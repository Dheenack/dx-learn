"""
Genetic Algorithm search engine.

Population-based evolution with tournament selection, subtree crossover,
typed mutation, elitism, fitness caching, duplicate elimination, early stopping,
and runtime limits.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from dxlearn.base.evolutionary_base import EvolutionarySearch
from dxlearn.encoding.grammar import PIPELINE_GRAMMAR
from dxlearn.encoding.node import PipelineNode, PreprocessorNode, ScalerNode, ClassifierNode
from dxlearn.encoding.tree import tree_to_pipeline
from dxlearn.evaluation.evaluator import Evaluator
from dxlearn.evaluation.objectives import Objectives
from dxlearn.evaluation.scalarizer import WeightedSumScalarizer, normalize_objectives_batch
from dxlearn.operators.crossover import subtree_crossover
from dxlearn.operators.mutation import mutate_pipeline_node
from dxlearn.operators.selection import tournament_selection
from dxlearn.search_space.registry import get_registry

logger = logging.getLogger(__name__)

_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextlib.contextmanager
def _blas_single_thread() -> Iterator[None]:
    """Force BLAS/OpenMP to one thread for reproducible CV (deterministic GA)."""
    saved: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in _BLAS_THREAD_VARS}
    try:
        for k in _BLAS_THREAD_VARS:
            os.environ[k] = "1"
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _tree_hash(node: PipelineNode) -> str:
    """Stable hash for duplicate detection."""
    parts = []
    if node.preprocessor.key:
        parts.append(f"prep:{node.preprocessor.key}:{sorted(node.preprocessor.params.items())}")
    else:
        parts.append("prep:None")
    parts.append(f"scaler:{node.scaler.key}:{sorted(node.scaler.params.items())}")
    parts.append(f"clf:{node.classifier.key}:{sorted(node.classifier.params.items())}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _baseline_seed_individuals() -> List[PipelineNode]:
    """
    Strong, grammar-valid baselines (warm start).

    StandardScaler + LogisticRegression is a stable default; optional preprocessor
    None keeps the search space valid on any feature count.
    """
    return [
        PipelineNode(
            preprocessor=PreprocessorNode(key=None, params={}),
            scaler=ScalerNode(key="StandardScaler", params={}),
            classifier=ClassifierNode(
                key="LogisticRegression",
                params={"C": 1.0, "max_iter": 2000, "solver": "lbfgs"},
            ),
        ),
    ]


def _create_random_individual(rng: Any) -> PipelineNode:
    """Create one random pipeline tree from the grammar."""
    registry = get_registry()
    choices_prep = PIPELINE_GRAMMAR.get_preprocessor_choices()
    choices_scaler = PIPELINE_GRAMMAR.get_scaler_choices()
    choices_clf = PIPELINE_GRAMMAR.get_classifier_choices()

    prep_key = choices_prep[int(rng.integers(0, len(choices_prep)))]
    prep_params = registry.sample_preprocessor_params(prep_key, rng) if prep_key else {}
    preprocessor = PreprocessorNode(key=prep_key, params=prep_params)

    scaler_key = choices_scaler[int(rng.integers(0, len(choices_scaler)))]
    scaler = ScalerNode(key=scaler_key, params=registry.sample_scaler_params(scaler_key, rng))

    clf_key = choices_clf[int(rng.integers(0, len(choices_clf)))]
    classifier = ClassifierNode(key=clf_key, params=registry.sample_classifier_params(clf_key, rng))

    return PipelineNode(preprocessor=preprocessor, scaler=scaler, classifier=classifier)


class GeneticSearch(EvolutionarySearch):
    """
    Grammar-constrained genetic algorithm for pipeline search.

    Supports elitism, tournament selection, subtree crossover, typed mutation,
    fitness caching, duplicate elimination, early stopping, and runtime limits.
    """

    def __init__(
        self,
        population_size: int = 30,
        generations: int = 20,
        elitism_count: int = 2,
        tournament_size: int = 3,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        early_stopping_generations: Optional[int] = 5,
        max_runtime: Optional[float] = 600.0,
        per_individual_timeout: Optional[float] = 60.0,
        cv: int = 5,
        alpha: float = 1.0,
        beta: float = 0.2,
        gamma: float = 0.01,
        random_state: Optional[int] = None,
        verbose: int = 1,
        n_jobs: int = -1,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            population_size=population_size,
            generations=generations,
            elitism_count=elitism_count,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            early_stopping_generations=early_stopping_generations,
            max_runtime=max_runtime,
            per_individual_timeout=per_individual_timeout,
            random_state=random_state,
            verbose=verbose,
            n_jobs=n_jobs,
            deterministic=deterministic,
            **kwargs,
        )
        self.tournament_size = tournament_size
        self.cv = cv
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Thread-based timeouts around CV break reproducibility (BLAS/thread scheduling).
        # When deterministic=True, evaluate synchronously; users who need wall timeouts
        # can set deterministic=False.
        _eval_timeout = None if deterministic else per_individual_timeout
        self._evaluator = Evaluator(
            cv=cv,
            scoring="accuracy",
            scalarizer=WeightedSumScalarizer(alpha=alpha, beta=beta, gamma=gamma),
            per_individual_timeout=_eval_timeout,
            random_state=random_state,
            n_jobs=1,
        )
        self._registry = get_registry()
        # Deterministic cache: tree hash -> raw Objectives (fitness is recomputed per population).
        self._cache: Dict[str, Objectives] = {}
        self._history: List[Dict[str, Any]] = []
        self._best_objectives: Optional[Objectives] = None
        self._best_accuracy: float = float("-inf")
        self._n_features: Optional[int] = None

        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if elitism_count >= population_size:
            raise ValueError("elitism_count must be strictly less than population_size")

    def _create_individual(self) -> PipelineNode:
        return _create_random_individual(self._get_rng())

    def _evaluate_population(
        self,
        population: List[PipelineNode],
        X: Any,
        y: Any,
    ) -> Tuple[List[float], List[Objectives]]:
        """
        Evaluate a population: raw objectives from CV (or cache), then
        population-level normalization and scalar fitness (WeightedSumScalarizer).
        """
        from typing import Optional as _Optional

        objectives_list: List[_Optional[Objectives]] = [None] * len(population)
        to_eval_indices: List[int] = []
        to_eval_inds: List[PipelineNode] = []

        for i, ind in enumerate(population):
            if self.deterministic:
                h = _tree_hash(ind)
                cached = self._cache.get(h)
                if cached is not None:
                    objectives_list[i] = cached
                    continue
            to_eval_indices.append(i)
            to_eval_inds.append(ind)

        if to_eval_inds:
            # Avoid joblib entirely in deterministic mode (loky/thread pools can leak
            # state and make identical seeds diverge between consecutive fits).
            if self.deterministic:
                eval_objs = [
                    self._evaluator.evaluate(ind, X, y, self._registry)[1]
                    for ind in to_eval_inds
                ]
            else:
                _, eval_objs = self._evaluator.evaluate_population(
                    to_eval_inds, X, y, registry=self._registry, n_jobs=self.n_jobs
                )
            for local_idx, (o, ind) in enumerate(zip(eval_objs, to_eval_inds)):
                idx = to_eval_indices[local_idx]
                objectives_list[idx] = o
                if self.deterministic:
                    self._cache[_tree_hash(ind)] = o

        objs_final: List[Objectives] = [o for o in objectives_list]  # type: ignore[list-item]
        norm_objs = normalize_objectives_batch(objs_final)
        fitnesses = [float(self._evaluator.scalarizer(no)) for no in norm_objs]
        return fitnesses, objs_final

    def _evaluate_fitness(self, individual: PipelineNode, X: Any, y: Any) -> Tuple[float, Objectives]:
        """Evaluate a single individual using the population-aware cache."""
        fitnesses, objectives = self._evaluate_population([individual], X, y)
        return fitnesses[0], objectives[0]

    def _select(
        self,
        population: List[PipelineNode],
        fitnesses: List[float],
        k: int,
    ) -> List[PipelineNode]:
        return tournament_selection(
            population,
            fitnesses,
            k=k,
            tournament_size=self.tournament_size,
            rng=self._get_rng(),
        )

    def _crossover(self, parent_a: PipelineNode, parent_b: PipelineNode) -> Tuple[PipelineNode, PipelineNode]:
        return subtree_crossover(parent_a, parent_b, self._get_rng())

    def _mutate(self, individual: PipelineNode) -> PipelineNode:
        return mutate_pipeline_node(individual, self.mutation_rate, self._get_rng())

    def _individual_to_pipeline(self, individual: PipelineNode) -> Any:
        return tree_to_pipeline(
            individual,
            registry=self._registry,
            n_features=self._n_features,
            random_state=self.random_state,
        )

    def fit(self, X: Any, y: Any) -> GeneticSearch:
        """Run the genetic search with early stopping and runtime limit."""
        ctx = _blas_single_thread() if self.deterministic else contextlib.nullcontext()
        with ctx:
            return self._fit_impl(X, y)

    def _fit_impl(self, X: Any, y: Any) -> GeneticSearch:
        self._rng = None
        if self.deterministic and self.random_state is not None:
            # Fresh processes are reproducible; re-seed legacy NumPy RNG so a second fit()
            # in the same interpreter matches a clean subprocess (sklearn still uses it).
            np.random.seed(int(self.random_state))
        rng = self._get_rng()
        start_time = time.perf_counter()
        saved_mutation_rate = float(self.mutation_rate)
        self._cache.clear()
        self._evaluator.clear_result_cache()
        self._history = []
        self._best_accuracy = float("-inf")
        self._n_features = int(np.asarray(X).shape[1])

        # Initial population: warm-start baselines, then random (with duplicate elimination)
        population: List[PipelineNode] = []
        seen: Set[str] = set()
        for ind in _baseline_seed_individuals():
            if len(population) >= self.population_size:
                break
            h = _tree_hash(ind)
            if h not in seen:
                seen.add(h)
                population.append(ind.copy())
        while len(population) < self.population_size:
            ind = self._create_individual()
            h = _tree_hash(ind)
            if h not in seen:
                seen.add(h)
                population.append(ind)
            if time.perf_counter() - start_time > (self.max_runtime or 1e9):
                break

        if self.verbose:
            logger.info("Initial population size: %d", len(population))

        # Evaluate initial population (with caching)
        fitnesses, objectives_list = self._evaluate_population(population, X, y)

        for o in objectives_list:
            if o.accuracy > self._best_accuracy:
                self._best_accuracy = float(o.accuracy)

        # Best so far
        best_idx = int(np.argmax(fitnesses))
        self._best_individual = population[best_idx].copy()
        self._best_fitness = float(fitnesses[best_idx])
        self._best_objectives = objectives_list[best_idx]
        generations_without_improvement = 0

        self._history.append({
            "generation": 0,
            "best_fitness": self._best_fitness,
            "best_accuracy": self._best_objectives.accuracy,
            "mean_fitness": float(np.mean(fitnesses)),
        })

        for gen in range(1, self.generations):
            if self.max_runtime and (time.perf_counter() - start_time) >= self.max_runtime:
                if self.verbose:
                    logger.info("Max runtime reached at generation %d", gen)
                break

            # Elitism: keep top elitism_count (by scalar fitness). lexsort breaks ties
            # by individual index so near-tie floats cannot flip order nondeterministically.
            fit_arr = np.asarray(fitnesses, dtype=float)
            sorted_idx = np.lexsort((np.arange(len(fit_arr), dtype=int), -fit_arr))
            elite = [population[i].copy() for i in sorted_idx[: self.elitism_count]]
            elite_fitnesses = [fitnesses[i] for i in sorted_idx[: self.elitism_count]]
            elite_hashes = {_tree_hash(e) for e in elite}

            # Selection for parents: need at least two parents whenever we breed
            num_offspring = self.population_size - len(elite)
            parent_pool_k = max(num_offspring, 2) if num_offspring > 0 else 2
            parents = self._select(population, fitnesses, k=parent_pool_k)

            # Crossover and mutation
            offspring: List[PipelineNode] = []
            i = 0
            while len(offspring) < num_offspring:
                if i + 1 >= len(parents):
                    i = 0
                p_a, p_b = parents[i], parents[i + 1]
                if rng.random() < self.crossover_rate:
                    c_a, c_b = self._crossover(p_a, p_b)
                    offspring.append(self._mutate(c_a))
                    if len(offspring) < num_offspring:
                        offspring.append(self._mutate(c_b))
                else:
                    offspring.append(self._mutate(p_a.copy()))
                i += 2

            # Duplicate elimination vs rest of run + elite; replace with new random individuals
            offspring_set: Set[str] = set()
            unique_offspring: List[PipelineNode] = []
            for ind in offspring:
                h = _tree_hash(ind)
                if h not in offspring_set and h not in seen and h not in elite_hashes:
                    offspring_set.add(h)
                    seen.add(h)
                    unique_offspring.append(ind)
                else:
                    new_ind = self._create_individual()
                    tries = 0
                    while _tree_hash(new_ind) in seen or _tree_hash(new_ind) in elite_hashes:
                        new_ind = self._create_individual()
                        tries += 1
                        if tries > 500:
                            break
                    nh = _tree_hash(new_ind)
                    seen.add(nh)
                    unique_offspring.append(new_ind)
            offspring = unique_offspring[:num_offspring]
            while len(offspring) < num_offspring:
                new_ind = self._create_individual()
                if _tree_hash(new_ind) not in seen and _tree_hash(new_ind) not in elite_hashes:
                    seen.add(_tree_hash(new_ind))
                    offspring.append(new_ind)

            population = elite + offspring
            fitnesses = list(elite_fitnesses) + [0.0] * len(offspring)
            objectives_list = [
                objectives_list[int(sorted_idx[j])] for j in range(self.elitism_count)
            ] + [None] * len(offspring)  # type: ignore[list-item]

            # Evaluate offspring (with caching); elite fitness/objectives already known
            off_fitnesses, off_objectives = self._evaluate_population(offspring, X, y)
            for j, ind in enumerate(offspring):
                idx = self.elitism_count + j
                fitnesses[idx] = off_fitnesses[j]
                objectives_list[idx] = off_objectives[j]

            for o in off_objectives:
                if o.accuracy > self._best_accuracy:
                    self._best_accuracy = float(o.accuracy)

            best_gen_idx = int(np.argmax(fitnesses))
            best_gen_fitness = fitnesses[best_gen_idx]
            if best_gen_fitness > self._best_fitness:
                self._best_fitness = best_gen_fitness
                self._best_individual = population[best_gen_idx].copy()
                self._best_objectives = objectives_list[best_gen_idx]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Adaptive mutation rate (restored to user value after fit() completes)
            if generations_without_improvement > 2:
                self.mutation_rate = min(0.5, self.mutation_rate * 1.5)
            else:
                self.mutation_rate = max(0.1, self.mutation_rate * 0.9)

            self._history.append({
                "generation": gen,
                "best_fitness": self._best_fitness,
                "best_accuracy": self._best_objectives.accuracy,
                "mean_fitness": float(np.mean(fitnesses)),
            })

            if self.verbose >= 1:
                logger.info(
                    "Gen %d: best_fitness=%.4f best_acc=%.4f",
                    gen, self._best_fitness, self._best_objectives.accuracy,
                )

            if self.early_stopping_generations and generations_without_improvement >= self.early_stopping_generations:
                if self.verbose:
                    logger.info("Early stopping at generation %d", gen)
                break

        self.mutation_rate = saved_mutation_rate
        return self

    def get_history(self) -> List[Dict[str, Any]]:
        """Return evolution history for dashboard."""
        return self._history

    def get_best_objectives(self) -> Optional[Objectives]:
        """Return objectives of the best individual."""
        return self._best_objectives
