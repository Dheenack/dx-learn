"""
Genetic Algorithm search engine.

Population-based evolution with tournament selection, subtree crossover,
typed mutation, elitism, fitness caching, duplicate elimination, early stopping,
and runtime limits.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from dxlearn.base.evolutionary_base import EvolutionarySearch
from dxlearn.encoding.grammar import PIPELINE_GRAMMAR
from dxlearn.encoding.node import PipelineNode, PreprocessorNode, ScalerNode, ClassifierNode
from dxlearn.encoding.tree import tree_to_pipeline
from dxlearn.evaluation.evaluator import Evaluator
from dxlearn.evaluation.objectives import Objectives
from dxlearn.evaluation.scalarizer import WeightedSumScalarizer
from dxlearn.operators.crossover import subtree_crossover
from dxlearn.operators.mutation import mutate_pipeline_node
from dxlearn.operators.selection import tournament_selection
from dxlearn.search_space.registry import get_registry

logger = logging.getLogger(__name__)


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
        self._evaluator = Evaluator(
            cv=cv,
            scoring="accuracy",
            scalarizer=WeightedSumScalarizer(alpha=alpha, beta=beta, gamma=gamma),
            per_individual_timeout=per_individual_timeout,
            random_state=random_state,
            n_jobs=1,
        )
        self._registry = get_registry()
        self._cache: Dict[str, Tuple[float, Objectives]] = {}
        self._history: List[Dict[str, Any]] = []
        self._best_objectives: Optional[Objectives] = None
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
        Evaluate a population of individuals with fitness caching.

        Uses Evaluator.evaluate_population under the hood, but only for
        individuals that are not already present in the cache. Cached
        results are reused across generations when deterministic=True.
        """
        from typing import Optional as _Optional

        fitnesses: List[_Optional[float]] = [None] * len(population)
        objectives_list: List[_Optional[Objectives]] = [None] * len(population)
        to_eval_indices: List[int] = []
        to_eval_inds: List[PipelineNode] = []

        for i, ind in enumerate(population):
            if self.deterministic:
                h = _tree_hash(ind)
                cached = self._cache.get(h)
                if cached is not None:
                    fitnesses[i], objectives_list[i] = cached
                    continue
            to_eval_indices.append(i)
            to_eval_inds.append(ind)

        if to_eval_inds:
            eval_fits, eval_objs = self._evaluator.evaluate_population(
                to_eval_inds, X, y, registry=self._registry, n_jobs=self.n_jobs
            )
            for local_idx, (f, o, ind) in enumerate(zip(eval_fits, eval_objs, to_eval_inds)):
                idx = to_eval_indices[local_idx]
                fitnesses[idx] = f
                objectives_list[idx] = o
                if self.deterministic:
                    self._cache[_tree_hash(ind)] = (f, o)

        # At this point all entries must be filled.
        return [float(f) for f in fitnesses], [o for o in objectives_list]  # type: ignore[arg-type]

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
        )

    def fit(self, X: Any, y: Any) -> GeneticSearch:
        """Run the genetic search with early stopping and runtime limit."""
        rng = self._get_rng()
        start_time = time.perf_counter()
        self._cache.clear()
        self._evaluator.clear_result_cache()
        self._history = []
        self._n_features = int(np.asarray(X).shape[1])

        # Initial population (with duplicate elimination)
        population: List[PipelineNode] = []
        seen: Set[str] = set()
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

            # Elitism: keep top elitism_count
            sorted_idx = np.argsort(fitnesses)[::-1]
            elite = [population[i].copy() for i in sorted_idx[: self.elitism_count]]
            elite_fitnesses = [fitnesses[i] for i in sorted_idx[: self.elitism_count]]

            # Selection for parents
            num_offspring = self.population_size - len(elite)
            parents = self._select(population, fitnesses, k=num_offspring)

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

            # Duplicate elimination: replace duplicates with new random individuals
            offspring_set: Set[str] = set()
            unique_offspring: List[PipelineNode] = []
            for ind in offspring:
                h = _tree_hash(ind)
                if h not in offspring_set and h not in seen:
                    offspring_set.add(h)
                    seen.add(h)
                    unique_offspring.append(ind)
                else:
                    new_ind = self._create_individual()
                    while _tree_hash(new_ind) in seen:
                        new_ind = self._create_individual()
                    unique_offspring.append(new_ind)
                    seen.add(_tree_hash(new_ind))
            offspring = unique_offspring[:num_offspring]
            while len(offspring) < num_offspring:
                new_ind = self._create_individual()
                if _tree_hash(new_ind) not in seen:
                    seen.add(_tree_hash(new_ind))
                    offspring.append(new_ind)

            population = elite + offspring
            fitnesses = list(elite_fitnesses) + [0.0] * len(offspring)
            objectives_list = [objectives_list[sorted_idx[j]] for j in range(self.elitism_count)] + [None] * len(offspring)

            # Evaluate offspring (with caching)
            off_fitnesses, off_objectives = self._evaluate_population(offspring, X, y)
            for j, ind in enumerate(offspring):
                idx = self.elitism_count + j
                fitnesses[idx] = off_fitnesses[j]
                objectives_list[idx] = off_objectives[j]

            best_gen_idx = int(np.argmax(fitnesses))
            best_gen_fitness = fitnesses[best_gen_idx]
            if best_gen_fitness > self._best_fitness:
                self._best_fitness = best_gen_fitness
                self._best_individual = population[best_gen_idx].copy()
                self._best_objectives = objectives_list[best_gen_idx]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

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

        return self

    def get_history(self) -> List[Dict[str, Any]]:
        """Return evolution history for dashboard."""
        return self._history

    def get_best_objectives(self) -> Optional[Objectives]:
        """Return objectives of the best individual."""
        return self._best_objectives
