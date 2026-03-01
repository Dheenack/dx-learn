"""
Base class for evolutionary algorithms.

Provides reusable mechanisms for population initialization, selection,
crossover, mutation, fitness evaluation, elitism, and early stopping.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple

from dxlearn.base.search_base import BaseSearch


class EvolutionarySearch(BaseSearch):
    """
    Base class for evolutionary algorithms.

    Provides reusable mechanisms for:
        - Population initialization
        - Selection
        - Crossover
        - Mutation
        - Fitness evaluation
        - Elitism
        - Early stopping

    Future Extensions:
        - NSGA2Search
        - IslandModelSearch
        - DistributedGASearch
    """

    def __init__(
        self,
        population_size: int = 30,
        generations: int = 20,
        elitism_count: int = 2,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        early_stopping_generations: Optional[int] = 5,
        max_runtime: Optional[float] = 600.0,
        per_individual_timeout: Optional[float] = 60.0,
        random_state: Optional[int] = None,
        verbose: int = 1,
        n_jobs: int = -1,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize evolutionary search parameters.

        Args:
            population_size: Number of individuals per generation.
            generations: Maximum number of generations.
            elitism_count: Number of best individuals to preserve.
            mutation_rate: Probability of mutation per individual.
            crossover_rate: Probability of crossover per pair.
            early_stopping_generations: Stop if no improvement for N generations.
            max_runtime: Global runtime limit in seconds.
            per_individual_timeout: Max seconds per individual evaluation.
            random_state: Seed for reproducibility.
            verbose: Verbosity level (0, 1, 2).
            n_jobs: Parallel jobs for evaluation (-1 = all cores).
            deterministic: If True, use seeded RNG everywhere.
            **kwargs: Additional arguments for subclasses.
        """
        self.population_size = population_size
        self.generations = generations
        self.elitism_count = elitism_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.early_stopping_generations = early_stopping_generations
        self.max_runtime = max_runtime
        self.per_individual_timeout = per_individual_timeout
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.deterministic = deterministic
        self._rng: Any = None
        self._best_individual: Any = None
        self._best_fitness: Optional[float] = None
        self._history: List[dict] = []

    def _get_rng(self) -> Any:
        """Return a numpy RandomState or Generator for reproducibility."""
        import numpy as np
        if self._rng is not None:
            return self._rng
        seed = self.random_state if self.deterministic else None
        self._rng = np.random.default_rng(seed)
        return self._rng

    @abstractmethod
    def _create_individual(self) -> Any:
        """Create a single random individual (e.g. pipeline tree)."""
        pass

    @abstractmethod
    def _evaluate_fitness(self, individual: Any, X: Any, y: Any) -> Tuple[float, Any]:
        """Evaluate fitness of one individual. Returns (scalar_fitness, objectives)."""
        pass

    @abstractmethod
    def _select(
        self,
        population: List[Any],
        fitnesses: List[float],
        k: int,
    ) -> List[Any]:
        """Select k individuals from population using fitnesses."""
        pass

    @abstractmethod
    def _crossover(self, parent_a: Any, parent_b: Any) -> Tuple[Any, Any]:
        """Perform crossover; return (child_a, child_b)."""
        pass

    @abstractmethod
    def _mutate(self, individual: Any) -> Any:
        """Mutate individual in-place or return new individual."""
        pass

    def fit(self, X: Any, y: Any) -> EvolutionarySearch:
        """Run evolutionary search. Subclasses implement _run_evolution."""
        raise NotImplementedError("Subclasses must implement fit using _run_evolution.")

    def get_best_pipeline(self) -> Any:
        """Return the best discovered pipeline."""
        if self._best_individual is None:
            raise RuntimeError("Search has not been run. Call fit() first.")
        return self._individual_to_pipeline(self._best_individual)

    @abstractmethod
    def _individual_to_pipeline(self, individual: Any) -> Any:
        """Convert encoded individual to sklearn Pipeline."""
        pass
