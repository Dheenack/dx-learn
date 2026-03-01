"""
DXClassifier: sklearn-compatible API for genetic pipeline search.

Public API: fit, predict, predict_proba, score, get_params, set_params, dashboard.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from dxlearn.base.estimator import BaseDXEstimator
from dxlearn.engine.genetic_search import GeneticSearch

logger = logging.getLogger(__name__)


class DXClassifier:
    """
    Genetic Algorithm driven AutoML classifier.

    Discovers optimal sklearn-compatible classification pipelines using
    grammar-constrained GA with multi-objective fitness (accuracy, time, complexity).
    """

    def __init__(
        self,
        population_size: int = 30,
        generations: int = 20,
        cv: int = 5,
        alpha: float = 1.0,
        beta: float = 0.2,
        gamma: float = 0.01,
        max_runtime: Optional[float] = 600.0,
        verbose: int = 1,
        n_jobs: int = -1,
        deterministic: bool = True,
        random_state: Optional[int] = 42,
        elitism_count: int = 2,
        tournament_size: int = 3,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        early_stopping_generations: Optional[int] = 5,
        per_individual_timeout: Optional[float] = 60.0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            population_size: Number of individuals per generation.
            generations: Maximum number of generations.
            cv: Cross-validation folds.
            alpha: Weight for accuracy in fitness.
            beta: Penalty for fit time.
            gamma: Penalty for complexity.
            max_runtime: Global runtime limit in seconds.
            verbose: Verbosity (0, 1, 2).
            n_jobs: Parallel jobs (-1 = all cores).
            deterministic: Use fixed random seed for reproducibility.
            random_state: Random seed.
            elitism_count: Number of best individuals to keep.
            tournament_size: Tournament size for selection.
            mutation_rate: Mutation probability per individual.
            crossover_rate: Crossover probability per pair.
            early_stopping_generations: Stop after N generations without improvement.
            per_individual_timeout: Max seconds per pipeline evaluation.
            **kwargs: Passed to GeneticSearch.
        """
        self.population_size = population_size
        self.generations = generations
        self.cv = cv
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_runtime = max_runtime
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.deterministic = deterministic
        self.random_state = random_state
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.early_stopping_generations = early_stopping_generations
        self.per_individual_timeout = per_individual_timeout
        self._kwargs = kwargs

        self._search = GeneticSearch(
            population_size=population_size,
            generations=generations,
            elitism_count=elitism_count,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            early_stopping_generations=early_stopping_generations,
            max_runtime=max_runtime,
            per_individual_timeout=per_individual_timeout,
            cv=cv,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            random_state=random_state,
            verbose=verbose,
            n_jobs=n_jobs,
            deterministic=deterministic,
            **kwargs,
        )
        self._estimator: Optional[BaseDXEstimator] = None
        self.best_pipeline_ = None
        self.best_score_: Optional[float] = None

    def fit(self, X: Any, y: Any, **fit_params: Any) -> DXClassifier:
        """Run genetic search and fit the best pipeline on full data."""
        self._search.fit(X, y)
        pipeline = self._search.get_best_pipeline()
        self.best_pipeline_ = pipeline
        objs = self._search.get_best_objectives()
        self.best_score_ = objs.accuracy if objs else None
        self._estimator = BaseDXEstimator(pipeline, best_score=self.best_score_)
        self._estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels."""
        if self._estimator is None:
            raise RuntimeError("Call fit() first.")
        return self._estimator.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities."""
        if self._estimator is None:
            raise RuntimeError("Call fit() first.")
        return self._estimator.predict_proba(X)

    def score(self, X: Any, y: Any) -> float:
        """Return mean accuracy on the given test data."""
        if self._estimator is None:
            raise RuntimeError("Call fit() first.")
        return self._estimator.score(X, y)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator (sklearn API)."""
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "cv": self.cv,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "max_runtime": self.max_runtime,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            "deterministic": self.deterministic,
            "random_state": self.random_state,
            "elitism_count": self.elitism_count,
            "tournament_size": self.tournament_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "early_stopping_generations": self.early_stopping_generations,
            "per_individual_timeout": self.per_individual_timeout,
            **self._kwargs,
        }

    def set_params(self, **params: Any) -> DXClassifier:
        """Set parameters (sklearn API). Rebuilds internal search."""
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._search = GeneticSearch(
            population_size=self.population_size,
            generations=self.generations,
            elitism_count=self.elitism_count,
            tournament_size=self.tournament_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            early_stopping_generations=self.early_stopping_generations,
            max_runtime=self.max_runtime,
            per_individual_timeout=self.per_individual_timeout,
            cv=self.cv,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            random_state=self.random_state,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            deterministic=self.deterministic,
            **self._kwargs,
        )
        return self

    def dashboard(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """
        Launch the local analytical dashboard (FastAPI) at http://127.0.0.1:8000.

        Requires: pip install dxlearn[dashboard]
        """
        try:
            from dxlearn.dashboard.api import run_dashboard
        except ImportError as e:
            raise ImportError(
                "Dashboard requires optional dependencies. Install with: pip install dxlearn[dashboard]"
            ) from e
        run_dashboard(
            search=self._search,
            host=host,
            port=port,
        )
