"""
DXClassifier: sklearn-compatible API for genetic pipeline search.

Public API: fit, predict, predict_proba, score, get_params, set_params, dashboard.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional

import numpy as np
from sklearn.utils.validation import check_X_y

from dxlearn.base.estimator import BaseDXEstimator
from dxlearn.engine.genetic_search import GeneticSearch, _blas_single_thread
from dxlearn.evaluation.evaluator import mean_cross_val_score

logger = logging.getLogger(__name__)

# Parameters stored as DXClassifier attributes (vs. extra **kwargs for GeneticSearch).
_DXCLASSIFIER_MAIN_PARAMS: frozenset[str] = frozenset(
    {
        "population_size",
        "generations",
        "cv",
        "alpha",
        "beta",
        "gamma",
        "max_runtime",
        "verbose",
        "n_jobs",
        "deterministic",
        "random_state",
        "elitism_count",
        "tournament_size",
        "mutation_rate",
        "crossover_rate",
        "early_stopping_generations",
        "per_individual_timeout",
    }
)


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
        # copy=True: avoid mutating caller arrays in-place (would break reproducibility
        # when the same X, y is passed to a second estimator.fit).
        X, y = check_X_y(X, y, multi_output=False, copy=True)
        ctx = _blas_single_thread() if self.deterministic else contextlib.nullcontext()
        with ctx:
            if self.deterministic and self.random_state is not None:
                np.random.seed(int(self.random_state))
            self._search.fit(X, y)
            pipeline = self._search.get_best_pipeline()
            self.best_pipeline_ = pipeline
            if pipeline is None:
                raise RuntimeError("GeneticSearch failed to produce a valid pipeline.")
            # Recompute CV score on the chosen pipeline (GA objectives can be 0.0 when
            # many candidates fail; this matches sklearn-style best_score_ semantics).
            try:
                self.best_score_ = mean_cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=self.cv,
                    random_state=self.random_state,
                    scoring="accuracy",
                )
            except Exception as exc:
                logger.warning(
                    "Could not compute cross_val score for best pipeline: %s; "
                    "falling back to search objectives.",
                    exc,
                )
                objs = self._search.get_best_objectives()
                if objs is not None and objs.accuracy >= 0.0:
                    self.best_score_ = objs.accuracy
                else:
                    self.best_score_ = None
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
        if not hasattr(self._estimator.pipeline, "predict_proba"):
            raise AttributeError(
                "Pipeline does not support predict_proba; the final estimator "
                "has no predict_proba method."
            )
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
        """Set parameters (sklearn API).

        Rebuilds internal ``GeneticSearch`` and clears fitted state. Updates are
        **atomic**: if ``GeneticSearch`` validation fails (e.g. invalid
        ``elitism_count`` / ``population_size``), no attributes or ``_search``
        are modified.
        """
        if not params:
            return self

        valid = set(self.get_params(deep=False).keys())
        unknown = set(params) - valid
        if unknown:
            raise ValueError(
                "Invalid parameter(s) for estimator DXClassifier: "
                f"{sorted(unknown)!r}. Valid parameters are {sorted(valid)!r}."
            )

        merged = dict(self.get_params(deep=False))
        merged.update(params)

        main_vals = {k: merged[k] for k in _DXCLASSIFIER_MAIN_PARAMS}
        extra_kwargs = {k: v for k, v in merged.items() if k not in _DXCLASSIFIER_MAIN_PARAMS}

        # Validate by constructing search first; only then mutate self.
        new_search = GeneticSearch(
            population_size=main_vals["population_size"],
            generations=main_vals["generations"],
            elitism_count=main_vals["elitism_count"],
            tournament_size=main_vals["tournament_size"],
            mutation_rate=main_vals["mutation_rate"],
            crossover_rate=main_vals["crossover_rate"],
            early_stopping_generations=main_vals["early_stopping_generations"],
            max_runtime=main_vals["max_runtime"],
            per_individual_timeout=main_vals["per_individual_timeout"],
            cv=main_vals["cv"],
            alpha=main_vals["alpha"],
            beta=main_vals["beta"],
            gamma=main_vals["gamma"],
            random_state=main_vals["random_state"],
            verbose=main_vals["verbose"],
            n_jobs=main_vals["n_jobs"],
            deterministic=main_vals["deterministic"],
            **extra_kwargs,
        )

        for k, v in main_vals.items():
            setattr(self, k, v)
        self._kwargs = dict(extra_kwargs)
        self._search = new_search
        self._estimator = None
        self.best_pipeline_ = None
        self.best_score_ = None
        return self

    def dashboard(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """
        Launch the local analytical dashboard (FastAPI) at http://127.0.0.1:8000.

        Requires: pip install dxlearn[dashboard]
        """
        if self._estimator is None or self.best_pipeline_ is None:
            raise RuntimeError("Call fit(X, y) before dashboard() to view evolution history.")
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
