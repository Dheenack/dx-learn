"""
Pipeline evaluator: cross-validation, timing, complexity, exception handling.

Uses StratifiedKFold by default; supports custom CV; penalizes invalid pipelines.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

from dxlearn.encoding.node import PipelineNode
from dxlearn.encoding.tree import tree_to_pipeline
from dxlearn.evaluation.objectives import Objectives
from dxlearn.evaluation.scalarizer import BaseScalarizer, WeightedSumScalarizer
from dxlearn.validation.pipeline_validator import (
    pipeline_node_cache_key,
    validate_pipeline,
)

logger = logging.getLogger(__name__)

# Penalized accuracy for failed / invalid pipelines (distinct from legitimate 0.0 score).
FAILED_ACCURACY: float = -1.0


def _cv_test_scores_array(result: Mapping[str, Any], scoring: Optional[str]) -> np.ndarray:
    """
    Extract per-fold test scores from ``cross_validate`` output.

    Compatible with sklearn: single metric ``"accuracy"`` → ``test_accuracy``;
    default estimator score → ``test_score``; named metrics → ``test_<scoring>``.
    """
    if scoring in (None, "score"):
        primary_keys = ("test_score",)
    else:
        primary_keys = (f"test_{scoring}", "test_score")

    for key in primary_keys:
        if key in result:
            return np.asarray(result[key], dtype=float)

    for key in sorted(result):
        if not key.startswith("test_"):
            continue
        if "time" in key:
            continue
        val = result[key]
        if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
            return np.asarray(val, dtype=float)

    raise KeyError(
        "No test score array found in cross_validate result; "
        f"scoring={scoring!r}, keys={list(result)!r}"
    )


def _run_with_timeout(fn: Any, timeout: float) -> Any:
    """Run ``fn()`` in a worker thread with a wall-clock timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        return future.result(timeout=timeout)


class Evaluator:
    """
    Evaluates pipeline trees via cross-validation.

    Measures mean accuracy, fit_time, predict_time, and complexity.
    Catches exceptions and returns penalized objectives.
    """

    def __init__(
        self,
        cv: int = 5,
        scoring: str = "accuracy",
        scalarizer: Optional[BaseScalarizer] = None,
        per_individual_timeout: Optional[float] = 60.0,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ) -> None:
        """
        Args:
            cv: Number of folds (or a CV splitter object).
            scoring: Scoring metric name (default ``\"accuracy\"``).
            scalarizer: Converts Objectives to scalar fitness; default WeightedSum.
            per_individual_timeout: Max seconds per pipeline evaluation (thread timeout).
            random_state: For CV splits.
            n_jobs: Parallel jobs for cross_validate (within evaluate, typically 1).
        """
        self.cv = cv
        self.scoring = scoring
        self.scalarizer = scalarizer or WeightedSumScalarizer()
        self.per_individual_timeout = per_individual_timeout
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._result_cache: Dict[str, Tuple[float, Objectives]] = {}

    def clear_result_cache(self) -> None:
        """Clear per-individual evaluation cache (e.g. at start of a new search)."""
        self._result_cache.clear()

    def evaluate(
        self,
        individual: PipelineNode,
        X: Any,
        y: Any,
        registry: Any = None,
    ) -> Tuple[float, Objectives]:
        """
        Evaluate one pipeline tree. Returns (scalar_fitness, objectives).

        On exception, timeout, failed pre-CV validation, or timeout: penalized
        objectives with ``accuracy=-1.0`` (not ``0.0``, which is ambiguous).
        """
        cache_key = pipeline_node_cache_key(individual)
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]

        complexity = float(individual.node_count())

        def _penalized() -> Tuple[float, Objectives]:
            objectives = Objectives(
                accuracy=FAILED_ACCURACY,
                fit_time=1e6,
                predict_time=1e6,
                complexity=complexity,
            )
            return self.scalarizer(objectives), objectives

        def _body() -> Tuple[float, Objectives]:
            n_features = int(np.asarray(X).shape[1])
            pipeline = tree_to_pipeline(
                individual,
                registry=registry,
                n_features=n_features,
                random_state=self.random_state,
            )
            if not validate_pipeline(pipeline, X, y):
                logger.warning(
                    "Pipeline failed lightweight pre-CV validation (complexity=%.0f).",
                    complexity,
                )
                return _penalized()

            cv_splitter = StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
            result = cross_validate(
                pipeline,
                X,
                y,
                cv=cv_splitter,
                scoring=self.scoring,
                n_jobs=1,
                return_train_score=False,
                error_score=np.nan,
            )
            test_scores = _cv_test_scores_array(result, self.scoring)
            test_scores = np.asarray(test_scores, dtype=float)
            if np.all(np.isnan(test_scores)):
                logger.warning(
                    "All CV folds failed during cross_validate (complexity=%.0f).",
                    complexity,
                )
                return _penalized()
            fit_times = result["fit_time"]
            score_times = result["score_time"]
            accuracy = float(np.round(np.nanmean(test_scores), 8))
            # Quantize timings aggressively: tiny wall-clock differences change
            # population min-max normalization and break deterministic GA selection.
            fit_time = float(np.round(np.mean(fit_times), 2))
            predict_time = float(np.round(np.mean(score_times), 2))
            objectives = Objectives(
                accuracy=accuracy,
                fit_time=fit_time,
                predict_time=predict_time,
                complexity=complexity,
            )
            fitness = self.scalarizer(objectives)
            return fitness, objectives

        try:
            timeout = self.per_individual_timeout
            if timeout is not None and float(timeout) > 0:
                out = _run_with_timeout(_body, float(timeout))
            else:
                out = _body()
            self._result_cache[cache_key] = out
            return out
        except FuturesTimeoutError:
            logger.warning(
                "Pipeline evaluation timed out after %.1fs (complexity=%.0f).",
                float(self.per_individual_timeout or 0.0),
                complexity,
                exc_info=True,
            )
            out = _penalized()
            self._result_cache[cache_key] = out
            return out
        except Exception:
            logger.warning(
                "Pipeline evaluation failed (complexity=%.0f).",
                complexity,
                exc_info=True,
            )
            out = _penalized()
            self._result_cache[cache_key] = out
            return out

    def evaluate_population(
        self,
        population: List[PipelineNode],
        X: Any,
        y: Any,
        registry: Any = None,
        n_jobs: int = -1,
    ) -> Tuple[List[float], List[Objectives]]:
        """
        Evaluate all individuals. Uses joblib for parallelism.

        Returns (list of scalar fitnesses, list of objective vectors).
        """
        from joblib import Parallel, delayed

        # Sequential backend avoids subtle ordering/thread issues when n_jobs==1.
        parallel_kw: Dict[str, Any] = {"n_jobs": n_jobs}
        if n_jobs == 1:
            parallel_kw["backend"] = "sequential"
        results = Parallel(**parallel_kw)(
            delayed(self.evaluate)(ind, X, y, registry) for ind in population
        )
        fitnesses = [r[0] for r in results]
        objectives_list = [r[1] for r in results]
        return fitnesses, objectives_list


def mean_cross_val_score(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    cv: int,
    random_state: Optional[int],
    scoring: str = "accuracy",
) -> float:
    """
    Mean test score from stratified k-fold CV (same defaults as Evaluator).

    Used to set ``best_score_`` on the final pipeline so it reflects real CV
    performance rather than GA-internal objectives that may be penalized.

    Folds that error use ``error_score=np.nan`` and are omitted from the mean.
    Raises if too few folds succeed (``valid.size < max(1, cv // 2)``).
    """
    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )
    result = cross_validate(
        estimator,
        X,
        y,
        cv=skf,
        scoring=scoring,
        n_jobs=1,
        error_score=np.nan,
    )
    scores = _cv_test_scores_array(result, scoring)
    scores = np.asarray(scores, dtype=float)
    valid = scores[~np.isnan(scores)]
    min_ok = max(1, cv // 2)
    if valid.size < min_ok:
        raise RuntimeError(
            f"Too many cross-validation failures: only {valid.size} of {cv} folds "
            f"succeeded (need at least {min_ok})."
        )
    if valid.size == 0:
        raise RuntimeError("All cross-validation folds failed for the best pipeline.")
    return float(np.mean(valid))
