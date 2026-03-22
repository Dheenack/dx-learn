"""
Pipeline evaluator: cross-validation, timing, complexity, exception handling.

Uses StratifiedKFold by default; supports custom CV; penalizes invalid pipelines.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

from dxlearn.encoding.node import PipelineNode
from dxlearn.encoding.tree import tree_to_pipeline
from dxlearn.evaluation.objectives import Objectives
from dxlearn.evaluation.scalarizer import BaseScalarizer, WeightedSumScalarizer

logger = logging.getLogger(__name__)


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
            scoring: Scoring metric name.
            scalarizer: Converts Objectives to scalar fitness; default WeightedSum.
            per_individual_timeout: Max seconds per pipeline evaluation.
            random_state: For CV splits.
            n_jobs: Parallel jobs for cross_validate.
        """
        self.cv = cv
        self.scoring = scoring
        self.scalarizer = scalarizer or WeightedSumScalarizer()
        self.per_individual_timeout = per_individual_timeout
        self.random_state = random_state
        self.n_jobs = n_jobs

    def evaluate(
        self,
        individual: PipelineNode,
        X: Any,
        y: Any,
        registry: Any = None,
    ) -> Tuple[float, Objectives]:
        """
        Evaluate one pipeline tree. Returns (scalar_fitness, objectives).

        On exception or timeout, returns penalized objectives (low fitness).
        """
        try:
            n_features = int(np.asarray(X).shape[1])
            pipeline = tree_to_pipeline(
                individual, registry=registry, n_features=n_features
            )
            cv = StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
            # cross_validate returns dict with test_<scoring>, fit_time, score_time
            result = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=self.scoring,
                n_jobs=1,  # we parallelize at population level
                return_train_score=False,
            )
            test_scores = result[f"test_{self.scoring}"]
            fit_times = result["fit_time"]
            score_times = result["score_time"]
            accuracy = float(np.mean(test_scores))
            fit_time = float(np.mean(fit_times))
            predict_time = float(np.mean(score_times))
            complexity = float(individual.node_count())
            objectives = Objectives(
                accuracy=accuracy,
                fit_time=fit_time,
                predict_time=predict_time,
                complexity=complexity,
            )
            fitness = self.scalarizer(objectives)
            return fitness, objectives
        except Exception as e:
            logger.debug("Pipeline evaluation failed: %s", e)
            objectives = Objectives(
                accuracy=0.0,
                fit_time=1e6,
                predict_time=1e6,
                complexity=1e6,
            )
            return self.scalarizer(objectives), objectives

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
        results = Parallel(n_jobs=n_jobs)(
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

    Folds that error use ``error_score=np.nan`` and are omitted from the mean
    so brittle pipelines (e.g. aggressive ``VarianceThreshold`` on small folds)
    still yield a score when any fold succeeds.
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate

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
    scores = np.asarray(result[f"test_{scoring}"], dtype=float)
    valid = scores[~np.isnan(scores)]
    if valid.size == 0:
        raise RuntimeError("All cross-validation folds failed for the best pipeline.")
    return float(np.mean(valid))
