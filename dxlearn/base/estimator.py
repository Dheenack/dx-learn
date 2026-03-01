"""
Base estimator wrapping the best pipeline for sklearn API compatibility.

Provides fit, predict, predict_proba, score, get_params, set_params.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class BaseDXEstimator:
    """
    Wrapper that exposes the best pipeline as an sklearn-compatible estimator.

    Used by DXClassifier to delegate fit/predict/score to the best
    discovered pipeline after search.
    """

    def __init__(self, pipeline: Any, best_score: Optional[float] = None) -> None:
        """Initialize with the best pipeline and optional score.

        Args:
            pipeline: sklearn Pipeline or estimator.
            best_score: Cross-validation score of the best pipeline.
        """
        self.pipeline = pipeline
        self.best_score_ = best_score

    def fit(self, X: Any, y: Any, **fit_params: Any) -> BaseDXEstimator:
        """Fit the underlying pipeline."""
        self.pipeline.fit(X, y, **fit_params)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities if the pipeline supports it."""
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        raise AttributeError(
            "The best pipeline does not support predict_proba. "
            "Use a classifier that supports it (e.g. LogisticRegression, RandomForest)."
        )

    def score(self, X: Any, y: Any) -> float:
        """Return mean accuracy on the given test data and labels."""
        return self.pipeline.score(X, y)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator (sklearn API)."""
        return {"pipeline": self.pipeline, "best_score": self.best_score_}

    def set_params(self, **params: Any) -> BaseDXEstimator:
        """Set parameters (sklearn API)."""
        if "pipeline" in params:
            self.pipeline = params["pipeline"]
        if "best_score" in params:
            self.best_score_ = params["best_score"]
        return self
