"""
Multi-objective evaluation metrics container.

Stores accuracy, fit_time, predict_time, complexity for each pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Objectives:
    """
    Container for multi-objective evaluation metrics.

    Attributes:
        accuracy: Mean cross-validation score (e.g. accuracy).
        fit_time: Mean training time across folds (seconds).
        predict_time: Mean inference time (seconds).
        complexity: Structural complexity score (e.g. node count).
    """

    accuracy: float
    fit_time: float
    predict_time: float
    complexity: float

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return (accuracy, fit_time, predict_time, complexity)."""
        return (self.accuracy, self.fit_time, self.predict_time, self.complexity)
