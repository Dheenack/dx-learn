"""
Scalarization strategies: convert multi-objective vector to single fitness.

Weighted sum: fitness = α*accuracy - β*fit_time - γ*complexity (and predict_time).
Future: Pareto ranking (NSGA-II), adaptive weighting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dxlearn.evaluation.objectives import Objectives


class BaseScalarizer(ABC):
    """
    Abstract scalarization strategy.

    Converts multi-objective vector into a single fitness score.
    Higher fitness is better.

    Future:
        - Pareto ranking (NSGA-II)
        - Adaptive weighting
    """

    @abstractmethod
    def __call__(self, objectives: Objectives) -> float:
        """Compute scalar fitness from objectives. Higher is better."""
        pass


class WeightedSumScalarizer(BaseScalarizer):
    """
    Linear weighted sum: α*accuracy - β*fit_time - γ*complexity.

    Optionally includes predict_time with a separate weight.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.2,
        gamma: float = 0.01,
        delta: float = 0.01,
        use_predict_time: bool = True,
    ) -> None:
        """
        Args:
            alpha: Weight for accuracy (maximize).
            beta: Penalty for fit_time (minimize).
            gamma: Penalty for complexity (minimize).
            delta: Penalty for predict_time (minimize).
            use_predict_time: If True, subtract delta * predict_time.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.use_predict_time = use_predict_time

    def __call__(self, objectives: Objectives) -> float:
        """fitness = α*accuracy - β*fit_time - γ*complexity [- δ*predict_time]."""
        f = (
            self.alpha * objectives.accuracy
            - self.beta * objectives.fit_time
            - self.gamma * objectives.complexity
        )
        if self.use_predict_time:
            f -= self.delta * objectives.predict_time
        return f
