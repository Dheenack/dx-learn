"""
Scalarization strategies: convert multi-objective vector to single fitness.

Weighted sum with optional population-level normalization of objectives before
scalarization (accuracy / times), plus log complexity penalty.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from dxlearn.evaluation.objectives import Objectives


def _normalize_column(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1d array to [0, 1]; degenerate range → 0.5."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo == 0:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def normalize_objectives_batch(objectives_list: List[Objectives]) -> List[Objectives]:
    """
    Min-max normalize accuracy, fit_time, and predict_time across a population.

    Failed runs (accuracy < 0) are clipped to 0 for normalization only.
    Complexity is left **raw** for ``log(1 + complexity)`` in the scalarizer.

    Args:
        objectives_list: Raw objectives from CV / penalties.

    Returns:
        New ``Objectives`` instances with normalized accuracy/times and raw complexity.
    """
    if not objectives_list:
        return []

    acc = np.array([max(0.0, float(o.accuracy)) for o in objectives_list], dtype=float)
    fit_t = np.array([float(o.fit_time) for o in objectives_list], dtype=float)
    pred_t = np.array([float(o.predict_time) for o in objectives_list], dtype=float)

    acc_n = _normalize_column(acc)
    fit_n = _normalize_column(fit_t)
    pred_n = _normalize_column(pred_t)

    out: List[Objectives] = []
    for i, o in enumerate(objectives_list):
        out.append(
            Objectives(
                accuracy=float(acc_n[i]),
                fit_time=float(fit_n[i]),
                predict_time=float(pred_n[i]),
                complexity=float(o.complexity),
            )
        )
    return out


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
    Weighted sum: α·accuracy − β·fit_time − δ·predict_time − γ·log(1 + complexity).

    Pass **population-normalized** accuracy and times (see ``normalize_objectives_batch``)
    for balanced multi-objective signal; complexity should remain raw (node count).
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
            gamma: Scale for log complexity penalty.
            delta: Penalty for predict_time (minimize).
            use_predict_time: If True, subtract delta * predict_time.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.use_predict_time = use_predict_time

    def __call__(self, objectives: Objectives) -> float:
        """Scalar fitness; complexity penalized as γ·log(1 + complexity)."""
        comp = max(0.0, float(objectives.complexity))
        complexity_term = self.gamma * math.log(1.0 + comp)
        f = (
            self.alpha * float(objectives.accuracy)
            - self.beta * float(objectives.fit_time)
            - complexity_term
        )
        if self.use_predict_time:
            f -= self.delta * float(objectives.predict_time)
        return f
