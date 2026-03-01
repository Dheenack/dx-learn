"""
Abstract base class for all search strategies in dxlearn.

Defines the minimal interface for any search algorithm (Genetic, Bayesian, Random, etc.)
to integrate with the dxlearn estimator API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSearch(ABC):
    """
    Abstract base class for all search strategies in dxlearn.

    This class defines the minimal interface required for
    any search algorithm (Genetic, Bayesian, Random, etc.)
    to integrate with the dxlearn estimator API.

    Future Extensions:
        - RegressionSearch
        - ClusteringSearch
        - MultiObjectiveSearch
    """

    @abstractmethod
    def fit(self, X: Any, y: Any) -> BaseSearch:
        """Run the search process and identify the best pipeline.

        Args:
            X: Training feature matrix.
            y: Target vector.

        Returns:
            self (for method chaining).
        """
        pass

    @abstractmethod
    def get_best_pipeline(self) -> Any:
        """Return the best discovered sklearn pipeline.

        Returns:
            An sklearn-compatible Pipeline or estimator.
        """
        pass
