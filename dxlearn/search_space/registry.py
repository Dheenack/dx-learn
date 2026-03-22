"""
Component registry: maps keys to sklearn classes and default hyperparameter bounds.

Used by tree_to_pipeline to build Pipeline instances from PipelineNode.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Type alias: (type, low, high) for numeric or (type, choices) for categorical
ParamSpec = Tuple[str, Any, Any]


def _int_bounded(low: int, high: int, rng: Any) -> int:
    return int(rng.integers(low, high + 1))


def _float_bounded(low: float, high: float, rng: Any) -> float:
    return float(rng.uniform(low, high))


def _log_uniform(low: float, high: float, rng: Any) -> float:
    import math
    log_low = math.log(low)
    log_high = math.log(high)
    return float(math.exp(rng.uniform(log_low, log_high)))


def _choice(choices: List[Any], rng: Any) -> Any:
    return rng.choice(choices)


class ComponentRegistry:
    """
    Registry of sklearn components and their hyperparameter specs.

    Supports building instances from key + params dict and sampling
    random valid params for a key.
    """

    def __init__(self) -> None:
        self._preprocessors: Dict[str, Type[BaseEstimator]] = {
            "PCA": PCA,
            "SelectKBest": SelectKBest,
            "PolynomialFeatures": PolynomialFeatures,
            "VarianceThreshold": VarianceThreshold,
        }
        self._scalers: Dict[str, Type[BaseEstimator]] = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
        }
        self._classifiers: Dict[str, Type[BaseEstimator]] = {
            "LogisticRegression": LogisticRegression,
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "SVC": SVC,
            "KNeighborsClassifier": KNeighborsClassifier,
            "DecisionTreeClassifier": DecisionTreeClassifier,
        }
        # param_specs: key -> param_name -> ("int"|"float"|"log"|"categorical", a, b)
        # for int/float/log: (low, high); for categorical: (choices list, _, _) stored as ("categorical", choices, None)
        self._preprocessor_param_specs: Dict[str, Dict[str, ParamSpec]] = {
            "PCA": {
                "n_components": ("float", 0.1, 0.99),
                "svd_solver": ("categorical", ["auto", "full"], None),
            },
            "SelectKBest": {
                "k": ("int", 1, 20),
            },
            "PolynomialFeatures": {
                "degree": ("int", 2, 3),
                "include_bias": ("categorical", [True, False], None),
            },
            "VarianceThreshold": {
                "threshold": ("float", 1e-5, 1.0),
            },
        }
        self._scaler_param_specs: Dict[str, Dict[str, ParamSpec]] = {}
        self._classifier_param_specs: Dict[str, Dict[str, ParamSpec]] = {
            "LogisticRegression": {
                "C": ("log", 0.01, 100.0),
                "max_iter": ("int", 100, 2000),
                # lbfgs only: saga touches global RNG and is sensitive to max_iter / thread order.
                "solver": ("categorical", ["lbfgs"], None),
            },
            "RandomForestClassifier": {
                "n_estimators": ("int", 10, 200),
                "max_depth": ("int", 2, 20),
                "min_samples_split": ("int", 2, 20),
            },
            "GradientBoostingClassifier": {
                "n_estimators": ("int", 10, 150),
                "max_depth": ("int", 2, 10),
                "learning_rate": ("log", 0.01, 0.5),
            },
            "SVC": {
                "C": ("log", 0.01, 100.0),
                "kernel": ("categorical", ["rbf", "linear", "poly"], None),
                "gamma": ("categorical", ["scale", "auto"], None),
                "probability": ("categorical", [True], None),
            },
            "KNeighborsClassifier": {
                "n_neighbors": ("int", 1, 30),
                "weights": ("categorical", ["uniform", "distance"], None),
            },
            "DecisionTreeClassifier": {
                "max_depth": ("int", 2, 20),
                "min_samples_split": ("int", 2, 20),
            },
        }

    def build_preprocessor(
        self,
        key: str,
        params: Optional[Dict[str, Any]] = None,
        n_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> BaseEstimator:
        """Build a preprocessor instance by key with given params.

        Args:
            key: Preprocessor registry key.
            params: Constructor keyword arguments.
            n_features: If set, ``SelectKBest`` ``k`` is clamped to
                ``min(k, n_features)`` to avoid sklearn warnings and redundant
                behavior when ``k`` exceeds the number of input features.
        """
        params = dict(params or {})
        cls = self._preprocessors.get(key)
        if cls is None:
            raise ValueError(f"Unknown preprocessor: {key}")
        if key == "SelectKBest":
            params.setdefault("score_func", f_classif)
            if n_features is not None and n_features > 0 and "k" in params:
                k_raw = params["k"]
                if isinstance(k_raw, (int, np.integer)):
                    params["k"] = min(int(k_raw), int(n_features))
        import inspect
        sig = inspect.signature(cls.__init__)
        if random_state is not None and "random_state" in sig.parameters:
            params.setdefault("random_state", random_state)
        valid = {k: v for k, v in params.items() if k in sig.parameters}
        return cls(**valid)

    def build_scaler(self, key: str, params: Optional[Dict[str, Any]] = None) -> BaseEstimator:
        """Build a scaler instance by key."""
        params = params or {}
        cls = self._scalers.get(key)
        if cls is None:
            raise ValueError(f"Unknown scaler: {key}")
        return cls(**params)

    def build_classifier(
        self,
        key: str,
        params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ) -> BaseEstimator:
        """Build a classifier instance by key."""
        params = dict(params or {})
        cls = self._classifiers.get(key)
        if cls is None:
            raise ValueError(f"Unknown classifier: {key}")
        if key == "SVC":
            params.setdefault("probability", True)
        import inspect
        sig = inspect.signature(cls.__init__)
        if random_state is not None and "random_state" in sig.parameters:
            params.setdefault("random_state", random_state)
        valid = {k: v for k, v in params.items() if k in sig.parameters}
        return cls(**valid)

    def sample_preprocessor_params(self, key: str, rng: Any) -> Dict[str, Any]:
        """Sample random valid params for a preprocessor key."""
        specs = self._preprocessor_param_specs.get(key, {})
        return self._sample_params(specs, rng)

    def sample_scaler_params(self, key: str, rng: Any) -> Dict[str, Any]:
        """Sample random valid params for a scaler key."""
        specs = self._scaler_param_specs.get(key, {})
        return self._sample_params(specs, rng)

    def sample_classifier_params(self, key: str, rng: Any) -> Dict[str, Any]:
        """Sample random valid params for a classifier key."""
        specs = self._classifier_param_specs.get(key, {})
        return self._sample_params(specs, rng)

    def _sample_params(self, specs: Dict[str, ParamSpec], rng: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for param_name, (kind, a, b) in specs.items():
            if kind == "int":
                out[param_name] = _int_bounded(int(a), int(b), rng)
            elif kind == "float":
                out[param_name] = _float_bounded(float(a), float(b), rng)
            elif kind == "log":
                out[param_name] = _log_uniform(float(a), float(b), rng)
            elif kind == "categorical":
                out[param_name] = _choice(a, rng)
            else:
                raise ValueError(f"Unknown param kind: {kind}")
        return out

    def get_preprocessor_keys(self) -> List[str]:
        return list(self._preprocessors.keys())

    def get_scaler_keys(self) -> List[str]:
        return list(self._scalers.keys())

    def get_classifier_keys(self) -> List[str]:
        return list(self._classifiers.keys())


_registry: Optional[ComponentRegistry] = None


def get_registry() -> ComponentRegistry:
    """Return the global component registry singleton."""
    global _registry
    if _registry is None:
        _registry = ComponentRegistry()
    return _registry
