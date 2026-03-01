"""
Grammar definition for v1 pipeline search space.

<Pipeline> ::= <OptionalPreprocessor> <Scaler> <Classifier>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Grammar:
    """
    Grammar-constrained search space definition.

    Defines valid components for each slot in the pipeline
    and optional hyperparameter bounds.
    """

    # Optional preprocessor: None or one of the listed keys
    optional_preprocessors: List[Optional[str]] = field(
        default_factory=lambda: [
            None,
            "PCA",
            "SelectKBest",
            "PolynomialFeatures",
            "VarianceThreshold",
        ]
    )
    scalers: List[str] = field(
        default_factory=lambda: ["StandardScaler", "MinMaxScaler", "RobustScaler"]
    )
    classifiers: List[str] = field(
        default_factory=lambda: [
            "LogisticRegression",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "SVC",
            "KNeighborsClassifier",
            "DecisionTreeClassifier",
        ]
    )
    # Hyperparameter specs: component_key -> { param_name -> (type, low, high or choices) }
    hyperparameter_specs: Dict[str, Dict[str, Tuple[str, Any, Any]]] = field(
        default_factory=dict
    )

    def get_preprocessor_choices(self) -> List[Optional[str]]:
        """Return list of valid optional preprocessor names (including None)."""
        return list(self.optional_preprocessors)

    def get_scaler_choices(self) -> List[str]:
        """Return list of valid scaler names."""
        return list(self.scalers)

    def get_classifier_choices(self) -> List[str]:
        """Return list of valid classifier names."""
        return list(self.classifiers)

    def get_hyperparameter_spec(
        self, component_key: str, param_name: str
    ) -> Optional[Tuple[str, Any, Any]]:
        """Return (type, low, high) or (type, choices) for a component's param."""
        specs = self.hyperparameter_specs.get(component_key, {})
        return specs.get(param_name)


# Singleton grammar instance for v1 (will be extended with hyperparameter_specs in registry)
PIPELINE_GRAMMAR = Grammar()
