"""
Tree node types for grammar-constrained pipeline representation.

PipelineNode -> PreprocessorNode (optional), ScalerNode, ClassifierNode
ClassifierNode may have HyperparameterNodes (stored as dict in practice).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HyperparameterNode:
    """A single hyperparameter: name and value."""

    name: str
    value: Any  # int, float, str, bool

    def copy(self) -> HyperparameterNode:
        """Return a deep copy."""
        return HyperparameterNode(name=self.name, value=self.value)


@dataclass
class PreprocessorNode:
    """Optional preprocessor: None or component key (e.g. PCA, SelectKBest)."""

    key: Optional[str]  # None means no preprocessor
    params: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> PreprocessorNode:
        return PreprocessorNode(key=self.key, params=dict(self.params))


@dataclass
class ScalerNode:
    """Scaler component: key and optional params."""

    key: str
    params: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> ScalerNode:
        return ScalerNode(key=self.key, params=dict(self.params))


@dataclass
class ClassifierNode:
    """Classifier with key and hyperparameters."""

    key: str
    params: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> ClassifierNode:
        return ClassifierNode(key=self.key, params=dict(self.params))


@dataclass
class PipelineNode:
    """
    Root node: OptionalPreprocessor + Scaler + Classifier.

    Depth and node count are strictly limited by the grammar.
    """

    preprocessor: PreprocessorNode
    scaler: ScalerNode
    classifier: ClassifierNode

    def copy(self) -> PipelineNode:
        return PipelineNode(
            preprocessor=self.preprocessor.copy(),
            scaler=self.scaler.copy(),
            classifier=self.classifier.copy(),
        )

    def node_count(self) -> int:
        """Total number of nodes (1 root + preprocessor + scaler + classifier + param nodes)."""
        n = 3  # preprocessor, scaler, classifier
        if self.preprocessor.key is not None:
            n += 1
        n += len(self.preprocessor.params)
        n += len(self.scaler.params)
        n += len(self.classifier.params)
        return n

    def depth(self) -> int:
        """Tree depth (flat structure in v1: depth 2)."""
        return 2
