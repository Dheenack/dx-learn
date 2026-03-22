"""
Lightweight validation of sklearn pipelines before full cross-validation.

Reduces wasted compute on structurally broken or incompatible pipelines.
"""

from __future__ import annotations

import hashlib
from typing import Any

from sklearn.base import clone

from dxlearn.encoding.node import PipelineNode


def pipeline_node_cache_key(individual: PipelineNode) -> str:
    """
    Stable cache key for a pipeline tree (matches GA duplicate-hash semantics).

    Args:
        individual: Encoded pipeline node.

    Returns:
        Hex digest string suitable for dict keys.
    """
    parts: list[str] = []
    if individual.preprocessor.key:
        parts.append(
            f"prep:{individual.preprocessor.key}:{sorted(individual.preprocessor.params.items())}"
        )
    else:
        parts.append("prep:None")
    parts.append(
        f"scaler:{individual.scaler.key}:{sorted(individual.scaler.params.items())}"
    )
    parts.append(
        f"clf:{individual.classifier.key}:{sorted(individual.classifier.params.items())}"
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def validate_pipeline(pipeline: Any, X: Any, y: Any, max_samples: int = 10) -> bool:
    """
    Fit a clone on a small subset to catch obvious failures before full CV.

    Args:
        pipeline: sklearn estimator or Pipeline.
        X: Feature matrix.
        y: Target vector.
        max_samples: Number of rows to use for the probe fit.

    Returns:
        True if probe fit succeeds, False otherwise.
    """
    try:
        n_rows = int(X.shape[0]) if hasattr(X, "shape") else len(X)
        n = min(max_samples, n_rows)
        if n <= 0:
            return False
        p = clone(pipeline)
        p.fit(X[:n], y[:n])
        return True
    except Exception:
        return False
