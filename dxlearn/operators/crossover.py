"""
Subtree crossover with depth control.

For v1 grammar, crossover swaps one of: preprocessor, scaler, or classifier
(subtree) between two parents to produce two children.
"""

from __future__ import annotations

from typing import Any, Tuple

from dxlearn.encoding.node import PipelineNode


def subtree_crossover(
    parent_a: PipelineNode,
    parent_b: PipelineNode,
    rng: Any,
    max_depth: int = 4,
) -> Tuple[PipelineNode, PipelineNode]:
    """
    Perform subtree crossover between two pipeline trees.

    With the v1 flat structure, we swap one of preprocessor / scaler / classifier
    between parents to get two children. Depth is preserved.

    Args:
        parent_a: First parent pipeline node.
        parent_b: Second parent pipeline node.
        rng: Random number generator.
        max_depth: Maximum allowed tree depth (unused in v1 flat structure).

    Returns:
        (child_a, child_b) as new PipelineNodes.
    """
    # Choose which slot to swap: 0=preprocessor, 1=scaler, 2=classifier
    slot = int(rng.integers(0, 3))
    child_a = parent_a.copy()
    child_b = parent_b.copy()

    if slot == 0:
        child_a.preprocessor = parent_b.preprocessor.copy()
        child_b.preprocessor = parent_a.preprocessor.copy()
    elif slot == 1:
        child_a.scaler = parent_b.scaler.copy()
        child_b.scaler = parent_a.scaler.copy()
    else:
        child_a.classifier = parent_b.classifier.copy()
        child_b.classifier = parent_a.classifier.copy()

    return child_a, child_b
