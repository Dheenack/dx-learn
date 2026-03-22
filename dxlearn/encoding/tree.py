"""
Convert PipelineNode to sklearn Pipeline and optionally parse back.

Tree representation is grammar-constrained: OptionalPreprocessor, Scaler, Classifier.
"""

from __future__ import annotations

from typing import Any, Optional

from sklearn.pipeline import Pipeline as SklearnPipeline

from dxlearn.encoding.node import PipelineNode, PreprocessorNode, ScalerNode, ClassifierNode
from dxlearn.search_space.registry import ComponentRegistry, get_registry


def tree_to_pipeline(
    tree: PipelineNode,
    registry: Optional[ComponentRegistry] = None,
    n_features: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Any:
    """
    Build an sklearn Pipeline from a PipelineNode.

    Args:
        tree: Root pipeline node (preprocessor + scaler + classifier).
        registry: Component registry; uses global if None.
        n_features: Number of input features (``X.shape[1]``). When set,
            ``SelectKBest`` uses ``k = min(k, n_features)``.
        random_state: If set, passed to sklearn steps that support it (reproducible CV).

    Returns:
        sklearn Pipeline or a single estimator if no steps.
    """
    if registry is None:
        registry = get_registry()
    steps: list[tuple[str, Any]] = []

    # Optional preprocessor
    if tree.preprocessor.key is not None:
        prep = registry.build_preprocessor(
            tree.preprocessor.key,
            tree.preprocessor.params,
            n_features=n_features,
        )
        steps.append(("preprocessor", prep))

    # Scaler
    scaler = registry.build_scaler(tree.scaler.key, tree.scaler.params)
    steps.append(("scaler", scaler))

    # Classifier
    clf = registry.build_classifier(
        tree.classifier.key, tree.classifier.params, random_state=random_state
    )
    steps.append(("classifier", clf))

    if not steps:
        return clf
    return SklearnPipeline(steps=steps)


def pipeline_to_tree(pipeline: Any) -> Optional[PipelineNode]:
    """
    Parse an sklearn Pipeline back into a PipelineNode (best-effort).

    Only supports pipelines built by tree_to_pipeline with standard step names.
    Returns None if the pipeline structure is not recognized.
    """
    if not hasattr(pipeline, "steps"):
        return None
    steps = getattr(pipeline, "steps", []) or []
    preprocessor = PreprocessorNode(key=None, params={})
    scaler_node: Optional[ScalerNode] = None
    classifier_node: Optional[ClassifierNode] = None
    registry = get_registry()

    for name, est in steps:
        cls_name = type(est).__name__
        if name == "preprocessor":
            if cls_name in registry.get_preprocessor_keys():
                preprocessor = PreprocessorNode(key=cls_name, params=getattr(est, "get_params", lambda: {})())
                specs = registry._preprocessor_param_specs.get(cls_name, {})
                preprocessor.params = {k: v for k, v in preprocessor.params.items() if k in specs}
        elif name == "scaler" and cls_name in registry.get_scaler_keys():
            scaler_node = ScalerNode(key=cls_name, params={})
        elif name == "classifier" and cls_name in registry.get_classifier_keys():
            params = getattr(est, "get_params", lambda: {})()
            specs = registry._classifier_param_specs.get(cls_name, {})
            classifier_node = ClassifierNode(key=cls_name, params={k: params[k] for k in specs if k in params})

    if scaler_node is None or classifier_node is None:
        return None
    return PipelineNode(preprocessor=preprocessor, scaler=scaler_node, classifier=classifier_node)
