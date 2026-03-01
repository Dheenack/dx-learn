"""
Typed mutation operators: categorical replacement, int/float/log-scale mutation.

Applied to pipeline tree nodes with depth control.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from dxlearn.encoding.node import PipelineNode, PreprocessorNode, ScalerNode, ClassifierNode
from dxlearn.encoding.grammar import PIPELINE_GRAMMAR
from dxlearn.search_space.registry import get_registry


def mutate_pipeline_node(
    node: PipelineNode,
    mutation_rate: float,
    rng: Any,
) -> PipelineNode:
    """
    Mutate a pipeline node in-place or return a new node.

    Applies with probability mutation_rate:
    - Categorical: replace preprocessor/scaler/classifier key with another valid choice.
    - Integer/float/log: mutate one hyperparameter within bounds.

    Args:
        node: Pipeline tree to mutate.
        mutation_rate: Probability of mutating each mutable part.
        rng: Random number generator.

    Returns:
        Mutated pipeline node (new copy).
    """
    registry = get_registry()
    out = node.copy()

    # Mutate preprocessor key
    if rng.random() < mutation_rate:
        choices = PIPELINE_GRAMMAR.get_preprocessor_choices()
        new_key = choices[int(rng.integers(0, len(choices)))]
        new_params = registry.sample_preprocessor_params(new_key, rng) if new_key else {}
        out.preprocessor = PreprocessorNode(key=new_key, params=new_params)
    # Mutate preprocessor params
    elif out.preprocessor.key and rng.random() < mutation_rate:
        out.preprocessor.params = _mutate_params(
            out.preprocessor.params,
            out.preprocessor.key,
            registry._preprocessor_param_specs,
            rng,
        )

    # Mutate scaler key
    if rng.random() < mutation_rate:
        choices = PIPELINE_GRAMMAR.get_scaler_choices()
        new_key = choices[int(rng.integers(0, len(choices)))]
        out.scaler = ScalerNode(key=new_key, params=registry.sample_scaler_params(new_key, rng))
    elif rng.random() < mutation_rate:
        out.scaler.params = _mutate_params(
            out.scaler.params,
            out.scaler.key,
            registry._scaler_param_specs,
            rng,
        )

    # Mutate classifier key
    if rng.random() < mutation_rate:
        choices = PIPELINE_GRAMMAR.get_classifier_choices()
        new_key = choices[int(rng.integers(0, len(choices)))]
        out.classifier = ClassifierNode(key=new_key, params=registry.sample_classifier_params(new_key, rng))
    elif rng.random() < mutation_rate:
        out.classifier.params = _mutate_params(
            out.classifier.params,
            out.classifier.key,
            registry._classifier_param_specs,
            rng,
        )

    return out


def _mutate_params(
    params: Dict[str, Any],
    key: str,
    specs_dict: Dict[str, Dict[str, Any]],
    rng: Any,
) -> Dict[str, Any]:
    """Mutate one random parameter in params according to specs. Returns new dict."""
    specs = specs_dict.get(key, {})
    if not specs:
        return dict(params)
    param_names = [p for p in specs if p in params]
    if not param_names:
        return dict(params)
    name = param_names[int(rng.integers(0, len(param_names)))]
    kind, a, b = specs[name]
    new_params = dict(params)
    if kind == "int":
        low, high = int(a), int(b)
        delta = max(1, (high - low) // 5)
        current = int(params[name])
        new_params[name] = int(rng.integers(max(low, current - delta), min(high + 1, current + delta + 1)))
    elif kind == "float":
        low, high = float(a), float(b)
        current = float(params[name])
        sigma = (high - low) * 0.2
        new_params[name] = float(rng.uniform(max(low, current - sigma), min(high, current + sigma)))
    elif kind == "log":
        low, high = float(a), float(b)
        log_low, log_high = math.log(low), math.log(high)
        current = float(params[name])
        log_cur = math.log(max(current, 1e-10))
        sigma = (log_high - log_low) * 0.2
        new_val = math.exp(rng.uniform(max(log_low, log_cur - sigma), min(log_high, log_cur + sigma)))
        new_params[name] = new_val
    elif kind == "categorical":
        choices = list(a)
        new_params[name] = rng.choice(choices)
    return new_params
