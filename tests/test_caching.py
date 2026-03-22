"""
Fitness caching: cache is used so same pipeline is not re-evaluated.
"""
import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from dxlearn import DXClassifier
from dxlearn.engine.genetic_search import GeneticSearch, _tree_hash
from dxlearn.encoding.node import PipelineNode, PreprocessorNode, ScalerNode, ClassifierNode


def test_cache_grows_during_fit():
    """After fit with deterministic=True, cache should contain entries."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DXClassifier(
        population_size=6,
        generations=3,
        cv=2,
        verbose=0,
        n_jobs=1,
        deterministic=True,
        random_state=42,
    )
    model.fit(X_train, y_train)
    assert len(model._search._cache) >= 1


def test_same_individual_same_hash():
    """Tree hash is deterministic for same structure."""
    tree = PipelineNode(
        preprocessor=PreprocessorNode(key=None, params={}),
        scaler=ScalerNode(key="StandardScaler", params={}),
        classifier=ClassifierNode(key="LogisticRegression", params={"C": 1.0}),
    )
    h1 = _tree_hash(tree)
    h2 = _tree_hash(tree.copy())
    assert h1 == h2
