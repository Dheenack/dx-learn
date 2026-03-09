"""
Edge cases: small population, elitism, empty/minimal data, invalid config.
"""
import pytest
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split

from dxlearn import DXClassifier
from dxlearn.engine.genetic_search import GeneticSearch
from dxlearn.encoding.tree import tree_to_pipeline
from dxlearn.encoding.node import PipelineNode, PreprocessorNode, ScalerNode, ClassifierNode
from dxlearn.search_space.registry import get_registry


def test_small_population_and_generations():
    """Minimal run should still complete without crash."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DXClassifier(population_size=3, generations=2, cv=2, verbose=0, n_jobs=1, random_state=42)
    model.fit(X_train, y_train)
    assert model.best_pipeline_ is not None
    _ = model.predict(X_test)


def test_elitism_not_exceeds_population():
    """Elitism count should be clamped or population large enough."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # population_size=5, elitism_count=2 is valid
    model = DXClassifier(
        population_size=5,
        generations=3,
        elitism_count=2,
        cv=2,
        verbose=0,
        n_jobs=1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    assert model.best_pipeline_ is not None


def test_deterministic_reproducibility():
    """Same seed and data should give same best_score_ (with small tolerance for float)."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)
    model1 = DXClassifier(
        population_size=8,
        generations=4,
        cv=2,
        verbose=0,
        n_jobs=1,
        deterministic=True,
        random_state=123,
    )
    model2 = DXClassifier(
        population_size=8,
        generations=4,
        cv=2,
        verbose=0,
        n_jobs=1,
        deterministic=True,
        random_state=123,
    )
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    assert model1.best_score_ == model2.best_score_
    # Predictions on same pipeline type might differ if different pipeline chosen; at least scores match
    np.testing.assert_equal(model1.predict(X_test), model1.best_pipeline_.predict(X_test))
    np.testing.assert_equal(model2.predict(X_test), model2.best_pipeline_.predict(X_test))


def test_binary_classification():
    """Works with binary labels."""
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DXClassifier(population_size=6, generations=3, cv=2, verbose=0, n_jobs=1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))


def test_max_runtime_does_not_crash():
    """Very short max_runtime should stop early without error."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DXClassifier(
        population_size=4,
        generations=100,
        cv=2,
        max_runtime=1.0,
        verbose=0,
        n_jobs=1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    # May have run only one or a few generations
    assert model.best_pipeline_ is not None


def test_tree_to_pipeline_no_preprocessor():
    """Pipeline with preprocessor=None builds correctly."""
    registry = get_registry()
    tree = PipelineNode(
        preprocessor=PreprocessorNode(key=None, params={}),
        scaler=ScalerNode(key="StandardScaler", params={}),
        classifier=ClassifierNode(key="LogisticRegression", params={"C": 1.0, "max_iter": 100}),
    )
    pipe = tree_to_pipeline(tree, registry=registry)
    assert pipe is not None
    assert hasattr(pipe, "fit")
    X, y = load_iris(return_X_y=True)
    pipe.fit(X, y)
    preds = pipe.predict(X[:5])
    assert len(preds) == 5


def test_evaluator_handles_invalid_pipeline_gracefully():
    """Evaluator returns penalized fitness on bad pipeline (e.g. invalid params)."""
    from dxlearn.evaluation.evaluator import Evaluator

    X, y = load_iris(return_X_y=True)
    # Build a tree that might fail (e.g. PCA n_components > 1.0 or invalid)
    tree = PipelineNode(
        preprocessor=PreprocessorNode(key="PCA", params={"n_components": 0.5, "svd_solver": "auto"}),
        scaler=ScalerNode(key="StandardScaler", params={}),
        classifier=ClassifierNode(key="LogisticRegression", params={"C": 1.0, "max_iter": 100}),
    )
    ev = Evaluator(cv=2, random_state=42)
    fitness, objs = ev.evaluate(tree, X, y)
    # Should not raise; if pipeline fails we get penalized objectives
    assert isinstance(fitness, (int, float))
    assert objs.accuracy >= 0 and objs.accuracy <= 1


def test_invalid_elitism_raises():
    """elitism_count >= population_size raises ValueError."""
    with pytest.raises(ValueError, match="elitism_count"):
        DXClassifier(population_size=2, elitism_count=2, random_state=42)


def test_invalid_population_size_raises():
    """population_size < 2 raises ValueError."""
    with pytest.raises(ValueError, match="population_size"):
        DXClassifier(population_size=1, random_state=42)
