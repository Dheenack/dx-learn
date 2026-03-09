"""
API contract tests: sklearn compatibility, fit/predict/score, get_params/set_params.
"""
import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from dxlearn import DXClassifier


def test_fit_returns_self():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DXClassifier(population_size=4, generations=2, cv=2, verbose=0, n_jobs=1, random_state=42)
    out = model.fit(X_train, y_train)
    assert out is model


def test_predict_before_fit_raises():
    model = DXClassifier(population_size=4, generations=2, cv=2, verbose=0, n_jobs=1, random_state=42)
    X, _ = load_iris(return_X_y=True)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(X)


def test_score_before_fit_raises():
    model = DXClassifier(population_size=4, generations=2, cv=2, verbose=0, n_jobs=1, random_state=42)
    X, y = load_iris(return_X_y=True)
    with pytest.raises(RuntimeError, match="fit"):
        model.score(X, y)


def test_predict_proba_before_fit_raises():
    model = DXClassifier(population_size=4, generations=2, cv=2, verbose=0, n_jobs=1, random_state=42)
    X, _ = load_iris(return_X_y=True)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(X)


def test_get_params_returns_dict():
    model = DXClassifier(population_size=10, generations=5, cv=3, random_state=42)
    params = model.get_params()
    assert isinstance(params, dict)
    assert params["population_size"] == 10
    assert params["random_state"] == 42


def test_set_params_returns_self():
    model = DXClassifier(population_size=10, generations=5, random_state=42)
    out = model.set_params(generations=3)
    assert out is model
    assert model.generations == 3


def test_best_pipeline_and_score_after_fit():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DXClassifier(population_size=6, generations=3, cv=2, verbose=0, n_jobs=1, random_state=42)
    model.fit(X_train, y_train)
    assert model.best_pipeline_ is not None
    assert model.best_score_ is not None
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert np.array_equal(preds, model.best_pipeline_.predict(X_test))


def test_predict_proba_when_supported():
    """predict_proba works when best pipeline supports it (e.g. LogisticRegression, SVC with probability=True)."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DXClassifier(population_size=6, generations=3, cv=2, verbose=0, n_jobs=1, random_state=42)
    model.fit(X_train, y_train)
    # All our classifiers (with SVC probability=True) support predict_proba when used in pipeline
    P = model.predict_proba(X_test)
    assert P.shape[0] == len(X_test)
    assert P.shape[1] == len(np.unique(y))
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_score_equals_pipeline_score():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DXClassifier(population_size=6, generations=3, cv=2, verbose=0, n_jobs=1, random_state=42)
    model.fit(X_train, y_train)
    expected = model.best_pipeline_.score(X_test, y_test)
    assert model.score(X_test, y_test) == expected
