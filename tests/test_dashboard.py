"""
Dashboard: optional dependency and API without starting server.
"""
import pytest


def test_dashboard_import_error_without_extra():
    """Without [dashboard] extra, dashboard() should raise ImportError with helpful message."""
    try:
        from dxlearn.dashboard.api import run_dashboard
        HAS_DASHBOARD = True
    except ImportError:
        HAS_DASHBOARD = False

    if HAS_DASHBOARD:
        pytest.skip("Dashboard dependencies installed; cannot test ImportError")

    from dxlearn import DXClassifier

    model = DXClassifier(population_size=2, generations=1, cv=2, verbose=0, n_jobs=1, random_state=42)
    with pytest.raises(ImportError) as exc_info:
        model.dashboard()
    assert "dashboard" in str(exc_info.value).lower() or "pip install" in str(exc_info.value)


def test_dashboard_data_structure_without_server():
    """Verify GeneticSearch exposes history for dashboard (no server start)."""
    from dxlearn import DXClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DXClassifier(population_size=4, generations=2, cv=2, verbose=0, n_jobs=1, random_state=42)
    model.fit(X_train, y_train)
    history = model._search.get_history()
    assert isinstance(history, list)
    if history:
        assert "generation" in history[0]
        assert "best_fitness" in history[0]
        assert "best_accuracy" in history[0]


def test_dashboard_before_fit_raises():
    """Calling dashboard() before fit() raises RuntimeError."""
    from dxlearn import DXClassifier

    model = DXClassifier(population_size=2, generations=1, cv=2, verbose=0, n_jobs=1, random_state=42)
    with pytest.raises(RuntimeError, match="fit.*before dashboard"):
        model.dashboard()
