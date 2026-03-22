"""
Run one deterministic DXClassifier fit; print score and best-tree hash.
Executed via subprocess by test_deterministic_reproducibility (not a pytest module).
"""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from dxlearn import DXClassifier
from dxlearn.engine.genetic_search import _tree_hash

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(X, y, random_state=99)
    model = DXClassifier(
        population_size=8,
        generations=4,
        cv=2,
        verbose=0,
        n_jobs=1,
        deterministic=True,
        random_state=123,
    )
    model.fit(X_train, y_train)
    h = _tree_hash(model._search._best_individual)
    print(f"{model.best_score_!r}\t{h}")
