from dxlearn import DXClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def test_dxclassifier_iris():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = DXClassifier(population_size=10, generations=5, cv=3, verbose=0, n_jobs=1, random_state=42)
    model.fit(X_train, y_train)
    assert model.best_score_ is not None
    assert model.score(X_test, y_test) > 0.9