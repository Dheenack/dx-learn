# dxlearn — Genetic Algorithm Driven AutoML Framework

**dxlearn** is a production-grade, research-ready AutoML Python package that discovers optimal classification pipelines using a **grammar-constrained Genetic Algorithm (GA)** built strictly on top of `scikit-learn`.

## Features

- **Grammar-constrained search**: Pipelines follow `<OptionalPreprocessor> <Scaler> <Classifier>`.
- **Multi-objective fitness**: Accuracy, fit time, predict time, and complexity (scalarized for selection).
- **sklearn-compatible API**: `fit`, `predict`, `predict_proba`, `score`, `get_params`, `set_params`.
- **Deterministic & reproducible**: Optional seeded RNG and fitness caching.
- **Extensible**: Base abstractions for regression, NSGA-II, and distributed GA (future).

## Requirements

- Python 3.11+
- numpy, scikit-learn, joblib

## Installation

```bash
pip install -e .
# With dashboard (FastAPI + uvicorn):
pip install -e ".[dashboard]"
```

## Quick Start

```python
from dxlearn import DXClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = DXClassifier(
    population_size=30,
    generations=20,
    cv=5,
    alpha=1.0,
    beta=0.2,
    gamma=0.01,
    max_runtime=600,
    verbose=2,
    n_jobs=-1,
    deterministic=True,
    random_state=42,
)

model.fit(X_train, y_train)
preds = model.predict(X_test)
print(model.score(X_test, y_test))
print(model.best_pipeline_)
print(model.best_score_)

# Optional: launch analytical dashboard
model.dashboard()  # requires pip install dxlearn[dashboard]
```

## Pipeline Grammar (v1)

- **OptionalPreprocessor**: `None` | `PCA` | `SelectKBest` | `PolynomialFeatures` | `VarianceThreshold`
- **Scaler**: `StandardScaler` | `MinMaxScaler` | `RobustScaler`
- **Classifier**: `LogisticRegression` | `RandomForestClassifier` | `GradientBoostingClassifier` | `SVC` | `KNeighborsClassifier` | `DecisionTreeClassifier`

## Fitness

Multi-objective vector: `(accuracy, fit_time, predict_time, complexity)`.  
Scalarized for selection: `α·accuracy − β·fit_time − γ·complexity − δ·predict_time` (default weights: α=1, β=0.2, γ=0.01).

## Dashboard

With `pip install dxlearn[dashboard]`, calling `model.dashboard()` starts a FastAPI server at `http://127.0.0.1:8000` with:

- Generation evolution curves (best fitness, best accuracy)
- Accuracy vs time scatter
- Mean fitness over generations
- Best metrics summary

## Package Layout

```
dxlearn/
├── base/           # BaseSearch, EvolutionarySearch, BaseDXEstimator
├── encoding/       # Grammar, tree, node (pipeline representation)
├── operators/      # Selection, crossover, mutation
├── search_space/   # Registry (scalers, preprocessors, classifiers)
├── evaluation/     # Evaluator, Objectives, Scalarizer
├── engine/         # GeneticSearch
├── dashboard/      # FastAPI dashboard (optional)
├── dxclassifier.py # Public API
└── config.py       # Defaults
```

## License

MIT.
