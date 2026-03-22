# `dxlearn` Comprehensive User Manual

Welcome to the `dxlearn` user manual! `dxlearn` is a production-grade, research-ready Automated Machine Learning (AutoML) Python package that discovers optimal classification pipelines using a **grammar-constrained Genetic Algorithm (GA)** strictly built on top of `scikit-learn`.

This manual covers everything from basic installation to the most advanced configurations of the `dxlearn` package.

---

## 1. Installation

To install the basic package, clone the repository or use pip to install it locally:

```bash
pip install dxlearn
```

To enable the analytical dashboard (which runs a FastAPI + uvicorn web server), install the `dashboard` extra:

```bash
pip install -e "dxlearn[dashboard]"
```

**Requirements:**
- Python 3.11+
- `numpy`
- `scikit-learn`
- `joblib`
- (Optional) `fastapi`, `uvicorn` (for the dashboard)

---

## 2. Core Concepts

`dxlearn` uses Evolutionary Search (a Genetic Algorithm) to evolve high-performing classification pipelines over multiple generations. 

### Grammar-Constrained Pipelines
To prevent the search space from growing out of control and generating poorly structured pipelines, `dxlearn` employs a grammar-based structure. Every generated pipeline follows this exact formula:

`[Optional Preprocessor] -> [Scaler] -> [Classifier]`

### Multi-Objective Fitness
Pipelines are not judged strictly on accuracy. `dxlearn` evaluates pipelines based on a custom scalarized multi-objective vector:
`Fitness = α * accuracy - β * fit_time - γ * complexity - δ * predict_time`
This ensures the final model is both accurate and computationally efficient.

---

## 3. Quick Start Guide

Here's the minimal setup required to train your first pipeline using `dxlearn`.

```python
from dxlearn import DXClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Prepare Data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize DXClassifier
model = DXClassifier(
    population_size=40,   # number of pipeline variants per generation
    generations=15,       # number of iterations of evolution
    cv=3,                 # k-folds for cross validation
    verbose=1,
    random_state=42       # for reproducibility
)

# 3. Fit the model to discover the best pipeline
model.fit(X_train, y_train)

# 4. Predict and evaluate
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print(f"Test Accuracy: {accuracy:.4f}")
print("Best Pipeline Found:", model.best_pipeline_)
print("Best Validation Score during CV:", model.best_score_)
```

---

## 4. API Reference

### `DXClassifier`

`DXClassifier` is the main entry point to the system and implements a fully `scikit-learn` compatible interface.

#### Initialization Parameters

You have deep control over the genetic algorithm's behavior:

- **`population_size`** (`int`, default `30`): Number of individual pipelines evaluated per generation.
- **`generations`** (`int`, default `20`): Maximum number of generations to run the search.
- **`cv`** (`int`, default `5`): Number of cross-validation folds.
- **`alpha`** (`float`, default `1.0`): Multiplier for `accuracy` in the fitness function.
- **`beta`** (`float`, default `0.2`): Penalty associated with `fit_time`.
- **`gamma`** (`float`, default `0.01`): Penalty associated with `complexity`.
- **`max_runtime`** (`float`, default `600.0`): Global maximum search time in seconds. Once reached, evolution returns the best individual so far.
- **`verbose`** (`int`, default `1`): Output verbosity level (0 = silent, 1 = basic, 2 = detailed).
- **`n_jobs`** (`int`, default `-1`): Number of parallel jobs for cross-validation and pipeline evaluation (-1 means all CPU cores).
- **`deterministic`** (`bool`, default `True`): If True, ensures repeatable results.
- **`random_state`** (`int`, default `42`): Seed for random number generators.
- **`elitism_count`** (`int`, default `2`): Number of top-performing individuals to carry over directly to the next generation without mutation.
- **`tournament_size`** (`int`, default `3`): Number of individuals randomly picked for tournament selection.
- **`mutation_rate`** (`float`, default `0.2`): The probability a pipeline will spontaneously mutate its structure or hyperparameters.
- **`crossover_rate`** (`float`, default `0.8`): The probability two pipelines will mate and exchange components.
- **`early_stopping_generations`** (`int`, default `5`): Stops the search early if the generation's best fitness has not improved for `N` generations.
- **`per_individual_timeout`** (`float`, default `60.0`): Max seconds allowed to evaluate a single pipeline. Prevents stalling on complex combinations.

#### Public Attributes
- **`best_pipeline_`**: The actual finalized `sklearn`-compliant `Pipeline` object found after `fit()`.
- **`best_score_`**: The maximum validation score attained by `best_pipeline_` via CV.

---

### Methods

#### `fit(X, y)`
Runs the multi-generational genetic search on `X` and `y`.
After the evolution process finds the absolute best pipeline, it automatically fits this best pipeline onto the **entire** dataset (`X`, `y`). Returns `self`.

#### `predict(X)`
Generates class label predictions on `X` using the best found pipeline.

#### `predict_proba(X)`
Generates class probability estimates on `X`. *(Note: the discovered classifier must support `predict_proba`, which most in the grammar do).*

#### `score(X, y)`
Returns the accuracy score on the given test data `X` and labels `y`.

#### `dashboard(host="127.0.0.1", port=8000)`
Spawns a local web server displaying visual analytics of the genetic algorithm run (Generational Evolution Curves, fitness scatterplots, etc.). Must be called **after** `fit()`.

#### `get_params(deep=True)` / `set_params(**params)`
Standard `scikit-learn` parameter access methods allowing dynamic updates to the search engine configs.

---

## 5. Components & Search Space

Pipelines are dynamically composed. `dxlearn` will randomly select, crossover, and mutate components from the following lists:

### 1. Preprocessors (Optional)
May be omitted (`None`), or apply structural transformations before scaling:
- `PCA` (Principal Component Analysis)
- `SelectKBest` (Univariate feature selection)
- `PolynomialFeatures`
- `VarianceThreshold`

### 2. Scalers (Mandatory)
Normalizes features:
- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`

### 3. Classifiers (Mandatory)
The predictive model:
- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `SVC` (Support Vector Classifier)
- `KNeighborsClassifier`
- `DecisionTreeClassifier`

*(Note: In the background, `dxlearn` also assigns optimal internal hyperparameters for these classes, e.g., the `C` parameter for `SVC` or `n_estimators` for `RandomForestClassifier`)*

---

## 6. Dashboard & Visualization

The `dxlearn` package includes a powerful diagnostic dashboard to visualize how the GA explored the search space.

```python
# Assuming model.fit(X_train, y_train) has already been called
model.dashboard(port=8000)
```

Point your browser to `http://127.0.0.1:8000`. The server runs via FastAPI and provides:
1. **Best Fitness vs Generation**: Line charts showing optimization progression.
2. **Mean Fitness**: Insights into population diversity.
3. **Accuracy vs Time Scatterplots**: Helps identify whether higher accuracy came at the cost of huge test/fit times.

---

## 7. Advanced Use Cases

### Forcing Strict Reproducibility
Evolutionary algorithms are highly stochastic. To guarantee identical pipelines across different identical script runs, strictly pass these limits:
```python
model = DXClassifier(
    deterministic=True, # Threading context limits standard BLAS variance
    random_state=42,    # Standard fixed seed
    n_jobs=1            # Disable multiprocessing randomness
)
```

### Global vs Individual Timeouts
When dealing with large datasets, evaluations can hang. `dxlearn` handles this gracefully:
```python
model = DXClassifier(
    max_runtime=3600,             # Entire search cuts off strictly after 1 hour
    per_individual_timeout=120    # If any single pipeline takes > 2 mins to fit, it is instantly killed
)
```

### Weighted Objective Penalties
If inference speed is more vital than accuracy to your application (e.g. edge-computing devices):
```python
model = DXClassifier(
    alpha=1.0,   # Keep Base Accuracy
    beta=0.0,    # Ignore Fit Time (Training can take forever, we don't care)
    gamma=0.01,  # Minor penalty to complex models
)
# Note: delta (predict_time penalty) will be implemented in future versions based on grammar configuration
```

