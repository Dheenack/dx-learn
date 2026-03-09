# dxlearn — Test Report and Risk Analysis

## 1. Test Summary

| Suite | Purpose |
|-------|--------|
| `testv1.py` | Smoke: Iris, fit/predict/score, best_score_ and test accuracy |
| `test_api_contract.py` | Sklearn API: fit returns self, predict/score/proba before fit raise, get_params/set_params, best_pipeline_ consistency |
| `test_edge_cases.py` | Small population/generations, elitism, deterministic reproducibility, binary classification, max_runtime, tree_to_pipeline, evaluator robustness |
| `test_dashboard.py` | Dashboard ImportError when [dashboard] not installed, history structure |
| `test_caching.py` | Cache growth during fit, tree hash stability |

---

## 2. Problems and Bugs Found

### 2.1 Critical / High

1. **SelectKBest `k` can exceed `n_features`**  
   Registry uses `k in (5, 50)`. On small datasets (e.g. Iris with 4 features) this triggers sklearn `UserWarning` and redundant behavior. **Risk**: Noisy test output; in production small datasets get repeated warnings.

2. **`dashboard()` before `fit()`**  
   Calling `model.dashboard()` before `fit()` runs without error but shows empty evolution (no history). **Risk**: Confusing UX; users may think the dashboard is broken.

3. **`set_params()` does not clear fitted state**  
   After `set_params(...)`, `_estimator`, `best_pipeline_`, and `best_score_` remain from the previous fit. Next `predict()` uses the old pipeline until `fit()` is called again. **Risk**: Sklearn compatibility; tools like `GridSearchCV` may expect that changing params and calling `fit()` is enough. Not clearing state is acceptable per sklearn docs, but we do not reset `_search` internal state (e.g. `_best_individual`), so the dashboard could show stale history after `set_params` + `fit()`.

4. **Elitism / population bounds**  
   If `elitism_count >= population_size` (e.g. population_size=2, elitism_count=2), offspring count becomes 0 and the loop can still run with empty `offspring`. **Risk**: Wasted generations or edge-case crashes; validation is missing.

### 2.2 Medium

5. **`testv2.py` blocks forever**  
   `model.dashboard()` starts a blocking server. **Risk**: Unsuitable as an automated test; CI would hang.

6. **`predict_proba` when final estimator is SVC**  
   Pipeline with SVC needs `probability=True` for `predict_proba`. Our registry does not set it, so pipelines ending in SVC raise when user calls `predict_proba`. **Risk**: Inconsistent API; some runs work, others raise.

7. **Global registry singleton**  
   `get_registry()` is process-wide. Tests or code that mutate the registry affect others. **Risk**: Flaky tests or hidden coupling in multi-component apps.

8. **Exception swallowing in evaluator**  
   `Evaluator.evaluate()` catches all `Exception` and returns penalized fitness. **Risk**: Real bugs (e.g. wrong data shape) get hidden; hard to debug.

### 2.3 Low / Nice-to-have

9. **No `__sklearn_is_fitted__`**  
   DXClassifier does not set sklearn’s fitted flag. **Risk**: `sklearn.base.is_fitted()` or third-party code may not recognize the estimator as fitted.

10. **Verbose logging**  
    At `verbose=1`, logging goes to root logger; no explicit handler. **Risk**: In libraries that configure logging, output may be missing or duplicated.

11. **Typo in testv2.py**  
    `print(model.score(xtest,ytest))` has an extra closing parenthesis. **Risk**: SyntaxError when run.

---

## 3. Potential Risks and Downfalls

- **Reproducibility**: With `deterministic=False` or different `n_jobs`, results can vary; documentation could state this clearly.
- **Memory**: Fitness cache and history grow with run length; very long runs could use significant memory.
- **Timeout**: `per_individual_timeout` is not enforced in the current evaluator (no actual timeout around `cross_validate`); long-running pipelines can block.
- **Dashboard port conflict**: If port 8000 is in use, `dashboard()` will fail; no retry or clear error.
- **Regression / multi-label**: Only classification is supported; misuse (e.g. regression targets) may surface as obscure sklearn or evaluator errors.

---

## 4. Refactoring Plan

1. **Cap SelectKBest `k`** in the registry (e.g. max 20) and/or at build time using `min(k, n_features)` when building the pipeline (requires passing `n_features` or doing it in evaluator/tree_to_pipeline). Prefer registry cap + doc for now.
2. **Validate elitism/population** in `GeneticSearch.__init__`: ensure `elitism_count < population_size` and optionally `population_size >= 2`.
3. **Improve `dashboard()` UX**: If `_search` has no history (e.g. not fitted), raise a clear error or show an “Run fit() first” message in the dashboard.
4. **Fix testv2.py**: Remove blocking `model.dashboard()` from the script or guard it with a flag so CI can run tests without starting the server; fix the print typo.
5. **SVC and predict_proba**: Either add `probability=True` to SVC params in the registry, or document that `predict_proba` is only supported when the best pipeline has it, and ensure `BaseDXEstimator.predict_proba` error message is clear.
6. **Optional: fitted state** after `set_params`: Consider clearing `_estimator`, `best_pipeline_`, `best_score_` in `set_params` so that “not fitted” is consistent until the next `fit()`.

Refactoring will implement (1), (2), (3), (4), and (5) as far as possible without breaking the existing API.
