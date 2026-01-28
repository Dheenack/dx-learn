import numpy as np

def shap_sparsity(shap_values, threshold=0.01):
    mean_abs = np.mean(np.abs(shap_values.values), axis=0)
    active = np.sum(mean_abs > threshold)
    return 1 - (active / len(mean_abs))
