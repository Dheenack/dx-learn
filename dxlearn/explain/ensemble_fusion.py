def weighted_shap(shap_list, weights):
    fused = 0
    for shap_vals, w in zip(shap_list, weights):
        fused += w * shap_vals.values
    return fused
