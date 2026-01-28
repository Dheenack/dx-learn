import shap

def compute_shap(model, X_background, X_sample):
    explainer = shap.Explainer(model.predict, X_background)
    shap_values = explainer(X_sample)
    return shap_values
