def attribute_pipeline(pipeline, shap_values):
    return {
        "pipeline_steps": [name for name, _ in pipeline.steps],
        "shap_sum": shap_values.values.sum(axis=1).tolist()
    }
