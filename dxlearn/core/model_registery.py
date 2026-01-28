class ModelRegistry:
    def __init__(self):
        self.models = []

    def register(self, model_info: dict):
        self.models.append(model_info)

    def list(self):
        return self.models
