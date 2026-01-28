import time
from autosklearn2 import AutoMLClassifier

def run_automl(X, y, time_budget=300):
    automl = AutoMLClassifier(
        time_budget=time_budget,
        n_jobs=-1
    )

    start = time.time()
    automl.fit(X, y)
    duration = time.time() - start

    return {
        "automl": automl,
        "models": automl.get_models(),
        "leaderboard": automl.leaderboard(),
        "training_time": duration
    }
