"""
Configuration and default parameters for dxlearn.

Centralizes tunable constants for the genetic search engine,
evaluation, and dashboard.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# GA defaults
# ---------------------------------------------------------------------------
DEFAULT_POPULATION_SIZE: int = 30
DEFAULT_GENERATIONS: int = 20
DEFAULT_ELITISM_COUNT: int = 2
DEFAULT_TOURNAMENT_SIZE: int = 3
DEFAULT_MUTATION_RATE: float = 0.2
DEFAULT_CROSSOVER_RATE: float = 0.8
DEFAULT_EARLY_STOPPING_GENERATIONS: int = 5
DEFAULT_MAX_RUNTIME_SECONDS: float = 600.0
DEFAULT_PER_INDIVIDUAL_TIMEOUT_SECONDS: float = 60.0

# ---------------------------------------------------------------------------
# Fitness scalarization (α, β, γ)
# ---------------------------------------------------------------------------
DEFAULT_ALPHA: float = 1.0   # accuracy weight
DEFAULT_BETA: float = 0.2    # fit_time penalty
DEFAULT_GAMMA: float = 0.01  # complexity penalty

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
DEFAULT_CV: int = 5
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_VERBOSE: int = 1
DEFAULT_N_JOBS: int = -1

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
DASHBOARD_HOST: str = "127.0.0.1"
DASHBOARD_PORT: int = 8000
DASHBOARD_URL: str = f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"

# ---------------------------------------------------------------------------
# Tree / grammar constraints
# ---------------------------------------------------------------------------
MAX_TREE_DEPTH: int = 4
MAX_NODE_COUNT: int = 50


def get_default_params() -> dict[str, Any]:
    """Return a dict of default parameters for DXClassifier / GeneticSearch."""
    return {
        "population_size": DEFAULT_POPULATION_SIZE,
        "generations": DEFAULT_GENERATIONS,
        "elitism_count": DEFAULT_ELITISM_COUNT,
        "tournament_size": DEFAULT_TOURNAMENT_SIZE,
        "mutation_rate": DEFAULT_MUTATION_RATE,
        "crossover_rate": DEFAULT_CROSSOVER_RATE,
        "early_stopping_generations": DEFAULT_EARLY_STOPPING_GENERATIONS,
        "max_runtime": DEFAULT_MAX_RUNTIME_SECONDS,
        "per_individual_timeout": DEFAULT_PER_INDIVIDUAL_TIMEOUT_SECONDS,
        "cv": DEFAULT_CV,
        "alpha": DEFAULT_ALPHA,
        "beta": DEFAULT_BETA,
        "gamma": DEFAULT_GAMMA,
        "random_state": DEFAULT_RANDOM_STATE,
        "verbose": DEFAULT_VERBOSE,
        "n_jobs": DEFAULT_N_JOBS,
        "deterministic": True,
    }
