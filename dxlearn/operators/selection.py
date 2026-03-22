"""
Selection operators: tournament and roulette.

Used by the GA engine to select parents for the next generation.
"""

from __future__ import annotations

from typing import Any, List


def tournament_selection(
    population: List[Any],
    fitnesses: List[float],
    k: int,
    tournament_size: int,
    rng: Any,
) -> List[Any]:
    """
    Select k individuals using tournament selection.

    Higher fitness is better. Each tournament picks the best of
    tournament_size random individuals.

    Args:
        population: List of individuals.
        fitnesses: Scalar fitness for each individual (higher is better).
        k: Number of individuals to select.
        tournament_size: Size of each tournament.
        rng: Random number generator.

    Returns:
        List of k selected individuals (with replacement).
    """
    n = len(population)
    if n == 0 or k == 0:
        return []
    tsize = min(tournament_size, n)
    selected = []
    for _ in range(k):
        # Unique contestants when possible → better diversity than sampling with replacement.
        if tsize <= n:
            indices = rng.choice(n, size=tsize, replace=False)
        else:
            indices = rng.integers(0, n, size=tsize)
        best_idx = int(indices[0])
        best_f = fitnesses[best_idx]
        for i in indices[1:]:
            ii = int(i)
            if fitnesses[ii] > best_f:
                best_f = fitnesses[ii]
                best_idx = ii
        selected.append(population[best_idx])
    return selected


def roulette_selection(
    population: List[Any],
    fitnesses: List[float],
    k: int,
    rng: Any,
) -> List[Any]:
    """
    Select k individuals using roulette (fitness-proportionate) selection.

    Fitnesses must be non-negative. Uses normalized weights.

    Args:
        population: List of individuals.
        fitnesses: Scalar fitness for each individual (higher is better).
        k: Number of individuals to select.
        rng: Random number generator.

    Returns:
        List of k selected individuals (with replacement).
    """
    n = len(population)
    if n == 0 or k == 0:
        return []
    import numpy as np
    arr = np.asarray(fitnesses, dtype=float)
    min_f = arr.min()
    # shift so all non-negative (avoid zero prob for worst)
    arr = arr - min_f + 1e-8
    probs = arr / arr.sum()
    indices = rng.choice(n, size=k, replace=True, p=probs)
    return [population[i] for i in indices]
