"""Pydantic schemas for dashboard API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore

if BaseModel is not None:

    class GenerationRecord(BaseModel):
        """One generation's summary."""

        generation: int
        best_fitness: float
        best_accuracy: float
        mean_fitness: float

    class DashboardData(BaseModel):
        """Aggregated data for the dashboard."""

        history: List[GenerationRecord]
        best_accuracy: Optional[float] = None
        best_fitness: Optional[float] = None

else:

    class GenerationRecord:
        generation: int = 0
        best_fitness: float = 0.0
        best_accuracy: float = 0.0
        mean_fitness: float = 0.0

    class DashboardData:
        history: List[Any] = []
        best_accuracy: Optional[float] = None
        best_fitness: Optional[float] = None
