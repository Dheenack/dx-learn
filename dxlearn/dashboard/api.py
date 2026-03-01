"""
FastAPI dashboard: evolution curves, accuracy vs time, Pareto, hyperparameter plots.

Launches at http://127.0.0.1:8000 when model.dashboard() is called.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from dxlearn.dashboard.schemas import DashboardData, GenerationRecord


def _get_dashboard_data(search: Any) -> dict[str, Any]:
    """Build dashboard payload from GeneticSearch instance."""
    history = getattr(search, "_history", []) or getattr(search, "get_history", lambda: [])()
    objs = getattr(search, "get_best_objectives", lambda: None)()
    records = [
        {
            "generation": h["generation"],
            "best_fitness": h["best_fitness"],
            "best_accuracy": h["best_accuracy"],
            "mean_fitness": h["mean_fitness"],
        }
        for h in history
    ]
    return {
        "history": records,
        "best_accuracy": objs.accuracy if objs else None,
        "best_fitness": getattr(search, "_best_fitness", None),
        "best_fit_time": objs.fit_time if objs else None,
        "best_predict_time": objs.predict_time if objs else None,
        "best_complexity": objs.complexity if objs else None,
    }


def run_dashboard(
    search: Any,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """
    Start the FastAPI dashboard server. Blocks until server is stopped.

    Args:
        search: GeneticSearch instance (after fit) with get_history(), get_best_objectives().
        host: Bind host.
        port: Bind port.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        from uvicorn import run as uvicorn_run
    except ImportError:
        raise ImportError("Install dashboard deps: pip install dxlearn[dashboard]")

    app = FastAPI(title="dxlearn Dashboard", version="0.1.0")
    _search_ref: Optional[Any] = None

    def set_search(s: Any) -> None:
        nonlocal _search_ref
        _search_ref = s

    set_search(search)

    @app.get("/api/data", response_model=None)
    def get_data() -> dict:
        if _search_ref is None:
            return {"history": [], "best_accuracy": None, "best_fitness": None}
        return _get_dashboard_data(_search_ref)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _get_html()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    uvicorn_run(app, host=host, port=port, log_level="warning")


def _get_html() -> str:
    """Return full HTML for the dashboard with embedded JS and Chart.js."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>dxlearn Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #0f0f12; color: #e4e4e7; }
    h1 { font-size: 1.5rem; margin-bottom: 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
    .card { background: #18181b; border-radius: 8px; padding: 16px; }
    .card h2 { font-size: 0.95rem; margin: 0 0 12px 0; color: #a1a1aa; }
    canvas { max-height: 220px; }
    .metrics { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 20px; }
    .metric { background: #18181b; padding: 12px 16px; border-radius: 8px; }
    .metric span { display: block; font-size: 0.8rem; color: #71717a; }
    .metric strong { font-size: 1.2rem; }
  </style>
</head>
<body>
  <h1>dxlearn — Evolution Dashboard</h1>
  <div class="metrics" id="metrics"></div>
  <div class="grid">
    <div class="card"><h2>Best fitness & accuracy over generations</h2><canvas id="chartEvolution"></canvas></div>
    <div class="card"><h2>Accuracy vs fit time (last generation)</h2><canvas id="chartAccuracyTime"></canvas></div>
    <div class="card"><h2>Mean fitness over generations</h2><canvas id="chartMeanFitness"></canvas></div>
  </div>
  <script>
    async function load() {
      const r = await fetch('/api/data');
      const d = await r.json();
      const history = d.history || [];
      const bestAcc = d.best_accuracy;
      const bestFitness = d.best_fitness;
      const bestFitTime = d.best_fit_time;
      const bestComplexity = d.best_complexity;

      const metricsEl = document.getElementById('metrics');
      metricsEl.innerHTML = [
        ['Best accuracy', bestAcc != null ? (bestAcc * 100).toFixed(2) + '%' : '—'],
        ['Best fitness', bestFitness != null ? bestFitness.toFixed(4) : '—'],
        ['Best fit time (s)', bestFitTime != null ? bestFitTime.toFixed(3) : '—'],
        ['Complexity', bestComplexity != null ? bestComplexity : '—']
      ].map(([label, value]) => '<div class="metric"><span>' + label + '</span><strong>' + value + '</strong></div>').join('');

      const gens = history.map(h => h.generation);
      const bestF = history.map(h => h.best_fitness);
      const bestA = history.map(h => h.best_accuracy);
      const meanF = history.map(h => h.mean_fitness);

      new Chart(document.getElementById('chartEvolution'), {
        type: 'line',
        data: {
          labels: gens,
          datasets: [
            { label: 'Best fitness', data: bestF, borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)', fill: true },
            { label: 'Best accuracy', data: bestA.map(a => a * 100), borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true }
          ]
        },
        options: { responsive: true, maintainAspectRatio: true, scales: { x: { title: { display: true, text: 'Generation' } } } }
      });

      new Chart(document.getElementById('chartAccuracyTime'), {
        type: 'scatter',
        data: {
          datasets: [{ label: 'Generations', data: history.map(h => ({ x: h.generation, y: h.best_accuracy * 100 })), backgroundColor: '#3b82f6' }]
        },
        options: { responsive: true, scales: { x: { title: { display: true, text: 'Generation' } }, y: { title: { display: true, text: 'Accuracy %' } } } }
      });

      new Chart(document.getElementById('chartMeanFitness'), {
        type: 'line',
        data: { labels: gens, datasets: [{ label: 'Mean fitness', data: meanF, borderColor: '#a855f7', fill: true, backgroundColor: 'rgba(168,85,247,0.1)' }] },
        options: { responsive: true, scales: { x: { title: { display: true, text: 'Generation' } } } }
      });
    }
    load();
  </script>
</body>
</html>
"""
