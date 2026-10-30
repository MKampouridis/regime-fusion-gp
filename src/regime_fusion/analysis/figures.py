"""Figures of a campaign, drawn from the tables the analysis computes (``python -m regime_fusion.analysis figures``).

Each figure sums up a whole campaign, so a campaign has a handful of figures,
not one per run or stock. Each is a small vector PDF (with a PNG beside it on
request) under ``reports/figures/<campaign>/`` in the data folder, with a fixed
name, so drawing it again replaces it. Figures redraw in seconds from the saved
results and are never committed (decided on 30 October 2026).

* ``gp_parameters``: the GP-parameter pilot (docs/PROTOCOL.md, section 9.1). For
  each population (shades of one blue, darker for larger) and tournament size
  (solid for the smaller, dashed for the larger): the median best training
  fitness and the median validation Sharpe ratio against evaluations, a dot at
  the candidate's number of generations and a ring round the choice.
* ``run_count``: the 100-run check (section 9): the 95th-percentile distance of
  an n-run median from the median of all runs, against n, with the protocol's
  limit.

Fonts are embedded as TrueType, since journals refuse Type 3 fonts. matplotlib
is an optional dependency (``pip install -e ".[figures]"``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from regime_fusion.analysis.pilot import CANDIDATE

SHADES = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # one hue, light to dark, for an ordered quantity
LINES = ("-", "--", ":", "-.")
INK, SECONDARY = "#0b0b0b", "#52514e"
STYLE = {
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titlelocation": "left",
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 200,
}


def save(figure: Figure, folder: Path, name: str, png: bool = False) -> list[Path]:
    """Writes ``name``.pdf (and .png) to ``folder``, replacing an earlier drawing, and returns the paths.
    Without ``png``, the PNG of an earlier drawing is removed, so no file is out of date."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{name}.{kind}" for kind in (("pdf", "png") if png else ("pdf",))]
    if not png:
        (folder / f"{name}.png").unlink(missing_ok=True)
    with mpl.rc_context(STYLE):
        for path in paths:
            figure.savefig(path)
    return paths


def gp_parameters(history: pd.DataFrame, checkpoints: pd.DataFrame, table: pd.DataFrame, choice: dict) -> Figure:
    """The GP-parameter pilot: training fitness and validation Sharpe ratio against evaluations
    (``table`` and ``choice`` from ``pilot.gp_parameters``)."""
    populations = sorted(table.index.get_level_values("population").unique())
    tournaments = sorted(table.index.get_level_values("tournament").unique())
    if len(populations) > len(SHADES) or len(tournaments) > len(LINES):
        raise ValueError(f"at most {len(SHADES)} populations and {len(LINES)} tournament sizes can be told apart")
    colour = dict(zip(populations, SHADES[len(SHADES) - len(populations):]))
    line = dict(zip(tournaments, LINES))
    keys = ["population", "tournament", "generation"]
    fitness = _candidates(history).groupby(keys)["best_fitness"].median()
    sharpe = _candidates(checkpoints).groupby(keys)["sharpe"].median() if len(checkpoints) else None
    with mpl.rc_context(STYLE):
        figure = Figure(figsize=(7.0, 2.8), layout="constrained")
        fitness_axes, sharpe_axes = figure.subplots(1, 2, sharex=True)
        for (population, tournament), row in table.iterrows():
            style = {"color": colour[population], "linestyle": line[tournament]}
            at, curve = row["generations"], fitness.loc[(population, tournament)]
            fitness_axes.plot(curve.index * population / 1000, curve.to_numpy(), **style)
            fitness_axes.plot(at * population / 1000, curve.loc[at], "o", color=style["color"], markersize=3.5)
            if sharpe is not None and (population, tournament) in sharpe.index.droplevel("generation"):
                marks = sharpe.loc[(population, tournament)]
                sharpe_axes.plot(marks.index * population / 1000, marks.to_numpy(), **style)
                sharpe_axes.plot(at * population / 1000, row["validation_sharpe"], "o", color=style["color"],
                                 markersize=3.5)
        chosen = choice["evaluations"] / 1000
        rings = [(fitness_axes, fitness.loc[(choice["population"], choice["tournament"], choice["generations"])])]
        if sharpe is not None:
            rings.append((sharpe_axes, choice["validation_sharpe"]))
        else:
            sharpe_axes.text(0.5, 0.5, "no validation checkpoints", transform=sharpe_axes.transAxes,
                             ha="center", color=SECONDARY)
        for axes, value in rings:
            axes.plot(chosen, value, "o", markersize=8, markerfacecolor="none", markeredgecolor=INK,
                      markeredgewidth=0.8)
            axes.annotate("chosen", (chosen, value), xytext=(7, -9), textcoords="offset points", fontsize=7,
                          bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none"})
        fitness_axes.set(title="(a) Best training fitness", xlabel="Evaluations (thousands)",
                         ylabel="Median over runs")
        sharpe_axes.set(title="(b) Sharpe ratio on the validation year", xlabel="Evaluations (thousands)",
                        ylabel="Median over runs")
        handles = [Line2D([], [], color=colour[p], label=f"population {p:,}") for p in populations]
        handles += [Line2D([], [], color=SECONDARY, linestyle=line[t], label=f"tournament {t}") for t in tournaments]
        figure.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    return figure


def run_count(counts: pd.DataFrame, limit: float = 0.05, runs: int = 30) -> Figure:
    """The 100-run check (``counts`` from ``pilot.run_count_check``), with the limit for keeping ``runs``."""
    with mpl.rc_context(STYLE):
        figure = Figure(figsize=(3.4, 2.4), layout="constrained")
        axes = figure.subplots()
        axes.plot(counts.index, counts["p95_distance"], "o-", color=SHADES[2], markersize=4)
        axes.axhline(limit, color=SECONDARY, linewidth=0.8, linestyle="--")
        axes.annotate(f"limit {limit:g}", (counts.index.max(), limit), xytext=(0, 3), textcoords="offset points",
                      ha="right", va="bottom", fontsize=7, color=SECONDARY)
        if runs in counts.index:
            value = counts.loc[runs, "p95_distance"]
            axes.annotate(f"{value:.3f}", (runs, value), xytext=(5, 4), textcoords="offset points", fontsize=7)
        axes.set(title="How far an n-run median strays", xlabel="Runs (n)",
                 ylabel="95th-percentile distance (Sharpe)", xticks=list(counts.index))
        axes.set_ylim(bottom=0)
    return figure


def _candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with each candidate's population and tournament size; rows of other rungs left out."""
    parts = frame["rung"].str.extract(CANDIDATE)
    keep = parts["base"].notna()
    return frame[keep].assign(population=parts.loc[keep, "population"].astype(int),
                              tournament=parts.loc[keep, "tournament"].astype(int))
