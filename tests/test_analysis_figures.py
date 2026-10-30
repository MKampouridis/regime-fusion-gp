import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from regime_fusion.analysis import cli, figures, pilot  # noqa: E402
from regime_fusion.data.store import DATA_DIR_ENV  # noqa: E402


def candidates():
    """Histories and validation checkpoints of four GP-parameter candidates, two stocks and two runs each."""
    rows, marks = [], []
    for population, tournament, converge_at in ((250, 2, 200), (500, 4, 100), (1000, 2, 40), (2000, 4, 20)):
        for stock in ("S1", "S2"):
            for run in range(2):
                for g in range(100_000 // population + 1):
                    rows.append({"rung": f"A1-p{population}-t{tournament}", "stock": stock, "test_year": 2019,
                                 "stage": 1, "run": run, "generation": g,
                                 "best_fitness": min(g / converge_at, 1.0) + 0.1 * run})
                    if g % 10 == 0:
                        marks.append({**rows[-1], "sharpe": 0.5 - 0.001 * g})
    return pd.DataFrame(rows), pd.DataFrame(marks)


def test_a_figure_is_a_small_pdf_with_truetype_fonts_that_drawing_again_replaces(tmp_path):
    history, marks = candidates()
    table, choice = pilot.gp_parameters(history, marks)
    figure = figures.gp_parameters(history, marks, table, choice)
    folder = tmp_path / "figures"
    assert figures.save(figure, folder, "gp_parameters", png=True) == [folder / "gp_parameters.pdf",
                                                                        folder / "gp_parameters.png"]
    pdf = (folder / "gp_parameters.pdf").read_bytes()
    assert len(pdf) < 100_000 and b"/Type3" not in pdf  # journals refuse Type 3 fonts
    assert figures.save(figure, folder, "gp_parameters") == [folder / "gp_parameters.pdf"]
    assert sorted(p.name for p in folder.iterdir()) == ["gp_parameters.pdf"]  # no PNG left out of date
    figures.gp_parameters(history, marks.iloc[:0], *pilot.gp_parameters(history, marks.iloc[:0]))  # no checkpoints
    counts = pd.DataFrame({"cells": 20, "sd": [0.06, 0.04, 0.03, 0.02], "p95_distance": [0.11, 0.07, 0.048, 0.03]},
                          index=pd.Index([10, 20, 30, 50], name="n"))
    assert figures.save(figures.run_count(counts), folder, "run_count")[0].stat().st_size < 100_000


def test_the_command_draws_what_the_campaign_has(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    with pytest.raises(SystemExit, match="nothing to draw"):
        cli.main(["figures", "--campaign", "gp-pilot"])
    history, marks = candidates()
    for (rung, stock, run), part in history.groupby(["rung", "stock", "run"]):
        path = tmp_path / "results" / "gp-pilot" / rung / stock / f"2019.stage1.run{run:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoints = marks[(marks["rung"] == rung) & (marks["stock"] == stock) & (marks["run"] == run)]
        path.write_text(json.dumps({"rung": rung, "stock": stock, "test_year": 2019, "stage": 1, "run": run,
                                    "history": part[["generation", "best_fitness"]].to_dict(orient="records"),
                                    "checkpoints": checkpoints[["generation", "best_fitness", "sharpe"]]
                                    .to_dict(orient="records")}))
    pd.DataFrame({"rung": "A0", "stock": "S1", "test_year": 2019, "stage": 1, "run": range(60),
                  "sharpe_10bps": np.random.default_rng(0).normal(size=60)}).to_parquet(
        tmp_path / "results" / "gp-pilot" / "runs.parquet")
    assert cli.main(["figures", "--campaign", "gp-pilot", "--png"]) == 0
    folder = tmp_path / "reports" / "figures" / "gp-pilot"
    assert sorted(p.name for p in folder.iterdir()) == ["gp_parameters.pdf", "gp_parameters.png",
                                                       "run_count.pdf", "run_count.png"]
