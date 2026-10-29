"""Configurations of the ablation ladder (``configs/rungs.yaml``).

A rung may override the common GP settings (``gp``) and scaling. The
GP-parameter pilot's candidates (``gp_pilot``) are generated as rungs named
``<rung>-p<population>-t<tournament>``, each with the same budget of
evaluations (population x generations).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from regime_fusion.engine.fitness import Settings
from regime_fusion.engine.gp import GPConfig

DEFAULT_RUNGS = Path(__file__).resolve().parents[3] / "configs" / "rungs.yaml"


@dataclass(frozen=True)
class Rung:
    name: str
    sources: tuple[str, ...]
    fusion: tuple[str, ...]
    regimes: str | None
    gp: GPConfig
    fitness: Settings
    max_branch_depth: int
    scaling: str
    spec: dict[str, Any]   # the resolved settings, for provenance
    finbert: bool = False  # robustness check: FinBERT headline sentiment joins the sentiment branch

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()[:12]


def load_rungs(path: str | Path = DEFAULT_RUNGS) -> dict[str, Rung]:
    raw = yaml.safe_load(Path(path).read_text())
    raw["rungs"] = {**raw["rungs"], **pilot_rungs(raw.get("gp_pilot"))}
    return {name: _rung(name, raw) for name in raw["rungs"]}


def pilot_rungs(grid: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The GP-parameter pilot's candidates as rung specifications."""
    if not grid:
        return {}
    budget = int(grid["evaluations"])
    return {f"{base}-p{population}-t{tournament}": {
                "extends": base,
                "gp": {"population": int(population), "generations": budget // int(population),
                       "tournament": int(tournament)}}
            for base in grid["rungs"] for population in grid["populations"] for tournament in grid["tournaments"]}


def exit_grid(path: str | Path = DEFAULT_RUNGS) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())["exit_grid"]


def finbert_source(test_year: int, path: str | Path = DEFAULT_RUNGS) -> str:
    """The headline source of the FinBERT check for a test year's window (``finbert_sources``)."""
    for source, years in yaml.safe_load(Path(path).read_text())["finbert_sources"].items():
        if test_year in years:
            return source
    raise ValueError(f"{path}: no FinBERT headline source for test year {test_year}")


def _resolved(name: str, rungs: dict[str, dict], seen=()) -> dict[str, Any]:
    if name in seen:
        raise ValueError(f"rung {name} extends itself")
    spec = dict(rungs[name])
    parent = spec.pop("extends", None)
    return {**(_resolved(parent, rungs, (*seen, name)) if parent else {}), **spec}


def _rung(name: str, raw: dict[str, Any]) -> Rung:
    spec = _resolved(name, raw["rungs"])
    gp_settings = {**raw["gp"], **spec.get("gp", {})}  # a rung may override the common settings
    max_depth = gp_settings.pop("max_branch_depth")
    gp_settings["first_parent"] = tuple(gp_settings["first_parent"])
    gp = GPConfig(**gp_settings, active_trading=bool(spec["active_trading"]), selection=spec["selection"])
    fitness = dict(raw["fitness"])
    fitness["weights"] = spec.get("weights", fitness["weights"])
    scaling = spec.get("scaling", raw["scaling"])  # a rung may override the common scaling
    resolved = {"rung": name, **spec, "gp": {**raw["gp"], **spec.get("gp", {})}, "fitness": fitness,
                "scaling": scaling}
    return Rung(name, tuple(spec["sources"]), tuple(spec["fusion"]), spec.get("regimes"), gp,
                Settings(**fitness), max_depth, scaling, resolved, bool(spec.get("finbert", False)))
