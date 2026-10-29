"""One view of the data checks, for the Phase 2 sign-off ("data validated").

Every step of the pipeline writes its own checks to ``reports/``. This module
reads them and lists, per stock, what needs a look:

* prices: sessions missing from the calendar, bad rows, other flags, and daily
  moves of more than 25% (possible bad data or corporate events);
* SEC filings: the flags of the sec step (such as years with fewer than four
  10-K and 10-Q filings);
* fundamental and technical indicators: indicators with values on too few days;
* sentiment: the share of sessions with news and the indicators' coverage;
* regimes: probability files missing for an input set, test year or stage.

Missing reports are listed rather than treated as failures, so the check can
run while a step (the GDELT extraction, say) is still going.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from regime_fusion.data import store

REPORTS = {"prices": "price_checks.json", "sec": "sec_checks.json", "fa": "fa_coverage.json",
           "ta": "ta_coverage.json", "sa": "sa_coverage.json"}
BIG_MOVE = 0.25
THIN_NEWS = 0.5  # share of sessions with at least one article


def load_reports(root: Path | None = None) -> dict[str, dict[str, Any]]:
    out = {}
    for name, filename in REPORTS.items():
        path = store.layer("reports", filename, root=root)
        out[name] = json.loads(path.read_text()) if path.exists() else {}
    return out


def stock_issues(stock: str, reports: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Issues of one stock, by data source; a source without a report entry says so."""
    out: dict[str, list[str]] = {}
    prices = reports["prices"].get(stock)
    if prices is None:
        out["prices"] = ["no price checks"]
    else:
        issues = list(prices.get("flags", []))
        if prices.get("missing_sessions"):
            issues.append(f"{prices['missing_sessions']} sessions without a bar")
        big = [f"{day} {move:+.0%}" for day, move in prices.get("big_moves", []) if abs(move) > BIG_MOVE]
        if big:
            issues.append("moves over 25%: " + ", ".join(big))
        out["prices"] = issues
    sec = reports["sec"].get(stock)
    out["sec"] = ["no SEC checks"] if sec is None else list(sec.get("flags", []))
    for source in ("fa", "ta"):
        entry = reports[source].get(stock)
        out[source] = (["no coverage report"] if entry is None
                       else [f"thin: {', '.join(entry['missing'])}"] if entry.get("missing") else [])
    news = reports["sa"].get(stock)
    if news is None:
        out["sa"] = ["no sentiment report"]
    else:
        share = news.get("sessions_with_news")
        if share is None or share != share:  # NaN: no session has GDELT coverage yet
            out["sa"] = ["no session with news coverage yet"]
        else:
            out["sa"] = [f"news on {share:.0%} of sessions"] if share < THIN_NEWS else []
    return out


def regime_gaps(settings: dict[str, Any], test_years: list[int], root: Path | None = None) -> list[str]:
    """Market-regime probability files that should exist and do not (per-stock regimes not included)."""
    missing = []
    for name, spec in settings["inputs"].items():
        if spec.get("attention", {}).get("scope") == "stock":
            continue
        for year in test_years:
            for stage in (1, 2):
                if not store.layer("processed", "regimes", name, f"{year}.stage{stage}.parquet", root=root).exists():
                    missing.append(f"{name} {year} stage {stage}")
    return missing


def markdown(rows: list[dict[str, Any]], regimes: list[str], pending: dict[str, Any]) -> str:
    """A table for reading: one row per stock with issues, then the regime files and news progress."""
    lines = ["# Data check", ""]
    if pending:
        lines += [f"GDELT news: {pending.get('gdelt_days_used', 0)} days extracted, "
                  f"{pending.get('gdelt_days_pending', '?')} still to come; "
                  f"{pending.get('sessions_covered', 0):.1%} of sessions covered.", ""]
    flagged = [r for r in rows if any(r["issues"].values())]
    lines += [f"{len(rows)} stocks checked; {len(flagged)} with something to look at.", ""]
    if flagged:
        lines += ["| Stock | Role | Issues |", "| --- | --- | --- |"]
        for row in flagged:
            text = "; ".join(f"{source}: {', '.join(items)}" for source, items in row["issues"].items() if items)
            lines.append(f"| {row['stock']} | {row['role']} | {text} |")
        lines.append("")
    lines += ["Missing regime files: " + (", ".join(regimes) if regimes else "none") + "."]
    return "\n".join(lines) + "\n"
