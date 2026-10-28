"""Command line for the data pipeline: ``python -m regime_fusion.data <command>``.

Run ``python -m regime_fusion.data --help`` for the list of commands. Downloads
are cached: a file already in the data folder is not fetched again unless
``--refresh`` is given.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from regime_fusion import __version__
from regime_fusion.data import constituents, store
from regime_fusion.data.settings import load_settings

CONSTITUENTS_FILE = Path(__file__).resolve().parents[3] / "configs" / "constituents_2015-01.csv"

log = logging.getLogger("regime_fusion.data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m regime_fusion.data", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser(
        "constituents",
        help="rebuild configs/constituents_2015-01.csv from the pinned Wikipedia revisions",
    )
    sub.set_defaults(run=run_constituents)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.run(args) or 0


def user_agent() -> str:
    """Identifies our requests to services that ask for a contact."""
    return f"regime-fusion-gp/{__version__} ({load_settings().sec_user_agent})"


def run_constituents(args: argparse.Namespace) -> int:
    texts = {}
    for key, revision in constituents.SNAPSHOT.items():
        payload = constituents.fetch(revision, user_agent())
        store.write_bytes(
            store.layer("raw", "wikipedia", f"{key}_{revision.revid}.json"), payload,
            source="Wikipedia API", page=revision.page, revid=revision.revid,
        )
        texts[key] = constituents.wikitext_of(payload, revision)
    table = constituents.combine(
        constituents.parse_sp100(texts["sp100"]), constituents.parse_sp500(texts["sp500"])
    )
    table.to_csv(CONSTITUENTS_FILE, index=False)
    log.info("wrote %d S&P 500 members (%d in the S&P 100) to %s",
             len(table), int(table["sp100"].sum()), CONSTITUENTS_FILE)
    return 0
