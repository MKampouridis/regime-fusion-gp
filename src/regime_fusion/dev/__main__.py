"""``python -m regime_fusion.dev e23 [--workers N] [--limit K]`` runs the E2 x E3 grid (finished runs
are skipped); ``python -m regime_fusion.dev e23-report`` summarises it."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from regime_fusion.dev import e23, report

log = logging.getLogger("regime_fusion.dev")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m regime_fusion.dev")
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("e23", help="run the E2 x E3 grid")
    sub.add_argument("--workers", type=int, default=4)
    sub.add_argument("--limit", type=int, help="run at most this many unfinished jobs (a timing test)")
    sub.add_argument("--modes", nargs="+", default=list(e23.MODES))
    sub.add_argument("--designs", nargs="+", default=list(e23.DESIGNS))
    sub.add_argument("--folds", nargs="+", default=list(e23.FOLDS))
    sub.add_argument("--runs", type=int, default=e23.RUNS)
    commands.add_parser("e23-report", help="summarise the E2 x E3 grid")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "e23-report":
        report.e23_report()
        return 0
    todo = [j for j in e23.jobs(args.modes, args.designs, args.folds, args.runs) if not j.path().exists()]
    if args.limit:
        todo = todo[:args.limit]
    log.info("%d runs to do with %d workers", len(todo), args.workers)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(e23.run_job, j) for j in todo]
        for i, future in enumerate(as_completed(futures), 1):
            log.info("%d/%d %s", i, len(todo), future.result())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
