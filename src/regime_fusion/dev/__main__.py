"""``python -m regime_fusion.dev e23 [--workers N] [--limit K]`` runs the E2 x E3 grid (finished runs
are skipped); ``python -m regime_fusion.dev e23-report`` summarises it."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from regime_fusion.dev import e8, e23, longshort, replicate, report

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
    sub.add_argument("--inputs", nargs="+", default=["level"], choices=e23.INPUTS)
    sub.add_argument("--fitness", nargs="+", default=["sharpe-sum"], choices=e23.FITNESSES)
    commands.add_parser("e23-report", help="summarise the E2 x E3 grid")
    sub = commands.add_parser("e6", help="run the long/short experiment")
    sub.add_argument("--workers", type=int, default=4)
    commands.add_parser("e6-report", help="summarise the long/short experiment")
    commands.add_parser("e7-simple", help="the best median-split rule on training, scored on validation")
    sub = commands.add_parser("e8", help="the long/short GP on the broad universe, six folds")
    sub.add_argument("--workers", type=int, default=4)
    commands.add_parser("e8-simple", help="E8's best median-split rule per fold")
    commands.add_parser("e8-report", help="summarise E8")
    sub = commands.add_parser("acttrade", help="replicate ActTrade's protocol on our data (tuning, then final runs)")
    sub.add_argument("--workers", type=int, default=4)
    commands.add_parser("acttrade-report", help="summarise the ActTrade replication")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "acttrade-report":
        replicate.report()
        return 0
    if args.command == "acttrade":
        for phase in ("tuning", "final"):
            todo = replicate.tuning_jobs() if phase == "tuning" else replicate.final_jobs()
            log.info("%s: %d runs with %d workers", phase, len(todo), args.workers)
            with ProcessPoolExecutor(args.workers) as pool:
                for i, future in enumerate(as_completed([pool.submit(replicate.run_job, j) for j in todo]), 1):
                    message = future.result()
                    if i % 100 == 0 or i == len(todo):
                        log.info("%s %d/%d %s", phase, i, len(todo), message)
            if phase == "tuning":
                log.info("chosen exits: %s", replicate.chosen_exits())
        replicate.report()
        return 0
    if args.command == "e8-simple":
        e8.simple_baseline()
        return 0
    if args.command == "e8-report":
        e8.report()
        return 0
    if args.command == "e8":
        todo = e8.jobs()
        log.info("%d E8 runs to do with %d workers", len(todo), args.workers)
        with ProcessPoolExecutor(args.workers) as pool:
            for i, future in enumerate(as_completed([pool.submit(e8.run_job, j) for j in todo]), 1):
                log.info("%d/%d %s", i, len(todo), future.result())
        return 0
    if args.command == "e7-simple":
        longshort.simple_baseline()
        return 0
    if args.command == "e6-report":
        longshort.report()
        return 0
    if args.command == "e6":
        todo = longshort.jobs()
        log.info("%d long/short runs to do with %d workers", len(todo), args.workers)
        with ProcessPoolExecutor(args.workers) as pool:
            for i, future in enumerate(as_completed([pool.submit(longshort.run_job, j) for j in todo]), 1):
                log.info("%d/%d %s", i, len(todo), future.result())
        return 0
    if args.command == "e23-report":
        report.e23_report()
        return 0
    todo = [j for j in e23.jobs(args.modes, args.designs, args.folds, args.runs, args.inputs, args.fitness) if not j.path().exists()]
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
