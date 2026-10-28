# regime-fusion-gp

Research code for *Learning When to Listen: regime-conditioned genetic programming for fusing fundamental, sentiment and technical signals* (in preparation).

The project evolves strongly-typed genetic programming (GP) trading strategies with separate fundamental (FA), sentiment (SA) and technical (TA) branches. Unlike ActTrade (Christodoulaki and Kampouridis, *Applied Soft Computing*, 2026), the way the branches are combined is itself evolved and conditioned on the market regime, which is detected with directional changes (DC).

## Status

Phase 1: set-up. Decisions to date are in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/regime_fusion/` | Python package |
| `src/regime_fusion/dc/` | Directional-change events and causal DC indicators |
| `configs/study.yaml` | Settled study design: periods, walk-forward windows, runs, costs, base seed |
| `docs/` | Decision log; the analysis protocol once it is frozen |
| `tests/` | Automated tests, including look-ahead checks and a cross-check against the group's earlier Java DC code |

## Set-up

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Data

No data are stored in this repository. Raw and processed data live outside it, in the folder named by the environment variable `RFGP_DATA_DIR` (default `./data`, which git ignores). Licences differ by source: FNSPID is CC BY-NC 4.0 (research use, no redistribution), Yahoo Finance data are for personal research use, and SEC EDGAR and GDELT are public.

## Rules the code follows

- **No look-ahead.** A value used on day *t* must be available by the close of day *t*; trades execute on day *t*+1. Every feature module is tested by recomputing it on truncated data (`regime_fusion.testing.check_no_lookahead`).
- **Development data only for design.** The recipe (algorithm settings, features, regime model) is designed on data up to the end of 2018. Test years 2019–2025 and the hold-out stocks are run once, after the recipe is frozen.
- **Reproducibility.** Randomness comes only from `regime_fusion.seeds`, and every run records its seed, configuration hash, code version and package versions (`regime_fusion.provenance`).
