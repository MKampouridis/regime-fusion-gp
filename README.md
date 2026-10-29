# regime-fusion-gp

Research code for *Learning When to Listen: regime-conditioned genetic programming for fusing fundamental, sentiment and technical signals* (in preparation).

The project evolves strongly-typed genetic programming (GP) trading strategies with separate fundamental (FA), sentiment (SA) and technical (TA) branches. Unlike ActTrade (Christodoulaki and Kampouridis, *Applied Soft Computing*, 2026), the way the branches are combined is itself evolved and conditioned on the market regime, which is detected with directional changes (DC).

## Status

Set-up and the directional-change module are done. The data pipeline (Phase 2) is under way. Built and run: the trading calendar and point-in-time rules, the universe (ActTrade's US stocks plus a sector-balanced draw, and 50 hold-out stocks), prices with spin-off corrections, market series, SEC fundamentals and ActTrade's twelve fundamental indicators. Running: the GDELT news extraction (about three days) and the FNSPID headline filter. Next: screening GDELT's company matches and building the daily news features. The plan is in [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) and decisions to date in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/regime_fusion/` | Python package |
| `src/regime_fusion/dc/` | Directional-change events and causal DC indicators |
| `src/regime_fusion/data/` | Data pipeline: downloads, trading calendar, point-in-time alignment, checks |
| `src/regime_fusion/engine/` | Typed GP (ActTrade and the ablation ladder), backtester, metrics, runs; see `docs/ENGINE.md` |
| `src/regime_fusion/regimes/` | Market regimes: Gaussian HMMs with filtered probabilities, per walk-forward window |
| `configs/study.yaml` | Settled study design: periods, walk-forward windows, runs, costs, base seed |
| `configs/data_sources.yaml` | Settings for the data pipeline (download range, rate limits, contacts) |
| `configs/rungs.yaml` | GP parameters, the ablation ladder and the exit-setting grid |
| `docs/` | Research plan, decision log, data pipeline (`DATA.md`), and the analysis protocol once it is frozen |
| `tests/` | Automated tests, including look-ahead checks and a cross-check against the group's earlier Java DC code |

## Set-up

```bash
python3 -m venv ~/.venvs/rfgp
source ~/.venvs/rfgp/bin/activate
pip install -e ".[dev,download]"   # "download" is needed only where data are downloaded
pytest
```

Keep the environment and the data folder outside cloud-synchronised folders such as Box, which would otherwise upload thousands of files.

## Data

No data are stored in this repository. Raw and processed data live outside it, in the folder named by the environment variable `RFGP_DATA_DIR` (default `./data`, which git ignores); [`docs/DATA.md`](docs/DATA.md) describes the layout and the point-in-time rules. Licences differ by source: FNSPID is CC BY-NC 4.0 (research use, no redistribution), Yahoo Finance data are for personal research use, and SEC EDGAR and GDELT are public.

## Rules the code follows

- **No look-ahead.** A value used on day *t* must be available by the close of day *t*; trades execute on day *t*+1. Every feature module is tested by recomputing it on truncated data (`regime_fusion.testing.check_no_lookahead`).
- **Development data only for design.** The recipe (algorithm settings, features, regime model) is designed on data up to the end of 2018. Test years 2019–2025 and the hold-out stocks are run once, after the recipe is frozen.
- **Reproducibility.** Randomness comes only from `regime_fusion.seeds`, and every run records its seed, configuration hash, code version and package versions (`regime_fusion.provenance`).
