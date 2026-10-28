# Notes for Claude

Research code for Michael Kampouridis's paper on regime-conditioned strongly-typed GP that fuses fundamental, sentiment and technical (FA/SA/TA) signals. Michael is the supervisor; Claude works as his research associate. Keep him informed, explain in plain English, and ask before changing a recorded design decision.

## Ground rules

- Python only. Package in `src/regime_fusion`, tests in `tests`; run `pytest` before every commit.
- No look-ahead: anything used on day t must be available by the close of t, and trades execute on t+1. Every feature module needs a test with `regime_fusion.testing.check_no_lookahead`.
- Design the recipe on data up to `period.development_end` (2018) only. Never tune on 2019–2025 or on the hold-out stocks.
- Randomness only through `regime_fusion.seeds`. Seeds depend on stock, test year and run number, never on the configuration.
- Every experiment run writes a provenance record (`regime_fusion.provenance`).
- Never commit data or results. Data live under `$RFGP_DATA_DIR`.
- Log decisions in `docs/DECISIONS.md` with the date and who decided.
- British English in documentation and comments.
- Work on a branch and open a pull request for Michael to review, unless he says otherwise.

## Environment

- The cloud workspace has 2 cores and cannot reach SEC, Yahoo, GDELT, BigQuery or Hugging Face, so download scripts run on Michael's Mac or on CERES.
- The main experiments run on CERES (University of Essex HPC); Michael submits the jobs. Ask him for scheduler and Python details, with the terminal commands to find them, when the time comes.
