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
- Push straight to `main` (Michael's preference) and summarise each change for him in plain English.
- The plan is in `docs/RESEARCH_PLAN.md`; decisions in `docs/DECISIONS.md` take precedence over it.

## Which model does what (10 November 2026)

Michael asked for the most capable model to do the complex thinking and cheaper models the simpler work, with Claude deciding case by case. He sets the main session's model and effort (`/model`, `/effort`); Claude cannot change them, so the saving comes from handing work to helpers on cheaper models (`.claude/agents/`).

- **The main session keeps:** research design; statistics and what they mean; anything about look-ahead, test data, the protocol or a recorded decision; what is put to Michael; the brief for every helper; and the review of whatever a helper returns. It also does any task of a few tool calls itself, since starting a helper costs more than that.
- **`builder` (Sonnet)** implements a written specification: code and tests for a settled design, or a document from an outline. Brief it with the files to read, the exact deliverables, the files it may touch and a check it must pass.
- **`runner` (Haiku)** runs commands and reports what happened: the test suite, scripts, listings, searches for where something is defined or used, the queue and logs on CERES, fetching results. It changes no files. Prefer it to the built-in Explore agent for plain look-ups, since Explore runs on a dearer model.
- **Where the helpers are not loaded** (a session that began before `.claude/agents/` existed), use the general-purpose agent with the model set to `sonnet` or `haiku` and the same kind of brief.
- **A helper's report is a claim to check.** Read the diff, rerun the deciding command and compare any number with its source before relying on it or telling Michael. Helpers never commit, push, submit jobs, download data or touch files outside their brief; the main session does those after its review.
- **Move a task up** to the main session when a helper fails twice or its result does not survive review. Do not start several helpers for second opinions on research conclusions unless Michael asks: agents in parallel spend credits quickly.

## Environment

- The cloud workspace has 2 cores and cannot reach SEC, Yahoo, GDELT, BigQuery or Hugging Face, so download scripts run on Michael's Mac or on CERES.
- The main experiments run on CERES (University of Essex HPC); Michael submits the jobs. Ask him for scheduler and Python details, with the terminal commands to find them, when the time comes.
