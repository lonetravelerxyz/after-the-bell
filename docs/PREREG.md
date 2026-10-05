# Pre-registration — test window 2026-08-11 → 2026-09-22 (project B)

Status: **FROZEN** 2026-09-23 ~14:45 UTC (this commit). The test window is run only after this commit. Nothing in this file may change after the test runs. Any
later addition goes under "Deviations", with a date.

## What is fixed

| Item | Value |
|---|---|
| Code | the commit that freezes this file (risk v1, prompt v1 `prompts/decide_v1.md`, `sentiment/config.py` as committed). Prompt v1 failed its dev explainability criteria (trials.log 14:40) and is frozen anyway, not tuned further |
| Model | `qwen3.8-max` via the hackathon gateway, temperature 0, `enable_thinking: false` |
| Data | `sentiment/snapshot/` as hashed in `MANIFEST.sha256` at that commit |
| Window | 2026-08-11 00:00 → 2026-09-22 24:00 UTC, decisions at 00/04/08/12/16/20 UTC |
| Universe | NVDA AAPL GOOGL META AMZN TSLA MSTR COIN HOOD CRCL (stock perps) |
| Costs | taker 6 bp, quoted half-spread per name (`static/quoted_spread.csv`), realised funding |

## Runs (each run once, in this order)

1. `baseline` v1: the rule policy, with no LLM decisions.
2. `llm` v1: **headline**.
3. `nonews` v1: the LLM with no evidence cards.
4. `shuffled` v1: the LLM with every card shown 7–21 days late.
5. `blinded` v1: only if Qwen credit remains after 1–4.

Any run that fails partway is resumed from the LLM cache; it is not re-designed.

## Headline and readings (decided now)

- **Headline**: the LLM v1 test-window net return, Sharpe, Sortino, max DD, win rate, turnover and
  cost ladder, labelled *estimated (replay)*. It is reported **whatever its sign**.
- **LLM increment**: Sharpe(llm) − Sharpe(baseline), block bootstrap (5-day blocks, 2000 paths,
  seed 20260923). "The LLM adds value" is claimed only if the 95% CI excludes 0.
- **Information use**:
  - "news matters" is claimed only if llm beats **both** nonews and shuffled with CIs excluding 0.
  - Otherwise the report says the agent's P&L is not attributable to the news it reads.
- **Signal tests**, one each, all reported:
  - (a) rank IC of the baseline score vs forward perp return at 4h, 24h and 72h, with Newey-West t;
  - (b) signed 24h demeaned return after rating cards, one observation per event;
  - (c) signed 24h demeaned return after `news/new` cards, one observation per ticker-day.

  With 5 tests (a counts as 3), |t| > 2.6 is needed before any one is called significant
  (Bonferroni at 5%).
- **Risk layer** (reported, not pass/fail): the count and P&L effect of each binding rule, and the
  book's realised beta to the equal-weight basket (expected within ±0.03).
- **Dev reference**: the dev results (v0 and v1) and `sentiment/DIAGNOSIS-dev.md` are shown next to
  the test results. The dev loss and the absence of signal in dev go in the report summary.

## What would change the plan

Nothing, for the test window. If a bug is found after the test runs, the fix and a rerun are added
under Deviations. Both the original and the rerun are reported, and the headline stays the
original unless the bug is shown to invalidate it.

## Deviations

- 2026-09-23 14:20 UTC, **timestamps**: the "FROZEN ~14:45 UTC" in the Status line was written ahead
  of time and is wrong. The freeze is commit `62286d9` at 13:32:47 UTC, and git is authoritative.
  The baseline test run started at 13:32:53 UTC, after the freeze.
- 2026-09-23 14:20 UTC, **code after freeze**: `sentiment/replay.py` (a `code_stamp` field) and
  `sentiment/metrics.py` (daily statistics on complete UTC days, counts of bootstrap paths) were
  edited at about 13:38 UTC, before the llm/nonews/shuffled test runs started on that tree. Neither
  edit touches decisions, risk actions or fills. Proof: every test run is rerun offline from a clean
  checkout of `62286d9`, and the log hashes are compared in `sentiment/out/freeze_rerun.json`.
- 2026-09-23 14:20 UTC, **signal test (c)**: the event direction now comes only from cards known at
  the entry bar. The previous version used every card of the ticker-day, which looks ahead. Forward
  windows must end before the window end. This was fixed before any test-window signal test was
  run; the dev before/after values are in `trials.log`.
