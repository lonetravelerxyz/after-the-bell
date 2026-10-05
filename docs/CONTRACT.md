# sentiment/ — module contract (project B, spec `docs/spec-B-market-sentiment.md`)

Every module owns its files and exposes exactly the interface below. Times are tz-aware UTC
`pandas.Timestamp`. Weights are fractions of equity (+ long, − short). Run modules as
`uv run python -m sentiment.<module>` from the project root (`b-sentiment-agent/` in the research monorepo, the
repo root of the exported public repo); every path below is relative to it (`sentiment/config.py`). B does not
import A4 (`../a4-session-factor/`): its read-only MCP client is vendored as `sentiment/mcp_client.py`, its quoted
spreads as `static/quoted_spread.csv` (byte-identical), and the shared cost constants are listed here.

## Constants — `sentiment/config.py`
```
UNIVERSE = ["NVDA","AAPL","GOOGL","META","AMZN","TSLA","MSTR","COIN","HOOD","CRCL"]   # perp symbol = f"{t}USDT"
INDEX_PERPS = ["SP500USDT","NDX100USDT"]        # context only in v0, not traded
REPLAY_START, DEV_END, TEST_START, REPLAY_END = "2026-06-22", "2026-08-10", "2026-08-11", "2026-09-22"   # UTC dates, inclusive
DECISION_HOURS = (0, 4, 8, 12, 16, 20)          # UTC decision clock
CARD_LOOKBACK_H = 72                            # evidence cards older than this are not shown
TAKER = 0.0006                                  # same as A4
HALF_SPREAD_FALLBACK = 0.0005                   # when static/quoted_spread.csv lacks the name
MAX_W_NAME, MAX_GROSS, MAX_NET = 0.15, 1.0, 0.30
STOP_LOSS_NAME = 0.05                           # v0 flat stop: flat the name, 24h cooldown
DAILY_KILL = 0.03                               # equity down 3% in 24h -> flat everything for 24h
OFF_HOURS_SCALE = 0.5                           # caps scaled outside US regular session (13:30-20:00 UTC Mon-Fri)
FUNDING_BLOCK = 0.001                           # |rate| >= 0.1%/8h blocks new exposure on the paying side
START_EQUITY = 10_000.0
RISK_VERSION = "v1"                             # "v0" | "v1"; env SENTIMENT_RISK_VERSION or --risk-version overrides
VOL_CAP_FLOOR = 0.33                            # v1 cap_i = MAX_W_NAME * clip(median vol_7d / vol_7d_i, VOL_CAP_FLOOR, 1)
STOP_K, STOP_FLOOR, STOP_CAP = 2.0, 0.03, 0.10  # v1 stop_i = clip(STOP_K * vol_7d_i at entry, STOP_FLOOR, STOP_CAP)
BETA_NEUTRAL = True                             # v1 beta step on/off
BETA_LOOKBACK_D, BETA_MIN_D = 60, 30            # beta_60d window; fewer daily returns -> beta 1.0
LLM_MODEL = "qwen3.8-max"                       # model id in LLM cache keys when BITGET_QWEN_MODEL is unset
```

## Data — `sentiment/data.py` (+ `sentiment/fetch_extra.py`, `sentiment/snapshot.py`)
Raw inputs (gitignored `data/raw/`, `config.RAW`: the project's `data/` when it exists, else the monorepo's shared
`../data/`): `news/news.parquet` (from `sentiment/fetch_news.py`),
`sentiment/ratings.parquet`, `sentiment/insider.parquet`, `sentiment/crypto_fng.parquet`,
`perp/{SYM}.parquet` (1h bars, ts ms = bar open), `funding/{SYM}.parquet` (rate, ts ms = settlement),
`spot/R{SYM}.parquet` (rToken 1h). `fetch_extra.py` fetches ratings, insider, crypto F&G and the two
index perps (1h history-candles, same format as A4's perp files).
`snapshot.py` freezes every input restricted to the universe and replay window into
`snapshot/*.parquet` + `snapshot/MANIFEST.sha256` (committed; replay reads the
snapshot first, raw otherwise). Universe perp bars start BETA_LOOKBACK_D + 3 days before REPLAY_START
(`--perp-history` adds them without touching the other frozen files), so beta_60d has a full window from
the first replay decision.

```
class Store:
    def __init__(self, snapshot: bool = True): ...
    def items(self, t, lookback_h) -> list[Item]         # news/rating/insider items with available_at in (t-lookback, t]
    def all_items(self) -> list[Item]                     # everything, for evidence extraction
    def bars(self, sym, t, n) -> pd.DataFrame             # last n CLOSED 1h bars with close time <= t (cols open high low close quote_vol, index = bar open ts)
    def next_open(self, sym, t) -> float | None           # open of the first bar starting in [t, t+1h) (fill price); None across a data gap
    def funding(self, sym, t, n) -> pd.Series             # last n settlements with ts <= t
    def settlements(self, sym, t0, t1) -> pd.Series       # settlements with t0 < ts <= t1 (for P&L)
    def fng(self, t) -> int | None                        # crypto F&G of the latest day whose value is available by t (day d available at d+1 00:00 UTC)
    def spot_premium(self, sym, t) -> float | None        # rToken close / perp close - 1 at the last closed bar <= t
```
`Item = {"id": str, "kind": "news"|"rating"|"insider", "available_at": Timestamp, "title": str, "text": str, "tickers": list[str] | None, "raw": dict}`.
Availability rules: news = `published_at` − 8h (the MCP stamps are UTC+8 wall time labelled Z; evidence in data.py), and in live mode no earlier than `first_seen`; rating = `rating_date` + 1 day 13:30 UTC (next US session);
insider = `filing_date` + 1 day 03:00 UTC (EDGAR dates Form 4s accepted until 22:00 New York time, i.e. up to
03:00 UTC the next day, with that filing date). `id` = sha1 of kind|available_at|title, stable across runs.

## Features — `sentiment/features.py`
`def market_state(store, t, version=None) -> dict[str, dict]` per ticker: `price, ret_4h, ret_24h, ret_7d,
vol_7d, funding_last, funding_7d_mean, spot_premium, us_session_open (bool)`, and under risk v1 (`version`,
default config.RISK_VERSION) `beta_60d` (trailing daily beta to the equal-weight universe basket, completed
UTC days only) and `beta_days` (daily returns it used; < BETA_MIN_D -> beta 1.0); plus key `"_market"`:
`fng, fng_regime ("extreme_fear".."extreme_greed"), sp500_ret_24h, ndx_ret_24h`. Uses only data <= t.

## LLM — `sentiment/llm.py`
```
def complete(messages: list[dict], *, purpose: str) -> str
```
OpenAI-compatible call to `BITGET_QWEN_*` (.env), temperature 0, `enable_thinking: false`. The model id in the
cache key is `BITGET_QWEN_MODEL` if set, else `config.LLM_MODEL` (pinned), so offline replay needs no `.env`.
Disk cache `cache/llm/{sha256(model+messages)}.json` storing request, response, usage,
purpose. Cache hit -> no network. `SENTIMENT_OFFLINE=1` -> cache miss raises `CacheMiss`.
Retries 5xx/timeouts 4x with backoff. `def usage_summary() -> dict` totals tokens from the cache.

## Evidence — `sentiment/evidence.py`
```
def cards_for(items: list[Item]) -> list[Card]           # news via LLM (one call per item, cached); ratings/insider deterministic
Card = {"item_id","kind","available_at","tickers": list[str],   # subset of UNIVERSE, may be empty (macro)
        "scope": "name"|"sector"|"market", "stance": int (-2..2), "strength": float (0..1),
        "horizon_h": int, "novelty": "new"|"recap"|"post_hoc", "summary": str (<=200 chars), "quote": str}
```
`post_hoc` marks items that explain a move already happened (e.g. "[Market Movement Analysis]").
Cards are persisted to `cache/cards.parquet` keyed by item_id. `cards_for` never returns partial
evidence: a failed news item raises `CacheMiss` / `LLMError` after the rest is saved (the live loop passes
`errors=[]` to collect failures instead and logs them as `evidence_errors`).

## Policies — `sentiment/agent.py`, `sentiment/baseline.py`
```
def decide(t, state: dict, cards: list[Card], book: dict[str, float], variant: str) -> Decision
Decision = {"targets": dict[str, float], "reasons": dict[str, str], "evidence": dict[str, list[str]],
            "confidence": float, "raw": str, "valid": bool}
```
`agent.decide(..., risk_ctx=None)`: without a risk context the prompt is v0 (`prompts/decide.md`, frozen
byte-identical to 94cbe51); with one (`replay.risk_context`, risk v1) it is v1 (`prompts/decide_v1.md` plus a
RISK table and a since_bp card column). The Decision carries `prompt_version`.
`agent.decide` (LLM) variants: `llm`, `nonews` (cards=[]), `shuffled` (cards passed in are already
shown late by the runner: true time + 7-21 days, never earlier), `blinded` (tickers aliased A..J, no
dates, prices as returns; summaries masked: company-unique names, dates, dollar amounts, price-target
levels, insider names/prices). An endpoint failure returns `valid: False` plus `error`.
`baseline.decide` (no LLM): per ticker score = Σ stance·strength·exp(-age_h/24) over name-scoped cards
+ 0.5·market-scoped; weight = clip(0.05·score, ±MAX_W_NAME); skips `post_hoc`.

## Risk — `sentiment/risk.py`
`def apply(t, decision, book, state, ledger_state, version=None) -> (targets: dict[str,float], actions:
list[dict])` — `version` "v0" | "v1", default config.RISK_VERSION. Enforces the limits in this order:
invalid -> hold; per-name cap; off-hours scale; funding block; stop-loss/cooldown; daily kill; [v1: beta
neutralisation]; net cap; gross cap (proportional scale-down). Each binding rule emits `{"rule", "ticker",
"before", "after"}`. The stop reference is the quantity-averaged entry (re-averaged when a position grows),
`ledger_state["entry"][tk] = {"price", "side"[, "stop"]}`. Off-hours and the funding block limit NEW
exposure: a held same-side position is kept up to the per-name cap.
- v0: per-name cap MAX_W_NAME; stop at STOP_LOSS_NAME; no beta step. Every run before 2026-09-23.
- v1: per-name cap `vol_caps` = MAX_W_NAME * clip(median vol_7d / vol_7d_i, VOL_CAP_FLOOR, 1) (flat without a
  vol); stop `stop_level` = clip(STOP_K * vol_7d_i, STOP_FLOOR, STOP_CAP) taken AT ENTRY and stored as the
  entry's `stop` (re-averaged by quantity on a top-up; an entry without one uses STOP_LOSS_NAME); rule
  `beta_neutral`: the minimal-L2 change of the weights with beta . w = 0 (beta = state beta_60d, missing ->
  1.0) inside every per-name bound the earlier rules left and with |sum w| <= MAX_NET (the net cap is solved
  inside the projection; per-name actions `beta_neutral`, then `net_cap` for the part the net cap forces).
  The net cap step then binds only when neutrality and the cap cannot both hold (fixed non-zero names).
`def effective_caps(t, state, version=None)` = per-name cap on new exposure now (what the prompt shows).
`def check(t, book, prices, ledger_state, version=None) -> (targets | None, actions)` runs the stop-loss and
daily kill on every hourly mark between decisions (v0 flat stop, v1 the entry's stop).

## Execution — `sentiment/sim.py` (replay) and `sentiment/live.py` (Bitget Demo)
```
class Broker:  # sim.SimBroker(store) and live.DemoBroker(dry_run: bool)
    def rebalance(self, t, targets: dict[str,float], equity: float, only=None) -> list[Fill]   # only: trade just these names
    def mark(self, t) -> dict   # {"equity", "positions": {tk: qty}, "cash", "fees", "spread", "funding"} as of t
Fill = {"ticker","side","qty","price","fee","half_spread_cost","ts","order_id"}
```
Sim fill = `store.next_open(sym, t)` ± half-spread (static/quoted_spread.csv, else fallback), fee
TAKER·notional; funding P&L at each settlement: position_qty·mark·rate, longs pay positive rates.
No-churn band (both brokers, `sim.below_band`): skip a same-side resize when |Δw| < max(0.25% of
equity, 25% of |target w|); opens, closes and flips always trade.
Live = `https://api.bitget.com` v2 mix orders with header `paptrading: 1`, productType USDT-FUTURES,
HMAC-SHA256 signing with `BITGET_DEMO_API_KEY/SECRET/PASSPHRASE`; `dry_run=True` (or keys absent)
logs intended orders, fills them at live-venue bid/ask and marks against live tickers (shadow ledger;
the demo book is 10-62 bp wide vs 0.3-2.5 bp live, so shadow uses the replay's cost regime). Every
order line carries the demo and the live quote. Orders are never re-sent after a timeout.

## Log — `sentiment/ledger.py`
`class Ledger(path)`: `append(record) -> record_with_hash`; record keys:
`ts, mode ("replay"|"live"), run_id, variant, event ("decision"|"risk_exit"), input_hash, evidence_ids,
llm_request_hash, decision, error, baseline_targets, risk_actions, targets, fills, skipped, state, marks,
equity, prev_hash, hash` (hash = sha256 of the canonical JSON of the record without `hash`; live records
add `wall_ts, data_warnings, evidence_errors`, and `pre_trade`). Risk v1 records add `risk_ctx,
risk_ctx_hash, risk_version, prompt_version`; v0 records have none of them (a record without
`risk_version` is v0). `def verify(path) -> bool`.

## Runner — `sentiment/replay.py`
`uv run python -m sentiment.replay --variant llm --start 2026-06-22 --end 2026-08-10 [--offline] [--risk-version v0|v1]`
-> `out/{variant}_{start}_{end}/log.jsonl`, `metrics.json`, `equity.csv`.
Hourly marking and hourly `risk.check` (a binding stop/kill writes a `risk_exit` record); decisions at
DECISION_HOURS; baseline targets computed (from the true, unshuffled cards) and logged every step even
for LLM variants. equity.csv = pre-trade mark of every hour (first row = START_EQUITY). An LLM endpoint
failure stops the replay (nothing cached, so not reproducible).
Live: `uv run python -m sentiment.live --run [--shadow]` (or `--once` per hour) steps the same
`replay.decision_step` / `risk_exit_step` on `DemoBroker` over a Store rebuilt from topped-up
data/raw -> `out/live/log.jsonl` (mode "live"), `equity.csv`. The risk version is read once per
replay run (live: per step; `--risk-version` or env SENTIMENT_RISK_VERSION). Under v0 the runner writes
the pre-v1 record format and prompt, so an offline v0 rerun reproduces the old `log_last_hash`.
metrics.json adds `risk_version, prompt_version, n_decisions_beta_fallback, n_decisions_beta_short`.
`sentiment/reconcile.py` reconciles each live record under its own risk version and rebuilds the v1 risk
context point-in-time (a logged context that differs is a `risk_ctx` mismatch).

## Metrics — `sentiment/metrics.py`
`def compute(run_dir) -> dict`: midnight-to-midnight UTC daily returns (N full days -> N returns) ->
Sharpe/Sortino (annualised √365, 24/7 market), max DD, total return and cost ladder (gross / fee /
spread / funding) against the first equity row (the pre-trade start), win rate (closed round trips;
qty <= 0 or non-finite fills ignored), turnover (annualised), trades, n_decisions, n_invalid,
n_risk_exits. `def increment(run_a, run_b, block_days=5, n=2000)`:
stationary block bootstrap CI of daily-return Sharpe difference.

## Tests — `tests/`
No look-ahead (Store never returns data after t; features at t unchanged when data after t is
deleted), ledger hash chain, sim cost arithmetic on a toy book, risk rules each bind once.
Run with `uv run pytest -q tests`.
