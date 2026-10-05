"""Run metrics and the LLM-vs-baseline increment (CONTRACT: Metrics).

`compute(run_dir)` reads `equity.csv` (ts, equity, gross_equity, fees, spread, funding; costs
cumulative as in sentiment/sim.py) and the records in `log.jsonl`. The first equity row is the
pre-trade mark at the start (START_EQUITY), so total return, drawdown, the first daily return and the
cost ladder all include the cost of building the first book. Returns are daily, from 00:00 UTC to
00:00 UTC (each day's return is labelled by its closing midnight; a run of N full days has N returns),
annualised with sqrt(365) because the perps trade 24/7. Only COMPLETE days count (`complete_days`): a
curve that starts after a midnight (the live run) or ends after its last midnight (the live run, an
aborted replay) keeps those partial days out of n_days, Sharpe and Sortino and reports them separately
(`partial_first_day_h`, `partial_last_day_h`, `partial_last_day_return`); total return and max DD still
use the whole curve. Max drawdown uses the full hourly curve. Win
rate counts closed round trips per ticker (flat -> position -> flat or flip), average-cost P&L after
fees, funding excluded (it is not attributed per trip); fills with qty <= 0 or no finite price are
ignored. Also counted: decisions, invalid decisions (held by the risk layer) and hourly risk exits.

`increment(run_a, run_b)` is a stationary block bootstrap (Politis-Romano, mean block
`block_days`, fixed seed) of the daily-return Sharpe difference a - b on the common complete days. A
path whose Sharpe is not finite (a resample with zero variance) is dropped; `n_boot_dropped` counts
them and `n_boot_used` = n - dropped is the number the CI and P(diff <= 0) come from.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANN = 365
SEED = 20260923


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def load_equity(run_dir) -> pd.DataFrame:
    e = pd.read_csv(Path(run_dir) / "equity.csv")
    e["ts"] = pd.to_datetime(e["ts"], utc=True)
    return e.set_index("ts").sort_index()


def daily_returns(equity: pd.Series) -> pd.Series:
    """Midnight-to-midnight UTC returns, labelled by the closing midnight. Buckets are (d-1 00:00, d 00:00],
    so the last value of each bucket is the midnight mark; a first point exactly on a midnight is the
    base of the first return, not a return of its own."""
    d = equity.resample("1D", closed="right", label="right").last().dropna()
    prev = d.shift(1)
    prev.iloc[0] = equity.iloc[0]
    r = (d / prev - 1).rename("ret")
    return r.iloc[1:] if equity.index[0] == d.index[0] else r


def complete_days(equity: pd.Series) -> tuple[pd.Series, dict]:
    """(the part of the curve spanning complete UTC days, partial-day info). The kept part runs from the
    first midnight at or after the first mark to the last midnight at or before the last mark, so its
    `daily_returns` are full 00:00-00:00 days only. Info: `partial_first_day_h` (hours before that first
    midnight; 0 when the curve contains no midnight), `partial_last_day_h` (hours the curve covers after its
    last midnight, or its whole span when it contains no midnight) and `partial_last_day_return` (last mark
    over the last mark at or before that midnight, else the first mark; None without a partial last day)."""
    if not len(equity):
        return equity, {"partial_first_day_h": 0.0, "partial_last_day_h": 0.0, "partial_last_day_return": None}
    first, last = equity.index[0], equity.index[-1]
    first_mid, last_mid = first.ceil("D"), last.floor("D")
    spans = first_mid <= last_mid                     # the curve contains at least one midnight
    full = equity[(equity.index >= first_mid) & (equity.index <= last_mid)] if spans else equity.iloc[:0]
    last_h = (last - max(last_mid, first)).total_seconds() / 3600
    part = None
    if last_h > 0:
        base = equity[equity.index <= last_mid]
        b = float(base.iloc[-1]) if len(base) else float(equity.iloc[0])
        part = float(equity.iloc[-1]) / b - 1 if b else None
    return full, {"partial_first_day_h": (first_mid - first).total_seconds() / 3600 if spans else 0.0,
                  "partial_last_day_h": last_h, "partial_last_day_return": part}


def complete_day_returns(equity: pd.Series) -> pd.Series:
    """`daily_returns` over complete UTC days only (see `complete_days`)."""
    full, _ = complete_days(equity)
    return daily_returns(full) if len(full) > 1 else pd.Series(dtype=float, name="ret")


def sharpe(r) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else float("nan")
    return float(r.mean() / sd * math.sqrt(ANN)) if sd and sd > 0 else float("nan")


def sortino(r) -> float:
    r = np.asarray(r, dtype=float)
    dd = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if len(r) else float("nan")
    return float(r.mean() / dd * math.sqrt(ANN)) if dd and dd > 0 else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def load_records(run_dir) -> list[dict]:
    p = Path(run_dir) / "log.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def load_fills(run_dir) -> list[dict]:
    return [f for r in load_records(run_dir) for f in (r.get("fills") or [])]


def _real_fill(f: dict) -> bool:
    try:
        return float(f["qty"]) > 0 and math.isfinite(float(f["price"]))
    except (KeyError, TypeError, ValueError):
        return False


def round_trips(fills: list[dict]) -> list[float]:
    """Closed round-trip P&L per ticker (average cost, fees included)."""
    book: dict[str, list[float]] = {}                 # ticker -> [pos, avg_price, trip_pnl]
    trips = []
    for f in sorted((f for f in fills if _real_fill(f)), key=lambda f: str(f["ts"])):
        q = float(f["qty"]) * (1 if f["side"] == "buy" else -1)
        px, fee = float(f["price"]), float(f.get("fee", 0.0))
        pos, avg, pnl = book.get(f["ticker"], [0.0, 0.0, 0.0])
        tol = 1e-9 * max(1.0, abs(q))
        if abs(pos) <= tol or np.sign(q) == np.sign(pos):
            pos, avg = pos + q, (pos * avg + q * px) / (pos + q)
            pnl -= fee
        else:
            close = min(abs(q), abs(pos))
            pnl += close * (px - avg) * np.sign(pos) - fee * close / abs(q)
            pos += np.sign(q) * close
            rest = abs(q) - close
            if abs(pos) <= tol:
                trips.append(pnl)
                pos, avg, pnl = 0.0, 0.0, 0.0
                if rest > tol:
                    pos, avg, pnl = np.sign(q) * rest, px, -fee * rest / abs(q)
        book[f["ticker"]] = [pos, avg, pnl]
    return trips


def compute(run_dir) -> dict:
    e = load_equity(run_dir)
    eq = e["equity"]
    r = complete_day_returns(eq)
    _, partial = complete_days(eq)
    days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400, 1e-9)
    recs = load_records(run_dir)
    fills = [f for r in recs for f in (r.get("fills") or []) if _real_fill(f)]
    decisions = [r for r in recs if r.get("event", "decision") == "decision"]
    traded = sum(float(f["qty"]) * float(f["price"]) for f in fills)
    trips = round_trips(fills)
    start = float(eq.iloc[0])
    last = e.iloc[-1]
    gross = float(last.get("gross_equity", last["equity"]))
    fees, spread, funding = (float(last.get(c, 0.0)) for c in ("fees", "spread", "funding"))
    ladder = {"gross": gross / start - 1,
              "after_fees": (gross - fees) / start - 1,
              "after_spread": (gross - fees - spread) / start - 1,
              "after_funding": (gross - fees - spread + funding) / start - 1}
    return _clean({
        "start": str(eq.index[0]), "end": str(eq.index[-1]), "n_days": len(r), **partial,
        "start_equity": start, "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1]) / start - 1,
        "sharpe": sharpe(r), "sortino": sortino(r), "max_dd": max_drawdown(eq),
        "gross_sharpe": sharpe(complete_day_returns(e["gross_equity"])) if "gross_equity" in e else None,
        "win_rate": (sum(p > 0 for p in trips) / len(trips)) if trips else float("nan"),
        "round_trips": len(trips), "trades": len(fills),
        "n_decisions": len(decisions),
        "n_invalid": sum(1 for r in decisions if not (r.get("decision") or {}).get("valid", True)),
        "n_risk_exits": sum(1 for r in recs if r.get("event") == "risk_exit"),
        "turnover_ann": traded / float(eq.mean()) * ANN / days,
        "costs": {"fees": fees, "spread": spread, "funding": funding},
        "ladder": ladder,
    })


def stationary_bootstrap_idx(t: int, n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """n x t index matrix: each path restarts at a random day with prob 1/block, else steps on (wrapping)."""
    idx = np.empty((n, t), dtype=int)
    idx[:, 0] = rng.integers(0, t, n)
    jump = rng.random((n, t)) < 1.0 / block
    fresh = rng.integers(0, t, (n, t))
    for j in range(1, t):
        idx[:, j] = np.where(jump[:, j], fresh[:, j], (idx[:, j - 1] + 1) % t)
    return idx


def increment(run_a, run_b, block_days: int = 5, n: int = 2000) -> dict:
    ra = complete_day_returns(load_equity(run_a)["equity"])
    rb = complete_day_returns(load_equity(run_b)["equity"])
    j = pd.concat([ra.rename("a"), rb.rename("b")], axis=1, join="inner").dropna()
    a, b = j["a"].to_numpy(), j["b"].to_numpy()
    diff = sharpe(a) - sharpe(b)
    if len(j):
        idx = stationary_bootstrap_idx(len(j), n, block_days, np.random.default_rng(SEED))
        boot = np.array([sharpe(a[i]) - sharpe(b[i]) for i in idx])
    else:
        boot = np.full(n, np.nan)
    dropped = int((~np.isfinite(boot)).sum())
    boot = boot[np.isfinite(boot)]
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (float("nan"),) * 2)
    return _clean({
        "run_a": str(run_a), "run_b": str(run_b), "n_days": len(j), "block_days": block_days,
        "n_boot": n, "n_boot_dropped": dropped, "n_boot_used": n - dropped, "seed": SEED,
        "sharpe_a": sharpe(a), "sharpe_b": sharpe(b),
        "sharpe_diff": diff, "ci95": [float(lo), float(hi)],
        "p_diff_le_0": float(np.mean(boot <= 0)) if len(boot) else float("nan"),
    })


if __name__ == "__main__":
    args = sys.argv[1:]
    out = increment(*args[:2]) if len(args) == 2 else compute(args[0])
    print(json.dumps(out, indent=2))
