"""B4 paper log: F&G-scaled MSTR long replayed on Bitget spot RMSTRUSDT 1h bars since the listing.

Each new F&G value (stamped d 00:00 UTC) becomes usable at d 00:00 + lag; the first bar opening at
or after that time executes the rebalance at its open (+/- half spread, plus the taker fee) when
|target - current weight| >= BAND. One hash-chained ledger record per new F&G value; hourly marks
at bar close go to equity.csv. Status: estimated (replayed fills, no orders sent).

Run: uv run python -m b4.paper [--lag-h 1]   -> out/b4/paper[_lagNh]/{log.jsonl,equity.csv,metrics.json}
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from b4.backtest import weight
from b4.config import BAND, FEE, FNG_LAG_H, HALF_SPREAD, OUT, PAPER_START, RAW, START_EQUITY, SYMBOL
from sentiment.ledger import Ledger, verify


def replay(lag_h: int) -> dict:
    bars = pd.read_parquet(RAW / "rtoken.parquet")
    bars["t"] = pd.to_datetime(bars.ts, unit="ms", utc=True)
    bars = bars[bars.t >= pd.Timestamp(PAPER_START, tz="UTC")].reset_index(drop=True)
    fg = pd.read_parquet(RAW / "fng.parquet")
    fg["avail"] = fg.date + pd.Timedelta(hours=lag_h)
    # the last value already usable at the start, then every later one
    fg = fg[fg.avail >= fg.avail[fg.avail <= bars.t.iloc[0]].max()].reset_index(drop=True)

    d = OUT / ("paper" if lag_h == FNG_LAG_H else f"paper_lag{lag_h}h")
    d.mkdir(parents=True, exist_ok=True)
    (d / "log.jsonl").unlink(missing_ok=True)
    led = Ledger(d / "log.jsonl")

    cash, qty, marks, n_trades, fees, spread = START_EQUITY, 0.0, [], 0, 0.0, 0.0
    queue = list(fg.itertuples(index=False))
    for b in bars.itertuples(index=False):
        due = [q for q in queue if q.avail <= b.t]
        if due:
            q = due[-1]                                    # several pending (gap in bars): act on the newest
            queue = [x for x in queue if x.avail > b.t]
            eq_open = cash + qty * b.open
            w_now = qty * b.open / eq_open
            target = float(weight(pd.Series([q.value])).iloc[0])
            rec = {"ts": b.t.isoformat(), "mode": "replay", "symbol": SYMBOL, "variant": f"fg_long_lag{lag_h}h",
                   "fng_date": q.date.date().isoformat(), "fng": int(q.value), "fng_available_at": q.avail.isoformat(),
                   "skipped_fng_dates": [x.date.date().isoformat() for x in due[:-1]],
                   "target_w": round(target, 6), "w_before": round(w_now, 6), "equity_before": round(eq_open, 4),
                   "order": None}
            if abs(target - w_now) >= BAND or (target == 0 and qty > 0):
                side = 1 if target > w_now else -1
                px = b.open * (1 + side * HALF_SPREAD)
                # quantity that brings the weight at the bar open to the target (costs come out of cash)
                dq = (target * eq_open - qty * b.open) / b.open if target > 0 else -qty
                if dq > 0:                                 # spot, no borrowing: the fee must fit in cash
                    dq = min(dq, cash / (px * (1 + FEE)))
                notional = abs(dq) * px
                fee = notional * FEE
                cash -= dq * px + fee
                qty += dq
                n_trades += 1
                fees += fee
                spread += abs(dq) * b.open * HALF_SPREAD
                rec["order"] = {"side": "buy" if dq > 0 else "sell", "qty": round(abs(dq), 6), "fill_px": round(px, 4),
                                "bar_open": b.open, "notional": round(notional, 4), "fee": round(fee, 4)}
            rec.update({"qty_after": round(qty, 6), "cash_after": round(cash, 4),
                        "equity_after_open": round(cash + qty * b.open, 4)})
            led.append(rec)
        marks.append({"ts": b.t, "close": b.close, "qty": qty, "cash": cash, "equity": cash + qty * b.close,
                      "w": qty * b.close / (cash + qty * b.close)})

    eq = pd.DataFrame(marks).set_index("ts")
    eq.to_csv(d / "equity.csv")
    ok = verify(d / "log.jsonl")
    n = sum(1 for ln in (d / "log.jsonl").read_text().splitlines() if ln.strip())
    daily = eq.equity.resample("1D").last().dropna()
    bh = eq.close.resample("1D").last().dropna()
    mean_w = float(eq.w.mean())

    def m(s: pd.Series) -> dict:
        r = s.pct_change().dropna()
        dd = float((s / s.cummax() - 1).min())
        return {"return": float(s.iloc[-1] / s.iloc[0] - 1), "sharpe_daily_365": float(r.mean() / r.std() * np.sqrt(365)),
                "max_dd": dd}
    const = START_EQUITY * (1 + mean_w * (bh / bh.iloc[0] - 1))   # exposure-matched, bought once, no costs
    res = {"window": [eq.index[0].isoformat(), eq.index[-1].isoformat()], "bars": len(eq), "lag_h": lag_h,
           "records": n, "ledger_ok": ok, "last_hash": led.last_hash, "trades": n_trades,
           "fees": round(fees, 2), "spread_cost": round(spread, 2), "mean_w": mean_w,
           "start_equity": START_EQUITY, "end_equity": float(eq.equity.iloc[-1]),
           "fg_long": m(pd.concat([pd.Series([START_EQUITY], index=[eq.index[0] - pd.Timedelta(hours=1)]), daily])),
           "buy_hold_rtoken": m(bh), "constant_mean_w": m(const),
           "fng_range": [int(fg.value.min()), int(fg.value.max())],
           "by_month": {str(k): {"fg_long": float(v.equity.iloc[-1] / v.equity.iloc[0] - 1),
                                 "rtoken": float(v.close.iloc[-1] / v.close.iloc[0] - 1), "mean_w": float(v.w.mean())}
                        for k, v in eq.groupby(eq.index.strftime("%Y-%m"))}}
    (d / "metrics.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag-h", type=int, default=FNG_LAG_H)
    res = replay(ap.parse_args().lag_h)
    print(json.dumps({k: v for k, v in res.items() if k != "by_month"}, indent=1))
    print(pd.DataFrame(res["by_month"]).T.map(lambda v: f"{v:+.1%}" if abs(v) < 5 else v).to_string())


if __name__ == "__main__":
    main()
