"""Crisis-derisk module v0: features, rules C1-C4, exposure policy, benchmarks, event study.

Clock: a signal is computed on the close of hourly bar i and acted on at the OPEN of bar i+1. Positions are
held in fractions of equity; equity marks at every bar open. Daily inputs (stablecoin prices, Augmento
counts, F&G) for date d become visible at d+1 00:00 UTC.

Run: uv run python -m studies.b2.backtest   -> out/b2/{signals.parquet, equity.csv, metrics.json, events.json}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from studies.b2 import config as C

H = pd.Timedelta(hours=1)
EVENTS = [  # (id, name, start UTC, type, split)
    ("E1", "COVID crash", "2020-03-12 00:00", "macro+liquidation", "dev"),
    ("E2", "May-19 deleveraging", "2021-05-19 00:00", "liquidation", "dev"),
    ("E3", "Terra/UST", "2022-05-09 00:00", "structural", "dev"),
    ("E4", "Celsius/3AC", "2022-06-12 00:00", "structural", "dev"),
    ("E5", "FTX", "2022-11-08 00:00", "structural", "dev"),
    ("E6", "SVB/USDC depeg", "2023-03-10 00:00", "structural", "holdout"),
    ("E7", "Yen carry unwind", "2024-08-05 00:00", "macro", "holdout"),
    ("E8", "Tariff drawdown", "2025-02-24 00:00", "macro-trend", "holdout"),
    ("E9", "10/11 liquidation", "2025-10-10 20:00", "macro+liquidation", "holdout"),
]


# ---------------------------------------------------------------- inputs

def load_perp() -> pd.DataFrame:
    p = pd.read_parquet(C.RAW / "perp.parquet")
    p["t"] = pd.to_datetime(p.ts, unit="ms", utc=True)
    p = p.set_index("t").sort_index()
    p = p[~p.index.duplicated()]
    full = pd.date_range(p.index[0], p.index[-1], freq="h")
    gaps = len(full) - len(p)
    p = p.reindex(full)
    p[["open", "high", "low", "close"]] = p[["open", "high", "low", "close"]].ffill()   # gap: carry the last price
    p.attrs["gaps"] = int(gaps)
    return p


def daily_to_hourly(s: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily series indexed by date d -> hourly series visible from d+1 00:00 UTC (step, ffill)."""
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index) + pd.Timedelta(days=1)
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def stable_min(index) -> tuple[pd.Series, pd.Series]:
    """Lowest daily stablecoin price across the list, and a 'fresh break' flag: some coin closed <= 1-DEPEG today
    while it closed >= 0.99 seven days earlier (a dead coin such as UST after May 2022 must not keep C3 on)."""
    st = pd.read_parquet(C.RAW / "stables.parquet")
    px = st.pivot(index="date", columns="sym", values="price").sort_index()
    major = px[[c for c in px.columns if c in C.MAJOR_STABLES]]
    fresh = ((major <= 1 - C.DEPEG) & (major.shift(7) >= 0.99)).any(axis=1)
    return daily_to_hourly(px.min(axis=1), index), daily_to_hourly(fresh.astype(float), index)


def social_share(index) -> tuple[pd.Series, pd.Series]:
    a = pd.read_parquet(C.RAW / "augmento.parquet")
    topics = [c for c in a.columns if c not in ("date", "source")]
    day = a.groupby("date")[topics].sum()
    total = day.sum(axis=1)
    crisis = day[list(C.CRISIS_TOPICS)].sum(axis=1)
    share = (crisis / total.replace(0, np.nan))
    q = share.rolling(365, min_periods=180).apply(lambda w: (w[:-1] <= w[-1]).mean() if len(w) > 1 else np.nan, raw=True)
    return daily_to_hourly(share, index), daily_to_hourly(q, index)


def dvol(index) -> pd.Series:
    """Deribit DVOL hourly close, visible from the end of its hour (candle at ts covers ts..ts+1h)."""
    path = C.RAW / "dvol.parquet"
    if not path.exists():
        return pd.Series(np.nan, index=index)
    d = pd.read_parquet(path)
    s = pd.Series(d.close.to_numpy(), index=pd.to_datetime(d.ts, unit="ms", utc=True) + H)
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def fng(index) -> pd.Series:
    f = pd.read_parquet(C.RAW / "fng.parquet").set_index("date").value
    return daily_to_hourly(f, index)


# ---------------------------------------------------------------- features and rules

def features(p: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=p.index)
    f["close"], f["open"] = p.close, p.open
    f["r_24h"] = p.close / p.close.shift(24) - 1
    lr = np.log(p.close).diff()
    f["rv_24h"] = lr.rolling(24).std() * np.sqrt(24 * 365)
    f["rv_med_90d"] = f.rv_24h.rolling(90 * 24, min_periods=30 * 24).median()
    f["stable_min"], f["stable_break"] = stable_min(p.index)
    f["social_share"], f["social_q"] = social_share(p.index)
    f["fng"] = fng(p.index)
    f["dvol"] = dvol(p.index)
    f["dvol_med_30d"] = f.dvol.rolling(30 * 24, min_periods=7 * 24).median()
    # rules (evaluated on the bar close; True = exit)
    f["C1"] = f.r_24h <= C.R24_CRASH
    f["C2"] = (f.rv_24h >= C.VOL_MULT * f.rv_med_90d) & (f.r_24h < 0)
    f["C3"] = f.stable_break > 0
    f["C4"] = (f.social_q >= C.SOCIAL_Q) & (f.r_24h < 0)
    f["C5"] = (f.dvol >= C.DVOL_MULT * f.dvol_med_30d) & (f.r_24h < 0)      # not part of v0's trigger
    f["trigger"] = f[["C1", "C2", "C3", "C4"]].any(axis=1)
    return f


def module_exposure(f: pd.DataFrame, base: pd.Series | None = None) -> pd.Series:
    """Position for bar i+1 decided on the close of bar i. `base` = exposure the module gates (default 1)."""
    base = pd.Series(1.0, index=f.index) if base is None else base
    trig, r24 = f.trigger.to_numpy(), f.r_24h.to_numpy()
    out, state, cd = np.zeros(len(f)), True, -1          # state True = in the market
    for i in range(len(f)):
        if trig[i]:
            state, cd = False, i + C.COOLDOWN_H
        elif not state and i >= cd and r24[i] > C.REENTRY_R24:
            state = True
        out[i] = base.iloc[i] if state else 0.0
    return pd.Series(out, index=f.index).shift(1).fillna(0.0)       # acted on next bar


def benchmarks(f: pd.DataFrame) -> dict[str, pd.Series]:
    daily = f.close[f.index.hour == 23]                            # bar closing at 00:00 UTC
    daily.index = daily.index + H                                  # decision time 00:00 UTC
    ma = daily.rolling(C.MA_DAYS, min_periods=C.MA_DAYS).mean()
    ma_pos = (daily > ma).astype(float).where(ma.notna(), 1.0)     # before the MA exists: long
    dret = np.log(daily).diff()
    vol30 = dret.rolling(30).std() * np.sqrt(365)
    vt = (C.VOL_TARGET / vol30).clip(upper=1.0).where(vol30.notna(), 1.0)

    def hourly(s):  # daily decision at 00:00 -> position from the bar opening at 00:00 onwards
        return s.reindex(s.index.union(f.index)).ffill().reindex(f.index).fillna(1.0)

    ma_h, vt_h = hourly(ma_pos), hourly(vt)
    return {"buy_hold": pd.Series(1.0, index=f.index), "ma200": ma_h, "voltarget": vt_h,
            "module": module_exposure(f), "module_ma200": module_exposure(f, ma_h.shift(-1).fillna(1.0))}


# ---------------------------------------------------------------- P&L and metrics

def equity(pos: pd.Series, open_px: pd.Series, funding_8h: float = 0.0) -> tuple[pd.Series, float]:
    r = open_px.shift(-1) / open_px - 1                             # open i -> open i+1
    cost = (pos.diff().abs().fillna(pos.abs())) * (C.TAKER + C.HALF_SPREAD)
    fund = pos * funding_8h / 8
    step = (1 + pos * r.fillna(0)) * (1 - cost) * (1 - fund)
    eq = step.cumprod().shift(1).fillna(1.0)
    return eq, float(cost.sum())


def metrics(eq: pd.Series, pos: pd.Series, cost_total: float) -> dict:
    d = eq[eq.index.hour == 0]
    dr = d.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1
    dd = eq / eq.cummax() - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else np.nan
    return {"total_return": float(eq.iloc[-1] - 1), "cagr": float(cagr), "ann_vol": float(dr.std() * np.sqrt(365)),
            "sharpe": float(sharpe), "max_dd": float(dd.min()), "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else np.nan,
            "time_in_market": float(pos.mean()), "position_changes": int((pos.diff().abs() > 1e-9).sum()),
            "cost_paid_frac": cost_total, "years": float(years)}


def yearly(eqs: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for y in sorted(set(next(iter(eqs.values())).index.year)):
        row = {"year": int(y)}
        for k, e in eqs.items():
            s = e[e.index.year == y]
            row[k] = float(s.iloc[-1] / s.iloc[0] - 1) if len(s) > 1 else np.nan
        rows.append(row)
    return rows


def exit_episodes(f: pd.DataFrame, pos: pd.Series) -> list[dict]:
    """Each out-of-market episode of the plain module: trigger, rules, what was avoided / missed."""
    px = f.open
    out = []
    p = pos.to_numpy()
    i = 1
    while i < len(p):
        if p[i] == 0 and p[i - 1] > 0:
            j = i
            while j < len(p) and p[j] == 0:
                j += 1
            t_exit, t_re = f.index[i], (f.index[j] if j < len(p) else None)
            rules = [r for r in ("C1", "C2", "C3", "C4", "C5") if r in f and f[r].iloc[i - 1]]
            win = px.iloc[i:i + 30 * 24]
            low = float(win.min() / px.iloc[i] - 1)
            out.append({"exit": str(t_exit), "reentry": str(t_re) if t_re is not None else None,
                        "rules": rules, "hours_out": int(j - i),
                        "min_30d_from_exit": low, "false_alarm": low > -0.10,
                        "missed": float(px.iloc[j] / px.iloc[i] - 1) if t_re is not None else None,
                        "fng_at_exit": None if pd.isna(f.fng.iloc[i]) else int(f.fng.iloc[i])})
            i = j
        else:
            i += 1
    return out


def event_study(f: pd.DataFrame, pos: pd.Series) -> list[dict]:
    px = f.open
    rows = []
    for eid, name, start, kind, split in EVENTS:
        t0 = pd.Timestamp(start, tz="UTC")
        if t0 < f.index[0] or t0 > f.index[-1] - pd.Timedelta(days=1):
            rows.append({"id": eid, "name": name, "start": start, "type": kind, "split": split, "covered": False})
            continue
        lo, hi = t0 - pd.Timedelta(days=3), t0 + pd.Timedelta(days=10)
        w = f.loc[lo:hi]
        trig = w.index[w.trigger]
        p0 = float(px.asof(t0))
        after = px.loc[t0:t0 + pd.Timedelta(days=30)]
        trough_t = after.idxmin()
        row = {"id": eid, "name": name, "start": start, "type": kind, "split": split, "covered": True,
               "btc_start_to_trough": float(after.min() / p0 - 1), "trough": str(trough_t),
               "triggered": len(trig) > 0}
        if len(trig):
            t1 = trig[0] + H                                             # acted on the next open
            rules = [r for r in ("C1", "C2", "C3", "C4", "C5") if r in f and f.loc[trig[0], r]]
            p1 = float(px.asof(t1))
            pe = pos.loc[t1:]
            re_idx = pe.index[pe > 0]
            t_re = re_idx[0] if len(re_idx) else None
            row.update({"first_trigger": str(trig[0]), "hours_after_start": float((trig[0] - t0) / H), "rules": rules,
                        "btc_trigger_to_trough": float(after.loc[t1:].min() / p1 - 1) if t1 <= after.index[-1] else None,
                        "reentry": str(t_re) if t_re is not None else None,
                        "btc_trigger_to_reentry": float(px.asof(t_re) / p1 - 1) if t_re is not None else None})
        rows.append(row)
    return rows


def ablation(f: pd.DataFrame) -> list[dict]:
    """Diagnostic only: the module with each rule alone and with each rule removed."""
    rows = []
    variants = {"C1 only": f.C1, "C2 only": f.C2, "C3 only": f.C3, "C4 only": f.C4,
                "without C1": f.C2 | f.C3 | f.C4, "without C2": f.C1 | f.C3 | f.C4,
                "without C3": f.C1 | f.C2 | f.C4, "without C4": f.C1 | f.C2 | f.C3, "all (v0)": f.trigger}
    for name, mask in variants.items():
        g = f.copy()
        g["trigger"] = mask
        pos = module_exposure(g)
        eq, cost = equity(pos, f.open)
        m = metrics(eq, pos, cost)
        ep = exit_episodes(g, pos)
        ev = event_study(g, pos)
        rows.append({"variant": name, "total_return": m["total_return"], "max_dd": m["max_dd"], "calmar": m["calmar"],
                     "sharpe": m["sharpe"], "time_in_market": m["time_in_market"], "episodes": len(ep),
                     "false_alarms": sum(1 for e in ep if e["false_alarm"]),
                     "events_hit": sum(1 for e in ev if e.get("triggered"))})
    return rows


def main() -> None:
    C.OUT.mkdir(parents=True, exist_ok=True)
    p = load_perp()
    f = features(p)
    pos = benchmarks(f)
    eqs, mets = {}, {}
    for k, s in pos.items():
        eq, cost = equity(s, f.open)
        eqs[k], mets[k] = eq, metrics(eq, s, cost)
    eq_f, cost_f = equity(pos["module"], f.open, C.FUNDING_8H)
    mets["module_funding_sens"] = metrics(eq_f, pos["module"], cost_f)
    eq_bf, cost_bf = equity(pos["buy_hold"], f.open, C.FUNDING_8H)
    mets["buy_hold_funding_sens"] = metrics(eq_bf, pos["buy_hold"], cost_bf)
    episodes = exit_episodes(f, pos["module"])
    events = event_study(f, pos["module"])
    fa = [e for e in episodes if e["false_alarm"]]
    rule_counts = {r: int(f[r].sum()) for r in ("C1", "C2", "C3", "C4")}
    first_rule = {r: sum(1 for e in episodes if e["rules"] and e["rules"][0] == r) for r in ("C1", "C2", "C3", "C4")}
    summary = {
        "window": [str(f.index[0]), str(f.index[-1])], "bars": int(len(f)), "bar_gaps_filled": p.attrs["gaps"],
        "rules": {"R24_CRASH": C.R24_CRASH, "VOL_MULT": C.VOL_MULT, "DEPEG": C.DEPEG, "SOCIAL_Q": C.SOCIAL_Q,
                  "COOLDOWN_H": C.COOLDOWN_H, "REENTRY_R24": C.REENTRY_R24, "TAKER": C.TAKER, "HALF_SPREAD": C.HALF_SPREAD,
                  "FUNDING_8H_sens": C.FUNDING_8H, "MA_DAYS": C.MA_DAYS, "VOL_TARGET": C.VOL_TARGET},
        "rule_hours_true": rule_counts, "episodes": len(episodes), "episodes_first_rule": first_rule,
        "false_alarms": len(fa), "false_alarm_rate": (len(fa) / len(episodes)) if episodes else None,
        "false_alarm_missed_mean": float(np.mean([e["missed"] for e in fa if e["missed"] is not None])) if fa else None,
        "hit_missed_mean": float(np.mean([e["missed"] for e in episodes if not e["false_alarm"] and e["missed"] is not None])) if episodes else None,
        "events_covered": sum(1 for e in events if e["covered"]),
        "events_triggered": sum(1 for e in events if e.get("triggered")),
        "metrics": mets, "yearly": yearly(eqs), "ablation": ablation(f), "label": "estimated (replay, simulated fills at next 1h open)"}
    (C.OUT / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    (C.OUT / "events.json").write_text(json.dumps({"events": events, "episodes": episodes}, indent=2, default=str))
    sig = f[["open", "close", "r_24h", "rv_24h", "rv_med_90d", "stable_min", "stable_break", "social_share", "social_q", "fng",
             "C1", "C2", "C3", "C4", "trigger"]].copy()
    for k, s in pos.items():
        sig[f"pos_{k}"] = s
    sig.to_parquet(C.OUT / "signals.parquet")
    pd.DataFrame(eqs).to_csv(C.OUT / "equity.csv")
    for k, m in mets.items():
        print(f"{k:24s} ret {m['total_return']:+8.1%} cagr {m['cagr']:+7.1%} vol {m['ann_vol']:5.1%} sharpe {m['sharpe']:5.2f} "
              f"maxDD {m['max_dd']:7.1%} calmar {m['calmar']:5.2f} in-mkt {m['time_in_market']:4.0%} changes {m['position_changes']}")
    print(f"episodes {len(episodes)}, false alarms {len(fa)}; events triggered {summary['events_triggered']}/{summary['events_covered']}")


if __name__ == "__main__":
    main()
