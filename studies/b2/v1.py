"""v1 trials of the crisis-derisk module: a small pre-declared grid evaluated on the DEV period only, the chosen
configuration then run ONCE on the holdout period. Records go to studies/b2/trials.log by hand.

Dev = 2019-08-01 .. 2022-12-31 (events E1-E5), holdout = 2023-01-01 .. now (E6-E9).

Grid (16 configurations; C1 and C3 kept at v0):
  C2: off | vol ratio >= 3.0 (98th pct of the ratio, measured in v0 diagnostics)
  C4: off | social share >= 95th pct on 2 consecutive days (and r_24h < 0)
  re-entry after the cooldown: 'recover' (open > exit price) | 'ma20' (daily close > 20-day MA)
  base exposure: 1 | 200-day MA rule
Selection on dev: Calmar of the module over its own base, minus the base's Calmar (the module must add to
what it gates), tie-break by fewer false alarms.

Run: uv run python -m studies.b2.v1  -> out/b2/v1_grid.json, v1_holdout.json, and metrics for the chosen config.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from studies.b2 import backtest as B
from studies.b2 import config as C

DEV_END = pd.Timestamp("2022-12-31 23:00", tz="UTC")
HOLD_START = pd.Timestamp("2023-01-01 00:00", tz="UTC")
GRID = {"c2": [None, 3.0], "c4": [None, (0.95, 2)], "c5": [None, C.DVOL_MULT], "reentry": ["recover", "ma20"], "base": ["one", "ma200"]}


def rules(f: pd.DataFrame, c2, c4, c5=None) -> pd.Series:
    trig = f.C1 | f.C3
    if c5 is not None:
        trig = trig | ((f.dvol >= c5 * f.dvol_med_30d) & (f.r_24h < 0))
    if c2 is not None:
        trig = trig | ((f.rv_24h >= c2 * f.rv_med_90d) & (f.r_24h < 0))
    if c4 is not None:
        q, days = c4
        hot = (f.social_q >= q)
        # 'days' consecutive daily observations: the hourly series is a step function, so require it to have been
        # true for days*24 hours
        hot = hot.rolling(days * 24, min_periods=days * 24).min().fillna(0) > 0
        trig = trig | (hot & (f.r_24h < 0))
    return trig


def exposure(f: pd.DataFrame, trig: pd.Series, reentry: str, base: pd.Series) -> pd.Series:
    t, r24, op = trig.to_numpy(), f.r_24h.to_numpy(), f.open.to_numpy()
    ma20 = f.ma20_ok.to_numpy()
    out, state, cd, exit_px = np.zeros(len(f)), True, -1, np.nan
    for i in range(len(f)):
        if t[i]:
            if state:
                exit_px = op[i]                                  # ~ the price at which we leave (next open)
            state, cd = False, i + C.COOLDOWN_H
        elif not state and i >= cd:
            ok = (op[i] > exit_px) if reentry == "recover" else bool(ma20[i])
            if ok:
                state = True
        out[i] = base.iloc[i] if state else 0.0
    return pd.Series(out, index=f.index).shift(1).fillna(0.0)


def evaluate(f: pd.DataFrame, pos: pd.Series, base_pos: pd.Series, lo, hi) -> dict:
    sl = slice(lo, hi)
    fw, pw, bw = f.loc[sl], pos.loc[sl], base_pos.loc[sl]
    eq, cost = B.equity(pw, fw.open)
    eqb, costb = B.equity(bw, fw.open)
    m, mb = B.metrics(eq, pw, cost), B.metrics(eqb, bw, costb)
    g = fw.copy()
    g["trigger"] = g["trigger_v1"]
    ep = B.exit_episodes(g, pw)
    fa = sum(1 for e in ep if e["false_alarm"])
    ev = [e for e in B.event_study(g, pw) if lo <= pd.Timestamp(e["start"], tz="UTC") <= hi]
    return {"module": m, "base": mb, "episodes": len(ep), "false_alarms": fa,
            "calmar_increment": (m["calmar"] - mb["calmar"]) if np.isfinite(m["calmar"]) and np.isfinite(mb["calmar"]) else np.nan,
            "dd_increment": m["max_dd"] - mb["max_dd"], "events": ev,
            "events_hit": sum(1 for e in ev if e.get("triggered")), "events_n": len(ev)}


def main() -> None:
    p = B.load_perp()
    f = B.features(p)
    bench = B.benchmarks(f)
    daily = f.close[f.index.hour == 23]
    daily.index = daily.index + B.H
    ma20 = (daily > daily.rolling(20).mean()).astype(float)
    f["ma20_ok"] = ma20.reindex(ma20.index.union(f.index)).ffill().reindex(f.index).fillna(1.0)
    bases = {"one": pd.Series(1.0, index=f.index), "ma200": bench["ma200"].shift(-1).fillna(1.0)}
    rows = []
    for c2, c4, c5, re, base in itertools.product(*GRID.values()):
        trig = rules(f, c2, c4, c5)
        f["trigger_v1"] = trig
        pos = exposure(f, trig, re, bases[base])
        base_pos = bases[base].shift(1).fillna(0.0)
        dev = evaluate(f, pos, base_pos, f.index[0], DEV_END)
        rows.append({"c2": c2, "c4": c4, "c5": c5, "reentry": re, "base": base, "dev": dev})
        d = dev
        print(f"c2={str(c2):4s} c4={'q95x2' if c4 else 'off ':5s} c5={str(c5):4s} re={re:7s} base={base:5s} | dev module ret {d['module']['total_return']:+7.1%} "
              f"DD {d['module']['max_dd']:6.1%} calmar {d['module']['calmar']:5.2f} | base calmar {d['base']['calmar']:5.2f} "
              f"incr {d['calmar_increment']:+5.2f} | ep {d['episodes']:3d} FA {d['false_alarms']:3d} hits {d['events_hit']}/{d['events_n']}")
    (C.OUT / "v1_grid.json").write_text(json.dumps(rows, indent=1, default=str))
    ok = [r for r in rows if np.isfinite(r["dev"]["calmar_increment"])]
    best = max(ok, key=lambda r: (r["dev"]["calmar_increment"], -r["dev"]["false_alarms"]))
    print("\nCHOSEN on dev:", {k: best[k] for k in ("c2", "c4", "c5", "reentry", "base")})
    # holdout, once
    trig = rules(f, best["c2"], best["c4"], best["c5"])
    f["trigger_v1"] = trig
    pos = exposure(f, trig, best["reentry"], bases[best["base"]])
    base_pos = bases[best["base"]].shift(1).fillna(0.0)
    hold = evaluate(f, pos, base_pos, HOLD_START, f.index[-1])
    full = evaluate(f, pos, base_pos, f.index[0], f.index[-1])
    # the same window for the other references
    refs = {}
    for k in ("buy_hold", "ma200", "voltarget"):
        eq, cost = B.equity(bench[k].loc[HOLD_START:], f.open.loc[HOLD_START:])
        refs[k] = B.metrics(eq, bench[k].loc[HOLD_START:], cost)
    res = {"chosen": {k: best[k] for k in ("c2", "c4", "c5", "reentry", "base")}, "dev": best["dev"], "holdout": hold,
           "holdout_refs": refs, "full": full, "label": "estimated (replay); dev used for selection, holdout run once"}
    (C.OUT / "v1_holdout.json").write_text(json.dumps(res, indent=1, default=str))
    h = hold
    print(f"\nHOLDOUT 2023-01..now: module ret {h['module']['total_return']:+7.1%} DD {h['module']['max_dd']:6.1%} calmar {h['module']['calmar']:5.2f} "
          f"sharpe {h['module']['sharpe']:4.2f} in-mkt {h['module']['time_in_market']:3.0%} | base ({best['base']}) ret {h['base']['total_return']:+7.1%} "
          f"DD {h['base']['max_dd']:6.1%} calmar {h['base']['calmar']:5.2f} | ep {h['episodes']} FA {h['false_alarms']} hits {h['events_hit']}/{h['events_n']}")
    for k, m in refs.items():
        print(f"  ref {k:9s} ret {m['total_return']:+7.1%} DD {m['max_dd']:6.1%} calmar {m['calmar']:5.2f} sharpe {m['sharpe']:4.2f}")
    for e in h["events"]:
        print(f"  {e['id']} {e['name']:22s} trig={e.get('triggered')} h={e.get('hours_after_start')} rules={e.get('rules')} "
              f"trig->trough={e.get('btc_trigger_to_trough')} trig->reentry={e.get('btc_trigger_to_reentry')}")
    sig = pd.DataFrame({"pos_v1": pos, "trigger_v1": trig})
    sig.to_parquet(C.OUT / "v1_signals.parquet")


if __name__ == "__main__":
    main()


def lead_time(f: pd.DataFrame) -> list[dict]:
    """Per event and per rule: the first hour the rule is true in [start-5d, start+10d], its lag to the event start,
    and BTC's move from that hour's next open to the 30-day trough after the event (what an exit there avoids).
    C2 at v0 (2x) and v1 (3x), C4 at v0 (q90) and v1 (q95 x 2 days), C5 = DVOL 1.25x."""
    out = []
    cands = {"C1": f.C1, "C2_v0(2x)": f.C2, "C2_v1(3x)": (f.rv_24h >= 3.0 * f.rv_med_90d) & (f.r_24h < 0),
             "C3": f.C3, "C4_v0(q90)": f.C4,
             "C4_v1(q95x2d)": ((f.social_q >= 0.95).rolling(48, min_periods=48).min().fillna(0) > 0) & (f.r_24h < 0),
             "C5_dvol": f.C5}
    for eid, name, start, kind, split in B.EVENTS:
        t0 = pd.Timestamp(start, tz="UTC")
        if t0 < f.index[0]:
            continue
        after = f.open.loc[t0:t0 + pd.Timedelta(days=30)]
        trough_t, trough = after.idxmin(), float(after.min())
        row = {"id": eid, "name": name, "start": start, "trough": str(trough_t),
               "btc_start_to_trough": float(trough / f.open.asof(t0) - 1)}
        for k, s in cands.items():
            w = s.loc[t0 - pd.Timedelta(days=5): t0 + pd.Timedelta(days=10)]
            hit = w.index[w.fillna(False).astype(bool)]
            if len(hit) and hit[0] + B.H <= trough_t:
                p1 = float(f.open.asof(hit[0] + B.H))
                row[k] = {"first": str(hit[0]), "lag_h": float((hit[0] - t0) / B.H), "avoidable": float(trough / p1 - 1)}
            elif len(hit):
                row[k] = {"first": str(hit[0]), "lag_h": float((hit[0] - t0) / B.H), "avoidable": 0.0, "after_trough": True}
            else:
                row[k] = None
        out.append(row)
    (C.OUT / "v1_leadtime.json").write_text(json.dumps(out, indent=1, default=str))
    return out
