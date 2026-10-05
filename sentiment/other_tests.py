"""The other strategies we tested after the v1 test window, read from their committed outputs (report section 7).

Each entry: id, title, question, design (what was fixed before the run), window, numbers, verdict, status and the
files the numbers come from. Nothing is recomputed here: the numbers are the ones the modules wrote (studies/b2/ -> out/b2/,
studies/b4/ -> out/b4/, studies/b5/ -> out/b5/), each run once against criteria written in its trials.log before the run. A missing
file gives status "not shipped" instead of failing the report.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from sentiment.config import ROOT

OUT = ROOT / "out"


def _j(rel: str):
    p = OUT / rel
    return json.loads(p.read_text()) if p.exists() else None


def _med(rows: list[dict], key: str) -> float | None:
    v = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(v) if v else None


def fg_cross_asset() -> dict:
    x = _j("b4/cross_asset.json")
    base = {"id": "fg_cross_asset", "title": "Crypto fear & greed as a contrarian signal, BTC and four crypto stocks",
            "question": "Buy fear and sell greed: does the crypto Fear & Greed index time BTC, MSTR, COIN, MARA and RIOT?",
            "design": ("long-only w = clip((75 - F&G)/50, 0, 1) and long/short w = clip((50 - F&G)/50, -1, 1), one "
                       "day lag, 10 bp per unit turnover; 2018-02..2022-12 and 2023-01..2026-10, each once; pass = beat "
                       "buy & hold with a block-bootstrap CI excluding 0 in both periods and beat the same rule on BTC "
                       "30-day momentum"),
            "window": ["2018-02-01", "2026-10-01"], "status": "estimated (daily closes, simulated)",
            "sources": ["studies/b4/cross_asset.py", "out/b4/cross_asset.json", "studies/b4/trials.log"]}
    if x is None:
        return {**base, "status": "not shipped"}
    s, st = x["summary"], x["strategies"]
    pick = {k: {c: round(v[c]["sharpe"], 4) for c in ("buy_hold", "fg_long", "fg_long_short", "mom_long", "ma200")
                if v[c]["sharpe"] == v[c]["sharpe"]} for k, v in st.items()}
    leg = {y: round(v["sum_of_daily"], 4) for y, v in x["short_leg_mstr"].items()}
    return {**base, "numbers": {
        "n_asset_periods": s["n_asset_periods"], "long_short_below_buy_hold": s["fg_long_short_below_bh"],
        "long_only_below_buy_hold": s["fg_long_below_bh"], "long_only_beats_buy_hold_ci": s["fg_long_beats_bh_ci_excl_0"],
        "p1_slopes_positive": s["p1_slopes_positive"], "p1_slopes_total": s["p1_slopes_total"],
        "resid_max_abs_t": round(s["resid_max_abs_t"], 4), "sharpe": pick, "mstr_short_leg_by_year": leg,
        "mstr_short_leg_2024": leg.get("2024"), "mstr_short_leg_2020": leg.get("2020")},
        "verdict": "not claimed: " + s["verdict"].split(": ", 1)[-1]}


def fg_mstr() -> dict:
    h, p, s = _j("b4/history.json"), _j("b4/paper/metrics.json"), _j("b4/paper_lag24h/metrics.json")
    base = {"id": "fg_mstr", "title": "Fear & greed on MSTR only, long-only, after its first bitcoin purchase",
            "question": "Restricted to MSTR as a bitcoin holder (from 2020-08-11) and to the long side: hold more in fear, less in greed?",
            "design": ("w = clip((75 - F&G)/50, 0, 1): full at F&G <= 25, flat at >= 75; 13 bp per unit turnover; "
                       "pass = Sharpe above buy & hold and above a constant weight with the same mean exposure, each with "
                       "a 95% block-bootstrap CI excluding 0. Paper log: the same rule on Bitget's RMSTRUSDT rToken, "
                       "1h bars since its listing, hash-chained"),
            "window": [None, None], "status": "estimated (daily closes and rToken bars, simulated fills)",
            "sources": ["studies/b4/backtest.py", "studies/b4/paper.py", "out/b4/history.json", "out/b4/paper/", "studies/b4/trials.log"]}
    if h is None or p is None:
        return {**base, "status": "not shipped"}
    r = h["rules"]
    fwd = h["fwd20_by_fg_bucket"]
    return {**base, "window": h["window"], "numbers": {
        "history": {k: {m: round(v[m], 4) for m in ("total", "cagr", "sharpe", "max_dd") if v.get(m) is not None}
                    | {"mean_w": round(v["mean_w"], 4)} for k, v in r.items()},
        "ci_vs_buy_hold": [round(v, 4) for v in h["ci_sharpe_fg_minus_buy_hold"]],
        "ci_vs_constant": [round(v, 4) for v in h["ci_sharpe_fg_minus_constant"]],
        "fwd20_le25": round(fwd["<=25"]["mean_fwd20"], 4), "fwd20_gt75": round(fwd[">75"]["mean_fwd20"], 4),
        "mean_w_2022": round(h["by_year"]["2022"]["mean_w"], 4), "mstr_2022": round(h["by_year"]["2022"]["mstr"], 4),
        "paper": {"window": p["window"], "records": p["records"], "trades": p["trades"], "ledger_ok": p["ledger_ok"],
                  "last_hash": p["last_hash"], "return": round(p["fg_long"]["return"], 4),
                  "max_dd": round(p["fg_long"]["max_dd"], 4), "rtoken_return": round(p["buy_hold_rtoken"]["return"], 4),
                  "constant_return": round(p["constant_mean_w"]["return"], 4), "mean_w": round(p["mean_w"], 4),
                  "return_lag24h": round(s["fg_long"]["return"], 4) if s else None}},
        "verdict": ("not claimed: Sharpe {:.2f} vs buy & hold {:.2f} and a constant {:.0%} weight {:.2f}; greed was "
                    "followed by higher MSTR returns, not lower".format(r["fg_long"]["sharpe"], r["buy_hold"]["sharpe"],
                                                                       r["fg_long"]["mean_w"],
                                                                       r["constant_mean_w"]["sharpe"]))}


def vol_managed() -> dict:
    s, d = _j("b5/summary.json"), _j("b5/daily_check.json")
    base = {"id": "vol_managed", "title": "No news: the ten names long, cut when volatility rises",
            "question": "Drop news, ratings and insider filings; hold the ten names long and scale each by its volatility. Better than holding the same average exposure?",
            "design": ("cap_i = 0.10 x min(1, median vol_7d over the last 60 days / vol_7d); decisions every 4h; v1's "
                       "simulator and costs; pass = Sharpe above a constant weight with the same mean exposure, CI "
                       "excluding 0, in the test window AND on 11 months of perp bars AND on 2021-26 daily stock closes"),
            "window": ["2021-01-04", "2026-10-01"], "status": "estimated (replay and daily closes, simulated fills)",
            "sources": ["studies/b5/run.py", "studies/b5/compare.py", "studies/b5/daily_check.py", "out/b5/summary.json",
                        "out/b5/daily_check.json", "studies/b5/trials.log"]}
    if s is None or d is None:
        return {**base, "status": "not shipped"}
    win = {}
    for w, x in s.items():
        v = x["variants"]
        win[w] = {"window": x["window"],
                  **{k: {"return": round(m["total_return"], 4), "sharpe": None if m["sharpe"] is None else round(m["sharpe"], 4),
                         "max_dd": round(m["max_dd"], 4), "trades": m["trades"], "gross": round(m["mean_gross_exposure"], 4)}
                     for k, m in v.items()},
                  "ci_rule_minus_const": None if x["ci"].get("rule-const") is None else [round(c, 4) for c in x["ci"]["rule-const"]]}
    return {**base, "numbers": {"replay": win, "daily": {
        "window": d["window"], "mean_gross_rule": round(d["mean_gross_rule"], 4),
        **{k: {m: round(d[k][m], 4) for m in ("total", "cagr", "sharpe", "max_dd")}
           for k in ("rule", "const_exposure_matched", "equal_weight")},
        "ci_rule_minus_const": [round(c, 4) for c in d["ci_sharpe_rule_minus_const"]]}},
        "verdict": ("not claimed: the rule's Sharpe is above constant exposure with a CI excluding 0 in no window (daily 2021-26 {:.2f} vs "
                    "{:.2f}, same max drawdown {:.0%}); the caps add turnover without lowering drawdown".format(
                        d["rule"]["sharpe"], d["const_exposure_matched"]["sharpe"], d["rule"]["max_dd"]))}


def crisis_derisk() -> dict:
    m, h = _j("b2/metrics.json"), _j("b2/v1_holdout.json")
    base = {"id": "crisis_derisk", "title": "Crisis exit on BTC from free public data",
            "question": "Exit BTC to zero when crash, volatility, stablecoin-depeg or crisis-chatter rules fire; back after a cooldown. Better than a 200-day moving average?",
            "design": ("v0: four rules fixed from the literature, 7-day cooldown; v1: 32 configurations chosen on "
                       "2019-08..2022-12, the chosen one run once on 2023-01..2026-09; pass = beat buy & hold and the "
                       "200-day MA on drawdown and Calmar, with false alarms costing less than the drawdown avoided"),
            "window": ["2019-08-01", "2026-09-26"], "status": "estimated (Bitget BTCUSDT 1h bars, simulated fills)",
            "sources": ["studies/b2/backtest.py", "studies/b2/v1.py", "out/b2/metrics.json", "out/b2/v1_holdout.json", "out/b2/REPORT.md",
                        "studies/b2/trials.log"]}
    if m is None or h is None:
        return {**base, "status": "not shipped"}
    mm = m["metrics"]
    pick = lambda v: {k: round(v[k], 4) for k in ("total_return", "sharpe", "max_dd", "calmar") if v.get(k) is not None}  # noqa: E731
    return {**base, "numbers": {
        "v0": {k: pick(mm[k]) for k in ("buy_hold", "ma200", "voltarget", "module")},
        "v0_episodes": m["episodes"], "v0_false_alarms": m["false_alarms"], "v0_false_alarm_rate": round(m["false_alarm_rate"], 4),
        "v0_events_triggered": m["events_triggered"], "v0_events_covered": m["events_covered"],
        "v1_holdout": {"module": pick(h["holdout"]["module"]), **{k: pick(v) for k, v in h["holdout_refs"].items()}},
        "v1_grid_size": 32},
        "verdict": ("not claimed: v0 returned {:+.0%} against buy & hold {:+.0%} with {:.0%} false alarms; no v1 "
                    "configuration beat the 200-day MA in the design period and the chosen one returned {:+.0%} on the "
                    "holdout against {:+.0%}".format(mm["module"]["total_return"], mm["buy_hold"]["total_return"],
                                                     m["false_alarm_rate"], h["holdout"]["module"]["total_return"],
                                                     h["holdout_refs"]["buy_hold"]["total_return"]))}


def incident_exit() -> dict:
    inc, dl, at = _j("b2/incidents.json"), _j("b2/delistings.json"), _j("b2/alert_timing_summary.json")
    base = {"id": "incident_exit", "title": "Incident exit: leave a coin on a hack or a delisting",
            "question": "After a hack or a Binance delisting notice, does the token keep falling, and does X (Twitter) know before the price does?",
            "design": ("event studies on Bitget 1-minute candles: hacks >= $20M from DefiLlama with a >= 5% 60-minute "
                       "drop as the alert; Binance delisting announcements; then the first post of security and "
                       "news accounts on X against the price alert"),
            "window": ["2021-01-01", "2026-09-26"], "status": "observed (event studies on exchange candles; survivorship understates the falls)",
            "sources": ["studies/b2/incident_pilot.py", "studies/b2/delist_pilot.py", "studies/b2/alert_timing.py", "out/b2/incidents.json",
                        "out/b2/delistings.json", "out/b2/alert_timing_summary.json"]}
    if inc is None or dl is None or at is None:
        return {**base, "status": "not shipped"}
    i = [r for r in inc if not r.get("skip")]
    d = [r for r in dl if not r.get("skip")]
    hk = at["hack"]
    return {**base, "numbers": {
        "hacks_n": len(i), "hacks_median_24h": round(_med(i, "r_24h"), 4), "hacks_median_30d": round(_med(i, "r_30d"), 4),
        "delist_n": len(d), "delist_median_1h": round(_med(d, "r_1h"), 4),
        "delist_median_to_delist": round(_med(d, "r_to_delist"), 4),
        "x_hacks_with_post": hk["with_tweet"], "x_hacks_x_first": hk["earlier_than_ref"],
        "x_hacks_median_lead_min": round(hk["median_lead_min"], 1),
        "x_delist_median_lead_min": round(at["delist"]["median_lead_min"], 2)},
        "verdict": ("observed, not a strategy: tokens keep falling after both kinds of event, but X trailed the price "
                    "alert in {} of {} hacks (median {:.0f} minutes late), so the edge would be unattended execution, "
                    "not foresight; not built further".format(hk["with_tweet"] - hk["earlier_than_ref"], hk["with_tweet"],
                                                              -hk["median_lead_min"]))}


def _fact(label: str, value, kind: str) -> dict:
    """One headline figure for the site: kind = pct | num | share | ci | int."""
    return {"label": label, "value": value, "kind": kind}


def facts(x: dict) -> list[dict]:
    """Three or four headline figures per test, in the order the site shows them (values from `numbers`)."""
    n = x.get("numbers")
    if not n:
        return []
    k = x["id"]
    if k == "fg_cross_asset":
        return [_fact("asset-periods where the long-only rule beats buy & hold (CI excl. 0)", n["long_only_beats_buy_hold_ci"], "int"),
                _fact("asset-periods where long/short trails buy & hold", n["long_short_below_buy_hold"], "int"),
                _fact("2018-22 slopes with greed followed by higher returns", f"{n['p1_slopes_positive']} of {n['p1_slopes_total']}", "text"),
                _fact("MSTR short leg in greed, 2024", n["mstr_short_leg_2024"], "pct")]
    if k == "fg_mstr":
        h, p = n["history"], n["paper"]
        return [_fact("Sharpe, F&G rule (MSTR daily)", h["fg_long"]["sharpe"], "num"),
                _fact("Sharpe, buy & hold", h["buy_hold"]["sharpe"], "num"),
                _fact("rToken paper log, rule", p["return"], "pct"),
                _fact("rToken over the same hours", p["rtoken_return"], "pct")]
    if k == "vol_managed":
        d, lg = n["daily"], n["replay"].get("long") or {}
        return [_fact("Sharpe, vol rule (daily 2021-26)", d["rule"]["sharpe"], "num"),
                _fact("Sharpe, same exposure without the rule", d["const_exposure_matched"]["sharpe"], "num"),
                _fact("max drawdown, both", d["rule"]["max_dd"], "pct"),
                _fact("trades in 11 months, rule vs constant", f"{(lg.get('rule') or {}).get('trades')} vs {(lg.get('const') or {}).get('trades')}", "text")]
    if k == "crisis_derisk":
        return [_fact("false alarms (v0)", n["v0_false_alarm_rate"], "share"),
                _fact("holdout return, chosen v1", n["v1_holdout"]["module"]["total_return"], "pct"),
                _fact("holdout return, 200-day MA", n["v1_holdout"]["ma200"]["total_return"], "pct")]
    if k == "incident_exit":
        return [_fact("median token, 30 days after a hack alert", n["hacks_median_30d"], "pct"),
                _fact("median token, notice to delisting", n["delist_median_to_delist"], "pct"),
                _fact("hacks where X posted before the price alert", f"{n['x_hacks_x_first']} of {n['x_hacks_with_post']}", "text")]
    return []


def collect() -> dict:
    items = [fg_cross_asset(), fg_mstr(), vol_managed(), crisis_derisk(), incident_exit()]
    for x in items:
        x["facts"] = facts(x)
    return {"note": ("Run after the v1 test window, each against criteria written in its trials.log before the run; "
                     "none changes the v1 submission, which stays as pre-registered."),
            "items": {x["id"]: x for x in items},
            "n_items": len(items), "n_claimed": 0}
