"""B5 vol-managed long book: replay of the 10 v1 names with volatility caps (spec: studies/b5/trials.log, entry 1).

  uv run python -m studies.b5.run --variant rule --start 2026-06-22 --end 2026-08-10 [--raw] [--const-w 0.08] [--offline]
  -> out/b5/{variant}_{start}_{end}/log.jsonl, equity.csv, metrics.json

cap_i(t) = FULL_W * min(1, vt_i / vol7_i): vol7 = the v1 market state's vol_7d at t; vt_i = median of vol7_i
sampled at each 00:00 UTC over the trailing VT_DAYS days (point-in-time, bars closed by the sample time),
at least VT_MIN samples, else the cap is FULL_W; no vol7 at t -> cap 0. Risk v2 clips every target to
[0, cap] and nothing else. Variants: rule (target = cap), llm (Qwen picks weights in [0, cap]; an invalid
reply holds the book clipped to the caps), const (every name at --const-w), buyhold (every name at FULL_W).
Execution is v1's SimBroker unchanged; decisions at config.DECISION_HOURS. The log is hash-chained and,
like v1, a function of the inputs and the LLM cache, so an --offline rerun reproduces it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment import agent, features, ledger, llm, metrics
from sentiment.config import DECISION_HOURS, OUT, START_EQUITY, UNIVERSE, perp
from sentiment.replay import MARK_KEYS, _day, _mark_row, book_weights, input_hash
from sentiment.sim import SimBroker

FULL_W = 0.10
VT_DAYS, VT_MIN = 60, 20
VARIANTS = ("rule", "llm", "const", "buyhold")
PROMPT = Path(__file__).resolve().parent / "prompts" / "decide.md"
PROMPT_VERSION = "b5-v1"
B5_OUT = OUT / "b5"


# ---------------------------------------------------------------- vol target

def vol7_daily(store, sym: str) -> pd.Series:
    """vol_7d of `sym` at each 00:00 UTC (hourly log returns of bars closed by then, last 7 days, x sqrt 24)."""
    d = store._perp.get(sym)
    if d is None or not len(d):
        return pd.Series(dtype=float)
    c = pd.Series(d["close"].to_numpy(float), index=d.index + pd.Timedelta(hours=1))   # close time
    lr = np.log(c).diff().dropna()
    days = pd.date_range(c.index[0].ceil("D"), c.index[-1].floor("D"), freq="D")
    out = {}
    for t in days:
        w = lr[(lr.index > t - pd.Timedelta(days=7)) & (lr.index <= t)]
        if len(w) > 1:
            out[t] = float(w.std(ddof=1) * math.sqrt(24))
    return pd.Series(out)


class VolTarget:
    def __init__(self, store):
        self.v = {tk: vol7_daily(store, perp(tk)) for tk in UNIVERSE}

    def at(self, tk: str, t: pd.Timestamp) -> tuple[float | None, int]:
        s = self.v[tk]
        s = s[(s.index <= t) & (s.index > t - pd.Timedelta(days=VT_DAYS))]
        return (float(s.median()) if len(s) >= VT_MIN else None), len(s)


def caps_at(t, state: dict, vt: VolTarget) -> tuple[dict[str, float], dict[str, float | None], dict[str, int]]:
    caps, tgt, n = {}, {}, {}
    for tk in UNIVERSE:
        v = (state.get(tk) or {}).get("vol_7d")
        tgt[tk], n[tk] = vt.at(tk, t)
        if v is None or not v > 0:
            caps[tk] = 0.0
        elif tgt[tk] is None:
            caps[tk] = FULL_W
        else:
            caps[tk] = round(FULL_W * min(1.0, tgt[tk] / v), 6)
    return caps, tgt, n


# ---------------------------------------------------------------- LLM policy

def system_prompt() -> str:
    return (PROMPT.read_text().replace("{{TICKERS}}", ", ".join(UNIVERSE))
            .replace("{{FULL_W}}", f"{FULL_W:.2f}"))


def user_message(t, state: dict, caps: dict, tgt: dict, book: dict) -> str:
    p, bp = agent._pct, agent._bp
    m = state.get("_market") or {}
    session = next((state[tk].get("us_session_open") for tk in UNIVERSE if tk in state), None)
    fng = m.get("fng")
    lines = [f"NOW: {t:%Y-%m-%d %H:%M} UTC ({t:%A})", f"US regular session: {'open' if session else 'closed'}",
             f"MARKET: crypto fear&greed {fng if fng is not None else 'na'} ({m.get('fng_regime') or 'na'}); "
             f"SP500 perp 24h {p(m.get('sp500_ret_24h'))}%; NDX100 perp 24h {p(m.get('ndx_ret_24h'))}%", "",
             "NAMES (ret/vol in %, funding/premium in bp, cap/book = weight)",
             "ticker ret_4h ret_24h ret_7d vol_7d vol_target cap funding funding_7d premium book"]
    for tk in UNIVERSE:
        s = state.get(tk) or {}
        lines.append(" ".join([tk, p(s.get("ret_4h")), p(s.get("ret_24h")), p(s.get("ret_7d")),
                               p(s.get("vol_7d")).lstrip("+"), p(tgt.get(tk)).lstrip("+"), f"{caps[tk]:.3f}",
                               bp(s.get("funding_last"), 2), bp(s.get("funding_7d_mean"), 2),
                               bp(s.get("spot_premium")), f"{book.get(tk, 0.0):.3f}"]))
    held = [f"{tk} {book[tk]:.3f}" for tk in UNIVERSE if abs(book.get(tk, 0.0)) > 1e-9]
    lines += ["", "CURRENT BOOK: " + (", ".join(held) if held else "flat"), "",
              "Return the JSON decision for every ticker."]
    return "\n".join(lines)


def llm_decide(t, state, caps, tgt, book) -> dict:
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_message(t, state, caps, tgt, book)}]
    req = llm.cache_key(llm.model_name(), messages)
    try:
        text = llm.complete(messages, purpose="decide:b5")
    except llm.LLMError as e:
        d = agent.parse_decision("", [])
        d.update(raw=f"LLMError: {e}", error=str(e))
    else:
        d = agent.parse_decision(text, [])
    return {**d, "llm_request_hash": req, "prompt_version": PROMPT_VERSION}


# ---------------------------------------------------------------- risk v2

def risk_v2(dec: dict, book: dict, caps: dict) -> tuple[dict[str, float], list[dict]]:
    want = dec["targets"] if dec.get("valid", True) else {tk: book.get(tk, 0.0) for tk in UNIVERSE}
    out, actions = {}, []
    if not dec.get("valid", True):
        actions.append({"rule": "invalid_hold", "ticker": None})
    for tk in UNIVERSE:
        w = float(want.get(tk, 0.0))
        c = min(max(w, 0.0), caps[tk])
        if abs(c - w) > 1e-9:
            actions.append({"rule": "vol_cap" if w > caps[tk] else "long_only", "ticker": tk,
                            "requested": round(w, 6), "applied": round(c, 6)})
        if c > 0:
            out[tk] = c
    return out, actions


# ---------------------------------------------------------------- runner

def run(variant: str, start, end, *, store, const_w: float | None = None, out_dir=None, quiet=False) -> Path:
    start, end_x = _day(start), _day(end) + pd.Timedelta(days=1)
    run_id = f"{variant}_{start:%Y-%m-%d}_{end_x - pd.Timedelta(days=1):%Y-%m-%d}"
    out = Path(out_dir) if out_dir else B5_OUT / run_id
    out.mkdir(parents=True, exist_ok=True)
    for f in ("log.jsonl", "equity.csv", "metrics.json"):
        (out / f).unlink(missing_ok=True)
    if variant == "const" and const_w is None:
        raise ValueError("const needs --const-w")
    vt = VolTarget(store)
    log = ledger.Ledger(out / "log.jsonl")
    broker = SimBroker(store, START_EQUITY)
    rows, gross = [], []
    for t in pd.date_range(start, end_x, freq="h"):
        m = broker.mark(t)
        rows.append(_mark_row(m))
        if t >= end_x or t.hour not in DECISION_HOURS:
            continue
        state = features.market_state(store, t, version="v0")      # no betas: B5 does not use them
        book = book_weights(m)
        caps, tgt, n_vt = caps_at(t, state, vt)
        if variant == "rule":
            dec = {"targets": dict(caps), "valid": True}
        elif variant == "const":
            dec = {"targets": {tk: const_w for tk in UNIVERSE}, "valid": True}
        elif variant == "buyhold":
            dec = {"targets": {tk: FULL_W for tk in UNIVERSE}, "valid": True}
        else:
            dec = llm_decide(t, state, caps, tgt, book)
        if variant in ("const", "buyhold"):                       # no vol cap: only long-only, [0, FULL_W]
            targets = {tk: min(max(float(w), 0.0), FULL_W) for tk, w in dec["targets"].items() if w > 0}
            actions = []
        else:
            targets, actions = risk_v2(dec, book, caps)
        fills = broker.rebalance(t, targets, m["equity"])
        m2 = broker.mark(t)
        rec = {"ts": t, "mode": "replay", "run_id": run_id, "variant": variant, "event": "decision",
               "input_hash": input_hash(t, state, []), "evidence_ids": [],
               "llm_request_hash": dec.get("llm_request_hash"),
               "decision": {k: dec.get(k) for k in ("targets", "reasons", "confidence", "raw", "valid")},
               "error": dec.get("error"), "caps": caps, "vol_target": tgt, "vol_target_n": n_vt,
               "risk_actions": actions, "targets": targets, "fills": fills,
               "skipped": list(broker.skipped), "state": state,
               "marks": {k: m2[k] for k in MARK_KEYS}, "equity": m2["equity"],
               "risk_version": "b5-v2", "prompt_version": dec.get("prompt_version")}
        if rec["error"]:
            raise llm.LLMError(f"{t}: decision failed ({rec['error']}); rerun to retry (replies are cached)")
        log.append(rec)
        gross.append(sum(targets.values()))
        if not quiet and t.hour == 0:
            print(f"{t:%Y-%m-%d} eq {m2['equity']:9.2f} gross {sum(targets.values()):.2f} fills {len(fills)}",
                  file=sys.stderr)
    pd.DataFrame(rows).to_csv(out / "equity.csv", index=False)
    res = metrics.compute(out)
    eq = pd.read_csv(out / "equity.csv")
    res.update({"run_id": run_id, "variant": variant, "const_w": const_w, "store": store.source,
                "mean_target_gross": float(np.mean(gross)) if gross else None,
                "mean_gross_exposure": float(eq["gross_exposure"].mean()),
                "prompt_version": PROMPT_VERSION if variant == "llm" else None, "risk_version": "b5-v2",
                "log_last_hash": log.last_hash, "log_verified": ledger.verify(out / "log.jsonl")})
    (out / "metrics.json").write_text(json.dumps(res, indent=2) + "\n")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--raw", action="store_true", help="data/raw instead of the committed snapshot")
    p.add_argument("--const-w", type=float, default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.offline:
        os.environ["SENTIMENT_OFFLINE"] = "1"
    from sentiment.data import Store
    out = run(a.variant, a.start, a.end, store=Store(snapshot=not a.raw), const_w=a.const_w, out_dir=a.out)
    res = json.loads((out / "metrics.json").read_text())
    keep = ("run_id", "store", "total_return", "sharpe", "max_dd", "trades", "turnover_ann", "costs",
            "mean_target_gross", "mean_gross_exposure", "n_invalid", "log_verified", "log_last_hash")
    print(json.dumps({k: res.get(k) for k in keep}, indent=1))


if __name__ == "__main__":
    main()
