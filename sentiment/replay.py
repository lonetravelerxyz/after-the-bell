"""Historical replay of the sentiment agent (contract: sentiment/CONTRACT.md, Runner; spec §2, §6).

  uv run python -m sentiment.replay --variant llm --start 2026-06-22 --end 2026-08-10 [--offline]
  -> sentiment/out/{variant}_{start}_{end}/log.jsonl, equity.csv, metrics.json

Clock: hourly from start 00:00 UTC to end + 1 day 00:00 UTC (dates inclusive). Every hour the
SimBroker marks the book (funding booked), the risk layer sees the equity and `risk.check` runs the
stop-loss and daily kill on that hourly mark; a binding rule closes the affected names at the next bar
open and writes a `risk_exit` record. At DECISION_HOURS the runner builds the point-in-time market
state, the evidence cards available in the last CARD_LOOKBACK_H hours, the baseline targets (always,
for every variant), the traded variant's decision (`baseline` = the rule policy itself), applies risk
and rebalances at the next bar open, writing a `decision` record. equity.csv holds the pre-trade mark
of every hour, so its first row is START_EQUITY and each trade's cost shows from the next hour on.

Evidence: cards for every item available in (start - lookback, end] are extracted once up front
through `evidence.cards_for` (cached, so each item costs one LLM call ever); a decision at t sees
only cards with available_at in (t - lookback, t]. An LLM endpoint failure in a decision stops the run
(nothing is cached for that prompt, so it could not be reproduced).

`shuffled` (the placebo of spec §6) shows every item's cards late, never early: at the true
availability time plus a lag drawn uniformly from SHUFFLE_LAG_D days, fixed per item by a seeded hash
(all cards of one item move together). The model then sees real but stale news, which carries no
timing edge, and it never sees an item before it existed. Items up to the maximum lag before the
window are extracted too, so the start of the window is not starved. The logged baseline targets of
a shuffled run are computed from the true (unshuffled) cards.

Risk version (config.RISK_VERSION, env SENTIMENT_RISK_VERSION, or `--risk-version`; read once per run).
v0 runs the code path of every run before risk v1 and writes records in exactly that format (no beta in
the state, no risk context, v0 prompt), so an offline v0 rerun reproduces the old record hashes. v1:

Risk context (agent prompt v1). Before the risk layer runs, `risk_context` builds the risk
context of the decision from the point-in-time market state, the Store (bars with close <= t only) and
the risk ledger state: per-name cap in force now (`risk.effective_caps(t, state)`) and the cap a held
same-side position keeps off-hours (`caps_held`), beta, cooldown,
entry price, P&L since entry and distance to the stop, book net and beta-weighted net, 24h returns, and
for every shown card the move of its tickers since the card's available_at. It is passed to the policy
as `risk_ctx=` when the policy accepts that keyword (agent.decide does, baseline.decide does not) and is
logged in every decision record (`risk_ctx`, `risk_ctx_hash`) with `risk_version` and `prompt_version`
(the policy's, None for the baseline); metrics.json carries both, plus how many decisions had a beta from
fewer than BETA_MIN_D daily returns (the 1.0 fallback) or fewer than BETA_LOOKBACK_D, and `code_stamp`
(git HEAD short sha at the start of the run, "-dirty" when the replay's code, prompts, snapshot or
quoted spreads differ from HEAD; see `code_stamp`).

Pinned card set (`--cards-from RUN_DIR`). News items are restricted to (a) items whose cards appear in
RUN_DIR/log.jsonl, i.e. whose item-id prefix (the part of a card id before the last "-") occurs in any
record's `evidence_ids`, `decision.evidence` or `risk_ctx.since_card`, plus (b) items whose extraction
is already cached: a row in cards.parquet under the current news extractor version, or an LLM cache
file for the item's extraction prompt (items that yielded no card are only in the LLM cache). Ratings
and insider items are deterministic and always kept. So a rerun sees the same news as the pinned run
wherever the old run could have seen it, never triggers a new extraction for an item the old run did
not have, and an item of (a) that is no longer cached still fails offline (CacheMiss) instead of being
dropped silently. metrics.json records `cards_from` and the number of news items dropped.

Everything in the log is a function of the inputs and the LLM cache (run_id is the run name, no
wall-clock values), so a rerun with --offline reproduces the same record hashes.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

from sentiment import agent, baseline, config, evidence, features, ledger, llm, metrics, risk
from sentiment.config import (CARD_LOOKBACK_H, DECISION_HOURS, MAX_W_NAME, OFF_HOURS_SCALE, OUT, START_EQUITY,
                              STOP_LOSS_NAME, UNIVERSE, perp)
from sentiment.sim import SimBroker

VARIANTS = ("baseline", *agent.VARIANTS)
MKT_REF = "SP500USDT"                                   # since_card move of market-scope cards
SINCE_MAX_GAP = pd.Timedelta(hours=6)                   # a price older than this at a time is not a price at it
SHUFFLE_SEED = 20260923
SHUFFLE_LAG_D = (7.0, 21.0)          # placebo lag range in days (uniform)
H = pd.Timedelta(hours=1)
MARK_KEYS = ("positions", "prices", "cash", "fees", "spread", "funding", "gross_equity")


def _day(s) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).normalize()


def _ts(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def input_hash(t, state: dict, card_ids: list[str]) -> str:
    body = ledger.canonical({"t": _ts(t).isoformat(), "state": state, "cards": sorted(card_ids)})
    return hashlib.sha256(body.encode()).hexdigest()


def window_items(store, start: pd.Timestamp, end_x: pd.Timestamp, extra_h: float = 0.0) -> list[dict]:
    lo = start - pd.Timedelta(hours=CARD_LOOKBACK_H + extra_h)
    return [it for it in store.all_items() if lo < _ts(it["available_at"]) <= end_x]


def shuffle_lag(item_id: str, seed: int = SHUFFLE_SEED) -> pd.Timedelta:
    """Placebo lag of one item: uniform in SHUFFLE_LAG_D days, from a seeded hash (whole seconds)."""
    u = int(hashlib.sha256(f"{seed}|{item_id}".encode()).hexdigest()[:12], 16) / 16 ** 12
    lo, hi = SHUFFLE_LAG_D
    return pd.Timedelta(seconds=round((lo + (hi - lo) * u) * 86400))


def shuffle_cards(cards: list[dict], seed: int = SHUFFLE_SEED) -> list[dict]:
    """Placebo: every card shown at its true availability + its item's lag (never earlier)."""
    out = [{**c, "available_at": _ts(c["available_at"]) + shuffle_lag(c["item_id"], seed),
            "true_available_at": _ts(c["available_at"])} for c in cards]
    assert all(c["available_at"] >= c["true_available_at"] for c in out)
    return sorted(out, key=lambda c: (c["available_at"], baseline.card_ref(c)))


def book_weights(mark: dict) -> dict[str, float]:
    eq = mark["equity"]
    return {tk: q * mark["prices"][tk] / eq for tk, q in mark["positions"].items()
            if tk in mark["prices"] and eq > 0}


def _mark_row(m: dict) -> dict:
    w = book_weights(m)
    return {"ts": m["ts"], "equity": m["equity"], "gross_equity": m["gross_equity"], "fees": m["fees"],
            "spread": m["spread"], "funding": m["funding"], "cash": m["cash"],
            "gross_exposure": sum(abs(v) for v in w.values()), "net_exposure": sum(w.values())}


def _live(cards: list[dict], t: pd.Timestamp) -> list[dict]:
    lo = t - pd.Timedelta(hours=CARD_LOOKBACK_H)
    return [c for c in cards if lo < _ts(c["available_at"]) <= t]


# ---------------------------------------------------------------- risk context (agent prompt v1)

def risk_version() -> str | None:
    """config.RISK_VERSION, read at call time like risk._version (None for a config without it)."""
    return getattr(config, "RISK_VERSION", None)


def _f(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _close_at(store, sym: str, t: pd.Timestamp) -> float | None:
    """Close of the last bar closed by t (Store.bars: close time <= t); None if none or older than SINCE_MAX_GAP."""
    b = store.bars(sym, t, 1)
    if not len(b) or b.index[-1] + pd.Timedelta(hours=1) < t - SINCE_MAX_GAP:
        return None
    return _f(b["close"].iloc[-1])


def _caps(t: pd.Timestamp, state: dict, version: str | None = None) -> tuple[dict[str, float], str]:
    fn = getattr(risk, "effective_caps", None)
    if callable(fn):
        raw = fn(t, state, version=version)
        caps = {tk: _f(v.get("cap") if isinstance(v, dict) else v) for tk, v in raw.items() if tk in UNIVERSE}
        return {tk: round(v, 4) for tk, v in caps.items() if v is not None}, "risk.effective_caps"
    # TODO(reviewer): fallback until risk.effective_caps(t, state) exists: config cap, off-hours scale only
    session = next((state[tk].get("us_session_open") for tk in UNIVERSE if tk in state), None)
    session = risk.us_session_open(t) if session is None else bool(session)
    cap = MAX_W_NAME * (1.0 if session else OFF_HOURS_SCALE)
    return {tk: round(cap, 4) for tk in UNIVERSE}, "config (fallback: no risk.effective_caps)"


def risk_context(t, *, store, state: dict, book: dict[str, float], lstate: dict, cards: list[dict],
                 version: str | None = None) -> dict:
    """The risk context of a decision at t (JSON-safe; see module docstring). Call it BEFORE risk.apply,
    which mutates `lstate`. `cards` are the cards the policy is given; only those the prompt shows
    (agent.select_cards) get a since_card entry. Point-in-time: prices are closes of bars closed by t
    (or by the card's available_at); betas = the market state's `beta_60d` (the ones the risk layer
    neutralises with; missing -> 1.0 as in risk v1). `version` = risk version (default config)."""
    t = _ts(t)
    ver = version or risk_version()
    caps, caps_src = _caps(t, state, ver)
    session = risk._session(t, state or {}, sorted(UNIVERSE))
    # off-hours a held same-side position is not cut to `caps` (new exposure) but kept up to the full per-name cap
    full = risk.vol_caps(state or {}, UNIVERSE) if ver == "v1" else {tk: MAX_W_NAME for tk in UNIVERSE}
    held = caps if session else {tk: round(full[tk], 4) for tk in caps}
    beta = {tk: _f((state.get(tk) or {}).get("beta_60d")) for tk in UNIVERSE}
    beta = {tk: 1.0 if b is None else round(b, 4) for tk, b in beta.items()}   # risk v1: missing beta -> 1.0
    w = {tk: float(v) for tk, v in book.items() if abs(v) > 1e-12}
    bnet = round(sum(v * beta.get(tk, 1.0) for tk, v in w.items()), 6)
    cooldown = {tk: _ts(u).isoformat() for tk, u in sorted((lstate.get("cooldown") or {}).items())
                if u is not None and _ts(u) > t}
    ku = lstate.get("kill_until")
    entry = {}
    for tk, e in sorted((lstate.get("entry") or {}).items()):
        side, px0 = int(e.get("side") or 0), _f(e.get("price"))
        if tk not in w or not px0 or (w[tk] > 0) != (side > 0):
            continue
        px = _f((state.get(tk) or {}).get("price"))
        stop = STOP_LOSS_NAME if ver == "v0" else (_f(e.get("stop")) or STOP_LOSS_NAME)  # as risk._stop_of
        pnl = None if px is None else side * (px / px0 - 1)
        entry[tk] = {"price": px0, "side": side, "stop": stop,
                     "pnl": None if pnl is None else round(pnl, 6),
                     "to_stop": None if pnl is None else round(pnl + stop, 6)}
    since: dict[str, dict[str, float | None]] = {}
    ref_now = _close_at(store, MKT_REF, t)
    for c in agent.select_cards(t, cards):
        at = _ts(c["available_at"])
        syms = [(tk, perp(tk)) for tk in c.get("tickers") or [] if tk in UNIVERSE]
        if not syms and c.get("scope") == "market":
            syms = [("_market", MKT_REF)]
        mv = {}
        for key, sym in syms:
            p0 = _close_at(store, sym, at)
            p1 = ref_now if sym == MKT_REF else _close_at(store, sym, t)
            mv[key] = round(p1 / p0 - 1, 6) if p0 and p1 else None
        since[agent._card_id(c)] = mv
    return {"t": t.isoformat(), "risk_version": ver, "caps": caps, "caps_source": caps_src,
            "us_session_open": session, "caps_held": held,
            "beta": beta, "book": {tk: round(v, 6) for tk, v in sorted(w.items())},
            "book_net": round(sum(w.values()), 6), "book_beta_net": bnet,
            "ret_24h": {tk: _f((state.get(tk) or {}).get("ret_24h")) for tk in UNIVERSE},
            "cooldown_until": cooldown, "kill_until": _ts(ku).isoformat() if ku and _ts(ku) > t else None,
            "entry": entry, "since_card": since}


@functools.lru_cache(maxsize=None)
def _accepts_risk_ctx(policy) -> bool:
    try:
        ps = inspect.signature(policy).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "risk_ctx" or p.kind is p.VAR_KEYWORD for p in ps)


def decision_step(t, *, store, broker, m: dict, lstate: dict, shown_cards: list[dict], true_cards: list[dict],
                  variant: str, policy, run_id: str, mode: str, version: str | None = None) -> tuple[dict, dict]:
    """One decision at t (shared by replay and the live loop): state, baseline, risk context, policy, risk,
    rebalance. `m` is the pre-trade mark at t. Returns (log record, post-trade mark). A policy endpoint
    failure is returned in the record (decision.valid False, `error` set); the caller decides whether to stop.
    `version` = risk version (default config.RISK_VERSION); v0 builds no risk context and writes the v0
    record format (see module docstring)."""
    ver = version or risk_version()
    state = features.market_state(store, t, version=ver)
    book = book_weights(m)
    base = baseline.decide(t, state, true_cards, book)
    lstate["equity"] = m["equity"]
    ctx = None
    if ver != "v0":
        ctx = risk_context(t, store=store, state=state, book=book, lstate=lstate,
                           cards=[] if variant == "nonews" else shown_cards, version=ver)
    if variant == "baseline":
        dec = base
    elif ctx is not None and _accepts_risk_ctx(policy):
        dec = policy(t, state, shown_cards, book, variant, risk_ctx=ctx)
    else:
        dec = policy(t, state, shown_cards, book, variant)
    targets, actions = risk.apply(t, dec, book, state, lstate, version=ver)
    fills = broker.rebalance(t, targets, m["equity"])
    m2 = broker.mark(t)
    ids = [baseline.card_ref(c) for c in shown_cards]
    rec = {
        "ts": t, "mode": mode, "run_id": run_id, "variant": variant, "event": "decision",
        "input_hash": input_hash(t, state, ids),
        "evidence_ids": dec.get("shown", ids), "llm_request_hash": dec.get("llm_request_hash"),
        "decision": {k: dec[k] for k in ("targets", "reasons", "evidence", "confidence", "raw", "valid")},
        "error": dec.get("error"),
        "baseline_targets": base["targets"], "risk_actions": actions, "targets": targets,
        "fills": fills, "skipped": list(getattr(broker, "skipped", [])), "state": state,
        "marks": {k: m2[k] for k in MARK_KEYS}, "equity": m2["equity"]}
    if ver != "v0":
        rec.update({"risk_ctx": ctx, "risk_ctx_hash": hashlib.sha256(ledger.canonical(ctx).encode()).hexdigest(),
                    "risk_version": ver, "prompt_version": None if variant == "baseline" else dec.get("prompt_version")})
    return rec, m2


def risk_exit_step(t, *, broker, m: dict, lstate: dict, variant: str, run_id: str, mode: str,
                   version: str | None = None) -> tuple[dict | None, dict]:
    """Hourly stop-loss / daily kill between decisions. Returns (log record or None, mark after)."""
    ver = version or risk_version()
    forced, actions = risk.check(t, book_weights(m), m["prices"], lstate, version=ver)
    if forced is None:
        return None, m
    names = None if any(a["rule"] in ("daily_kill", "kill") for a in actions) else \
        {a["ticker"] for a in actions if a["ticker"]}
    fills = broker.rebalance(t, forced, m["equity"], only=names)
    m2 = broker.mark(t)
    rec = {"ts": t, "mode": mode, "run_id": run_id, "variant": variant, "event": "risk_exit",
           "input_hash": None, "evidence_ids": [], "llm_request_hash": None, "decision": None, "error": None,
           "baseline_targets": None, "risk_actions": actions, "targets": forced, "fills": fills,
           "skipped": list(getattr(broker, "skipped", [])), "state": None,
           "marks": {k: m2[k] for k in MARK_KEYS}, "equity": m2["equity"]}
    if ver != "v0":
        rec.update({"risk_ctx": None, "risk_ctx_hash": None, "risk_version": ver,
                    "prompt_version": None if variant == "baseline" else agent.PROMPT_VERSION})
    return rec, m2


def beta_history(log_path) -> dict[str, int | None]:
    """Decisions whose state has a beta estimated from fewer than BETA_MIN_D daily returns for some name (the
    1.0 fallback: the beta step is then dollar neutralisation for those names) or fewer than BETA_LOOKBACK_D
    (a shorter history than intended). None for a log without beta_days (risk v0)."""
    n_fb = n_short = n = 0
    for ln in Path(log_path).read_text().splitlines():
        r = json.loads(ln)
        st = r.get("state") or {}
        days = [(st.get(tk) or {}).get("beta_days") for tk in UNIVERSE]
        if r.get("event") != "decision" or any(d is None for d in days):
            continue
        n += 1
        n_fb += min(days) < config.BETA_MIN_D
        n_short += min(days) < config.BETA_LOOKBACK_D
    return {"n_decisions_beta_fallback": n_fb if n else None, "n_decisions_beta_short": n_short if n else None}


# ---------------------------------------------------------------- pinned card set (--cards-from)

def _item_prefix(card_id: str) -> str:
    card_id = str(card_id)
    return card_id.rsplit("-", 1)[0] if "-" in card_id else card_id[:10]


CODE_PATHS = (":(glob)sentiment/*.py", "sentiment/prompts", "snapshot", "static/quoted_spread.csv")


def code_stamp() -> str | None:
    """git HEAD short sha, plus "-dirty" when the replay's inputs under git (sentiment/*.py, the prompts, the
    snapshot, the quoted spreads) differ from HEAD or are untracked; None outside a git checkout."""
    import subprocess
    try:
        git = lambda *a: subprocess.run(["git", *a], cwd=config.ROOT, capture_output=True, text=True,  # noqa: E731
                                        check=True, timeout=30).stdout.strip()
        sha = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--", *CODE_PATHS)
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha or None


def logged_item_prefixes(run_dir) -> set[str]:
    """Item-id prefixes of every card a run logged: evidence_ids, decision.evidence, risk_ctx.since_card."""
    out: set[str] = set()
    for ln in (Path(run_dir) / "log.jsonl").read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        ids = list(r.get("evidence_ids") or [])
        for v in ((r.get("decision") or {}).get("evidence") or {}).values():
            ids += v if isinstance(v, list) else [v]
        ids += list(((r.get("risk_ctx") or {}).get("since_card") or {}).keys())
        out |= {_item_prefix(i) for i in ids if i}
    return out


def extraction_cached(items: list[dict]) -> set[str]:
    """Ids of news items whose extraction is cached: a cards.parquet row under the current news extractor,
    or an LLM cache file for the extraction prompt (needed for items that yielded no card). No LLM calls."""
    d = evidence.load_cards()
    ver = evidence.news_version()
    have = set(d.loc[d["extractor"] == ver, "item_id"]) if len(d) and "extractor" in d else set()
    return {it["id"] for it in items if it["kind"] == "news"
            and (it["id"] in have or llm.cached(evidence.news_messages(it)) is not None)}


def pin_items(items: list[dict], run_dir) -> tuple[list[dict], list[dict]]:
    """(kept, dropped): news items restricted to the pinned run's logged items + cached extractions;
    ratings and insider items always kept (see module docstring)."""
    logged = logged_item_prefixes(run_dir)
    cached = extraction_cached(items)
    kept, dropped = [], []
    for it in items:
        ok = it["kind"] != "news" or it["id"] in cached or it["id"][:10] in logged
        (kept if ok else dropped).append(it)
    return kept, dropped


def run(variant: str, start, end, *, store=None, out_dir=None, cards_fn=None, decide_fn=None,
        quiet: bool = False, cards_from=None) -> Path:
    """Replay [start, end] (UTC dates, inclusive); returns the run directory. `cards_from` = a previous run
    directory whose card set is pinned (see module docstring)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; one of {VARIANTS}")
    if store is None:
        from sentiment.data import Store
        store = Store()
    start, end_x = _day(start), _day(end) + pd.Timedelta(days=1)
    run_id = f"{variant}_{start:%Y-%m-%d}_{end_x - pd.Timedelta(days=1):%Y-%m-%d}"
    out = Path(out_dir) if out_dir else OUT / run_id
    pinned = None if cards_from is None else config.run_dir(cards_from)
    if pinned is not None and pinned.resolve() == out.resolve():
        raise ValueError("--cards-from must name another run directory (this one is cleared first)")
    out.mkdir(parents=True, exist_ok=True)
    for f in ("log.jsonl", "equity.csv", "metrics.json"):
        (out / f).unlink(missing_ok=True)

    extra_h = SHUFFLE_LAG_D[1] * 24 if variant == "shuffled" else 0.0
    items = window_items(store, start, end_x, extra_h)
    dropped: list[dict] = []
    if pinned is not None:
        items, dropped = pin_items(items, pinned)
    true_cards = (cards_fn or evidence.cards_for)(items)
    shown_all = shuffle_cards(true_cards) if variant == "shuffled" else true_cards
    policy = decide_fn or (baseline.decide if variant == "baseline" else agent.decide)
    if not quiet:
        print(f"replay {run_id}: {len(items)} items -> {len(true_cards)} cards"
              + (f" (pinned to {cards_from}: {len(dropped)} news items dropped)" if cards_from else ""),
              file=sys.stderr)

    ver = risk_version()
    risk._version(ver)                                    # fail early on an unknown version
    stamp = code_stamp()                                  # the code this run starts with
    log = ledger.Ledger(out / "log.jsonl")
    broker = SimBroker(store, START_EQUITY)
    lstate: dict = {}
    rows = []
    for t in pd.date_range(start, end_x, freq="h"):
        m = broker.mark(t)
        rows.append(_mark_row(m))                         # pre-trade mark of the hour
        risk.observe(lstate, t, m["equity"])
        if t >= end_x:
            continue
        if t.hour in DECISION_HOURS:
            rec, m = decision_step(t, store=store, broker=broker, m=m, lstate=lstate,
                                   shown_cards=_live(shown_all, t), true_cards=_live(true_cards, t),
                                   variant=variant, policy=policy, run_id=run_id, mode="replay", version=ver)
            if rec["error"]:
                raise llm.LLMError(f"{t}: decision failed ({rec['error']}); replay stopped: an uncached "
                                   "failure cannot be reproduced. Rerun to retry.")
        else:
            rec, m = risk_exit_step(t, broker=broker, m=m, lstate=lstate, variant=variant, run_id=run_id,
                                    mode="replay", version=ver)
            if rec is None:
                continue
        log.append(rec)
        if not quiet:
            bound = sorted({a["rule"] for a in rec["risk_actions"]})
            valid = (rec["decision"] or {}).get("valid", True)
            print(f"{t:%m-%d %H:%M} {rec['event'][:8]:8} eq {m['equity']:9.2f} "
                  f"tg {json.dumps({k: round(v, 3) for k, v in rec['targets'].items()})} "
                  f"fills {len(rec['fills'])}{' risk ' + ','.join(bound) if bound else ''}"
                  f"{'' if valid else ' INVALID'}", file=sys.stderr)

    pd.DataFrame(rows).to_csv(out / "equity.csv", index=False)
    res = metrics.compute(out)
    res.update({"run_id": run_id, "variant": variant, "code_stamp": stamp, "n_items": len(items), "n_cards": len(true_cards),
                "prompt_version": None if variant == "baseline" else ("v0" if ver == "v0" else agent.PROMPT_VERSION),
                "risk_version": ver, **beta_history(out / "log.jsonl"),
                "cards_from": None if cards_from is None else str(cards_from),
                "n_items_dropped_by_pin": len(dropped),
                "log_last_hash": log.last_hash, "log_verified": ledger.verify(out / "log.jsonl")})
    (out / "metrics.json").write_text(json.dumps(res, indent=2) + "\n")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", default="llm", choices=VARIANTS)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--offline", action="store_true", help="LLM cache only (SENTIMENT_OFFLINE=1)")
    p.add_argument("--raw", action="store_true", help="read data/raw instead of the committed snapshot")
    p.add_argument("--risk-version", default=None, choices=risk.VERSIONS,
                   help="risk layer + prompt version (default config.RISK_VERSION; v0 reproduces runs before v1)")
    p.add_argument("--out", default=None, help="run directory (default out/{variant}_{start}_{end})")
    p.add_argument("--cards-from", default=None, metavar="RUN_DIR",
                   help="pin the news set: only items logged by RUN_DIR plus items whose extraction is cached")
    a = p.parse_args(argv)
    if a.offline:
        os.environ["SENTIMENT_OFFLINE"] = "1"
    if a.risk_version:
        config.RISK_VERSION = a.risk_version
    from sentiment.data import Store
    out = run(a.variant, a.start, a.end, store=Store(snapshot=not a.raw), out_dir=a.out, cards_from=a.cards_from)
    res = json.loads((out / "metrics.json").read_text())
    keep = ("run_id", "code_stamp", "prompt_version", "risk_version", "n_decisions_beta_fallback", "n_decisions_beta_short", "cards_from", "n_items_dropped_by_pin", "n_decisions",
            "n_invalid", "n_risk_exits", "n_cards", "total_return", "sharpe", "max_dd",
            "win_rate", "round_trips", "trades", "turnover_ann", "costs", "log_verified", "log_last_hash")
    print(json.dumps({k: res.get(k) for k in keep}, indent=1))
    if not a.offline:
        print(json.dumps(llm.usage_summary(), indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
