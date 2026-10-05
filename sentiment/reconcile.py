"""Replay-vs-live reconciliation (spec §2 "回放与实盘对账"; contract: Runner, Log, Execution).

  uv run python -m sentiment.reconcile [--live-dir out/live] [--out DIR] [--snapshot]
  uv run python -m sentiment.reconcile --self-check [RUN_DIR] [--news-offset-h 0]   (replay log relabelled
      live in a temp dir; --news-offset-h 0 reproduces runs made before news became available at
      published_at - 8h, e.g. the 2026-07-01..03 smoke runs)
  -> <out, default live-dir>/reconcile.csv (one row per order) and reconcile.json (per record + summary)

For every record of a live log (mode "live") at time t this recomputes what the replay code path
decides at the same t over the same Store and the same LLM cache, then runs the replay fill model on
the live targets. Nothing here calls the network: SENTIMENT_OFFLINE=1 is forced and LLM replies are
read from the cache only.

Positions. Records are grouped by broker mode (`live_{variant}_{mode}` run_id, else the orders.jsonl
lines of their fills): the live loop keeps one book per mode ({mode}_state.json, or the demo account),
so a shadow -> demo switch in one directory does not carry the shadow book into the demo run. The
first record of a mode starts from the positions the record itself implies (post-trade positions
minus its signed fills; a record's `pre_trade` {positions, prices, equity} is used when logged), not
from flat; later records carry the previous record's post-trade positions of the same mode, and a
record whose implied pre-trade positions differ from the carried ones is listed (`positions_break`).
The risk state is one walk over all records, as live.step keeps one runner_state.json for all modes.

Decision (decision records). state = features.market_state(store, t); cards = the cards of the items
available in (t - CARD_LOOKBACK_H, t], read from cache/cards.parquet (rating/insider cards
rebuilt in memory, news cards whose cached extraction is missing from the table rebuilt from the
cached reply; nothing is extracted or written); book = the live pre-trade book: the logged
`pre_trade` book when the record has one, else the pre-trade positions at the replay mark prices
(last closed bar <= t) over the pre-trade equity of the live equity.csv. The prompt is rebuilt with
`agent.build_messages` and hashed; `agent.decide` is run only when that prompt is cached. A hash
different from the record's `llm_request_hash` is a mismatch, and the record says which inputs
differ: state fields (late bar / funding / F&G top-ups), cards only one side saw (with the item's
availability, published_at - 8h and first_seen), the book, the system prompt or the model id, plus a
line diff against the live prompt when that is in the cache. The risk state (entry prices,
cooldowns, daily kill) is rebuilt by walking the live equity.csv and log in order with the live
decisions, so the replay decision is risk-filtered against the state the live agent had.

Versions. Each record is reconciled under its own risk version (`risk_version`; absent = v0, the records
written before risk v1 and by a loop still running that code): the state is rebuilt with beta_60d only
under v1, the prompt is the v0 prompt (decide.md, no risk context) or the v1 prompt, and every
risk.apply / risk.check of the walk and of the replay decision uses that version (hours between records:
the version of the latest record), so a v0 record is never judged by v1 caps, stops or beta step.

Risk context (v1 records). The context the v1 prompt shows (RISK table: caps, betas, cooldowns, kill,
P&L since entry and distance to the stop, book net and beta net; since_bp of every shown card) is NOT
taken from the record: it is rebuilt point-in-time with `replay.risk_context` from the reconciliation
Store, the rebuilt state, the reference book and a copy of the rebuilt risk state, and the replay prompt
is built from the rebuilt context. A logged context that differs (`risk_ctx_diffs`, numbers equal within
CTX_REL_TOL) is a 'risk_ctx' mismatch input, so a live loop that showed the model a context from leaky or
different bars, or a wrong cap / cooldown / kill, fails prompt and decision identity.

The book an LLM prompt showed is read back from the cached live request and checked against the
reference book (`book_check`): exactly at 3 decimals against a logged pre-trade book, else within
BOOK_TOL of the book at replay marks (live marks are venue tickers taken after the bar close, so the
two differ by a few bp). A shown book outside that is a "book" mismatch input and the replay prompt
is then built from the reference book, so a wrong live book shows in prompt and decision identity.
Inside it the replay prompt uses the shown book, so identity re-checks state, cards, system prompt
and model given the book. Identical = same prompt hash (the rule inputs for `baseline`) and the same
decision targets (1e-9); a live decision whose LLM call failed (record `error`, or its prompt not in
the cache) is unverifiable (None), not a mismatch. Target L1 = sum |live - replay| of the post-risk
targets. The risk layer reads the book (off-hours clamp to |b|, funding block, invalid -> hold), so
without a logged pre-trade book post-risk targets are compared twice: exactly (`targets_identical`,
`risk_rebuild_exact`) and within the L1 a mark gap can explain (`targets_match`, `risk_rebuild_ok`:
2 x sum |b| x gap per held name, gap = |live mark / replay mark - 1| + MARK_GAP_SLACK from the
record's own marks or orders.jsonl quotes, else MARK_GAP_DEFAULT; no tolerance at a record whose
positions break or whose shown book fails `book_check`); `n_targets_book_driven` counts the
differences explained that way.

Hourly exits. At every hour between decisions `risk.check` runs on the replay book (live positions at
replay marks) with a copy of the rebuilt risk state, and its rules are compared with the live
risk_exit record at that hour, or with its absence (`exits`: live-only, replay-only, rule differences,
and each disputed stop's loss at the replay mark).

Fills (decision and risk-exit records). A SimBroker holding the live pre-trade positions rebalances to
the live targets at t (next 1h bar open +- the quoted half-spread; a risk exit trades only its names).
Fills are netted per (ticker, side) and paired: gap_bp = s * (live price / sim price - 1) * 1e4 with
s = +1 for buys and -1 for sells, so positive = live paid worse. mid_gap_bp does the same with the
live-venue mid logged in orders.jsonl against the bar open (price source and timing, no spread);
demo_mid_gap_bp with the demo-book mid, and demo_live_mid_bp = demo mid / live mid - 1. The summary is
split by broker mode: demo fills are observed (Bitget Demo), shadow fills are estimated (simulated
at live-venue quotes), so a shadow gap compares two fill models.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment import agent, baseline, evidence, features, ledger, llm, replay, risk
from sentiment.config import (CARD_LOOKBACK_H, DECISION_HOURS, KILL_COOLDOWN_H, OUT, STOP_COOLDOWN_H,
                              perp)
from sentiment.data import news_available
from sentiment.sim import SimBroker

LIVE_DIR = OUT / "live"
SMOKE_RUN = OUT / "llm_2026-07-01_2026-07-03"
TOL = 1e-9                  # weights equal within this
BOOK_TOL = 0.002            # |shown - reference| weight per name allowed against the book at replay marks
MARK_GAP_DEFAULT = 0.005    # live-vs-replay mark gap assumed for a held name with no live price logged
MARK_GAP_SLACK = 0.0005     # added to an observed gap (the logged live mark is taken seconds after the pre-trade one)
POS_TOL = 1e-8              # contract quantities equal within this (shadow positions are rounded to 1e-10)
CTX_REL_TOL = 1e-9          # risk-context numbers equal within this (entry prices differ in the last ulp)
BROKER_MODES = ("shadow", "demo")
FILL_STATUS = {"demo": "observed (Bitget Demo fills) vs estimated (replay fill model)",
               "shadow": ("estimated (shadow ledger: simulated fills at live-venue quotes) vs estimated "
                          "(replay fill model)"),
               "unknown": "unknown broker mode (e.g. a relabelled replay) vs estimated (replay fill model)"}
MAX_DIFF_LINES = 40
CSV_COLS = ["ts", "event", "variant", "mode", "ticker", "status", "side", "qty_live", "qty_sim", "px_live",
            "px_sim", "gap_bp", "live_mid", "sim_mid", "mid_gap_bp", "demo_mid", "demo_mid_gap_bp",
            "demo_live_mid_bp", "live_order_ids", "sim_order_ids"]
HINT = {"price": "perp bars", "ret_4h": "perp bars", "ret_24h": "perp bars", "ret_7d": "perp bars",
        "vol_7d": "perp bars", "funding_last": "funding", "funding_7d_mean": "funding",
        "spot_premium": "spot/perp bars", "us_session_open": "clock", "fng": "crypto F&G",
        "fng_regime": "crypto F&G", "sp500_ret_24h": "index perp bars", "ndx_ret_24h": "index perp bars",
        "beta_60d": "perp bars (daily closes)", "beta_days": "perp bars (daily closes)"}


def _ts(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _num(x) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _norm(obj):
    """What a JSON log reader gets back for obj."""
    return json.loads(ledger.canonical(obj))


def record_version(rec: dict | None) -> str:
    """Risk version a record was written under: its `risk_version`, else v0 (records before risk v1)."""
    return (rec or {}).get("risk_version") or "v0"


def _flat(x, prefix: str = "") -> dict[str, object]:
    if isinstance(x, dict):
        out: dict[str, object] = {}
        for k, v in x.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
        return out or {prefix: {}}
    return {prefix: x}


def ctx_diffs(live: dict | None, rep: dict) -> list[dict]:
    """Fields of the logged risk context that differ from the rebuilt one (numbers within CTX_REL_TOL)."""
    if live is None:
        return [{"field": "risk_ctx", "live": None, "replay": "rebuilt"}]
    a, b = _flat(_norm(live)), _flat(_norm(rep))
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) \
                and not isinstance(y, bool):
            same = math.isclose(float(x), float(y), rel_tol=CTX_REL_TOL, abs_tol=1e-12)
        else:
            same = x == y
        if not same:
            out.append({"field": k, "live": x, "replay": y})
    return out


# ---------------------------------------------------------------- inputs

def load_log(path, modes=("live",)) -> list[dict]:
    """Records of these modes from a JSONL log, in file order; missing file -> []; bad lines skipped."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and (modes is None or r.get("mode") in modes):
            out.append(r)
    return out


def load_equity(path) -> dict[pd.Timestamp, float]:
    """Pre-trade equity per hour from equity.csv (last row wins on a repeated hour)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    d = pd.read_csv(path)
    d = d[pd.to_numeric(d.get("equity"), errors="coerce").notna()] if "equity" in d else d.iloc[:0]
    return {_ts(t): float(e) for t, e in zip(d["ts"], d["equity"])}


def load_orders(path) -> dict[str, dict]:
    """orders.jsonl filled lines keyed by fill order_id (live-venue quote per order)."""
    path = Path(path)
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        f = r.get("fill") if isinstance(r, dict) else None
        if r.get("status") == "filled" and isinstance(f, dict) and f.get("order_id"):
            out[str(f["order_id"])] = r
    return out


def card_reader():
    """Read-only stand-in for evidence.cards_for: cached cards of the current extractors, rating/insider
    cards rebuilt in memory. A news item with no row in cards.parquet is rebuilt from its cached
    extraction reply when there is one (an extraction that returned no cards leaves no row, and is not
    missing); otherwise it is never extracted, and the items of the last call without either are in
    `read.missing`."""
    versions = {"news": evidence.news_version(), "rating": evidence.RATING_VERSION,
                "insider": evidence.INSIDER_VERSION}
    table = evidence.load_cards()
    table = table[table["extractor"].isin(set(versions.values()))] if len(table) else table
    by_item: dict[str, list[dict]] = {}
    for r in table.to_dict("records"):
        if r["extractor"] == versions.get(r["kind"]):
            by_item.setdefault(r["item_id"], []).append(r)

    def read(items: list[dict]) -> list[dict]:
        read.missing = []
        out = []
        for it in items:
            rows = by_item.get(it["id"])
            if rows is None and it["kind"] in ("rating", "insider"):
                c = (evidence.rating_card if it["kind"] == "rating" else evidence.insider_card)(it)
                rows = evidence._rows(it, [c] if c else [], versions[it["kind"]])
            elif rows is None and it["kind"] == "news":
                rows = _cached_news_rows(it, versions["news"])
            if rows is None:
                read.missing.append(it)
                continue
            out += [evidence._to_card(r) for r in rows]
        return sorted(out, key=lambda c: (c["available_at"], c["card_id"]))
    read.missing = []
    return read


def _cached_news_rows(item: dict, version: str) -> list[dict] | None:
    """Card rows of a news item from its cached extraction reply (as evidence._news_cards parses it),
    [] for a reply with no cards, None when the extraction is not cached. Never calls the LLM."""
    text = llm.cached(evidence.news_messages(item))
    if text is None:
        return None
    try:
        parsed = llm.parse_json(text)
    except ValueError:
        parsed = {"cards": []}
    raw = parsed.get("cards", []) if isinstance(parsed, dict) else parsed
    return evidence._rows(item, evidence.normalize(raw, item), version)


# ---------------------------------------------------------------- book and risk rebuild

def pre_trade_book(store, t, positions: dict[str, float], equity: float,
                   last_prices: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """(weights, prices) of `positions` at t marked like SimBroker.mark: last closed bar <= t, else the
    last price seen."""
    prices = {}
    for tk in positions:
        b = store.bars(perp(tk), t, 1)
        px = float(b["close"].iloc[-1]) if b is not None and len(b) else last_prices.get(tk)
        if px is not None:
            prices[tk] = px
    w = {tk: q * prices[tk] / equity for tk, q in positions.items() if tk in prices and equity > 0}
    return w, prices


def _mirror_exit(t: pd.Timestamp, rec: dict, lstate: dict) -> None:
    """Apply a logged risk exit to the rebuilt risk state (after risk.check with an empty book has
    done the book-independent part: observe, kill trigger and expiry)."""
    for a in rec.get("risk_actions") or []:
        if a.get("rule") == "stop_loss" and a.get("ticker"):
            lstate.setdefault("cooldown", {})[a["ticker"]] = t + pd.Timedelta(hours=STOP_COOLDOWN_H)
            lstate.setdefault("entry", {}).pop(a["ticker"], None)
        elif a.get("rule") == "daily_kill" and not lstate.get("kill_until"):
            lstate["kill_until"] = t + pd.Timedelta(hours=KILL_COOLDOWN_H)


def broker_mode(rec: dict, orders: dict[str, dict]) -> str | None:
    """'shadow' / 'demo' from the live run_id (live_{variant}_{mode}), else from the orders.jsonl lines
    of the record's fills; None when neither says (e.g. a relabelled replay)."""
    rid = str(rec.get("run_id") or "")
    if rid.startswith("live_") and rid.rsplit("_", 1)[-1] in BROKER_MODES:
        return rid.rsplit("_", 1)[-1]
    modes = {orders[str(f.get("order_id"))].get("mode") for f in rec.get("fills") or []
             if str(f.get("order_id")) in orders} - {None}
    return modes.pop() if len(modes) == 1 else None


def _signed(fills: list[dict] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in fills or []:
        q = _num(f.get("qty"))
        if q and q > 0 and f.get("side") in ("buy", "sell"):
            out[f["ticker"]] = out.get(f["ticker"], 0.0) + (q if f["side"] == "buy" else -q)
    return out


def _positions(d: dict | None) -> dict[str, float]:
    return {k: float(v) for k, v in (d or {}).items() if _num(v)}


def implied_pre_positions(rec: dict) -> dict[str, float]:
    """Pre-trade positions a record implies: its logged `pre_trade.positions`, else its post-trade
    positions minus its signed fills."""
    pre = rec.get("pre_trade")
    if isinstance(pre, dict) and isinstance(pre.get("positions"), dict):
        return _positions(pre["positions"])
    pos = _positions((rec.get("marks") or {}).get("positions"))
    for tk, q in _signed(rec.get("fills")).items():
        pos[tk] = pos.get(tk, 0.0) - q
    return {k: v for k, v in pos.items() if abs(v) > POS_TOL}


def positions_diff(a: dict[str, float], b: dict[str, float]) -> list[str]:
    return sorted(k for k in set(a) | set(b)
                  if abs(a.get(k, 0.0) - b.get(k, 0.0)) > POS_TOL + 1e-9 * max(abs(a.get(k, 0.0)), abs(b.get(k, 0.0))))


def logged_book(rec: dict) -> dict[str, float] | None:
    """The live pre-trade book weights when the record logs `pre_trade` {positions, prices, equity}."""
    pre = rec.get("pre_trade")
    if not isinstance(pre, dict) or not isinstance(pre.get("positions"), dict):
        return None
    try:
        return replay.book_weights({"equity": float(pre["equity"]), "positions": _positions(pre["positions"]),
                                    "prices": _positions(pre.get("prices"))})
    except (KeyError, TypeError, ValueError):
        return None


def mark_gaps(t: pd.Timestamp, rec: dict, book: dict[str, float], prices: dict[str, float],
              orders: dict[str, dict]) -> dict[str, float]:
    """Relative gap between the live mark and the replay mark per held name: the record's own
    post-trade marks, else the live-venue quote of this record's order on the name, + MARK_GAP_SLACK;
    MARK_GAP_DEFAULT when neither exists."""
    live_px = _positions((rec.get("marks") or {}).get("prices"))
    for f in rec.get("fills") or []:
        o = orders.get(str(f.get("order_id"))) or {}
        q = o.get("live_quote") or {}
        px = _num(q.get("mark")) or _num(q.get("mid"))
        if px and f.get("ticker") not in live_px:
            live_px[f["ticker"]] = px
    out = {}
    for tk in book:
        a, b = live_px.get(tk), prices.get(tk)
        out[tk] = abs(a / b - 1) + MARK_GAP_SLACK if a and b else MARK_GAP_DEFAULT
    return out


def _rules(actions: list[dict] | None) -> list[str]:
    return sorted({f"{a.get('rule')}:{a.get('ticker') or '*'}" for a in actions or []})


def exit_check(t: pd.Timestamp, rec: dict | None, *, store, positions: dict[str, float], equity: float,
               last_prices: dict[str, float], lstate: dict, version: str = "v0") -> dict:
    """The replay's hourly risk.check at t (live positions at replay marks, a copy of the rebuilt risk
    state, risk `version`) against the live risk_exit record at t, or its absence."""
    book, prices = pre_trade_book(store, t, positions, equity, last_prices)
    entry = copy.deepcopy(lstate.get("entry") or {})
    forced, acts = risk.check(t, book, prices, copy.deepcopy(lstate), version=version)
    rep = _rules(acts) if forced is not None else []
    live = _rules(rec.get("risk_actions")) if rec else []
    out = {"ts": t, "live_exit": rec is not None, "replay_exit": forced is not None, "live_rules": live,
           "replay_rules": rep, "identical": live == rep, "risk_version": version}
    if not out["identical"]:
        disputed = {r.split(":", 1)[1] for r in set(live) ^ set(rep)}
        out["stop_distance"] = [
            {"ticker": tk, "entry": e["price"], "side": e["side"], "replay_mark": prices[tk],
             "loss_at_replay_mark": e["side"] * (prices[tk] / e["price"] - 1), "stop": -risk._stop_of(e, version)}
            for tk, e in sorted(entry.items()) if tk in disputed and tk in prices and e.get("price")]
    return out


# ---------------------------------------------------------------- decision diff

def state_diffs(live: dict | None, rep: dict) -> list[dict]:
    out = []
    live, rep = live or {}, _norm(rep)
    for key in sorted(set(live) | set(rep)):
        a, b = live.get(key) or {}, rep.get(key) or {}
        for f in sorted(set(a) | set(b)):
            x, y = a.get(f), b.get(f)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                same = math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=1e-15)
            else:
                same = x == y
            if not same:
                out.append({"field": f"{key}.{f}", "live": x, "replay": y, "source": HINT.get(f, "?")})
    return out


def _item_index(store) -> dict[str, dict]:
    return {it["id"][:10]: it for it in store.all_items()}


def _why(cid: str, t: pd.Timestamp, idx: dict[str, dict]) -> dict:
    it = idx.get(cid.rsplit("-", 1)[0][:10])
    if it is None:
        return {"card": cid, "item": None, "note": "item not in the reconciliation Store"}
    raw = it.get("raw") or {}
    at = _ts(it["available_at"])
    d = {"card": cid, "kind": it["kind"], "title": str(it.get("title") or "")[:100], "available_at": at}
    if it["kind"] == "news" and raw.get("published_at") is not None:
        d["published_minus_8h"] = news_available(raw["published_at"])
        d["first_seen"] = raw.get("first_seen")
    lo = t - pd.Timedelta(hours=CARD_LOOKBACK_H)
    d["note"] = ("available after t (first_seen / late arrival)" if at > t else
                 "older than the card lookback" if at <= lo else
                 "available by t in the Store now (arrived after the live step, or a missing card)")
    return d


def _book_line(messages: list[dict] | None) -> str | None:
    if not messages:
        return None
    return next((ln for ln in messages[-1]["content"].splitlines() if ln.startswith("CURRENT BOOK:")), None)


def live_messages(req_hash: str | None) -> list[dict] | None:
    if not req_hash:
        return None
    f = llm.CACHE_DIR / f"{req_hash}.json"
    try:
        return json.loads(f.read_text())["request"]["messages"] if f.exists() else None
    except (OSError, ValueError, KeyError):
        return None


def prompt_diff(live_msgs: list[dict] | None, rep_msgs: list[dict]) -> tuple[list[str], list[str]]:
    """(input categories that differ, unified diff lines of the user message)."""
    if live_msgs is None:
        return ["live prompt not in cache"], []
    cats = []
    if live_msgs == rep_msgs:
        return ["model id"], []
    if live_msgs[0]["content"] != rep_msgs[0]["content"]:
        cats.append("system prompt")
    if _book_line(live_msgs) != _book_line(rep_msgs):
        cats.append("book")
    a, b = live_msgs[-1]["content"].splitlines(), rep_msgs[-1]["content"].splitlines()
    diff = [ln for ln in difflib.unified_diff(a, b, "live", "replay", n=0, lineterm="")][:MAX_DIFF_LINES]
    return cats, diff


def _l1(a: dict | None, b: dict | None) -> float | None:
    if a is None or b is None:
        return None
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in set(a) | set(b)))


def _same(a: dict | None, b: dict | None) -> bool:
    d = _l1(a, b)
    return d is not None and d <= TOL


def prompt_book(messages: list[dict] | None, blinded: bool = False) -> dict[str, float] | None:
    """Book weights exactly as a decision prompt showed them (3 decimals, from the NAMES table; a held
    name that rounds to 0.000 keeps its sign from the CURRENT BOOK line). None without a prompt."""
    if not messages:
        return None
    lines = messages[-1]["content"].splitlines()
    i = next((k for k, ln in enumerate(lines) if ln.startswith("ticker ret_4h")), None)
    if i is None:
        return None
    tk = (lambda x: agent.UNALIAS.get(x, x)) if blinded else (lambda x: x)
    book: dict[str, float] = {}
    for ln in lines[i + 1:]:
        if not ln.strip():
            break
        tok = ln.split()
        book[tk(tok[0])] = float(tok[-1])
    held = (_book_line(messages) or "").removeprefix("CURRENT BOOK:").strip()
    for part in [] if held in ("", "flat") else held.split(", "):
        name, val = part.rsplit(" ", 1)
        if not book.get(tk(name)):
            book[tk(name)] = math.copysign(1e-6, float(val))
    return {k: v for k, v in book.items() if v != 0}


def book_check(shown: dict[str, float] | None, ref: dict[str, float], exact: bool) -> dict | None:
    """The book a prompt showed against the reference book: equal at 3 decimals (`exact`, a logged
    pre-trade book) or within BOOK_TOL per name (the book at replay marks). None without a prompt."""
    if shown is None:
        return None
    names = sorted(set(shown) | set(ref))
    gap = {k: abs(shown.get(k, 0.0) - ref.get(k, 0.0)) for k in names}
    bad = [k for k in names
           if (f"{shown.get(k, 0.0):+.3f}" != f"{ref.get(k, 0.0):+.3f}" if exact else gap[k] > BOOK_TOL)]
    return {"reference": "logged pre-trade book" if exact else "book at replay marks",
            "tol": 0.0 if exact else BOOK_TOL,
            "max_abs_diff": max(gap.values(), default=0.0), "ok": not bad,
            "diffs": [{"ticker": k, "shown": shown.get(k, 0.0), "reference": ref.get(k, 0.0)} for k in bad]}


def recompute(t: pd.Timestamp, rec: dict, *, store, cards_fn, book: dict[str, float], lstate: dict,
              idx: dict[str, dict], book_exact: bool = False, targets_tol: float = TOL) -> dict:
    """The replay decision at t for one live decision record, and how it differs from the live one.
    `book` is the reference pre-trade book (logged when `book_exact`, else at replay marks) for the risk
    layer and the baseline; the LLM prompt gets the book the live prompt showed when that is cached and
    passes `book_check`, else the reference book. `targets_tol` = the post-risk target L1 a mark gap can
    explain (TOL with a logged book)."""
    variant = rec.get("variant") or "llm"
    ver = record_version(rec)
    state = features.market_state(store, t, version=ver)
    items = replay._live(store.all_items(), t)
    cards = replay._live(cards_fn(items), t)
    missing = list(getattr(cards_fn, "missing", None) or [])
    base = baseline.decide(t, state, cards, book)
    out = {"ts": t, "event": "decision", "variant": variant, "risk_version": ver, "replay_source": None,
           "mismatch_inputs": [], "prompt_diff": [],
           "missing_cards": [{"item": it["id"], "title": str(it.get("title"))[:100]} for it in missing]}
    ctx = None
    if ver != "v0" and variant in replay.VARIANTS and variant != "shuffled":
        # rebuilt point-in-time as the replay does (never copied from the record), then checked against it
        ctx = replay.risk_context(t, store=store, state=state, book=book, lstate=copy.deepcopy(lstate),
                                  cards=[] if variant == "nonews" else cards, version=ver)
        out["risk_ctx_diffs"] = ctx_diffs(rec.get("risk_ctx"), ctx)
        out["risk_ctx_identical"] = not out["risk_ctx_diffs"]
        if out["risk_ctx_diffs"]:
            out["mismatch_inputs"].append("risk_ctx")
    live_dec = rec.get("decision") or {}
    unverifiable = None
    ref_name = "logged pre-trade book" if book_exact else "replay marks"
    if variant == "baseline":
        shown = [baseline.card_ref(c) for c in cards]
        dec, out["replay_source"] = base, "baseline rule"
        out["prompt_identical"] = replay.input_hash(t, state, shown) == rec.get("input_hash")
    elif variant in agent.VARIANTS and variant != "shuffled":
        live_msgs = live_messages(rec.get("llm_request_hash"))
        if rec.get("error"):
            unverifiable = "live LLM call failed (record error): the live decision is a hold, not a model reply"
        elif live_msgs is None:
            unverifiable = "live prompt not in the LLM cache: the live model reply cannot be re-read"
        pbook = prompt_book(live_msgs, variant == "blinded")
        out["book_check"] = book_check(pbook, book, book_exact)
        if pbook is None:
            out["book_source"] = f"{ref_name} (live prompt not cached)"
            pbook = book
        elif out["book_check"]["ok"]:
            out["book_source"] = "live prompt"
        else:
            out["book_source"] = f"{ref_name} (the live prompt's book differs from it)"
            out["mismatch_inputs"].append("book")
            pbook = book
        shown_cards = [] if variant == "nonews" else agent.select_cards(t, cards)
        shown = [agent._card_id(c) for c in shown_cards]
        msgs = agent.build_messages(t, state, shown_cards, pbook, variant, ctx)   # ctx: rebuilt (v1) or None (v0)
        req = llm.cache_key(llm.model_name(), msgs)
        out["replay_request_hash"] = req
        out["prompt_identical"] = req == rec.get("llm_request_hash")
        if llm.cached(msgs) is not None:
            dec = agent.decide(t, state, cards, pbook, variant, risk_ctx=ctx)   # cache hit: no network
            assert dec["llm_request_hash"] == req
            out["replay_source"] = "cache (live prompt)" if out["prompt_identical"] else "cache (replay prompt)"
        else:
            dec = None
            out["replay_source"] = "not cached: replay decision unknown (no LLM calls)"
        if not out["prompt_identical"]:
            cats, out["prompt_diff"] = prompt_diff(live_msgs, msgs)
            out["mismatch_inputs"] += [] if unverifiable else cats
    else:
        return {**out, "replay_source": f"variant {variant!r} not reconcilable", "prompt_identical": None,
                "decision_identical": None}
    live_ids, rep_ids = list(rec.get("evidence_ids") or []), shown
    out["cards_live_only"] = [_why(c, t, idx) for c in live_ids if c not in set(rep_ids)]
    out["cards_replay_only"] = [_why(c, t, idx) for c in rep_ids if c not in set(live_ids)]
    out["state_diffs"] = state_diffs(rec.get("state"), state)
    out["input_identical"] = replay.input_hash(t, state, [baseline.card_ref(c) for c in cards]) == rec.get("input_hash")
    if out["state_diffs"]:
        out["mismatch_inputs"].insert(0, "state")
    if out["cards_live_only"] or out["cards_replay_only"]:
        out["mismatch_inputs"].insert(0, "evidence")
    rep_targets = None
    if dec is not None:
        rep_targets, actions = risk.apply(t, dec, book, state, copy.deepcopy(lstate), version=ver)
        out["replay_risk_actions"] = _norm(actions)
    out["live_decision_targets"] = live_dec.get("targets")
    out["replay_decision_targets"] = None if dec is None else dec["targets"]
    out["live_targets"] = rec.get("targets")
    out["replay_targets"] = rep_targets
    out["decision_l1"] = None if dec is None else _l1(live_dec.get("targets"), dec["targets"])
    out["target_l1"] = _l1(rec.get("targets"), rep_targets)
    out["targets_identical"] = _same(rec.get("targets"), rep_targets)
    out["targets_tol"] = targets_tol
    out["targets_match"] = out["target_l1"] is not None and out["target_l1"] <= targets_tol
    out["baseline_l1"] = _l1(rec.get("baseline_targets"), base["targets"])
    same_dec = dec is not None and _same(live_dec.get("targets"), dec["targets"]) \
        and bool(live_dec.get("valid", True)) == bool(dec.get("valid", True))
    out["decision_identical"] = bool(out["prompt_identical"] and same_dec)
    if unverifiable or (out["prompt_identical"] and dec is None):
        out["decision_identical"] = None                      # live call failed, nothing cached on either side
        out["unverifiable"] = unverifiable or "prompt identical but not cached: the live call failed"
    out["mismatch_inputs"] = list(dict.fromkeys(out["mismatch_inputs"]))
    return out


# ---------------------------------------------------------------- fills

def _net(fills: list[dict]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for f in fills or []:
        q, p = _num(f.get("qty")), _num(f.get("price"))
        if not q or q <= 0 or p is None:
            continue
        d = out.setdefault((f["ticker"], f["side"]), {"qty": 0.0, "notional": 0.0, "ids": []})
        d["qty"] += q
        d["notional"] += q * p
        d["ids"].append(str(f.get("order_id")))
    return out


def _order_mid(ids: list[str], orders: dict[str, dict], quote: str) -> float | None:
    """Quantity-weighted mid of one quote ('live_quote' / 'demo_quote') over the orders.jsonl lines of ids."""
    mids = []
    for i in ids:
        o = orders.get(i)
        if o is None:
            continue
        m = (o.get(quote) or {}).get("mid") or (o.get("mid") if quote == "live_quote" else None)
        mids.append((_num((o.get("fill") or {}).get("qty")), _num(m)))
    mids = [(q, m) for q, m in mids if q and m]
    return sum(q * m for q, m in mids) / sum(q for q, _ in mids) if mids else None


def fill_rows(t: pd.Timestamp, rec: dict, sim_fills: list[dict], store, orders: dict[str, dict],
              mode: str | None = None) -> list[dict]:
    live, sim = _net(rec.get("fills")), _net(sim_fills)
    rows = []
    for tk, side in sorted(set(live) | set(sim)):
        a, b = live.get((tk, side)), sim.get((tk, side))
        s = 1 if side == "buy" else -1
        px_l = a["notional"] / a["qty"] if a else None
        px_s = b["notional"] / b["qty"] if b else None
        live_mid = _order_mid(a["ids"], orders, "live_quote") if a else None
        demo_mid = _order_mid(a["ids"], orders, "demo_quote") if a else None
        sim_mid = store.next_open(perp(tk), t)
        rows.append({
            "ts": t, "event": rec.get("event"), "variant": rec.get("variant"), "mode": mode or "unknown",
            "ticker": tk, "status": "matched" if a and b else "live_only" if a else "sim_only", "side": side,
            "qty_live": a and a["qty"], "qty_sim": b and b["qty"], "px_live": px_l, "px_sim": px_s,
            "gap_bp": s * (px_l / px_s - 1) * 1e4 if a and b else None,
            "live_mid": live_mid, "sim_mid": sim_mid,
            "mid_gap_bp": s * (live_mid / sim_mid - 1) * 1e4 if live_mid and sim_mid else None,
            "demo_mid": demo_mid,
            "demo_mid_gap_bp": s * (demo_mid / sim_mid - 1) * 1e4 if demo_mid and sim_mid else None,
            "demo_live_mid_bp": (demo_mid / live_mid - 1) * 1e4 if demo_mid and live_mid else None,
            "live_order_ids": ",".join(a["ids"]) if a else "", "sim_order_ids": ",".join(b["ids"]) if b else ""})
    return rows


# ---------------------------------------------------------------- driver

def reconcile(live_dir, *, store=None, cards_fn=None, out_dir=None, write: bool = True) -> dict:
    """Reconcile <live_dir>/log.jsonl (mode live) against the replay path; writes reconcile.csv/.json
    to out_dir (default live_dir) and returns the summary dict."""
    live_dir = Path(live_dir)
    recs = load_log(live_dir / "log.jsonl")
    eq = load_equity(live_dir / "equity.csv")
    orders = load_orders(live_dir / "orders.jsonl")
    if store is None and recs:
        from sentiment.data import Store
        store = Store(snapshot=False)
    cards_fn = cards_fn or (card_reader() if recs else None)
    idx = _item_index(store) if recs else {}
    by_t = {_ts(r["ts"]): r for r in recs}
    timeline = sorted(set(eq) | set(by_t))
    lstate: dict = {}
    ver = record_version(recs[0]) if recs else "v0"   # risk version of the latest record (hours between records)
    held: dict[str, dict[str, float]] = {}        # post-trade positions per broker mode
    mode = None                                   # broker mode of the latest record
    last_prices: dict[str, float] = {}
    decisions, exits, rows, breaks = [], [], [], []
    for t in timeline:
        e = eq.get(t)
        if e is not None:
            risk.observe(lstate, t, e)
        rec = by_t.get(t)
        if rec is not None:
            ver = record_version(rec)
            mode = broker_mode(rec, orders)
            implied = implied_pre_positions(rec)
            if mode not in held:
                positions, pos_src = implied, f"implied by the record (first record of broker mode {mode or 'unknown'})"
                if implied:                            # an account that was not flat, or a missed first order
                    breaks.append({"ts": t, "mode": mode, "kind": "first record of the mode starts non-flat",
                                   "tickers": sorted(implied), "implied": implied})
            else:
                positions, pos_src = held[mode], "carried from the previous record of this broker mode"
                diff = positions_diff(positions, implied)
                if diff:
                    breaks.append({"ts": t, "mode": mode, "kind": "carried != implied by the record",
                                   "tickers": diff, "carried": {k: positions.get(k, 0.0) for k in diff},
                                   "implied": {k: implied.get(k, 0.0) for k in diff}})
        else:
            positions = held.get(mode, {})
        if t.hour not in DECISION_HOURS and e is not None and (rec is None or rec.get("event") != "decision"):
            exits.append(exit_check(t, rec, store=store, positions=positions, equity=e,
                                    last_prices=last_prices, lstate=lstate, version=ver))
            if rec is None:
                risk.check(t, {}, {}, lstate, version=ver)   # the book-independent part of the live check
        if rec is None:
            continue
        e_src = "equity.csv" if e is not None else "post-trade equity (no equity.csv row)"
        e = e if e is not None else float(rec.get("equity") or 0.0)
        lbook = logged_book(rec)
        book, prices = pre_trade_book(store, t, positions, e, last_prices)
        if lbook is not None:
            book, targets_tol = lbook, TOL
        elif breaks and breaks[-1]["ts"] == t:         # the positions themselves are in doubt, not only the marks
            targets_tol = TOL
        else:
            gaps = mark_gaps(t, rec, book, prices, orders)
            targets_tol = 2 * sum(abs(w) * gaps[tk] for tk, w in book.items()) + TOL
        if rec.get("event") == "decision":
            lstate["equity"] = e
            d = recompute(t, rec, store=store, cards_fn=cards_fn, book=book, lstate=lstate, idx=idx,
                          book_exact=lbook is not None, targets_tol=targets_tol)
            if d.get("book_check") and not d["book_check"]["ok"]:     # a wrong book is not a mark gap
                targets_tol = d["targets_tol"] = TOL
                d["targets_match"] = d["targets_identical"]
            live_targets, _ = risk.apply(t, rec.get("decision") or {}, book, rec.get("state") or {}, lstate,
                                         version=ver)
            l1 = d["risk_rebuild_l1"] = _l1(live_targets, rec.get("targets"))
            d["risk_rebuild_exact"] = l1 is not None and l1 <= TOL
            d["risk_rebuild_ok"] = l1 is not None and l1 <= targets_tol
            d.update(equity_source=e_src, mode=mode or "unknown", positions_source=pos_src,
                     risk_book_source="logged pre-trade book" if lbook is not None else "replay marks")
            decisions.append(d)
            names = None
        else:
            risk.check(t, {}, {}, lstate, version=ver)
            _mirror_exit(t, rec, lstate)
            acts = rec.get("risk_actions") or []
            names = None if any(a.get("rule") in ("daily_kill", "kill") for a in acts) else \
                {a["ticker"] for a in acts if a.get("ticker")}
        sim = SimBroker(store, e)
        sim.positions = dict(positions)
        sim_fills = sim.rebalance(t, rec.get("targets") or {}, e, only=names)
        rows += fill_rows(t, rec, sim_fills, store, orders, mode)
        marks = rec.get("marks") or {}
        held[mode] = {k: float(v) for k, v in (marks.get("positions") or {}).items() if v}
        last_prices.update({k: float(v) for k, v in (marks.get("prices") or prices).items() if _num(v)})

    summary = summarize(recs, decisions, rows, live_dir, store, exits=exits, breaks=breaks)
    if write:
        out = Path(out_dir or live_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=CSV_COLS).to_csv(out / "reconcile.csv", index=False)
        (out / "reconcile.json").write_text(json.dumps(_clean({
            "summary": summary, "decisions": decisions, "positions_breaks": breaks,
            "exits": [x for x in exits if x["live_exit"] or x["replay_exit"] or not x["identical"]]}),
            indent=1, default=str) + "\n")
    return summary


def _clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _q(v: list[float], p: float) -> float | None:
    return float(np.percentile(v, p)) if v else None


def fill_stats(rows: list[dict]) -> dict:
    gaps = [r["gap_bp"] for r in rows if r["gap_bp"] is not None]
    col = lambda k: [r[k] for r in rows if r.get(k) is not None]  # noqa: E731
    return {
        "n_orders": len(rows), "n_matched": len(gaps),
        "n_live_only": sum(r["status"] == "live_only" for r in rows),
        "n_sim_only": sum(r["status"] == "sim_only" for r in rows),
        "gap_bp_median": _q(gaps, 50), "gap_bp_p90": _q(gaps, 90),
        "gap_bp_mean": float(np.mean(gaps)) if gaps else None,
        "abs_gap_bp_p90": _q([abs(g) for g in gaps], 90), "abs_gap_bp_max": max(map(abs, gaps)) if gaps else None,
        "mid_gap_bp_median": _q(col("mid_gap_bp"), 50), "mid_gap_bp_p90": _q(col("mid_gap_bp"), 90),
        "demo_mid_gap_bp_median": _q(col("demo_mid_gap_bp"), 50),
        "demo_live_mid_bp_median": _q(col("demo_live_mid_bp"), 50),
        "abs_demo_live_mid_bp_p90": _q([abs(x) for x in col("demo_live_mid_bp")], 90),
    }


def _rel(p: Path) -> str:
    """Path as shipped: relative to the project root when inside it (no machine-specific home path)."""
    from sentiment.config import ROOT
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def summarize(recs: list[dict], decisions: list[dict], rows: list[dict], live_dir: Path, store,
              exits: list[dict] | None = None, breaks: list[dict] | None = None) -> dict:
    n = len(decisions)
    exits, breaks = exits or [], breaks or []
    ident = [d["decision_identical"] for d in decisions]
    prompt = [d.get("prompt_identical") for d in decisions]
    l1 = [d["target_l1"] for d in decisions if d.get("target_l1") is not None]
    books = [d["book_check"] for d in decisions if d.get("book_check")]
    ctxs = [bool(d["risk_ctx_identical"]) for d in decisions if "risk_ctx_identical" in d]
    rate = lambda k: sum(bool(d.get(k)) for d in decisions) / n if n else None  # noqa: E731
    reasons: dict[str, int] = {}
    for d in decisions:
        for c in d.get("mismatch_inputs") or []:
            reasons[c] = reasons.get(c, 0) + 1
    modes = sorted({r["mode"] for r in rows} | {d.get("mode", "unknown") for d in decisions})
    by_mode = {m: {"status": FILL_STATUS.get(m, FILL_STATUS["unknown"]),
                   **fill_stats([r for r in rows if r["mode"] == m])} for m in modes}
    log = live_dir / "log.jsonl"
    return {
        "live_dir": _rel(live_dir), "log_exists": log.exists(),
        "log_verified": ledger.verify(log) if log.exists() else None,
        "store": getattr(store, "source", type(store).__name__ if store is not None else None),
        "llm_model": llm.model_name(),
        "status": by_mode[modes[0]]["status"] if len(modes) == 1 else
        ("mixed broker modes: see fills_by_mode" if modes else "nothing reconciled yet"),
        "broker_modes": modes,
        "identity_scope": ("LLM decision identity re-checks state, cards, system prompt and model given the book the "
                           "live prompt showed; that book is checked separately (book_check: exact against a logged "
                           f"pre-trade book, else within {BOOK_TOL} weight of the book at replay marks) and a failed "
                           "check makes the prompt a mismatch"),
        "n_live_records": len(recs), "n_decisions": n,
        "n_risk_exits": sum(r.get("event") == "risk_exit" for r in recs),
        "n_identical": sum(x is True for x in ident), "n_unverifiable": sum(x is None for x in ident),
        "n_live_errors": sum(bool(d.get("unverifiable", "").startswith("live LLM call failed")) for d in decisions),
        "decision_identity_rate": sum(x is True for x in ident) / n if n else None,
        "prompt_identity_rate": sum(x is True for x in prompt) / n if n else None,
        "n_book_checked": len(books), "n_book_mismatch": sum(not b["ok"] for b in books),
        "book_check_max_abs_diff": max((b["max_abs_diff"] for b in books), default=None),
        "risk_rebuild_ok_rate": rate("risk_rebuild_ok"), "risk_rebuild_exact_rate": rate("risk_rebuild_exact"),
        "risk_versions": {v: sum(d.get("risk_version") == v for d in decisions)
                          for v in sorted({d.get("risk_version") or "v0" for d in decisions})},
        "n_risk_ctx_checked": len(ctxs), "n_risk_ctx_mismatch": sum(not x for x in ctxs),
        "risk_ctx_identity_rate": sum(ctxs) / len(ctxs) if ctxs else None,
        "targets_identity_rate": rate("targets_identical"), "targets_match_rate": rate("targets_match"),
        "n_targets_book_driven": sum(bool(d.get("targets_match")) and not d.get("targets_identical")
                                     for d in decisions),
        "mismatch_inputs": reasons,
        # why inputs differ: cards the Store yields at t now but the live step did not see (items that reached
        # the Store after the step, e.g. ratings/insider filings fetched later with an earlier available_at),
        # cards the live step saw but the Store does not yield at t now, and live steps whose data refresh
        # logged a source error (HTTP 503 etc.), so that step decided on stale inputs
        "n_decisions_replay_only_cards": sum(bool(d.get("cards_replay_only")) for d in decisions),
        "n_decisions_live_only_cards": sum(bool(d.get("cards_live_only")) for d in decisions),
        "n_records_source_errors": sum(any("HTTP Error" in str(w) or "503" in str(w)
                                           for w in (r.get("data_warnings") or [])) for r in recs),
        "target_l1_mean": float(np.mean(l1)) if l1 else None, "target_l1_median": _q(l1, 50),
        "target_l1_max": max(l1) if l1 else None,
        "n_hourly_checks": len(exits), "n_exit_identical": sum(x["identical"] for x in exits),
        "exit_identity_rate": sum(x["identical"] for x in exits) / len(exits) if exits else None,
        "n_live_only_exits": sum(x["live_exit"] and not x["replay_exit"] for x in exits),
        "n_replay_only_exits": sum(x["replay_exit"] and not x["live_exit"] for x in exits),
        "n_exit_rule_diff": sum(x["live_exit"] and x["replay_exit"] and not x["identical"] for x in exits),
        "n_positions_breaks": len(breaks),
        **fill_stats(rows),
        "fills_by_mode": by_mode,
    }


def fabricate(run_dir, dest) -> Path:
    """Self-check input: a replay run's log relabelled mode 'live' (re-chained) plus its equity.csv."""
    run_dir, dest = Path(run_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "log.jsonl").unlink(missing_ok=True)
    lg = ledger.Ledger(dest / "log.jsonl")
    for r in load_log(run_dir / "log.jsonl", modes=None):
        lg.append({**r, "mode": "live"})
    (dest / "equity.csv").unlink(missing_ok=True)
    if (run_dir / "equity.csv").exists():                   # written at the end of a replay
        (dest / "equity.csv").write_text((run_dir / "equity.csv").read_text())
    return dest


def self_check_failures(s: dict) -> list[str]:
    """What a replay log relabelled live must reproduce exactly (same marks, same code path): every
    decision, prompt, post-risk target, risk rebuild and hourly exit, every order at 0 bp."""
    want = {"decision_identity_rate": 1.0, "prompt_identity_rate": 1.0, "targets_identity_rate": 1.0,
            "risk_rebuild_exact_rate": 1.0, "exit_identity_rate": 1.0}
    out = [f"{k} {s.get(k)}" for k, v in want.items() if s.get(k) != v]
    if (s.get("target_l1_max") or 0.0) > TOL:
        out.append(f"target_l1_max {s['target_l1_max']}")
    for k in ("n_book_mismatch", "n_risk_ctx_mismatch", "n_positions_breaks", "n_live_only", "n_sim_only"):
        if s.get(k):
            out.append(f"{k} {s[k]}")
    if s.get("n_matched") != s.get("n_orders") or (s.get("abs_gap_bp_max") or 0.0) >= 1e-6:
        out.append(f"orders matched {s.get('n_matched')}/{s.get('n_orders')}, max |gap| {s.get('abs_gap_bp_max')} bp")
    if not s.get("n_decisions"):
        out.append("no decisions")
    return out


def _fmt(x, nd=2) -> str:
    return "na" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def print_summary(s: dict, out: Path | None = None) -> None:
    print(f"reconcile {s['live_dir']}: {s['n_live_records']} live records, {s['n_decisions']} decisions, "
          f"{s['n_risk_exits']} risk exits (store {s['store']}, log verified {s['log_verified']})")
    if not s["n_decisions"] and not s["n_orders"]:
        print("  nothing to reconcile yet (no live decision records)")
    else:
        rate = s["decision_identity_rate"]
        print(f"  decision identity {s['n_identical']}/{s['n_decisions']}"
              f" ({'na' if rate is None else f'{rate:.0%}'}); prompt identity "
              f"{_fmt(s['prompt_identity_rate'])}; post-risk targets {_fmt(s['targets_identity_rate'])}; "
              f"unverifiable {s['n_unverifiable']}; "
              f"risk rebuild ok {_fmt(s['risk_rebuild_ok_rate'])}; mismatch inputs {s['mismatch_inputs'] or '-'}")
        print(f"  post-risk targets within the mark gap {_fmt(s['targets_match_rate'])} "
              f"({s['n_targets_book_driven']} book-driven); risk rebuild exact {_fmt(s['risk_rebuild_exact_rate'])}; "
              f"live LLM errors {s['n_live_errors']}; prompt book checked {s['n_book_checked']}, mismatched "
              f"{s['n_book_mismatch']} (max |diff| {_fmt(s['book_check_max_abs_diff'], 4)})")
        print(f"  target L1 mean {_fmt(s['target_l1_mean'], 4)} median {_fmt(s['target_l1_median'], 4)} "
              f"max {_fmt(s['target_l1_max'], 4)}; risk versions {s['risk_versions']}; risk context rebuilt "
              f"{s['n_risk_ctx_checked']}, mismatched {s['n_risk_ctx_mismatch']}")
        print(f"  hourly risk checks {s['n_hourly_checks']}: identical {s['n_exit_identical']}, live-only exits "
              f"{s['n_live_only_exits']}, replay-only exits {s['n_replay_only_exits']}, rule differences "
              f"{s['n_exit_rule_diff']}; position breaks {s['n_positions_breaks']}")
        for m, f in s["fills_by_mode"].items():
            print(f"  orders [{m}: {f['status']}] {f['n_orders']}: matched {f['n_matched']}, live only "
                  f"{f['n_live_only']}, sim only {f['n_sim_only']}; gap bp (+ = live worse) median "
                  f"{_fmt(f['gap_bp_median'])} p90 {_fmt(f['gap_bp_p90'])} mean {_fmt(f['gap_bp_mean'])}; live mid "
                  f"vs bar open median {_fmt(f['mid_gap_bp_median'])}; demo mid vs live mid median "
                  f"{_fmt(f['demo_live_mid_bp_median'])}")
    if out is not None:
        print(f"  -> {out / 'reconcile.csv'}, {out / 'reconcile.json'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--live-dir", default=str(LIVE_DIR))
    p.add_argument("--out", default=None, help="output directory (default: the live dir)")
    p.add_argument("--snapshot", action="store_true", help="use the committed snapshot Store (default data/raw)")
    p.add_argument("--self-check", nargs="?", const=str(SMOKE_RUN), default=None, metavar="RUN_DIR",
                   help="relabel a replay run's log as live in a temp dir and reconcile it (expects 100%% / 0 bp)")
    p.add_argument("--news-offset-h", type=float, default=None,
                   help="news stamp offset in hours (default: data.NEWS_STAMP_OFFSET); 0 = the rule before ff3bb00")
    a = p.parse_args(argv)
    os.environ["SENTIMENT_OFFLINE"] = "1"                     # reconciliation never calls the LLM
    from sentiment import data
    from sentiment.data import Store
    if a.news_offset_h is not None:
        data.NEWS_STAMP_OFFSET = pd.Timedelta(hours=a.news_offset_h)
        print(f"note: news available at published_at - {a.news_offset_h:g}h (not the current rule)")
    if a.self_check:
        tmp = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="reconcile_selfcheck_"))
        live_dir = fabricate(a.self_check, tmp)
        s = reconcile(live_dir, store=Store(snapshot=True), out_dir=tmp)
        print_summary(s, tmp)
        fails = self_check_failures(s)
        print(f"self-check {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}: replay log {a.self_check} "
              "relabelled live")
        sys.exit(0 if not fails else 1)
    live_dir = Path(a.live_dir)
    recs = load_log(live_dir / "log.jsonl")
    store = Store(snapshot=a.snapshot) if recs else None
    out = Path(a.out) if a.out else live_dir
    s = reconcile(live_dir, store=store, out_dir=out)
    print_summary(s, out)


if __name__ == "__main__":
    main()
