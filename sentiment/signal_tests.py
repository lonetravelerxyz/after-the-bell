"""The three pre-registered signal tests (docs/PREREG.md, "Signal tests"), for any window.

  uv run python -m sentiment.signal_tests --start 2026-06-22 --end 2026-08-10 --out out/signal_tests_dev.json

(a) Rank IC of the baseline score. At every decision time t (DECISION_HOURS, window dates inclusive) the
    per-ticker score of `baseline.scores` over the cards available in (t - CARD_LOOKBACK_H, t], the same
    cards the replay gives `baseline.decide`, is rank-correlated (Spearman, average ranks for ties) across
    the 10 names with the forward perp return from t to t + h, h = 4h, 24h, 72h. A decision whose scores
    (or returns) are all equal has no IC and is skipped. Estimate = mean IC over decision times; t =
    Newey-West (Bartlett) with lag = h in decision steps (1, 6, 18), for the overlap of the windows.
(b) Rating cards (kind rating, stance != 0). One observation per rating event = per (ticker, UTC day of
    available_at) (every rating dated d becomes available at d + 1 day 13:30 UTC, so a ticker-day is one
    event): the 24h demeaned return from the event's entry (the entry bar of its first card), signed by
    sign(sum of sign(stance)) over the event's cards KNOWN AT THAT ENTRY (entry bar <= the event's entry;
    a later card of the same day is not known when the position would be taken). An event whose known
    card signs cancel has no observation. Estimate = mean over events, t = mean / (sd / sqrt(n)).
(c) The same for news cards with novelty "new" (kind news, scope name, stance != 0), one observation per
    (ticker, UTC day of available_at). Until 2026-09-23 the sign summed ALL the day's cards, including
    cards available hours after the entry (a look-ahead; see report DEVIATIONS "signal-test-c-lookahead");
    that version is kept as the descriptive `all_day_sign` field, never used for the bar.

Prices (Store, the committed snapshot by default): every return is perp open to open. From a decision
time t the entry is `Store.next_open(sym, t)` (the replay's fill bar); from a card the entry is the open
of the first 1h bar at or after available_at. Demeaned = the name's return minus the equal-weight mean
return of the universe names priced over the same interval. A forward window must end by the window end
(end date + 1 day 00:00 UTC) STRICTLY BEFORE it: the snapshot keeps only bars that close by the test
window's end, so an exit exactly at the end has no price in the test window (the dev window has one); both
windows drop it the same way (`n_skipped_window_end`). A dev run never reads test-window prices and a test
run never reads prices after it. A forward window with too few priced names is `n_skipped_no_price`, a
constant score is `n_skipped_no_ic`. Only events with available_at inside the window count.

Significance: 5 tests (a counts as 3), Bonferroni at 5% -> `significant` = |t| > 2.6 (PREREG).
Status per test: "not significant" when |t| <= 2.6; otherwise "in-sample" for a window that overlaps the
dev window (up to DEV_END) and "observed" for a window inside the test period.

Cards (read-only, no LLM call, nothing written to the cache): ratings and insider filings are rebuilt
deterministically from the Store items (`evidence.rating_card` / `insider_card`, same ids as the replay);
news cards are read from `cache/cards.parquet` under the current news extractor. The news set is
pinned to a replay run (`--cards-from RUN_DIR`, default the window's baseline run directory
`baseline_v1_{start}_{end}` or `baseline_{start}_{end}` if it exists): only news items whose cards that run
logged count, because cards.parquet can hold news items the run never saw (see DIAGNOSIS-dev.md). The
output records the card source, and, when the pinned run is a baseline run, how many of its logged
`baseline_targets` the recomputed scores reproduce (clip(0.05 x score, +-MAX_W_NAME)).
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

from sentiment import baseline, config, evidence
from sentiment.config import CARD_LOOKBACK_H, DECISION_HOURS, DEV_END, MAX_W_NAME, OUT, UNIVERSE

IC_HORIZONS_H = (4, 24, 72)
EVENT_HORIZON_H = 24
DECISION_STEP_H = 4                        # DECISION_HOURS are 4h apart
T_BAR = 2.6                                # PREREG: 5 tests, Bonferroni at 5%
N_TESTS = 5
MIN_NAMES = 3                              # fewer priced names at a decision time -> no IC


def _ts(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _day(s) -> pd.Timestamp:
    return _ts(s).normalize()


def _f(x) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else float(x)


# ---------------------------------------------------------------- statistics

def nw_t(x, lag: int) -> tuple[float, float]:
    """(mean, Newey-West t of the mean) with Bartlett weights 1 - l/(lag+1); lag 0 = the plain t with the
    1/T variance. NaN when fewer than 2 observations or zero variance."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else math.nan), math.nan
    m = float(x.mean())
    e = x - m
    s = float(e @ e) / n
    for lg in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - lg / (lag + 1)) * float(e[lg:] @ e[:-lg]) / n
    return m, (m / math.sqrt(s / n) if s > 0 else math.nan)


def plain_t(x) -> tuple[float, float]:
    """(mean, mean / (sd / sqrt(n))) with the sample sd."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else math.nan), math.nan
    sd = float(x.std(ddof=1))
    return float(x.mean()), (float(x.mean()) / (sd / math.sqrt(n)) if sd > 0 else math.nan)


def spearman(a, b) -> float:
    """Rank correlation with average ranks; NaN if either side is constant."""
    ra = pd.Series(a, dtype=float).rank().to_numpy()
    rb = pd.Series(b, dtype=float).rank().to_numpy()
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _result(est, t, n, **extra) -> dict:
    t = _f(t)
    return {"n": int(n), "estimate": _f(est), "t": t, "significant": bool(t is not None and abs(t) > T_BAR),
            **extra}


# ---------------------------------------------------------------- prices

class Prices:
    """Perp opens and demeaned open-to-open returns from a Store (memoised next_open)."""

    def __init__(self, store):
        self.store = store
        self._open: dict[tuple[str, pd.Timestamp], float | None] = {}

    def open(self, tk: str, t: pd.Timestamp) -> float | None:
        k = (tk, t)
        if k not in self._open:
            p = self.store.next_open(tk, t)
            self._open[k] = p if p is not None and p > 0 and math.isfinite(p) else None
        return self._open[k]

    def returns(self, t0: pd.Timestamp, t1: pd.Timestamp) -> dict[str, float]:
        out = {}
        for tk in UNIVERSE:
            p0, p1 = self.open(tk, t0), self.open(tk, t1)
            if p0 is not None and p1 is not None:
                out[tk] = p1 / p0 - 1
        return out

    def demeaned(self, t0: pd.Timestamp, t1: pd.Timestamp) -> dict[str, float]:
        r = self.returns(t0, t1)
        if not r:
            return {}
        mu = sum(r.values()) / len(r)
        return {tk: v - mu for tk, v in r.items()}


# ---------------------------------------------------------------- tests

def decision_times(start, end) -> list[pd.Timestamp]:
    s, e = _day(start), _day(end) + pd.Timedelta(days=1)
    return [t for t in pd.date_range(s, e, freq="h", inclusive="left") if t.hour in DECISION_HOURS]


def live_cards(cards: list[dict], t: pd.Timestamp) -> list[dict]:
    lo = t - pd.Timedelta(hours=CARD_LOOKBACK_H)
    return [c for c in cards if lo < _ts(c["available_at"]) <= t]


def score_panel(cards: list[dict], times: list[pd.Timestamp]) -> dict[pd.Timestamp, dict[str, float]]:
    """baseline.scores at every decision time from the cards available in the lookback (point-in-time)."""
    cards = sorted(cards, key=lambda c: _ts(c["available_at"]))
    return {t: baseline.scores(t, live_cards(cards, t))[0] for t in times}


def ic_test(scores: dict[pd.Timestamp, dict[str, float]], prices: Prices, end_x: pd.Timestamp,
            horizon_h: int) -> dict:
    ics, skipped_const, skipped_end, skipped_price = [], 0, 0, 0
    for t in sorted(scores):
        t1 = t + pd.Timedelta(hours=horizon_h)
        if t1 >= end_x:                                    # the exit must lie strictly inside the window
            skipped_end += 1
            continue
        r = prices.demeaned(t, t1)
        names = [tk for tk in UNIVERSE if tk in r]
        if len(names) < MIN_NAMES:
            skipped_price += 1
            continue
        ic = spearman([scores[t][tk] for tk in names], [r[tk] for tk in names])
        if math.isnan(ic):
            skipped_const += 1
            continue
        ics.append(ic)
    lag = horizon_h // DECISION_STEP_H
    m, t = nw_t(ics, lag)
    return _result(m, t, len(ics), horizon_h=horizon_h, nw_lag=lag, n_skipped_no_ic=skipped_const,
                   n_skipped_window_end=skipped_end, n_skipped_no_price=skipped_price,
                   ic_positive_share=_f(np.mean(np.asarray(ics) > 0)) if ics else None)


def entry_time(available_at) -> pd.Timestamp:
    """Open of the first 1h bar at or after available_at."""
    return _ts(available_at).ceil("h")


def event_test(cards: list[dict], prices: Prices, start: pd.Timestamp, end_x: pd.Timestamp,
               horizon_h: int = EVENT_HORIZON_H) -> dict:
    """Signed demeaned return after these cards, one observation per (ticker, UTC day of availability): the
    demeaned return from the event's entry (its first card's entry bar) x sign(sum of sign(stance)) over the
    cards known at that entry (entry bar <= the event's entry); an event whose known signs cancel has no
    direction and no observation. `all_day_sign` = the same with every card of the day (look-ahead;
    descriptive only)."""
    rows, skipped_end, skipped_price = [], 0, 0
    for c in sorted(cards, key=lambda c: (_ts(c["available_at"]), str(c.get("card_id")))):
        at = _ts(c["available_at"])
        if not (start <= at < end_x) or int(c["stance"]) == 0:
            continue
        e0 = entry_time(at)
        e1 = e0 + pd.Timedelta(hours=horizon_h)
        for tk in [tk for tk in c.get("tickers") or [] if tk in UNIVERSE]:
            if e1 >= end_x:                                # the exit must lie strictly inside the window
                skipped_end += 1
                continue
            r = prices.demeaned(e0, e1)
            if tk not in r:
                skipped_price += 1
                continue
            rows.append({"ticker": tk, "day": at.normalize(), "entry": e0, "sign": float(np.sign(c["stance"])),
                         "ret": r[tk]})
    extra = {"horizon_h": horizon_h, "n_cards": len(rows), "n_skipped_window_end": skipped_end,
             "n_skipped_no_price": skipped_price}
    if not rows:
        return _result(math.nan, math.nan, 0, estimate_bp=None, n_ticker_days=0, n_dropped_net_zero=0,
                       n_later_cards_ignored=0, all_day_sign=None, **extra)
    d = pd.DataFrame(rows).sort_values(["ticker", "day", "entry"], kind="stable")
    key = ["ticker", "day"]
    d["event_entry"] = d.groupby(key)["entry"].transform("min")
    known = d[d["entry"] <= d["event_entry"]]                 # cards known when the event's position is taken
    net, first = known.groupby(key, sort=True)["sign"].sum(), known.groupby(key, sort=True)["ret"].first()
    ev = (np.sign(net) * first)[net != 0]
    m, t = plain_t(ev.to_numpy())
    net_all = d.groupby(key, sort=True)["sign"].sum()          # look-ahead version, descriptive only
    ev_all = (np.sign(net_all) * first)[net_all != 0]
    m_all, t_all = plain_t(ev_all.to_numpy())
    card_mean = float((d["sign"] * d["ret"]).mean())
    return _result(m, t, len(ev), estimate_bp=_f(m * 1e4), median_bp=_f(float(ev.median()) * 1e4) if len(ev) else None,
                   hit_rate=_f(float((ev > 0).mean())) if len(ev) else None, n_ticker_days=len(net),
                   n_dropped_net_zero=int((net == 0).sum()), n_later_cards_ignored=len(d) - len(known), **extra,
                   card_level_mean_bp=_f(card_mean * 1e4),
                   all_day_sign={"what": "sign from every card of the ticker-day, including cards available after "
                                         "the entry (look-ahead; descriptive, not used for the bar)",
                                 "n": len(ev_all), "estimate_bp": _f(m_all * 1e4), "t": _f(t_all),
                                 "n_dropped_net_zero": int((net_all == 0).sum())})


def rating_cards(cards: list[dict]) -> list[dict]:
    return [c for c in cards if c.get("kind") == "rating" and c.get("scope") == "name"]


def news_new_cards(cards: list[dict]) -> list[dict]:
    return [c for c in cards if c.get("kind") == "news" and c.get("scope") == "name" and c.get("novelty") == "new"]


def sample_label(start, end) -> str:
    s, e = _day(start), _day(end)
    return "dev" if e <= _day(DEV_END) else "test" if s > _day(DEV_END) else "mixed"


def status(res: dict, sample: str) -> str:
    if not res["significant"]:
        return "not significant"
    return "observed" if sample == "test" else "in-sample"


def run_tests(store, start, end, cards: list[dict]) -> dict:
    """All five pre-registered tests on [start, end] (UTC dates, inclusive) from these cards."""
    s, end_x = _day(start), _day(end) + pd.Timedelta(days=1)
    prices = Prices(store)
    times = decision_times(start, end)
    scores = score_panel(cards, times)
    sample = sample_label(start, end)
    tests = {f"a_ic_{h}h": ic_test(scores, prices, end_x, h) for h in IC_HORIZONS_H}
    tests["b_rating_24h"] = event_test(rating_cards(cards), prices, s, end_x)
    tests["c_news_new_24h"] = event_test(news_new_cards(cards), prices, s, end_x)
    for k, v in tests.items():
        v["status"] = status(v, sample)
    return {"window": {"start": f"{s:%Y-%m-%d}", "end": f"{end_x - pd.Timedelta(days=1):%Y-%m-%d}",
                       "sample": sample, "n_decision_times": len(times)},
            "t_bar": T_BAR, "n_tests": N_TESTS,
            "definitions": {
                "a": "mean cross-sectional Spearman IC of baseline.scores (cards available in (t-72h, t]) vs "
                     "forward perp open-to-open return (demeaned; rank-invariant) at 4/24/72h; Newey-West t, "
                     "lag = horizon / 4h; forward window must end strictly before the window end",
                "b": "rating cards with stance != 0, one observation per (ticker, UTC day of available_at): "
                     "24h demeaned perp return from the first bar open at/after the first card's available_at, "
                     "signed by sign(sum of sign(stance)) over the cards known at that entry bar; net-zero events "
                     "dropped; the exit must lie strictly before the window end; t = mean / (sd/sqrt(n)); "
                     "card_level_mean_bp and all_day_sign are descriptive only",
                "c": "as b for news cards with scope name, novelty new, stance != 0; one observation per "
                     "(ticker, UTC day)",
                "significant": "|t| > 2.6 (PREREG Bonferroni bar for 5 tests)"},
            "tests": tests, "_scores": scores}


# ---------------------------------------------------------------- cards (read-only)

def logged_ids(run_dir) -> tuple[set[str], str | None, list[dict]]:
    """(card ids a run logged, its variant, its decision records' ts + baseline_targets)."""
    ids: set[str] = set()
    variant, base = None, []
    for ln in (Path(run_dir) / "log.jsonl").read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        variant = variant or r.get("variant")
        ids |= set(r.get("evidence_ids") or [])
        for v in ((r.get("decision") or {}).get("evidence") or {}).values():
            ids |= set(v if isinstance(v, list) else [v])
        ids |= set(((r.get("risk_ctx") or {}).get("since_card") or {}).keys())
        if r.get("event") == "decision":
            base.append({"ts": _ts(r["ts"]), "baseline_targets": r.get("baseline_targets") or {}})
    return ids, variant, base


def _prefix(card_id: str) -> str:
    return str(card_id).rsplit("-", 1)[0]


def default_run_dir(start, end) -> Path | None:
    s, e = f"{_day(start):%Y-%m-%d}", f"{_day(end):%Y-%m-%d}"
    for name in (f"baseline_v1_{s}_{e}", f"baseline_{s}_{e}"):
        p = OUT / name
        if (p / "log.jsonl").exists():
            return p
    return None


def window_cards(store, start, end, cards_from=None) -> tuple[list[dict], dict]:
    """Cards of every item available in (start - lookback, end + 1 day], read-only (see module docstring).
    Returns (cards, provenance)."""
    s, end_x = _day(start), _day(end) + pd.Timedelta(days=1)
    lo = s - pd.Timedelta(hours=CARD_LOOKBACK_H)
    items = [it for it in store.all_items() if lo < _ts(it["available_at"]) <= end_x]
    out: list[dict] = []
    for it in items:
        if it["kind"] in ("rating", "insider"):
            c = (evidence.rating_card if it["kind"] == "rating" else evidence.insider_card)(it)
            if c:
                out.append({"card_id": f"{it['id'][:10]}-0", "item_id": it["id"], "kind": it["kind"],
                            "available_at": _ts(it["available_at"]), **c})
    news_ids = {it["id"]: it for it in items if it["kind"] == "news"}
    d = evidence.load_cards()
    ver = evidence.news_version()
    rows = d[(d["kind"] == "news") & (d["extractor"] == ver) & d["item_id"].isin(set(news_ids))] if len(d) else d
    prov = {"cards_file": str(evidence.CARDS_FILE.relative_to(config.ROOT)), "news_extractor": ver,
            "news_items_in_window": len(news_ids), "news_items_with_cards": int(rows["item_id"].nunique()) if len(rows) else 0}
    if cards_from is not None:
        ids, variant, _ = logged_ids(cards_from)
        pinned = {_prefix(i) for i in ids}
        window_prefixes = {k[:10] for k in news_ids}
        have = set(rows["card_id"]) if len(rows) else set()
        missing = sorted(i for i in ids if _prefix(i) in window_prefixes and i not in have)
        if missing:
            raise RuntimeError(f"{len(missing)} news card(s) logged by {cards_from} are not in {evidence.CARDS_FILE} "
                               f"under {ver} (first: {missing[0]}); the pinned card set cannot be rebuilt")
        before = prov["news_items_with_cards"]
        rows = rows[rows["item_id"].str[:10].isin(pinned)] if len(rows) else rows
        src = Path(cards_from).resolve()
        prov.update({"cards_from": str(src.relative_to(config.ROOT)) if src.is_relative_to(config.ROOT) else str(src),
                     "cards_from_variant": variant, "logged_card_ids": len(ids),
                     "news_items_dropped_by_pin": before - (int(rows["item_id"].nunique()) if len(rows) else 0)})
    else:
        ids = None
        prov["cards_from"] = None
    for r in rows.to_dict("records") if len(rows) else []:
        out.append(evidence._to_card(r))
    out.sort(key=lambda c: (c["available_at"], c["card_id"]))
    prov["n_cards"] = len(out)
    prov["n_cards_by_kind"] = {k: sum(c["kind"] == k for c in out) for k in ("news", "rating", "insider")}
    if ids is not None:
        prov["logged_cards_rebuilt"] = sum(c["card_id"] in ids for c in out)
    return out, prov


def check_baseline_targets(scores: dict[pd.Timestamp, dict[str, float]], logged: list[dict]) -> dict:
    """Decisions whose logged baseline_targets equal clip(0.05 x recomputed score) to 1e-6 on every name."""
    ok = n = 0
    for r in logged:
        sc = scores.get(r["ts"])
        if sc is None:
            continue
        n += 1
        want = {tk: round(max(-MAX_W_NAME, min(MAX_W_NAME, baseline.K_WEIGHT * v)), 6) for tk, v in sc.items()}
        got = r["baseline_targets"]
        ok += all(abs(want[tk] - float(got.get(tk, 0.0))) <= 1e-6 for tk in UNIVERSE)
    return {"decisions_compared": n, "decisions_matching": ok}


# ---------------------------------------------------------------- CLI

def compute(start, end, *, store=None, cards_from="auto", snapshot: bool = True) -> dict:
    os.environ.setdefault("SENTIMENT_OFFLINE", "1")          # nothing here calls the LLM; belt and braces
    if store is None:
        from sentiment.data import Store
        store = Store(snapshot=snapshot)
    run_dir = default_run_dir(start, end) if cards_from == "auto" else (config.run_dir(cards_from) if cards_from else None)
    cards, prov = window_cards(store, start, end, run_dir)
    res = run_tests(store, start, end, cards)
    scores = res.pop("_scores")
    prov["store_source"] = getattr(store, "source", None)
    if run_dir is not None:
        _, variant, logged = logged_ids(run_dir)
        if variant == "baseline":
            prov["baseline_targets_check"] = check_baseline_targets(scores, logged)
    res["cards"] = prov
    res["generated_by"] = "sentiment/signal_tests.py"
    return res


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", required=True, help="JSON output file")
    p.add_argument("--cards-from", default="auto", metavar="RUN_DIR",
                   help="pin the news cards to this run's log (default: the window's baseline run if present)")
    p.add_argument("--no-pin", action="store_true", help="use every cached news card of the window")
    p.add_argument("--raw", action="store_true", help="read data/raw instead of the committed snapshot")
    a = p.parse_args(argv)
    res = compute(a.start, a.end, cards_from=None if a.no_pin else a.cards_from, snapshot=not a.raw)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    for k, v in res["tests"].items():
        est = v.get("estimate_bp", v["estimate"])
        print(f"{k:15} n {v['n']:4}  est {est if est is None else round(est, 4)!s:>9}  "
              f"t {v['t'] if v['t'] is None else round(v['t'], 2)!s:>6}  {v['status']}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
