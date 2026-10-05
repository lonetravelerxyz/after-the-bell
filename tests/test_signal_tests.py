"""Pre-registered signal tests (sentiment/signal_tests.py, PREREG.md "Signal tests") on synthetic prices."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from sentiment import evidence, signal_tests as st
from sentiment.config import UNIVERSE

H = pd.Timedelta(hours=1)
D0 = pd.Timestamp("2026-07-01", tz="UTC")


class FakeStore:
    """next_open(tk, t) = 100 * exp(drift[tk] * hours since D0), plus optional log-price steps."""

    def __init__(self, drift=None, jumps=None, items=()):
        self.drift = drift or {tk: 0.0 for tk in UNIVERSE}
        self.jumps = jumps or {}                     # {tk: (t_from, log_jump)} step in log price from t_from on
        self._items = list(items)
        self.calls = []

    def next_open(self, tk, t):
        t = pd.Timestamp(t)
        self.calls.append(t)
        lp = self.drift[tk] * (t - D0) / H
        j = self.jumps.get(tk)
        if j and t >= j[0]:
            lp += j[1]
        return 100.0 * math.exp(lp)

    def all_items(self):
        return list(self._items)


def card(cid, at, tk, stance, strength=1.0, kind="news", novelty="new", scope="name"):
    return {"card_id": cid, "item_id": cid.split("-")[0] + "0" * 40, "kind": kind, "available_at": pd.Timestamp(at),
            "tickers": [tk] if scope == "name" else [], "scope": scope, "sector": "", "stance": stance,
            "strength": strength, "horizon_h": 24, "novelty": novelty, "summary": "", "quote": ""}


# ---- statistics ---------------------------------------------------------------------------------
def test_nw_t_lag0_is_plain_t_with_population_variance():
    x = np.array([0.1, -0.2, 0.3, 0.05, 0.15, -0.05])
    m, t = st.nw_t(x, 0)
    assert m == pytest.approx(x.mean())
    assert t == pytest.approx(x.mean() / math.sqrt(x.var(ddof=0) / len(x)))


def test_nw_t_shrinks_on_autocorrelated_series():
    rng = np.random.default_rng(0)
    e = rng.normal(0.02, 0.1, 400)
    x = np.convolve(e, np.ones(6) / 6, mode="valid")        # overlapping 6-step sums: positive autocorrelation
    _, t0 = st.nw_t(x, 0)
    _, t6 = st.nw_t(x, 6)
    assert abs(t6) < abs(t0)
    assert math.isnan(st.nw_t([1.0], 3)[1]) and math.isnan(st.nw_t([1.0, 1.0, 1.0], 1)[1])


def test_spearman_ties_and_constant():
    assert st.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert st.spearman([0, 0, 1], [1, 2, 3]) == pytest.approx(np.corrcoef([1.5, 1.5, 3], [1, 2, 3])[0, 1])
    assert math.isnan(st.spearman([1, 1, 1], [1, 2, 3]))


def test_significance_bar_and_status():
    assert st._result(0.1, 2.61, 10)["significant"] and not st._result(0.1, 2.6, 10)["significant"]
    assert st._result(-0.1, -3.0, 10)["significant"] and not st._result(0.1, math.nan, 10)["significant"]
    sig, ns = st._result(1, 3.0, 5), st._result(1, 1.0, 5)
    assert st.status(ns, "test") == "not significant"
    assert st.status(sig, "dev") == "in-sample" and st.status(sig, "test") == "observed"
    assert st.sample_label("2026-06-22", "2026-08-10") == "dev"
    assert st.sample_label("2026-08-11", "2026-09-22") == "test"
    assert st.sample_label("2026-08-01", "2026-08-20") == "mixed"


# ---- (a) rank IC ----------------------------------------------------------------------------------
def _ranked_world(sign=1.0):
    """Ticker i drifts at sign * i bp/h and has one fresh +1 card of strength (i+1)/10 at D0 00:30."""
    drift = {tk: sign * i * 1e-4 for i, tk in enumerate(UNIVERSE)}
    cards = [card(f"c{i}-0", D0 + 0.5 * H, tk, 1, (i + 1) / 10) for i, tk in enumerate(UNIVERSE)]
    return FakeStore(drift), cards


def test_ic_perfect_rank_and_window_end():
    store, cards = _ranked_world()
    res = st.run_tests(store, "2026-07-01", "2026-07-02", cards)
    a4, a24, a72 = (res["tests"][f"a_ic_{h}h"] for h in (4, 24, 72))
    # 12 decision times; the first (00:00) has no card yet -> constant scores, no IC
    assert res["window"]["n_decision_times"] == 12
    # the forward window must end strictly before 07-03 00:00 (the snapshot has no bar opening at the test
    # window's end): 4h drops the 07-02 20:00 decision, 24h every decision from 07-02 00:00 on
    assert a4["n_skipped_no_ic"] == 1 and a4["n_skipped_window_end"] == 1 and a4["n"] == 10
    assert a4["estimate"] == pytest.approx(1.0) and a4["n_skipped_no_price"] == 0
    assert a24["n_skipped_window_end"] == 6 and a24["n"] == 5 and a24["estimate"] == pytest.approx(1.0)
    assert a72["n"] == 0 and a72["estimate"] is None and a72["t"] is None and not a72["significant"]
    assert (a4["nw_lag"], a24["nw_lag"], a72["nw_lag"]) == (1, 6, 18)
    assert max(store.calls) < pd.Timestamp("2026-07-03", tz="UTC")      # no price at or after the window end
    store2, cards2 = _ranked_world(sign=-1.0)
    assert st.run_tests(store2, "2026-07-01", "2026-07-02", cards2)["tests"]["a_ic_4h"]["estimate"] == pytest.approx(-1.0)


def test_scores_point_in_time_and_lookback():
    t = D0 + 8 * H
    cards = [card("a-0", t - 1 * H, "NVDA", 1), card("b-0", t + 1 * H, "AAPL", 2),     # future: ignored
             card("c-0", t - 73 * H, "TSLA", -2), card("d-0", t - 2 * H, "META", 1, novelty="post_hoc")]
    sc = st.score_panel(cards, [t])[t]
    assert sc["NVDA"] == pytest.approx(math.exp(-1 / 24))
    assert sc["AAPL"] == 0 and sc["TSLA"] == 0 and sc["META"] == 0


# ---- (b)/(c) event tests ---------------------------------------------------------------------------
def test_event_net_sign_first_entry_and_demeaning():
    s, end_x = pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-07-04", tz="UTC")
    # NVDA jumps +10% (log) at 15:00 on 07-01; everything else flat
    store = FakeStore(jumps={"NVDA": (D0 + 15 * H, 0.1)})
    r = math.exp(0.1) - 1
    dm = r - r / len(UNIVERSE)                       # demeaned against the equal-weight universe
    cards = [
        card("r1-0", D0 + 13.5 * H, "NVDA", 1, kind="rating"),     # entry 14:00 -> captures the jump
        card("r2-0", D0 + 13.5 * H, "NVDA", 2, kind="rating"),     # same event: one observation
        card("r3-0", D0 + 13.5 * H, "NVDA", -1, kind="rating"),    # net sign still +1
        card("r4-0", D0 + 13.5 * H + 24 * H, "AAPL", 1, kind="rating"),
        card("r5-0", D0 + 13.5 * H + 24 * H, "AAPL", -1, kind="rating"),   # net zero: dropped
        card("r6-0", D0 + 14 * H, "TSLA", 0, kind="rating"),       # stance 0: ignored
        card("r7-0", D0 - 2 * H, "NVDA", 1, kind="rating"),        # before the window: ignored
        card("r8-0", end_x - 10 * H, "NVDA", 1, kind="rating"),    # forward window past the end: skipped
    ]
    res = st.event_test(st.rating_cards(cards), st.Prices(store), s, end_x)
    assert res["n"] == 1 and res["n_ticker_days"] == 2 and res["n_dropped_net_zero"] == 1
    assert res["n_skipped_window_end"] == 1 and res["n_cards"] == 5
    assert res["estimate"] == pytest.approx(dm) and res["estimate_bp"] == pytest.approx(dm * 1e4)
    assert res["t"] is None and not res["significant"]                 # one observation: no t


def test_event_uses_first_card_of_the_ticker_day():
    s, end_x = D0, D0 + pd.Timedelta(days=4)
    store = FakeStore(jumps={"NVDA": (D0 + 15 * H, 0.1)})
    later = card("n2-0", D0 + 16.2 * H, "NVDA", -1)                     # after the jump; same ticker-day
    first = card("n1-0", D0 + 10 * H, "NVDA", -1)                       # entry 10:00, before the jump
    res = st.event_test(st.news_new_cards([later, first]), st.Prices(store), s, end_x)
    r = math.exp(0.1) - 1
    assert res["n"] == 1 and res["estimate"] == pytest.approx(-(r - r / len(UNIVERSE)))


def test_card_filters():
    cs = [card("a-0", D0, "NVDA", 1), card("b-0", D0, "NVDA", 1, novelty="recap"),
          card("c-0", D0, "NVDA", 1, novelty="post_hoc"), card("d-0", D0, "NVDA", 1, scope="market"),
          card("e-0", D0, "NVDA", 1, kind="rating"), card("f-0", D0, "NVDA", 1, kind="insider")]
    assert [c["card_id"] for c in st.news_new_cards(cs)] == ["a-0"]
    assert [c["card_id"] for c in st.rating_cards(cs)] == ["e-0"]


def test_event_t_stat_is_plain():
    days = pd.date_range(D0, periods=30, freq="D")
    cards = [card(f"k{i}-0", d + 2 * H, UNIVERSE[i % len(UNIVERSE)], 1 if i % 3 else -1) for i, d in enumerate(days)]
    drift = {tk: (i - 4.5) * 1e-4 for i, tk in enumerate(UNIVERSE)}
    store = FakeStore(drift)
    s, end_x = D0, D0 + pd.Timedelta(days=31)
    res = st.event_test(cards, st.Prices(store), s, end_x)
    prices = st.Prices(store)
    obs = [np.sign(c["stance"]) * prices.demeaned(c["available_at"].ceil("h"),
                                                  c["available_at"].ceil("h") + 24 * H)[c["tickers"][0]] for c in cards]
    m, t = st.plain_t(obs)
    assert res["n"] == 30 and res["estimate"] == pytest.approx(m) and res["t"] == pytest.approx(t)


# ---- card loading (read-only) and CLI ----------------------------------------------------------------
def _item(kind, iid, at, raw, tickers=None):
    return {"id": iid, "kind": kind, "available_at": pd.Timestamp(at), "title": "t", "text": "", "tickers": tickers,
            "raw": raw}


def test_window_cards_pin_and_baseline_check(tmp_path, monkeypatch):
    n1, n2 = "a" * 40, "b" * 40
    rat = _item("rating", "c" * 40, D0 + 13.5 * H, {"symbol": "NVDA", "action": "调高评级", "rating_current": "买入",
                                                   "rating_previous": "中性"}, ["NVDA"])
    items = [_item("news", n1, D0 + 2 * H, {}), _item("news", n2, D0 + 3 * H, {}), rat]
    rows = pd.DataFrame([
        {"card_id": f"{n1[:10]}-0", "item_id": n1, "kind": "news", "available_at": D0 + 2 * H, "tickers": ["NVDA"],
         "scope": "name", "sector": "", "stance": 1, "strength": 1.0, "horizon_h": 24, "novelty": "new",
         "summary": "", "quote": "", "extractor": evidence.news_version()},
        {"card_id": f"{n2[:10]}-0", "item_id": n2, "kind": "news", "available_at": D0 + 3 * H, "tickers": ["AAPL"],
         "scope": "name", "sector": "", "stance": -1, "strength": 1.0, "horizon_h": 24, "novelty": "new",
         "summary": "", "quote": "", "extractor": evidence.news_version()}])
    monkeypatch.setattr(evidence, "load_cards", lambda: rows.copy())
    store = FakeStore(items=items)
    all_cards, prov = st.window_cards(store, "2026-07-01", "2026-07-02")
    assert prov["cards_from"] is None and prov["n_cards_by_kind"] == {"news": 2, "rating": 1, "insider": 0}
    # a baseline run that only logged n1 and the rating: n2 is dropped by the pin
    run = tmp_path / "baseline_run"
    run.mkdir()
    t = D0 + 4 * H
    sc = st.score_panel(all_cards, [t])[t]
    logged = {tk: round(0.05 * v, 6) for tk, v in sc.items() if tk != "AAPL" and abs(v) > 1e-9}
    recs = [{"ts": t.isoformat(), "event": "decision", "variant": "baseline",
             "evidence_ids": [f"{n1[:10]}-0", f"{'c' * 10}-0"], "baseline_targets": logged}]
    (run / "log.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    pinned, prov = st.window_cards(store, "2026-07-01", "2026-07-02", run)
    assert [c["card_id"] for c in pinned if c["kind"] == "news"] == [f"{n1[:10]}-0"]
    assert prov["news_items_dropped_by_pin"] == 1 and prov["logged_cards_rebuilt"] == 2
    sc2 = st.score_panel(pinned, [t])
    assert st.check_baseline_targets(sc2, [{"ts": t, "baseline_targets": logged}]) == \
        {"decisions_compared": 1, "decisions_matching": 1}
    # a logged news card that the cache no longer has cannot be rebuilt
    (run / "log.jsonl").write_text(json.dumps({**recs[0], "evidence_ids": [f"{n1[:10]}-9"]}) + "\n")
    with pytest.raises(RuntimeError):
        st.window_cards(store, "2026-07-01", "2026-07-02", run)


def test_compute_writes_json(tmp_path, monkeypatch):
    store, cards = _ranked_world()
    monkeypatch.setattr(st, "window_cards", lambda *a, **k: (cards, {"n_cards": len(cards)}))
    res = st.compute("2026-07-01", "2026-07-02", store=store, cards_from=None)
    assert set(res["tests"]) == {"a_ic_4h", "a_ic_24h", "a_ic_72h", "b_rating_24h", "c_news_new_24h"}
    assert res["t_bar"] == 2.6 and res["n_tests"] == 5
    for v in res["tests"].values():
        assert {"n", "estimate", "t", "significant", "status"} <= set(v)
    json.dumps(res)                                                      # JSON-safe
    c = res["tests"]["c_news_new_24h"]
    assert c["n"] == 10 and c["estimate"] == pytest.approx(0.0, abs=1e-12)  # all 10 names, same entry: demeaned sum 0


# ---- review fixes: no look-ahead in the event sign, the window end, missing prices -------------------
def test_event_sign_only_from_cards_known_at_entry():
    s, end_x = D0, D0 + pd.Timedelta(days=4)
    store = FakeStore(jumps={"NVDA": (D0 + 15 * H, 0.1)})
    r = math.exp(0.1) - 1
    dm = r - r / len(UNIVERSE)
    first = card("n1-0", D0 + 9.5 * H, "NVDA", 1)                      # entry 10:00, before the jump
    same_bar = card("n2-0", D0 + 9.8 * H, "NVDA", 1)                   # same entry bar: known at entry
    late = [card(f"l{i}-0", D0 + 16.5 * H, "NVDA", -1) for i in range(3)]   # after the jump: not known at 10:00
    res = st.event_test(st.news_new_cards([first, same_bar, *late]), st.Prices(store), s, end_x)
    assert res["n"] == 1 and res["estimate"] == pytest.approx(dm)       # sign +1 from the two known cards
    assert res["n_later_cards_ignored"] == 3
    assert res["all_day_sign"]["n"] == 1 and res["all_day_sign"]["estimate_bp"] == pytest.approx(-dm * 1e4)
    # a later card that would cancel the event does not drop it
    res = st.event_test(st.news_new_cards([first, late[0]]), st.Prices(store), s, end_x)
    assert res["n"] == 1 and res["n_dropped_net_zero"] == 0 and res["all_day_sign"]["n_dropped_net_zero"] == 1


class GappyStore(FakeStore):
    """No bar opens at or after `stop` (like the snapshot at the test window's end), and none at `hole`."""

    def __init__(self, stop, hole=None, **kw):
        super().__init__(**kw)
        self.stop, self.hole = stop, hole

    def next_open(self, tk, t):
        t = pd.Timestamp(t)
        if t >= self.stop or t == self.hole:
            return None
        return super().next_open(tk, t)


def test_window_end_is_treated_alike_and_missing_prices_have_their_own_reason():
    _, cards = _ranked_world()
    drift = {tk: i * 1e-4 for i, tk in enumerate(UNIVERSE)}
    end_x = pd.Timestamp("2026-07-03", tz="UTC")
    priced = st.run_tests(FakeStore(drift), "2026-07-01", "2026-07-02", cards)["tests"]
    cut = st.run_tests(GappyStore(end_x, drift=drift), "2026-07-01", "2026-07-02", cards)["tests"]
    for k in ("a_ic_4h", "a_ic_24h", "a_ic_72h"):             # a priced and an unpriced window end give the same
        assert {f: priced[k][f] for f in ("n", "n_skipped_window_end", "n_skipped_no_ic", "n_skipped_no_price")} == \
            {f: cut[k][f] for f in ("n", "n_skipped_window_end", "n_skipped_no_ic", "n_skipped_no_price")}
    hole = st.run_tests(GappyStore(end_x, hole=D0 + 12 * H, drift=drift), "2026-07-01", "2026-07-02", cards)["tests"]
    a4 = hole["a_ic_4h"]                                       # 08:00 -> 12:00 and 12:00 -> 16:00 lose a price
    assert a4["n_skipped_no_price"] == 2 and a4["n_skipped_no_ic"] == 1 and a4["n"] == priced["a_ic_4h"]["n"] - 2
