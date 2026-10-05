"""Execution layer: sim cost arithmetic, funding sign, each risk rule binds, ledger chain, metrics."""
import json

import numpy as np
import pandas as pd
import pytest

from sentiment import ledger, metrics, risk, sim
from sentiment.config import (DAILY_KILL, FUNDING_BLOCK, MAX_GROSS, MAX_NET, MAX_W_NAME,
                              OFF_HOURS_SCALE, START_EQUITY, TAKER, UNIVERSE)

T0 = pd.Timestamp("2026-07-01 14:00", tz="UTC")         # Wednesday, inside the US session
H = pd.Timedelta(hours=1)


@pytest.fixture(autouse=True)
def _risk_v0(monkeypatch):
    """These tests encode the v0 risk rules (flat per-name cap, flat 5% stop, no beta step); v1 is
    covered in test_risk_v1.py."""
    from sentiment import config
    monkeypatch.setattr(config, "RISK_VERSION", "v0")


class FakeStore:
    """1h bars from a price path per symbol: bar i opens at T0 + i h with open = close = path[i]."""

    def __init__(self, paths: dict[str, list[float]], funding: dict[str, dict] | None = None):
        self.b = {s: pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "quote_vol": 1.0},
                                  index=pd.DatetimeIndex([T0 + i * H for i in range(len(p))]))
                  for s, p in paths.items()}
        self.f = {s: pd.Series(v, dtype=float).sort_index() for s, v in (funding or {}).items()}

    def bars(self, sym, t, n):
        d = self.b[sym]
        return d[d.index + H <= t].tail(n)

    def next_open(self, sym, t):
        d = self.b[sym]
        d = d[d.index >= t]
        return float(d["open"].iloc[0]) if len(d) else None

    def settlements(self, sym, t0, t1):
        s = self.f.get(sym, pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC")))
        return s[(s.index > t0) & (s.index <= t1)]


# ---- sim --------------------------------------------------------------------------------------
def test_toy_book_costs_exact():
    hs = sim.half_spread("NVDA")
    assert hs == pytest.approx(0.4499336347884595 / 2 / 1e4)
    store = FakeStore({"NVDAUSDT": [100.0, 100.0, 110.0, 110.0]})
    br = sim.SimBroker(store, 10_000.0)
    # buy 10% of equity at the open of bar 0
    f1 = br.rebalance(T0, {"NVDA": 0.10}, 10_000.0)
    assert len(f1) == 1 and f1[0]["side"] == "buy"
    assert f1[0]["qty"] == pytest.approx(10.0)
    assert f1[0]["price"] == pytest.approx(100 * (1 + hs))
    assert f1[0]["fee"] == pytest.approx(TAKER * 10 * 100 * (1 + hs))
    assert f1[0]["half_spread_cost"] == pytest.approx(10 * 100 * hs)
    # sell everything at the open of bar 2 (110)
    f2 = br.rebalance(T0 + 2 * H, {}, 10_000.0)
    assert f2[0]["side"] == "sell" and f2[0]["qty"] == pytest.approx(10.0)
    assert f2[0]["price"] == pytest.approx(110 * (1 - hs))
    m = br.mark(T0 + 3 * H)
    fees = TAKER * 10 * (100 * (1 + hs) + 110 * (1 - hs))
    spread = 10 * hs * (100 + 110)
    assert m["positions"] == {}
    assert m["fees"] == pytest.approx(fees)
    assert m["spread"] == pytest.approx(spread)
    assert m["funding"] == 0
    assert m["equity"] == pytest.approx(10_000 + 100 - fees - spread)
    assert m["gross_equity"] == pytest.approx(10_100.0)


def test_sim_fallback_spread_and_churn_filter():
    assert sim.half_spread("NOT_A_TICKER") == sim.HALF_SPREAD_FALLBACK
    store = FakeStore({"NVDAUSDT": [100.0] * 4})
    br = sim.SimBroker(store, 10_000.0)
    br.rebalance(T0, {"NVDA": 0.10}, 10_000.0)
    assert br.rebalance(T0 + H, {"NVDA": 0.102}, 10_000.0) == []       # 0.2% < 0.25%: skipped
    assert br.rebalance(T0 + H, {"NVDA": 0.11}, 10_000.0) == []        # 1% < 25% of 0.11: skipped
    assert len(br.rebalance(T0 + 2 * H, {"NVDA": 0.14}, 10_000.0)) == 1  # 4% >= 25% of 0.14: trades


def test_band_stops_decay_churn():
    """Regression: a baseline-style weight decaying by exp(-4/24) per 4h step used to be re-traded at
    every decision (fixed 0.25%-of-equity band); the relative band only trades after it drifts >= 25%."""
    store = FakeStore({"NVDAUSDT": [100.0] * 40})
    br = sim.SimBroker(store, 10_000.0)
    w, n = 0.15, 0
    for step in range(8):
        n += len(br.rebalance(T0 + 4 * step * H, {"NVDA": w}, 10_000.0))
        w *= np.exp(-4 / 24)
    assert n == 4                                   # open + 3 resizes (the fixed band gave 8 fills)
    assert sim.below_band(0.0225, 0.1275) and not sim.below_band(0.05, 0.10)
    assert not sim.below_band(0.003, 0.003)         # opening a small position clears MIN_TRADE
    # flips and full closes always trade
    assert len(br.rebalance(T0 + 33 * H, {"NVDA": -0.003}, 10_000.0)) == 1
    assert len(br.rebalance(T0 + 34 * H, {}, 10_000.0)) == 1


def test_sim_skips_gap_and_trades_only_subset():
    """Regression: across a data gap the sim used to fill at the next existing bar's open (hours later)."""
    class GapStore(FakeStore):
        def next_open(self, sym, t):
            d = self.b[sym]
            d = d[(d.index >= t) & (d.index < t + H)]
            return float(d["open"].iloc[0]) if len(d) else None
    store = GapStore({"NVDAUSDT": [100.0] * 3 + [101.0] * 5, "AAPLUSDT": [50.0] * 8})
    store.b["NVDAUSDT"] = store.b["NVDAUSDT"].drop(index=[T0 + 3 * H, T0 + 4 * H])
    br = sim.SimBroker(store, 10_000.0)
    br.rebalance(T0, {"NVDA": 0.1, "AAPL": 0.1}, 10_000.0)
    fills = br.rebalance(T0 + 3 * H, {"NVDA": -0.1, "AAPL": -0.1}, 10_000.0)
    assert [f["ticker"] for f in fills] == ["AAPL"] and br.skipped == [{"ticker": "NVDA", "reason": "no bar opening within 1h"}]
    assert br.positions["NVDA"] > 0
    fills = br.rebalance(T0 + 5 * H, {"AAPL": -0.1}, 10_000.0, only={"NVDA"})
    assert [(f["ticker"], f["side"]) for f in fills] == [("NVDA", "sell")] and "NVDA" not in br.positions


def test_funding_sign():
    fund = {"NVDAUSDT": {T0 + 2 * H: 0.001}}
    for w, sign in ((0.10, -1), (-0.10, 1)):                 # long pays a positive rate, short receives
        store = FakeStore({"NVDAUSDT": [100.0, 100.0, 100.0, 100.0]}, fund)
        br = sim.SimBroker(store, 10_000.0)
        br.mark(T0)
        br.rebalance(T0, {"NVDA": w}, 10_000.0)
        m = br.mark(T0 + 3 * H)
        assert m["funding"] == pytest.approx(sign * 10 * 100 * 0.001)
        # settlement before the position existed is not booked
        br2 = sim.SimBroker(FakeStore({"NVDAUSDT": [100.0] * 4}, {"NVDAUSDT": {T0: 0.001}}), 10_000.0)
        br2.mark(T0)
        br2.rebalance(T0, {"NVDA": w}, 10_000.0)
        assert br2.mark(T0 + 3 * H)["funding"] == 0


# ---- risk -------------------------------------------------------------------------------------
def _state(price=100.0, funding=0.0, session=True):
    return {tk: {"price": price, "funding_last": funding, "us_session_open": session} for tk in UNIVERSE}


def _dec(targets, valid=True):
    return {"targets": targets, "reasons": {}, "evidence": {}, "confidence": 0.5, "raw": "", "valid": valid}


def _rules(actions):
    return [a["rule"] for a in actions]


def test_risk_invalid_holds():
    book = {"NVDA": 0.1}
    tg, acts = risk.apply(T0, _dec({"NVDA": 0.0}, valid=False), book, _state(), {})
    assert tg == book and _rules(acts) == ["invalid"]
    tg, acts = risk.apply(T0, _dec({"NVDA": float("nan")}), book, _state(), {})
    assert tg == book and _rules(acts) == ["invalid"]


def test_risk_name_cap():
    tg, acts = risk.apply(T0, _dec({"NVDA": 0.4, "AAPL": 0.05}), {}, _state(), {})
    assert tg == {"NVDA": MAX_W_NAME, "AAPL": 0.05}
    assert _rules(acts) == ["name_cap"] and acts[0]["ticker"] == "NVDA"


def test_risk_off_hours():
    t = pd.Timestamp("2026-07-01 02:00", tz="UTC")
    tg, acts = risk.apply(t, _dec({"NVDA": 0.15}), {}, _state(session=False), {})
    assert tg == {"NVDA": pytest.approx(MAX_W_NAME * OFF_HOURS_SCALE)} and _rules(acts) == ["off_hours"]
    # an existing position is not forced down off-hours
    tg, acts = risk.apply(t, _dec({"NVDA": 0.15}), {"NVDA": 0.15}, _state(session=False), {})
    assert tg == {"NVDA": 0.15} and acts == []
    assert not risk.us_session_open(pd.Timestamp("2026-07-04 15:00", tz="UTC"))   # Saturday
    assert risk.us_session_open(pd.Timestamp("2026-07-01 13:30", tz="UTC"))


def test_risk_funding_block():
    st = _state(funding=FUNDING_BLOCK)
    tg, acts = risk.apply(T0, _dec({"NVDA": 0.1, "AAPL": -0.1}), {"NVDA": 0.05}, st, {})
    assert tg == {"NVDA": 0.05, "AAPL": -0.1}                 # long capped at the held size, short free
    assert _rules(acts) == ["funding_block"]
    st = _state(funding=-FUNDING_BLOCK)
    tg, acts = risk.apply(T0, _dec({"NVDA": -0.1}), {}, st, {})
    assert tg == {} and _rules(acts) == ["funding_block"]


def test_risk_stop_loss_then_cooldown():
    ls = {}
    tg, _ = risk.apply(T0, _dec({"NVDA": 0.1}), {}, _state(100.0), ls)
    assert ls["entry"]["NVDA"] == {"price": 100.0, "side": 1}
    tg, acts = risk.apply(T0 + 4 * H, _dec({"NVDA": 0.1}), {"NVDA": 0.1}, _state(94.0), ls)
    assert tg == {} and _rules(acts) == ["stop_loss"]
    tg, acts = risk.apply(T0 + 8 * H, _dec({"NVDA": 0.1}), {}, _state(94.0), ls)
    assert tg == {} and _rules(acts) == ["cooldown"]
    tg, acts = risk.apply(T0 + 29 * H, _dec({"NVDA": 0.1}), {}, _state(94.0), ls)
    assert tg == {"NVDA": 0.1} and acts == [] and "NVDA" not in ls["cooldown"]


def test_risk_topup_reaverages_entry():
    """Regression: a same-side top-up kept the first entry price, so the stop fired only at -21% on the
    added lot (buy 0.05 @100, raise to 0.15 @120, stop at 95). Entry is now the quantity-weighted average."""
    ls = {}
    risk.apply(T0, _dec({"NVDA": 0.05}), {}, _state(100.0), ls)
    risk.apply(T0 + 4 * H, _dec({"NVDA": 0.15}), {"NVDA": 0.05}, _state(120.0), ls)
    avg = (0.05 / 120 * 100 + 0.10 / 120 * 120) / (0.15 / 120)
    assert ls["entry"]["NVDA"] == {"price": pytest.approx(avg), "side": 1}
    tg, acts = risk.apply(T0 + 8 * H, _dec({"NVDA": 0.15}), {"NVDA": 0.15}, _state(avg * 0.949), ls)
    assert tg == {} and _rules(acts) == ["stop_loss"]
    ls = {}                                         # shrinking keeps the entry
    risk.apply(T0, _dec({"NVDA": 0.15}), {}, _state(100.0), ls)
    risk.apply(T0 + 4 * H, _dec({"NVDA": 0.05}), {"NVDA": 0.15}, _state(110.0), ls)
    assert ls["entry"]["NVDA"]["price"] == 100.0


def test_risk_check_hourly_stop_and_kill():
    """Regression: stop-loss and daily kill were only evaluated at the 4-hourly decisions."""
    ls = {}
    risk.apply(T0, _dec({"NVDA": 0.1, "AAPL": 0.1}), {}, _state(100.0), ls)
    risk.observe(ls, T0 + H, START_EQUITY)
    assert risk.check(T0 + H, {"NVDA": 0.1, "AAPL": 0.1}, {"NVDA": 97.0, "AAPL": 99.0}, ls) == (None, [])
    forced, acts = risk.check(T0 + 2 * H, {"NVDA": 0.1, "AAPL": 0.1}, {"NVDA": 94.0, "AAPL": 99.0}, ls)
    assert forced == {"AAPL": 0.1} and _rules(acts) == ["stop_loss"] and acts[0]["ticker"] == "NVDA"
    assert "NVDA" not in ls["entry"] and "NVDA" in ls["cooldown"]
    tg, acts = risk.apply(T0 + 4 * H, _dec({"NVDA": 0.1, "AAPL": 0.1}), {"AAPL": 0.1}, _state(100.0), ls)
    assert "NVDA" not in tg and _rules(acts) == ["cooldown"]
    # kill between decisions: equity 3%+ below its 24h high at an off-decision hour -> flat everything
    risk.observe(ls, T0 + 5 * H, START_EQUITY * (1 - DAILY_KILL) - 1)
    forced, acts = risk.check(T0 + 5 * H, {"AAPL": 0.1}, {"AAPL": 99.0}, ls)
    assert forced == {} and _rules(acts) == ["daily_kill", "kill"] and ls["kill_until"] is not None


def test_replay_stops_between_decisions(tmp_path):
    """Regression: a -8% gap one hour after a decision was only acted on at the next 4h decision."""
    from sentiment import replay
    T = pd.Timestamp("2026-07-01", tz="UTC")
    path = [100.0] * 24 * 3
    for i in range(13, len(path)):                  # long opened at 12:00; 13:00 open -8%
        path[i] = 92.0
    idx = pd.DatetimeIndex([T - 10 * 24 * H + i * H for i in range(24 * 14)])

    class S:
        def __init__(self):
            p = np.r_[np.full(24 * 10, 100.0), path, np.full(len(idx) - 24 * 10 - len(path), 92.0)]
            self.b = pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "quote_vol": 1.0}, index=idx)

        def bars(self, sym, t, n):
            return self.b[self.b.index + H <= t].tail(n)

        def next_open(self, sym, t):
            d = self.b[self.b.index >= t]
            return float(d["open"].iloc[0]) if len(d) else None

        def funding(self, sym, t, n):
            return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))

        def settlements(self, sym, t0, t1):
            return self.funding(sym, t1, 0)

        def fng(self, t):
            return 50

        def spot_premium(self, sym, t):
            return None

        def all_items(self):
            return []

    def pol(t, state, cards, book, variant):
        return {"targets": {"NVDA": 0.1} if t.hour == 12 and t.day == 1 else dict(book), "reasons": {},
                "evidence": {}, "confidence": 0.5, "raw": "{}", "valid": True, "shown": [], "llm_request_hash": None}
    out = replay.run("llm", "2026-07-01", "2026-07-01", store=S(), out_dir=tmp_path, cards_fn=lambda items: [],
                     decide_fn=pol, quiet=True)
    recs = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    ex = [r for r in recs if r["event"] == "risk_exit"]
    assert len(ex) == 1 and ex[0]["ts"].startswith("2026-07-01 14:00")        # first mark showing the loss
    assert _rules(ex[0]["risk_actions"]) == ["stop_loss"] and ex[0]["fills"][0]["side"] == "sell"
    assert ledger.verify(out / "log.jsonl")
    m = json.loads((out / "metrics.json").read_text())
    assert m["n_risk_exits"] == 1 and m["n_decisions"] == 6 and m["n_invalid"] == 0


def test_risk_daily_kill():
    ls = {}
    risk.observe(ls, T0, START_EQUITY)
    ls["equity"] = START_EQUITY * (1 - DAILY_KILL) - 1
    book = {"NVDA": 0.1, "AAPL": -0.1}
    tg, acts = risk.apply(T0 + 4 * H, _dec({"NVDA": 0.1, "AAPL": -0.1}), book, _state(), ls)
    assert tg == {} and _rules(acts) == ["daily_kill", "kill", "kill"]
    tg, acts = risk.apply(T0 + 8 * H, _dec({"NVDA": 0.1}), {}, _state(), ls)
    assert tg == {} and _rules(acts) == ["kill"]
    tg, acts = risk.apply(T0 + 29 * H, _dec({"NVDA": 0.1}), {}, _state(), ls)
    assert tg == {"NVDA": 0.1} and acts == []


def test_risk_net_cap():
    tg, acts = risk.apply(T0, _dec({"NVDA": 0.15, "AAPL": 0.15, "META": 0.15, "TSLA": -0.05}), {}, _state(), {})
    assert sum(tg.values()) == pytest.approx(MAX_NET)
    assert tg["TSLA"] == -0.05 and set(_rules(acts)) == {"net_cap"} and len(acts) == 3


def test_risk_gross_cap():
    t = {tk: (0.15 if i % 2 else -0.15) for i, tk in enumerate(UNIVERSE)}       # gross 1.5, net 0
    tg, acts = risk.apply(T0, _dec(t), {}, _state(), {})
    assert sum(abs(v) for v in tg.values()) == pytest.approx(MAX_GROSS)
    assert set(_rules(acts)) == {"gross_cap"} and len(acts) == len(UNIVERSE)


# ---- ledger -----------------------------------------------------------------------------------
def test_ledger_chain_and_tamper(tmp_path):
    p = tmp_path / "log.jsonl"
    lg = ledger.Ledger(p)
    r1 = lg.append({"ts": T0, "mode": "replay", "equity": 10_000.0, "targets": {"NVDA": 0.1}})
    assert r1["prev_hash"] == ledger.GENESIS
    lg2 = ledger.Ledger(p)                                   # reopening continues the chain
    r2 = lg2.append({"ts": T0 + H, "mode": "replay", "equity": 10_001.5, "fills": []})
    assert r2["prev_hash"] == r1["hash"] and ledger.verify(p)
    lines = p.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["equity"] = 20_000.0
    p.write_text(ledger.canonical(rec) + "\n" + lines[1] + "\n")
    assert not ledger.verify(p)
    p.write_text(lines[1] + "\n")                            # deleting a line breaks the chain too
    assert not ledger.verify(p)


# ---- metrics ----------------------------------------------------------------------------------
def _write_run(d, equity, fills=()):
    d.mkdir()
    ts = pd.date_range("2026-07-01", periods=len(equity), freq="h", tz="UTC")
    pd.DataFrame({"ts": ts, "equity": equity, "gross_equity": np.asarray(equity) + 30.0,
                  "fees": 10.0, "spread": 5.0, "funding": -15.0}).to_csv(d / "equity.csv", index=False)
    lg = ledger.Ledger(d / "log.jsonl")
    lg.append({"ts": ts[0], "fills": list(fills)})
    return d


def test_metrics_synthetic(tmp_path):
    n = 24 * 20
    eq = 10_000 * np.cumprod(np.r_[1.0, np.full(n - 1, 1.0001)])
    eq[200:260] *= 0.9                                       # a 10% dip that recovers
    fills = [{"ticker": "NVDA", "side": "buy", "qty": 10, "price": 100, "fee": 1, "ts": "2026-07-01 01"},
             {"ticker": "NVDA", "side": "sell", "qty": 10, "price": 105, "fee": 1, "ts": "2026-07-02 01"},
             {"ticker": "AAPL", "side": "sell", "qty": 5, "price": 100, "fee": 1, "ts": "2026-07-03 01"},
             {"ticker": "AAPL", "side": "buy", "qty": 10, "price": 102, "fee": 1, "ts": "2026-07-04 01"},
             {"ticker": "AAPL", "side": "sell", "qty": 5, "price": 101, "fee": 1, "ts": "2026-07-05 01"}]
    m = metrics.compute(_write_run(tmp_path / "a", eq, fills))
    # hourly marks 07-01 00:00 .. 07-20 23:00: 19 complete days, the 20th (23h) is partial and not counted
    assert m["n_days"] == 19 and m["partial_last_day_h"] == 23 and m["partial_first_day_h"] == 0
    assert m["partial_last_day_return"] == pytest.approx(eq[-1] / eq[-24] - 1)
    assert m["trades"] == 5
    assert m["total_return"] == pytest.approx(eq[-1] / eq[0] - 1)
    assert m["max_dd"] == pytest.approx(eq[200] / eq[199] - 1)
    assert m["round_trips"] == 3 and m["win_rate"] == pytest.approx(1 / 3)   # +48, -11.5 (flip), -6.5
    assert m["ladder"]["after_funding"] == pytest.approx(m["total_return"])
    assert m["ladder"]["gross"] > m["ladder"]["after_fees"] > m["ladder"]["after_spread"] > m["ladder"]["after_funding"]
    r = metrics.daily_returns(pd.Series(eq, index=pd.date_range("2026-07-01", periods=n, freq="h", tz="UTC")))
    r = r.iloc[:-1]                                          # drop the partial last day
    assert m["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(365))
    assert json.dumps(m, allow_nan=False)


def test_increment_bootstrap(tmp_path):
    rng = np.random.default_rng(1)
    n = 24 * 60
    base = 10_000 * np.cumprod(1 + rng.normal(0, 0.002, n))
    better = base * np.cumprod(np.full(n, 1.0002))
    a = _write_run(tmp_path / "a", better)
    b = _write_run(tmp_path / "b", base)
    out = metrics.increment(a, b, block_days=5, n=500)
    assert out["n_days"] == 59 and out["sharpe_diff"] > 0          # 60 x 24 marks: the 60th day is partial
    assert out["ci95"][0] > 0 and out["p_diff_le_0"] < 0.05
    assert metrics.increment(a, b, n=500) == out            # fixed seed: deterministic
    assert out["n_boot_dropped"] == 0 and out["n_boot_used"] == 500
    same = metrics.increment(b, b, n=200)
    assert same["sharpe_diff"] == 0 and same["ci95"] == [0.0, 0.0]


def test_increment_counts_dropped_paths(tmp_path):
    """A resample of a flat day sequence has zero variance: its Sharpe is not finite and the path is dropped."""
    ts = pd.date_range("2026-07-01", periods=24 * 6 + 1, freq="h", tz="UTC")          # 6 complete days
    eq = np.full(len(ts), 10_000.0)
    eq[24 * 3:] = 10_100.0                                   # one up day, five flat days
    a = _write_run(tmp_path / "a", eq)
    b = _write_run(tmp_path / "b", np.linspace(10_000.0, 10_060.0, len(ts)))
    out = metrics.increment(a, b, n=300)
    assert out["n_days"] == 6 and out["n_boot_dropped"] > 0
    assert out["n_boot_used"] + out["n_boot_dropped"] == 300


def test_partial_first_and_last_day(tmp_path):
    """A curve from 10:00 on day 1 to 05:00 on day 4: days 2 and 3 are complete, the rest is reported apart."""
    ts = pd.date_range("2026-07-01 10:00", "2026-07-04 05:00", freq="h", tz="UTC")
    eq = pd.Series(10_000.0 + np.arange(len(ts)), index=ts)
    full, info = metrics.complete_days(eq)
    assert full.index[0] == pd.Timestamp("2026-07-02", tz="UTC") and full.index[-1] == pd.Timestamp("2026-07-04", tz="UTC")
    assert info["partial_first_day_h"] == 14 and info["partial_last_day_h"] == 5
    assert info["partial_last_day_return"] == pytest.approx(eq.iloc[-1] / eq[pd.Timestamp("2026-07-04", tz="UTC")] - 1)
    assert list(metrics.complete_day_returns(eq).index) == [pd.Timestamp(f"2026-07-0{d}", tz="UTC") for d in (3, 4)]
    one_day = eq[eq.index < pd.Timestamp("2026-07-02", tz="UTC")]        # no midnight at all
    _, info = metrics.complete_days(one_day)
    assert info["partial_first_day_h"] == 0 and info["partial_last_day_h"] == 13
    assert len(metrics.complete_day_returns(one_day)) == 0


def test_metrics_base_is_pre_trade_and_days_are_midnight_to_midnight(tmp_path):
    """Regression: metrics used the post-trade mark as the base (dropping the first build's costs) and
    made an extra one-hour 'day' out of the final end+1 00:00 mark."""
    ts = pd.date_range("2026-07-01", "2026-07-04", freq="h", tz="UTC")      # 3 days, 73 hourly marks
    eq = pd.Series(np.linspace(10_000.0, 9_970.0, len(ts)), index=ts)
    eq.iloc[1:] -= 2.0                                                     # the first build's costs
    r = metrics.daily_returns(eq)
    mid = eq[eq.index.hour == 0]
    assert len(r) == 3 and r.to_numpy() == pytest.approx((mid / mid.shift(1) - 1).dropna().to_numpy())
    d = tmp_path / "run"
    d.mkdir()
    pd.DataFrame({"ts": ts, "equity": eq.to_numpy(), "gross_equity": eq.to_numpy() + 2.0, "fees": 1.5, "spread": 0.5,
                  "funding": 0.0}).assign(gross_equity=lambda x: np.r_[10_000.0, x.gross_equity[1:]]).to_csv(
        d / "equity.csv", index=False)
    ledger.Ledger(d / "log.jsonl").append({"ts": ts[0], "event": "decision", "fills": [
        {"ticker": "NVDA", "side": "buy", "qty": 0.0, "price": float("nan"), "fee": 0.0, "ts": "x"},
        {"ticker": "NVDA", "side": "sell", "qty": 1.0, "price": 100.0, "fee": 0.06, "ts": "y"}],
        "decision": {"valid": False}})
    m = metrics.compute(d)
    assert m["start_equity"] == 10_000.0 and m["n_days"] == 3
    assert m["total_return"] == pytest.approx(eq.iloc[-1] / 10_000.0 - 1)
    assert m["ladder"]["after_funding"] == pytest.approx(m["total_return"])
    assert m["trades"] == 1 and m["n_invalid"] == 1                     # qty-0 / NaN fill ignored
    assert metrics.round_trips([{"ticker": "A", "side": "buy", "qty": 0.0, "price": float("nan"), "fee": 0, "ts": "1"}]) == []
