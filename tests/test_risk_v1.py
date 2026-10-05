"""Risk layer v1: point-in-time betas, vol-scaled per-name cap, vol-based stop, beta neutralisation, v0 kept."""
import numpy as np
import pandas as pd
import pytest

from sentiment import config, risk
from sentiment.config import (MAX_GROSS, MAX_NET, MAX_W_NAME, OFF_HOURS_SCALE, STOP_CAP, STOP_FLOOR, STOP_K,
                              STOP_LOSS_NAME, UNIVERSE, VOL_CAP_FLOOR, perp)
from sentiment.data import Store
from sentiment.features import betas, daily_closes, market_state

T0 = pd.Timestamp("2026-07-01 14:00", tz="UTC")         # Wednesday, inside the US session
OFF = pd.Timestamp("2026-07-01 02:00", tz="UTC")
H = pd.Timedelta(hours=1)
LOW = 0.02                                               # the median vol in the fixtures
VOLS = {**{tk: LOW for tk in UNIVERSE}, "MSTR": 0.06, "COIN": 0.04, "HOOD": 0.30, "NVDA": 0.01}
BETAS = {"NVDA": 0.6, "AAPL": 0.3, "GOOGL": 0.5, "META": 0.7, "AMZN": 0.6, "TSLA": 1.2,
         "MSTR": 1.9, "COIN": 1.7, "HOOD": 1.6, "CRCL": 1.8}


def _state(price=100.0, vols=None, bet=None, session=True, funding=0.0):
    vols = VOLS if vols is None else vols
    return {tk: {"price": price, "funding_last": funding, "us_session_open": session,
                 "vol_7d": vols.get(tk), "beta_60d": (bet or {}).get(tk)} for tk in UNIVERSE}


def _dec(targets, valid=True):
    return {"targets": targets, "reasons": {}, "evidence": {}, "confidence": 0.5, "raw": "", "valid": valid}


def _rules(actions):
    return [a["rule"] for a in actions]


def _bw(w, bet):
    return sum(bet.get(tk, 1.0) * v for tk, v in w.items())


@pytest.fixture(autouse=True)
def _risk_v1(monkeypatch):
    """Pin v1 so these tests do not depend on the config default (checked separately below)."""
    monkeypatch.setattr(config, "RISK_VERSION", "v1")


@pytest.fixture
def no_beta(monkeypatch):
    monkeypatch.setattr(config, "BETA_NEUTRAL", False)


def test_config_v1_default_and_fixed_parameters():
    from pathlib import Path
    assert '\nRISK_VERSION = "v1"' in Path(config.__file__).read_text()             # the shipped default
    assert risk._version(None) == "v1" and risk._version("v0") == "v0"
    assert (VOL_CAP_FLOOR, STOP_K, STOP_FLOOR, STOP_CAP) == (0.33, 2.0, 0.03, 0.10)
    assert config.BETA_NEUTRAL is True and (config.BETA_LOOKBACK_D, config.BETA_MIN_D) == (60, 30)
    with pytest.raises(ValueError):
        risk.apply(T0, _dec({}), {}, _state(), {}, version="v2")


# ---- (b) vol-scaled per-name cap ----------------------------------------------------------------
def test_vol_caps_scale_with_median_over_vol():
    st = _state()
    assert risk.median_vol(st) == pytest.approx(LOW)
    caps = risk.vol_caps(st, UNIVERSE)
    assert caps["AAPL"] == pytest.approx(MAX_W_NAME)                            # at the median: full cap
    assert caps["NVDA"] == pytest.approx(MAX_W_NAME)                            # below the median: clipped at 1
    assert caps["COIN"] == pytest.approx(MAX_W_NAME * 0.5)
    assert caps["MSTR"] == pytest.approx(MAX_W_NAME * max(VOL_CAP_FLOOR, LOW / 0.06))
    assert caps["HOOD"] == pytest.approx(MAX_W_NAME * VOL_CAP_FLOOR)            # floored
    st["CRCL"]["vol_7d"] = None                                                 # no vol at t -> flat cap
    assert risk.vol_caps(st, ["CRCL"])["CRCL"] == MAX_W_NAME
    assert risk.vol_caps({}, ["NVDA"]) == {"NVDA": MAX_W_NAME}


def test_high_vol_name_gets_smaller_cap(no_beta):
    tg, acts = risk.apply(T0, _dec({"AAPL": 0.15, "COIN": 0.15, "HOOD": -0.15}), {}, _state(), {})
    assert tg == {"AAPL": 0.15, "COIN": pytest.approx(0.075), "HOOD": pytest.approx(-MAX_W_NAME * VOL_CAP_FLOOR)}
    assert _rules(acts) == ["name_cap", "name_cap"] and {a["ticker"] for a in acts} == {"COIN", "HOOD"}
    # an existing position above the vol cap is forced down (hard rule), like the v0 flat cap
    tg, acts = risk.apply(T0, _dec({"COIN": 0.15}), {"COIN": 0.15}, _state(), {})
    assert tg == {"COIN": pytest.approx(0.075)} and _rules(acts) == ["name_cap"]


def test_off_hours_multiplies_the_vol_cap(no_beta):
    tg, acts = risk.apply(OFF, _dec({"AAPL": 0.15, "COIN": 0.15}), {}, _state(session=False), {})
    assert tg == {"AAPL": pytest.approx(MAX_W_NAME * OFF_HOURS_SCALE),
                  "COIN": pytest.approx(MAX_W_NAME * 0.5 * OFF_HOURS_SCALE)}
    assert _rules(acts) == ["name_cap", "off_hours", "off_hours"]
    # a held position is kept off-hours up to the vol cap, not the flat one
    tg, _ = risk.apply(OFF, _dec({"COIN": 0.15}), {"COIN": 0.07}, _state(session=False), {})
    assert tg == {"COIN": pytest.approx(0.07)}
    caps = risk.effective_caps(OFF, _state(session=False))
    assert caps["COIN"] == pytest.approx(MAX_W_NAME * 0.5 * OFF_HOURS_SCALE)
    assert risk.effective_caps(T0, _state())["AAPL"] == pytest.approx(MAX_W_NAME)
    assert risk.effective_caps(T0, _state(), version="v0")["HOOD"] == MAX_W_NAME


# ---- (c) vol-based stop -------------------------------------------------------------------------
def test_stop_level_scales_with_vol_floor_and_cap():
    st = _state()
    assert risk.stop_level(st, "AAPL") == pytest.approx(STOP_K * LOW)            # 4%
    assert risk.stop_level(st, "NVDA") == pytest.approx(STOP_FLOOR)              # 2% -> floor 3%
    assert risk.stop_level(st, "MSTR") == pytest.approx(STOP_CAP)                # 12% -> cap 10%
    assert risk.stop_level(st, "COIN") == pytest.approx(0.08)
    st["AAPL"]["vol_7d"] = None                                                  # no vol -> median vol
    assert risk.stop_level(st, "AAPL") == pytest.approx(STOP_K * LOW)
    assert risk.stop_level({}, "AAPL") == STOP_LOSS_NAME                         # nothing at all


def test_stop_threshold_scales_with_vol(no_beta):
    ls = {}
    risk.apply(T0, _dec({"AAPL": 0.1, "COIN": 0.05, "NVDA": -0.1}), {}, _state(), ls)
    assert ls["entry"]["AAPL"] == {"price": 100.0, "side": 1, "stop": pytest.approx(0.04)}
    assert ls["entry"]["COIN"]["stop"] == pytest.approx(0.08)
    assert ls["entry"]["NVDA"]["stop"] == pytest.approx(0.03)
    book = {"AAPL": 0.1, "COIN": 0.05, "NVDA": -0.1}
    # -3.9% on AAPL (stop 4%), -7% on COIN (stop 8%; v0 would stop at 5%), NVDA short +2.9% (stop 3%)
    px = {"AAPL": 96.1, "COIN": 93.0, "NVDA": 102.9}
    assert risk.check(T0 + H, book, px, ls) == (None, [])
    forced, acts = risk.check(T0 + 2 * H, book, {"AAPL": 95.9, "COIN": 93.0, "NVDA": 103.1}, ls)
    assert forced == {"COIN": 0.05} and _rules(acts) == ["stop_loss", "stop_loss"]
    assert {a["ticker"] for a in acts} == {"AAPL", "NVDA"}
    forced, acts = risk.check(T0 + 3 * H, {"COIN": 0.05}, {"COIN": 91.9}, ls)
    assert forced == {} and _rules(acts) == ["stop_loss"]
    # same move under v0: COIN stops at the flat 5%
    ls0 = {}
    risk.apply(T0, _dec({"COIN": 0.05}), {}, _state(), ls0, version="v0")
    assert ls0["entry"]["COIN"] == {"price": 100.0, "side": 1}
    assert risk.check(T0 + H, {"COIN": 0.05}, {"COIN": 94.9}, ls0, version="v0")[0] == {}


def test_stop_uses_vol_at_entry_not_current(no_beta):
    ls = {}
    risk.apply(T0, _dec({"AAPL": 0.1}), {}, _state(), ls)                      # entry stop 4%
    hot = _state(vols={**VOLS, "AAPL": 0.05})                                   # vol doubles after entry
    tg, acts = risk.apply(T0 + 4 * H, _dec({"AAPL": 0.1}), {"AAPL": 0.1}, {**hot, "AAPL": {**hot["AAPL"], "price": 95.9}}, ls)
    assert tg == {} and _rules(acts) == ["name_cap", "stop_loss"] and "AAPL" in ls["cooldown"]  # (cap: vol 5%)
    tg, acts = risk.apply(T0 + 8 * H, _dec({"AAPL": 0.1}), {}, _state(95.9), ls)
    assert tg == {} and _rules(acts) == ["cooldown"]


def test_topup_reaverages_stop_by_quantity(no_beta):
    ls = {}
    risk.apply(T0, _dec({"COIN": 0.05}), {}, _state(), ls)                      # 0.05 @100, stop 8%
    risk.apply(T0 + 4 * H, _dec({"COIN": 0.075}), {"COIN": 0.05}, _state(125.0, vols={**VOLS, "COIN": LOW}), ls)
    q_old, q_add = 0.05 / 125, 0.025 / 125
    e = ls["entry"]["COIN"]
    assert e["price"] == pytest.approx((q_old * 100 + q_add * 125) / (q_old + q_add))
    assert e["stop"] == pytest.approx((q_old * 0.08 + q_add * 0.04) / (q_old + q_add))
    ls["entry"]["NVDA"] = {"price": 100.0, "side": 1}                           # recorded without a stop (v0)
    assert risk.check(T0 + 5 * H, {"NVDA": 0.1}, {"NVDA": 95.1}, ls) == (None, [])
    assert risk.check(T0 + 6 * H, {"NVDA": 0.1}, {"NVDA": 94.9}, ls)[0] == {}   # falls back to STOP_LOSS_NAME


# ---- (d) beta neutralisation --------------------------------------------------------------------
def test_beta_neutral_closed_form_is_minimal_change():
    rng = np.random.default_rng(7)
    w = {tk: float(x) for tk, x in zip(UNIVERSE, rng.uniform(-0.03, 0.03, len(UNIVERSE)))}
    lo = {tk: -1.0 for tk in UNIVERSE}
    hi = {tk: 1.0 for tk in UNIVERSE}
    out = risk.beta_neutral(w, BETAS, lo, hi)
    b = np.array([BETAS[tk] for tk in UNIVERSE])
    w0 = np.array([w[tk] for tk in UNIVERSE])
    w1 = np.array([out[tk] for tk in UNIVERSE])
    assert b @ w1 == pytest.approx(0.0, abs=1e-15)
    assert w1 == pytest.approx(w0 - b * (b @ w0) / (b @ b), abs=1e-15)       # the spec's closed form
    for _ in range(200):                                                        # any other neutral book is farther
        z = rng.normal(size=len(b)) * 0.01
        other = w1 + z - b * (b @ z) / (b @ b)
        assert b @ other == pytest.approx(0.0, abs=1e-12)
        assert np.linalg.norm(other - w0) >= np.linalg.norm(w1 - w0) - 1e-15


def test_beta_neutral_respects_bounds_kkt():
    w = {"A": 0.10, "B": 0.10, "C": -0.02, "D": 0.0}
    beta = {"A": 1.9, "B": 0.5, "C": 0.6, "D": 1.0}
    lo = {"A": -0.05, "B": -0.15, "C": -0.05, "D": 0.0}                          # D is locked (e.g. stopped)
    hi = {"A": 0.10, "B": 0.15, "C": 0.05, "D": 0.0}
    out = risk.beta_neutral(w, beta, lo, hi)
    assert _bw(out, beta) == pytest.approx(0.0, abs=1e-12)
    assert all(lo[k] - 1e-15 <= out[k] <= hi[k] + 1e-15 for k in w) and out["D"] == 0.0
    free = [k for k in "ABC" if lo[k] + 1e-12 < out[k] < hi[k] - 1e-12]
    lams = {k: (w[k] - out[k]) / beta[k] for k in free}
    lam = next(iter(lams.values()))
    assert all(v == pytest.approx(lam) for v in lams.values())                  # common multiplier
    for k in "ABC":                                                             # bound names: clipped KKT form
        assert out[k] == pytest.approx(max(lo[k], min(hi[k], w[k] - lam * beta[k])))
    # unreachable neutrality: the closest reachable exposure
    out = risk.beta_neutral({"A": 0.1, "B": 0.1}, {"A": 1.0, "B": 1.0}, {"A": 0.05, "B": 0.0}, {"A": 0.1, "B": 0.1})
    assert out == {"A": 0.05, "B": 0.0}


def test_apply_beta_neutral_step_and_actions():
    st = _state(vols={tk: LOW for tk in UNIVERSE}, bet=BETAS)
    tg0 = {"NVDA": 0.08, "MSTR": -0.03, "AAPL": 0.02}
    tg, acts = risk.apply(T0, _dec(tg0), {}, st, {})
    assert _bw(tg, BETAS) == pytest.approx(0.0, abs=1e-12)
    bn = [a for a in acts if a["rule"] == "beta_neutral"]
    assert _rules(acts) == ["beta_neutral"] * len(UNIVERSE)                     # every name moves (all betas != 0)
    assert len({a["ticker"] for a in bn}) == len(bn)
    b = np.array([BETAS[tk] for tk in UNIVERSE])
    w0 = np.array([tg0.get(tk, 0.0) for tk in UNIVERSE])
    want = w0 - b * (b @ w0) / (b @ b)
    assert np.array([tg.get(tk, 0.0) for tk in UNIVERSE]) == pytest.approx(want, abs=1e-15)
    for a in bn:
        assert a["before"] == pytest.approx(tg0.get(a["ticker"], 0.0)) and a["after"] == pytest.approx(tg[a["ticker"]])
    # missing betas -> 1.0: neutral = dollar neutral
    tg, _ = risk.apply(T0, _dec({"NVDA": 0.1}), {}, _state(vols={tk: LOW for tk in UNIVERSE}), {})
    assert sum(tg.values()) == pytest.approx(0.0, abs=1e-12) and tg["NVDA"] == pytest.approx(0.09)
    # an already neutral book is left alone
    tg, acts = risk.apply(T0, _dec({"NVDA": 0.1, "AAPL": -0.1}), {}, _state(vols={tk: LOW for tk in UNIVERSE},
                                                                            bet={"NVDA": 0.5, "AAPL": 0.5}), {})
    assert tg == {"NVDA": 0.1, "AAPL": -0.1} and acts == []


def test_beta_neutral_keeps_caps_and_locked_names():
    ls = {"cooldown": {"TSLA": T0 + 10 * H}}
    st = _state(bet=BETAS)                                                      # HOOD cap 0.0495, COIN 0.075
    st["CRCL"]["price"] = None                                                  # no price: cannot trade -> fixed
    st["META"]["funding_last"] = config.FUNDING_BLOCK                           # longs pay: no new long META
    tg0 = {"AAPL": -0.15, "GOOGL": -0.15, "NVDA": -0.10, "AMZN": -0.15, "CRCL": 0.02}
    book = {"CRCL": 0.02}
    tg, acts = risk.apply(T0, _dec(tg0), book, st, ls)
    caps = risk.vol_caps(st, UNIVERSE)
    assert all(abs(tg.get(tk, 0.0)) <= caps[tk] + 1e-12 for tk in UNIVERSE)
    assert "TSLA" not in tg and tg.get("META", 0.0) <= 0 and tg["CRCL"] == 0.02
    bn = [a for a in acts if a["rule"] == "beta_neutral"]
    assert {a["ticker"] for a in bn}.isdisjoint({"TSLA", "META", "CRCL"})
    assert not [a for a in acts if a["rule"] == "name_cap" and acts.index(a) > acts.index(bn[0])]
    pre_net = [a for a in acts if a["rule"] in ("net_cap", "gross_cap")]
    if not pre_net:                                                            # neutral unless net/gross scaled after
        assert _bw(tg, BETAS) == pytest.approx(0.0, abs=1e-12)
    assert abs(sum(tg.values())) <= MAX_NET + 1e-12 and sum(abs(v) for v in tg.values()) <= MAX_GROSS + 1e-12


def test_beta_neutral_then_net_and_gross_caps():
    st = _state(vols={tk: LOW for tk in UNIVERSE}, bet={tk: (2.0 if i < 5 else 0.2) for i, tk in enumerate(UNIVERSE)})
    tg, acts = risk.apply(T0, _dec({tk: (-0.15 if i < 5 else 0.15) for i, tk in enumerate(UNIVERSE)}), {}, st, {})
    rules = _rules(acts)
    assert rules.index("beta_neutral") < rules.index("net_cap")                 # net cap applies after
    assert abs(sum(tg.values())) == pytest.approx(MAX_NET)
    assert sum(abs(v) for v in tg.values()) <= MAX_GROSS + 1e-12


def test_kill_and_invalid_under_v1():
    st = _state(bet=BETAS)
    ls = {"equity": 9_600.0, "equity_hist": [(T0 - H, 10_000.0)]}
    tg, acts = risk.apply(T0, _dec({"NVDA": 0.1}), {"NVDA": 0.1, "AAPL": -0.1}, st, ls)
    assert tg == {} and "beta_neutral" not in _rules(acts)
    # an invalid decision holds the book; the risk layer still neutralises the held book's beta
    tg, acts = risk.apply(T0, _dec({}, valid=False), {"NVDA": 0.1}, _state(bet=BETAS), {})
    assert _rules(acts)[0] == "invalid" and _bw(tg, BETAS) == pytest.approx(0.0, abs=1e-12)


# ---- (e) v0 path unchanged ----------------------------------------------------------------------
def test_v0_path_unchanged(monkeypatch):
    st = _state(bet=BETAS)
    dec = _dec({"HOOD": 0.4, "COIN": 0.15, "AAPL": -0.05})
    tg, acts = risk.apply(T0, dec, {}, st, {}, version="v0")
    assert tg == {"HOOD": MAX_W_NAME, "COIN": 0.15, "AAPL": -0.05}                # flat cap, no beta step
    assert _rules(acts) == ["name_cap"]
    ls = {}
    risk.apply(T0, _dec({"HOOD": 0.1}), {}, st, ls, version="v0")
    assert ls["entry"]["HOOD"] == {"price": 100.0, "side": 1}                   # no stop recorded
    tg, acts = risk.apply(T0 + 4 * H, _dec({"HOOD": 0.1}), {"HOOD": 0.1}, _state(94.9), ls, version="v0")
    assert tg == {} and _rules(acts) == ["stop_loss"]                           # flat 5% even for a 30% vol name
    monkeypatch.setattr(config, "RISK_VERSION", "v0")                           # the config switch reaches v0
    assert risk.apply(T0, dec, {}, st, {}) == risk.apply(T0, dec, {}, st, {}, version="v0")
    ls = {"entry": {"HOOD": {"price": 100.0, "side": 1, "stop": 0.10}}}
    assert risk.check(T0, {"HOOD": 0.1}, {"HOOD": 94.9}, ls)[0] == {}           # v0 ignores a recorded stop


# ---- (a) point-in-time betas --------------------------------------------------------------------
D0 = pd.Timestamp("2026-05-01", tz="UTC")
TRUE_BETA = {tk: b for tk, b in zip(UNIVERSE, (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 0.9, 1.1, 1.0))}


def _frames(n_days: int, seed: int = 3, noise: float = 0.0) -> pd.DataFrame:
    """Hourly bars flat within each UTC day; day d's close = prev * (1 + beta_i * f_d + noise)."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.02, n_days)
    rows = []
    for tk in UNIVERSE:
        p, rets = 100.0, f * TRUE_BETA[tk] + rng.normal(0, noise, n_days)
        for d in range(n_days):
            p *= 1 + rets[d]
            for h in range(24):
                ts = D0 + pd.Timedelta(days=d, hours=h)
                rows.append({"sym": perp(tk), "ts": int(ts.timestamp() * 1000), "open": p, "high": p, "low": p,
                             "close": p, "quote_vol": 1.0})
    return pd.DataFrame(rows)


def _store(perp_frame) -> Store:
    s = Store.__new__(Store)
    s.source = "test"
    s._init({"perp": perp_frame})
    return s


def test_betas_recover_true_beta_to_the_basket():
    s = _store(_frames(75))
    t = D0 + pd.Timedelta(days=70, hours=13)
    got = betas(s, t)
    mean = np.mean(list(TRUE_BETA.values()))
    for tk in UNIVERSE:                                                         # no noise: exact up to fp
        assert got[tk] == pytest.approx(TRUE_BETA[tk] / mean, rel=1e-9)
    st = market_state(s, t)
    assert {tk: st[tk]["beta_60d"] for tk in UNIVERSE} == got
    assert {"price", "ret_4h", "ret_24h", "ret_7d", "vol_7d", "funding_last", "funding_7d_mean",
            "spot_premium", "us_session_open"} <= set(st["NVDA"])               # backward compatible


def test_betas_min_history_and_completed_days_only():
    s = _store(_frames(75))
    assert set(betas(s, D0 + pd.Timedelta(days=25)).values()) == {1.0}         # < 30 daily returns
    t = D0 + pd.Timedelta(days=31)                                              # 31 closes -> exactly 30 returns
    assert len(daily_closes(s, perp("NVDA"), t, 61)) == 31
    assert betas(s, t)["MSTR"] != 1.0
    assert set(betas(s, D0 + pd.Timedelta(days=30, hours=23)).values()) == {1.0}  # 29 returns (day 30 in progress)
    t = D0 + pd.Timedelta(days=70, hours=13)
    dc = daily_closes(s, perp("NVDA"), t, 61)
    assert dc.index[-1] == D0 + pd.Timedelta(days=69) and len(dc) == 61          # today (in progress) excluded


def test_betas_point_in_time():
    fr = _frames(75, noise=0.01)
    s = _store(fr)
    t = D0 + pd.Timedelta(days=66, hours=9)
    ms = int(t.timestamp() * 1000)
    cut = _store(fr[fr.ts + 3_600_000 <= ms])                                   # delete bars closing after t
    assert betas(cut, t) == betas(s, t)
    assert market_state(cut, t) == market_state(s, t)
    wild = fr.copy()                                                            # rewrite today's bars and the future
    day0 = int(t.floor("D").timestamp() * 1000)
    wild.loc[wild.ts >= day0, "close"] *= np.where(wild.loc[wild.ts >= day0, "sym"] == "MSTRUSDT", 3.0, 0.5)
    assert betas(_store(wild), t) == betas(s, t)                                # only completed days count
    first = t.floor("D") - pd.Timedelta(days=61)                                # the window's first close day
    trimmed = _store(fr[fr.ts >= int(first.timestamp() * 1000)])
    assert betas(trimmed, t) == pytest.approx(betas(s, t), rel=1e-12)          # older data is not used
    shorter = _store(fr[fr.ts >= int((first + pd.Timedelta(days=1)).timestamp() * 1000)])
    assert betas(shorter, t) != betas(s, t)                                     # ...but the window's first day is


def test_state_to_risk_end_to_end_is_beta_neutral():
    s = _store(_frames(75, noise=0.01))
    t = D0 + pd.Timedelta(days=70, hours=16)                                    # Sunday 16:00: off-hours
    st = market_state(s, t)
    tg, acts = risk.apply(t, _dec({"MSTR": 0.15, "AAPL": 0.15, "NVDA": -0.05}), {}, st, {})
    bet = {tk: st[tk]["beta_60d"] for tk in UNIVERSE}
    caps = risk.effective_caps(t, st)
    assert all(abs(tg.get(tk, 0.0)) <= caps[tk] + 1e-12 for tk in UNIVERSE)
    if not any(a["rule"] in ("net_cap", "gross_cap") for a in acts):
        assert _bw(tg, bet) == pytest.approx(0.0, abs=1e-12)
    assert "beta_neutral" in _rules(acts)
