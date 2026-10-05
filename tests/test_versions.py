"""Risk/prompt versions end to end: the v0 path reproduces the pre-v1 record format and prompt (frozen
decide.md), the version is selectable without editing code, the v1 beta step keeps the book beta-neutral
when the net cap binds (net cap inside the projection), off-hours held caps reach the prompt, and the
snapshot carries enough perp history for a full 60-day beta from the first replay decision."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from sentiment import agent, config, features, llm, replay, risk
from sentiment.config import MAX_GROSS, MAX_NET, MAX_W_NAME, OFF_HOURS_SCALE, PROMPTS, ROOT, UNIVERSE
from tests.test_replay import START, FakeStore, _state, fake_cards

H = pd.Timedelta(hours=1)
T0 = pd.Timestamp("2026-07-01 14:00", tz="UTC")          # Wednesday, US session open
DECIDE_V0_SHA256 = "773497d5b3411bf1896ffff7de96383fb1348e1c154ffa93eefe9b13b03019cc"   # decide.md at 94cbe51
V0_KEYS = {"ts", "mode", "run_id", "variant", "event", "input_hash", "evidence_ids", "llm_request_hash", "decision",
           "error", "baseline_targets", "risk_actions", "targets", "fills", "skipped", "state", "marks", "equity",
           "prev_hash", "hash"}
V1_KEYS = V0_KEYS | {"risk_ctx", "risk_ctx_hash", "risk_version", "prompt_version"}
REPLY = '{"targets": {"NVDA": 0.1, "AAPL": -0.1, "MSTR": 0.05}, "reasons": {}, "evidence": {}, "confidence": 0.5}'


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    monkeypatch.setattr(llm, "model_name", lambda: "fake-model")
    prompts = []

    def complete(messages, *, purpose):
        prompts.append(messages)
        return REPLY
    monkeypatch.setattr(llm, "complete", complete)
    return prompts


def _recs(run):
    return [json.loads(ln) for ln in (run / "log.jsonl").read_text().splitlines()]


# ---- prompt versions --------------------------------------------------------------------------------
def test_v0_prompt_file_is_frozen():
    """decide.md is the v0 prompt every pre-v1 run (and any process still running that code) reads."""
    assert hashlib.sha256(agent.PROMPT_FILES["v0"].read_bytes()).hexdigest() == DECIDE_V0_SHA256
    assert agent.PROMPT_FILES["v0"] == ROOT / "sentiment" / "prompts" / "decide.md" == PROMPTS / "decide.md"
    v0 = agent.system_prompt(version="v0")
    assert "{{" not in v0 and "RISK" not in v0 and "new exposure is capped at half" in v0
    v1 = agent.system_prompt()
    assert "{{" not in v1 and "BETA-NEUTRALISES" in v1 and "cap_held" in v1


def test_prompt_version_follows_risk_ctx(fake_llm):
    st = _state()
    no = agent.build_messages(T0, st, [], {}, "llm")
    assert no[0]["content"] == agent.system_prompt(version="v0") and "RISK (" not in no[1]["content"]
    d = agent.decide(T0, st, [], {}, "llm")
    assert d["prompt_version"] == "v0" and fake_llm[-1] == no
    ctx = {"caps": {tk: 0.15 for tk in UNIVERSE}, "beta": {}, "book_net": 0.0, "book_beta_net": 0.0}
    d = agent.decide(T0, st, [], {}, "llm", risk_ctx=ctx)
    assert d["prompt_version"] == "v1" and fake_llm[-1][0]["content"] == agent.system_prompt(version="v1")


# ---- v0 record format ------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["baseline", "llm"])
def test_v0_run_writes_the_pre_v1_record_format(tmp_path, monkeypatch, fake_llm, variant):
    monkeypatch.setattr(config, "RISK_VERSION", "v0")
    out = replay.run(variant, "2026-07-01", "2026-07-01", store=FakeStore(), out_dir=tmp_path, cards_fn=fake_cards,
                     quiet=True)
    recs = _recs(out)
    assert recs and all(set(r) == V0_KEYS for r in recs)
    dec = [r for r in recs if r["event"] == "decision"]
    assert dec and all(not set(r["state"][tk]) & {"beta_60d", "beta_days"} for r in dec for tk in UNIVERSE)
    if variant == "llm":
        assert fake_llm and all(m[0]["content"] == agent.system_prompt(version="v0") for m in fake_llm)
        assert not any("RISK (" in m[1]["content"] or "since_bp" in m[1]["content"] for m in fake_llm)
    m = json.loads((out / "metrics.json").read_text())
    assert m["risk_version"] == "v0" and m["prompt_version"] == (None if variant == "baseline" else "v0")
    assert m["n_decisions_beta_fallback"] is None


def test_v0_state_has_no_beta_v1_state_has_beta_and_days(monkeypatch):
    st = FakeStore()
    t = START + 16 * H
    s0 = features.market_state(st, t, version="v0")
    s1 = features.market_state(st, t, version="v1")
    assert "beta_60d" not in s0["NVDA"] and {"beta_60d", "beta_days"} <= set(s1["NVDA"])
    assert {k: {f: v for f, v in s1[k].items() if f not in ("beta_60d", "beta_days")} for k in s1} == s0
    monkeypatch.setattr(config, "RISK_VERSION", "v0")
    assert features.market_state(st, t) == s0                                  # default follows the config


def test_v1_run_logs_versions_and_beta_fallback_counts(tmp_path, monkeypatch, fake_llm):
    monkeypatch.setattr(config, "RISK_VERSION", "v1")
    out = replay.run("llm", "2026-07-01", "2026-07-01", store=FakeStore(), out_dir=tmp_path, cards_fn=fake_cards,
                     quiet=True)
    recs = _recs(out)
    dec = [r for r in recs if r["event"] == "decision"]
    assert all(set(r) == V1_KEYS for r in recs)
    assert all(r["risk_version"] == "v1" and r["prompt_version"] == "v1" and r["risk_ctx"] for r in dec)
    # FakeStore has 9 days of bars before the window: every beta is the 1.0 fallback, and the metrics say so
    assert all(r["state"][tk]["beta_days"] < config.BETA_MIN_D for r in dec for tk in UNIVERSE)
    m = json.loads((out / "metrics.json").read_text())
    assert m["n_decisions_beta_fallback"] == m["n_decisions_beta_short"] == len(dec) == m["n_decisions"]


def test_risk_version_is_selectable_without_editing_code(tmp_path, monkeypatch):
    seen = []

    def fake_run(variant, start, end, **kw):
        seen.append(config.RISK_VERSION)
        d = tmp_path / "r"
        d.mkdir(exist_ok=True)
        (d / "metrics.json").write_text("{}")
        return d
    monkeypatch.setattr(replay, "run", fake_run)
    monkeypatch.setattr(config, "RISK_VERSION", "v1")
    replay.main(["--variant", "baseline", "--start", "2026-07-01", "--end", "2026-07-01", "--risk-version", "v0"])
    assert seen == ["v0"]
    with pytest.raises(SystemExit):
        replay.main(["--start", "2026-07-01", "--end", "2026-07-01", "--risk-version", "v9"])
    env = {**os.environ, "SENTIMENT_RISK_VERSION": "v0"}
    got = subprocess.run([sys.executable, "-c", "from sentiment import config; print(config.RISK_VERSION)"],
                         capture_output=True, text=True, env=env, cwd=ROOT, check=True).stdout.strip()
    assert got == "v0"


def test_run_rejects_unknown_version(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RISK_VERSION", "v9")
    with pytest.raises(ValueError, match="unknown risk version"):
        replay.run("baseline", "2026-07-01", "2026-07-01", store=FakeStore(), out_dir=tmp_path, cards_fn=fake_cards,
                   quiet=True)


# ---- net cap inside the beta projection ---------------------------------------------------------------
LIVE_BETAS = {"AAPL": 0.04, "AMZN": 0.25, "GOOGL": 0.22, "META": 0.47, "NVDA": 0.51, "TSLA": 0.74, "COIN": 1.88,
              "MSTR": 2.10, "CRCL": 2.02, "HOOD": 1.76}


def _rstate(bet, vol=0.03, session=True):
    return {tk: {"price": 100.0, "funding_last": 0.0, "us_session_open": session, "vol_7d": vol,
                 "beta_60d": bet.get(tk)} for tk in UNIVERSE}


def _dec(tg):
    return {"targets": tg, "reasons": {}, "evidence": {}, "confidence": 0.5, "raw": "", "valid": True}


def _bw(w, bet):
    return sum(bet.get(tk, 1.0) * v for tk, v in w.items())


def test_net_cap_binding_keeps_the_book_beta_neutral():
    """Regression: the net cap used to scale only the dominant side after the beta step, which re-created
    (here tripled) the beta exposure the step had removed."""
    tg0 = {**{tk: 0.15 for tk in ("AAPL", "AMZN", "GOOGL", "META", "NVDA", "TSLA")}, "MSTR": -0.10, "COIN": -0.10}
    tg, acts = risk.apply(T0, _dec(tg0), {}, _rstate(LIVE_BETAS), {}, version="v1")
    rules = [a["rule"] for a in acts]
    assert "beta_neutral" in rules and "net_cap" in rules
    assert max(i for i, r in enumerate(rules) if r == "beta_neutral") < rules.index("net_cap")
    assert abs(_bw(tg, LIVE_BETAS)) <= 1e-9                                  # was -0.172 (raw decision: -0.064)
    assert abs(sum(tg.values())) <= MAX_NET + 1e-12
    assert abs(sum(tg.values())) == pytest.approx(MAX_NET)                  # the cap binds, no more than needed
    assert sum(abs(v) for v in tg.values()) <= MAX_GROSS + 1e-12
    assert all(abs(v) <= MAX_W_NAME + 1e-12 for v in tg.values())
    for a in (a for a in acts if a["rule"] == "net_cap"):                   # per-name, chained after the beta step
        assert a["ticker"] in UNIVERSE and a["after"] == pytest.approx(tg.get(a["ticker"], 0.0))


def _qp(w, beta, lo, hi, max_net):
    """Reference solution of the same projection with a generic solver."""
    from scipy.optimize import minimize
    names = sorted(w)
    w0, b = np.array([w[k] for k in names]), np.array([beta[k] for k in names])
    cons = [{"type": "eq", "fun": lambda x: b @ x, "jac": lambda x: b},
            {"type": "ineq", "fun": lambda x: max_net - x.sum(), "jac": lambda x: -np.ones_like(x)},
            {"type": "ineq", "fun": lambda x: max_net + x.sum(), "jac": lambda x: np.ones_like(x)}]
    r = minimize(lambda x: ((x - w0) ** 2).sum(), np.clip(np.zeros_like(w0), [lo[k] for k in names],
                                                        [hi[k] for k in names]),
                 jac=lambda x: 2 * (x - w0), bounds=[(lo[k], hi[k]) for k in names], constraints=cons,
                 method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
    assert r.success, r.message
    return float(((r.x - w0) ** 2).sum())


def test_joint_projection_is_the_constrained_minimum():
    rng = np.random.default_rng(11)
    n_bound = 0
    for i in range(60):
        # megacaps (first 6) low beta and mostly long, crypto names high beta and mostly short: the book shape
        # where neutralising pushes the net up (every other case fully random)
        lowb = rng.uniform(0.0, 0.8, 6) if i % 2 else rng.uniform(0.0, 2.5, 6)
        bet = {tk: float(x) for tk, x in zip(UNIVERSE, [*lowb, *rng.uniform(1.4, 2.5, 4)])}
        caps = {tk: float(x) for tk, x in zip(UNIVERSE, rng.uniform(0.05, 0.15, len(UNIVERSE)))}
        raw = [*rng.uniform(0.0, 0.15, 6), *rng.uniform(-0.15, 0.03, 4)] if i % 2 else rng.normal(0.05, 0.1, 10)
        w = {tk: float(np.clip(x, -caps[tk], caps[tk])) for tk, x in zip(UNIVERSE, raw)}
        lo, hi = {tk: -c for tk, c in caps.items()}, dict(caps)
        out = risk.beta_neutral(w, bet, lo, hi, max_net=MAX_NET)
        assert abs(_bw(out, bet)) <= 1e-9 and abs(sum(out.values())) <= MAX_NET + 1e-12
        assert all(lo[k] - 1e-15 <= out[k] <= hi[k] + 1e-15 for k in w)
        plain = risk.beta_neutral(w, bet, lo, hi)
        n_bound += abs(sum(plain.values())) > MAX_NET
        obj = sum((out[k] - w[k]) ** 2 for k in w)
        assert obj <= _qp(w, bet, lo, hi, MAX_NET) + 1e-9                   # no feasible book is closer
    assert n_bound >= 10                                                      # the net cap was exercised


def test_net_cap_without_binding_leaves_the_plain_projection():
    bet = {"NVDA": 0.5, "AAPL": 0.5, "MSTR": 2.0}
    w = {"NVDA": 0.1, "AAPL": 0.05, "MSTR": -0.02}
    lo, hi = {k: -0.15 for k in w}, {k: 0.15 for k in w}
    assert risk.beta_neutral(w, bet, lo, hi, max_net=MAX_NET) == risk.beta_neutral(w, bet, lo, hi)


def test_random_v1_books_meet_every_limit_and_stay_neutral():
    rng = np.random.default_rng(5)
    for i in range(40):
        bet = {tk: float(x) for tk, x in zip(UNIVERSE, rng.uniform(0.0, 2.3, len(UNIVERSE)))}
        vols = {tk: float(x) for tk, x in zip(UNIVERSE, rng.uniform(0.01, 0.08, len(UNIVERSE)))}
        st = {tk: {"price": 100.0, "funding_last": 0.0, "us_session_open": bool(i % 2), "vol_7d": vols[tk],
                   "beta_60d": bet[tk]} for tk in UNIVERSE}
        dec = _dec({tk: float(x) for tk, x in zip(UNIVERSE, rng.normal(0.04, 0.12, len(UNIVERSE)))})
        tg, _ = risk.apply(T0, dec, {}, st, {}, version="v1")
        caps = risk.effective_caps(T0 if i % 2 else T0 - 8 * H, st, version="v1")
        assert all(abs(tg.get(tk, 0.0)) <= caps[tk] + 1e-12 for tk in UNIVERSE)
        assert abs(sum(tg.values())) <= MAX_NET + 1e-12 and sum(map(abs, tg.values())) <= MAX_GROSS + 1e-12
        assert abs(_bw(tg, bet)) <= 1e-9


# ---- off-hours held caps in the risk context and prompt ------------------------------------------------
def test_off_hours_context_shows_the_cap_a_held_position_keeps(monkeypatch):
    monkeypatch.setattr(config, "RISK_VERSION", "v1")
    st = FakeStore()
    t = START + 2 * H                                                        # 02:00 UTC: US session closed
    state = features.market_state(st, t)
    ctx = replay.risk_context(t, store=st, state=state, book={"NVDA": 0.1}, lstate={}, cards=[])
    full = risk.vol_caps(state, UNIVERSE)
    assert ctx["us_session_open"] is False
    for tk in UNIVERSE:
        assert ctx["caps"][tk] == pytest.approx(full[tk] * OFF_HOURS_SCALE, abs=1e-4)
        assert ctx["caps_held"][tk] == pytest.approx(full[tk], abs=1e-4)
    # the risk layer really keeps a held same-side position above `cap`, up to `caps_held`
    tk = "NVDA"
    held = min(ctx["caps_held"][tk], ctx["caps"][tk] * 1.5)
    tg, acts = risk.apply(t, _dec({tk: held}), {tk: held}, {k: v for k, v in state.items() if k != "_market"}, {},
                          version="v1")
    assert not [a for a in acts if a["rule"] in ("off_hours", "name_cap") and a["ticker"] == tk]
    u = agent.build_messages(t, state, [], {tk: 0.1}, "llm", ctx)[1]["content"]
    assert "US session closed: cap limits NEW exposure only" in u and f"NVDA {ctx['caps_held']['NVDA']:.3f}" in u
    b = agent.build_messages(t, state, [], {tk: 0.1}, "blinded", ctx)[1]["content"]
    assert f"{agent.ALIAS['NVDA']} {ctx['caps_held']['NVDA']:.3f}" in b and "NVDA" not in b
    on = replay.risk_context(START + 16 * H, store=st, state=features.market_state(st, START + 16 * H),
                             book={}, lstate={}, cards=[])
    assert on["us_session_open"] is True and on["caps_held"] == on["caps"]
    assert "US session closed" not in agent.build_messages(START + 16 * H, _state(), [], {}, "llm", on)[1]["content"]


def test_risk_context_version_argument(monkeypatch):
    monkeypatch.setattr(config, "RISK_VERSION", "v1")
    st = FakeStore()
    t = START + 16 * H
    state = features.market_state(st, t)
    ls = {"entry": {"NVDA": {"price": 100.0, "side": 1, "stop": 0.07}}}
    c0 = replay.risk_context(t, store=st, state=state, book={"NVDA": 0.1}, lstate=ls, cards=[], version="v0")
    c1 = replay.risk_context(t, store=st, state=state, book={"NVDA": 0.1}, lstate=ls, cards=[])
    assert c0["risk_version"] == "v0" and c0["entry"]["NVDA"]["stop"] == config.STOP_LOSS_NAME
    assert set(c0["caps"].values()) == {MAX_W_NAME}
    assert c1["risk_version"] == "v1" and c1["entry"]["NVDA"]["stop"] == 0.07


# ---- snapshot history ---------------------------------------------------------------------------------
def test_snapshot_has_a_full_beta_history_at_the_first_decision():
    from sentiment.data import Store
    s = Store(snapshot=True)
    if s.source != "snapshot":
        pytest.skip("no snapshot")
    t = pd.Timestamp(config.REPLAY_START, tz="UTC")
    _, n = features.beta_stats(s, t)
    assert set(n.values()) == {config.BETA_LOOKBACK_D}                      # no 1.0 fallback, no short window
    from sentiment import snapshot
    assert snapshot.check()
