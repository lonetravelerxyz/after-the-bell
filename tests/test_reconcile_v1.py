"""Reconciliation under risk v1 and across versions: the v1 risk context is rebuilt point-in-time (never
copied from the record) so a wrong logged context is a mismatch; each record is judged under its own risk
version, so v0 records reconcile exactly while the config default is v1 (state, prompt, risk rebuild and
hourly stops); a log that switches from v0 to v1 reconciles too."""
import json

import pandas as pd
import pytest

from sentiment import config, ledger, reconcile, replay
from tests.test_reconcile import CrashStore, _decisions, _run, cached_llm, item_cards  # noqa: F401
from tests.test_replay import FakeStore

H = pd.Timedelta(hours=1)
T_TAMPER = pd.Timestamp("2026-07-01 08:00", tz="UTC")     # off-hours, cards in the lookback


@pytest.fixture
def v1(monkeypatch):
    monkeypatch.setattr(config, "RISK_VERSION", "v1")


@pytest.mark.parametrize("variant", ["llm", "blinded", "nonews", "baseline"])
def test_v1_relabelled_replay_reconciles_with_rebuilt_risk_ctx(cached_llm, monkeypatch, v1, variant):
    run = _run(cached_llm, variant)
    live = reconcile.fabricate(run, cached_llm / "live")
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards, out_dir=cached_llm / "o")
    assert s["risk_versions"] == {"v1": 12} and s["n_risk_ctx_checked"] == 12 and s["n_risk_ctx_mismatch"] == 0
    assert s["risk_ctx_identity_rate"] == 1.0 and not reconcile.self_check_failures(s), \
        reconcile.self_check_failures(s)
    d = next(iter(_decisions(cached_llm / "o").values()))
    assert d["risk_ctx_identical"] and d["risk_ctx_diffs"] == [] and d["risk_version"] == "v1"


def _tamper(kind):
    def edit(ctx, t):
        if kind == "beta":
            ctx["beta"] = {tk: 9.99 for tk in ctx["beta"]}
        elif kind == "since":
            ctx["since_card"] = {cid: {k: 0.5 for k in mv} for cid, mv in ctx["since_card"].items()}
        elif kind == "cooldown":
            ctx["cooldown_until"] = {**ctx["cooldown_until"], "NVDA": (t + 10 * H).isoformat()}
        elif kind == "kill":
            ctx["kill_until"] = (t + 10 * H).isoformat()
        elif kind == "caps":
            ctx["caps"] = {tk: 0.15 for tk in ctx["caps"]}
        return ctx
    return edit


@pytest.mark.parametrize("kind", ["beta", "since", "cooldown", "kill", "caps"])
def test_wrong_live_risk_ctx_is_a_mismatch(cached_llm, monkeypatch, v1, kind):
    """A live loop that showed the model a context from leaky / different bars or a wrong cap, cooldown or
    kill: the logged context and prompt are self-consistent, so copying the context (the old code) passed
    prompt and decision identity. The rebuilt context exposes it."""
    real = replay.risk_context
    edit = _tamper(kind)

    def buggy(t, **kw):
        ctx = real(t, **kw)
        return edit(ctx, t) if pd.Timestamp(t) == T_TAMPER else ctx
    monkeypatch.setattr(replay, "risk_context", buggy)
    run = _run(cached_llm)                                  # the "live" run: its T_TAMPER prompt showed the bad ctx
    monkeypatch.setattr(replay, "risk_context", real)
    recs = reconcile.load_log(run / "log.jsonl", modes=None)
    r = next(r for r in recs if pd.Timestamp(r["ts"]) == T_TAMPER)
    assert r["risk_ctx"]["since_card"] and reconcile.live_messages(r["llm_request_hash"])   # cached, cards shown
    live = reconcile.fabricate(run, cached_llm / "live")
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards, out_dir=cached_llm / "o")
    dec = _decisions(cached_llm / "o")
    d = dec[T_TAMPER]
    assert d["risk_ctx_identical"] is False and "risk_ctx" in d["mismatch_inputs"]
    field = {"beta": "beta.", "since": "since_card.", "cooldown": "cooldown_until.NVDA", "kill": "kill_until",
             "caps": "caps."}[kind]
    assert any(x["field"].startswith(field) for x in d["risk_ctx_diffs"]), d["risk_ctx_diffs"]
    assert d["prompt_identical"] is False and d["decision_identical"] is not True
    assert s["n_risk_ctx_mismatch"] == 1 and s["mismatch_inputs"]["risk_ctx"] == 1
    assert any(f.startswith("n_risk_ctx_mismatch") for f in reconcile.self_check_failures(s))
    assert all(x["risk_ctx_identical"] for t, x in dec.items() if t != T_TAMPER)


def test_ctx_diffs_tolerance_and_missing_ctx():
    a = {"entry": {"NVDA": {"price": 361.6385124264642, "pnl": 0.01}}, "caps": {"NVDA": 0.15}}
    b = {"entry": {"NVDA": {"price": 361.6385124264643, "pnl": 0.01}}, "caps": {"NVDA": 0.15}}
    assert reconcile.ctx_diffs(a, b) == []                                  # last-ulp entry price
    assert [x["field"] for x in reconcile.ctx_diffs(a, {**b, "caps": {"NVDA": 0.1}})] == ["caps.NVDA"]
    assert reconcile.ctx_diffs(None, b)[0]["field"] == "risk_ctx"
    assert reconcile.record_version({}) == "v0" and reconcile.record_version({"risk_version": "v1"}) == "v1"


# ---- v0 records while the default is v1 ----------------------------------------------------------------
def _v0_run(tmp_path, monkeypatch, variant="baseline", store=None, **kw):
    monkeypatch.setattr(config, "RISK_VERSION", "v0")
    run = replay.run(variant, "2026-07-01", "2026-07-02", store=store or FakeStore(), out_dir=tmp_path / "run",
                     cards_fn=item_cards, quiet=True, **kw)
    monkeypatch.setattr(config, "RISK_VERSION", "v1")      # the reconciler runs with the new default
    return run


@pytest.mark.parametrize("variant", ["baseline", "llm"])
def test_v0_records_reconcile_exactly_under_the_v1_default(cached_llm, monkeypatch, variant):
    run = _v0_run(cached_llm, monkeypatch, variant)
    assert all("risk_version" not in r for r in reconcile.load_log(run / "log.jsonl", modes=None))
    live = reconcile.fabricate(run, cached_llm / "live")
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards, out_dir=cached_llm / "o")
    assert s["risk_versions"] == {"v0": 12} and s["n_risk_ctx_checked"] == 0
    assert s["risk_rebuild_exact_rate"] == 1.0 and not s["mismatch_inputs"]
    assert not reconcile.self_check_failures(s), reconcile.self_check_failures(s)
    for d in _decisions(cached_llm / "o").values():
        assert d["input_identical"] and not d["state_diffs"] and d["risk_version"] == "v0"


def test_v0_stops_are_not_judged_by_v1_thresholds(tmp_path, monkeypatch):
    """A 3.5% drop on AMZN: under v0 (flat 5% stop) no exit; the v1 stop of a ~2% daily-vol name (~4%) does
    fire. The reconciler must use the record's version, not report replay-only exits."""
    store = CrashStore(drop=0.035)
    run = _v0_run(tmp_path, monkeypatch, store=store)
    assert not [r for r in reconcile.load_log(run / "log.jsonl", modes=None) if r["event"] == "risk_exit"]
    s = reconcile.reconcile(reconcile.fabricate(run, tmp_path / "live"), store=CrashStore(drop=0.035),
                            cards_fn=item_cards)
    assert s["exit_identity_rate"] == 1.0 and s["n_replay_only_exits"] == 0
    # the same path under v1 thresholds (what the old reconciler did) stops the name
    monkeypatch.setattr(config, "RISK_VERSION", "v1")
    v1 = replay.run("baseline", "2026-07-01", "2026-07-02", store=CrashStore(drop=0.035), out_dir=tmp_path / "v1",
                    cards_fn=item_cards, quiet=True)
    assert [r for r in reconcile.load_log(v1 / "log.jsonl", modes=None) if r["event"] == "risk_exit"]


def test_log_switching_from_v0_to_v1_reconciles(tmp_path, monkeypatch):
    """The live loop restarted on the new code: v0 records (shadow book) then v1 records (demo book)."""
    monkeypatch.setattr(config, "RISK_VERSION", "v0")
    a = replay.run("baseline", "2026-07-01", "2026-07-02", store=FakeStore(), out_dir=tmp_path / "a",
                   cards_fn=item_cards, quiet=True)
    monkeypatch.setattr(config, "RISK_VERSION", "v1")
    b = replay.run("baseline", "2026-07-02", "2026-07-02", store=FakeStore(), out_dir=tmp_path / "b",
                   cards_fn=item_cards, quiet=True)
    cut = pd.Timestamp("2026-07-02", tz="UTC")
    ra = [r for r in reconcile.load_log(a / "log.jsonl", modes=None) if pd.Timestamp(r["ts"]) < cut]
    rb = reconcile.load_log(b / "log.jsonl", modes=None)
    live = tmp_path / "live"
    live.mkdir()
    lg = ledger.Ledger(live / "log.jsonl")
    for rs, mode in ((ra, "shadow"), (rb, "demo")):
        for r in rs:
            lg.append({**r, "run_id": f"live_baseline_{mode}", "mode": "live"})
    ea, eb = pd.read_csv(a / "equity.csv"), pd.read_csv(b / "equity.csv")
    pd.concat([ea[pd.to_datetime(ea["ts"]) < cut], eb]).to_csv(live / "equity.csv", index=False)
    s = reconcile.reconcile(live, store=FakeStore(), cards_fn=item_cards, out_dir=tmp_path / "o")
    assert s["risk_versions"] == {"v0": 6, "v1": 6} and s["n_risk_ctx_checked"] == 6
    assert s["decision_identity_rate"] == 1.0 and s["risk_rebuild_exact_rate"] == 1.0
    assert s["n_risk_ctx_mismatch"] == 0 and not s["mismatch_inputs"], s["mismatch_inputs"]
    out = json.loads((tmp_path / "o" / "reconcile.json").read_text())
    assert {d["risk_version"] for d in out["decisions"]} == {"v0", "v1"}
