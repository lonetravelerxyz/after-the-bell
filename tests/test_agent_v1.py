"""Agent prompt v1: risk context in the prompt (RISK table + since_bp card column), no identity leak in the
blinded variant, point-in-time risk context, versions and risk_ctx in the log, --cards-from pinning,
and an offline replay smoke with a fake LLM (no Qwen calls)."""
import hashlib
import json
import re

import pandas as pd
import pytest

from sentiment import agent, config, evidence, ledger, llm, replay, risk
from sentiment.config import UNIVERSE, perp
from sentiment.evidence import NAMES
from tests.test_replay import BLIND_LEAK, START, FakeStore, _state, card, fake_cards

H = pd.Timedelta(hours=1)
T = pd.Timestamp("2026-07-01 16:00", tz="UTC")          # Wednesday, US session open
ENTRY_PX = 123.4567


def _ctx(**over):
    ctx = {"t": T.isoformat(), "risk_version": "v1", "caps": {tk: 0.15 for tk in UNIVERSE} | {"MSTR": 0.048},
           "caps_source": "test", "beta": {tk: 1.0 for tk in UNIVERSE} | {"MSTR": 1.83, "AAPL": 0.41},
           "book": {"NVDA": 0.1, "MSTR": -0.05}, "book_net": 0.05, "book_beta_net": 0.0085,
           "ret_24h": {tk: -0.02 for tk in UNIVERSE},
           "cooldown_until": {"COIN": (T + 7 * H).isoformat()}, "kill_until": None,
           "entry": {"NVDA": {"price": ENTRY_PX, "side": 1, "stop": 0.06, "pnl": 0.0123, "to_stop": 0.0723}},
           "since_card": {"n1-0": {"NVDA": 0.0035}, "m1-0": {"_market": -0.0012}, "t1-0": {"TSLA": None}}}
    ctx.update(over)
    return ctx


CARDS = [card("n1-0", T - 5 * H, 2, 0.7, summary="NVIDIA beat estimates on July 30, 2026"),
         card("m1-0", T - 2 * H, -1, 0.4, scope="market", summary="Fed hike fears"),
         card("t1-0", T - 1 * H, 1, 0.5, tickers=("TSLA",), summary="Tesla deliveries")]


def _user(variant, ctx):
    return agent.build_messages(T, _state(), agent.select_cards(T, CARDS), {"NVDA": 0.1, "MSTR": -0.05},
                                variant, ctx)[1]["content"]


# ---- prompt ---------------------------------------------------------------------------------------------
def test_prompt_contains_risk_table_and_since_column():
    u = _user("llm", _ctx())
    assert "RISK (applied after you" in u
    assert "book net +0.050; beta-weighted net +0.009" in u
    assert "ticker beta cap cooldown_h pnl_entry to_stop" in u
    rows = {ln.split()[0]: ln.split() for ln in u.splitlines() if ln.split() and ln.split()[0] in UNIVERSE}
    assert rows["NVDA"][-5:] == ["1.00", "0.150", "-", "+1.23", "7.23"]
    assert rows["MSTR"][-5:] == ["1.83", "0.048", "-", "-", "-"]            # held short, no entry recorded
    assert rows["COIN"][-5:] == ["1.00", "0.150", "7", "-", "-"]            # 7h of cooldown left
    assert "id | age_h | since_bp | scope subject" in u
    assert "n1-0 | 5 | NVDA +35 | name NVDA" in u
    assert "m1-0 | 2 | mkt -12 | market" in u
    assert "t1-0 | 1 | - | name TSLA" in u                                  # no price -> '-'
    assert str(ENTRY_PX) not in u and "123.4" not in u                     # entry level never in the prompt
    assert "DAILY KILL" not in u
    k = _user("llm", _ctx(kill_until=(T + 10 * H).isoformat()))
    assert "DAILY KILL in force for 10h more" in k
    sp = agent.system_prompt()
    assert "BETA-NEUTRALISES" in sp and "RELATIVE views" in sp and "since_bp" in sp and "{{" not in sp
    assert f"{config.DAILY_KILL * 100:g}% drawdown" in sp


def test_prompt_without_risk_ctx_has_no_risk_block():
    u = agent.build_messages(T, _state(), agent.select_cards(T, CARDS), {}, "llm")[1]["content"]
    assert "RISK (" not in u and "since_bp" not in u
    assert "id | age_h | scope subject" in u


def test_nonews_prompt_keeps_risk_table():
    u = agent.build_messages(T, _state(), [], {"NVDA": 0.1}, "nonews", _ctx(since_card={}))[1]["content"]
    assert "ticker beta cap cooldown_h pnl_entry to_stop" in u and "EVIDENCE: not provided" in u


def test_blinded_risk_fields_leak_no_identity_dates_or_levels():
    ctx = _ctx(kill_until=(T + 10 * H).isoformat())
    u = _user("blinded", ctx)
    assert not BLIND_LEAK.search(u), BLIND_LEAK.search(u)
    for tk in UNIVERSE:
        for w in [tk, *NAMES[tk], *agent.IDENTITY[tk]]:
            assert not re.search(rf"\b{re.escape(w)}\b", u), w
    assert str(ENTRY_PX) not in u and "123.4" not in u
    assert T.isoformat()[:10] not in u and ":00" not in u                  # no dates, no clock times
    a = agent.ALIAS
    rows = {ln.split()[0]: ln.split() for ln in u.splitlines() if ln.split() and ln.split()[0] in a.values()}
    assert rows[a["MSTR"]][1:3] == ["1.83", "0.048"] and rows[a["COIN"]][3] == "7"
    assert f"n1-0 | 5 | {a['NVDA']} +35 | name {a['NVDA']}" in u and "| mkt -12 |" in u
    assert "DAILY KILL in force for 10h more" in u


# ---- risk context ---------------------------------------------------------------------------------------
class CutStore(FakeStore):
    """FakeStore with every bar that closes after `cut` deleted (a point-in-time check)."""

    def __init__(self, cut, **kw):
        super().__init__(**kw)
        self.b = {s: d[d.index + H <= cut] for s, d in self.b.items()}


def _lstate():
    return {"entry": {"NVDA": {"price": 100.0, "side": 1, "stop": 0.07}, "AAPL": {"price": 100.0, "side": -1}},
            "cooldown": {"COIN": T + 3 * H, "HOOD": str(T - H)}, "kill_until": None}


def test_risk_context_values_and_point_in_time():
    from sentiment import features
    st = FakeStore()
    t = START + 16 * H
    state = features.market_state(st, t)
    book = {"NVDA": 0.1, "AAPL": -0.05}
    cards = [card("a-0", t - 5 * H, 1, tickers=("NVDA",)), card("m-0", t - 2.5 * H, 1, scope="market"),
             card("s-0", t - H, 1, scope="sector", sector="crypto")]
    ctx = replay.risk_context(t, store=st, state=state, book=book, lstate=_lstate(), cards=cards)
    json.dumps(ctx)                                                        # JSON-safe
    close = lambda sym, at: float(st.bars(sym, at, 1)["close"].iloc[-1])  # noqa: E731
    nv = perp("NVDA")
    assert ctx["since_card"]["a-0"]["NVDA"] == pytest.approx(close(nv, t) / close(nv, t - 5 * H) - 1, abs=1e-6)
    assert ctx["since_card"]["m-0"]["_market"] == pytest.approx(
        close("SP500USDT", t) / close("SP500USDT", t - 2.5 * H) - 1, abs=1e-6)
    assert ctx["since_card"]["s-0"] == {}
    assert ctx["beta"] == {tk: round(state[tk]["beta_60d"], 4) if "beta_60d" in state[tk] else 1.0 for tk in UNIVERSE}
    assert ctx["book_net"] == pytest.approx(0.05)
    assert ctx["book_beta_net"] == pytest.approx(0.1 * ctx["beta"]["NVDA"] - 0.05 * ctx["beta"]["AAPL"], abs=1e-6)
    assert ctx["ret_24h"]["NVDA"] == state["NVDA"]["ret_24h"]
    assert list(ctx["cooldown_until"]) == ["COIN"]                         # HOOD's cooldown is over
    px = state["NVDA"]["price"]
    e = ctx["entry"]["NVDA"]
    stop = config.STOP_LOSS_NAME if replay.risk_version() == "v0" else 0.07
    assert e["pnl"] == pytest.approx(px / 100 - 1, abs=1e-6)
    assert e["to_stop"] == pytest.approx(px / 100 - 1 + stop, abs=1e-6)
    assert ctx["entry"]["AAPL"]["stop"] == config.STOP_LOSS_NAME            # no stop recorded -> config stop
    assert ctx["risk_version"] == getattr(config, "RISK_VERSION", None)
    # nothing after t is used: the same context from a store cut at t
    cut = CutStore(t)
    ctx2 = replay.risk_context(t, store=cut, state=features.market_state(cut, t), book=book, lstate=_lstate(),
                               cards=cards)
    assert ctx2 == ctx
    # a card from the future or outside the lookback gets no entry (only shown cards)
    late = replay.risk_context(t, store=st, state=state, book=book, lstate={}, cards=[card("f-0", t + H, 1)])
    assert late["since_card"] == {} and late["entry"] == {} and late["cooldown_until"] == {}


def test_risk_context_v0_stop_and_caps(monkeypatch):
    from sentiment import features
    st = FakeStore()
    t = START + 2 * H                                                       # 02:00 UTC: US session closed
    state = features.market_state(st, t)
    monkeypatch.setattr(config, "RISK_VERSION", "v0")
    monkeypatch.delattr(risk, "effective_caps", raising=False)
    ctx = replay.risk_context(t, store=st, state=state, book={"NVDA": 0.1}, lstate=_lstate(), cards=[])
    assert ctx["entry"]["NVDA"]["stop"] == config.STOP_LOSS_NAME and ctx["risk_version"] == "v0"
    assert set(ctx["caps"].values()) == {config.MAX_W_NAME * config.OFF_HOURS_SCALE}
    assert ctx["caps_source"].startswith("config")


def test_risk_context_uses_effective_caps(monkeypatch):
    from sentiment import features
    st = FakeStore()
    t = START + 16 * H
    seen = []

    def eff(tt, state, version=None):
        seen.append((tt, version))
        return {tk: 0.1 if tk == "MSTR" else 0.15 for tk in UNIVERSE}
    monkeypatch.setattr(risk, "effective_caps", eff, raising=False)
    ctx = replay.risk_context(t, store=st, state=features.market_state(st, t), book={}, lstate={}, cards=[],
                              version="v1")
    assert seen == [(t, "v1")] and ctx["caps"]["MSTR"] == 0.1 and ctx["caps_source"] == "risk.effective_caps"


# ---- runner ---------------------------------------------------------------------------------------------
REPLY = '{"targets": {"NVDA": 0.1, "AAPL": -0.1}, "reasons": {"NVDA": "r"}, "evidence": {}, "confidence": 0.5}'


@pytest.fixture
def fake_llm(monkeypatch):
    """Offline mode plus a fake model: records every prompt, never touches the network or the cache."""
    monkeypatch.setenv("SENTIMENT_OFFLINE", "1")
    monkeypatch.setattr(llm, "model_name", lambda: "fake-model")
    prompts = []

    def complete(messages, *, purpose):
        prompts.append((purpose, messages))
        return REPLY
    monkeypatch.setattr(llm, "complete", complete)
    return prompts


def _recs(run):
    return [json.loads(ln) for ln in (run / "log.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("variant", ["llm", "blinded"])
def test_offline_smoke_logs_risk_ctx_and_versions(tmp_path, fake_llm, variant):
    a = replay.run(variant, "2026-07-01", "2026-07-01", store=FakeStore(), out_dir=tmp_path / "a",
                   cards_fn=fake_cards, quiet=True)
    assert fake_llm and all(p == f"decide:{variant}" for p, _ in fake_llm)
    for _, msgs in fake_llm:
        u = msgs[1]["content"]
        assert "ticker beta cap cooldown_h pnl_entry to_stop" in u
        if variant == "blinded":
            assert not BLIND_LEAK.search(u) and not any(re.search(rf"\b{tk}\b", u) for tk in UNIVERSE)
    assert any("since_bp" in m[1]["content"] for _, m in fake_llm)
    recs = [r for r in _recs(a) if r["event"] == "decision"]
    assert recs and ledger.verify(a / "log.jsonl")
    for r in recs:
        assert r["prompt_version"] == agent.PROMPT_VERSION == "v1"
        assert r["risk_version"] == getattr(config, "RISK_VERSION", None)
        assert r["risk_ctx_hash"] == hashlib.sha256(ledger.canonical(r["risk_ctx"]).encode()).hexdigest()
        assert set(r["risk_ctx"]["since_card"]) == set(r["evidence_ids"])
    held = [r for r in recs[1:] if r["risk_ctx"]["book"]]
    assert held and all(r["risk_ctx"]["entry"] for r in held)              # entries from the risk ledger state
    m = json.loads((a / "metrics.json").read_text())
    assert m["prompt_version"] == "v1" and m["risk_version"] == getattr(config, "RISK_VERSION", None)
    assert m["cards_from"] is None and m["n_items_dropped_by_pin"] == 0
    b = replay.run(variant, "2026-07-01", "2026-07-01", store=FakeStore(), out_dir=tmp_path / "b",
                   cards_fn=fake_cards, quiet=True)
    assert [r["hash"] for r in _recs(a)] == [r["hash"] for r in _recs(b)]  # deterministic


def test_baseline_run_logs_ctx_without_prompt_version(tmp_path):
    out = replay.run("baseline", "2026-07-01", "2026-07-01", store=FakeStore(), out_dir=tmp_path,
                     cards_fn=fake_cards, quiet=True)
    recs = [r for r in _recs(out) if r["event"] == "decision"]
    assert all(r["prompt_version"] is None and r["risk_ctx"] for r in recs)
    assert json.loads((out / "metrics.json").read_text())["prompt_version"] is None


def test_policy_gets_risk_ctx_only_when_it_accepts_it(tmp_path):
    got = []

    def new(t, state, cards, book, variant, *, risk_ctx=None):
        got.append(risk_ctx)
        return {"targets": {}, "reasons": {}, "evidence": {}, "confidence": 0.0, "raw": "{}", "valid": True,
                "shown": [c["card_id"] for c in cards], "llm_request_hash": "h"}
    replay.run("llm", "2026-07-01", "2026-07-01", store=FakeStore(), out_dir=tmp_path, cards_fn=fake_cards,
               decide_fn=new, quiet=True)
    assert got and all(isinstance(c, dict) and "caps" in c for c in got)
    assert replay._accepts_risk_ctx(agent.decide) and not replay._accepts_risk_ctx(lambda t, s, c, b, v: None)


# ---- --cards-from ---------------------------------------------------------------------------------------
class ExtraStore(FakeStore):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.items += [
            {"id": "newstale00", "kind": "news", "available_at": START + 3 * H, "title": "x", "text": "",
             "tickers": None, "raw": {}},
            {"id": "newcached0", "kind": "news", "available_at": START + 4 * H, "title": "y", "text": "",
             "tickers": None, "raw": {}},
            {"id": "rating0001", "kind": "rating", "available_at": START + 6 * H, "title": "z", "text": "",
             "tickers": None, "raw": {}}]


def test_cards_from_pins_news_to_logged_plus_cached(tmp_path, monkeypatch):
    pinned = replay.run("baseline", "2026-07-01", "2026-07-01", store=FakeStore(), out_dir=tmp_path / "pin",
                        cards_fn=fake_cards, quiet=True)
    logged = replay.logged_item_prefixes(pinned)
    assert {"it0", "it-10"} <= logged and not any(p.endswith("-0") for p in logged)
    ver = evidence.news_version()
    monkeypatch.setattr(evidence, "load_cards", lambda: pd.DataFrame(
        {"item_id": ["newcached0", "newstale00"], "extractor": [ver, "news:stale"]}))
    monkeypatch.setattr(llm, "cached", lambda messages: None)                # nothing else in the LLM cache
    # 'newstale00' is not logged by the pinned run and only has a card under an old extractor -> dropped
    seen = []

    def cards_fn(items):
        seen.append({it["id"] for it in items})
        return fake_cards(items)
    out = replay.run("baseline", "2026-07-01", "2026-07-01", store=ExtraStore(), out_dir=tmp_path / "new",
                     cards_fn=cards_fn, quiet=True, cards_from=pinned)
    ids = seen[0]
    assert "newcached0" in ids and "rating0001" in ids and "newstale00" not in ids
    assert {"it0", "it1", "it2"} <= ids                                     # the pinned run's news
    m = json.loads((out / "metrics.json").read_text())
    assert m["cards_from"] == str(pinned) and m["n_items_dropped_by_pin"] >= 1
    # an item whose extraction prompt is in the LLM cache is kept even without a card row
    monkeypatch.setattr(llm, "cached", lambda messages: "[]" if "TITLE: x" in messages[1]["content"] else None)
    kept, dropped = replay.pin_items(ExtraStore().all_items(), pinned)
    assert "newstale00" in {it["id"] for it in kept}
    with pytest.raises(ValueError, match="another run directory"):
        replay.run("baseline", "2026-07-01", "2026-07-01", store=FakeStore(), out_dir=pinned, cards_fn=fake_cards,
                   quiet=True, cards_from=pinned)

