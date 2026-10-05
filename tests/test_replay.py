"""Policies and runner: baseline formula, LLM prompt/parse per variant, replay determinism, no look-ahead."""
import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from sentiment import agent, baseline, config, ledger, llm, replay
from sentiment.config import CARD_LOOKBACK_H, DECISION_HOURS, INDEX_PERPS, MAX_W_NAME, UNIVERSE, perp
from sentiment.evidence import NAMES

H = pd.Timedelta(hours=1)
T = pd.Timestamp("2026-07-01 16:00", tz="UTC")          # Wednesday, US session open


@pytest.fixture(autouse=True)
def _risk_v0(monkeypatch):
    """These tests encode the v0 risk rules and record format (flat cap and stop, no beta step, no risk
    context); the v1 counterparts are in test_versions.py, test_agent_v1.py and test_risk_v1.py."""
    monkeypatch.setattr(config, "RISK_VERSION", "v0")


def card(cid, at, stance, strength=1.0, scope="name", tickers=("NVDA",), novelty="new", sector="", summary="s"):
    return {"card_id": cid, "item_id": cid.split("-")[0], "kind": "news", "available_at": pd.Timestamp(at),
            "tickers": list(tickers) if scope == "name" else [], "scope": scope, "sector": sector,
            "stance": stance, "strength": strength, "horizon_h": 48, "novelty": novelty, "summary": summary,
            "quote": ""}


# ---- baseline ---------------------------------------------------------------------------------
def test_baseline_formula():
    cards = [card("a-0", T - 12 * H, 2, 0.5),                                  # NVDA +2*.5*e^-.5
             card("b-0", T - 24 * H, -1, 0.8, tickers=("AAPL",)),             # AAPL -.8*e^-1
             card("c-0", T - 6 * H, 1, 0.6, scope="market"),                  # all names +.5*.6*e^-.25
             card("d-0", T - 1 * H, 2, 1.0, novelty="post_hoc"),              # skipped
             card("e-0", T - 1 * H, 2, 1.0, scope="sector", sector="ai_semis"),  # skipped
             card("f-0", T + 1 * H, 2, 1.0)]                                    # future: ignored
    d = baseline.decide(T, {}, cards, {})
    mkt = 0.5 * 0.6 * math.exp(-0.25)
    assert d["valid"] and d["targets"]["NVDA"] == pytest.approx(0.05 * (math.exp(-0.5) + mkt), abs=1e-6)
    assert d["targets"]["AAPL"] == pytest.approx(0.05 * (-0.8 * math.exp(-1) + mkt), abs=1e-6)
    assert d["targets"]["TSLA"] == pytest.approx(0.05 * mkt, abs=1e-6)
    assert d["evidence"]["NVDA"] == ["a-0", "c-0"]
    big = [card(f"g{i}-0", T, 2, 1.0) for i in range(5)]                       # score 10 -> 0.5 -> clipped
    assert baseline.decide(T, {}, big, {})["targets"]["NVDA"] == MAX_W_NAME


# ---- agent prompt -----------------------------------------------------------------------------
def _state(price=123.4567):
    s = {tk: {"price": price, "ret_4h": 0.01, "ret_24h": -0.02, "ret_7d": 0.05, "vol_7d": 0.03,
              "funding_last": 0.0001, "funding_7d_mean": 0.00005, "spot_premium": -0.0004,
              "us_session_open": True} for tk in UNIVERSE}
    s["_market"] = {"fng": 40, "fng_regime": "fear", "sp500_ret_24h": 0.003, "ndx_ret_24h": None}
    return s


CARDS = [card("n1-0", T - 5 * H, 2, 0.7, summary="NVIDIA beat estimates on July 30, 2026 and Apple's deal"),
         card("m1-0", T - 2 * H, -1, 0.4, scope="market", summary="Fed hike fears"),
         card("p1-0", T - 1 * H, 1, 0.5, tickers=("TSLA",), novelty="post_hoc", summary="Tesla surges on July 1"),
         card("old-0", T - (CARD_LOOKBACK_H + 1) * H, 2, 0.9, summary="too old"),
         card("fut-0", T + H, 2, 0.9, summary="from the future")]


def test_prompt_llm_variant():
    msgs = agent.build_messages(T, _state(), agent.select_cards(T, CARDS), {"NVDA": 0.1}, "llm")
    u = msgs[1]["content"]
    assert "2026-07-01 16:00 UTC" in u and "NVDA" in u and "n1-0" in u
    assert "POST_HOC" in u and "too old" not in u and "from the future" not in u
    assert "123.4" not in u                                    # price levels never shown
    assert "CURRENT BOOK: NVDA +0.100" in u
    assert "{{" not in msgs[0]["content"]


def test_prompt_nonews_and_blinded():
    u = agent.build_messages(T, _state(), [], {}, "nonews")[1]["content"]
    assert "EVIDENCE: not provided" in u and "n1-0" not in u
    b = agent.build_messages(T, _state(), agent.select_cards(T, CARDS), {"NVDA": 0.1}, "blinded")
    text = b[0]["content"] + b[1]["content"]
    for tk in UNIVERSE:
        assert not re.search(rf"\b{tk}\b", text), tk
        for name in NAMES[tk]:
            assert not re.search(rf"\b{re.escape(name)}\b", b[1]["content"]), name
    assert "2026" not in b[1]["content"] and "July" not in b[1]["content"]
    assert f"Company {agent.ALIAS['NVDA']}" in b[1]["content"]
    assert f"CURRENT BOOK: {agent.ALIAS['NVDA']} +0.100" in b[1]["content"]


def test_parse_decision_and_aliases():
    shown = ["n1-0", "m1-0"]
    txt = '```json\n{"targets": {"NVDA": 0.12, "AAPL": 0, "XYZ": 0.05}, "reasons": {"NVDA": "beat"}, ' \
          '"evidence": {"NVDA": ["n1-0", "made-up"]}, "confidence": 1.7}\n```'
    d = agent.parse_decision(txt, shown)
    assert d["valid"] and d["targets"] == {"NVDA": 0.12, "XYZ": 0.05}     # risk drops XYZ
    assert d["evidence"] == {"NVDA": ["n1-0"]} and d["confidence"] == 1.0
    a = agent.ALIAS["TSLA"]
    d = agent.parse_decision(json.dumps({"targets": {a: -0.1}, "reasons": {f"Company {a}": "x"}}), shown, True)
    assert d["targets"] == {"TSLA": -0.1} and d["reasons"] == {"TSLA": "x"}
    for bad in ("I think NVDA goes up", '{"targets": [1, 2]}', '{"targets": {"NVDA": "lots"}}'):
        assert agent.parse_decision(bad, shown)["valid"] is False


def test_decide_calls_llm_once_and_handles_errors(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "model_name", lambda: "test-model")

    def fake(messages, *, purpose):
        calls.append((messages, purpose))
        return '{"targets": {"NVDA": 0.1}, "reasons": {"NVDA": "r"}, "evidence": {"NVDA": ["n1-0"]}, "confidence": 0.6}'
    monkeypatch.setattr(llm, "complete", fake)
    d = agent.decide(T, _state(), CARDS, {}, "llm")
    assert d["valid"] and d["targets"] == {"NVDA": 0.1} and len(calls) == 1 and calls[0][1] == "decide:llm"
    assert d["shown"] == ["p1-0", "m1-0", "n1-0"] and d["llm_request_hash"] == llm.cache_key("test-model", calls[0][0])
    assert agent.decide(T, _state(), CARDS, {}, "nonews")["shown"] == []

    def boom(messages, *, purpose):
        raise llm.LLMError("down")
    monkeypatch.setattr(llm, "complete", boom)
    d = agent.decide(T, _state(), CARDS, {}, "llm")
    assert d["valid"] is False and d["raw"].startswith("LLMError") and d["error"] == "down"

    def miss(messages, *, purpose):
        raise llm.CacheMiss("offline")
    monkeypatch.setattr(llm, "complete", miss)
    with pytest.raises(llm.CacheMiss):
        agent.decide(T, _state(), CARDS, {}, "llm")


def test_select_cards_budget_drops_post_hoc_first():
    many = [card(f"p{i}-0", T - i * 0.1 * H, 1, novelty="post_hoc") for i in range(agent.MAX_CARDS)]
    many += [card("keep-0", T - 70 * H, 1)]
    got = [c["card_id"] for c in agent.select_cards(T, many)]
    assert len(got) == agent.MAX_CARDS and "keep-0" in got


# ---- runner -----------------------------------------------------------------------------------
START = pd.Timestamp("2026-07-01", tz="UTC")


class FakeStore:
    """Deterministic random-walk bars for every perp, flat funding, no spot, constant F&G."""

    def __init__(self, days=4, seed=7):
        rng = np.random.default_rng(seed)
        idx = pd.date_range(START - pd.Timedelta(days=9), START + pd.Timedelta(days=days + 1), freq="h")
        self.b = {}
        for sym in [perp(t) for t in UNIVERSE] + INDEX_PERPS:
            c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
            o = np.r_[c[0], c[:-1]]
            self.b[sym] = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                                        "quote_vol": 1.0}, index=idx)
        f_idx = pd.date_range(idx[0], idx[-1], freq="8h")
        self.f = {perp(t): pd.Series(0.0001, index=f_idx) for t in UNIVERSE}
        self.items = [{"id": f"it{i}", "kind": "news", "available_at": START + i * 5 * H, "title": f"t{i}",
                       "text": "", "tickers": None, "raw": {}} for i in range(-10, 20)]

    def bars(self, sym, t, n):
        d = self.b[sym]
        return d[d.index + H <= t].tail(n)

    def next_open(self, sym, t):
        d = self.b[sym]
        d = d[d.index >= t]
        return float(d["open"].iloc[0]) if len(d) else None

    def funding(self, sym, t, n):
        s = self.f[sym]
        return s[s.index <= t].tail(n)

    def settlements(self, sym, t0, t1):
        s = self.f[sym]
        return s[(s.index > t0) & (s.index <= t1)]

    def fng(self, t):
        return 50

    def spot_premium(self, sym, t):
        return None

    def all_items(self):
        return list(self.items)


def fake_cards(items):
    out = []
    for i, it in enumerate(items):
        tk = UNIVERSE[i % len(UNIVERSE)]
        out.append(card(f"{it['id']}-0", it["available_at"], (-1) ** i * 2, 0.9, tickers=(tk,)))
        out[-1]["item_id"] = it["id"]
    return out


def _hashes(run_dir):
    return [json.loads(ln)["hash"] for ln in (run_dir / "log.jsonl").read_text().splitlines()]


def test_replay_baseline_writes_verifiable_deterministic_run(tmp_path):
    a = replay.run("baseline", "2026-07-01", "2026-07-02", store=FakeStore(), out_dir=tmp_path / "a",
                   cards_fn=fake_cards, quiet=True)
    b = replay.run("baseline", "2026-07-01", "2026-07-02", store=FakeStore(), out_dir=tmp_path / "b",
                   cards_fn=fake_cards, quiet=True)
    assert ledger.verify(a / "log.jsonl") and _hashes(a) == _hashes(b)
    recs = [json.loads(ln) for ln in (a / "log.jsonl").read_text().splitlines()]
    assert {r["event"] for r in recs} <= {"decision", "risk_exit"}
    recs = [r for r in recs if r["event"] == "decision"]
    assert len(recs) == 2 * len(DECISION_HOURS)
    assert {r["mode"] for r in recs} == {"replay"} and recs[0]["run_id"] == "baseline_2026-07-01_2026-07-02"
    assert any(r["fills"] for r in recs) and all(r["decision"]["targets"] == r["baseline_targets"] for r in recs)
    eq = pd.read_csv(a / "equity.csv")
    assert len(eq) == 2 * 24 + 1 and {"ts", "equity", "gross_equity", "fees", "spread", "funding"} <= set(eq)
    assert eq["equity"].iloc[0] == 10_000.0                         # pre-trade mark at the start
    m = json.loads((a / "metrics.json").read_text())
    assert m["log_verified"] and m["n_decisions"] == 12 and m["trades"] > 0


def test_replay_policy_sees_no_future_cards_and_logs_baseline(tmp_path):
    seen = []

    def spy(t, state, cards, book, variant):
        seen.append((t, cards, state))
        return {"targets": {"NVDA": 0.5, "AAPL": -0.1}, "reasons": {}, "evidence": {}, "confidence": 0.5,
                "raw": "{}", "valid": True, "shown": [c["card_id"] for c in cards], "llm_request_hash": "h"}
    out = replay.run("llm", "2026-07-01", "2026-07-01", store=FakeStore(), out_dir=tmp_path,
                     cards_fn=fake_cards, decide_fn=spy, quiet=True)
    assert len(seen) == len(DECISION_HOURS)
    for t, cards, state in seen:
        assert all(t - CARD_LOOKBACK_H * H < c["available_at"] <= t for c in cards)
        assert state["NVDA"]["price"] is not None
    recs = [json.loads(ln) for ln in (out / "log.jsonl").read_text().splitlines()]
    r0, r = recs[0], recs[DECISION_HOURS.index(16)]                # 00:00 off-hours, 16:00 in session
    assert r0["targets"]["NVDA"] == pytest.approx(MAX_W_NAME * 0.5)
    assert {a["rule"] for a in r0["risk_actions"]} >= {"name_cap", "off_hours"}
    assert r["targets"]["NVDA"] == MAX_W_NAME and any(a["rule"] == "name_cap" for a in r["risk_actions"])
    assert r["llm_request_hash"] == "h" and "baseline_targets" in r and r["decision"]["valid"]


def test_shuffle_cards_never_earlier_deterministic():
    """Regression: the placebo used to permute available_at across the window, so ~half the cards were
    shown before they existed (future information in the 'null'). Now every card is shown late by
    7-21 days, the same lag for all cards of an item, deterministically."""
    items = FakeStore().all_items()
    cards = fake_cards(items)
    cards.append({**cards[0], "card_id": "it-10-1"})               # second card of the first item
    s1, s2 = replay.shuffle_cards(cards), replay.shuffle_cards(cards)
    assert [(c["card_id"], c["available_at"]) for c in s1] == [(c["card_id"], c["available_at"]) for c in s2]
    lo, hi = (pd.Timedelta(days=d) for d in replay.SHUFFLE_LAG_D)
    for c in s1:
        assert c["available_at"] >= c["true_available_at"]
        assert lo <= c["available_at"] - c["true_available_at"] <= hi
    assert len({c["available_at"] - c["true_available_at"] for c in s1}) > len(items) // 2   # lags differ
    by_item = {}
    for c in s1:
        by_item.setdefault(c["item_id"], set()).add(c["available_at"])
    assert all(len(v) == 1 for v in by_item.values())


def test_shuffled_run_shows_no_future_cards_and_logs_true_baseline(tmp_path):
    seen = []

    def spy(t, state, cards, book, variant):
        seen.append((t, cards))
        return {"targets": {}, "reasons": {}, "evidence": {}, "confidence": 0.0, "raw": "{}", "valid": True,
                "shown": [c["card_id"] for c in cards], "llm_request_hash": "h"}
    out = replay.run("shuffled", "2026-07-07", "2026-07-09", store=FakeStore(days=12), out_dir=tmp_path,
                     cards_fn=fake_cards, decide_fn=spy, quiet=True)
    assert seen and all(c["true_available_at"] <= c["available_at"] <= t for t, cards in seen for c in cards)
    assert any(cards for _, cards in seen)                         # stale cards do reach the placebo
    recs = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines() if '"decision"' in x]
    true = fake_cards(FakeStore().all_items())
    for r in recs:
        t = pd.Timestamp(r["ts"])
        live = [c for c in true if t - CARD_LOOKBACK_H * H < c["available_at"] <= t]
        assert r["baseline_targets"] == baseline.decide(t, {}, live, {})["targets"]
    assert any(r["baseline_targets"] for r in recs)


BLIND_LEAK = re.compile(
    r"PT \d|\$\s?\d|\b(?:at|@) ?\d[\d,]*\.\d|\b\d{1,2}/\d{1,2}\b|\b20\d\d\b|"
    rf"\b{agent.MONTHS}\b|\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b")
LEAKY = [
    ("rating", "NVDA", "KGI Securities downgrades NVDA Outperform -> Hold, PT 315"),
    ("rating", "META", "Susquehanna maintains META Positive, PT 900 -> 650 (-28%)"),
    ("rating", "MSTR", "BTIG maintains MSTR Buy, PT 350 -> 250 (-29%)"),
    ("insider", "META", "Mark Zuckerberg (CEO) sold 12,345 META shares at 700.12 ($8.6M)"),
    ("news", "TSLA", "Tesla surged Sept 17 on reports of ~5,000 Optimus robot orders and a Cybercab display"),
    ("news", "GOOGL", "Honda will integrate Google Gemini; earnings due 7/29 after the close on Tuesday"),
    ("news", "CRCL", "Circle rose as USDC circulation hit $73.3B, Arc mainnet in September"),
    ("news", "MSTR", "Strategy (MSTR) surged ~12.7% as a crypto proxy; holds ~850K BTC at $66,000 average"),
    ("news", "NVDA", "In September, Nvidia and Jensen Huang said Blackwell demand is strong (Q3 FY2027)"),
]


def test_blinded_prompt_hides_identity_levels_and_dates():
    """Regression: blinded prompts leaked price levels, product/person names and numeric dates."""
    cards = [{**card(f"x{i}-0", T - (i + 1) * H, 1, tickers=(tk,), summary=summ), "kind": kind}
             for i, (kind, tk, summ) in enumerate(LEAKY)]
    u = agent.build_messages(T, _state(), agent.select_cards(T, cards), {}, "blinded")[1]["content"]
    assert not BLIND_LEAK.search(u), BLIND_LEAK.search(u)
    for tk in UNIVERSE:
        for w in [tk, *NAMES[tk], *agent.IDENTITY[tk]]:
            assert not re.search(rf"\b{re.escape(w)}\b", u), w
    assert "PT -28%" in u and "notional 1-10M USD" in u and "(CEO) sold" in u
    assert "12.7%" in u                                            # returns are evidence, not identity


def test_blinded_real_cards_have_no_levels_or_identity():
    from sentiment import evidence
    if not evidence.CARDS_FILE.exists():
        pytest.skip("no cards cache")
    d = evidence.load_cards()
    ident = re.compile(r"\b(?:" + "|".join(re.escape(w) for ws in agent.IDENTITY.values() for w in ws) + r")\b")
    for c in d[d["scope"] == "name"].to_dict("records"):
        b = agent.blind_summary(c)
        assert not BLIND_LEAK.search(b) and not ident.search(b), b


def test_replay_stops_on_llm_endpoint_error(tmp_path):
    """Regression: an endpoint failure used to be logged as an invalid hold with nothing cached, so the
    run could not be reproduced offline. The replay now stops."""
    def down(t, state, cards, book, variant):
        d = agent.parse_decision("", [])
        return {**d, "raw": "LLMError: HTTP 503", "error": "HTTP 503", "shown": [], "llm_request_hash": "h"}
    with pytest.raises(llm.LLMError, match="replay stopped"):
        replay.run("llm", "2026-07-01", "2026-07-01", store=FakeStore(), out_dir=tmp_path, cards_fn=fake_cards,
                   decide_fn=down, quiet=True)
