"""Point-in-time guarantees of sentiment.data.Store and sentiment.features.market_state."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from sentiment.config import INDEX_PERPS, UNIVERSE, perp
from sentiment.data import (Store, fng_available, insider_available, load_raw, load_snapshot, news_available,
                            rating_available)
from sentiment.features import market_state, us_session_open

H = pd.Timedelta(hours=1)
TIMES = [pd.Timestamp(x, tz="UTC") for x in (
    "2026-06-22 00:00", "2026-06-29 13:37", "2026-07-03 14:00", "2026-07-15 20:00", "2026-08-03 16:00",
    "2026-08-10 23:59:59", "2026-08-11 00:00", "2026-08-28 13:30", "2026-09-07 08:00", "2026-09-22 20:00")]


def _cut(frames: dict[str, pd.DataFrame], t: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Delete everything that becomes available after t (bars by close time)."""
    ms = int(t.timestamp() * 1000)
    out = {}
    for k, d in frames.items():
        if k in ("perp", "spot"):
            out[k] = d[d.ts + 3_600_000 <= ms]
        elif k == "funding":
            out[k] = d[d.ts <= ms]
        else:
            col, avail = {"news": ("published_at", news_available), "ratings": ("rating_date", rating_available),
                          "insider": ("filing_date", insider_available), "crypto_fng": ("date", fng_available)}[k]
            out[k] = d[d[col].map(avail) <= t]
    return out


def _store(frames) -> Store:
    s = Store.__new__(Store)
    s.source = "test"
    s._init(frames)
    return s


@pytest.fixture(scope="module")
def frames():
    s = Store(snapshot=True)
    f = load_snapshot() if s.source == "snapshot" else load_raw()
    if "perp" not in f or not len(f["perp"]):
        pytest.skip("no snapshot and no raw data")
    return f


@pytest.fixture(scope="module")
def store(frames):
    return _store(frames)


@pytest.mark.parametrize("t", TIMES)
def test_items_never_after_t(store, t):
    got = store.items(t, 72)
    assert all(t - pd.Timedelta(hours=72) < it["available_at"] <= t for it in got)
    want = [it for it in store.all_items() if t - pd.Timedelta(hours=72) < it["available_at"] <= t]
    assert [it["id"] for it in got] == [it["id"] for it in want]


@pytest.mark.parametrize("t", TIMES)
def test_bars_funding_fng_premium_never_after_t(store, t):
    for sym in [perp(tk) for tk in UNIVERSE] + INDEX_PERPS:
        b = store.bars(sym, t, 500)
        assert len(b) and (b.index + H <= t).all()
        full = store._perp[sym]
        later = full[full.index + H <= t]
        assert b.index[-1] == later.index[-1]                     # the latest closed bar, not an older one
        assert len(store.bars(sym, t, 3)) == 3
    for tk in UNIVERSE:
        f = store.funding(perp(tk), t, 50)
        assert (f.index <= t).all() and len(f) <= 50
        s = store.settlements(perp(tk), t - pd.Timedelta(days=1), t)
        assert ((s.index > t - pd.Timedelta(days=1)) & (s.index <= t)).all()
    v = store.fng(t)
    avail = store._fng[store._fng.index <= t]
    assert v == (int(avail.iloc[-1]) if len(avail) else None)


@pytest.mark.parametrize("t", TIMES)
def test_truncated_store_gives_same_answers(frames, store, t):
    cut = _store(_cut(frames, t))
    assert market_state(cut, t) == market_state(store, t)
    assert [i["id"] for i in cut.items(t, 72)] == [i["id"] for i in store.items(t, 72)]
    assert cut.fng(t) == store.fng(t)
    for tk in UNIVERSE:
        sym = perp(tk)
        pd.testing.assert_frame_equal(cut.bars(sym, t, 200), store.bars(sym, t, 200))
        pd.testing.assert_series_equal(cut.funding(sym, t, 30), store.funding(sym, t, 30))
        assert cut.spot_premium(sym, t) == store.spot_premium(sym, t)


def test_next_open_is_first_bar_at_or_after_t(store):
    d = store._perp["NVDAUSDT"]
    t0 = pd.Timestamp("2026-08-03 12:00", tz="UTC")
    assert store.next_open("NVDA", t0) == d.at[t0, "open"]                       # exactly on a bar open
    assert store.next_open("NVDAUSDT", t0 + pd.Timedelta(minutes=30)) == d.at[t0 + H, "open"]
    assert store.next_open("NVDA", t0 - pd.Timedelta(seconds=1)) == d.at[t0, "open"]
    assert store.next_open("NVDA", d.index[-1] + pd.Timedelta(minutes=1)) is None


def test_availability_rules_and_stable_ids(frames, store):
    items = store.all_items()
    assert len({i["id"] for i in items}) == len(items)
    for it in items:
        assert it["id"] == hashlib.sha1(f"{it['kind']}|{it['available_at'].isoformat()}|{it['title']}".encode()).hexdigest()
        assert it["available_at"].tzinfo is not None
        r = it["raw"]
        if it["kind"] == "rating":
            assert it["available_at"] == pd.Timestamp(r["rating_date"], tz="UTC") + pd.Timedelta(hours=37, minutes=30)
        elif it["kind"] == "insider":
            assert it["available_at"] == pd.Timestamp(r["filing_date"], tz="UTC") + pd.Timedelta(days=1, hours=3)
            assert r["transaction_type"] in ("P", "S")
        else:
            assert it["tickers"] is None and it["available_at"] == pd.Timestamp(r["published_at"]) - pd.Timedelta(hours=8)
    assert [i["id"] for i in _store(frames).all_items()] == [i["id"] for i in items]   # stable across builds


def test_fng_day_visible_only_next_day():
    f = pd.DataFrame({"date": ["2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z"], "value": [20, 80]})
    s = _store({"crypto_fng": f})
    assert s.fng(pd.Timestamp("2026-08-01 23:59", tz="UTC")) is None
    assert s.fng(pd.Timestamp("2026-08-02 00:00", tz="UTC")) == 20
    assert s.fng(pd.Timestamp("2026-08-02 23:00", tz="UTC")) == 20
    assert s.fng(pd.Timestamp("2026-08-03 00:00", tz="UTC")) == 80


def test_toy_bars_close_time():
    ts = [int(pd.Timestamp(f"2026-08-01 0{h}:00", tz="UTC").timestamp() * 1000) for h in range(3)]
    p = pd.DataFrame({"sym": "NVDAUSDT", "ts": ts, "open": [1.0, 2, 3], "high": 1.0, "low": 1.0,
                      "close": [1.5, 2.5, 3.5], "quote_vol": 1.0})
    s = _store({"perp": p})
    t = pd.Timestamp("2026-08-01 01:30", tz="UTC")
    assert s.bars("NVDA", t, 5)["close"].tolist() == [1.5]      # the 01:00 bar is still open
    assert s.next_open("NVDA", t) == 3.0
    assert s.bars("NVDA", pd.Timestamp("2026-08-01 02:00", tz="UTC"), 5)["close"].tolist() == [1.5, 2.5]


def test_us_session():
    assert us_session_open(pd.Timestamp("2026-08-03 13:30", tz="UTC"))           # Mon 09:30 EDT
    assert not us_session_open(pd.Timestamp("2026-08-03 20:00", tz="UTC"))
    assert not us_session_open(pd.Timestamp("2026-08-01 15:00", tz="UTC"))       # Saturday
    assert not us_session_open(pd.Timestamp("2026-09-07 15:00", tz="UTC"))       # Labor Day


def test_insider_evening_filing_not_visible_before_public():
    """Regression: EDGAR gives Form 4s accepted until 22:00 New York time that day's filing date. META's
    0000950103-26-012279 (filing date 2026-08-12) was accepted 20:37:22 EDT = 2026-08-13 00:37 UTC; the
    00:00 UTC decision must not see it (it used to: available_at was filing_date + 1 day 00:00)."""
    accepted = pd.Timestamp("2026-08-12 20:37:22", tz="America/New_York").tz_convert("UTC")
    at = insider_available("2026-08-12")
    assert at >= accepted
    for day in ("2026-07-15", "2026-12-15"):                        # EDT and EST: 22:00 NY is public by then
        latest = pd.Timestamp(f"{day} 22:00", tz="America/New_York").tz_convert("UTC")
        assert insider_available(day) >= latest
    ins = pd.DataFrame([{"symbol": "META", "transaction_type": "S", "filing_date": "2026-08-12",
                         "owner_name": "X", "owner_title": "CFO", "securities_transacted": 100.0,
                         "transaction_price": 700.0, "transaction_date": "2026-08-10", "securities_owned": 1000.0}])
    s = _store({"insider": ins})
    assert s.items(pd.Timestamp("2026-08-13 00:00", tz="UTC"), 72) == []
    assert [i["kind"] for i in s.items(pd.Timestamp("2026-08-13 04:00", tz="UTC"), 72)] == ["insider"]


def test_ticker_without_funding_does_not_break_market_state(frames):
    """Regression: an absent funding series used to be a RangeIndex Series and market_state raised TypeError."""
    f = dict(frames)
    f["funding"] = frames["funding"][frames["funding"]["sym"] != "CRCLUSDT"]
    s = _store(f)
    t = pd.Timestamp("2026-07-01 00:00", tz="UTC")
    st = market_state(s, t)
    assert st["CRCL"]["funding_last"] is None and st["CRCL"]["funding_7d_mean"] is None
    assert st["NVDA"]["funding_last"] is not None
    assert len(s.funding("CRCL", t, 10)) == 0 and len(s.settlements("CRCL", t - 24 * H, t)) == 0
    assert isinstance(s.funding("CRCL", t, 10).index, pd.DatetimeIndex)


def test_gap_in_bars_no_late_fill_no_stale_price():
    """Regression: across a data gap next_open used to return the open of a bar hours later, and features
    showed the pre-gap close as the current price."""
    idx = [pd.Timestamp("2026-07-01 00:00", tz="UTC") + h * H for h in range(24) if not 10 <= h <= 15]
    p = pd.DataFrame({"sym": "NVDAUSDT", "ts": [int(t.timestamp() * 1000) for t in idx], "open": 100.0,
                      "high": 100.0, "low": 100.0, "close": 100.0, "quote_vol": 1.0})
    s = _store({"perp": p})
    t = pd.Timestamp("2026-07-01 12:00", tz="UTC")
    assert s.next_open("NVDA", t) is None                                    # 16:00 bar is 4h later
    assert s.next_open("NVDA", pd.Timestamp("2026-07-01 15:30", tz="UTC")) == 100.0   # 16:00 within 1h
    st = market_state(s, t)["NVDA"]
    assert st["price"] is None and st["ret_4h"] is None and st["ret_24h"] is None
    assert market_state(s, pd.Timestamp("2026-07-01 11:00", tz="UTC"))["NVDA"]["price"] == 100.0  # 1h old: fine


def test_news_stamps_are_utc8_and_first_seen_wins():
    from sentiment.data import news_available
    stamp = "2026-09-23T17:01:31Z"                        # served at 10:52 UTC: UTC+8 wall time
    assert news_available(stamp) == pd.Timestamp("2026-09-23 09:01:31", tz="UTC")
    assert news_available(stamp, "2026-09-23 10:00") == pd.Timestamp("2026-09-23 10:00", tz="UTC")
    assert news_available(stamp, "2026-09-23 08:00") == pd.Timestamp("2026-09-23 09:01:31", tz="UTC")
