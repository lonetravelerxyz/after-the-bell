"""Freeze project B's inputs into snapshot/ (committed) so the replay is reproducible offline.

Keeps UNIVERSE + INDEX_PERPS and the window 2026-06-01 00:00 .. REPLAY_END + 1 day 00:00 UTC by
*availability* time (bar close, settlement, item available_at, F&G day + 1): nothing that became
available after the replay window (the live period) enters the snapshot. Perp bars start earlier, at
PERP_START = REPLAY_START - (BETA_LOOKBACK_D + 3) days, so the risk-v1 beta (60 daily returns) is estimated
from a full history from the first decision on instead of falling back to 1.0 for the first weeks. News
keeps its full columns; ratings and insider keep every MCP column for the universe.

Writes snapshot/{perp,spot,funding,news,ratings,insider,crypto_fng}.parquet and
MANIFEST.sha256 (`sha256  file`, sorted by file name).

Run: uv run python -m sentiment.snapshot [--check | --perp-history]
  --perp-history  only (re)write perp.parquet with the bars before 2026-06-01 from data/raw, keeping every
                  bar already in the snapshot unchanged (checked), and update its manifest line; the other
                  files are left as frozen (raw news has since been topped up, so a full refreeze differs).
"""
from __future__ import annotations

import hashlib
import sys

import pandas as pd

from sentiment.config import BETA_LOOKBACK_D, REPLAY_END, REPLAY_START, SNAP, UNIVERSE
from sentiment.data import fng_available, insider_available, load_raw, news_available, rating_available

START = pd.Timestamp("2026-06-01", tz="UTC")
PERP_START = pd.Timestamp(REPLAY_START, tz="UTC") - pd.Timedelta(days=BETA_LOOKBACK_D + 3)   # beta history
END = pd.Timestamp(REPLAY_END, tz="UTC") + pd.Timedelta(days=1)   # availability <= END
HOUR_MS = 3_600_000
MANIFEST = SNAP / "MANIFEST.sha256"


def _ms(t: pd.Timestamp) -> int:
    return int(t.timestamp() * 1000)


def freeze(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict raw frames to the universe and the window (by availability time)."""
    lo, hi = _ms(START), _ms(END)
    out = {}
    for k in ("perp", "spot"):
        d = frames[k]
        k_lo = _ms(min(START, PERP_START)) if k == "perp" else lo
        out[k] = d[(d.ts >= k_lo) & (d.ts + HOUR_MS <= hi)][["sym", "ts", "open", "high", "low", "close", "quote_vol"]]
    f = frames["funding"]
    out["funding"] = f[(f.ts >= lo) & (f.ts <= hi)][["sym", "ts", "rate"]]
    for k, col, avail in (("news", "published_at", news_available), ("ratings", "rating_date", rating_available),
                          ("insider", "filing_date", insider_available), ("crypto_fng", "date", fng_available)):
        d = frames[k]
        if "symbol" in d.columns:
            d = d[d["symbol"].isin(UNIVERSE)]
        at = d[col].map(avail)
        out[k] = d[(at >= START) & (at <= END)]
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(frames: dict[str, pd.DataFrame]) -> None:
    SNAP.mkdir(parents=True, exist_ok=True)
    for old in SNAP.glob("*.parquet"):
        old.unlink()
    for k, d in frames.items():
        d = d.copy()
        d.attrs = {}
        d.to_parquet(SNAP / f"{k}.parquet", index=False, compression="zstd")
    files = sorted(SNAP.glob("*.parquet"))
    MANIFEST.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))


def check() -> bool:
    """True when every file in MANIFEST.sha256 matches and no parquet is unlisted."""
    listed = dict(ln.split("  ")[::-1] for ln in MANIFEST.read_text().splitlines() if ln.strip())
    ok = set(listed) == {p.name for p in SNAP.glob("*.parquet")}
    return ok and all(sha256(SNAP / n) == h for n, h in listed.items())


def perp_history(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The snapshot's perp bars plus raw bars from PERP_START up to the first bar already frozen per symbol.
    Raises if a bar already in the snapshot would change."""
    old = pd.read_parquet(SNAP / "perp.parquet")
    new = freeze({"perp": raw["perp"], **{k: raw[k] for k in raw if k != "perp"}})["perp"]
    first = old.groupby("sym").ts.min()
    add = new[new.ts < new.sym.map(first).fillna(float("inf"))]
    out = pd.concat([add, old], ignore_index=True).sort_values(["sym", "ts"], kind="stable").reset_index(drop=True)
    kept = out.merge(old, on=["sym", "ts"], suffixes=("", "_old"))
    assert len(kept) == len(old) and all((kept[c] == kept[c + "_old"]).all()
                                         for c in ("open", "high", "low", "close", "quote_vol"))
    return out[old.columns]


def write_perp(perp: pd.DataFrame) -> None:
    perp = perp.copy()
    perp.attrs = {}
    perp.to_parquet(SNAP / "perp.parquet", index=False, compression="zstd")
    lines = [ln for ln in MANIFEST.read_text().splitlines() if ln.strip() and not ln.endswith("  perp.parquet")]
    lines.append(f"{sha256(SNAP / 'perp.parquet')}  perp.parquet")
    MANIFEST.write_text("".join(ln + "\n" for ln in sorted(lines, key=lambda x: x.split("  ")[1])))


def main() -> None:
    if "--perp-history" in sys.argv:
        perp = perp_history(load_raw())
        write_perp(perp)
        g = perp.groupby("sym").ts.agg(["min", "max", "size"])
        g[["min", "max"]] = g[["min", "max"]].apply(lambda s: pd.to_datetime(s, unit="ms", utc=True))
        print(g.to_string())
        print("snapshot manifest", "OK" if check() else "MISMATCH")
        return
    if "--check" in sys.argv:
        ok = check()
        print("snapshot manifest", "OK" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)
    frames = freeze(load_raw())
    write(frames)
    total = 0
    for p in sorted(SNAP.glob("*.parquet")):
        total += p.stat().st_size
        print(f"{p.name:22} {len(frames[p.stem]):>6} rows {p.stat().st_size / 1024:>8.0f} KB")
    print(f"total {total / 1e6:.2f} MB; window {START} .. {END} (availability); manifest {MANIFEST}")
    bars = frames["perp"].groupby("sym").ts.agg(["min", "max", "size"])
    bars[["min", "max"]] = bars[["min", "max"]].apply(lambda s: pd.to_datetime(s, unit="ms", utc=True))
    print(bars.to_string())


if __name__ == "__main__":
    main()
