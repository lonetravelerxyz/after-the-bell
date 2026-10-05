"""Point-in-time news archive from Bitget's data MCP (`news_label_search`), project 2.

Pulls every item for the given labels in weekly windows and keeps `published_at` as the
availability time: in replay the agent at time t only sees items with published_at <= t.

Run: uv run python -m sentiment.fetch_news [start] [end]   (defaults 2026-06-21 .. today)
Writes data/raw/news/news.parquet (gitignored; `config.RAW`) and prints coverage.
"""
from __future__ import annotations

import sys

import pandas as pd

from sentiment.config import RAW
from sentiment.mcp_client import BitgetMCP

OUT = RAW / "news" / "news.parquet"
LABELS = [1, 2, 6, 7, 9]   # labels that returned items on 2026-09-23; 0, 3-5, 8, 10+ were empty


def fetch(start: str, end: str) -> pd.DataFrame:
    m, rows = BitgetMCP(), []
    for lab in LABELS:
        for s in pd.date_range(start, end, freq="7D"):
            e = s + pd.Timedelta(days=7)
            for page in range(1, 50):
                r = m.query("news_label_search", label=lab, start_time=str(s.date()), end_time=str(e.date()),
                            page_size=100, page=page)
                rows += [{"label": lab, **x} for x in r]
                if len(r) < 100:
                    break
    d = pd.DataFrame(rows)
    d["published_at"] = pd.to_datetime(d["published_at"], format="ISO8601", utc=True)
    d["labels"] = d["labels"].map(lambda x: ",".join(map(str, x)) if isinstance(x, list) else str(x))
    d = (d.groupby(["published_at", "title"], as_index=False)
          .agg(content=("content", "first"), labels=("labels", "first"), query_labels=("label", lambda s: sorted(set(s)))))
    return d.sort_values("published_at").reset_index(drop=True)


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-06-21"
    end = sys.argv[2] if len(sys.argv) > 2 else str(pd.Timestamp.now("UTC").date())
    d = fetch(start, end)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.assign(query_labels=d["query_labels"].map(lambda x: ",".join(map(str, x)))).to_parquet(OUT)
    daily = d.set_index("published_at").resample("D").size()
    print(f"{len(d)} unique items {d.published_at.min()} -> {d.published_at.max()}")
    print(f"per day: median {daily.median():.0f}, min {daily.min()}, max {daily.max()}, empty days {(daily == 0).sum()}")
    print(d["labels"].str.split(",").explode().value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
