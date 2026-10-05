# Other studies

Three ideas tested after the pre-registered agent. None of them is the submission. Each directory has its own `trials.log`, written before the run. Outputs stay in `out/b2/`, `out/b4/` and `out/b5/`. Section 7 of the report reads those files and does not recompute them.

| Directory | Idea |
|---|---|
| `b2/` | Crisis exit on BTC, plus the incident and delisting pilots |
| `b4/` | Crypto Fear & Greed, on MSTR and on BTC plus four crypto stocks |
| `b5/` | Volatility-managed long book of the ten names |

From the repo root:

```bash
uv run python -m studies.b2.backtest
uv run python -m studies.b4.backtest
uv run python -m studies.b5.compare
```
