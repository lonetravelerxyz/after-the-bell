"""Project B report: report/REPORT.md, metrics.json and figures from the runs in out/.

  uv run python -m sentiment.report [--out out] [--report-dir report]
                                    [--reconcile PATH] [--signal-tests PATH]

Reads, never writes, the run directories `{variant}[_{tag}]_{start}_{end}/` (tag = risk/prompt version,
e.g. `llm_v1_2026-08-11_2026-09-22`; no tag = v0), `live/`, `live/reconcile.json` (written by
`sentiment.reconcile`) and `signal_tests_test.json`. It never calls the model (no LLM import; decision
prompts are read from the LLM cache files by request hash).

Expected runs (always listed, with status `complete` / `incomplete` (a replay in progress or aborted) /
`not run yet`): the pre-registered test-window runs of docs/PREREG.md (v1: baseline, llm, nonews,
shuffled; blinded optional) and the dev reference (v0 llm and baseline on the full dev window, v1 llm
and baseline on the first half). Any other run directory found is listed too.

Everything is first collected into one dict, written as metrics.json, and REPORT.md is rendered from
that dict alone (`render(json.loads(metrics.json))` reproduces REPORT.md), so every number in the report
comes from metrics.json. Sections:

Sections follow project A4's report: Summary (judging table, what, the <= 8 findings of `summary`, the other
tests); 1 hypothesis and prior evidence; 2 data and universe (snapshot counts, cards, the knowledge-cutoff probe,
data facts); 3 agent and portfolio construction (the PREREG run list); 4 does the news predict returns (signal
tests, test and dev, dev diagnosis); 5 performance net of costs (headline, LLM increment with its claim rule, dev
reference, all runs, cost ladders, figures); 6 is it the news (placebos with the information-use claim rule,
realised beta); 7 robustness: the other strategies tested after the test window (sentiment/other_tests.py); 8 risk
layer, every run and three decisions end to end; 9 live log, deviations and errata, trials, limits; 10 references
(checked against Crossref); 11 reproduce. A reading whose runs are missing says "not run yet".

Labels: replay numbers are "estimated (replay)". Live numbers are "observed" only for fills on Bitget Demo
(spec section 2); a shadow-ledger live run (live-venue quotes, simulated fills) is "estimated", and a live
run whose broker mode is unknown is "unverified". Daily statistics use complete UTC days only
(`metrics.complete_days`); a partial first or last day is reported separately.

Provenance: a replay whose log cites evidence items that do not exist under the current data rules
(item ids are sha1(kind|available_at|title), so a changed availability rule renames them), or whose
metrics.json carries a legacy sha256 `code_stamp` different from `code_hash()`, is `stale`: it keeps its
row but is kept out of the readings, comparisons and figures. A git `code_stamp` (short sha, "-dirty"
when uncommitted; written by sentiment.replay) is compared with the commit that froze PREREG.md: the
report lists the replay-path files that differ, it does not decide on its own that they matter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentiment import ledger, metrics, other_tests
from sentiment.config import (CACHE, DECISION_HOURS, DEV_END, DOCS, OUT, PREREG_FILE, PROMPTS, REPLAY_END, REPLAY_START,
                              REPORT_DIR, ROOT, SNAP, TEST_START, TRIALS_FILE, UNIVERSE, perp)

VARIANTS = ("llm", "baseline", "nonews", "shuffled", "blinded")
LLM_VARIANTS = ("llm", "nonews", "shuffled", "blinded")
CONTROLS = ("nonews", "shuffled", "blinded")
DEV_HALF_END = "2026-07-15"                         # the v1 prompt check (trials.log entry 5, committed in 62286d9)
WINDOWS = {"test": (TEST_START, REPLAY_END), "dev": (REPLAY_START, DEV_END), "dev-half": (REPLAY_START, DEV_HALF_END)}
# (window, version, variant, optional): the pre-registered test runs, then the dev reference
EXPECTED = ([("test", "v1", v, v == "blinded") for v in VARIANTS]
            + [("dev", "v0", "llm", False), ("dev", "v0", "baseline", False),
               ("dev-half", "v1", "llm", False), ("dev-half", "v1", "baseline", False)])
LLM_CACHE = CACHE / "llm"
PROVENANCE_FILE = ROOT / "PROVENANCE.json"          # only in the exported public repo (scripts/export_public.sh)
DIAGNOSIS_FILE = DOCS / "DIAGNOSIS-dev.md"
RUN_RE = re.compile(r"^(?P<variant>[a-z]+)(?:_(?P<tag>v\d+))?_(?P<start>\d{4}-\d\d-\d\d)_(?P<end>\d{4}-\d\d-\d\d)$")
READ_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError, ZeroDivisionError)
MIN_DAYS_READ = 20          # below this the increment CI and P(diff <= 0) are not shown
N_CHAINS = 3
N_CHAIN_FOCUS = 3           # tickers per decision whose reasons and cited cards are shown
TEXT_MAX = 320
SIGNAL_T_BAR = 2.6          # PREREG: 5 tests (a counts as 3), Bonferroni at 5%
BETA_BAND = 0.03            # PREREG: realised beta to the EW basket expected within +-0.03
EPS_W = 1e-4                # weight changes below this are not shown in a decision chain
# the site's pack (web/design/packs/cartesian): stone and ink, series told apart by stroke pattern first, tone second
COLOR = {"llm": "#1A1A1A", "baseline": "#8A8178", "nonews": "#9A9185", "shuffled": "#5A5A5A",
         "blinded": "#857C72", "live": "#1A1A1A"}
DASH = {"llm": "-", "baseline": (0, (6, 4)), "nonews": (0, (2, 6)), "shuffled": (0, (1.5, 3)),
        "blinded": (0, (8, 3, 2, 3)), "live": "-"}
INK, MUTED, GRID, PAPER = "#1A1A1A", "#5A5A5A", "#E2DBD1", "#F3EFE9"
RULE_PRIORITY = ("invalid", "universe", "name_cap", "off_hours", "funding_block", "stop_loss", "cooldown",
                 "daily_kill", "kill", "beta_neutral", "net_cap", "gross_cap")
RULE_TEXT = {
    "invalid": "LLM output invalid -> hold the current book", "universe": "ticker outside the universe -> 0",
    "name_cap": "per-name cap (v1: vol-scaled)", "off_hours": "outside the US session the cap is scaled for new exposure",
    "funding_block": "|funding| >= FUNDING_BLOCK blocks new exposure on the paying side",
    "stop_loss": "name down its stop from entry -> flat + 24h cooldown (v1: stop = clip(2 x vol_7d, 3%, 10%))",
    "cooldown": "name in stop-loss cooldown -> 0", "daily_kill": "equity down DAILY_KILL from its 24h high -> flat all",
    "kill": "daily kill in force -> name flat", "beta_neutral": "v1: minimal-L2 change to a beta-neutral book",
    "net_cap": "net exposure cap MAX_NET", "gross_cap": "gross exposure cap MAX_GROSS"}
# spec section 2: only Bitget Demo fills are exchange fills ("observed"); the shadow ledger simulates its fills
LIVE_LABEL = {"demo": "observed (Bitget Demo fills)",
              "shadow": "estimated (shadow ledger: simulated fills at live-venue quotes)"}
UNKNOWN_LIVE_LABEL = "unverified (broker mode unknown)"
REPLAY_LABEL = "estimated (replay)"
SIGNAL_TESTS = (
    ("a_ic_4h", "(a) rank IC of the baseline score vs forward perp return, 4h, Newey-West t"),
    ("a_ic_24h", "(a) rank IC of the baseline score vs forward perp return, 24h, Newey-West t"),
    ("a_ic_72h", "(a) rank IC of the baseline score vs forward perp return, 72h, Newey-West t"),
    ("b_rating_24h", "(b) signed 24h demeaned return after rating cards, one observation per event"),
    ("c_news_new_24h", "(c) signed 24h demeaned return after news/new cards, one observation per ticker-day"))
# Dated deviations and errata. PREREG.md is frozen, so they are carried here (and as trials.log entries) and
# shown in the report summary and section 9b; git commit times are authoritative, hand-written stamps are not.
DEVIATIONS = (
    {"date": "2026-09-23", "id": "timestamps",
     "title": "hand-written freeze and trials.log stamps are wrong (git commit times are authoritative)",
     "text": ("PREREG.md says \"FROZEN 2026-09-23 ~14:45 UTC\", and trials.log entries 2-5 are stamped 12:10, 12:10, "
              "13:30 and 14:40 UTC; these stamps were written by hand ahead of time. Per git, entries 2-3 (v0 dev "
              "results) were committed in dac95b5 at 12:07:55 UTC, entry 4 (risk v1) in 2ed0093 at 12:59:36 UTC, and "
              "entry 5 (prompt-v1 check) with the freeze in 62286d9 at 13:32:47 UTC. The test runs started after "
              "62286d9; the freeze time printed in section 3 is read from git.")},
    {"date": "2026-09-23", "id": "code-after-freeze",
     "title": "replay.py and metrics.py edited after the freeze, before the llm/nonews/shuffled test runs",
     "text": ("sentiment/replay.py (adds the code_stamp field to metrics.json) and sentiment/metrics.py (daily "
              "statistics over complete UTC days, bootstrap path counts) were edited after the freeze commit and were "
              "uncommitted when the llm, nonews and shuffled test runs started (the baseline test run used the "
              "freeze-commit code). By inspection neither edit reaches a decision, risk action or fill (metrics are "
              "computed after the replay loop); this counts as verified only when a rerun from a clean checkout of "
              "the freeze commit ends on the same log_last_hash (freeze_rerun.json, section 9b).")},
    {"date": "2026-09-23", "id": "signal-test-c-lookahead",
     "title": "signal test (c): direction only from cards known at the event's entry",
     "text": ("Test (c) took each ticker-day's direction (and whether it counts) from all cards of that UTC day, "
              "including cards available hours after the entry bar its return is measured from: a look-ahead. Fixed "
              "before the test-window signal tests were run: the direction now comes only from the cards available "
              "at the event's entry bar; the all-day version is kept as a descriptive field (all_day_sign). Test "
              "(b) is unaffected (a rating ticker-day has one availability time).")},
    {"date": "2026-09-23", "id": "signal-test-window-end",
     "title": "signal tests: a forward window must end strictly before the window end",
     "text": ("The snapshot keeps only bars that close by the window end, so an exit exactly at the test window's "
              "end had no price (the dev window had one) and was counted as 'no IC' / 'no price'. Now every forward "
              "window must end strictly before the window end in both windows, and a missing price is its own skip "
              "reason. Changed before the test-window signal tests were run.")},
    {"date": "2026-09-23", "id": "project-split-paths",
     "title": "project split (075a0d4): config.py, evidence.py, llm.py and replay.py differ from the freeze in paths only",
     "text": ("The research repo was split into one directory per project (075a0d4 moved sentiment/ to "
              "b-sentiment-agent/sentiment/ without edits). After the move, config.py (data, snapshot, cache, out and "
              ".env locations relative to the project directory), evidence.py (news_items reads config.RAW), llm.py "
              "(reads config.ENV_FILE) and replay.py (git pathspecs of code_stamp; --cards-from written before the "
              "split resolved via config.run_dir) were edited for paths only, so the code-provenance list shows them "
              "as differing from the freeze commit. No decision, risk action, fill or metric changed: offline replays "
              "after the edits end on the same log_last_hash as the shipped runs.")},
)
# Five-line digest of docs/DIAGNOSIS-dev.md (synthesis and section 2), qualitative on purpose: the
# numbers stay in that document, where each one carries its own derivation.
DIAGNOSIS_DIGEST = (
    ("The v0 LLM's dev loss is real against flat, but it cannot be told apart from the fixed-rule baseline or from "
     "long or short equal-weight basket benchmarks; costs are a small part of it and the risk layer did not cause it."),
    ("The largest piece is market and crypto-sector timing: the book was on the wrong side of the swings of the "
     "crypto-beta names, not structurally tilted."),
    ("A flat stop and sizing that ignored volatility sent most of the trip loss through MSTR, COIN, HOOD and CRCL; "
     "this motivated risk v1 (vol-scaled caps and stops, beta neutralisation), fixed before the test window."),
    ("No exploitable sentiment signal in dev: the baseline score has no predictive rank IC, cards mostly describe the "
     "move that already happened, fresh news does not beat costs, and no test clears a multiple-testing bar."),
    ("The test window is a different regime (crypto fear flips to greed, far fewer earnings), so dev results should "
     "not be used to predict the test window in either direction."),
)


# ---------------------------------------------------------------- helpers

def _clean(x):
    """JSON-safe: non-finite floats -> None, numpy scalars -> Python, timestamps -> ISO strings."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def _num(x) -> float | None:
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _short(s, n: int = TEXT_MAX) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _days(start: str, end: str) -> int:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days + 1


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rel(p) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------- discovery

def run_name(variant: str, version: str, start: str, end: str) -> str:
    return f"{variant}_{start}_{end}" if version == "v0" else f"{variant}_{version}_{start}_{end}"


def _window_of(start: str, end: str) -> str:
    return next((w for w, se in WINDOWS.items() if se == (start, end)), f"{start}..{end}")


def discover(out_root: Path) -> list[dict]:
    """The expected runs (EXPECTED), every other run directory found, and live."""
    out_root = Path(out_root)
    specs, seen = [], set()

    def add(variant, version, start, end, optional=False, expected=True):
        name = run_name(variant, version, start, end)
        if name in seen:
            return
        seen.add(name)
        specs.append({"run_id": name, "window": _window_of(start, end), "variant": variant, "version": version,
                      "start": start, "end": end, "mode": "replay", "dir": str(out_root / name),
                      "optional": optional, "expected": expected})

    for w, ver, v, opt in EXPECTED:
        add(v, ver, *WINDOWS[w], optional=opt)
    if out_root.exists():
        for d in sorted(out_root.iterdir()):
            g = RUN_RE.match(d.name) if d.is_dir() else None
            if g and g["variant"] in VARIANTS:
                add(g["variant"], g["tag"] or "v0", g["start"], g["end"], expected=False)
    specs.append({"run_id": "live", "window": "live", "variant": "llm", "version": None, "start": None, "end": None,
                  "mode": "live", "dir": str(out_root / "live"), "optional": False, "expected": True})
    return specs


def _count_lines(p: Path) -> int:
    try:
        with p.open() as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return 0


def run_status(spec: dict) -> str:
    d = Path(spec["dir"])
    if not d.is_dir():
        return "not run yet"
    if spec["mode"] == "live":
        return "live" if (d / "equity.csv").exists() else "incomplete"
    return "complete" if (d / "metrics.json").exists() and (d / "equity.csv").exists() else "incomplete"


# ---------------------------------------------------------------- per run

def live_modes(d: Path) -> list[str]:
    return sorted(p.stem[: -len("_state")] for p in Path(d).glob("*_state.json") if p.stem != "runner_state")


def live_label(modes: list[str]) -> str:
    if len(modes) == 1 and modes[0] in LIVE_LABEL:
        return LIVE_LABEL[modes[0]]
    if not modes:
        return UNKNOWN_LIVE_LABEL
    return "mixed: " + "; ".join(f"{m} {LIVE_LABEL.get(m, 'unknown mode')}" for m in modes)


CODE_FILES = ("config", "data", "features", "evidence", "agent", "baseline", "risk", "sim", "replay", "llm",
              "ledger", "metrics")


def code_hash() -> str:
    """sha256 over the replay code path (sentiment/{CODE_FILES}.py and the prompts): the legacy
    `code_stamp` format; a metrics.json carrying a 64-hex stamp is checked against it."""
    h = hashlib.sha256()
    files = [ROOT / "sentiment" / f"{m}.py" for m in CODE_FILES]
    for p in files + sorted(PROMPTS.glob("*.md")):
        h.update(p.name.encode() + b"\0" + (p.read_bytes() if p.exists() else b"") + b"\0")
    return h.hexdigest()


def current_item_ids() -> set[str] | None:
    """10-char id prefixes of every item the current data rules build from the snapshot (None if unreadable)."""
    try:
        from sentiment import data
        return {it["id"][:10] for it in data.build_items(data.load_snapshot())}
    except READ_ERRORS:
        return None


def provenance(d: Path, recs: list[dict], item_ids: set[str] | None, legacy_hash: str | None) -> dict:
    """Is a replay a product of the current data rules? Its cited evidence items must exist under the current
    item ids, and a legacy sha256 code_stamp must equal the current code hash."""
    cited = sorted({str(c).rsplit("-", 1)[0][:10] for r in recs for c in r.get("evidence_ids") or []})
    rec = _read_json(d / "metrics.json") or {}
    out = {"n_evidence_items": len(cited), "why": []}
    if item_ids is None:
        out.update(n_unknown_items=None, items_checked=False)
    else:
        unknown = [c for c in cited if c not in item_ids]
        out.update(n_unknown_items=len(unknown), unknown_examples=unknown[:5], items_checked=True)
        if unknown:
            out["why"].append(f"{len(unknown)} of {len(cited)} cited evidence items do not exist under the current "
                              "data rules: made under an older availability rule, e.g. before the UTC+8 news fix")
    st = rec.get("code_stamp")
    if legacy_hash and isinstance(st, str) and re.fullmatch(r"[0-9a-f]{64}", st) and st != legacy_hash:
        out["why"].append("metrics.json code_stamp (sha256) differs from the current replay code")
    out["stale"] = bool(out["why"])
    return out


def evaluate(spec: dict, item_ids: set[str] | None = None, legacy_hash: str | None = None) -> dict:
    """Status, recomputed metrics, provenance and ledger check of one run (read only)."""
    d = Path(spec["dir"])
    row = {k: spec[k] for k in ("run_id", "window", "variant", "version", "start", "end", "mode", "optional",
                                "expected")}
    row["status"] = run_status(spec)
    row["live_modes"] = live_modes(d) if spec["mode"] == "live" and d.is_dir() else []
    row["label"] = live_label(row["live_modes"]) if spec["mode"] == "live" else REPLAY_LABEL
    log = d / "log.jsonl"
    row["n_log_records"] = _count_lines(log) if log.exists() else 0
    if row["status"] in ("not run yet", "incomplete"):
        row["note"] = (("optional run (PREREG: only if Qwen credit remains)" if spec["optional"] else "directory missing")
                       if row["status"] == "not run yet" else
                       f"no metrics.json yet ({row['n_log_records']} log records so far): in progress or aborted")
        return row
    try:
        m = metrics.compute(d)
    except READ_ERRORS as exc:                                 # a live file mid-write, an empty curve, ...
        row.update(status="unreadable", note=f"metrics.compute failed: {type(exc).__name__}: {exc}")
        return row
    rec = _read_json(d / "metrics.json") or {}
    row["metrics"] = m
    row["n_items"], row["n_cards"] = rec.get("n_items"), rec.get("n_cards")
    row["risk_version"] = rec.get("risk_version") or ("v0" if spec["mode"] == "replay" else None)
    row["prompt_version"] = rec.get("prompt_version") if "prompt_version" in rec else (
        None if spec["variant"] == "baseline" or spec["mode"] == "live" else "v0")
    row["code_stamp"] = rec.get("code_stamp")
    row["cards_from"] = rec.get("cards_from")
    notes = []
    if spec["mode"] == "replay":
        row["provenance"] = provenance(d, metrics.load_records(d), item_ids, legacy_hash)
        if row["provenance"]["stale"]:
            row["status"], row["label"] = "stale", f"{REPLAY_LABEL}, stale"
            notes += row["provenance"]["why"]
        if spec["version"] != row["risk_version"]:
            notes.append(f"directory tag {spec['version']} but metrics.json risk_version {row['risk_version']}")
    row["log_verified"] = ledger.verify(log) if log.exists() else None
    last = None
    if log.exists() and row["n_log_records"]:
        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        last = json.loads(lines[-1]).get("hash")
    row["log_last_hash"] = last
    row["recorded_last_hash"] = rec.get("log_last_hash")
    row["hash_matches_recorded"] = (last == rec.get("log_last_hash")) if rec else None
    row["n_hours"] = len(metrics.load_equity(d))
    row["note"] = "; ".join(notes) or None
    return row


# ---------------------------------------------------------------- git provenance

PREREG_DEVIATIONS = "## Deviations"


def prereg_parts(text: str) -> tuple[str, str]:
    """(the fixed part of PREREG.md above "## Deviations", the Deviations section's body). PREREG allows dated
    additions under Deviations after the freeze, so only the fixed part must stay identical."""
    head, sep, tail = text.partition("\n" + PREREG_DEVIATIONS)
    return head.rstrip(), tail.strip() if sep else ""


def _sha(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode()).hexdigest()


def _utc(t: str | None) -> str | None:
    try:
        return pd.Timestamp(t).tz_convert("UTC").isoformat() if t else None
    except (ValueError, TypeError):
        return None


def git_info(provenance_file: Path | None = None) -> dict:
    """HEAD, the commit that froze PREREG.md (the first commit whose PREREG.md says **FROZEN**) and its git
    commit time, and whether PREREG's fixed part (above "## Deviations") changed since; Deviations entries
    are reported separately. In the exported public repo the research repo's history is absent: the freeze
    facts then come from PROVENANCE.json, written by scripts/export_public.sh."""
    pf = Path(provenance_file) if provenance_file else PROVENANCE_FILE
    prov = _read_json(pf) if pf.exists() else None
    cur = PREREG_FILE.read_text() if PREREG_FILE.exists() else None
    fixed_now, dev_now = prereg_parts(cur) if cur is not None else (None, None)
    out = {"head": _git("rev-parse", "--short", "HEAD"),
           "prereg_sha256": sha256_file(PREREG_FILE) if PREREG_FILE.exists() else None,
           "prereg_deviations_now": dev_now}
    if isinstance(prov, dict) and prov.get("prereg_freeze_commit"):
        out.update(source=f"{_rel(pf)} (exported repo; the research repo's git history is not shipped)",
                   prereg_freeze_commit=prov["prereg_freeze_commit"], prereg_freeze_time=prov.get("prereg_freeze_time"))
        fixed_sha, dev_then = prov.get("prereg_fixed_sha256_at_freeze"), prov.get("prereg_deviations_at_freeze")
        out["export_code_checks"] = prov.get("code_checks") or {}
    else:
        freeze, path = prereg_freeze()
        then = _git("show", f"{freeze}:{path}") if freeze else None
        fixed_then, dev_then = prereg_parts(then) if then is not None else (None, None)
        fixed_sha = _sha(fixed_then)
        out.update(source="git", prereg_freeze_commit=freeze,
                   prereg_freeze_time=_git("show", "-s", "--format=%cI", freeze) if freeze else None)
    out["prereg_freeze_time_utc"] = _utc(out["prereg_freeze_time"])
    out["prereg_fixed_sha256_at_freeze"] = fixed_sha
    out["prereg_deviations_at_freeze"] = dev_then
    out["prereg_changed_since_freeze"] = (None if fixed_sha is None or fixed_now is None
                                          else _sha(fixed_now) != fixed_sha)
    out["prereg_deviations_added"] = (None if dev_then is None or dev_now is None else dev_now != dev_then)
    return out


def prereg_freeze() -> tuple[str | None, str | None]:
    """(short sha, PREREG.md's path at that commit relative to the git top level) of the first commit whose
    PREREG.md says **FROZEN**, following the file across renames (sentiment/PREREG.md before the project split
    075a0d4, b-sentiment-agent/docs/PREREG.md after it). (None, None) outside git or when never frozen."""
    log = _git("log", "--follow", "--format=%x00%h", "--name-only", "-S**FROZEN**", "--", _rel(PREREG_FILE))
    hits = [c.split() for c in (log or "").split("\0") if c.strip()]   # newest first; --follow cannot --reverse
    return (hits[-1][0], hits[-1][-1]) if hits and len(hits[-1]) >= 2 else (None, None)


# the replay's code path under git: exactly the modules a replay imports (CODE_FILES), the prompts, the
# snapshot and the quoted spreads, relative to ROOT. report.py, signal_tests.py, live.py, reconcile.py,
# fetch_*.py are not on it. LEGACY_REPLAY_CODE_PATHS: the same files in the research repo before the project
# split (075a0d4), relative to the git top level (A4's static/quoted_spread.csv is vendored byte-identical).
REPLAY_CODE_PATHS = tuple(f"sentiment/{m}.py" for m in CODE_FILES) + ("sentiment/prompts", "snapshot",
                                                                        "static/quoted_spread.csv")
LEGACY_REPLAY_CODE_PATHS = tuple(f"sentiment/{m}.py" for m in CODE_FILES) + ("sentiment/prompts", "sentiment/snapshot",
                                                                               "static/quoted_spread.csv")


def _code_pathspecs() -> list[str]:
    """REPLAY_CODE_PATHS as git pathspecs from the top level, plus the pre-split paths when ROOT is a
    subdirectory of the checkout (the research monorepo), so commits on both sides of the split compare."""
    prefix = _git("rev-parse", "--show-prefix") or ""
    specs = [f":(top){prefix}{p}" for p in REPLAY_CODE_PATHS]
    return specs + ([f":(top){p}" for p in LEGACY_REPLAY_CODE_PATHS] if prefix else [])


def _changed(name_status: str) -> list[str]:
    """Paths of a `git diff --name-status -M` listing, pure renames (R100: the split moved the file, content
    identical) left out."""
    out = []
    for ln in name_status.splitlines():
        f = ln.split("\t")
        if len(f) >= 2 and f[0] != "R100":
            out.append(f[-1])
    return out


def code_check(stamp, freeze: str | None) -> dict:
    """A run's git code_stamp against the PREREG freeze commit: same commit, which replay-path files
    (REPLAY_CODE_PATHS) differ, or not checkable (no stamp, legacy sha256 stamp, unknown commit). A "-dirty"
    stamp is written by sentiment.replay when any sentiment/*.py, prompt, snapshot or spread file was
    uncommitted at the start of the run, replay path or not: git cannot show which."""
    if not stamp:
        return {"stamp": None, "check": "no code_stamp recorded (run made before replay wrote one)"}
    if re.fullmatch(r"[0-9a-f]{64}", str(stamp)):
        return {"stamp": stamp, "check": "legacy sha256 stamp (checked by provenance)"}
    sha, dirty = str(stamp).removesuffix("-dirty"), str(stamp).endswith("-dirty")
    out = {"stamp": stamp, "commit": sha, "dirty": dirty}
    if not freeze:
        return {**out, "check": "PREREG freeze commit unknown"}
    files = _git("diff", "--name-status", "-M", freeze, sha, "--", *_code_pathspecs())
    if files is None:
        return {**out, "check": f"commit {sha} not found in this checkout"}
    out["files_differing_from_freeze"] = _changed(files)
    same = not out["files_differing_from_freeze"]
    out["check"] = ((f"replay code identical to the PREREG freeze commit {freeze}" if same else
                     f"replay code differs from the PREREG freeze commit {freeze} in "
                     + ", ".join(out["files_differing_from_freeze"]))
                    + ("; the run also had uncommitted changes (-dirty), not verifiable from git" if dirty else ""))
    out["matches_freeze"] = same and not dirty
    return out


def replay_path_changes(freeze: str | None) -> list[str] | None:
    """Replay-path files that differ now (working tree, untracked included) from the freeze commit."""
    if not freeze:
        return None
    specs = _code_pathspecs()
    diff = _git("diff", "--name-status", "-M", freeze, "--", *specs)
    new = _git("ls-files", "--others", "--exclude-standard", "--full-name", "--", *specs)
    if diff is None or new is None:
        return None
    return sorted(set(_changed(diff)) | set(new.split()))


def freeze_rerun(path: Path, runs: list[dict]) -> dict:
    """<out>/freeze_rerun.json: {"freeze_commit": sha, "runs": {run_id: {"log_last_hash": h}}}, the last hashes
    of test runs replayed offline from a clean checkout of the freeze commit. A shipped run whose hash equals
    its rerun's ran the freeze-commit replay code in effect, whatever its code_stamp says."""
    d = _read_json(path)
    if not isinstance(d, dict):
        return {"status": "not run yet", "path": _rel(path), "runs": {}}
    got = d.get("runs") or {}
    by = {r["run_id"]: r for r in runs}
    res = {}
    for rid, x in got.items():
        r = by.get(rid) or {}
        mine, theirs = r.get("log_last_hash"), (x or {}).get("log_last_hash")
        res[rid] = {"rerun_last_hash": theirs, "run_last_hash": mine,
                    "identical": None if not (mine and theirs) else mine == theirs}
    return {"status": "present", "path": _rel(path), "freeze_commit": d.get("freeze_commit"), "runs": res}


def code_provenance(runs: list[dict], git: dict, rerun: dict) -> dict:
    """Did the pre-registered test runs use the freeze-commit replay code? Per run: its code check, and
    whether a rerun from the freeze commit reproduced its log (freeze_rerun). One verdict line for the
    README and the form."""
    freeze = git.get("prereg_freeze_commit")
    rows = {}
    for r in runs:
        if r["mode"] != "replay" or r["window"] != "test" or r["status"] != "complete":
            continue
        c = r.get("code") or {}
        rr = (rerun.get("runs") or {}).get(r["run_id"]) or {}
        verified = c.get("matches_freeze") is True or rr.get("identical") is True
        rows[r["run_id"]] = {"stamp": c.get("stamp"), "check": c.get("check"),
                             "matches_freeze": c.get("matches_freeze"),
                             "rerun_identical": rr.get("identical"), "verified": verified}
    unverified = sorted(k for k, v in rows.items() if not v["verified"])
    broken = sorted(k for k, v in rows.items() if v["rerun_identical"] is False)
    if not rows:
        verdict = "no complete test-window run yet"
    elif broken:
        verdict = (f"a rerun from the freeze commit {freeze} does NOT reproduce {', '.join(broken)}: those runs did "
                   "not use the frozen replay code")
    elif unverified:
        verdict = (f"{len(unverified)} of {len(rows)} test runs are not yet verified against the freeze commit "
                   f"{freeze} ({', '.join(unverified)}): they started on a working tree with replay-path edits made "
                   "after the freeze (see Deviations); verified only by a rerun from a clean checkout of the freeze "
                   "commit ending on the same log_last_hash")
    else:
        verdict = (f"all {len(rows)} test runs used the freeze-commit {freeze} replay code (git stamp, or a rerun "
                   "from the freeze commit ending on the same log_last_hash)")
    return {"freeze_commit": freeze, "freeze_time_utc": git.get("prereg_freeze_time_utc"),
            "replay_path_changed_since_freeze": replay_path_changes(freeze) if git.get("source") == "git" else None,
            "rerun": rerun, "runs": rows, "n_unverified": len(unverified), "verdict": verdict}


# ---------------------------------------------------------------- increment and controls

def _ci_side(ci) -> str:
    lo, hi = (ci or [None, None])
    if lo is None or hi is None:
        return "n/a"
    return "above 0" if lo > 0 else "below 0" if hi < 0 else "includes 0"


READING = {
    "baseline": {"above 0": "the LLM agent beats the fixed-rule baseline",
                 "below 0": "the LLM agent trails the fixed-rule baseline",
                 "includes 0": "the LLM agent is not distinguishable from the fixed-rule baseline"},
    # one control alone never establishes a news effect (PREREG: llm must beat BOTH nonews and shuffled), so
    # the "above 0" texts only describe the comparison; the combined claim is readings.information.verdict
    "nonews": {"above 0": ("llm beats nonews (95% CI above 0); on its own this does not establish a news effect "
                           "(PREREG needs llm to beat both nonews and shuffled)"),
               "below 0": "the agent does better without news: the result is not a news effect",
               "includes 0": "the agent without news is not distinguishable from it: the result cannot be attributed to news"},
    "shuffled": {"above 0": ("llm beats shuffled (95% CI above 0); on its own this does not establish a "
                             "news-timing effect (PREREG needs llm to beat both nonews and shuffled)"),
                 "below 0": "stale (shuffled) news does better: the result is not a news-timing effect",
                 "includes 0": "stale (shuffled) news does as well: no evidence that news timing matters"},
    "blinded": {"above 0": "names and dates help: consistent with the model using name priors (a leakage risk)",
                "below 0": "the blinded agent does better: no sign that name priors help",
                "includes 0": "blinding names and dates makes no significant difference: no sign of reliance on name priors"},
}


def reading(inc: dict, other: str, claim: bool = True) -> str:
    """The increment in words. `claim=False` (any window but the pre-registered test window): the difference
    and CI only, no claim sentence (an in-sample check whose P&L is not a criterion)."""
    if (inc.get("n_days") or 0) < MIN_DAYS_READ:
        return (f"llm − {other} Sharpe {_f(inc.get('sharpe_diff'))} on {inc.get('n_days')} common days: "
                f"n < {MIN_DAYS_READ}, CI and P(diff ≤ 0) not computed; too short to read either way.")
    side = _ci_side(inc.get("ci95"))
    ci = inc.get("ci95") or [None, None]
    base = (f"llm − {other} Sharpe {_f(inc.get('sharpe_diff'))}, 95% CI [{_f(ci[0])}, {_f(ci[1])}] "
            f"({side}, {inc.get('n_days')} days, {inc.get('n_boot_used')} of {inc.get('n_boot')} bootstrap paths used)")
    if side == "n/a":
        return f"{base}: no finite bootstrap; not readable."
    if not claim:
        return f"{base}: in-sample check outside the pre-registered test window; P&L not a criterion, no claim read."
    return f"{base}: {READING[other][side]}."


def distinct_paths(n_days: int, inc: dict) -> int:
    """Distinct resampling paths of `metrics.increment`'s bootstrap (same seed and index draw): on a few days
    most paths repeat."""
    if not n_days:
        return 0
    idx = metrics.stationary_bootstrap_idx(n_days, int(inc["n_boot"]), inc["block_days"],
                                           np.random.default_rng(inc.get("seed", metrics.SEED)))
    return len({tuple(i) for i in idx})


def comparisons(runs: list[dict]) -> dict:
    """Per window and version: llm vs baseline and llm vs each control (stationary block bootstrap). Only
    complete, current (not stale) runs are compared."""
    by = {(r["window"], r["version"], r["variant"]): r for r in runs if r["mode"] == "replay"}
    out = {}
    for w, ver in dict.fromkeys((r["window"], r["version"]) for r in runs if r["mode"] == "replay"):
        llm_run = by.get((w, ver, "llm"))
        entry = {"window": w, "version": ver, "llm_run": llm_run["run_id"] if llm_run else None,
                 "llm_status": llm_run["status"] if llm_run else "not run yet", "vs": {}}
        for other in ("baseline", *CONTROLS):
            o = by.get((w, ver, other))
            item = {"run": o["run_id"] if o else None, "status": o["status"] if o else "not run yet"}
            if llm_run and llm_run["status"] == "complete" and o and o["status"] == "complete":
                try:
                    inc = metrics.increment(llm_run["dir_path"], o["dir_path"])
                    inc["n_distinct_paths"] = distinct_paths(inc["n_days"], inc)
                    inc["run_a"], inc["run_b"] = llm_run["run_id"], o["run_id"]
                    item.update(increment=inc, reading=reading(inc, other, claim=w == "test"))
                except READ_ERRORS as exc:
                    item.update(reading=f"increment failed: {type(exc).__name__}: {exc}")
            else:
                missing = [f"{x} {r['status'] if r else 'not run yet'}" for x, r in (("llm", llm_run), (other, o))
                           if not r or r["status"] != "complete"]
                item["reading"] = f"not computed: {', '.join(missing)} (only complete, current runs are compared)"
            entry["vs"][other] = item
        out[f"{w} {ver}"] = entry
    return out


# ---------------------------------------------------------------- market data (snapshot)

def load_perp(snap: Path) -> dict[str, pd.DataFrame]:
    """Universe 1h bars from the snapshot (index = bar open, UTC); {} when missing."""
    p = Path(snap) / "perp.parquet"
    if not p.exists():
        return {}
    try:
        d = pd.read_parquet(p, columns=["sym", "ts", "open", "close"])
    except (OSError, ValueError, KeyError):
        return {}
    out = {}
    for tk in UNIVERSE:
        g = d[d["sym"] == perp(tk)].sort_values("ts")
        if len(g):
            out[tk] = pd.DataFrame({"open": g["open"].to_numpy(float), "close": g["close"].to_numpy(float)},
                                   index=pd.DatetimeIndex(pd.to_datetime(g["ts"].to_numpy(), unit="ms", utc=True)))
    return out


def basket_daily(bars: dict[str, pd.DataFrame]) -> pd.Series:
    """Equal-weight universe basket daily return, labelled by its closing midnight (as metrics.daily_returns):
    each name's daily close is the close of its last bar opening that day (close time <= the next midnight);
    returns over consecutive days only; basket = mean of the names' returns that day."""
    rets = {}
    for tk, b in bars.items():
        c = b["close"].groupby(b.index.floor("D")).last()
        c.index = c.index + pd.Timedelta(days=1)
        full = c.reindex(pd.date_range(c.index[0], c.index[-1], freq="D"))
        rets[tk] = full / full.shift(1) - 1
    return pd.DataFrame(rets).mean(axis=1, skipna=True).dropna() if rets else pd.Series(dtype=float)


def realised_beta(run_dir, basket: pd.Series) -> dict:
    """OLS beta of the book's daily GROSS return (before fees, spread, funding; complete days) on the EW basket."""
    if basket is None or not len(basket):
        return {"status": "no basket (snapshot perp bars missing)"}
    e = metrics.load_equity(run_dir)
    col = "gross_equity" if "gross_equity" in e else "equity"
    r = metrics.complete_day_returns(e[col])
    j = pd.concat([r.rename("r"), basket.rename("b")], axis=1, join="inner").dropna()
    n = len(j)
    if n < 3 or float(j["b"].var(ddof=1)) <= 0:
        return {"status": f"too few days ({n})", "n_days": n}
    x, y = j["b"].to_numpy(), j["r"].to_numpy()
    xc = x - x.mean()
    beta = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    alpha = float(y.mean() - beta * x.mean())
    resid = y - alpha - beta * x
    se = math.sqrt(float((resid ** 2).sum()) / (n - 2) / float((xc ** 2).sum())) if n > 2 else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"status": "computed", "basis": f"daily {col.replace('_', ' ')} returns vs EW basket", "n_days": n,
            "beta": beta, "t": beta / se if se and se > 0 else None,
            "r2": 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else None,
            "alpha_bp_day": alpha * 1e4, "within_band": abs(beta) <= BETA_BAND}


# ---------------------------------------------------------------- risk layer

def _next_decision(t: pd.Timestamp) -> pd.Timestamp:
    for k in range(1, 25):
        if (t + pd.Timedelta(hours=k)).hour in DECISION_HOURS:
            return t + pd.Timedelta(hours=k)
    return t + pd.Timedelta(hours=4)


def _open_at(bars: dict[str, pd.DataFrame], tk: str, t: pd.Timestamp) -> float | None:
    b = bars.get(tk)
    if b is None:
        return None
    i = b.index.searchsorted(t, side="left")
    if i >= len(b) or b.index[i] - t >= pd.Timedelta(hours=1):
        return None
    return _num(b["open"].iloc[i])


def risk_counts(recs: list[dict], bars: dict[str, pd.DataFrame] | None = None) -> dict:
    """Binding-rule counts, split by where the rule bound (decision-time apply vs hourly check), the weight each
    moved, and an estimated first-order P&L effect: (after - before) x equity x the name's move from the bar
    opening at the record to the bar opening at the next decision, frictionless (rules chain, so the effects
    add up; path effects such as cooldowns and later decisions are ignored)."""
    c = {"decision": Counter(), "hourly_check": Counter()}
    moved, effect, priced = Counter(), Counter(), Counter()
    bars = bars or {}
    for r in recs:
        src = "decision" if r.get("event", "decision") == "decision" else "hourly_check"
        t = _ts(r["ts"]) if r.get("ts") else None
        eq = _num(r.get("equity"))
        th = _next_decision(t) if t is not None else None
        for a in r.get("risk_actions") or []:
            rule = a["rule"]
            c[src][rule] += 1
            b, af = _num(a.get("before")), _num(a.get("after"))
            if rule == "daily_kill" or b is None or af is None:
                continue
            moved[rule] += abs(b - af)
            tk = a.get("ticker")
            if tk and t is not None and eq:
                p0, p1 = _open_at(bars, tk, t), _open_at(bars, tk, th)
                if p0 and p1:
                    effect[rule] += (af - b) * eq * (p1 / p0 - 1)
                    priced[rule] += 1
    rules = sorted(set(c["decision"]) | set(c["hourly_check"]),
                   key=lambda k: RULE_PRIORITY.index(k) if k in RULE_PRIORITY else 99)
    return {"by_rule": {k: {"decision": c["decision"][k], "hourly_check": c["hourly_check"][k],
                            "weight_moved": moved[k], "pnl_effect_est": effect[k] if priced[k] else None,
                            "n_priced": priced[k], "meaning": RULE_TEXT.get(k, "")} for k in rules},
            "n_records_with_action": sum(1 for r in recs if r.get("risk_actions")),
            "n_records": len(recs), "effects_priced": bool(bars)}


# ---------------------------------------------------------------- requested -> risk -> executed -> fills

def _fill(f: dict) -> dict:
    return {k: f.get(k) for k in ("ticker", "side", "qty", "price", "fee", "half_spread_cost", "ts", "order_id")}


def _signed_notional(f: dict) -> float:
    q, p = _num(f.get("qty")), _num(f.get("price"))
    if not q or not p or q <= 0:
        return 0.0
    return q * p * (1 if f.get("side") == "buy" else -1)


def prompt_cards(req_hash: str | None, llm_cache: Path = LLM_CACHE) -> dict[str, dict] | None:
    """Evidence lines of the decision prompt the model received (the cached request of `req_hash`), by card
    id, parsed by the EVIDENCE header (v0: no since_bp column; v1: since_bp); None when not cached."""
    rec = _read_json(Path(llm_cache) / f"{req_hash}.json") if req_hash else None
    try:
        user = rec["request"]["messages"][-1]["content"]
    except (TypeError, KeyError, IndexError):
        return None
    out, cols = {}, None
    for ln in user.splitlines():
        parts = [p.strip() for p in ln.split(" | ")]
        if parts[0] == "id" and "summary" in parts:
            cols = parts
            continue
        if cols is None or len(parts) < len(cols):
            continue
        row = dict(zip(cols[:-1], parts[: len(cols) - 1]))
        subj = row.get("scope subject", "").split(" ", 1)
        try:
            stance = int(row.get("stance", ""))
        except ValueError:
            continue
        novelty = row.get("novelty", "")
        out[parts[0]] = {"age_h": _num(row.get("age_h")), "since_bp": row.get("since_bp"), "scope": subj[0],
                         "subject": subj[1] if len(subj) > 1 else "", "stance": stance,
                         "strength": _num(row.get("strength")), "horizon_h": _num(row.get("horizon_h")),
                         "novelty": "post_hoc" if novelty.startswith("POST_HOC") else novelty,
                         "summary": " | ".join(parts[len(cols) - 1:])}
    return out


def _shown_card(cid: str, t: pd.Timestamp, prompt: dict[str, dict] | None, blinded: bool) -> dict:
    """A cited card as the decision prompt showed it (never from the mutable card cache)."""
    p = (prompt or {}).get(cid)
    if p is None:
        return {"card_id": cid, "in_prompt_text": False}
    tks = []
    if p["scope"] == "name":
        from sentiment.agent import UNALIAS
        tks = [UNALIAS.get(x, x) if blinded else x for x in p["subject"].split(",") if x]
    age = p.get("age_h")
    return {"card_id": cid, "in_prompt_text": True, "scope": p["scope"], "tickers": tks,
            "subject": p["subject"], "stance": p["stance"], "strength": p["strength"], "novelty": p["novelty"],
            "age_h": age, "available_at": (t - pd.Timedelta(hours=age)).isoformat() if age is not None else None,
            "since_bp": p.get("since_bp"), "summary": _short(p["summary"])}


def decision_chains(run_id: str, recs: list[dict], n: int = N_CHAINS, llm_cache: Path = LLM_CACHE) -> list[dict]:
    """The n decision records with the largest executed rebalance (sum over names of |signed fill notional| /
    equity), each as requested (model target) -> risk actions rule by rule -> executed (post-risk target and the
    weight the fills moved) -> fills, per name. Cited cards: only those in the record's logged shown set
    (`evidence_ids`), with the text the cached prompt showed."""
    cand, prev_by_i, prev = [], {}, {}
    for i, r in enumerate(recs):
        prev_by_i[i] = dict(prev)
        eq = _num(r.get("equity"))
        if r.get("event", "decision") == "decision" and r.get("decision") and eq:
            size = sum(abs(_signed_notional(f)) for f in r.get("fills") or []) / eq
            if size > 0:
                cand.append((size, i))
        prev = {k: float(v) for k, v in (r.get("targets") or {}).items()}
    cand.sort(key=lambda c: (-c[0], c[1]))
    out = []
    for size, i in cand[:n]:
        r = recs[i]
        t = _ts(r["ts"])
        eq = float(r["equity"])
        dec = r["decision"] or {}
        req = {k: float(v) for k, v in (dec.get("targets") or {}).items() if _num(v) is not None}
        fin = {k: float(v) for k, v in (r.get("targets") or {}).items()}
        before = prev_by_i[i]
        book = ((r.get("risk_ctx") or {}).get("book")) or None
        acts: dict[str | None, list[dict]] = {}
        for a in r.get("risk_actions") or []:
            acts.setdefault(a.get("ticker"), []).append(a)
        exe: dict[str, float] = {}
        fills: dict[str, list[dict]] = {}
        for f in r.get("fills") or []:
            exe[f["ticker"]] = exe.get(f["ticker"], 0.0) + _signed_notional(f) / eq
            fills.setdefault(f["ticker"], []).append(_fill(f))
        rows = []
        for tk in UNIVERSE:
            steps = [{"rule": a["rule"], "before": a.get("before"), "after": a.get("after")} for a in acts.get(tk, [])]
            moved = any(abs((_num(s["before"]) or 0) - (_num(s["after"]) or 0)) >= EPS_W for s in steps
                        if _num(s["before"]) is not None and _num(s["after"]) is not None)
            rq = req.get(tk, 0.0)
            if not (abs(rq - before.get(tk, 0.0)) >= EPS_W or tk in fills or moved):
                continue
            rows.append({"ticker": tk, "prev_target": before.get(tk, 0.0),
                         "book_before": (book or {}).get(tk, 0.0) if book is not None else None,
                         "requested": rq, "risk_steps": steps, "final_target": fin.get(tk, 0.0),
                         "executed_dw": exe.get(tk, 0.0), "fills": fills.get(tk, [])})
        shown = [str(c) for c in r.get("evidence_ids") or []]
        shown_set = set(shown)
        prompt = prompt_cards(r.get("llm_request_hash"), llm_cache)
        blinded = r.get("variant") == "blinded"
        focus = sorted((x for x in rows if x["fills"]), key=lambda x: -abs(x["executed_dw"]))[:N_CHAIN_FOCUS]
        for x in focus:
            cited = [str(c) for c in (dec.get("evidence") or {}).get(x["ticker"], [])]
            x["reason"] = _short((dec.get("reasons") or {}).get(x["ticker"]), 600)
            x["cards"] = [_shown_card(c, t, prompt, blinded) for c in cited if c in shown_set]
            x["cited_not_shown"] = [c for c in cited if c not in shown_set]
            for k in x["cards"]:
                k["off_ticker"] = k.get("scope") == "name" and x["ticker"] not in (k.get("tickers") or [])
        out.append({"run": run_id, "ts": r["ts"], "equity": eq, "executed_size": size,
                    "confidence": dec.get("confidence"), "valid": dec.get("valid"),
                    "prompt_version": r.get("prompt_version") or ("v0" if r.get("variant") != "baseline" else None),
                    "risk_version": r.get("risk_version") or "v0",
                    "n_cards_shown": len(shown), "prompt_cached": prompt is not None,
                    "book_actions": [{"rule": a["rule"], "before": a.get("before"), "after": a.get("after")}
                                     for a in acts.get(None, [])],
                    "rows": rows, "focus": [x["ticker"] for x in focus],
                    "llm_request_hash": r.get("llm_request_hash"), "record_hash": r.get("hash")})
    return out


# ---------------------------------------------------------------- PREREG readings

def signal_tests(path: Path, window: tuple[str, str] | None = None) -> dict:
    """signal_tests_{test,dev}.json (sentiment/signal_tests.py) -> rows with the |t| > SIGNAL_T_BAR bar.
    Accepted shapes: {"tests": {id: {...}}} (what signal_tests.py writes), {"tests": [...]}, a list, or {id: {...}};
    each test gives a t statistic (`t`, `t_stat`, `nw_t` or `t_nw`) and optionally `estimate_bp` (shown in bp) or
    `estimate` (or ic / mean / value), `n`. Pre-registered ids are matched first. `window` = the expected
    (start, end): a file for another window is flagged.

    Each row: `significant` (|t| > SIGNAL_T_BAR) and `status` in the report's label vocabulary: "not significant",
    else "observed (replay prices)" on the pre-registered test window and "in-sample" on any other window."""
    d = _read_json(path)
    pre = [{"id": i, "what": w} for i, w in SIGNAL_TESTS]
    if d is None:
        return {"status": "not run yet", "path": _rel(path), "bar": SIGNAL_T_BAR,
                "rows": [{**p, "status": "not run yet"} for p in pre]}
    tests = d.get("tests") if isinstance(d, dict) and "tests" in d else d
    if isinstance(tests, dict):
        tests = [{"id": k, **v} for k, v in tests.items() if isinstance(v, dict)]
    elif not isinstance(tests, list):
        tests = []
    tests = [x for x in tests or [] if isinstance(x, dict)]
    pick = lambda x, ks: next((x[k] for k in ks if k in x and x[k] is not None), None)  # noqa: E731
    parsed = []
    for x in tests:
        tv = _num(pick(x, ("t", "t_stat", "nw_t", "t_nw")))
        bp = _num(x.get("estimate_bp"))
        parsed.append({"id": str(pick(x, ("id", "name")) or f"test {len(parsed) + 1}"),
                       "what": pick(x, ("what", "description", "label")),
                       "estimate": bp if bp is not None else _num(pick(x, ("estimate", "ic", "mean", "value"))),
                       "estimate_unit": "bp" if bp is not None else x.get("unit"),
                       "n": pick(x, ("n", "n_obs")), "t": tv,
                       "significant": tv is not None and abs(tv) > SIGNAL_T_BAR})
    by_id = {p["id"]: p for p in parsed}
    rows = []
    for p in pre:
        got = by_id.pop(p["id"], None)
        rows.append({**p, **got, "what": p["what"]} if got else {**p, "status": "missing from the file"})
    extra = list(by_id.values())
    if extra and all(r["status"] == "missing from the file" for r in rows):
        rows = extra                                           # the file uses its own ids: show it as given
        extra = []
    w = d.get("window") if isinstance(d, dict) else None
    ws = (f"{w.get('start')}..{w.get('end')}" if isinstance(w, dict) else str(w)) if w else None
    ok = None if not (window and isinstance(w, dict)) else (w.get("start"), w.get("end")) == tuple(window)
    sig_label = "observed (replay prices)" if window and tuple(window) == WINDOWS["test"] and ok else "in-sample"
    for r in rows + extra:
        if "significant" in r:
            r["status"] = ("not computed (no t)" if r["t"] is None else
                           sig_label if r["significant"] else "not significant")
    n_sig = sum(bool(r.get("significant")) for r in rows)
    return {"status": "complete", "path": _rel(path), "bar": SIGNAL_T_BAR, "window": ws, "window_ok": ok,
            "rows": rows, "extra_rows": extra, "n_tests": len(rows), "n_significant": n_sig,
            "generated_by": d.get("generated_by") if isinstance(d, dict) else None}


def _claim_increment(item: dict) -> dict:
    inc = item.get("increment")
    if not inc:
        return {"claim": None, "verdict": f"not run yet ({item.get('reading')})"}
    side = _ci_side(inc.get("ci95"))
    if side == "above 0":
        return {"claim": True, "verdict": "claimed: the LLM adds value (95% CI excludes 0, above it)"}
    if side == "below 0":
        return {"claim": False, "verdict": "not claimed: the LLM trails the baseline (95% CI excludes 0, below it)"}
    if side == "includes 0":
        return {"claim": False, "verdict": "not claimed: the 95% CI includes 0"}
    return {"claim": None, "verdict": "not readable: no finite bootstrap"}


def _claim_control(item: dict, other: str) -> dict:
    """One information-use comparison: does llm beat `other` with a 95% CI excluding 0?"""
    inc = item.get("increment")
    if not inc:
        return {"claim": None, "verdict": f"not run yet ({item.get('reading')})"}
    side = _ci_side(inc.get("ci95"))
    text = {"above 0": f"llm beats {other} (95% CI excludes 0, above it)",
            "below 0": f"llm trails {other} (95% CI excludes 0, below it)",
            "includes 0": f"llm not distinguishable from {other} (95% CI includes 0)"}
    return {"claim": side == "above 0" if side in text else None,
            "verdict": text.get(side, "not readable: no finite bootstrap")}


def prereg_readings(runs: list[dict], comps: dict, sig: dict, git: dict) -> dict:
    """PREREG decision rules, applied mechanically to the test-window v1 runs."""
    by = {r["variant"]: r for r in runs if r["mode"] == "replay" and r["window"] == "test" and r["version"] == "v1"}
    vs = (comps.get("test v1") or {}).get("vs", {})
    h = by.get("llm")
    head = {"run": h["run_id"] if h else None, "status": h["status"] if h else "not run yet",
            "label": REPLAY_LABEL, "rule": "reported whatever its sign"}
    if h and h["status"] == "complete":
        head.update(metrics=h["metrics"], risk_version=h.get("risk_version"), prompt_version=h.get("prompt_version"),
                    versions_ok=h.get("risk_version") == "v1" and h.get("prompt_version") == "v1",
                    code=h.get("code"))
    inc = {**_claim_increment(vs.get("baseline", {})), "increment": vs.get("baseline", {}).get("increment"),
           "status": vs.get("baseline", {}).get("status", "not run yet")}
    info = {o: {**_claim_control(vs.get(o, {}), o), "increment": vs.get(o, {}).get("increment"),
                "status": vs.get(o, {}).get("status", "not run yet")} for o in ("nonews", "shuffled")}
    sides = {o: _ci_side((info[o]["increment"] or {}).get("ci95")) if info[o]["increment"] else None for o in info}
    if any(s is None for s in sides.values()):
        miss = ", ".join(f"{o} {info[o]['status']}" for o, s in sides.items() if s is None)
        info_claim = {"claim": None, "verdict": f"not run yet ({miss}; llm {head['status']})"}
    elif all(s == "above 0" for s in sides.values()):
        info_claim = {"claim": True, "verdict": "claimed: news matters (llm beats both nonews and shuffled, both 95% "
                                                "CIs exclude 0)"}
    else:
        why = "; ".join(f"vs {o}: CI {s}" for o, s in sides.items())
        info_claim = {"claim": False, "verdict": f"not claimed ({why}): the agent's P&L is not attributable to the "
                                                 "news it reads"}
    risk = {}
    for v in VARIANTS:
        r = by.get(v)
        if r and r["status"] == "complete":
            risk[r["run_id"]] = {"variant": v, "beta": r.get("beta"), "risk": r.get("risk")}
    blinded = by.get("blinded")
    return {"window": dict(zip(("start", "end"), WINDOWS["test"])), "prereg": _rel(PREREG_FILE),
            "prereg_freeze_commit": git.get("prereg_freeze_commit"),
            "prereg_freeze_time_utc": git.get("prereg_freeze_time_utc"),
            "prereg_changed_since_freeze": git.get("prereg_changed_since_freeze"),
            "runs": {v: {"run": (by.get(v) or {}).get("run_id") or run_name(v, "v1", *WINDOWS["test"]),
                         "status": (by.get(v) or {}).get("status", "not run yet"),
                         "note": (by.get(v) or {}).get("note")} for v in VARIANTS},
            "headline": head, "increment": inc,
            "information": {**info, **info_claim}, "signal_tests": sig, "risk": risk,
            "blinded": {"status": blinded["status"] if blinded else "not run yet",
                        "reading": vs.get("blinded", {}).get("reading")}}


# ---------------------------------------------------------------- dev reference

def trials(path: Path) -> dict:
    """trials.log (JSONL): every entry's window, change and verdict; the explainability FAIL picked out."""
    rows = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return {"status": "missing", "path": _rel(path), "entries": [], "explainability": None}
    for ln in lines:
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        rows.append({"entry": len(rows) + 1, "ts": d.get("ts"), "window": d.get("window"),
                     "change": _short(d.get("change"), 200),
                     "verdict": d.get("verdict"), "criteria": d.get("pre_declared_criteria"),
                     "result": d.get("result")})
    fail = next((r for r in rows if str(r.get("verdict") or "").upper().startswith("FAIL")
                 and "explainab" in str(r.get("verdict") or "").lower()), None)
    exp = None
    if fail:
        res = fail.get("result") or {}
        keys = ("cooldown_rerequests", "off_hours_clips", "net_cap_clips", "beta_neutral_actions")
        exp = {"entry": fail["entry"], "ts": fail["ts"], "window": fail["window"], "criteria": fail["criteria"],
               "verdict": fail["verdict"], "before_after": {k: res[k] for k in keys if k in res}}
    return {"status": "present", "path": _rel(path), "n_entries": len(rows), "entries": rows, "explainability": exp}


def dev_reference(runs: list[dict], comps: dict, trials_path: Path, diagnosis_path: Path, sig: dict) -> dict:
    keep = [(w, ver, v) for w, ver, v, _ in EXPECTED if w != "test"]
    by = {(r["window"], r["version"], r["variant"]): r for r in runs if r["mode"] == "replay"}
    rows = []
    for w, ver, v in keep:
        r = by.get((w, ver, v))
        rows.append({"run": r["run_id"] if r else run_name(v, ver, *WINDOWS[w]), "window": w, "version": ver,
                     "variant": v, "status": r["status"] if r else "not run yet",
                     "metrics": (r or {}).get("metrics"), "beta": (r or {}).get("beta"),
                     "label": (r or {}).get("label", REPLAY_LABEL) + ", in-sample"})
    incs = {k: {"reading": c["vs"]["baseline"].get("reading"), "increment": c["vs"]["baseline"].get("increment")}
            for k, c in comps.items() if k in ("dev v0", "dev-half v1")}
    dp = Path(diagnosis_path)
    return {"runs": rows, "increments": incs, "signal_tests": sig,
            "diagnosis": {"path": _rel(dp), "exists": dp.exists(), "digest": list(DIAGNOSIS_DIGEST),
                          "sha256": sha256_file(dp) if dp.exists() else None,
                          "status": "in-sample / estimated (replay), v0 dev window"},
            "trials": trials(trials_path)}


# ---------------------------------------------------------------- live

def version_summary(by: dict) -> str:
    """One line: the live records per risk version and its prompt versions, first and last record time. The
    live loop switched risk/prompt versions while running, so its log is not all the pre-registered v1."""
    if not by:
        return "no live records"
    parts = []
    for v in sorted(by):
        x = by[v]
        pv = ", ".join(f"prompt {k} {n}" for k, n in sorted(x.get("prompt_versions", {}).items())) or "no decisions"
        parts.append(f"risk {v}: {x['n_records']} records ({pv}), {str(x.get('first_ts'))[:16]} → "
                     f"{str(x.get('last_ts'))[:16]} UTC")
    return "; ".join(parts)


def gap_label(fills_by_mode: dict) -> str:
    """Status label of the replay-vs-live fill gap: observed only when every matched live fill is a Bitget Demo
    fill; a shadow gap compares two fill models (live-venue quotes vs the replay's next-bar open)."""
    modes = sorted(m for m, f in (fills_by_mode or {}).items() if (f.get("n_matched") or 0) > 0)
    if not modes:
        return "not computed (no matched live order)"
    if modes == ["demo"]:
        return "observed (Bitget Demo fills vs the replay simulator)"
    if "demo" not in modes:
        return "estimated (shadow ledger: live-quote fill model vs the replay's fill model)"
    return "mixed: observed for Demo fills, estimated for shadow fills (see fills_by_mode)"


def live_section(out_root: Path, reconcile_path: Path | None, live_row: dict | None) -> dict:
    d = Path(out_root) / "live"
    if not d.is_dir():
        return {"status": "not run yet", "dir": _rel(d), "by_risk_version": {}, "versions": "no live records",
                "reconcile": {"status": "not run yet"}}
    recs = metrics.load_records(d)
    by: dict[str, dict] = {}
    for r in recs:
        v = r.get("risk_version") or "v0"
        x = by.setdefault(v, {"n_records": 0, "n_decisions": 0, "n_risk_exits": 0, "n_fills": 0, "n_invalid": 0,
                              "n_errors": 0, "first_ts": r.get("ts"), "last_ts": r.get("ts"), "prompt_versions": {}})
        x["n_records"] += 1
        x["last_ts"] = r.get("ts")
        if r.get("event", "decision") == "decision":
            x["n_decisions"] += 1
            if not (r.get("decision") or {}).get("valid", True):
                x["n_invalid"] += 1
            pv = r.get("prompt_version") or "v0"
            x["prompt_versions"][pv] = x["prompt_versions"].get(pv, 0) + 1
        else:
            x["n_risk_exits"] += 1
        x["n_fills"] += len(r.get("fills") or [])
        x["n_errors"] += bool(r.get("error")) + len(r.get("evidence_errors") or [])
    modes = live_modes(d)
    out = {"status": (live_row or {}).get("status", "incomplete"), "dir": _rel(d), "modes": modes,
           "label": live_label(modes), "n_records": len(recs), "by_risk_version": by,
           "versions": version_summary(by),
           "log_verified": ledger.verify(d / "log.jsonl") if (d / "log.jsonl").exists() else None,
           "metrics": (live_row or {}).get("metrics"), "n_hours": (live_row or {}).get("n_hours")}
    rp = Path(reconcile_path) if reconcile_path else d / "reconcile.json"
    rj = _read_json(rp)
    if not isinstance(rj, dict) or "summary" not in rj:
        out["reconcile"] = {"status": "not run yet", "path": _rel(rp),
                            "cmd": "uv run python -m sentiment.reconcile"}
        return out
    s = rj["summary"]
    keys = ("n_live_records", "n_decisions", "n_risk_exits", "n_identical", "n_unverifiable",
            "decision_identity_rate", "prompt_identity_rate", "targets_identity_rate", "targets_match_rate",
            "risk_rebuild_ok_rate", "risk_versions", "n_risk_ctx_checked", "n_risk_ctx_mismatch",
            "n_hourly_checks", "n_exit_identical", "exit_identity_rate", "n_live_only_exits", "n_replay_only_exits",
            "n_positions_breaks", "n_orders", "n_matched", "gap_bp_median", "gap_bp_p90", "abs_gap_bp_max",
            "mismatch_inputs", "n_decisions_replay_only_cards", "n_decisions_live_only_cards",
            "n_records_source_errors", "log_verified", "store")
    rec = {"status": "present", "path": _rel(rp), **{k: s.get(k) for k in keys},
           "fills_by_mode": {m: {k: f.get(k) for k in ("n_orders", "n_matched", "gap_bp_median", "gap_bp_p90",
                                                       "mid_gap_bp_median", "demo_live_mid_bp_median")}
                                 | {"label": LIVE_LABEL.get(m, m)}
                             for m, f in (s.get("fills_by_mode") or {}).items()}}
    rec["covers_log"] = s.get("n_live_records") == len(recs)
    rec["identity_label"] = "observed (the replay code recomputes each logged live decision)"
    rec["gap_label"] = gap_label(rec["fills_by_mode"])
    out["reconcile"] = rec
    return out


# ---------------------------------------------------------------- leakage audit

def cutoff_probe(path: Path) -> dict:
    d = _read_json(path)
    if not d:
        return {"status": "not run yet", "path": _rel(path)}
    rows = []
    for m in d.get("months", []):
        err = m.get("p2_median_abs_log_err")
        rows.append({"month": m.get("month"), "p1_n": m.get("p1_n"), "p1_acc": _num(m.get("p1_acc")),
                     "p2_n": m.get("p2_n"), "p2_median_abs_log_err": _num(err),
                     "p2_nonfinite": err is not None and _num(err) is None})
    first = next((r["month"] for r in rows if (r["p2_median_abs_log_err"] or 0) >= 0.10), None)
    gap = None
    if first:
        a, b = pd.Period(first, "M"), pd.Period(REPLAY_START[:7], "M")
        gap = (b - a).n
    return {"status": "complete", "path": _rel(path), "model": d.get("model"), "months": rows,
            "first_month_p2_err_ge_10pct": first, "months_from_that_to_replay_start": gap}


def news_timestamp_note() -> str:
    from sentiment import data
    doc = data.__doc__ or ""
    i = doc.find("Note on news timestamps")
    return " ".join(doc[i:].split()) if i >= 0 else ""


def manifest(snap: Path) -> dict:
    p = Path(snap) / "MANIFEST.sha256"
    if not p.exists():
        return {"status": "missing", "path": _rel(p)}
    files = {}
    for ln in p.read_text().splitlines():
        if ln.strip():
            h, name = ln.split("  ", 1)
            f = Path(snap) / name
            files[name] = {"sha256": h, "matches": f.exists() and sha256_file(f) == h}
    unlisted = sorted(q.name for q in Path(snap).glob("*.parquet") if q.name not in files)
    return {"status": "present", "path": _rel(p), "manifest_sha256": sha256_file(p), "files": files,
            "unlisted": unlisted, "check_ok": all(v["matches"] for v in files.values()) and not unlisted}


def llm_cache_summary(cache_dir: Path = LLM_CACHE) -> dict:
    tot = {"responses": 0, "prompt_tokens": 0, "completion_tokens": 0}
    by: dict[str, int] = {}
    for f in sorted(Path(cache_dir).glob("*.json")) if Path(cache_dir).exists() else []:
        rec = _read_json(f)
        if not isinstance(rec, dict):
            continue
        u = rec.get("usage") or {}
        tot["responses"] += 1
        tot["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        tot["completion_tokens"] += int(u.get("completion_tokens") or 0)
        by[rec.get("purpose", "?")] = by.get(rec.get("purpose", "?"), 0) + 1
    return {**tot, "by_purpose": dict(sorted(by.items())), "path": _rel(cache_dir)}


# ---------------------------------------------------------------- figures

def _style(ax, ylabel: str) -> None:
    ax.set_facecolor(PAPER)
    ax.figure.set_facecolor(PAPER)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)


def figures(runs: list[dict], report_dir: Path) -> dict[str, list[str]]:
    """Per window with >= 1 complete, current run (stale runs are left out): equity (% from start), drawdown,
    daily-return distribution (complete days). Titles carry the runs' status label."""
    for old in report_dir.glob("*.png"):
        old.unlink()
    by_w: dict[str, list[dict]] = {}
    for r in runs:
        if r.get("metrics") and r["status"] != "stale":
            by_w.setdefault(r["window"], []).append(r)
    out: dict[str, list[str]] = {}
    for w, rs in by_w.items():
        curves = {}
        for r in rs:
            key = "live" if r["mode"] == "live" else (r["variant"] if len({x["version"] for x in rs}) == 1
                                                      else f"{r['variant']} {r['version']}")
            curves[key] = (r["variant"] if r["mode"] == "replay" else "live", metrics.load_equity(r["dir_path"])["equity"])
        slug = re.sub(r"[^A-Za-z0-9]+", "_", w).strip("_")
        label = "; ".join(dict.fromkeys(r["label"] for r in rs))
        files = []

        fig, ax = plt.subplots(figsize=(9, 3.8))
        for k, (v, eq) in curves.items():
            ax.plot(eq.index, (eq / eq.iloc[0] - 1) * 100, color=COLOR[v], lw=2.0 if v in ("llm", "live") else 1.4, label=k,
                    ls=DASH[v] if not k.endswith(" v0") else (0, (4, 2)))
        ax.axhline(0, color=MUTED, lw=0.8)
        _style(ax, "return since start, %")
        ax.set_title(f"Equity, {w} ({label}, net of fees, spread, funding)", fontsize=10, color=INK, loc="left")
        if len(curves) > 1:
            ax.legend(frameon=False, fontsize=8, ncol=len(curves))
        fig.autofmt_xdate()
        fig.tight_layout(); fig.savefig(report_dir / f"equity_{slug}.png", dpi=140); plt.close(fig)
        files.append(f"equity_{slug}.png")

        fig, ax = plt.subplots(figsize=(9, 3.0))
        for k, (v, eq) in curves.items():
            ax.plot(eq.index, (eq / eq.cummax() - 1) * 100, color=COLOR[v], lw=1.4, label=k,
                    ls=DASH[v] if not k.endswith(" v0") else (0, (4, 2)))
        _style(ax, "drawdown, %")
        ax.set_title(f"Drawdown from running peak, {w} ({label}, hourly marks)", fontsize=10, color=INK, loc="left")
        if len(curves) > 1:
            ax.legend(frameon=False, fontsize=8, ncol=len(curves))
        fig.autofmt_xdate()
        fig.tight_layout(); fig.savefig(report_dir / f"drawdown_{slug}.png", dpi=140); plt.close(fig)
        files.append(f"drawdown_{slug}.png")

        fig, ax = plt.subplots(figsize=(max(4.5, 1.6 * len(curves) + 2), 3.2))
        rng = np.random.default_rng(0)
        names = list(curves)
        for i, k in enumerate(names):
            v, eq = curves[k]
            r = metrics.complete_day_returns(eq).to_numpy() * 100
            if len(r) >= 5:
                ax.boxplot([r], positions=[i], widths=0.5, showfliers=False,
                           medianprops={"color": INK}, boxprops={"color": MUTED},
                           whiskerprops={"color": MUTED}, capprops={"color": MUTED})
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(r)), r, s=20, color=COLOR[v], alpha=0.8, zorder=3,
                       edgecolors="white", linewidths=0.5)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_xticks(range(len(names)), [f"{k}\n(n={len(metrics.complete_day_returns(curves[k][1]))})" for k in names])
        _style(ax, "daily return, %")
        ax.set_title(f"Daily returns (complete 00:00-00:00 UTC days), {w} ({label})", fontsize=10, color=INK, loc="left")
        fig.tight_layout(); fig.savefig(report_dir / f"daily_{slug}.png", dpi=140); plt.close(fig)
        files.append(f"daily_{slug}.png")
        out[w] = files
    return out


# ---------------------------------------------------------------- formatting

def _f(x, nd: int = 2) -> str:
    v = _num(x)
    return "n/a" if v is None else f"{v:+.{nd}f}".replace("-", "−")


def _p(x, nd: int = 2) -> str:
    v = _num(x)
    return "n/a" if v is None else f"{v * 100:+.{nd}f}%".replace("-", "−")


def _why_not_identical(rc: dict) -> str:
    """Why live decisions are not reproduced: the inputs the Store yields at t now differ from what the live
    step saw (late-arriving items, source outages). Counts come from reconcile.json; empty when all identical."""
    n, ident = rc.get("n_decisions") or 0, rc.get("n_identical") or 0
    if n == 0 or ident == n or rc.get("n_decisions_replay_only_cards") is None:
        return ""
    return (f". Inputs, not code, explain the rest: the Store now yields cards at t that the live step did not see "
            f"(items that reached the Store after the step) in {rc['n_decisions_replay_only_cards']} of {n} "
            f"decisions, the live step saw cards the Store no longer yields at t in "
            f"{rc['n_decisions_live_only_cards']}, and {rc['n_records_source_errors']} of "
            f"{rc.get('n_live_records', n)} live steps logged a data-source error (HTTP 503) and decided on stale "
            f"inputs; every decision whose inputs match is reproduced exactly")
def _share(x) -> str:
    """Unsigned percentage for rates in [0, 1] (accuracy, win rate, identity rates)."""
    v = _num(x)
    return "n/a" if v is None else f"{v * 100:.0f}%"


def _bp(x) -> str:
    v = _num(x)
    return "n/a" if v is None else f"{v:+.1f} bp".replace("-", "−")


def _w(x) -> str:
    v = _num(x)
    return "n/a" if v is None else f"{v:+.3f}".replace("-", "−")


def _i(x) -> str:
    return "n/a" if x is None else str(x)


def _prob(x) -> str:
    v = _num(x)
    return "n/a" if v is None else f"{v:.2f}"


def _money(x) -> str:
    v = _num(x)
    return "n/a" if v is None else f"{v:,.2f}".replace("-", "−")


def _md(s) -> str:
    return " ".join(str(s if s is not None else "").split()).replace("|", "\\|")


def _ci(inc: dict | None) -> str:
    ci = (inc or {}).get("ci95") or [None, None]
    return f"[{_f(ci[0])}, {_f(ci[1])}]"


# ---------------------------------------------------------------- summary (<= 8 lines)

def summary(m: dict) -> list[str]:
    """At most 8 lines, from the metrics dict: the headline, the three PREREG claims, then every finding that
    weakens the headline (dev loss, no dev signal, the explainability FAIL), then live and integrity."""
    rd, dev = m["readings"], m["dev_reference"]
    out = []
    h = rd["headline"]
    if h.get("metrics"):
        x = h["metrics"]
        out.append(f"Headline (pre-registered: {h['run']}, {x['n_days']} days, {h['label']}, reported whatever its "
                   f"sign): return {_p(x['total_return'])}, Sharpe {_f(x['sharpe'])}, Sortino {_f(x['sortino'])}, "
                   f"max DD {_p(x['max_dd'])}, win rate {_share(x['win_rate'])} ({x['round_trips']} trips), costs "
                   f"{_p(x['ladder']['gross'])} gross -> {_p(x['ladder']['after_funding'])} net.")
    else:
        out.append(f"Headline (pre-registered: {h['run'] or run_name('llm', 'v1', *WINDOWS['test'])}, test window "
                   f"{rd['window']['start']}..{rd['window']['end']}): {h['status']}; no out-of-sample result yet.")
    inc = rd["increment"]
    out.append("LLM vs baseline (test): " + (f"Sharpe diff {_f(inc['increment']['sharpe_diff'])}, 95% CI "
                                              f"{_ci(inc['increment'])}; " if inc.get("increment") else "")
               + f"{inc['verdict']}.")
    info, sig = rd["information"], rd["signal_tests"]
    sig_s = (f"{sig['n_significant']} of {sig['n_tests']} signal tests clear |t| > {_f(sig['bar'], 1).lstrip('+')}"
             + (f" (the file is for {sig['window']}, not the test window)" if sig.get("window_ok") is False else "")
             if sig["status"] == "complete" else "signal tests not run yet")
    betas = [f"{v['variant']} {_f((v.get('beta') or {}).get('beta'), 3)}" for v in rd["risk"].values()
             if (v.get("beta") or {}).get("status") == "computed"]
    out.append(f"Information use (test): {info['verdict']}; {sig_s}"
               + (f"; realised beta to the EW basket {', '.join(betas)} (expected within ±{BETA_BAND})" if betas else "")
               + ".")
    dv = {(r["window"], r["version"], r["variant"]): r for r in dev["runs"]}
    parts, lost, half_days = [], [], None
    for key, name in ((("dev", "v0", "llm"), "v0 LLM"), (("dev", "v0", "baseline"), "v0 baseline"),
                      (("dev-half", "v1", "llm"), "v1 LLM (first half)"),
                      (("dev-half", "v1", "baseline"), "v1 baseline (first half)")):
        x = (dv.get(key) or {}).get("metrics")
        if x:
            parts.append(f"{name} {_p(x['total_return'])} (Sharpe {_f(x['sharpe'])}, {x['n_days']} days)")
            if key[0] == "dev" and (_num(x["total_return"]) or 0) < 0:
                lost.append(name)
            if key[0] == "dev-half":
                half_days = x["n_days"]
    if parts:
        out.append("Dev window (in-sample, estimated): " + "; ".join(parts) + "."
                   + (f" Lost over the full dev window: {', '.join(lost)}." if lost else "")
                   + (f" The first-half v1 runs ({half_days} days) were a prompt/risk check; P&L was not a criterion."
                      if half_days is not None else ""))
    else:
        out.append("Dev window: no complete dev run.")
    ds = dev["signal_tests"]
    if dev["diagnosis"]["exists"] or ds["status"] == "complete":
        out.append("No sentiment signal in dev: "
                   + (f"{ds['n_significant']} of {ds['n_tests']} pre-registered signal tests clear |t| > "
                      f"{_f(ds['bar'], 1).lstrip('+')} (in-sample); " if ds["status"] == "complete" else "")
                   + "DIAGNOSIS-dev.md finds no predictive rank IC in the baseline score, cards that mostly describe past "
                   "moves and fresh news that does not beat costs; dev results do not predict test.")
    ex = dev["trials"].get("explainability")
    if ex:
        ba = "; ".join(f"{k.replace('_', ' ')} {' -> '.join(str(v) for v in val)}"
                       for k, val in ex["before_after"].items() if isinstance(val, list))
        out.append(f"Prompt v1 FAILED its pre-declared explainability criteria (trials.log entry {ex.get('entry')}, "
                   f"{ex['window']}): {ba}; frozen anyway, not tuned. Explainability is shown per decision instead.")
    dvn, cp = rd.get("deviations") or {}, rd.get("code_provenance") or {}
    notes = dvn.get("notes") or []
    if notes or dvn.get("prereg_section_added_since_freeze"):
        out.append(f"Deviations and errata ({len(notes)}, dated; PREREG.md stays frozen, so they are listed in section "
                   "9b and trials.log): " + "; ".join(n["title"] for n in notes)
                   + ("; PREREG's Deviations section has entries added since the freeze"
                      if dvn.get("prereg_section_added_since_freeze") else "")
                   + (f". Test-run code: {cp['verdict']}" if cp.get("verdict") else "") + ".")
    lv = m["live"]
    live = f"Live paper log: {lv['status']}"
    if lv.get("metrics"):
        x = lv["metrics"]
        live = (f"Live paper log ({lv['label']}): {lv['n_hours']} hourly marks, {lv['n_records']} records "
                f"({lv.get('versions')}), {x['n_days']} complete days, return {_p(x['total_return'])}; too short for "
                "Sharpe or win rate")
    rc = lv.get("reconcile") or {}
    if rc.get("status") == "present":
        live += (f"; replay reconciliation: {rc['n_identical']}/{rc['n_decisions']} decisions identical, orders "
                 f"{rc['n_matched']}/{rc['n_orders']} matched, fill gap {rc.get('gap_label')}")
        live += _why_not_identical(rc)
    else:
        live += "; replay reconciliation not run yet"
    issues = []
    bad = [r["run_id"] for r in m["runs"] if r.get("log_verified") is False]
    if bad:
        issues.append(f"ledger hash chain BROKEN in {', '.join(bad)}")
    stale = [r["run_id"] for r in m["runs"] if r["status"] == "stale"]
    if stale:
        issues.append(f"stale runs kept out of readings: {', '.join(stale)}")
    if rd.get("prereg_changed_since_freeze"):
        issues.append("PREREG.md's fixed part (above Deviations) changed since its freeze commit")
    out.append(live + (". Integrity: " + "; ".join(issues) if issues else "") + ".")
    return out[:8]


# Verified against Crossref (DataCite for the arXiv paper) on 2026-10-02; cited in sections 1, 4-7 and 9.
REFERENCES = (
    ("baker2006", "Baker, M., Wurgler, J. (2006). Investor sentiment and the cross-section of stock returns. *Journal of Finance*, 61(4), 1645–1680. https://doi.org/10.1111/j.1540-6261.2006.00885.x"),
    ("cederburg2020", "Cederburg, S., O'Doherty, M. S., Wang, F., Yan, X. S. (2020). On the performance of volatility-managed portfolios. *Journal of Financial Economics*, 138(1), 95–117. https://doi.org/10.1016/j.jfineco.2020.04.015"),
    ("chan2003", "Chan, W. S. (2003). Stock price reaction to news and no-news: Drift and reversal after headlines. *Journal of Financial Economics*, 70(2), 223–260. https://doi.org/10.1016/S0304-405X(03)00146-6"),
    ("da2015", "Da, Z., Engelberg, J., Gao, P. (2015). The sum of all FEARS: Investor sentiment and asset prices. *Review of Financial Studies*, 28(1), 1–32. https://doi.org/10.1093/rfs/hhu072"),
    ("faber2007", "Faber, M. T. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management*, 9(4), 69–79. https://doi.org/10.3905/jwm.2007.674809"),
    ("glasserman2023", "Glasserman, P., Lin, C. (2023). Assessing look-ahead bias in stock return predictions generated by GPT sentiment analysis. *arXiv:2309.17322*. https://doi.org/10.48550/arXiv.2309.17322"),
    ("harvey2016", "Harvey, C. R., Liu, Y., Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68. https://doi.org/10.1093/rfs/hhv059"),
    ("lo2002", "Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36–52. https://doi.org/10.2469/faj.v58.n4.2453"),
    ("lopezlira2023", "Lopez-Lira, A., Tang, Y. (2023). Can ChatGPT forecast stock price movements? Return predictability and large language models. *SSRN working paper 4412788*. https://doi.org/10.2139/ssrn.4412788"),
    ("moreira2017", "Moreira, A., Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611–1644. https://doi.org/10.1111/jofi.12513"),
    ("newey1987", "Newey, W. K., West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708. https://doi.org/10.2307/1913610"),
    ("politis1994", "Politis, D. N., Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313. https://doi.org/10.1080/01621459.1994.10476870"),
    ("sarkar2024", "Sarkar, S. K., Vafa, K. (2024). Lookahead bias in pretrained language models. *SSRN working paper 4754678*. https://doi.org/10.2139/ssrn.4754678"),
    ("tetlock2007", "Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance*, 62(3), 1139–1168. https://doi.org/10.1111/j.1540-6261.2007.01232.x"),
    ("tetlock2008", "Tetlock, P. C., Saar-Tsechansky, M., Macskassy, S. (2008). More than words: Quantifying language to measure firms' fundamentals. *Journal of Finance*, 63(3), 1437–1467. https://doi.org/10.1111/j.1540-6261.2008.01362.x"),
)


def data_summary(snap: Path, cards_path: Path = CACHE / "cards.parquet") -> dict:
    """Section 2: what the frozen inputs hold (counts and date ranges from snapshot/), and the cards extracted from
    exactly those items (cards of items fetched later by the live loop are left out, so the counts do not drift)."""
    out: dict = {"universe": list(UNIVERSE), "snapshot": {}}
    try:
        from sentiment.data import Store
        frames = {f.stem: pd.read_parquet(f) for f in sorted(Path(snap).glob("*.parquet"))}
        for k, d in frames.items():
            row = {"rows": int(len(d))}
            if k in ("perp", "spot", "funding") and len(d):
                ts = pd.to_datetime(d["ts"], unit="ms", utc=True)
                row.update(symbols=int(d["sym"].nunique()), first=str(ts.min())[:10], last=str(ts.max())[:16])
            out["snapshot"][k] = row
        n = frames.get("news")
        if n is not None and len(n):
            out["snapshot"]["news"].update(first=str(n["published_at"].min())[:10], last=str(n["published_at"].max())[:10])
        items = Store(snapshot=True).all_items()
        ids = {it["id"] for it in items}
        out["items"] = dict(Counter(it["kind"] for it in items))
        if Path(cards_path).exists():
            c = pd.read_parquet(cards_path)
            c = c[c["item_id"].isin(ids)]
            out["cards"] = {"total": int(len(c)), "by_kind": {k: int(v) for k, v in c["kind"].value_counts().items()},
                            "news_by_novelty": {k: int(v) for k, v in c.loc[c["kind"] == "news", "novelty"].value_counts().items()}}
            nn = out["cards"]["news_by_novelty"]
            out["cards"]["news_post_hoc_share"] = (nn.get("post_hoc", 0) / sum(nn.values())) if nn else None
        out["status"] = "complete"
    except READ_ERRORS as exc:
        out["status"] = f"failed: {type(exc).__name__}: {exc}"
    return out


def design_params() -> dict:
    """Section 3: the parameters of the agent, the risk layer and the simulator (config.py)."""
    from sentiment import config as C
    return {"decision_hours_utc": list(DECISION_HOURS), "card_lookback_h": C.CARD_LOOKBACK_H,
            "max_w_name": C.MAX_W_NAME, "max_gross": C.MAX_GROSS, "max_net": C.MAX_NET,
            "vol_cap_floor": C.VOL_CAP_FLOOR, "stop_k": C.STOP_K, "stop_floor": C.STOP_FLOOR, "stop_cap": C.STOP_CAP,
            "stop_cooldown_h": C.STOP_COOLDOWN_H, "daily_kill": C.DAILY_KILL, "off_hours_scale": C.OFF_HOURS_SCALE,
            "funding_block": C.FUNDING_BLOCK, "taker": C.TAKER, "half_spread_fallback": C.HALF_SPREAD_FALLBACK,
            "model": C.LLM_MODEL, "start_equity": C.START_EQUITY}


def limits(m: dict) -> dict:
    """Section 9: Lo (2002) standard error of the headline Sharpe (iid daily returns) and the number of strategies tried."""
    h = (m["readings"]["headline"] or {}).get("metrics") or {}
    n, sr = h.get("n_days"), _num(h.get("sharpe"))
    se = None
    if n and sr is not None:
        sd = sr / math.sqrt(365)
        se = math.sqrt((1 + 0.5 * sd * sd) / n) * math.sqrt(365)
    return {"headline_n_days": n, "headline_sharpe": sr, "headline_sharpe_se": se,
            "headline_sharpe_ci95": None if se is None else [sr - 1.96 * se, sr + 1.96 * se],
            "n_other_tests": (m.get("other_tests") or {}).get("n_items")}


# ---------------------------------------------------------------- build

def build(out_root: Path = OUT, report_dir: Path = REPORT_DIR, snap: Path = SNAP, llm_cache: Path = LLM_CACHE,
          now: str | None = None, item_ids="auto", reconcile: Path | None = None, signal: Path | None = None,
          trials_path: Path = TRIALS_FILE, diagnosis_path: Path = DIAGNOSIS_FILE) -> dict:
    """`item_ids`: the current item id prefixes for the provenance check ("auto" = built from the snapshot under
    the current data rules; None skips the item check). `reconcile` defaults to out/live/reconcile.json,
    `signal` to out/signal_tests_test.json."""
    out_root, report_dir = Path(out_root), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    item_ids = current_item_ids() if isinstance(item_ids, str) and item_ids == "auto" else item_ids
    legacy = code_hash()
    git = git_info()
    bars = load_perp(snap)
    basket = basket_daily(bars)
    runs = []
    for spec in discover(out_root):
        r = evaluate(spec, item_ids, legacy)
        r["dir"], r["dir_path"] = _rel(spec["dir"]), spec["dir"]
        runs.append(r)
    rec_cache = {}
    for r in runs:
        if r.get("metrics"):
            rec_cache[r["run_id"]] = metrics.load_records(r["dir_path"])
            r["risk"] = risk_counts(rec_cache[r["run_id"]], bars)
            if r["mode"] == "replay":
                r["code"] = code_check(r.get("code_stamp"), git.get("prereg_freeze_commit"))
                exported = (git.get("export_code_checks") or {}).get(r["run_id"])
                if exported and not r["code"].get("files_differing_from_freeze") and "matches_freeze" not in r["code"]:
                    r["code"] = {**exported, "check": f"{exported.get('check')} (checked in the research repo at "
                                                      "export, PROVENANCE.json)"}
                try:
                    r["beta"] = realised_beta(r["dir_path"], basket)
                except READ_ERRORS as exc:
                    r["beta"] = {"status": f"failed: {type(exc).__name__}: {exc}"}
    comps = comparisons(runs)
    figs = figures(runs, report_dir)
    rerun = freeze_rerun(out_root / "freeze_rerun.json", runs)

    # decision chains: the headline run when complete, else the newest dev reference LLM run
    prefs = [("test", "v1", "llm"), ("dev-half", "v1", "llm"), ("dev", "v0", "llm")]
    by = {(r["window"], r["version"], r["variant"]): r for r in runs if r["mode"] == "replay"}
    chain_run = next((by[k]["run_id"] for k in prefs if k in by and by[k]["status"] == "complete"), None)
    if chain_run is None:
        chain_run = next((r["run_id"] for r in runs if r["mode"] == "replay" and r["status"] == "complete"
                          and r["variant"] in LLM_VARIANTS), None)
    live_row = next((r for r in runs if r["mode"] == "live"), None)
    m = {
        "generated_at": now or pd.Timestamp.now(tz="UTC").floor("s").isoformat(),
        "out_dir": _rel(out_root),
        "windows": {w: {"start": s, "end": e} for w, (s, e) in WINDOWS.items()},
        "git": git,
        "runs": [{k: v for k, v in r.items() if k != "dir_path"} for r in runs],
        "comparisons": comps,
        "readings": {**prereg_readings(runs, comps, signal_tests(Path(signal) if signal else
                                                                 out_root / "signal_tests_test.json", WINDOWS["test"]), git),
                     "code_provenance": code_provenance(runs, git, rerun),
                     "deviations": {"notes": [dict(d) for d in DEVIATIONS],
                                    "prereg_section": git.get("prereg_deviations_now"),
                                    "prereg_section_added_since_freeze": git.get("prereg_deviations_added")}},
        "dev_reference": dev_reference(runs, comps, trials_path, diagnosis_path,
                                       signal_tests(out_root / "signal_tests_dev.json", WINDOWS["dev"])),
        "chains": {"run": chain_run, "run_status": next((r["status"] for r in runs if r["run_id"] == chain_run), None),
                   "items": decision_chains(chain_run, rec_cache[chain_run], llm_cache=llm_cache) if chain_run else []},
        "live": live_section(out_root, reconcile, live_row),
        "leakage": {"cutoff_probe": cutoff_probe(out_root / "cutoff_probe.json"),
                    "news_timestamp_note": news_timestamp_note(), "snapshot_manifest": manifest(snap)},
        "reproduction": {"llm_cache": llm_cache_summary(llm_cache), "commands": commands(runs),
                         "items_checked": item_ids is not None},
        "figures": figs,
        "data": data_summary(snap),
        "design": design_params(),
        "other_tests": other_tests.collect(),
        "references": [{"id": k, "text": t} for k, t in REFERENCES],
    }
    m["limits"] = limits(m)
    m = _clean(m)
    m["summary"] = summary(m)
    (report_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (report_dir / "REPORT.md").write_text(render(m))
    return m


JUDGE_OUT = "judge-out"          # reruns go here, never over the shipped run directories


def _replay_cmd(spec: dict, cards_from: str | None) -> str:
    v, ver, start, end = spec["variant"], spec["version"], spec["start"], spec["end"]
    return (f"SENTIMENT_OFFLINE=1 uv run python -m sentiment.replay --variant {v} --start {start} --end {end} "
            f"--offline --risk-version {ver} --out {JUDGE_OUT}/{run_name(v, ver, start, end)}"
            + (f" --cards-from {cards_from}" if cards_from else ""))


def _run_arg(p: str | None) -> str | None:
    """A run directory as a command-line argument from ROOT: a pre-split `cards_from` ("sentiment/out/<run>")
    becomes "out/<run>" (config.run_dir resolves either form)."""
    if not p:
        return p
    q = Path(p)
    return str(Path("out", *q.parts[2:])) if not q.is_absolute() and q.parts[:2] == ("sentiment", "out") else p


def commands(runs: list[dict] | None = None) -> list[dict]:
    """How to reproduce. Every replay writes under judge-out/ (sentiment.replay deletes log.jsonl, equity.csv
    and metrics.json in its --out directory before it starts, so a rerun into a shipped run directory would
    destroy the shipped log if it stopped partway); compare its log_last_hash with the shipped run's. A run
    pinned to another run's cards (metrics.json `cards_from`) gets the same --cards-from."""
    by = {r["run_id"]: r for r in runs or []}
    rows = [{"what": "tests", "cmd": "uv run pytest -q tests"},
            {"what": "snapshot integrity", "cmd": "uv run python -m sentiment.snapshot --check"},
            {"what": "every shipped replay, offline, into judge-out/, checked against its logged last hash "
                     "(public repo)", "cmd": "uv run python scripts/judge_replay.py"}]
    specs = [{"variant": v, "version": ver, "start": WINDOWS[w][0], "end": WINDOWS[w][1], "window": w}
             for w, ver, v, _ in EXPECTED]
    for sp in specs:
        rid = run_name(sp["variant"], sp["version"], sp["start"], sp["end"])
        r = by.get(rid) or {}
        rows.append({"what": (f"{sp['variant']} {sp['version']}, {sp['window']} window (offline, cached LLM replies; "
                              f"compare judge-out/{rid}/metrics.json log_last_hash with out/{rid}/)"),
                     "cmd": _replay_cmd(sp, _run_arg(r.get("cards_from")))})
    rows += [{"what": "ledger hash chains", "cmd": "uv run python -m sentiment.ledger out/*/log.jsonl"},
             {"what": "live shadow loop (Bitget live quotes, no key needed)", "cmd": "uv run python -m sentiment.live --run --shadow"},
             {"what": "replay-vs-live reconciliation", "cmd": "uv run python -m sentiment.reconcile"},
             {"what": "this report", "cmd": "uv run python -m sentiment.report"}]
    return rows


# ---------------------------------------------------------------- markdown

def _metrics_row(name: str, status: str, label: str, x: dict | None, beta: dict | None = None) -> str:
    if not x:
        return f"| {name} | **{status}** | {_md(label)} | – | – | – | – | – | – | – | – | – | – |"
    days = f"{x['n_days']}" + (f" (+{x['partial_last_day_h']:.0f}h partial)" if (x.get("partial_last_day_h") or 0) > 0 else "")
    b = _f((beta or {}).get("beta"), 3) if (beta or {}).get("status") == "computed" else "–"
    return (f"| {name} | {status} | {_md(label)} | {days} | {_p(x['total_return'])} | {_f(x['sharpe'])} | "
            f"{_f(x['sortino'])} | {_p(x['max_dd'])} | {_share(x['win_rate'])} ({x['round_trips']}) | "
            f"{_f(x['turnover_ann'], 1)}× | {_p(x['ladder']['gross'])} | {_p(x['ladder']['after_funding'])} | {b} |")


METRIC_HEAD = ["| Run | Status | Label | Days | Net return | Sharpe | Sortino | Max DD | Win rate (trips) | Turnover (ann.) | Gross | Net of costs | Beta to EW |",
               "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]


def _inc_row(name: str, item: dict) -> str:
    inc = item.get("increment")
    if not inc:
        return f"| {name} | {_md(item.get('status'))} | – | – | – | – | – | – | {_md(item.get('verdict') or item.get('reading'))} |"
    return (f"| {name} | complete | {inc['n_days']} | {_f(inc['sharpe_a'])} | {_f(inc['sharpe_b'])} | "
            f"{_f(inc['sharpe_diff'])} | {_ci(inc)} | {inc.get('n_boot_used')} / {inc.get('n_boot')} "
            f"({inc.get('n_boot_dropped')} dropped) | {_md(item.get('verdict') or item.get('reading'))} |")


INC_HEAD = ["| llm vs | Status | Days | Sharpe llm | Sharpe other | Diff | 95% CI | Paths used / drawn | Reading |",
            "|---|---|---:|---:|---:|---:|---|---|---|"]


def _risk_table(rk: dict) -> list[str]:
    if not rk or not rk.get("by_rule"):
        return ["No rule bound."]
    L = ["| Rule | At decision | Hourly | Weight moved | P&L effect (est., USDT) | Meaning |", "|---|---:|---:|---:|---:|---|"]
    for k, v in rk["by_rule"].items():
        eff = _money(v["pnl_effect_est"]) if v.get("pnl_effect_est") is not None else "–"
        L.append(f"| {k} | {v['decision']} | {v['hourly_check']} | {_num(v['weight_moved']) or 0:.3f} | {eff} | {_md(v['meaning'])} |")
    return L


def _signal_table(sig: dict) -> list[str]:
    L = [f"Source `{sig['path']}`" + (f", window {sig['window']}" if sig.get("window") else "")
         + (" (**not the expected window**)" if sig.get("window_ok") is False else "")
         + (": not run yet." if sig["status"] != "complete" else "."), "",
         "| Test | What | Estimate | n | t | Status |", "|---|---|---:|---:|---:|---|"]
    for r in sig["rows"]:
        est = (_f(r["estimate"], 1 if r.get("estimate_unit") == "bp" else 4)
               + (f" {r['estimate_unit']}" if r.get("estimate_unit") else "")) if r.get("estimate") is not None else "–"
        L.append(f"| {r['id']} | {_md(r.get('what'))} | {est} | {_i(r.get('n'))} | "
                 f"{_f(r.get('t')) if r.get('t') is not None else '–'} | {r['status']} |")
    for r in sig.get("extra_rows") or []:
        L.append(f"| {r['id']} (not pre-registered) | {_md(r.get('what'))} | {_f(r.get('estimate'), 4)} | "
                 f"{_i(r.get('n'))} | {_f(r.get('t'))} | {r['status']}, not used for the bar |")
    if sig["status"] == "complete" and sig["n_tests"] != len(SIGNAL_TESTS):
        L += ["", f"The file has {sig['n_tests']} tests; PREREG fixes {len(SIGNAL_TESTS)}."]
    return L


def _u(x, nd: int = 2) -> str:
    """Unsigned number (parameters, not results)."""
    v = _num(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def _cite(m: dict, *ids: str) -> str:
    """[n] for reference ids, n = position in m['references'] (1-based)."""
    pos = {r["id"]: i for i, r in enumerate(m.get("references") or [], 1)}
    return "[" + ", ".join(str(pos[i]) for i in ids if i in pos) + "]"


def _judging(m: dict) -> list[str]:
    rd = m["readings"]
    x = (rd["headline"] or {}).get("metrics")
    if not x:
        return ["**Judging focus.** No complete test-window run yet."]
    inc, info, sig, lv = rd["increment"], rd["information"], rd["signal_tests"], m["live"]
    rc = lv.get("reconcile") or {}
    rows = [("Return", f"{_p(x['total_return'])} over {x['n_days']} days"),
            ("Sharpe (Sortino)", f"{_f(x['sharpe'])} ({_f(x['sortino'])})"),
            ("Max drawdown", _p(x["max_dd"])),
            ("Win rate", f"{_share(x['win_rate'])} of {x['round_trips']} round trips; turnover {_u(x['turnover_ann'], 1)}× equity a year"),
            ("LLM over a fixed rule on the same evidence",
             (f"Sharpe difference {_f(inc['increment']['sharpe_diff'])}, 95% CI {_ci(inc['increment'])}: " if inc.get("increment") else "")
             + inc["verdict"]),
            ("Does the news matter?", f"{info['verdict']}; {sig.get('n_significant')} of {sig.get('n_tests')} signal tests significant"),
            ("Replayable", (f"{rc.get('n_identical')} of {rc.get('n_decisions')} live decisions reproduced exactly; every shipped "
                            "replay reruns offline to its logged hash" if rc.get("status") == "present" else "reconciliation not run yet"))]
    return ([(f"**Judging focus.** Pre-registered test window {rd['window']['start']} → {rd['window']['end']}, net of taker fee, "
              "quoted half-spread and funding; every number is *estimated (replay)*."), "",
             "| Metric | Value |", "|:--|:--|"] + [f"| {a} | {_md(b)} |" for a, b in rows])


def render(m: dict) -> str:
    """REPORT.md from the metrics dict alone (every number printed here is a value in metrics.json). Section order
    follows project A4's report: hypothesis, data, construction, prediction tests, performance, alternative
    explanations, robustness, risk, limits, references, reproduction."""
    rd, dev, lv = m["readings"], m["dev_reference"], m["live"]
    c = lambda *ids: _cite(m, *ids)  # noqa: E731
    L = ["# After the Bell — an LLM news agent on Bitget stock perps", "",
         (f"Generated {m['generated_at']} by `uv run python -m sentiment.report` from `{m['out_dir']}/`. Every number "
          "below is a value in `report/metrics.json`. Replay numbers are **estimated (replay)**: simulated "
          "fills at the next 1h open ± quoted half-spread, taker fee, realised funding. Live numbers are labelled by "
          "broker mode: only Bitget Demo fills are **observed**; a shadow ledger (live-venue quotes, simulated fills) "
          "is **estimated**."), "",
         "## Summary", ""]
    L += _judging(m)
    L += ["", ("**What.** An agent reads every news item, analyst action and insider filing about ten Bitget stock perps as it "
               "arrives, turns each into an evidence card, and every four hours sets a target book citing the cards behind "
               "each position. A deterministic risk layer has the last word. The same code runs on a historical and on the "
               "wall clock, every LLM reply is cached and every record is hash-chained, so the paper log can be rerun."), "",
          "**Findings.**", ""]
    L += [f"- {_md(s)}" for s in m["summary"]]
    ot = m.get("other_tests") or {}
    if ot.get("items"):
        L += [f"- Also tested after the test window (section 7): {ot['n_items']} other strategies, each against criteria "
              f"written before its run; {ot['n_claimed']} cleared them. The submission stays the pre-registered v1."]

    # 1. hypothesis
    L += ["", "## 1. Hypothesis and prior evidence", "",
          ("News moves prices, and part of the move comes late. Media pessimism predicts price pressure that later "
           f"reverses {c('tetlock2007')}; the negative words in firm news predict earnings and returns {c('tetlock2008')}; "
           f"prices drift after news headlines but reverse after large moves without news {c('chan2003')}; broad sentiment "
           f"measures predict the cross-section of returns {c('baker2006', 'da2015')}. Headlines scored by a large language "
           f"model predict next-day returns {c('lopezlira2023')}, but a pretrained model may simply remember what happened "
           f"after its training cutoff {c('glasserman2023', 'sarkar2024')}: the replay therefore starts more than a year "
           "after the model's measured cutoff (section 2)."), "",
          ("Bitget's US-stock perps trade around the clock; the stocks and the desks that read the news about them do "
           "not. News lands after the close, overnight and at weekends, and the perp is where it can be priced first. "
           "Two hypotheses were frozen in `docs/PREREG.md` before the test window was run, each with a claim rule: "
           "(a) an LLM that reads the cards and sets the book beats a fixed rule on the same cards (section 5); "
           "(b) the agent's P&L depends on the news it reads, i.e. it beats the same agent with no news and with the "
           "news shown weeks late (section 6). **Neither cleared its rule.**")]

    # 2. data
    d = m.get("data") or {}
    sn, cards = d.get("snapshot") or {}, d.get("cards") or {}
    L += ["", "## 2. Data and universe", ""]
    if d.get("status") == "complete":
        perp_s = sn.get("perp") or {}
        L += [(f"- **Universe.** {len(d['universe'])} stock perps with news coverage and full data: "
               f"{', '.join(d['universe'])}. Chosen in 2026-09 from the names on Bitget's demo venue, i.e. with hindsight."),
              (f"- **Prices.** Hourly perp bars of {perp_s.get('symbols')} symbols (the ten names plus the S&P 500 and "
               f"Nasdaq-100 index perps) from {perp_s.get('first')} to {perp_s.get('last')} UTC, rToken spot bars for the "
               f"premium, funding settlements ({(sn.get('funding') or {}).get('rows')} rows from "
               f"{(sn.get('funding') or {}).get('first')}). Bitget public API v2."),
              (f"- **Evidence.** From Bitget's data MCP: {(sn.get('news') or {}).get('rows')} news items "
               f"({(sn.get('news') or {}).get('first')} → {(sn.get('news') or {}).get('last')}), "
               f"{(sn.get('ratings') or {}).get('rows')} analyst rating and price-target rows, "
               f"{(sn.get('insider') or {}).get('rows')} insider filings, and {(sn.get('crypto_fng') or {}).get('rows')} "
               "days of the crypto Fear & Greed index. Each item becomes visible at its availability time only "
               "(ratings the next session, filings the next day, the index the day after)."),
              (f"- **Cards.** {cards.get('total')} evidence cards from those items "
               f"({', '.join(f'{k} {v}' for k, v in (cards.get('by_kind') or {}).items())}); of the news cards "
               f"{_share(cards.get('news_post_hoc_share'))} are *post hoc*: they explain a move that already happened."),
              (f"- **Windows.** Dev {m['windows']['dev']['start']} → {m['windows']['dev']['end']} (prompt and risk design); "
               f"test {m['windows']['test']['start']} → {m['windows']['test']['end']} (pre-registered, each run once); "
               "live from 2026-09-23.")]
    else:
        L.append(f"Data summary: {d.get('status', 'n/a')}.")
    lk = m["leakage"]
    pr = lk["cutoff_probe"]
    L += ["", ("**What the model knew** (`sentiment/probes/cutoff_probe.py`, closed-book, observed). P1 = direction of the "
               "month's move for a set of names; P2 = median |log error| of the recalled month-end close."), ""]
    if pr.get("status") != "complete":
        L.append(f"Not run yet (`{pr['path']}` missing).")
    else:
        L += [f"Model `{pr['model']}`.", "", "| Month | P1 n | P1 accuracy | P2 n | P2 median abs log err |",
              "|---|---:|---:|---:|---:|"]
        for r in pr["months"]:
            err = "non-finite" if r["p2_nonfinite"] else (_f(r["p2_median_abs_log_err"], 3) if r["p2_median_abs_log_err"] is not None else "n/a")
            L.append(f"| {r['month']} | {_i(r['p1_n'])} | {_share(r['p1_acc'])} | {_i(r['p2_n'])} | {err} |")
        if pr.get("first_month_p2_err_ge_10pct"):
            L += ["", (f"The P2 error first reaches 10% in {pr['first_month_p2_err_ge_10pct']}, "
                       f"{pr['months_from_that_to_replay_start']} months before the replay window starts: the model "
                       "cannot recall the window's prices from its parameters.")]
    L += ["", f"**News timestamps (UTC+8).** From `sentiment/data.py`: {_md(lk['news_timestamp_note'])}", "",
          "Facts about the data we measured along the way:", "",
          "| # | Fact | Why it matters |", "|---|---|---|",
          "| 1 | The data MCP stamps `published_at` with a `Z` suffix, but the stamps are Beijing wall time | items appeared up to 8 hours \"in the future\"; every item is shifted and the store is tested for it |",
          "| 2 | Ratings and insider filings arrive late | a live step can see fewer cards than the store holds afterwards (section 9) |",
          "| 3 | Stock-perp long/short and taker-volume histories are empty; only crypto has them | sentiment from positioning is not available for stocks |",
          "| 4 | Funding history reaches back ~90 days | costs before 2026-06-22 can only be estimated |",
          "| 5 | The news MCP returned HTTP 503 for large parts of 2026-09-25 → 28 | the live loop logged the gaps and decided on stale inputs (section 9) |"]

    # 3. construction
    dz = m.get("design") or {}
    L += ["", "## 3. Agent and portfolio construction", "",
          ("1. **Evidence cards (LLM stage 1).** Each news item is turned once into cards: tickers or scope, stance −2..+2, "
           "strength, horizon, novelty (new / recap / post hoc) and a quote. Ratings and filings become cards by rule. "
           "Cached by prompt hash, reused by replay and live."),
          (f"2. **Decision (LLM stage 2).** At {', '.join(f'{h:02d}' for h in dz.get('decision_hours_utc', []))} UTC the model "
           f"`{dz.get('model')}` (temperature 0) sees the market state of the ten names, the cards of the last "
           f"{dz.get('card_lookback_h')} hours, its risk context and the book, and returns a weight, a reason and the card ids "
           "for every name (`sentiment/prompts/decide_v1.md`)."),
          ("3. **Fixed-rule baseline.** The same cards summed by stance × strength with an age decay and mapped to weights by a "
           "formula (`docs/CONTRACT.md`). The LLM's increment is measured against it."),
          (f"4. **Risk layer v1** (deterministic, `sentiment/risk.py`): per-name cap {_u(dz.get('max_w_name'))} of equity scaled down by "
           f"volatility (floor {_u(dz.get('vol_cap_floor'))}×), ×{_u(dz.get('off_hours_scale'))} for new exposure outside the US "
           f"session, funding block at |funding| ≥ {_u((dz.get('funding_block') or 0) * 1e4, 0)} bp, stop at clip({_u(dz.get('stop_k'), 1)} × vol, "
           f"{_share(dz.get('stop_floor'))}, {_share(dz.get('stop_cap'))}) with a {dz.get('stop_cooldown_h')}h cooldown, a daily kill "
           f"at {_share(dz.get('daily_kill'))}, beta neutralisation, net cap {_u(dz.get('max_net'))}, gross cap {_u(dz.get('max_gross'))}."),
          (f"5. **Execution.** Replay: next 1h bar open ± each name's quoted half-spread, taker fee {_u((dz.get('taker') or 0) * 100, 2)}%, "
           f"settled funding; {_money(dz.get('start_equity'))} USDT. Live: Bitget Demo when keys are set, else a shadow ledger at "
           "live quotes. One hash-chained record per decision either way."), "",
          "| Rule | What it does |", "|---|---|"]
    L += [f"| {k} | {_md(v)} |" for k, v in RULE_TEXT.items()]
    frz = rd.get("prereg_freeze_commit")
    L += ["", (f"Rules for reading the results: `{rd['prereg']}`, frozen at commit `{frz or 'unknown'}`"
               + (f", committed {rd['prereg_freeze_time_utc']} per git" if rd.get("prereg_freeze_time_utc") else "")
               + ("; **its fixed part changed since**" if rd.get("prereg_changed_since_freeze") else "")
               + ". Risk v1 and prompt v1 were designed on the dev window and frozen before the test runs."), "",
          "| Pre-registered run (v1, test) | Status | Note |", "|---|---|---|"]
    for v, x in rd["runs"].items():
        L.append(f"| {x['run']} | {x['status']} | {_md(x.get('note'))} |")

    # 4. prediction
    sig = rd["signal_tests"]
    L += ["", "## 4. Does the news predict returns?", "",
          (f"Five tests fixed in PREREG, each run once ((a) counts as three horizons). With five tests, |t| > {sig['bar']} is "
           f"needed before any one is called significant (Bonferroni at 5%); t statistics are Newey-West {c('newey1987')}."), "",
          "**Test window**", ""] + _signal_table(sig)
    L += ["", "**Dev window** (same definitions; in-sample)", ""] + _signal_table(dev["signal_tests"])
    dg = dev["diagnosis"]
    L += ["", f"**Dev diagnosis** ([{dg['path']}](../{dg['path']}), {dg['status']}"
          + ("" if dg["exists"] else ", **file missing**") + "):", ""]
    L += [f"{i}. {_md(s)}" for i, s in enumerate(dg["digest"], 1)]

    # 5. performance
    h = rd["headline"]
    L += ["", "## 5. Performance net of costs", "", "### 5a. Headline: LLM v1, test window", "",
          f"Reported whatever its sign; status label **{h['label']}**.", ""] + METRIC_HEAD
    L.append(_metrics_row(h["run"] or "llm v1 test", h["status"], h["label"], h.get("metrics"),
                          (rd["risk"].get(h["run"] or "", {}) or {}).get("beta")))
    if h.get("metrics"):
        x = h["metrics"]
        lad, cc = x["ladder"], x["costs"]
        L += ["", (f"Cost ladder: gross {_p(lad['gross'])} → after fees {_p(lad['after_fees'])} → after spread "
                   f"{_p(lad['after_spread'])} → after funding {_p(lad['after_funding'])} (fees {_money(cc['fees'])}, "
                   f"spread {_money(cc['spread'])}, funding P&L {_money(cc['funding'])} USDT). Trades {x['trades']}, "
                   f"decisions {x['n_decisions']}, invalid {x['n_invalid']}, risk exits {x['n_risk_exits']}.")]
        if not h.get("versions_ok"):
            L.append(f"**Version check failed**: risk {h.get('risk_version')}, prompt {h.get('prompt_version')} (PREREG: v1, v1).")
        if h.get("code"):
            L.append(f"Code: `{h['code'].get('stamp') or 'no stamp'}`: {_md(h['code'].get('check'))}.")
    inc = rd["increment"]
    L += ["", "### 5b. LLM increment over the fixed-rule baseline", "",
          (f"Sharpe(llm) − Sharpe(baseline), stationary block bootstrap {c('politis1994')} (mean block 5 days, 2000 paths, "
           "seed 20260923) on common complete days. **Claim rule**: \"the LLM adds value\" only if the 95% CI excludes 0."), ""] + INC_HEAD
    L.append(_inc_row("baseline", inc))
    L += ["", f"**Reading**: {_md(inc['verdict'])}."]
    L += ["", "### 5c. Dev window (in-sample)", "",
          ("The development runs, shown next to the test results as PREREG requires. v0 = the first risk layer and "
           "prompt, full dev window; v1 = the risk layer and prompt frozen for the test, first half of the dev window. "
           "In-sample: risk v1 was designed after the v0 dev diagnosis."), ""] + METRIC_HEAD
    for r in dev["runs"]:
        L.append(_metrics_row(f"{r['run']} ({r['version']})", r["status"], r["label"], r.get("metrics"), r.get("beta")))
    L += ["", "LLM vs baseline in dev:", ""]
    for k, x in dev["increments"].items():
        L.append(f"- {k} (in-sample): {_md(x['reading'])}")
    L += ["", "### 5d. All runs", "",
          "| Run | Window | Variant | Version | Status | Label | Code | Days | Return | Sharpe | Max DD | Trades | Ledger | Note |",
          "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for r in m["runs"]:
        x = r.get("metrics")
        if not x:
            L.append(f"| {r['run_id']} | {r['window']} | {r['variant']} | {_i(r.get('version'))} | **{r['status']}** | "
                     f"{_md(r['label'])} | – | – | – | – | – | – | – | {_md(r.get('note'))} |")
            continue
        ver = {True: "verified", False: "BROKEN", None: "no log"}[r.get("log_verified")]
        days = f"{x['n_days']}"
        if (x.get("partial_first_day_h") or 0) > 0:
            days += f" (+{x['partial_first_day_h']:.0f}h partial first day)"
        if (x.get("partial_last_day_h") or 0) > 0:
            days += f" (+{x['partial_last_day_h']:.0f}h partial last day, {_p(x.get('partial_last_day_return'))}, not counted)"
        status = f"**{r['status']}**" if r["status"] == "stale" else r["status"]
        L.append(f"| {r['run_id']} | {r['window']} | {r['variant']} | {_i(r.get('risk_version') or r.get('version'))} | "
                 f"{status} | {_md(r['label'])} | {'`' + r['code_stamp'] + '`' if r.get('code_stamp') else '–'} | {days} | {_p(x['total_return'])} | "
                 f"{_f(x['sharpe'])} | {_p(x['max_dd'])} | {x['trades']} | {ver} | {_md(r.get('note'))} |")
    L += ["", "Cost ladder (return on start equity after each cost layer; costs in USDT):", "",
          "| Run | Gross | After fees | After spread | After funding | Fees | Spread | Funding P&L | Decisions | Invalid | Risk exits |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in m["runs"]:
        x = r.get("metrics")
        if x:
            lad, cc = x["ladder"], x["costs"]
            L.append(f"| {r['run_id']} | {_p(lad['gross'])} | {_p(lad['after_fees'])} | {_p(lad['after_spread'])} | "
                     f"{_p(lad['after_funding'])} | {_money(cc['fees'])} | {_money(cc['spread'])} | {_money(cc['funding'])} | "
                     f"{x['n_decisions']} | {x['n_invalid']} | {x['n_risk_exits']} |")
    L += ["", ("Sharpe and Sortino: daily 00:00-00:00 UTC returns × √365, complete days only (a partial first or last "
               "day is shown, not counted). Max DD on hourly marks. Win rate: closed round trips after fees, funding "
               "excluded. A **stale** run was made under older data rules or code (its note says why): listed, not "
               "compared or charted.")]
    other = {k: cm for k, cm in m["comparisons"].items() if k not in ("test v1",)}
    if other:
        L += ["", "Increments in the other windows (estimated; paths with a non-finite Sharpe are dropped):", ""] + INC_HEAD
        for k, cm in other.items():
            for o, item in cm["vs"].items():
                if item.get("increment"):
                    L.append(_inc_row(f"{o} ({k})", item))
    for w in ("test", "dev", "dev-half"):
        files = m["figures"].get(w) or []
        if files:
            L += ["", f"Figures, {w} window:", ""] + [f"![{f}]({f})" for f in files]

    # 6. alternative explanations
    info = rd["information"]
    L += ["", "## 6. Is it the news, or something else?", "",
          ("Three placebos run through the same replay, costs and risk layer: **no news** (the agent sees prices, funding "
           "and the index only), **shuffled** (every card shown 7–21 days after it appeared, never before) and **blinded** "
           "(names and dates masked). **Claim rule**: \"news matters\" only if the agent beats **both** no news and "
           "shuffled with CIs excluding 0; otherwise its P&L cannot be attributed to the news it reads."), ""] + INC_HEAD
    for o in ("nonews", "shuffled"):
        L.append(_inc_row(o, info[o]))
    L += ["", f"**Reading**: {_md(info['verdict'])}."]
    if rd["blinded"]["status"] != "not run yet":
        L.append(f"Blinded control (optional): {rd['blinded']['status']}; {_md(rd['blinded'].get('reading'))}.")
    else:
        L.append("Blinded control (optional, only if Qwen credit remains): not run yet.")
    L += ["", (f"**Market exposure.** The risk layer neutralises beta, so the P&L should not be the market: realised beta "
               f"(OLS slope of the book's daily gross return on the equal-weight basket of the ten perps) is expected within "
               f"±{BETA_BAND}."), "", "| Run | Beta | t | R² | Days | Inside the band |", "|---|---:|---:|---:|---:|---|"]
    for rid, x in rd["risk"].items():
        b = x.get("beta") or {}
        if b.get("status") == "computed":
            L.append(f"| {rid} | {_f(b['beta'], 3)} | {_f(b.get('t'))} | {_prob(b.get('r2'))} | {b['n_days']} | "
                     f"{'yes' if b.get('within_band') else '**no**'} |")
        else:
            L.append(f"| {rid} | {b.get('status', 'n/a')} | | | | |")

    # 7. robustness / other tests
    L += ["", "## 7. Robustness: what else we tried", "",
          (_md(ot.get("note") or "") + " Each module has its own code, trials.log and outputs; the numbers below are read "
           "from those outputs."), ""]
    items = ot.get("items") or {}
    if items:
        L += ["| Strategy | Window | Evidence | Verdict |", "|---|---|---|---|"]
        for x in items.values():
            w = x.get("window") or [None, None]
            L.append(f"| {_md(x['title'])} | {w[0] or '–'} → {w[1] or '–'} | {_md(x['status'])} | {_md(x.get('verdict', 'not shipped'))} |")
    fc = (items.get("fg_cross_asset") or {}).get("numbers")
    if fc:
        sh = fc["sharpe"]
        L += ["", "### 7a. Crypto fear & greed, contrarian", "", _md(items["fg_cross_asset"]["question"]),
              "", f"Design: {_md(items['fg_cross_asset']['design'])}.", "",
              "| Asset, period | Buy & hold | F&G long-only | F&G long/short | Same rule on BTC momentum | 200-day MA " + c("faber2007") + " |",
              "|---|---:|---:|---:|---:|---:|"]
        for k, v in sh.items():
            L.append(f"| {k.replace('|', ', ').replace('BTC-USD', 'BTC')} | {_f(v.get('buy_hold'))} | {_f(v.get('fg_long'))} | "
                     f"{_f(v.get('fg_long_short'))} | {_f(v.get('mom_long'))} | {_f(v.get('ma200'))} |")
        L += ["", (f"Sharpe ratios, net of 10 bp per unit turnover. The long-only rule beats buy & hold with a CI excluding 0 in "
                   f"{fc['long_only_beats_buy_hold_ci']} of {fc['n_asset_periods']} asset-periods; the long/short rule is below "
                   f"buy & hold in {fc['long_short_below_buy_hold']}. In 2018-22, {fc['p1_slopes_positive']} of "
                   f"{fc['p1_slopes_total']} slopes of the 5/20/60-day forward return on the index are positive: greed was "
                   "followed by higher returns. Shorting MSTR in greed lost (sum of daily returns) "
                   f"{_p(fc['mstr_short_leg_2020'], 0)} in 2020 and {_p(fc['mstr_short_leg_2024'], 0)} in 2024. Net of 60-day "
                   f"beta × BTC, no stock's return depends on the index (largest |t| {_f(fc['resid_max_abs_t'])}).")]
    fm = (items.get("fg_mstr") or {}).get("numbers")
    if fm:
        hh, pp = fm["history"], fm["paper"]
        L += ["", "### 7b. Fear & greed on MSTR, long-only, with a paper log on the rToken", "",
              _md(items["fg_mstr"]["question"]), "", f"Design: {_md(items['fg_mstr']['design'])}.", "",
              "| MSTR daily, " + " → ".join(items["fg_mstr"]["window"]) + " | Total | CAGR | Sharpe | Max DD | Mean weight |",
              "|---|---:|---:|---:|---:|---:|"]
        for k, lab in (("buy_hold", "Buy & hold"), ("fg_long", "F&G rule"), ("constant_mean_w", "Constant, same mean weight"),
                       ("mom_twin", "Same rule on BTC momentum"), ("ma200", "200-day MA")):
            v = hh.get(k) or {}
            L.append(f"| {lab} | {_p(v.get('total'), 0)} | {_p(v.get('cagr'), 0)} | {_f(v.get('sharpe'))} | {_p(v.get('max_dd'), 0)} | {_share(v.get('mean_w'))} |")
        L += ["", (f"Sharpe difference, rule − buy & hold: 95% CI [{_f(fm['ci_vs_buy_hold'][0])}, {_f(fm['ci_vs_buy_hold'][1])}]; "
                   f"rule − constant: [{_f(fm['ci_vs_constant'][0])}, {_f(fm['ci_vs_constant'][1])}]. MSTR's mean 20-day forward "
                   f"return was {_p(fm['fwd20_le25'], 1)} after F&G ≤ 25 and {_p(fm['fwd20_gt75'], 1)} after F&G > 75. In 2022 the "
                   f"rule held {_share(fm['mean_w_2022'])} on average while MSTR returned {_p(fm['mstr_2022'], 0)}."),
              "", (f"Paper log on Bitget's RMSTRUSDT rToken, {pp['window'][0][:16]} → {pp['window'][1][:16]} UTC: {pp['records']} "
                   f"hash-chained records (chain {'verified' if pp['ledger_ok'] else 'BROKEN'}, last `{pp['last_hash'][:16]}…`), "
                   f"{pp['trades']} trades; rule {_p(pp['return'], 1)} (max DD {_p(pp['max_dd'], 1)}) against the rToken "
                   f"{_p(pp['rtoken_return'], 1)} and a constant {_share(pp['mean_w'])} weight {_p(pp['constant_return'], 1)}; "
                   f"with the index used a day later, {_p(pp['return_lag24h'], 1)}.")]
    vm = (items.get("vol_managed") or {}).get("numbers")
    if vm:
        L += ["", "### 7c. No news: the ten names long, scaled by volatility", "",
              (_md(items["vol_managed"]["question"]) + f" Volatility-managed portfolios raise Sharpe ratios for factors "
               f"{c('moreira2017')}, much less reliably out of sample and for single assets {c('cederburg2020')}."), "",
              f"Design: {_md(items['vol_managed']['design'])}.", "",
              "| Window | Rule | Same mean exposure | Full (10% each) | Rule − same exposure, Sharpe 95% CI | Trades rule / constant |",
              "|---|---|---|---|---|---|"]
        for w, x in vm["replay"].items():
            fmt = lambda v: "–" if not v else f"{_p(v['return'], 1)}, Sharpe {_f(v['sharpe'])}, DD {_p(v['max_dd'], 1)}"  # noqa: E731
            ci = x.get("ci_rule_minus_const")
            L.append(f"| {w} {x['window'][0]} → {x['window'][1]} | {fmt(x.get('rule'))} | {fmt(x.get('const'))} | "
                     f"{fmt(x.get('buyhold'))} | {'[' + _f(ci[0]) + ', ' + _f(ci[1]) + ']' if ci else 'too short'} | "
                     f"{(x.get('rule') or {}).get('trades')} / {(x.get('const') or {}).get('trades')} |")
        dd = vm["daily"]
        L.append(f"| daily stock closes {dd['window'][0]} → {dd['window'][1]} | CAGR {_p(dd['rule']['cagr'], 0)}, Sharpe "
                 f"{_f(dd['rule']['sharpe'])}, DD {_p(dd['rule']['max_dd'], 0)} | CAGR {_p(dd['const_exposure_matched']['cagr'], 0)}, "
                 f"Sharpe {_f(dd['const_exposure_matched']['sharpe'])}, DD {_p(dd['const_exposure_matched']['max_dd'], 0)} | "
                 f"CAGR {_p(dd['equal_weight']['cagr'], 0)}, Sharpe {_f(dd['equal_weight']['sharpe'])}, DD {_p(dd['equal_weight']['max_dd'], 0)} | "
                 f"[{_f(dd['ci_rule_minus_const'][0])}, {_f(dd['ci_rule_minus_const'][1])}] | – |")
        L += ["", ("This book is long about 90% of equity, so its test-window gain is the August-September rally, not the "
                   "rule: the same exposure without the rule made about as much. It is a different book from v1, not a tuning of it.")]
    cd = (items.get("crisis_derisk") or {}).get("numbers")
    if cd:
        v0, ho = cd["v0"], cd["v1_holdout"]
        row = lambda lab, v: f"| {lab} | {_p(v.get('total_return'), 0)} | {_f(v.get('sharpe'))} | {_p(v.get('max_dd'), 0)} | {_f(v.get('calmar'))} |"  # noqa: E731
        L += ["", "### 7d. Crisis exit on BTC", "", _md(items["crisis_derisk"]["question"]), "",
              f"Design: {_md(items['crisis_derisk']['design'])}.", "",
              "| v0, 2019-08 → 2026-09 | Total | Sharpe | Max DD | Calmar |", "|---|---:|---:|---:|---:|",
              row("Buy & hold", v0["buy_hold"]), row("200-day MA", v0["ma200"]), row("40% vol target", v0["voltarget"]),
              row("Crisis module v0", v0["module"]), "",
              (f"v0: {cd['v0_episodes']} exits, {cd['v0_false_alarms']} false alarms ({_share(cd['v0_false_alarm_rate'])}); "
               f"{cd['v0_events_triggered']} of {cd['v0_events_covered']} named crises triggered, most after they began."), "",
              "| v1 holdout, 2023-01 → 2026-09 | Total | Sharpe | Max DD | Calmar |", "|---|---:|---:|---:|---:|",
              row("Chosen configuration", ho["module"]), row("Buy & hold", ho["buy_hold"]), row("200-day MA", ho["ma200"])]
    ie = (items.get("incident_exit") or {}).get("numbers")
    if ie:
        L += ["", "### 7e. Incident exit: hacks and delistings", "", _md(items["incident_exit"]["question"]), "",
              f"Design: {_md(items['incident_exit']['design'])}.", "",
              (f"After {ie['hacks_n']} hacks with Bitget minute data, the median token return from the price alert was "
               f"{_p(ie['hacks_median_24h'], 1)} over 24 hours and {_p(ie['hacks_median_30d'], 1)} over 30 days. After {ie['delist_n']} "
               f"Binance delisting notices it was {_p(ie['delist_median_1h'], 1)} over the first hour and "
               f"{_p(ie['delist_median_to_delist'], 1)} to the delisting date. On X, security accounts posted first in "
               f"{ie['x_hacks_x_first']} of {ie['x_hacks_with_post']} hacks; the median first post came "
               f"{abs(ie['x_hacks_median_lead_min']):.0f} minutes after the price alert."),
              "", "**Reading**: " + _md(items["incident_exit"]["verdict"]) + "."]

    # 8. risk
    L += ["", "## 8. Risk layer and failure modes", "",
          ("### 8a. Test window"), "",
          (f"Rule effects: count and an estimated first-order P&L effect per binding rule ((after − before) × equity × the "
           "name's move to the next decision, frictionless; chained rules add up; later path effects such as cooldowns are "
           "ignored).")]
    if not rd["risk"]:
        L += ["", "Not run yet: no complete test-window run."]
    for rid, x in rd["risk"].items():
        b = x.get("beta") or {}
        bs = (f"beta {_f(b['beta'], 3)} (t {_f(b.get('t'))}, R² {_prob(b.get('r2'))}, {b['n_days']} days): "
              + ("inside" if b.get("within_band") else "**outside**") + f" ±{BETA_BAND}"
              if b.get("status") == "computed" else f"beta: {b.get('status', 'n/a')}")
        L += ["", f"**{rid}**: {bs}.", ""] + _risk_table(x.get("risk"))
    L += ["", ("**Known failure mode.** Beta neutralisation opens hedge positions the model did not ask for; those weights cite "
               "no card. A decision chain below shows the requested weight, each rule that changed it and the executed weight.")]
    L += ["", "### 8b. Every run", "",
          ("Binding-rule counts: `at decision` = `risk.apply` at the 4-hourly decision, `hourly` = `risk.check` between "
           "decisions. Weight moved = Σ|before − after| in fractions of equity (the daily kill's before/after are "
           "equity values and are not summed).")]
    rows = [r for r in m["runs"] if r.get("risk")]
    if not rows:
        L += ["", "No completed run: nothing to count."]
    for r in rows:
        rk = r["risk"]
        L += ["", (f"**{r['run_id']}** ({_md(r['label'])}): {rk['n_records_with_action']} of {rk['n_records']} log "
                   "records carry a binding rule."), ""] + _risk_table(rk)
    ch = m["chains"]
    L += ["", "### 8c. Decisions: requested → risk actions → executed → fills", ""]
    if not ch["items"]:
        L.append("No completed LLM run: no decisions to show.")
    else:
        L += [(f"From `{ch['run']}` ({ch['run_status']}): the {len(ch['items'])} decisions with the largest executed "
               "rebalance (Σ |fill notional| / equity). Per name: previous target, the model's requested weight, each "
               "risk rule that bound (before → after, in order), the post-risk target, the weight the fills actually "
               "moved, and the fills. Names whose weights moved less than "
               f"{EPS_W} and did not trade are left out; a post-risk target that differs from the previous one with "
               "no fill is inside the no-churn band. Cited cards are only those in the decision's logged shown set, "
               "with the text of the cached prompt the model received.")]
    for i, cx in enumerate(ch["items"], 1):
        L += ["", (f"#### 8c.{i}. {cx['ts']} · executed rebalance {_p(cx['executed_size'])} of equity "
                   f"({_money(cx['equity'])} USDT)"), "",
              (f"Risk {cx['risk_version']}, prompt {_i(cx['prompt_version'])}, confidence {_f(cx['confidence'])}, "
               f"{cx['n_cards_shown']} cards shown" + ("" if cx["prompt_cached"] else "; prompt not in the LLM cache")
               + ("; whole-book rules: " + "; ".join(a["rule"] for a in cx["book_actions"]) if cx["book_actions"] else "")
               + "."), "",
              "| Name | Previous | Requested | Risk actions (in order) | Post-risk target | Executed Δw | Fills |",
              "|---|---:|---:|---|---:|---:|---|"]
        for x in cx["rows"]:
            steps = "; ".join(f"{s['rule']} {_w(s['before'])} → {_w(s['after'])}" if _num(s["before"]) is not None
                              else s["rule"] for s in x["risk_steps"]) or "none"
            fl = "; ".join(f"{f['side']} {_f(f['qty'], 4).lstrip('+')} @ {_money(f['price'])} (fee {_money(f['fee'])})"
                           for f in x["fills"]) or "–"
            L.append(f"| {x['ticker']} | {_w(x['prev_target'])} | {_w(x['requested'])} | {_md(steps)} | "
                     f"{_w(x['final_target'])} | {_w(x['executed_dw'])} | {_md(fl)} |")
        for x in (r for r in cx["rows"] if "reason" in r):
            L += ["", f"**{x['ticker']}** (executed {_w(x['executed_dw'])}): "
                  + (f"“{_md(x['reason'])}”" if x["reason"] else "no reason given by the model")]
            if not x["cards"]:
                L.append("- no shown card cited")
            for k in x["cards"]:
                if not k["in_prompt_text"]:
                    L.append(f"- `{k['card_id']}`: shown (logged id), prompt text not in the LLM cache")
                    continue
                tks = ",".join(k.get("tickers") or []) or k.get("scope")
                flag = f" **(cited for {x['ticker']} but about {tks})**" if k.get("off_ticker") else ""
                since = f", since {k['since_bp']} bp" if k.get("since_bp") not in (None, "-") else ""
                age = f"{k['age_h']:.0f}h old" if _num(k.get("age_h")) is not None else "age n/a"
                L.append(f"- `{k['card_id']}` [{_md(tks)}] {age}{since}, stance {k['stance']:+d}, "
                         f"strength {_prob(k['strength'])}, {k['novelty']}: {_md(k['summary'])}{flag}")
            if x["cited_not_shown"]:
                L.append(f"- cited but not in the prompt (ignored): {', '.join(x['cited_not_shown'])}")
        L += ["", f"Log record `{cx['record_hash']}`, LLM request `{cx['llm_request_hash']}`."]

    # 9. live, deviations, limits
    L += ["", "## 9. Live log, deviations and limits", "", "### 9a. Live paper log", ""]
    if lv["status"] == "not run yet":
        L.append(f"Not run yet (`{lv['dir']}` missing).")
    else:
        x = lv.get("metrics")
        L += [(f"`{lv['dir']}`, broker mode {', '.join(lv['modes']) or 'unknown'}: **{lv['label']}**. "
               f"{lv['n_records']} log records; ledger "
               + {True: "verified", False: "**BROKEN**", None: "no log"}[lv.get("log_verified")] + "."), ""]
        if x:
            L.append(f"Equity: {lv['n_hours']} hourly marks, return {_p(x['total_return'])}, max DD {_p(x['max_dd'])}, "
                     f"{x['n_days']} complete days (partial first day {x.get('partial_first_day_h', 0):.0f}h, partial last day "
                     f"{x.get('partial_last_day_h', 0):.0f}h), fees {_money(x['costs']['fees'])}, spread "
                     f"{_money(x['costs']['spread'])} USDT. Too short for Sharpe or win rate to mean anything.")
        L += ["", "| Risk version | Records | Decisions | Risk exits | Fills | Invalid | Errors | Prompt versions | First | Last |",
              "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
        for v, s in lv["by_risk_version"].items():
            L.append(f"| {v} | {s['n_records']} | {s['n_decisions']} | {s['n_risk_exits']} | {s['n_fills']} | "
                     f"{s['n_invalid']} | {s['n_errors']} | {', '.join(f'{k} {n}' for k, n in s['prompt_versions'].items()) or '–'} | "
                     f"{s['first_ts']} | {s['last_ts']} |")
        rc = lv["reconcile"]
        L += ["", "**Replay-vs-live reconciliation**"
              + (f" (`{rc['path']}`" + ("" if rc.get("covers_log", True) else ", **older than the log**") + ")"
                 if rc.get("path") else "") + ":", ""]
        if rc["status"] != "present":
            L.append(f"not run yet (`{rc.get('cmd', 'uv run python -m sentiment.reconcile')}`).")
        else:
            L += [(f"{rc['n_live_records']} live records, {rc['n_decisions']} decisions: identical "
                   f"{rc['n_identical']} ({_share(rc['decision_identity_rate'])}), unverifiable {rc['n_unverifiable']}; "
                   f"prompt identity {_share(rc['prompt_identity_rate'])}, post-risk targets identical "
                   f"{_share(rc['targets_identity_rate'])}, risk context rebuilt {rc['n_risk_ctx_checked']} "
                   f"(mismatched {rc['n_risk_ctx_mismatch']}); hourly checks {rc['n_hourly_checks']}, exit identity "
                   f"{_share(rc['exit_identity_rate'])}; position breaks {rc['n_positions_breaks']}; mismatch inputs "
                   f"{_md(json.dumps(rc.get('mismatch_inputs') or {}))}; store {rc.get('store')}"
                   + _why_not_identical(rc) + "."), "",
                  "| Broker mode | Label | Orders | Matched | Fill gap median (bp, + = live worse) | p90 | Live mid vs bar open (median) | Demo mid vs live mid (median) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
            for mode, f in rc["fills_by_mode"].items():
                L.append(f"| {mode} | {f['label']} | {f['n_orders']} | {f['n_matched']} | {_bp(f['gap_bp_median'])} | "
                         f"{_bp(f['gap_bp_p90'])} | {_bp(f['mid_gap_bp_median'])} | {_bp(f['demo_live_mid_bp_median'])} |")
            L.append("\nA shadow gap compares two fill models (live quotes vs the replay's next-bar open); only Demo "
                     f"fills are exchange fills. Decision identity: **{rc.get('identity_label')}**; fill gap: "
                     f"**{rc.get('gap_label')}**.")
        files = m["figures"].get("live") or []
        if files:
            L += ["", "Figures, live:", ""] + [f"![{f}]({f})" for f in files]

    dvn, cp = rd.get("deviations") or {}, rd.get("code_provenance") or {}
    L += ["", "### 9b. Deviations and errata", "",
          ("PREREG.md is frozen, so dated deviations and errata are listed here and appended to trials.log; git "
           "commit times are authoritative. PREREG's own Deviations section: "
           + (f"\"{_md(dvn.get('prereg_section'))}\"" if dvn.get("prereg_section") is not None else "unreadable")
           + (" (**entries added since the freeze**)." if dvn.get("prereg_section_added_since_freeze") else ".")), ""]
    L += [f"- **{_md(n['date'])}, {_md(n['title'])}.** {_md(n['text'])}" for n in dvn.get("notes") or []]
    chg = cp.get("replay_path_changed_since_freeze")
    L += ["", f"**Test-run code**: {_md(cp.get('verdict') or 'n/a')}."
          + (f" Replay-path files that differ now from the freeze commit: {', '.join(chg) or 'none'}." if chg is not None
             else "")]
    rr = cp.get("rerun") or {}
    if cp.get("runs"):
        L += ["", "| Test run | Code stamp | Code check | Rerun from the freeze commit |", "|---|---|---|---|"]
        for rid, x in cp["runs"].items():
            rri = {True: "same log_last_hash", False: "**different log_last_hash**",
                   None: "not run yet"}[x.get("rerun_identical")]
            L.append(f"| {rid} | `{x.get('stamp') or '–'}` | {_md(x.get('check'))} | {rri} |")
    L += ["", f"Freeze rerun file `{rr.get('path')}`: {rr.get('status')}."]
    tr = dev["trials"]
    L += ["", f"**Trials** (`{tr['path']}`, {tr.get('n_entries', 0)} entries):", ""]
    if tr["entries"]:
        L += ["| When (UTC) | Window | Change | Verdict |", "|---|---|---|---|"]
        for e in tr["entries"]:
            L.append(f"| {_md((e['ts'] or '')[:16])} | {_md(e['window'])} | {_md(e['change'])} | {_md(_short(e.get('verdict') or '–', 240))} |")
    ex = tr.get("explainability")
    if ex:
        L += ["", (f"**Prompt v1 explainability: FAIL** ({_md(ex['ts'][:16])}). Criteria declared before the run: "
                   f"{_md(ex['criteria'])}. Before → after: "
                   + "; ".join(f"{k.replace('_', ' ')} {' → '.join(str(v) for v in val)}"
                               for k, val in ex["before_after"].items() if isinstance(val, list))
                   + ". The prompt was frozen anyway; explainability is shown per decision in section 8c.")]
    lm = m.get("limits") or {}
    L += ["", "### 9c. What this does not prove", ""]
    if lm.get("headline_sharpe_se") is not None:
        L.append(f"- **Short sample.** {lm['headline_n_days']} days: the headline Sharpe {_f(lm['headline_sharpe'])} has a standard "
                 f"error of about {_u(lm['headline_sharpe_se'])} {c('lo2002')} (95% interval {_f(lm['headline_sharpe_ci95'][0])} to "
                 f"{_f(lm['headline_sharpe_ci95'][1])}).")
    L += [("- **Estimated fills.** Every replay number and the whole live log are simulated fills; no Bitget Demo key was "
           "set, so nothing here is an exchange fill."),
          ("- **Many trials.** Besides the pre-registered runs, the dev variants in trials.log and the "
           f"{lm.get('n_other_tests') or 0} strategies of section 7 were tried. The more one tries, the higher the bar a "
           f"single good result must clear {c('harvey2016')}; none of them cleared even its own."),
          ("- **One model, one universe.** One LLM (`" + str(dz.get("model")) + "`) at temperature 0, ten names chosen in "
           "2026-09; a different model, prompt or universe is a different experiment."),
          "- **Cards describe more than they predict.** A large share of news cards explain a move that already happened (section 2)."]

    # 10. references
    L += ["", "## 10. References", "", "Each entry was checked against Crossref (DataCite for the arXiv paper).", ""]
    L += [f"{i}. {r['text']}" for i, r in enumerate(m.get("references") or [], 1)]

    # 11. reproduce
    rp, g = m["reproduction"], m["git"]
    L += ["", "## 11. Reproduce", "",
          ("The replay is a function of the snapshot, the code and the LLM cache (run_id is the run name, no wall-clock "
           "values), so an offline rerun (`SENTIMENT_OFFLINE=1`, no `.env` needed) reproduces its log hashes on the platform "
           "that wrote them, macOS on Apple silicon (CI checks it there); a cache miss stops the run instead of calling the "
           "model. On Linux x86_64 the risk v1 runs differ from their first record: the 60-day beta (pandas mean, variance "
           "and covariance) rounds differently in its last bits, and the weights and fills follow it; the v0 runs, which "
           "have no beta, match on both. Runs are checked against the current data rules by their "
           "cited evidence items" + ("" if rp.get("items_checked", True) else " (item check skipped in this build)")
           + f". Git HEAD `{g.get('head') or 'unknown'}`; PREREG frozen at `{g.get('prereg_freeze_commit') or 'unknown'}`."),
          "",
          (f"LLM cache `{rp['llm_cache']['path']}`: {rp['llm_cache']['responses']} responses "
           f"({', '.join(f'{k} {v}' for k, v in rp['llm_cache']['by_purpose'].items()) or 'none'}), "
           f"{rp['llm_cache']['prompt_tokens']:,} prompt + {rp['llm_cache']['completion_tokens']:,} completion tokens."), ""]
    mf = lk["snapshot_manifest"]
    if mf["status"] != "present":
        L.append(f"**Snapshot**: `{mf['path']}` missing.")
    else:
        L += [(f"**Snapshot**: `{mf['path']}` sha256 `{mf['manifest_sha256']}`; every file matches: "
               f"{'yes' if mf['check_ok'] else 'NO'}{'; unlisted: ' + ', '.join(mf['unlisted']) if mf['unlisted'] else ''}."), "",
              "| File | sha256 | Matches |", "|---|---|---|"]
        for name, v in mf["files"].items():
            L.append(f"| {name} | `{v['sha256']}` | {'yes' if v['matches'] else 'NO'} |")
    L += ["", "| Run | Ledger verify | Records | Last hash | Matches the run's metrics.json | Current rules | Code vs PREREG freeze |",
          "|---|---|---:|---|---|---|---|"]
    for r in m["runs"]:
        if r.get("metrics"):
            ver = {True: "ok", False: "BROKEN", None: "no log"}[r.get("log_verified")]
            match = {True: "yes", False: "NO", None: "n/a"}[r.get("hash_matches_recorded")]
            pv = r.get("provenance")
            cur = ("n/a (live)" if pv is None else "**STALE**" if pv["stale"] else
                   f"yes ({pv['n_evidence_items']} cited items found)" if pv.get("items_checked") else "not checked")
            code = _md((r.get("code") or {}).get("check")) if r.get("code") else "n/a"
            L.append(f"| {r['run_id']} | {ver} | {r['n_log_records']} | `{r.get('log_last_hash') or '–'}` | {match} "
                     f"| {cur} | {code} |")
    L += ["", "```bash"] + [f"# {cmd['what']}\n{cmd['cmd']}" for cmd in rp["commands"]] + ["```"]
    return "\n".join(L).rstrip() + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(OUT), help="run directories (default out)")
    p.add_argument("--report-dir", default=str(REPORT_DIR))
    p.add_argument("--reconcile", default=None, help="reconcile.json (default <out>/live/reconcile.json)")
    p.add_argument("--signal-tests", default=None, help="signal tests JSON (default <out>/signal_tests_test.json)")
    a = p.parse_args(argv)
    m = build(Path(a.out), Path(a.report_dir), reconcile=a.reconcile, signal=a.signal_tests)
    done = sum(1 for r in m["runs"] if r.get("metrics"))
    print(f"{_rel(Path(a.report_dir) / 'REPORT.md')}: {len(m['runs'])} runs ({done} with metrics), "
          f"figures {sum(len(v) for v in m['figures'].values())}", file=sys.stderr)


if __name__ == "__main__":
    main()
