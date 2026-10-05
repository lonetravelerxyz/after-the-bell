"""Cached OpenAI-compatible chat client for project B (contract: sentiment/CONTRACT.md, "LLM").

Every call is keyed by sha256(model + canonical JSON of messages) and stored as
sentiment/cache/llm/{key}.json with the request, the response text, token usage and purpose.
A cache hit never touches the network, so a replay over cached prompts is deterministic and runs
offline. SENTIMENT_OFFLINE=1 turns a cache miss into `CacheMiss` instead of a paid call.

Credentials: BITGET_QWEN_API_KEY / BITGET_QWEN_BASE_URL from the environment or .env, read only when a
network call is needed. The model id (part of every cache key) is BITGET_QWEN_MODEL when set, else the
pinned `config.LLM_MODEL`, so an offline replay from the committed cache needs no credentials at all. Calls use temperature 0 and
enable_thinking=false. `parse_json` is the shared tolerant JSON extractor for model output.

Run: uv run python -m sentiment.llm   -> prints usage_summary() of the cache
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from sentiment.config import CACHE, ENV_FILE, LLM_MODEL

CACHE_DIR = CACHE / "llm"
RETRIES = 4                     # retries after the first attempt, for 5xx / 429 / timeouts
BACKOFF_S = 2.0                 # 2, 4, 8, 16 s
TIMEOUT_S = 180


class CacheMiss(RuntimeError):
    """Raised when SENTIMENT_OFFLINE=1 and the prompt is not in the cache."""


class LLMError(RuntimeError):
    """Raised when the endpoint fails after all retries, or returns an unusable response."""


def _env() -> dict[str, str]:
    env_file = ENV_FILE
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return {k: os.environ.get(k, "") for k in ("BITGET_QWEN_API_KEY", "BITGET_QWEN_BASE_URL", "BITGET_QWEN_MODEL")}


def model_name() -> str:
    """The model id used in cache keys: BITGET_QWEN_MODEL (env or .env), else config.LLM_MODEL."""
    return os.environ.get("BITGET_QWEN_MODEL") or _env()["BITGET_QWEN_MODEL"] or LLM_MODEL


def cache_key(model: str, messages: list[dict]) -> str:
    canon = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((model + "\n" + canon).encode()).hexdigest()


def offline() -> bool:
    return os.environ.get("SENTIMENT_OFFLINE", "").strip() not in ("", "0", "false", "False")


def _post(cfg: dict[str, str], body: dict) -> dict:
    url = cfg["BITGET_QWEN_BASE_URL"].rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {cfg['BITGET_QWEN_API_KEY']}", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=TIMEOUT_S).read())


def _call(model: str, messages: list[dict]) -> tuple[str, dict]:
    cfg = _env()
    if not cfg["BITGET_QWEN_API_KEY"] or not cfg["BITGET_QWEN_BASE_URL"]:
        raise LLMError("BITGET_QWEN_API_KEY / BITGET_QWEN_BASE_URL not set (env or .env)")
    body = {"model": model, "temperature": 0, "enable_thinking": False, "messages": messages}
    err: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            out = _post(cfg, body)
            text = out["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise LLMError(f"no text content in response: {str(out)[:200]}")
            return text, out.get("usage") or {}
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:     # 4xx other than rate limit: do not retry
                try:
                    detail = e.read()[:300]
                except Exception:  # noqa: BLE001 - body is optional context only
                    detail = b""
                raise LLMError(f"HTTP {e.code}: {detail!r}") from e
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            err = e
        if attempt < RETRIES:
            time.sleep(BACKOFF_S * 2 ** attempt)
    raise LLMError(f"failed after {RETRIES + 1} attempts: {err!r}")


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def cached(messages: list[dict]) -> str | None:
    """Cached response text for these messages, or None (never calls the network)."""
    f = CACHE_DIR / f"{cache_key(model_name(), messages)}.json"
    return json.loads(f.read_text())["response"] if f.exists() else None


def complete(messages: list[dict], *, purpose: str) -> str:
    """Chat completion text for `messages`; cache first, network on a miss (unless offline)."""
    model = model_name()
    key = cache_key(model, messages)
    f = CACHE_DIR / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())["response"]
    if offline():
        raise CacheMiss(f"{purpose}: {key[:12]} not cached and SENTIMENT_OFFLINE=1")
    t0 = time.time()
    text, usage = _call(model, messages)
    _write_atomic(f, {"key": key, "model": model, "purpose": purpose, "request": {"messages": messages},
                      "response": text, "usage": usage, "latency_s": round(time.time() - t0, 2),
                      "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return text


def usage_summary() -> dict:
    """Token totals over the whole cache: overall and per purpose."""
    tot = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    by: dict[str, dict] = {}
    for f in sorted(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []:
        try:
            rec = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        u = rec.get("usage") or {}
        p = by.setdefault(rec.get("purpose", "?"), {k: 0 for k in tot})
        for d in (tot, p):
            d["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                d[k] += int(u.get(k) or 0)
    return {**tot, "by_purpose": by}


# ---------------------------------------------------------------- tolerant JSON extraction

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_SMART = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _balanced(text: str, start: int) -> str | None:
    """The balanced {...} or [...] span starting at text[start], respecting JSON strings."""
    open_c = text[start]
    close_c = "}" if open_c == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repairs(s: str) -> list[str]:
    fixed = re.sub(r",\s*([}\]])", r"\1", s)                                  # trailing commas
    py = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", re.sub(r"\bNone\b", "null", fixed)))
    return [s, fixed, py, py.translate(_SMART)]


def _first(text: str, opener: str) -> tuple[int, int, dict | list] | None:
    """(start, end, value) of the first balanced span opened by `opener` that parses as JSON."""
    for i, c in enumerate(text):
        if c != opener:
            continue
        span = _balanced(text, i)
        if span is None:
            continue
        for s in _repairs(span):
            try:
                return i, i + len(span), json.loads(s)
            except json.JSONDecodeError:
                pass
    return None


def parse_json(text: str) -> dict | list:
    """First JSON object/array in model output: tolerates code fences, prose around it, <think>
    blocks, trailing commas, Python literals and smart quotes. An object wins over an earlier
    bracket in prose ("see [1]"), unless that array encloses it. Raises ValueError if none parses."""
    if not isinstance(text, str):
        raise ValueError(f"expected str, got {type(text).__name__}")
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for cand in [m.group(1) for m in _FENCE.finditer(body)] + [body]:
        obj, arr = _first(cand, "{"), _first(cand, "[")
        if obj and arr and arr[0] < obj[0] < arr[1]:
            return arr[2]
        if obj or arr:
            return (obj or arr)[2]
    raise ValueError(f"no JSON object in model output: {text[:200]!r}")


if __name__ == "__main__":
    print(json.dumps(usage_summary(), indent=1))
