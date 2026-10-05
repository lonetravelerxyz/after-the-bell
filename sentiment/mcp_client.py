"""Minimal client for Bitget's official read-only data MCP (https://agent.bitget.com/mcp, no API key).
Vendored byte-for-byte from A4 (a4-session-factor/src/mcp_client.py) except this line, so B does not import A4's code.

The server speaks streamable-HTTP MCP and exposes two tools: `guide` (catalog) and `do_query`.
"""
from __future__ import annotations

import json
import time
import urllib.request

URL = "https://agent.bitget.com/mcp"
_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
            "User-Agent": "a4-session-factor/0.1"}


class BitgetMCP:
    def __init__(self) -> None:
        self.sid, self._id = None, 0
        self.sid, _ = self._post({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "a4", "version": "0.1"}}})
        try:
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:  # noqa: BLE001 - the notification has no body on some deployments
            pass

    def _post(self, body: dict) -> tuple[str | None, str]:
        h = dict(_HEADERS)
        if self.sid:
            h["Mcp-Session-Id"] = self.sid
        r = urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers=h), timeout=60)
        return r.headers.get("mcp-session-id"), r.read().decode()

    def query(self, entry_id: str, **params) -> list[dict]:
        """Run one catalog entry; returns the `results` list. Retries transient 5xx from the backend."""
        for attempt in range(5):
            self._id += 1
            _, txt = self._post({"jsonrpc": "2.0", "id": self._id, "method": "tools/call",
                                 "params": {"name": "do_query", "arguments": {"entry_id": entry_id, "params": params}}})
            data = [ln[5:].strip() for ln in txt.splitlines() if ln.startswith("data:")]
            out = json.loads(json.loads(data[-1])["result"]["content"][0]["text"])
            if out.get("success"):
                res = out["data"]
                return res.get("results", res) if isinstance(res, dict) else res
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"{entry_id} {params}: {str(out)[:200]}")
