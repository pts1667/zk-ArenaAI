"""Minimal, robust ControllerAI WebSocket client used by the Zero-K bot.

The game waits for this process while it is in synchronous mode, so the
transport is on the critical path.  ``websockets`` (with its C speedups) and
``orjson`` are substantially faster than the older pure-Python websocket
implementation, especially for the large per-frame observations.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

import requests
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

try:
    import orjson

    def _loads(raw: str | bytes) -> Any:
        return orjson.loads(raw)

    def _dumps(payload: Any) -> str:
        return orjson.dumps(payload).decode("utf-8")

except Exception:  # pragma: no cover - fallback for minimal installations
    import json

    def _loads(raw: str | bytes) -> Any:
        return json.loads(raw)

    def _dumps(payload: Any) -> str:
        return json.dumps(payload, separators=(",", ":"))


class QueryError(RuntimeError):
    pass


class Client:
    """One team's ControllerAI connection."""

    def __init__(self, master_url: str, team_id: int, log: Callable[[str], None]):
        self.master_url = master_url.rstrip("/")
        self.team_id = int(team_id)
        self.log = log
        self.ws = None
        self.inbox: deque[dict] = deque()
        self.connected_at = 0.0
        self.last_message_at = 0.0
        self._query_seq = 0
        self._closed = False

    # ------------------------------------------------------------------ setup
    def discover(self, timeout: float = 180.0) -> str:
        deadline = time.time() + timeout
        last_error = ""
        while time.time() < deadline:
            try:
                response = requests.get(self.master_url + "/list", timeout=2.0)
                response.raise_for_status()
                data = response.json()
                inst = data.get(str(self.team_id))
                if inst and inst.get("reachable", True) and inst.get("wsUrl"):
                    return inst["wsUrl"]
            except Exception as exc:  # master not up yet, etc.
                last_error = str(exc)
            time.sleep(0.1)
        raise RuntimeError(f"ControllerAI team {self.team_id} did not appear: {last_error}")

    def connect(self, timeout: float = 180.0) -> None:
        ws_url = self.discover(timeout)
        self.log(f"connecting websocket {ws_url}")
        self.ws = ws_connect(
            ws_url,
            open_timeout=timeout,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=2.0,
            max_size=64 * 1024 * 1024,
            compression=None,
            proxy=None,
        )
        self.connected_at = time.time()
        self.last_message_at = time.time()

    def close(self) -> None:
        self._closed = True
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    # --------------------------------------------------------------- transport
    def send(self, payload: Any) -> None:
        if self.ws is None or self._closed:
            raise RuntimeError("websocket is not connected")
        if isinstance(payload, (dict, list)):
            self.ws.send(_dumps(payload))
        else:
            self.ws.send(str(payload))

    def send_commands(self, commands: list[dict]) -> None:
        if not commands:
            return
        # Sending an array in one message is cheaper than one message per command.
        self.send(commands)

    def _recv_ws(self, timeout: float | None = None) -> dict | None:
        """Read only from the websocket, never from the inbox."""
        if self.ws is None or self._closed:
            return None
        try:
            raw = self.ws.recv(timeout=timeout)
        except TimeoutError:
            return None
        except ConnectionClosed:
            self._closed = True
            return None
        except OSError:
            self._closed = True
            return None
        if not raw:
            return None
        try:
            msg = _loads(raw)
        except Exception:
            return None
        if not isinstance(msg, dict):
            return None
        self.last_message_at = time.time()
        return msg

    def recv(self, timeout: float | None = None) -> dict | None:
        """Return the next message, or None on timeout/close."""
        if self.inbox:
            return self.inbox.popleft()
        return self._recv_ws(timeout)

    def wait_message(
        self,
        predicate: Callable[[dict], bool],
        timeout: float = 10.0,
    ) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.recv(timeout=min(0.25, max(0.01, deadline - time.time())))
            if msg is None:
                continue
            if predicate(msg):
                return msg
            self.inbox.append(msg)
        return None

    # ------------------------------------------------------------------ query
    query_time = 0.0
    query_counts: dict = {}
    # Game frame of the current observation; set by the host each step.
    frame = 0
    # Results of these queries change slowly (terrain, build-site occupancy),
    # so identical queries within the TTL (in frames) reuse the answer instead
    # of a blocking round trip.  Placement loops repeat the same probes on
    # every retry, which made them the bulk of all queries.
    CACHE_TTL = {"elevation_at": 300, "can_build_at": 90, "closest_build_site": 90}
    _cache: dict = {}
    cache_hits = 0

    def query(self, query: dict, timeout: float = 10.0) -> Any:
        ttl = self.CACHE_TTL.get(query.get("type"))
        key = None
        if ttl:
            key = _dumps(query)
            hit = self._cache.get(key)
            if hit is not None and self.frame - hit[0] < ttl:
                self.cache_hits += 1
                return hit[1]
        t0 = time.time()
        try:
            res = self._query(query, timeout)
            if key is not None:
                if len(self._cache) > 20000:
                    self._cache = {
                        k: v for k, v in self._cache.items() if self.frame - v[0] < 300
                    }
                self._cache[key] = (self.frame, res)
            return res
        finally:
            dt = time.time() - t0
            self.query_time += dt
            k = str(query.get("type"))
            n, t = self.query_counts.get(k, (0, 0.0))
            self.query_counts[k] = (n + 1, t + dt)

    def _query(self, query: dict, timeout: float = 10.0) -> Any:
        self._query_seq += 1
        request_id = f"zkbot-{self._query_seq}"
        message = {"type": "query", "requestId": request_id, "query": query}
        self.send(message)
        deadline = time.time() + timeout
        deferred: list[dict] = []

        def matches(msg: dict) -> bool:
            typ = msg.get("type")
            data = msg.get("data") or {}
            if typ == "query_result" and data.get("requestId") == request_id:
                return True
            if typ == "query_error" and data.get("requestId") == request_id:
                raise QueryError(str(data.get("message") or data))
            return False

        try:
            while self.inbox:
                msg = self.inbox.popleft()
                if matches(msg):
                    return (msg.get("data") or {}).get("result")
                deferred.append(msg)
            while time.time() < deadline:
                msg = self._recv_ws(timeout=min(0.25, max(0.01, deadline - time.time())))
                if msg is None:
                    continue
                if matches(msg):
                    return (msg.get("data") or {}).get("result")
                deferred.append(msg)
            raise QueryError(f"query timed out: {query.get('type')}")
        finally:
            if deferred:
                self.inbox.extendleft(reversed(deferred))

    # --------------------------------------------------------------- settings
    def set_settings(self, settings: dict) -> None:
        self.send({"type": "set_settings", "settings": settings})

    def finish_frame(self) -> None:
        self.send({"type": "finish_frame"})
