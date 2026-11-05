"""Progress of a run, for the on-screen dashboard (standing rule of the author, 2026-10-23).

A ``Tracker`` counts the items of one stage (persons, steps, pairs) and writes a small JSON snapshot to
``data/private/progress/<run>.json`` at most every ``every`` seconds and when the stage ends. The snapshot holds
counters only (no personal text): stage, done/total, rate, ETA, status, the next step, and the client's call
counters (network calls, cache hits, failures, 429, 5xx), so that ``rerun.dashboard`` can draw the bars.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from . import config

PROGRESS_DIR = config.PRIVATE / "progress"
CLIENT_KEYS = ("calls", "cache_hits", "failed", "http_429", "http_5xx", "invalid_json", "network_error")


class Tracker:
    def __init__(self, run: str, stage: str, total: int, *, unit: str = "items", client=None,
                 directory: Path | None = None, every: float = 2.0, clock=time.time, next_step: str = "",
                 heartbeat: float | None = 20.0):
        self.run, self.stage, self.total, self.unit = run, stage, max(0, int(total)), unit
        self.client, self.every, self.clock, self.next_step = client, every, clock, next_step
        self.dir = Path(directory) if directory else PROGRESS_DIR
        self.done = 0
        self.note = ""
        self.status = "running"
        self.started = clock()
        self._last_write = 0.0
        self.last_advance = None
        self._lock = threading.Lock()
        self._client0 = self._client_counts()
        self.write(force=True)
        # heartbeat: the snapshot (calls, 429, elapsed) is refreshed even when items finish slowly, so that the
        # dashboard does not report a slow stage as stopped
        self._stop = threading.Event()
        if heartbeat:
            t = threading.Thread(target=self._beat, args=(heartbeat,), daemon=True)
            t.start()

    def _beat(self, period: float):
        while not self._stop.wait(period):
            self.write(force=True)

    # ---- counting --------------------------------------------------------------------------------------------------
    def advance(self, n: int = 1):
        with self._lock:
            self.done += n
            self.last_advance = self.clock()
        self.write()

    def set_note(self, text: str):
        self.note = text
        self.write(force=True)

    def close(self, status: str = "done"):
        self._stop.set()
        self.status = status
        self.write(force=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close("done" if exc_type is None else f"stopped: {exc_type.__name__}")
        return False

    # ---- snapshot --------------------------------------------------------------------------------------------------
    def _client_counts(self) -> dict:
        stats = getattr(self.client, "stats", None) or {}
        return {k: int(stats.get(k, 0)) for k in CLIENT_KEYS}

    def snapshot(self) -> dict:
        now = self.clock()
        elapsed = max(now - self.started, 1e-9)
        rate = self.done / elapsed * 60.0
        left = max(self.total - self.done, 0)
        eta = (left / (self.done / elapsed)) if self.done else None
        cur = self._client_counts()
        delta = {k: cur[k] - self._client0.get(k, 0) for k in CLIENT_KEYS}
        return {"run": self.run, "stage": self.stage, "unit": self.unit, "total": self.total, "done": self.done,
                "fraction": (self.done / self.total) if self.total else 1.0, "status": self.status,
                "started": self.started, "updated": now, "elapsed_s": round(elapsed, 1),
                "rate_per_min": round(rate, 2), "eta_s": None if eta is None else round(eta, 1),
                "next_step": self.next_step, "note": self.note, "client": delta, "pid": os.getpid(),
                "limits": self.client.model_limits() if hasattr(self.client, "model_limits") else {},
                "last_advance": self.last_advance}

    def write(self, force: bool = False):
        now = self.clock()
        if not force and now - self._last_write < self.every:
            return
        self._last_write = now
        snap = self.snapshot()
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{self.run}.json"
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
        for _ in range(5):   # Windows: the dashboard may be reading the file at this moment
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.05)
        tmp.unlink(missing_ok=True)


def tick(tracker):
    """``tracker.advance()`` if a tracker is given (stage functions accept ``tracker=None``)."""
    if tracker is not None:
        tracker.advance()
