"""Text embeddings with the local ``bge-m3`` model (semantic novelty and PES clusters, spec 4.5).

``Embedder(texts) -> (n, d)`` array of unit vectors, in the order of the texts. Vectors are cached per text
(sha256 of model + text) in a JSONL file under ``data/private/``; they are stored as float16 and always used
after that quantisation, so a cache hit is exactly equal to the vector returned by the fresh call. A batch
that keeps failing raises ``RuntimeError``: a missing embedding is never replaced by zeros.
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np

from . import config

EMBED_MODEL = config.EMBEDDER
BATCH_SIZE = 64


def _http(url, body, timeout):
    import requests
    r = requests.post(url, json=body, timeout=timeout)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {}


class Embedder:
    def __init__(self, cache_path=None, transport=_http, model: str = EMBED_MODEL, batch_size: int = BATCH_SIZE,
                 max_attempts: int = 4, timeout: float = 300, sleep=time.sleep, private_root=config.PRIVATE):
        self.model, self.batch_size, self.max_attempts = model, batch_size, max_attempts
        self._transport, self._timeout, self._sleep = transport, timeout, sleep
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._path = Path(cache_path) if cache_path else None
        self._needs_newline = False
        if self._path:
            if not self._path.resolve().is_relative_to(Path(private_root).resolve()):
                raise ValueError("embedding cache must be under data/private/ (it holds personal text)")
            self._load()

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{text}".encode("utf-8")).hexdigest()

    @staticmethod
    def _pack(v: np.ndarray) -> str:
        return base64.b64encode(v.astype(np.float16).tobytes()).decode("ascii")

    @staticmethod
    def _unpack(s: str) -> np.ndarray:
        return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)

    def _load(self):
        if not self._path.exists():
            return
        raw = self._path.read_bytes()
        self._needs_newline = bool(raw) and not raw.endswith(b"\n")
        for line in raw.decode("utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
                self._cache[rec["key"]] = self._unpack(rec["vec"])
            except (ValueError, KeyError, TypeError):
                continue  # damaged or truncated line

    def _store(self, items: list[tuple[str, np.ndarray]]):
        for k, v in items:
            self._cache[k] = v
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            if self._needs_newline:
                f.write("\n")
                self._needs_newline = False
            for k, v in items:
                f.write(json.dumps({"key": k, "vec": self._pack(v)}) + "\n")

    def _fetch(self, texts: list[str]) -> list[np.ndarray]:
        url = config.LOCAL_URL + "/api/embed"
        last = ""
        for attempt in range(self.max_attempts):
            try:
                status, payload = self._transport(url, {"model": self.model, "input": texts}, self._timeout)
            except Exception as e:  # noqa: BLE001
                status, payload, last = 0, {}, type(e).__name__
            embs = payload.get("embeddings") if isinstance(payload, dict) else None
            if status == 200 and isinstance(embs, list) and len(embs) == len(texts):
                return [np.asarray(e, dtype=np.float32) for e in embs]
            if status == 200:
                raise RuntimeError(f"embedding count {len(embs or [])} != {len(texts)} texts")
            last = last or f"http_{status}"
            self._sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"embedding failed after {self.max_attempts} attempts ({last})")

    def __call__(self, texts) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        keys = [self._key(t) for t in texts]
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self._cache))
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i:i + self.batch_size]
            vecs = self._fetch(batch)
            # quantise to float16 now, so that fresh and cached vectors are identical
            q = [np.frombuffer(v.astype(np.float16).tobytes(), dtype=np.float16).astype(np.float32) for v in vecs]
            self._store([(self._key(t), v) for t, v in zip(batch, q)])
        out = np.vstack([self._cache[k] for k in keys]).astype(np.float32)
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
