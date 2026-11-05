"""LLM client: key pool, rate limit, retry, cache and checkpoint, usage log.

Every call of the rerun goes through ``LLMClient.chat``. Design (spec section 10):

* cache key = sha256 of (model, messages, options, schema); a hit returns the stored
  answer with ``cached=True``;
* the cache file is also the checkpoint: every successful call is appended to a JSONL
  file at once, so a new process resumes without repeating completed calls; failed
  calls are never stored (they are retried on resume);
* cloud models use the keys round-robin; 429 puts a key in a cool-down with
  exponential back-off and moves to the next key; 401/403 disables the key; 5xx,
  timeouts and network errors are retried with back-off up to ``max_attempts``;
* with a schema the answer must be a JSON object or array exactly as returned
  (no code fences, no prose around it): otherwise it is retried
  ``invalid_json_retries`` times and then reported as ``invalid_json``. A retry is a
  different request (the wrong answer is appended with a correction and the seed is
  SEED + retry), because an identical request at temperature 0 gives the same output;
  the cache key stays that of the original request and ``LLMResult.retries`` records
  how many retries were needed;
* a JSON wrapped in exactly ONE code fence (```json or ``` first, ``` last, nothing else outside,
  spaces excluded) is accepted and counted as ``fenced`` per model (decision 2026-10-20: some models,
  e.g. gemma4:31b, always answer that way); prose around, several blocks or a truncated JSON stay
  ``invalid_json``;
* an answer cut by the output limit (``done_reason == "length"``) is not an invalid JSON: it is
  retried with ``num_predict`` doubled up to ``NUM_PREDICT_CAP`` (32,768; 16,384 when the caller
  gave none), counted per model (``truncated:<model>``) and, if still cut at the cap, reported as
  ``truncated``; reasoning models spend part of the limit thinking;
* 429 and 401/403 do not consume ``max_attempts``: rate limits are bounded by a total
  waiting time per call (``rate_limit_max_wait``), then the call fails with
  ``rate_limited``;
* cache and usage log hold personal text: they must live under ``data/private/``;
* keys are never written anywhere: the usage log holds the key index only, and every
  error text is redacted before it is stored or returned.

The HTTP layer is injectable (``transport``) so that tests need no network.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config

LLM_TIMEOUT = 600
NUM_PREDICT_CAP = 32768
NUM_PREDICT_DEFAULT_START = 16384
RATE_LIMIT_MAX_WAIT = 1800.0  # seconds of waiting for a free key, per call
CORRECTION = "Return only valid JSON matching the schema, no other text"


@dataclass
class LLMResult:
    text: str = ""
    json: object | None = None
    ok: bool = False
    error: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    key_index: int | None = None
    cached: bool = False
    retries: int = 0
    fenced: bool = False  # the JSON came wrapped in exactly one code fence (accepted, counted per model)
    truncations: int = 0  # times the answer hit the output limit (done_reason "length") and was retried


_FENCE = re.compile(r"\s*```(?:json)?\s*(.*?)\s*```\s*", re.DOTALL | re.IGNORECASE)


def parse_json_answer(content: str):
    """(parsed, fenced) for an answer that must be a JSON object or array; raises ValueError.

    Accepts the bare JSON or the JSON wrapped in exactly one code fence and nothing else."""
    try:
        parsed, fenced = json.loads(content), False
    except ValueError:
        m = _FENCE.fullmatch(content)
        if not m or "```" in m.group(1):
            raise
        parsed, fenced = json.loads(m.group(1)), True
    if not isinstance(parsed, (dict, list)):
        raise ValueError("not an object or array")
    return parsed, fenced


def cache_key(model, messages, options, schema, think=None) -> str:
    # ``think`` enters the key only when given, so that the keys of requests without it do not change
    parts = [model, messages, options, schema] + ([{"think": think}] if think is not None else [])
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_keys(path=config.KEYS_PATH) -> list[str]:
    """Key strings only (names and types are ignored). Never print the result."""
    with open(path, encoding="utf-8") as f:
        return [e["key"] for e in json.load(f)["CLOUD_API_KEYS"]]


def load_key_types(path=config.KEYS_PATH) -> list[str]:
    """Plan of each key ('pro' or 'free', lower case; 'free' when the field is missing), in the order of
    ``load_keys``."""
    with open(path, encoding="utf-8") as f:
        return [str(e.get("type") or "free").lower() for e in json.load(f)["CLOUD_API_KEYS"]]


def http_transport(url, headers, body, timeout):
    import requests
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    try:
        payload = r.json()
    except ValueError:
        payload = None
    return r.status_code, payload


QUOTA_COOLDOWN = 1800.0   # seconds a key rests after "Free usage limit" (5-hour or weekly cap of the free plan)


def quota_kind(payload) -> str | None:
    """Kind of a 429 answer of Ollama Cloud: 'monthly' (monthly usage limit of the account), 'window' (free usage
    limit: 5-hour or weekly cap), or None (an ordinary rate limit)."""
    msg = ""
    if isinstance(payload, dict):
        msg = str(payload.get("error", ""))
    msg = msg.lower()
    if "monthly usage limit" in msg:
        return "monthly"
    if "usage limit" in msg:
        return "window"
    return None


class AdaptiveLimit:
    """Limit of calls in flight for one model, adapted to the 429 answers of the cloud.

    Every ``period`` seconds the outcomes of the last ``window`` seconds are checked: if the share of 429 is above
    ``max_429_share`` the limit is halved (not below ``floor``); if there was no 429 and the limit was reached at
    least once, it grows by one (not above ``ceiling``, the configured value). Used as a context manager around a
    call; ``record(rate_limited)`` is called for every answer of the server.
    """

    def __init__(self, ceiling: int, *, floor: int = 2, start: int | None = None, window: float = 60.0,
                 period: float = 30.0, max_429_share: float = 0.05, min_events: int = 10, clock=time.monotonic):
        self.ceiling, self.floor = max(1, int(ceiling)), max(1, min(int(floor), int(ceiling)))
        self.limit = min(self.ceiling, max(self.floor, int(start or ceiling)))
        self.window, self.period, self.max_share, self.min_events = window, period, max_429_share, min_events
        self._clock = clock
        self._cond = threading.Condition()
        self._in_flight = 0
        self._saturated = False
        self._events: list[tuple[float, bool]] = []
        self._last_adjust = clock()
        self.changes: list[tuple[float, int]] = []

    def __enter__(self):
        with self._cond:
            while self._in_flight >= self.limit:
                self._saturated = True
                self._cond.wait(timeout=1.0)
            self._in_flight += 1
            if self._in_flight >= self.limit:
                self._saturated = True
        return self

    def __exit__(self, *exc):
        with self._cond:
            self._in_flight -= 1
            self._cond.notify_all()
        return False

    def record(self, rate_limited: bool):
        with self._cond:
            now = self._clock()
            self._events.append((now, bool(rate_limited)))
            self._events = [e for e in self._events if now - e[0] <= self.window]
            if now - self._last_adjust < self.period:
                return
            n = len(self._events)
            share = sum(1 for _, r in self._events if r) / n if n else 0.0
            old = self.limit
            if n >= self.min_events and share > self.max_share:
                self.limit = max(self.floor, self.limit // 2)
                self._events.clear()
            elif share == 0.0 and self._saturated:
                self.limit = min(self.ceiling, self.limit + 1)
            self._saturated = False
            self._last_adjust = now
            if self.limit != old:
                self.changes.append((now, self.limit))
                self._cond.notify_all()


class LLMClient:
    def __init__(self, keys, transport=http_transport, cache_path=None, usage_path=None,
                 sleep=time.sleep, clock=time.monotonic, max_attempts=6,
                 invalid_json_retries=2, backoff_base=2.0, backoff_cap=120.0,
                 min_interval=0.0, max_concurrency=2, timeout=LLM_TIMEOUT,
                 digests=None, rate_limit_max_wait=RATE_LIMIT_MAX_WAIT,
                 private_root=config.PRIVATE, events_path=None, model_concurrency=None,
                 key_types=None, paid_models=(), paid_key_concurrency=None, grow_truncated=True):
        """``key_types``: plan of each key ('pro'/'free'); with it, the ``paid_models`` (no longer in the free plan
        since 2026-10-23) go only to the paid keys and the other models only to the free keys, which spares the
        paid credit. Without it every model may use every key.
        ``grow_truncated``: an answer cut by the output limit is retried with a doubled limit (right for the reasoning
        judges). With False it fails at once as ``truncated``: the annotators (D99) are small models that, when they
        do not close the answer, repeat themselves, and doubling the limit only spends hours on a loop."""
        self._keys = list(keys)
        self._transport = transport
        self._sleep = sleep
        self._clock = clock
        self.max_attempts = max_attempts
        self.invalid_json_retries = invalid_json_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.min_interval = min_interval
        self.timeout = timeout
        self.grow_truncated = grow_truncated
        self.rate_limit_max_wait = rate_limit_max_wait
        self.digests = dict(digests or {})
        self.stats = Counter()
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        root = Path(private_root).resolve()
        self._events_path = Path(events_path) if events_path else None
        for label, pth in (("cache_path", cache_path), ("usage_path", usage_path), ("events_path", events_path)):
            if pth and not Path(pth).resolve().is_relative_to(root):
                raise ValueError(f"{label} must be under {root} (it holds personal text)")
        n = len(self._keys)
        self._dead = [False] * n
        self._cool_until = [0.0] * n
        self._next_free = [0.0] * n
        self._strikes = [0] * n
        self._paid_models = set(paid_models)
        self._pools = None
        if key_types is not None:
            paid = [i for i, t in enumerate(key_types) if str(t).lower() == "pro"]
            self._pools = {"paid": set(paid), "free": set(range(n)) - set(paid)}
        per_key = [paid_key_concurrency if self._pools and i in self._pools["paid"] and paid_key_concurrency
                   else max_concurrency for i in range(n)]
        self._sem = [threading.Semaphore(k) for k in per_key]
        # calls in flight per model (a model absent from the dict is not limited): adaptive limit that halves on a
        # burst of 429 and grows by one while calls are clean (main-PC phase, 2026-10-23)
        self._model_sem = {m: AdaptiveLimit(int(k), start=config.MODEL_CONCURRENCY_START.get(m), clock=self._clock)
                           for m, k in (model_concurrency or {}).items()}
        self._rr: dict[int, int] = {}
        self._cache: dict[str, dict] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        self._usage_path = Path(usage_path) if usage_path else None
        self._needs_newline = False
        self._load_cache()

    def set_model_concurrency(self, model: str, n: int | None):
        """Change the limit of calls in flight of a model (None removes it); only between runs of calls."""
        if n:
            self._model_sem[model] = AdaptiveLimit(int(n), clock=self._clock)
        else:
            self._model_sem.pop(model, None)

    def model_limits(self) -> dict:
        """{model: current limit of calls in flight} (shown by the dashboard)."""
        return {m: lim.limit for m, lim in self._model_sem.items()}

    def _limit_outcome(self, model: str, rate_limited: bool):
        lim = self._model_sem.get(model)
        if lim is not None:
            lim.record(rate_limited)

    def __repr__(self):  # never expose keys
        return f"LLMClient(keys={len(self._keys)}, cached={len(self._cache)})"

    # ---- cache / checkpoint -------------------------------------------------
    def _load_cache(self):
        if not self._cache_path or not self._cache_path.exists():
            return
        raw = self._cache_path.read_bytes()
        self._needs_newline = bool(raw) and not raw.endswith(b"\n")
        for line in raw.decode("utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
                self._cache[rec["key"]] = rec
            except (ValueError, KeyError, TypeError):
                continue  # truncated or damaged line (process killed mid-write)

    def _append(self, path: Path | None, rec: dict):
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(path, "a", encoding="utf-8") as f:
            if path == self._cache_path and self._needs_newline:
                f.write("\n")
                self._needs_newline = False
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- public -------------------------------------------------------------
    supports_affinity = True

    def chat(self, model, messages, options=None, schema=None, think=None, affinity=None) -> LLMResult:
        """``think``: reasoning effort for models that support it ("low", "medium", "high"; a bool for the
        others); sent as the top-level ``think`` field of the request and part of the cache key.
        ``affinity``: an integer; calls with the same value go to the same key when it is free (not in the cache
        key)."""
        opts = {**config.DEFAULT_OPTIONS, **(options or {})}
        key = cache_key(model, messages, opts, schema, think)
        hit = self._cache.get(key)
        if hit:
            self._inc("cache_hits")
            return LLMResult(text=hit["text"], json=hit.get("json"), ok=True,
                             tokens_in=hit.get("tokens_in", 0), tokens_out=hit.get("tokens_out", 0),
                             seconds=hit.get("seconds", 0.0), key_index=hit.get("key_index"),
                             cached=True, retries=hit.get("retries", 0), fenced=hit.get("fenced", False),
                             truncations=hit.get("truncations", 0))
        msem = self._model_sem.get(model)
        if msem:
            with msem:
                res = self._call(model, messages, opts, schema, think, affinity)
        else:
            res = self._call(model, messages, opts, schema, think, affinity)
        rec = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "model": model, "digest": self.digests.get(model), "key_index": res.key_index,
               "ok": res.ok, "error": res.error, "retries": res.retries, "truncations": res.truncations,
               "tokens_in": res.tokens_in,
               "tokens_out": res.tokens_out, "seconds": round(res.seconds, 3), "cached": False}
        self._append(self._usage_path, rec)
        if res.ok:
            stored = {"key": key, "text": res.text, "json": res.json, "model": model,
                      "tokens_in": res.tokens_in, "tokens_out": res.tokens_out,
                      "seconds": res.seconds, "key_index": res.key_index,
                      "retries": res.retries, "fenced": res.fenced, "truncations": res.truncations,
                      "date": rec["date"]}
            self._cache[key] = stored
            self._append(self._cache_path, stored)
        else:
            self._inc("failed")
        return res

    def _event(self, kind: str, model: str, idx):
        """Rate-limit and server-error events (key index only, never the key): the real parallelism of the plan."""
        self._append(self._events_path, {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         "kind": kind, "model": model, "key_index": idx})

    def _inc(self, name: str, n: int = 1):
        with self._stats_lock:
            self.stats[name] += n

    def key_error_rate(self) -> float:
        bad = self.stats["http_429"] + self.stats["http_401"] + self.stats["http_403"]
        return bad / max(self.stats["calls"], 1)

    # ---- internals ----------------------------------------------------------
    def _redact(self, text: str) -> str:
        for k in self._keys:
            if k:
                text = text.replace(k, "[key]")
        return text

    def _pool(self, model):
        """Indices of the keys a model may use (None: all)."""
        if self._pools is None:
            return None
        return self._pools["paid" if model in self._paid_models else "free"]

    def _acquire(self, waited, pool=None, prefer=None):
        """Pick the next usable key (among ``pool`` when given). Returns (index, wait_seconds); (None, 0) if no
        valid key is left; (-1, 0) if waiting for a free key would exceed the per-call budget.
        ``waited`` is a one-element list accumulating the seconds spent waiting."""
        while True:
            with self._lock:
                alive = [i for i in range(len(self._keys)) if not self._dead[i] and (pool is None or i in pool)]
                if not alive:
                    return None, 0.0
                now = self._clock()
                free = [i for i in alive if self._cool_until[i] <= now]
                if not free:
                    wait = min(self._cool_until[i] for i in alive) - now
                    if waited[0] + max(wait, 0.001) > self.rate_limit_max_wait:
                        return -1, 0.0
                else:
                    wait = 0.0
                    n = len(self._keys)
                    # one turn per pool: with a shared turn the paid model would push the free ones back to the
                    # same key every time
                    if prefer is not None:
                        # the key of this affinity among the usable ones of the pool, else the next free one
                        start = alive[prefer % len(alive)]
                        idx = next(i for i in ((start + j) % n for j in range(n)) if i in free)
                    else:
                        rr = self._rr.get(id(pool), 0)
                        idx = next(i for i in ((rr + j) % n for j in range(n)) if i in free)
                        self._rr[id(pool)] = (idx + 1) % n
                    slot = max(now, self._next_free[idx])
                    self._next_free[idx] = slot + self.min_interval
                    return idx, max(slot - now, 0.0)
            wait = max(wait, 0.001)
            waited[0] += wait
            self._sleep(wait)

    def _backoff(self, n):
        return min(self.backoff_base * (2 ** max(n - 1, 0)), self.backoff_cap)

    def _call(self, model, messages, opts, schema, think=None, affinity=None) -> LLMResult:
        local = config.is_local(model)
        attempts = json_bad = failures = trunc = 0
        waited = [0.0]
        error = ""
        sent_messages, sent_opts = messages, opts
        while attempts < self.max_attempts:
            body = {"model": model, "messages": sent_messages, "stream": False,
                    "options": sent_opts}
            if schema is not None:
                body["format"] = schema
            if think is not None:
                body["think"] = think
            idx = None
            headers = {"Content-Type": "application/json"}
            if local:
                url = config.LOCAL_URL + "/api/chat"
            else:
                idx, wait = self._acquire(waited, self._pool(model), affinity)
                if idx is None:
                    return LLMResult(error="no_valid_key", retries=json_bad)
                if idx == -1:
                    self._inc("rate_limited")
                    return LLMResult(error="rate_limited", retries=json_bad)
                url = config.CLOUD_URL + "/api/chat"
                headers["Authorization"] = "Bearer " + self._keys[idx]
                if wait > 0:
                    self._sleep(wait)
            attempts += 1
            self._inc("calls")
            t0 = self._clock()
            sem = self._sem[idx] if idx is not None else None
            if sem:
                sem.acquire()
            try:
                status, payload = self._transport(url, headers, body, self.timeout)
            except Exception as e:  # noqa: BLE001 - any transport failure is retryable
                self._inc("network_error")
                error = self._redact(f"network_error: {type(e).__name__}: {e}")
                failures += 1
                self._sleep(self._backoff(failures))
                continue
            finally:
                if sem:
                    sem.release()
            seconds = self._clock() - t0

            if status == 200 and isinstance(payload, dict):
                self._limit_outcome(model, False)
                if idx is not None:
                    with self._lock:
                        self._strikes[idx] = 0
                content = (payload.get("message") or {}).get("content") or ""
                tin = int(payload.get("prompt_eval_count") or 0)
                tout = int(payload.get("eval_count") or 0)
                if payload.get("done_reason") == "length":
                    # cut by the output limit: not an invalid JSON; retry with a doubled limit
                    trunc += 1
                    self._inc("truncated")
                    self._inc(f"truncated:{model}")
                    cur = sent_opts.get("num_predict")
                    nxt = NUM_PREDICT_DEFAULT_START if not cur or cur < 0 else min(NUM_PREDICT_CAP, cur * 2)
                    if not self.grow_truncated or cur and cur >= NUM_PREDICT_CAP or nxt == cur:
                        self._inc("failed_truncated")
                        return LLMResult(error="truncated", tokens_in=tin, tokens_out=tout, seconds=seconds,
                                         key_index=idx, retries=json_bad, truncations=trunc)
                    sent_opts = {**sent_opts, "num_predict": nxt}
                    attempts -= 1  # does not consume attempts
                    continue
                if schema is None:
                    return LLMResult(text=content, ok=True, tokens_in=tin, tokens_out=tout,
                                     seconds=seconds, key_index=idx, truncations=trunc)
                try:
                    parsed, fenced = parse_json_answer(content)
                    if fenced:
                        self._inc("fenced")
                        self._inc(f"fenced:{model}")
                    return LLMResult(text=content, json=parsed, ok=True, tokens_in=tin,
                                     tokens_out=tout, seconds=seconds, key_index=idx,
                                     retries=json_bad, fenced=fenced, truncations=trunc)
                except ValueError:
                    json_bad += 1
                    error = "invalid_json"
                    if json_bad > self.invalid_json_retries:
                        self._inc("invalid_json")
                        return LLMResult(error="invalid_json", tokens_in=tin, tokens_out=tout,
                                         seconds=seconds, key_index=idx, retries=json_bad - 1)
                    # a different request: the wrong answer plus a correction, new seed
                    sent_messages = list(messages) + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": CORRECTION}]
                    sent_opts = {**opts, "seed": opts.get("seed", config.SEED) + json_bad}
                    continue
            if status == 429 and idx is not None:
                self._inc("http_429")
                quota = quota_kind(payload)
                if quota == "monthly":
                    # the account's monthly free usage is over: the key is useless until next month
                    self._inc("quota_monthly")
                    self._event("quota_monthly", model, idx)
                    with self._lock:
                        self._dead[idx] = True
                elif quota == "window":
                    # free usage window (5-hour or weekly cap) reached: pause the key, do not shrink the limit
                    # of calls in flight, which is about concurrency, not quota
                    self._inc("quota_window")
                    self._event("quota_window", model, idx)
                    with self._lock:
                        self._cool_until[idx] = self._clock() + QUOTA_COOLDOWN
                else:
                    self._event("http_429", model, idx)
                    self._limit_outcome(model, True)
                    with self._lock:
                        self._strikes[idx] += 1
                        self._cool_until[idx] = self._clock() + self._backoff(self._strikes[idx])
                error = "http_429"
                attempts -= 1  # rate limits do not consume attempts (bounded by the wait budget)
                continue
            if status in (401, 403) and idx is not None:
                self._inc(f"http_{status}")
                self._event(f"http_{status}", model, idx)
                with self._lock:
                    self._dead[idx] = True
                error = f"http_{status}"
                attempts -= 1
                continue
            if status >= 500 or status in (0, 408, 429):
                self._inc("http_5xx" if status >= 500 else f"http_{status}")
                self._event("http_5xx" if status >= 500 else f"http_{status}", model, idx)
                error = f"http_{status}"
                failures += 1
                self._sleep(self._backoff(failures))
                continue
            self._inc(f"http_{status}")  # other 4xx (model not found, bad request, 402 paid model): permanent
            if status == 402:
                self._event("http_402", model, idx)
            return LLMResult(error=f"http_{status}", seconds=seconds, key_index=idx)
        return LLMResult(error=error or "max_attempts", key_index=None, retries=json_bad)


def fetch_digests(models, keys=None, timeout=60) -> dict:
    """Best-effort model digests (local and cloud /api/tags) for the usage log."""
    import requests
    out = {}
    for base, hdr in ((config.LOCAL_URL, {}),
                      (config.CLOUD_URL, {"Authorization": "Bearer " + keys[0]} if keys else None)):
        if hdr is None:
            continue
        try:
            r = requests.get(base + "/api/tags", headers=hdr, timeout=timeout)
            for m in r.json().get("models", []):
                name = m.get("name") or m.get("model")
                if name in models or any(name.startswith(x + ":") for x in models):
                    out.setdefault(name, m.get("digest"))
        except Exception:  # noqa: BLE001
            continue
    return out
