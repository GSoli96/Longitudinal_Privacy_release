"""Local judge panel of Cases A, B, D, H (decisions D91/D92, 2026-10-27), for runs that must not touch the cloud.

Port of ``local_panel.py`` of the laboratory kit, without the fallback to cloud votes (a new run has none). Call
``setup()`` in the process, before ``run_case.main``; ``teardown()`` undoes it (tests).

Names with the "-local" suffix are served by the local Ollama (``config.is_local``); they are aliases made with
``ollama cp``. A different name means a different cache key, so local votes never mix with cloud votes.

What ``setup`` fixes (as in the runs of A, B, D, H):
* panel ``(gemma4:31b-local, qwen3.5:9b, nemotron-3-nano:30b-local)``: the first two vote on every attribute, the third
  breaks ties (Case C runs it on every attribute, as a full-panel case);
* one attribute per call, one call at a time per model, judges without reasoning (``think=False``);
* context: 262,144 tokens as the cloud judges, but 65,536 when the prompt fits (98.5% of them); qwen3.5:9b keeps the
  full context (the 32,768 variant was not identical to it on the pilot);
* the third judge answers with at most 8,192 tokens and the client does not double the cap on truncation;
* a prompt that would need the big context is not sent to a local judge (six attempts in a row timed out after 600 s in
  the first calibration, about an hour per call): the call counts as failed (``too_long_for_local``) unless its vote is
  already in the cache. None of the Case C judge prompts is that long.
"""
from __future__ import annotations

from . import config, llm

GEMMA = "gemma4:31b-local"
THIRD = "nemotron-3-nano:30b-local"
QWEN = config.LOCAL_JUDGE
# digests of the local aliases used in the runs of 2026-10-27/28 (shown by ``ollama list``): a check, not a constraint
DIGESTS = {GEMMA: "17ba34c06c80", THIRD: "7487f38ecf24"}

BIG_CTX = 262144
SMALL_CTX = 65536
THIRD_NUM_PREDICT = 8192

_LOCAL: set = set()
_STATE: dict = {"orig_chat": None, "saved": None}


def setup(third: str = THIRD, third_think=None, cap_third: bool = True):
    """Put the local panel in ``config`` and install the guard. Returns the panel."""
    if _STATE["saved"] is None:
        _STATE["saved"] = {"NUM_CTX": dict(config.NUM_CTX), "MODEL_CTX": dict(config.MODEL_CTX),
                           "JUDGE_BATCH_SIZE": dict(config.JUDGE_BATCH_SIZE),
                           "JUDGE_CTX_SMALL": dict(config.JUDGE_CTX_SMALL), "JUDGE_THINK": dict(config.JUDGE_THINK),
                           "MODEL_CONCURRENCY": dict(config.MODEL_CONCURRENCY), "PANEL": config.PANEL,
                           "NUM_PREDICT_CAP": llm.NUM_PREDICT_CAP}
    _LOCAL.clear()
    for m in (GEMMA, third):
        config.NUM_CTX[m] = BIG_CTX
        config.MODEL_CTX[m] = BIG_CTX
        config.JUDGE_BATCH_SIZE[m] = 1
        config.JUDGE_CTX_SMALL[m] = SMALL_CTX
        config.MODEL_CONCURRENCY[m] = 1
        _LOCAL.add(m)
    config.JUDGE_THINK[GEMMA] = False
    if third == THIRD and third_think is None:
        third_think = False
    if third_think is not None:
        config.JUDGE_THINK[third] = third_think
    config.MODEL_CONCURRENCY[QWEN] = 1
    config.PANEL = (GEMMA, QWEN, third)
    if cap_third:
        llm.NUM_PREDICT_CAP = THIRD_NUM_PREDICT
    _guard(third if cap_third else None)
    return config.PANEL


def _guard(capped_model: str | None):
    if _STATE["orig_chat"] is not None:
        return
    orig = llm.LLMClient.chat
    _STATE["orig_chat"] = orig

    def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
        if model in _LOCAL:
            if capped_model and model == capped_model:
                options = {**(options or {}), "num_predict": THIRD_NUM_PREDICT}
            if (options or {}).get("num_ctx", 0) > SMALL_CTX:
                opts = {**config.DEFAULT_OPTIONS, **(options or {})}
                if llm.cache_key(model, messages, opts, schema, think) not in self._cache:
                    self._inc("too_long_for_local")
                    self._inc("failed")
                    return llm.LLMResult(error="too_long_for_local")
        return orig(self, model, messages, options, schema, think, affinity)

    llm.LLMClient.chat = chat


def teardown():
    """Restore ``config`` and the client as before ``setup`` (for tests; a run never needs it)."""
    if _STATE["orig_chat"] is not None:
        llm.LLMClient.chat = _STATE["orig_chat"]
        _STATE["orig_chat"] = None
    saved = _STATE["saved"]
    if saved is not None:
        for name in ("NUM_CTX", "MODEL_CTX", "JUDGE_BATCH_SIZE", "JUDGE_CTX_SMALL", "JUDGE_THINK", "MODEL_CONCURRENCY"):
            target = getattr(config, name)
            target.clear()
            target.update(saved[name])
        config.PANEL = saved["PANEL"]
        llm.NUM_PREDICT_CAP = saved["NUM_PREDICT_CAP"]
        _STATE["saved"] = None
    _LOCAL.clear()
