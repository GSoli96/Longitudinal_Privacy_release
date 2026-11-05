"""Local LLM detector of the RQ4 stage (plan ``docs/plans/rq34.md`` §4, family C). Zero-shot and few-shot.

The prompts are those of the paper (``src.detection.ZERO_SHOT`` / ``FEW_SHOT``, Italian, answer 0 or 1). Differences
from ``src.detection.parse_binary_label``, which returns 0 when it finds no digit:

* ``parse_label`` accepts an answer that is a single 0 or 1 (surrounding spaces, quotes or a final full stop
  allowed) and returns ``None`` for anything else; the record is then counted as ``unparseable`` and left out of the
  metrics, never turned into a label.
* a failed call is counted as ``failed`` and left out in the same way. ``summary`` reports both counts, so the
  coverage of every detector is visible next to its metrics.

The answer is a label, not a probability, so ``score`` equals ``pred`` (AUC is then the balanced accuracy and the
AUPRC a single point; the report says so). The few-shot examples come from the training records only, taken in
the order of their ids (``pick_examples``), 300 characters each as in the paper. The model is called with
``think=False`` and a short output limit; the caller declares it as a detector (a judge may be reused, and then the
report says so). Records are dicts with ``id``, ``y`` (1 = phishing), ``subject``/``body`` or ``text``, and the
grouping keys ``person``, ``case`` and ``generator``."""
from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src import detection

MODES = ("zero", "few")
EXAMPLE_CHARS = 300
NUM_PREDICT = 16
_LABEL = re.compile(r"""^\s*["'`(\[]*\s*([01])\s*["'`)\]]*\s*[.!]?\s*$""")


def message_text(rec: dict) -> str:
    """The text the detector sees: subject and body, or ``text`` when the record has no body."""
    if rec.get("body") is not None or rec.get("subject") is not None:
        return f"Oggetto: {rec.get('subject') or ''}\n\n{rec.get('body') or ''}".strip()
    return rec.get("text") or ""


def parse_label(text) -> int | None:
    """0 or 1 when the whole answer is that digit (see the module docstring), else ``None``."""
    m = _LABEL.match(text or "")
    return int(m.group(1)) if m else None


def pick_examples(train: list[dict], n_per_class: int = 3) -> list[dict]:
    """The first ``n_per_class`` positives and negatives of the training records in the order of their ids."""
    ordered = sorted(train, key=lambda r: str(r.get("id", "")))
    pos = [r for r in ordered if r["y"] == 1][:n_per_class]
    neg = [r for r in ordered if r["y"] == 0][:n_per_class]
    return pos + neg


def build_prompt(rec: dict, mode: str, examples: list[dict] | None = None) -> str:
    if mode == "zero":
        return detection.ZERO_SHOT.format(email=message_text(rec))
    if mode == "few":
        if not examples:
            raise ValueError("few-shot mode needs examples")
        shots = "\n\n".join(f"Email: {message_text(e)[:EXAMPLE_CHARS]}...\nClassificazione: {int(e['y'])}"
                            for e in examples)
        return detection.FEW_SHOT.format(examples=shots, email=message_text(rec))
    raise ValueError(f"mode must be one of {MODES}")


def classify_one(client, model: str, rec: dict, mode: str, examples=None, *, seed: int = 42,
                 options_extra: dict | None = None) -> dict:
    """``{"id", "pred", "error"}``: ``pred`` is 0, 1 or ``None`` (then ``error`` says why)."""
    prompt = build_prompt(rec, mode, examples)
    res = client.chat(model, [{"role": "user", "content": prompt}],
                      options={"temperature": 0.0, "seed": seed, "num_predict": NUM_PREDICT,
                               **(options_extra or {})}, think=False)
    if not res.ok:
        return {"id": rec.get("id"), "pred": None, "error": "call_failed:" + (res.error or "unknown")}
    pred = parse_label(res.text)
    if pred is None:
        return {"id": rec.get("id"), "pred": None, "error": "unparseable"}
    return {"id": rec.get("id"), "pred": pred, "error": ""}


def evaluate(client, model: str, test: list[dict], mode: str, *, train: list[dict] | None = None,
             n_per_class: int = 3, workers: int = 2, seed: int = 42, options_extra: dict | None = None,
             examples: list[dict] | None = None) -> dict:
    """``{"records": [scored test record, ...], "summary": {...}}`` for ``detect_metrics``. Only records with a
    valid answer are scored; the others are counted in the summary by reason."""
    if mode == "few" and examples is None:
        examples = pick_examples(train or [], n_per_class)
    if mode != "few":
        examples = None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        answers = list(pool.map(lambda r: classify_one(client, model, r, mode, examples, seed=seed,
                                                                       options_extra=options_extra), test))
    scored, errors = [], Counter()
    for r, a in zip(test, answers):
        if a["pred"] is None:
            errors[a["error"]] += 1
            continue
        scored.append({"y": r["y"], "score": float(a["pred"]), "pred": a["pred"], "person": r.get("person"),
                       "case": r.get("case"), "generator": r.get("generator"), "id": r.get("id")})
    n = len(test)
    summary = {"model": model, "mode": mode, "n_test": n, "n_scored": len(scored), "n_excluded": n - len(scored),
               "errors": dict(errors), "coverage": (len(scored) / n) if n else float("nan"),
               "n_examples": len(examples or [])}
    return {"records": scored, "summary": summary}
