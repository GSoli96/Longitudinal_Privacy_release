"""Inference of attributes not stated in the documents (plan T4, spec section 4.3).

The instruction is the original one of the paper (identical in every case of the original code);
only its last sentence changes: the original asked for "a list of inferred attributes (no need for
valid JSON)", the rerun asks for a JSON object with a fixed schema, one item per attribute with the
attribute, a short category hint and the evidence (the passage of the text that supports it).
Special-category data (GDPR art. 9) are NOT requested; categories are assigned afterwards by a
separate classification pass (spec 10b.4).

* A text that does not fit the usable context is read in overlapping windows (``steps.windows``);
  the attributes of the windows are merged by normalised text (duplicates collapse, the first
  window and all windows are recorded) and the step is flagged ``chunked``.
* The answer is never parsed line by line. A JSON that does not match the schema is retried with a
  correction (the wrong answer plus a message, a new seed); after ``max_structure_retries`` the
  window is a failure and none of its text becomes an attribute. A failed window makes the step
  ``ok=False``; the attributes of the other windows are kept and the step is flagged ``partial``.
* ``evidence_in_text`` records whether the cited evidence occurs in the window text after normalisation
  (``evidence_match`` exact, or fuzzy with similarity >= ``config.EVIDENCE_FUZZY_RATIO`` on the best
  aligned window of the same length; ``evidence_ratio`` the similarity): diagnostic only, the attribute
  is kept either way and judging decides.
* ``num_ctx`` is fixed per model (``config.NUM_CTX``), never chosen per step.
* A step identical to the previous one is not run again: ``infer_person`` carries the result forward.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from . import config, steps

NUM_PREDICT = config.ATTACKER_OUTPUT_RESERVE  # reasoning + JSON; the client doubles it if cut
MAX_STRUCTURE_RETRIES = 2

INSTRUCTION = """
You are an analyst.

Given the following administrative records of a public employee,
infer ONLY information that is NOT explicitly stated but can be deduced.

Return a JSON object {{"attributes": [...]}} with one item per inferred attribute. Each item has:
"attribute" (the inferred statement, one fact), "category" (a short free-form label) and
"evidence" (the passage of the text that supports the inference, copied from the text).
If nothing can be inferred, return {{"attributes": []}}. Return only the JSON object.

TEXT:
----------------
{document}
----------------
"""

CORRECTION = ('The JSON does not match the required structure ({problem}). Return only a JSON object '
              '{"attributes": [{"attribute": "...", "category": "...", "evidence": "..."}]} where every item '
              'has a non-empty "attribute" and a non-empty "evidence" text, and nothing else.')

SCHEMA = {"type": "object",
          "properties": {"attributes": {"type": "array", "items": {
              "type": "object",
              "properties": {"attribute": {"type": "string"}, "category": {"type": "string"},
                             "evidence": {"type": "string"}},
              "required": ["attribute", "evidence"]}}},
          "required": ["attributes"]}


@dataclass
class Attribute:
    text: str
    category_hint: str
    evidence: str
    window: int                  # first window in which the attribute appeared
    windows: tuple = ()          # all windows
    evidence_in_text: bool = False   # evidence found in the window text (exact or fuzzy >= threshold)
    evidence_match: str = "none"     # "exact" | "fuzzy" | "none"
    evidence_ratio: float = 0.0      # 1.0 for exact; best similarity otherwise


@dataclass
class StepResult:
    step: object
    attributes: list = field(default_factory=list)
    ok: bool = True
    error: str = ""
    windows: int = 1
    chunked: bool = False
    partial: bool = False
    failed_windows: list = field(default_factory=list)
    structure_retries: int = 0
    empty_retries: int = 0
    evidence_exact: int = 0
    evidence_fuzzy: int = 0
    evidence_not_found: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    carried: bool = False


INSTRUCTION_NO_EVIDENCE = """
You are an analyst.

Given the following administrative records of a public employee,
infer ONLY information that is NOT explicitly stated but can be deduced.

Return a JSON object {{"attributes": [...]}} with one item per inferred attribute. Each item has:
"attribute" (the inferred statement, one fact) and "category" (a short free-form label).
If nothing can be inferred, return {{"attributes": []}}. Return only the JSON object.

TEXT:
----------------
{document}
----------------
"""

CORRECTION_NO_EVIDENCE = ('The JSON does not match the required structure ({problem}). Return only a JSON object '
                          '{"attributes": [{"attribute": "...", "category": "..."}]} where every item '
                          'has a non-empty "attribute" text, and nothing else.')

SCHEMA_NO_EVIDENCE = {"type": "object",
                      "properties": {"attributes": {"type": "array", "items": {
                          "type": "object",
                          "properties": {"attribute": {"type": "string"}, "category": {"type": "string"}},
                          "required": ["attribute"]}}},
                      "required": ["attributes"]}

VARIANTS = ("evidence", "no_evidence")   # "no_evidence": same instruction and schema without the evidence field
DEFAULT_VARIANT = "no_evidence"          # main run (author decision after the pilot); "evidence" is the sensitivity analysis


def schema_for(variant: str = "evidence") -> dict:
    return SCHEMA if variant == "evidence" else SCHEMA_NO_EVIDENCE


def build_prompt(document: str, variant: str = "evidence") -> str:
    return (INSTRUCTION if variant == "evidence" else INSTRUCTION_NO_EVIDENCE).format(document=document)


def _norm(t: str) -> str:
    return re.sub(r"[\W_]+", " ", t.casefold()).strip()


_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "«": '"', "»": '"'})


def _squash(t: str) -> str:
    return re.sub(r"\s+", " ", t.translate(_QUOTES)).strip().casefold()


def anchor_evidence(evidence: str, text_squashed: str) -> tuple[str, float]:
    """How the cited evidence is anchored in the (already squashed) text: ("exact", 1.0) if it occurs
    after normalisation; ("fuzzy", ratio) if the best aligned window of the same length has a similarity
    of at least ``config.EVIDENCE_FUZZY_RATIO``; otherwise ("none", best ratio). Nothing is discarded."""
    ev = _squash(evidence)
    if not ev:
        return "none", 0.0
    if ev in text_squashed:
        return "exact", 1.0
    from rapidfuzz import fuzz
    if len(ev) > len(text_squashed):
        ratio = fuzz.ratio(ev, text_squashed) / 100.0
    else:
        ratio = fuzz.partial_ratio(ev, text_squashed) / 100.0
    return ("fuzzy" if ratio >= config.EVIDENCE_FUZZY_RATIO else "none"), round(ratio, 4)


def _validate(js, require_evidence: bool = True) -> tuple[list, str]:
    """(items, problem): items are (attribute, category, evidence) tuples; problem is '' if valid."""
    if not isinstance(js, dict) or "attributes" not in js:
        return [], 'missing key "attributes"'
    lst = js["attributes"]
    if not isinstance(lst, list):
        return [], '"attributes" is not a list'
    out = []
    for i, it in enumerate(lst):
        if not isinstance(it, dict):
            return [], f"item {i + 1} is not an object"
        a, e, c = it.get("attribute"), it.get("evidence"), it.get("category", "")
        if not isinstance(a, str) or not a.strip():
            return [], f'item {i + 1} has no "attribute" text'
        if require_evidence and (not isinstance(e, str) or not e.strip()):
            return [], f'item {i + 1} has no "evidence" text'
        out.append((a.strip(), c.strip() if isinstance(c, str) else "",
                    e.strip() if isinstance(e, str) else ""))
    return out, ""


def _run_window(client, model, text, num_ctx, max_structure_retries, variant="evidence"):
    """-> (items, error, structure_retries, empty_retries, tokens_in, tokens_out, seconds)."""
    prompt = build_prompt(text, variant)
    schema = schema_for(variant)
    correction = CORRECTION if variant == "evidence" else CORRECTION_NO_EVIDENCE
    messages = [{"role": "user", "content": prompt}]
    retries = empty_retries = tries = tin = tout = 0
    secs = 0.0
    structure_left = max_structure_retries
    while True:
        opts = {"num_ctx": num_ctx, "num_predict": NUM_PREDICT, "seed": config.SEED + config.ATTACKER_SEED_OFFSET + tries,
                **config.ATTACKER_OPTIONS}
        r = client.chat(model, messages, opts, schema, think=config.attacker_think_for(model))
        tin += r.tokens_in
        tout += r.tokens_out
        secs += r.seconds
        if not r.ok:
            return [], r.error or "failed", retries, empty_retries, tin, tout, secs
        items, problem = _validate(r.json, variant == "evidence")
        if not problem:
            if not items and empty_retries < config.ATTACKER_EMPTY_RETRIES:
                empty_retries += 1  # same request, new seed: an empty list is usually spurious
                tries += 1
                continue
            return items, "", retries, empty_retries, tin, tout, secs
        if structure_left <= 0:
            return [], "invalid_structure", retries, empty_retries, tin, tout, secs
        structure_left -= 1
        retries += 1
        tries += 1
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": r.text},
                    {"role": "user", "content": correction.replace("{problem}", problem)}]


def split_step(text: str, usable_tokens: int, model: str, chars_per_token: float | None = None) -> list[str]:
    """The windows of a step text for the attacker (one if it fits). The judges read the same windows when the
    step is chunked, so that the originating window of an attribute means the same text for everyone."""
    cpt = chars_per_token or config.TOKEN_CHARS.get(model, config.TOKEN_CHARS_DEFAULT)
    if steps.estimate_tokens(text, cpt) <= usable_tokens:
        return [text]
    return steps.windows(text, usable_tokens, cpt)


def infer_step(client, step, model: str, *, usable_tokens: int, chars_per_token: float | None = None,
               num_ctx: int | None = None, max_structure_retries: int = MAX_STRUCTURE_RETRIES,
               variant: str = DEFAULT_VARIANT) -> StepResult:
    cpt = chars_per_token or config.TOKEN_CHARS.get(model, config.TOKEN_CHARS_DEFAULT)
    wins = split_step(step.text, usable_tokens, model, cpt)
    res = StepResult(step=step, windows=len(wins), chunked=len(wins) > 1)
    merged: dict[str, Attribute] = {}
    for w, text in enumerate(wins):
        ctx = num_ctx or config.num_ctx_for(model)  # fixed per model: never chosen per step
        items, error, retries, empties, tin, tout, secs = _run_window(client, model, text, ctx, max_structure_retries,
                                                                      variant)
        res.structure_retries += retries
        res.empty_retries += empties
        res.tokens_in += tin
        res.tokens_out += tout
        res.seconds += secs
        if error:
            res.failed_windows.append(w)
            res.error = res.error or error
            continue
        squashed = _squash(text)
        for a, c, e in items:
            key = _norm(a)
            if key in merged:
                prev = merged[key]
                if w not in prev.windows:
                    prev.windows = prev.windows + (w,)
                continue
            match, ratio = anchor_evidence(e, squashed) if variant == "evidence" else ("n/a", 0.0)
            merged[key] = Attribute(a, c, e, window=w, windows=(w,), evidence_in_text=match in ("exact", "fuzzy"),
                                    evidence_match=match, evidence_ratio=ratio)
    res.attributes = list(merged.values())
    res.evidence_exact = sum(1 for a in res.attributes if a.evidence_match == "exact")
    res.evidence_fuzzy = sum(1 for a in res.attributes if a.evidence_match == "fuzzy")
    res.evidence_not_found = sum(1 for a in res.attributes if a.evidence_match == "none")
    res.ok = not res.failed_windows
    res.partial = (not res.ok) and bool(res.attributes)
    return res


def infer_person(client, person, model: str, *, usable_tokens: int, **kw) -> list[StepResult]:
    """Results for the steps of a person; a step identical to the previous one is carried forward."""
    out: list[StepResult] = []
    for st in steps.build_steps(person):
        if st.identical_to_prev and out:
            out.append(replace(out[-1], step=st, carried=True))
        else:
            out.append(infer_step(client, st, model, usable_tokens=usable_tokens, **kw))
    return out
