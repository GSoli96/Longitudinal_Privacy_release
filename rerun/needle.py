"""Needle-in-a-haystack check of a model's usable context (plan T3).

Three invented verification codes are placed at the start, in the middle and at the end of a long
text of the target length; the model must return the three. A missing code, or a prompt token
count far below the expected length, means the prompt was cut (the original defect: a silent
truncation by the server). ``num_ctx`` is always passed explicitly. The result holds numbers and
booleans only, never the text.
"""
from __future__ import annotations

import random
import re

CODE_RE = re.compile(r"\b[A-Z]{5}-\d{4}-[A-Z]{6}\b")
_SYL = "BCDFGLMNPRSTV"
_VOW = "AEIOU"
NOTE = "Nota di servizio: il codice di verifica numero {i} e' {code}."
QUESTION = ("Nel testo sopra ci sono tre note di servizio, ciascuna con un codice di verifica "
            "(numero 1, 2 e 3). Restituisci i tre codici esattamente come scritti, nell'ordine "
            "in cui compaiono, in JSON: {{\"codes\": [\"...\", \"...\", \"...\"]}}.")
SCHEMA = {"type": "object", "properties": {"codes": {"type": "array", "items": {"type": "string"}}},
          "required": ["codes"]}
TRUNCATION_RATIO = 0.8  # prompt tokens below 80 % of the expected length: the prompt was cut


def _word(rnd, n):
    return "".join(rnd.choice(_SYL if i % 2 == 0 else _VOW) for i in range(n))


def make_codes(seed: int) -> list[str]:
    rnd = random.Random(seed)
    out = []
    while len(out) < 3:
        c = f"{_word(rnd, 5)}-{rnd.randint(1000, 9999)}-{_word(rnd, 6)}"
        if CODE_RE.fullmatch(c) and c not in out:
            out.append(c)
    return out


def build_haystack(texts: list[str], target_chars: int, seed: int = 0) -> tuple[str, list[str]]:
    """A text of about ``target_chars`` made of ``texts`` (cycled) with the three notes inserted."""
    codes = make_codes(seed)
    pieces, size, i = [], 0, 0
    while size < target_chars:
        t = texts[i % len(texts)]
        pieces.append(t)
        size += len(t) + 2
        i += 1
    body = "\n\n".join(pieces)[:target_chars]
    cut = lambda frac: body.rfind("\n", 0, int(len(body) * frac)) + 1  # a line start near the fraction
    a, b = cut(0.01), cut(0.5)
    notes = [NOTE.format(i=k + 1, code=c) for k, c in enumerate(codes)]
    text = body[:a] + notes[0] + "\n" + body[a:b] + notes[1] + "\n" + body[b:] + "\n" + notes[2] + "\n"
    return text, codes


def run_needle(client, model: str, texts: list[str], *, target_tokens: int, chars_per_token: float,
               num_ctx: int, seed: int = 0) -> dict:
    hay, codes = build_haystack(texts, int(target_tokens * chars_per_token), seed)
    expected_tokens = len(hay) / chars_per_token
    r = client.chat(model, [{"role": "user", "content": hay + "\n\n" + QUESTION}],
                    {"num_ctx": num_ctx, "num_predict": 4096})
    res = {"model": model, "target_tokens": target_tokens, "num_ctx": num_ctx, "chars": len(hay),
           "ok": r.ok, "error": r.error, "prompt_tokens": r.tokens_in, "seconds": round(r.seconds, 1),
           "found": [False, False, False], "all_found": False, "truncated": None}
    if not r.ok:
        return res
    got = CODE_RE.findall(r.text or "")  # read from the raw text: a JSON in a code fence is not a truncation
    res["found"] = [c in got for c in codes]
    res["all_found"] = all(res["found"])
    res["truncated"] = bool(r.tokens_in and r.tokens_in < TRUNCATION_RATIO * expected_tokens)
    return res


# ---- judge answer format --------------------------------------------------------------------------
FIELDS = ("plausible", "coherent", "realistic", "explicit")
_ATTRS = ["ha ricoperto un incarico dirigenziale presso una direzione generale",
          "ha conseguito una laurea magistrale in giurisprudenza",
          "coordina un ufficio con piu di dieci dipendenti",
          "ha partecipato a un corso di formazione sulla trasparenza amministrativa",
          "e stato nominato componente di una commissione di valutazione",
          "ha maturato esperienza nella gestione di appalti pubblici",
          "risiede nella provincia in cui ha sede l ufficio",
          "ha una retribuzione annua superiore alla media del livello",
          "ha svolto un incarico presso un altro ente nel periodo precedente",
          "parla correntemente una lingua straniera"]
JUDGE_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, **{f: {"type": "string", "enum": ["yes", "no"]} for f in FIELDS}},
    "required": ["id", *FIELDS]}}}, "required": ["verdicts"]}
JUDGE_INSTRUCTION = ("Sopra c e il testo dei documenti di una persona. Qui sotto ci sono {n} affermazioni numerate. "
                     "Per ciascuna rispondi con yes o no per: plausible, coherent, realistic, explicit. "
                     "Rispondi solo in JSON: {{\"verdicts\": [{{\"id\": 1, \"plausible\": \"yes\", \"coherent\": \"yes\", "
                     "\"realistic\": \"no\", \"explicit\": \"no\"}}, ...]}} con un elemento per affermazione.")


def validate_verdicts(js, n: int) -> list[str]:
    """Names of the format problems of a judge answer (empty list = valid)."""
    if not isinstance(js, dict) or "verdicts" not in js:
        return ["missing_key"]
    v = js["verdicts"]
    if not isinstance(v, list):
        return ["not_a_list"]
    problems = []
    if len(v) != n:
        problems.append("wrong_count")
    ids = [x.get("id") if isinstance(x, dict) else None for x in v]
    if sorted(i for i in ids if isinstance(i, int)) != list(range(1, len(v) + 1)) or len(set(ids)) != len(ids):
        problems.append("bad_ids")
    if any(not isinstance(x, dict) or any(x.get(f) not in ("yes", "no") for f in FIELDS) for x in v):
        problems.append("bad_value")
    return problems


def run_judge_format(client, model: str, text: str, *, n_attrs: int = 40, num_ctx: int, raw_probe=None) -> dict:
    """One judge call with ``n_attrs`` invented attributes: is the JSON answer valid?

    ``raw_probe(prompt)`` (optional) returns the raw answer of a failed call, used only to tell a JSON
    wrapped in a code fence from a really broken answer; it is never stored."""
    attrs = "\n".join(f"{i}. L interessato {_ATTRS[(i - 1) % len(_ATTRS)]} ({i})." for i in range(1, n_attrs + 1))
    prompt = f"{text}\n\nAffermazioni:\n{attrs}\n\n{JUDGE_INSTRUCTION.format(n=n_attrs)}"
    r = client.chat(model, [{"role": "user", "content": prompt}],
                    {"num_ctx": num_ctx, "num_predict": 16384}, JUDGE_SCHEMA)
    res = {"model": model, "ok": r.ok, "error": r.error, "valid": False, "n_returned": 0, "problems": [],
           "prompt_tokens": r.tokens_in, "completion_tokens": r.tokens_out, "seconds": round(r.seconds, 1),
           "fenced_but_valid_inside": None}
    if not r.ok:
        if raw_probe is not None:
            import json as _json
            raw = (raw_probe(prompt) or "").strip()
            m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
            if m:
                try:
                    res["fenced_but_valid_inside"] = not validate_verdicts(_json.loads(m.group(1)), n_attrs)
                except ValueError:
                    res["fenced_but_valid_inside"] = False
        return res
    js = r.json
    res["problems"] = validate_verdicts(js, n_attrs)
    res["valid"] = not res["problems"]
    if isinstance(js, dict) and isinstance(js.get("verdicts"), list):
        res["n_returned"] = len(js["verdicts"])
    return res
