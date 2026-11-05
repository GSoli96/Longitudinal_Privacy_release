"""Text artefact checks of the RQ4 stage (plan ``docs/plans/rq34.md`` §4). Pure functions, no I/O.

The question: does a detector separate generated phishing from generated legitimate mail because of the content or
because of surface cues the generator leaves (the name of the organisation, fixed urgency formulae, the signature
block, template placeholders)? ``clean_text`` removes those cues; the caller runs the detectors again on the
cleaned texts and compares. The cues are removed in this order, each counted apart:

* ``placeholder``: ``[...]``, ``{{...}}``, ``{...}``, ``<...>`` and runs of underscores or dots used as blanks;
* ``organisation``: the given names and domains, whole words only and case-insensitive (so the placeholder "ente" does
  not cut words such as "presente");
* ``urgency``: the fixed formulae of ``URGENCY_PATTERNS``;
* ``signature``: from the last closing formula ("Cordiali saluti", "Distinti saluti", "In attesa di un riscontro",
  "Grazie", "Cordialmente", "Buon lavoro") to the end of the text.

``clean_record`` applies it to ``subject`` and ``body`` of a record and keeps the counts; the record is copied, the
input is never changed. A text that becomes empty is kept empty (the caller decides), never replaced."""
from __future__ import annotations

import re
from collections import Counter

CATEGORIES = ("placeholder", "organisation", "urgency", "signature")

_PLACEHOLDER = re.compile(r"\[[^\]\n]{0,80}\]|\{\{[^}\n]{0,80}\}\}|\{[^}\n]{0,80}\}|<[^>\n]{0,80}>|_{3,}|\.{4,}")
URGENCY_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\burgent\w*", r"\bimmediat\w*", r"\bsubito\b", r"\bal più presto\b", r"\bil prima possibile\b",
    r"\bsenza (?:ulteriore )?ritardo\b", r"\bentro (?:oggi|domani|la giornata|la fine della giornata|\d+ ore)\b",
    r"\bscadenz\w*", r"\bASAP\b", r"\bdeadline\b", r"\bcon la massima priorità\b", r"\btempestiv\w*"))
_CLOSING = re.compile(r"(?:^|\n)\s*(?:cordiali saluti|distinti saluti|cordialmente|in attesa di un (?:suo )?riscontro|"
                      r"buon lavoro|grazie(?: per la collaborazione| in anticipo)?|saluti)\b", re.I)


def _count_sub(pattern: re.Pattern, text: str) -> tuple[str, int]:
    return pattern.subn("", text)


def _organisation_pattern(names) -> re.Pattern | None:
    names = [n for n in names if n and n.strip()]
    if not names:
        return None
    alt = "|".join(re.escape(n.strip()) for n in sorted(set(names), key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.I)


def clean_text(text: str, *, organisation=(), domains=()) -> tuple[str, Counter]:
    """``(cleaned text, Counter of removals per category)``."""
    counts: Counter = Counter()
    text = text or ""
    text, n = _count_sub(_PLACEHOLDER, text)
    counts["placeholder"] += n
    org = _organisation_pattern([*organisation, *domains])
    if org is not None:
        text, n = _count_sub(org, text)
        counts["organisation"] += n
    for pat in URGENCY_PATTERNS:
        text, n = _count_sub(pat, text)
        counts["urgency"] += n
    last = None
    for m in _CLOSING.finditer(text):
        last = m
    if last is not None:
        text = text[:last.start()]
        counts["signature"] += 1
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, counts


def clean_record(rec: dict, *, organisation=(), domains=()) -> dict:
    """A copy of ``rec`` with ``subject`` and ``body`` cleaned and ``cleaned_counts`` added."""
    out = dict(rec)
    total: Counter = Counter()
    for field in ("subject", "body", "text"):
        if isinstance(rec.get(field), str):
            out[field], c = clean_text(rec[field], organisation=organisation, domains=domains)
            total.update(c)
    out["cleaned_counts"] = {k: total.get(k, 0) for k in CATEGORIES}
    return out


def clean_records(records, *, organisation=(), domains=()) -> tuple[list[dict], dict]:
    """The cleaned records and a summary: removals per category and per class (``y`` or ``label``), and the number
    of records that became empty."""
    cleaned = [clean_record(r, organisation=organisation, domains=domains) for r in records]
    by_class: dict = {}
    empty = 0
    for orig, new in zip(records, cleaned):
        y = orig.get("y", orig.get("label"))
        d = by_class.setdefault(str(y), Counter())
        d.update(new["cleaned_counts"])
        d["n"] += 1
        body = new.get("body", new.get("text"))
        if isinstance(body, str) and not body.strip():
            empty += 1
    return cleaned, {"by_class": {k: dict(v) for k, v in by_class.items()}, "n_empty": empty,
                     "n_records": len(cleaned)}
