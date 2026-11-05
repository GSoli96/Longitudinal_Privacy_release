"""Exposure steps and text windows (plan T3, spec section 4.1-4.2).

* ``build_steps``: the text a model sees at 25/50/75/100 % of a person's documents. The
  documents come already ordered by ``corpus`` (chronological for C, E, F with undated ones
  last in their original order; portal-section order for G). A step holds the first
  ``ceil(k * n / 100)`` documents; a step whose text equals the previous one is flagged
  ``identical_to_prev`` and is not run again.
* ``windows`` / ``window_spans``: a text longer than the usable context is cut into windows of
  at most ``ctx_tokens * chars_per_token`` characters with 10 % overlap, cut at line boundaries
  when there is one in the last fifth of the window; together they cover the whole text.
* Token budget: ``estimate_tokens`` is a conservative character-based estimate (default 3.5
  characters per token for Italian; per-model ratios measured with the server's own token
  counts are in ``config.TOKEN_CHARS``), ``usable_input_tokens`` subtracts the output and
  prompt reserves from a model's context, ``choose_num_ctx`` picks the smallest power-of-two
  context that fits, capped at the model maximum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

CHARS_PER_TOKEN = 3.5
FRACTIONS = (25, 50, 75, 100)  # percent
DOC_HEADER = "[Document {i}]"
OVERLAP = 0.10
MIN_WINDOW_CHARS = 500
NUM_CTX_SIZES = tuple(2 ** k for k in range(13, 21))  # 8,192 ... 1,048,576


@dataclass
class Step:
    k: int                      # 25 / 50 / 75 / 100
    n_docs: int
    text: str
    identical_to_prev: bool
    dated_docs: int
    undated_docs: int
    chars: int = 0              # cumulative characters of the step (amount-of-text axis)


def build_steps(person, fractions=None) -> list[Step]:
    """Steps of one person; ``[]`` for a person without text. ``fractions`` defaults to the module's ``FRACTIONS``
    (read at call time: ``run_case --final-only`` sets it to the last step alone)."""
    fractions = FRACTIONS if fractions is None else fractions
    if person.status == "no_text":
        return []
    docs = person.docs
    n = len(docs)
    out, prev_text = [], None
    for k in fractions:
        m = min(n, max(1, -(-k * n // 100)))  # ceil(k * n / 100), at least one document
        parts = [f"{DOC_HEADER.format(i=i)}\n{d.text.strip()}" for i, d in enumerate(docs[:m], 1) if d.text.strip()]
        text = "\n\n".join(parts)
        dated = sum(1 for d in docs[:m] if d.date)
        out.append(Step(k=k, n_docs=m, text=text, identical_to_prev=(prev_text is not None and text == prev_text),
                        dated_docs=dated, undated_docs=m - dated, chars=len(text)))
        prev_text = text
    return out


def estimate_tokens(text_or_chars, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    n = text_or_chars if isinstance(text_or_chars, int) else len(text_or_chars)
    return math.ceil(n / chars_per_token) if n else 0


def usable_input_tokens(model_ctx: int, *, output_reserve: int, prompt_overhead: int) -> int:
    usable = model_ctx - output_reserve - prompt_overhead
    if usable <= 0:
        raise ValueError("context smaller than the reserves")
    return usable


def choose_num_ctx(tokens_needed: int, model_max: int) -> int:
    """Smallest power-of-two context (>= 8,192) holding ``tokens_needed``, capped at ``model_max``."""
    for size in NUM_CTX_SIZES:
        if size >= tokens_needed:
            return min(size, model_max)
    return model_max


def window_spans(text: str, ctx_tokens: int, chars_per_token: float = CHARS_PER_TOKEN,
                 overlap: float = OVERLAP) -> list[tuple[int, int]]:
    wchars = int(ctx_tokens * chars_per_token)
    if wchars < MIN_WINDOW_CHARS:
        raise ValueError(f"window of {wchars} characters is too small")
    n = len(text)
    if n <= wchars:
        return [(0, n)]
    ov = int(overlap * wchars)
    spans, start = [], 0
    while True:
        end = min(start + wchars, n)
        if end < n:  # cut at the last line break of the final fifth of the window, if any
            cut = text.rfind("\n", start + int(0.8 * wchars), end)
            if cut != -1:
                end = cut + 1
        spans.append((start, end))
        if end >= n:
            return spans
        nxt = end - ov
        nl = text.find("\n", nxt, end)  # start the next window at a line start inside the overlap
        if nl != -1 and nl + 1 < end:
            nxt = nl + 1
        start = max(nxt, start + 1)


def windows(text: str, ctx_tokens: int, chars_per_token: float = CHARS_PER_TOKEN,
            overlap: float = OVERLAP) -> list[str]:
    return [text[a:b] for a, b in window_spans(text, ctx_tokens, chars_per_token, overlap)]
