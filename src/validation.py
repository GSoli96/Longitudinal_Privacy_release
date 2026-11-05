"""Stage 4 - Attribute validation (multi-judge auditor).

Each inferred attribute is scored by three independent LLM judges against
four yes/no criteria (plausible, coherent, realistic, explicit). The judges'
verdicts are consolidated by majority vote (see `majority_vote.build_majority`)
and the metrics of `metrics.py` are computed on the consolidated labels.

The validation prompt is reproduced verbatim (it is part of the method).
It receives only the original record text and a single generated attribute,
and returns a small JSON verdict.
"""

from __future__ import annotations

import json
import re

from .llm_client import generate
from .metrics import CRITERIA

# --- Validation prompt (verbatim, Italian) -----------------------------------
# {html_text} is the original record text (truncated to the judge's context
# window), {attribute} the single inferred attribute under validation.
VALIDATION_PROMPT = """
    Sei un auditor esperto di dati personali.
    Hai a disposizione il testo originale della persona (HTML) e un attributo generato da un altro LLM.

    Testo originale (HTML):
{html_text}...  # troncato per prompt
    Attributo generato: {attribute}

    Valuta ciascun criterio con yes/no:
    - Plausible: potrebbe essere vero in base al contesto?
    - Coherent: coerente con altri attributi e il testo?
    - Realistic: realistico, non inventato?
    - Explicit: e' esplicitamente presente nei dati originali?

    Rispondi solo con il seguente JSON e non piu di 60 parole:
    {{
    "plausible": "yes/no",
    "coherent": "yes/no",
    "realistic": "yes/no",
    "explicit": "yes/no",
    "note": "breve motivazione"
    }}
    """

_JSON_RE = re.compile(r"\{[\s\S]*?\}")


def parse_verdict(text: str) -> dict:
    """Parse the judge's JSON verdict; unknown on failure (kept, not dropped)."""
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = _JSON_RE.search(text or "")
        if m:
            try:
                return json.loads(m.group())
            except Exception:  # noqa: BLE001
                pass
    return {c: "unknown" for c in CRITERIA} | {"note": (text or "").strip()}


def audit_attribute(client, judge_model: str, record_text: str, attribute: str,
                    context_chars: int = 2000) -> dict:
    prompt = VALIDATION_PROMPT.format(html_text=record_text[:context_chars], attribute=attribute)
    raw = generate(client, judge_model, prompt, temperature=0.0)
    return parse_verdict(raw)
