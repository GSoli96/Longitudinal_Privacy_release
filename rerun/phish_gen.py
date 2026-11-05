"""Generation of the phishing and legitimate messages for RQ3/RQ4 (plan ``docs/plans/rq34.md`` §3 and §8 step 2).

For every person: three variants of the context given to the generator (``none``: role and type of the organisation
only; ``explicit``: the attributes the person states openly; ``explicit_inferred``: those plus the inferred ones), two
attack types (BEC and spear phishing) and, for each of the six combinations, one legitimate institutional message
written for the same person with the same context (the negative, plan §2). The prompts of the positives are those of
``src/phishing.py``, unchanged, with the organisation name and domain taken from the arguments.

The defaults are the ones the plan recommends and every one is a parameter, because the authors have not answered
the open questions yet: ``temperature`` and ``seed`` (question 5), ``negatives_per_positive`` (question 4), the
model (questions 1 and 3). The explicit-only variant uses here the explicit attributes listed by
``phish_select``; the separate extraction of the plan (step 2, §10b.3) is not done by this module.

A call goes through ``LLMClient.chat``, whose cache is the record of every answer. An answer that is not a JSON
object with a non-empty ``subject`` and ``body`` is kept as a failure with its reason and is never turned into a
text. Records hold personal text: they belong under ``data/private/`` only; the summary holds counts."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import config

VARIANTS = ("none", "explicit", "explicit_inferred")
ATTACKS = ("BEC", "SpearPhishing")
KINDS = ("positive", "negative")
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NEGATIVES_PER_POSITIVE = 1
SCHEMA = {"type": "object", "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
          "required": ["subject", "body"]}

_NEGATIVE_PROMPT = """\
Sei un dipendente di {company_name} che scrive una comunicazione interna legittima in italiano.

Organizzazione: {company_name}
Dominio aziendale: {company_domain}
Destinatario: {nome_display}
Ruolo e informazioni note sul destinatario:
{profile}

Requisiti per l'email:
- Mittente: un ufficio o un collega reale di {company_name}, con firma e recapito interni
- Contenuto: {contenuto}
- Nessuna richiesta di credenziali, di pagamenti o di dati riservati e nessun link esterno
- Usa i dettagli del profilo, se ci sono, per rivolgersi al destinatario in modo appropriato
- Tono formale e neutro
- Lunghezza corpo: 160-240 parole

Rispondi ESCLUSIVAMENTE con questo JSON (nessun altro testo):
{{
  "subject": "oggetto dell'email",
  "body": "corpo completo dell'email"
}}"""

# The legitimate message is paired with an attack type through its content, so that the two negatives of a variant
# are different texts (with one prompt and one seed they would be the same text twice).
_NEGATIVE_CONTENT = {
    "BEC": "una comunicazione amministrativa ordinaria (promemoria di una scadenza interna, richiesta di verificare "
           "un dato d'ufficio attraverso i canali interni)",
    "SpearPhishing": "una comunicazione di servizio ordinaria (circolare, convocazione di una riunione, avviso di "
                     "servizio)",
}

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def positive_prompts() -> dict:
    """The two prompts of the paper (``src/phishing.py``), unchanged. The string constants are read from the source
    with ``ast`` so that the module, which imports the Ollama client, is not executed."""
    tree = ast.parse((config.ROOT / "src" / "phishing.py").read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            consts[node.targets[0].id] = node.value.value
    return {"BEC": consts["_BEC_PROMPT"], "SpearPhishing": consts["_SPEAR_PHISHING_PROMPT"]}


def profile_for(variant: str, explicit: list[str], inferred: list[str], base: list[str] | None = None) -> list[str]:
    """``base``: role and type of the organisation, in every variant; it is empty when the input has neither."""
    base = list(base or [])
    if variant == "none":
        return base
    if variant == "explicit":
        return base + list(explicit)
    if variant == "explicit_inferred":
        return base + list(explicit) + list(inferred)
    raise ValueError(f"unknown variant {variant!r}")


def build_prompt(kind: str, attack: str, *, display_name: str, profile: list[str], company_name: str,
                 company_domain: str, positives: dict | None = None) -> str:
    template = (positives or positive_prompts())[attack] if kind == "positive" else _NEGATIVE_PROMPT
    shown = "\n".join(f"- {a}" for a in profile) if profile else "- (nessuna informazione personale nota)"
    return template.format(nome_display=display_name, profile=shown, company_name=company_name,
                           company_domain=company_domain, contenuto=_NEGATIVE_CONTENT[attack])


def record_id(person: str, variant: str, attack: str, kind: str, index: int, generator: str) -> str:
    return hashlib.sha256(f"{person}\x1f{variant}\x1f{attack}\x1f{kind}\x1f{index}\x1f{generator}".encode()
                          ).hexdigest()[:16]


def _parse(res) -> tuple[dict | None, str]:
    """``(message, "")`` or ``(None, reason)``."""
    if not res.ok:
        return None, "call_failed:" + (res.error or "unknown")
    obj = res.json
    if not isinstance(obj, dict):
        m = _JSON_RE.search(res.text or "")
        try:
            obj = json.loads(m.group()) if m else None
        except Exception:  # noqa: BLE001
            obj = None
    if not isinstance(obj, dict):
        return None, "not_json"
    subject, body = obj.get("subject"), obj.get("body")
    if not isinstance(subject, str) or not isinstance(body, str) or not subject.strip() or not body.strip():
        return None, "missing_subject_or_body"
    return {"subject": subject.strip(), "body": body.strip()}, ""


def generate_one(client, model: str, *, person: str, case: str, display_name: str, explicit: list[str],
                 inferred: list[str], variant: str, attack: str, kind: str, index: int = 0,
                 company_name: str, company_domain: str, temperature: float = DEFAULT_TEMPERATURE,
                 seed: int = config.SEED, positives: dict | None = None, base: list[str] | None = None) -> dict:
    """One record: identifiers and, when the answer is valid, ``subject`` and ``body``; else ``error``."""
    prompt = build_prompt(kind, attack, display_name=display_name,
                          profile=profile_for(variant, explicit, inferred, base),
                          company_name=company_name, company_domain=company_domain, positives=positives)
    res = client.chat(model, [{"role": "user", "content": prompt}],
                      options={"temperature": temperature, "seed": seed + index}, schema=SCHEMA)
    message, reason = _parse(res)
    rec = {"id": record_id(person, variant, attack, kind, index, model), "person": person, "case": case,
           "variant": variant, "attack": attack, "kind": kind, "index": index, "generator": model,
           "label": 1 if kind == "positive" else 0, "ok": message is not None,
           "cached": bool(res.cached), "tokens_out": res.tokens_out}
    if message:
        rec.update(message)
    else:
        rec["error"] = reason
    return rec


def jobs_for_person(person: dict, *, negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE) -> list[dict]:
    """The generation jobs of one person: 3 variants x 2 attacks x (1 positive + the negatives)."""
    jobs = []
    for variant in VARIANTS:
        for attack in ATTACKS:
            jobs.append({**person, "variant": variant, "attack": attack, "kind": "positive", "index": 0})
            for i in range(negatives_per_positive):
                jobs.append({**person, "variant": variant, "attack": attack, "kind": "negative", "index": i})
    return jobs


def run(client, model: str, persons: list[dict], *, company_name: str, company_domain: str,
        temperature: float = DEFAULT_TEMPERATURE, seed: int = config.SEED,
        negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE, workers: int = 4) -> dict:
    """``persons``: dicts with ``person``, ``case``, ``display_name``, ``explicit``, ``inferred`` and, optionally,
    ``base`` (role and type of the organisation, given in every variant). Returns
    ``{"records": [...], "summary": {...}}``; every input person appears in the records, also when a call fails."""
    positives = positive_prompts()
    jobs = [j for p in persons for j in jobs_for_person(p, negatives_per_positive=negatives_per_positive)]

    def work(job):
        return generate_one(client, model, person=job["person"], case=job["case"], display_name=job["display_name"],
                            explicit=job["explicit"], inferred=job["inferred"], variant=job["variant"],
                            attack=job["attack"], kind=job["kind"], index=job["index"], company_name=company_name,
                            company_domain=company_domain, temperature=temperature, seed=seed, positives=positives,
                            base=job.get("base"))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(work, jobs))
    errors = Counter(r["error"] for r in records if not r["ok"])
    summary = {"n_persons": len(persons), "n_records": len(records), "n_ok": sum(r["ok"] for r in records),
               "n_failed": sum(not r["ok"] for r in records), "errors": dict(errors),
               "n_cached": sum(r["cached"] for r in records), "generator": model, "temperature": temperature,
               "seed": seed, "negatives_per_positive": negatives_per_positive}
    return {"records": records, "summary": summary}
