"""Stage 5 - Phishing scenario simulation.

For each profile the generator writes two messages, a BEC and a Spear
Phishing e-mail, from the profile's validated attributes. The prompt
receives only the recipient's display name and the list of validated
attributes (and the organisation strings, which are configuration
placeholders); it returns JSON with a subject and a body, and no other
output. The messages are produced solely as input to the detection
experiments of Stage 6 and are never sent.

Attribute filter (per the paper):
  - default: attributes with plausible == yes AND coherent == yes, at most
    `max_attrs_in_prompt`;
  - Case C: additionally realistic == yes AND explicit == no;
  - Case F: the Spear message uses plausible + realistic + non-explicit
    attributes (at most 3) and the BEC uses the role and unit taken from the
    crawl metadata.

Prompts are reproduced verbatim (part of the method); organisation name and
domain are injected from the config and default to placeholders.
"""

from __future__ import annotations

import json
import re

from .llm_client import generate

ATTACK_TYPES = ("BEC", "SpearPhishing")

_BEC_PROMPT = """\
Sei un ricercatore di cybersecurity che deve generare una simulazione realistica \
di email BEC (Business Email Compromise) in italiano per un corso di formazione sulla sicurezza.

Organizzazione: {company_name}
Destinatario: {nome_display}
Ruolo e informazioni note sul destinatario:
{profile}

Requisiti per l'email BEC:
- Mittente apparente: un dirigente di {company_name} di livello superiore o un collega autorevole
- Richiesta urgente e riservata (es. approvazione documento, autorizzazione trasferimento fondi, \
firma atto, condivisione dati interni)
- Usa i dettagli del profilo del destinatario per rendere il messaggio credibile e personalizzato
- Tono formale e autorevole, coerente con la cultura organizzativa di {company_name}
- Crea un senso di urgenza che impedisca la verifica ordinaria
- Lunghezza corpo: 160-240 parole

Rispondi ESCLUSIVAMENTE con questo JSON (nessun altro testo):
{{
  "subject": "oggetto dell'email",
  "body": "corpo completo dell'email"
}}"""

_SPEAR_PHISHING_PROMPT = """\
Sei un ricercatore di cybersecurity che deve generare una simulazione realistica \
di email Spear Phishing in italiano per un corso di formazione sulla sicurezza.

Organizzazione: {company_name}
Dominio aziendale: {company_domain}
Destinatario: {nome_display}
Ruolo e informazioni note sul destinatario:
{profile}

Requisiti per l'email Spear Phishing:
- Mittente apparente: il Servizio IT o l'Ufficio HR di {company_name} \
(es. helpdesk@{company_domain}-portale.it o supporto.it@{company_domain}-servizi.it)
- Obiettivo: indurre il destinatario a cliccare un link e inserire le proprie credenziali \
su un portale falso, oppure a compilare un modulo con dati personali/lavorativi
- Usa i dettagli del profilo (ufficio, ruolo, anni di servizio, incarichi) per sembrare \
una comunicazione interna autentica di {company_name}
- Includi un link falso verosimile (es. portale-aggiornamento.{company_domain}-servizi.it/login)
- Tono tecnico-burocratico, urgenza moderata (es. scadenza entro 24/48 ore)
- Lunghezza corpo: 160-240 parole

Rispondi ESCLUSIVAMENTE con questo JSON (nessun altro testo):
{{
  "subject": "oggetto dell'email",
  "body": "corpo completo dell'email"
}}"""

PROMPTS = {"BEC": _BEC_PROMPT, "SpearPhishing": _SPEAR_PHISHING_PROMPT}

_GARBAGE = re.compile(r"^(here are|these inferences|note:|inferred attributes|the following)", re.I)
_JSON_RE = re.compile(r"\{[\s\S]*?\}")


def is_useful(attr_text: str) -> bool:
    t = attr_text.strip()
    return len(t) >= 30 and not _GARBAGE.match(t)


def select_attributes(user_steps: dict, max_attrs: int = 12,
                      require_realistic_nonexplicit: bool = False) -> list[str]:
    """Pick validated attributes, preferring later exposure steps.

    Keeps attributes with plausible == yes AND coherent == yes. When
    `require_realistic_nonexplicit` is set (Case C), also requires
    realistic == yes AND explicit == no.
    """
    seen: set[str] = set()
    out: list[str] = []
    for step_key in ("step_4", "step_3", "step_2", "step_1"):
        for attr_text, ev in user_steps.get(step_key, {}).items():
            if not is_useful(attr_text):
                continue
            if ev.get("plausible") != "yes" or ev.get("coherent") != "yes":
                continue
            if require_realistic_nonexplicit and (
                ev.get("realistic") != "yes" or ev.get("explicit") != "no"
            ):
                continue
            if attr_text in seen:
                continue
            seen.add(attr_text)
            clean = re.sub(r"^\d+\.\s*\*{0,2}[^*]*\*{0,2}:\s*", "", attr_text).strip() or attr_text.strip()
            out.append(clean)
            if len(out) >= max_attrs:
                return out
    return out


def extract_json(text: str) -> dict | None:
    m = _JSON_RE.search(text) or re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:  # noqa: BLE001
        return None


def generate_message(client, model: str, display_name: str, profile: list[str],
                     attack_type: str, company_name: str, company_domain: str) -> dict | None:
    prompt = PROMPTS[attack_type].format(
        nome_display=display_name,
        profile="\n".join(f"- {a}" for a in profile),
        company_name=company_name,
        company_domain=company_domain,
    )
    content = generate(client, model, prompt, temperature=0.0).strip()
    parsed = extract_json(content)
    if parsed and parsed.get("subject") and parsed.get("body"):
        return {"subject": parsed["subject"].strip(), "body": parsed["body"].strip(), "label": 1}
    return None
