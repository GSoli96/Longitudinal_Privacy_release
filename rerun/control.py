"""Control without documents (spec 7.5, plan T11).

The attacker receives only the person's role (in generic terms) and the type of organisation: no name, no
documents, the same instruction and sampling as in the main run, the evidence-free schema (there is no text to
quote). The same judges judge the attributes against that one-line text (so "explicit" means stated in the role
line). The DIR with the documents (final step) is compared with the DIR of the control person by person; the
difference comes with a cluster bootstrap interval over persons. Persons without a role are excluded and counted.

Roles are extracted by one model (``config.CATEGORY_MODEL``) from the first step text of the person, asked for a
generic job title or function and level, without names of people, units, offices, places or organisations; the
answer is kept only if it is short (the extraction cannot detect a name by itself: the roles file stays under
``data/private/`` and is checked by sampling before use).
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, infer, metrics, run_case, stats, steps

MAX_ROLE_CHARS = 120
ROLE_NUM_PREDICT = 1024
ROLE_TEXT_CHARS = 15000
ROLE_NUM_CTX = 16384                 # 15,000 characters of text plus the answer

ROLE_SCHEMA = {"type": "object", "properties": {"role": {"type": "string"}}, "required": ["role"]}

ROLE_PROMPT = """
You are given administrative documents about a public employee.

State the person's role in generic terms: the job title or function and its level (for example "senior manager of an office" or "executive director"), without any name of a person, unit, office, city or organisation.
If the role cannot be determined, answer with an empty role.

DOCUMENTS:
----------------
{text}
----------------

Answer only with JSON: {{"role": "..."}}
"""


def control_text(role: str, org_type: str | None) -> str:
    return f"Role: {role}\nOrganisation type: {org_type or 'public administration'}"


def control_step(role: str, org_type: str | None):
    t = control_text(role, org_type)
    return steps.Step(k=100, n_docs=0, text=t, identical_to_prev=False, dated_docs=0, undated_docs=0, chars=len(t))


def clean_role(x) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"\s+", " ", x).strip()
    return x if 0 < len(x) <= MAX_ROLE_CHARS else ""


def save_roles(roles: dict, path: Path):
    Path(path).write_text(json.dumps(roles, ensure_ascii=False, indent=1), encoding="utf-8")


def load_roles(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_roles(persons, client, *, model: str = config.CATEGORY_MODEL, workers: int = 4):
    """-> ({person id: role}, info). Persons whose role stays empty are not in the dict."""
    persons = list(persons)

    def one(p):
        st = steps.build_steps(p)
        if not st:
            return p.id, ""
        prompt = ROLE_PROMPT.format(text=st[0].text[:ROLE_TEXT_CHARS])
        opts = {"num_ctx": min(config.num_ctx_for(model), ROLE_NUM_CTX), "num_predict": ROLE_NUM_PREDICT, "seed": config.SEED}
        r = client.chat(model, [{"role": "user", "content": prompt}], opts, ROLE_SCHEMA, think=config.JUDGE_THINK.get(model))
        return p.id, (clean_role(r.json.get("role")) if r.ok and isinstance(r.json, dict) else "")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        res = list(ex.map(one, persons))
    roles = {pid: role for pid, role in res if role}
    return roles, {"persons": len(persons), "with_role": len(roles), "no_role": len(persons) - len(roles)}


def run_control_infer(persons, roles: dict, client, out_path: Path, *, model: str = config.ATTACKER,
                      workers: int = 4) -> dict:
    todo = [p for p in persons if p.id in roles]
    t0 = time.time()

    def one(p):
        res = infer.infer_step(client, control_step(roles[p.id], p.org_type), model, usable_tokens=10 ** 6,
                               variant="no_evidence")
        return {"person": p.id, "step": 100, "variant": "no_evidence", "carried": False, "n_docs": 0, "chars": 0,
                "dated_docs": 0, "undated_docs": 0, "ok": res.ok, "error": res.error, "windows": 1, "chunked": False,
                "partial": False, "failed_windows": [], "structure_retries": res.structure_retries,
                "empty_retries": res.empty_retries, "evidence_exact": 0, "evidence_fuzzy": 0, "evidence_none": 0,
                "tokens_in": res.tokens_in, "tokens_out": res.tokens_out, "seconds": round(res.seconds, 2),
                "carried_from": None,
                "attributes": [{"id": run_case.attr_id(p.id, 100, n), "text": a.text, "category_hint": a.category_hint,
                                "evidence": "", "window": 0, "windows": [0], "evidence_in_text": False,
                                "evidence_match": "n/a", "evidence_ratio": 0.0} for n, a in enumerate(res.attributes, 1)]}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = list(ex.map(one, todo))
    run_case._write_jsonl(Path(out_path), rows)
    return {"persons": len(rows), "no_role": len(list(persons)) - len(todo),
            "attributes": sum(len(r["attributes"]) for r in rows), "failed": sum(1 for r in rows if not r["ok"]),
            "empty": sum(1 for r in rows if r["ok"] and not r["attributes"]),
            "seconds_wall": round(time.time() - t0, 1)}


def run_control_judge(persons, roles: dict, infer_path: Path, client, judges, out_path: Path, *,
                      workers: int = 4, sequential: bool = False, full_panel: bool = False) -> dict:
    """The panel and the voting scheme of the main run (``sequential``, ``full_panel`` as in ``run_case.run_judge``),
    so that the control is judged exactly like the condition with documents."""
    by = {p.id: p for p in persons}
    return run_case.run_judge(persons, infer_path, client, judges, out_path, attacker=config.ATTACKER,
                              usable_tokens=10 ** 6, workers=workers, sequential=sequential, full_panel=full_panel,
                              texts_for=lambda pid, row: [control_text(roles[pid], by[pid].org_type)])


def _per_person_dir(records) -> dict:
    by = {}
    for r in records:
        if r.step == 100:
            by.setdefault(r.person, []).append(r)
    return {p: metrics.basic_metrics(rs) for p, rs in by.items() if metrics.basic_metrics(rs)["n_complete"]}


def compare_with_documents(docs_infer, docs_judge, ctl_infer, ctl_judge, *, n_resamples: int = 2000) -> dict:
    """Paired DIR (final step with documents vs control) per person, difference = documents - control."""
    d = _per_person_dir(run_case.build_records(docs_infer, docs_judge))
    c = _per_person_dir(run_case.build_records(ctl_infer, ctl_judge))
    both = sorted(set(d) & set(c))
    diffs = [d[p]["DIR"] - c[p]["DIR"] for p in both]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    # explicit share over the attributes (pooled), like the other attribute-level rates of the paper
    expl = lambda ms: sum(x["E"] * x["n_complete"] for x in ms) / sum(x["n_complete"] for x in ms) if ms else float("nan")
    return {"n_persons": len(both), "persons_without_pair": len((set(d) | set(c)) - set(both)),
            "DIR_documents": {"mean": mean([d[p]["DIR"] for p in both])},
            "DIR_control": {"mean": mean([c[p]["DIR"] for p in both])},
            "DIR_difference": stats.cluster_bootstrap_ci(diffs, lambda xs: sum(xs) / len(xs) if xs else float("nan"),
                                                         n_resamples=n_resamples),
            "IQS": {"documents": mean([d[p]["IQS"] for p in both]), "control": mean([c[p]["IQS"] for p in both])},
            "attributes": {"documents": sum(d[p]["n_complete"] for p in both),
                           "control": sum(c[p]["n_complete"] for p in both)},
            "explicit_share": {"documents": expl([d[p] for p in both]), "control": expl([c[p] for p in both])}}
