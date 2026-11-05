"""Ablations and mitigation policies on one case (spec §7.3, §7.4, §10b.1; plan ``docs/plans/ablations-E.md``).

Every condition is a pure function from a list of ``Person`` to a list of ``Person`` with the same ids, so that the
steps of the inference, the texts the judges read and the metrics all come from the same transformed documents
(``steps.build_steps`` works on ``person.docs``). A condition is named by a short spec string:

``docs:N``       the first N documents of the person's order (chronological, undated last); N = 1, 2, 4, 8.
                 A person with fewer than N documents keeps all of them (counted in the info as ``fewer_than_n``).
``order:P``      the same documents in a seeded random order, permutation P = 1, 2, 3 (seed ``"seed:person:P"``).
``years:K``      mitigation, window: documents dated within K years before the person's latest dated document;
                 undated documents are dropped (they cannot be placed in a window). Persons left without documents
                 are excluded and counted.
``last:K``       mitigation, K documents per person: the K most recent (the last K of the person's order).
``full``          no change: the baseline of the other conditions, run with the same attacker, judges and persons.
``generalise``   mitigation, generalisation of fields by rules: dates reduced to the year, amounts to their order of
                 magnitude, e-mail addresses, telephone numbers, tax codes and IBANs replaced by a placeholder.

Assumptions where the specification is silent (to confirm with the authors): the first / last N documents instead of a
random sample, undated documents dropped by the year window, rule-based generalisation instead of a rewriting model.

Information kept (``info_kept``): share of characters and of documents that survive the policy, pooled over the
persons and as the mean of the per-person shares. Aggregates only."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from dataclasses import replace
from pathlib import Path

from . import config, control, run_case

DOC_COUNTS = (1, 2, 4, 8)
PERMUTATIONS = (1, 2, 3)
POLICY_K = {"years": (1, 3), "last": (2, 4)}      # two values of k for each window / count policy (assumption)


def parse(spec: str):
    name, _, param = spec.partition(":")
    if name in ("docs", "order", "years", "last"):
        if not param.isdigit() or int(param) < 1:
            raise ValueError(f"condition {spec!r}: a positive integer is needed after the colon")
        return name, int(param)
    if name in ("generalise", "full") and not param:
        return name, None
    raise ValueError(f"unknown condition {spec!r}")


def tag_for(spec: str) -> str:
    name, k = parse(spec)
    return {"docs": f"abl-docs{k}", "order": f"abl-order{k}", "years": f"mit-years{k}", "last": f"mit-last{k}",
            "generalise": "mit-generalise", "full": "abl-full"}[name]


# ---- the transformations ---------------------------------------------------------------------------------------------
def first_docs(person, n: int):
    return replace(person, docs=list(person.docs[:n]))


def shuffled(person, perm: int, seed: int = config.SEED):
    docs = list(person.docs)
    random.Random(f"{seed}:{person.id}:{perm}").shuffle(docs)
    return replace(person, docs=docs)


def last_docs(person, k: int):
    return replace(person, docs=list(person.docs[-k:]))


def last_years(person, years: int):
    dated = [d for d in person.docs if d.date]
    if not dated:
        return replace(person, docs=[])
    newest = max(d.date for d in dated)
    keep = [d for d in person.docs if d.date and (newest - d.date).days <= round(365.25 * years)]
    return replace(person, docs=keep)


_DATE_NUM = re.compile(r"\b\d{1,2}\s*[/.\-]\s*\d{1,2}\s*[/.\-]\s*((?:19|20)\d{2})\b")
_MONTHS = "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
_DATE_WORD = re.compile(rf"\b\d{{1,2}}(?:°|º)?\s+(?:{_MONTHS})\s+((?:19|20)\d{{2}})\b", re.I)
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
_IBAN = re.compile(r"\bIT\s?\d{2}\s?[A-Z]\s?(?:\d\s?){10}(?:[A-Z0-9]\s?){12}\b", re.I)
_CF = re.compile(r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b", re.I)
_PHONE = re.compile(r"(?<![\d/.])(?:\+39\s?)?(?:0\d{1,3}|3\d{2})[\s./-]?\d{6,8}(?!\d)")
_AMOUNT = re.compile(r"(?:€\s?(\d[\d.]*(?:,\d+)?)|(\d[\d.]*(?:,\d+)?)\s?(?:€|euro\b))", re.I)


def _bucket(m: re.Match) -> str:
    raw = (m.group(1) or m.group(2)).replace(".", "").replace(",", ".")
    try:
        v = float(raw)
    except ValueError:
        return m.group(0)
    return "euro dell'ordine di 10^%d" % int(math.floor(math.log10(v))) if v >= 1 else "euro (meno di 1)"


def generalise_text(text: str) -> str:
    t = _EMAIL.sub("[e-mail]", text)
    t = _IBAN.sub("[IBAN]", t)
    t = _CF.sub("[codice fiscale]", t)
    t = _DATE_NUM.sub(r"\1", t)
    t = _DATE_WORD.sub(r"\1", t)
    t = _AMOUNT.sub(_bucket, t)
    return _PHONE.sub("[telefono]", t)


def generalised(person):
    return replace(person, docs=[replace(d, text=generalise_text(d.text)) for d in person.docs])


# ---- application -------------------------------------------------------------------------------------------------------
def apply(persons, spec: str):
    """-> (transformed persons, info). Persons left without a document are removed and counted (never given a label)."""
    name, k = parse(spec)
    persons = list(persons)
    if name == "docs":
        out = [first_docs(p, k) for p in persons]
    elif name == "order":
        out = [shuffled(p, k) for p in persons]
    elif name == "years":
        out = [last_years(p, k) for p in persons]
    elif name == "last":
        out = [last_docs(p, k) for p in persons]
    elif name == "full":
        out = list(persons)
    else:
        out = [generalised(p) for p in persons]
    kept = [p for p in out if p.docs]
    info = {"spec": spec, "persons_in": len(persons), "persons_out": len(kept),
            "persons_excluded_without_documents": len(persons) - len(kept)}
    if name == "docs":
        info["fewer_than_n"] = sum(1 for p in persons if len(p.docs) < k)
    info["info_kept"] = info_kept(persons, out)
    return kept, info


def info_kept(before, after) -> dict:
    """Share of characters and of documents kept, pooled and as the mean over persons (persons with text only; the
    character share of a person is capped at 1)."""
    b = {p.id: p for p in before}
    tot_c0 = tot_c1 = tot_d0 = tot_d1 = 0
    shares_c, shares_d = [], []
    for p in after:
        o = b.get(p.id)
        if o is None or not o.docs:
            continue
        c0 = sum(len(d.text) for d in o.docs)
        c1 = min(sum(len(d.text) for d in p.docs), c0)         # generalising can lengthen a text: never more than all
        tot_c0, tot_c1, tot_d0, tot_d1 = tot_c0 + c0, tot_c1 + c1, tot_d0 + len(o.docs), tot_d1 + len(p.docs)
        shares_c.append(c1 / c0 if c0 else 1.0)
        shares_d.append(len(p.docs) / len(o.docs))
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else float("nan")
    return {"characters_pooled": round(tot_c1 / tot_c0, 4) if tot_c0 else float("nan"),
            "documents_pooled": round(tot_d1 / tot_d0, 4) if tot_d0 else float("nan"),
            "characters_mean_person": mean(shares_c), "documents_mean_person": mean(shares_d)}


def compare(base_infer, base_judge, var_infer, var_judge, *, n_resamples: int = 2000) -> dict:
    """Paired DIR at the final step, baseline minus variant, over the persons that have attributes in both
    (``control.compare_with_documents`` with the names of an ablation)."""
    r = control.compare_with_documents(base_infer, base_judge, var_infer, var_judge, n_resamples=n_resamples)
    return {"n_persons": r["n_persons"], "persons_without_pair": r["persons_without_pair"],
            "DIR_baseline": r["DIR_documents"], "DIR_variant": r["DIR_control"],
            "DIR_reduction": r["DIR_difference"], "IQS": {"baseline": r["IQS"]["documents"], "variant": r["IQS"]["control"]},
            "attributes": {"baseline": r["attributes"]["documents"], "variant": r["attributes"]["control"]},
            "explicit_share": {"baseline": r["explicit_share"]["documents"], "variant": r["explicit_share"]["control"]}}


# ---- reports (aggregates only) ---------------------------------------------------------------------------------------
def _rows(case, tag, base=config.PRIVATE):
    d = run_case.stage_dir(case, base, tag)
    return d, run_case._read_jsonl(d / "infer.jsonl"), run_case._read_jsonl(d / "judge.jsonl")


def report(case: str, specs, baseline_tag: str, *, n_resamples: int = 2000, base=config.PRIVATE) -> dict:
    """Per condition: paired DIR reduction against the baseline run on the same persons, information kept, persons."""
    _, b_inf, b_jud = _rows(case, baseline_tag, base)
    out = {}
    for spec in specs:
        d, v_inf, v_jud = _rows(case, tag_for(spec), base)
        if not v_jud:
            out[spec] = {"status": "not run"}
            continue
        cond = json.loads((d / "condition.json").read_text(encoding="utf-8")) if (d / "condition.json").exists() else {}
        ids = set(cond.get("persons") or [r["person"] for r in v_inf])
        res = compare([r for r in b_inf if r["person"] in ids], [r for r in b_jud if r["person"] in ids], v_inf, v_jud,
                      n_resamples=n_resamples)
        res["persons_run"] = len(ids)
        res["info_kept"] = (cond.get("ablation") or {}).get("info_kept")
        res["persons_excluded_without_documents"] = (cond.get("ablation") or {}).get("persons_excluded_without_documents")
        out[spec] = res
    return out


def variability(case: str, tags, *, baseline_tag: str | None = None, base=config.PRIVATE) -> dict:
    """Repetitions of the whole inference (``--seed-offset``): final-step DIR of each run on the persons all runs have,
    its spread between runs, and the mean absolute per-person difference between runs (the noise of one run)."""
    runs = {}
    for t in ([baseline_tag] if baseline_tag else []) + list(tags):
        _, inf, jud = _rows(case, t, base)
        if jud:
            runs[t] = control._per_person_dir(run_case.build_records(inf, jud))
    if len(runs) < 2:
        return {"runs": sorted(runs), "status": "fewer than two runs"}
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    means = {t: statistics.fmean(r[p]["DIR"] for p in common) for t, r in runs.items()} if common else {}
    names = sorted(runs)
    pair_diffs = [statistics.fmean(abs(runs[a][p]["DIR"] - runs[b][p]["DIR"]) for p in common)
                  for i, a in enumerate(names) for b in names[i + 1:]] if common else []
    return {"runs": names, "persons_in_all_runs": len(common),
            "DIR_by_run": {t: round(v, 4) for t, v in means.items()},
            "DIR_sd_between_runs": round(statistics.stdev(means.values()), 4) if len(means) > 1 else float("nan"),
            "DIR_range": round(max(means.values()) - min(means.values()), 4) if means else float("nan"),
            "mean_abs_person_difference_between_runs": round(statistics.fmean(pair_diffs), 4) if pair_diffs else float("nan")}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Report of the ablations / mitigation conditions of one case")
    ap.add_argument("--case", required=True)
    ap.add_argument("--baseline-tag", required=True, help="stage directory tag of the run to compare with (e.g. gloc)")
    ap.add_argument("--specs", default="", help="comma-separated conditions, e.g. docs:1,docs:2,years:1")
    ap.add_argument("--repetitions", default="", help="comma-separated tags of the repeated runs, e.g. rep1,rep2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"case": a.case, "baseline_tag": a.baseline_tag}
    if a.specs:
        res["conditions"] = report(a.case, [x for x in a.specs.split(",") if x], a.baseline_tag)
    if a.repetitions:
        res["repetitions"] = variability(a.case, [x for x in a.repetitions.split(",") if x], baseline_tag=a.baseline_tag)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
