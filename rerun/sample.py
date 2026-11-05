"""Stratified sample for the human validation (spec 7.1, ``docs/codebook.md``) and the annotation sheets.

* Candidates: the judged attributes of the cases (``candidates_from_stages``). Outcome classes of the majority
  vote: ``dangerous`` (P = C = R = 1, E = 0), ``explicit`` (E = 1), ``rejected`` (E = 0 and not dangerous),
  ``undecided`` (a criterion undecided: not sampled, the human labels would have nothing to be compared with).
* The sample has ``per_case`` attributes per case, drawn round-robin over the (step x outcome) cells so that steps
  and outcomes are balanced; a small or empty cell is filled from the others; ``max_per_person`` keeps one person
  from dominating. Deterministic for a seed.
* Annotation files: one CSV sheet per annotator with the columns of the codebook (item id, case, step, attribute,
  category, evidence, documents, then the labels to fill), one text file per item with the step text the model saw,
  a calibration subset shared by the annotators (same items, different order) and a key file with the judges'
  labels and the person, which the annotators never see. Everything is written under a private directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CRITERIA = ("plausible", "coherent", "realistic", "explicit")
OUTCOMES = ("dangerous", "explicit", "rejected")
COLUMNS = ["item_id", "case", "step", "attribute", "category", "evidence", "documents", "plausible", "coherent",
           "realistic", "explicit", "supported", "special_category", "notes", "annotator", "seconds"]


@dataclass
class Candidate:
    case: str
    person: str
    step: int
    attr_id: str
    text: str
    category: str
    evidence: str
    majority: str            # four characters P C R E: 1 = yes, 0 = no, u = undecided
    votes: dict = field(default_factory=dict)


def outcome(bits: str) -> str:
    if "u" in bits:
        return "undecided"
    if bits == "1110":
        return "dangerous"
    return "explicit" if bits[3] == "1" else "rejected"


def candidates_from_stages(case: str, infer_rows, judge_rows) -> list[Candidate]:
    jud = {j["attr_id"]: j for j in judge_rows}
    out = []
    for r in infer_rows:
        if r["carried"]:
            continue
        for a in r["attributes"]:
            j = jud.get(a["id"])
            if j is None:
                continue
            bits = "".join({"yes": "1", "no": "0"}.get(j["majority"][c], "u") for c in CRITERIA)
            out.append(Candidate(case, r["person"], r["step"], a["id"], a["text"], a.get("category_hint", ""),
                                 a.get("evidence", ""), bits, j["votes"]))
    return out


def draw_sample(cands, per_case: int, seed: int = 42, max_per_person: int | None = None) -> list[Candidate]:
    rnd = random.Random(seed)
    cells: dict[tuple, list] = defaultdict(list)
    for c in sorted(cands, key=lambda c: c.attr_id):
        o = outcome(c.majority)
        if o != "undecided":
            cells[(c.step, o)].append(c)
    for v in cells.values():
        rnd.shuffle(v)
    taken, per_person = [], defaultdict(int)
    progress, rnd_no = True, 0
    while len(taken) < per_case and progress:
        progress = False
        # the cells are visited step by step inside each outcome, and the outcome order rotates from round to round,
        # so that the cells left over by the last (partial) round spread over the steps
        order = sorted(cells, key=lambda k: ((OUTCOMES.index(k[1]) + rnd_no) % len(OUTCOMES), k[0]))
        rnd_no += 1
        for key in order:
            if len(taken) >= per_case:
                break
            pool = cells[key]
            while pool:
                c = pool.pop()
                if max_per_person and per_person[c.person] >= max_per_person:
                    continue
                per_person[c.person] += 1
                taken.append(c)
                progress = True
                break
    return taken


def draw_sample_all_cases(cands, per_case: int, seed: int = 42, **kw) -> list[Candidate]:
    by = defaultdict(list)
    for c in cands:
        by[c.case].append(c)
    out = []
    for case in sorted(by):
        out += draw_sample(by[case], per_case, seed + sum(map(ord, case)), **kw)
    return out


def item_id(c: Candidate, salt: str) -> str:
    return "I-" + hashlib.sha256(f"{salt}|{c.case}|{c.attr_id}".encode("utf-8")).hexdigest()[:10]


def write_annotation_files(sample, out_dir, *, salt: str, annotators=("A1", "A2"), calibration: int = 40,
                           step_text, seed: int = 42) -> dict:
    """Sheets, documents, key and calibration list under ``out_dir`` (private: it holds personal text)."""
    out = Path(out_dir)
    (out / "documents").mkdir(parents=True, exist_ok=True)
    items = [(item_id(c, salt), c) for c in sample]
    for iid, c in items:
        (out / "documents" / f"{iid}.txt").write_text(step_text(c), encoding="utf-8")
    rnd = random.Random(seed)
    cal_ids = [iid for iid, _ in rnd.sample(items, min(calibration, len(items)))]
    paths = {"documents_dir": out / "documents"}
    for ann in annotators:
        r = random.Random(f"{seed}-{ann}")
        cal = [it for it in items if it[0] in cal_ids]
        rest = [it for it in items if it[0] not in cal_ids]
        r.shuffle(cal)
        r.shuffle(rest)
        path = out / f"annotation_{ann}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for iid, c in cal + rest:
                w.writerow({"item_id": iid, "case": c.case, "step": c.step, "attribute": c.text,
                            "category": c.category, "evidence": c.evidence, "documents": f"documents/{iid}.txt",
                            "annotator": ann})
        paths[f"sheet_{ann}"] = path
    with open(out / "key.jsonl", "w", encoding="utf-8") as f:
        for iid, c in items:
            f.write(json.dumps({"item_id": iid, "attr_id": c.attr_id, "person": c.person, "case": c.case,
                                "step": c.step, "majority": c.majority, "votes": c.votes}, ensure_ascii=False) + "\n")
    with open(out / "calibration.jsonl", "w", encoding="utf-8") as f:
        for iid in cal_ids:
            f.write(json.dumps({"item_id": iid}) + "\n")
    paths["key"], paths["calibration"] = out / "key.jsonl", out / "calibration.jsonl"
    return paths
