"""Assembly of the RQ3/RQ4 corpus from the generated records (plan ``docs/plans/rq34.md`` §8 step 3).

Input: the persons that went into the generation (``person``, ``case``) and the records of ``phish_gen.run``.
Output: one row per valid message with person, case, variant, attack, kind (positive or negative), generator and
text, written once to ``data/private/phish/`` with a manifest that keeps the SHA-256 of the file and counts only.

The check that matters (plan §8): every input person appears in the corpus. A person whose calls all failed is not
dropped silently; ``assemble`` raises ``IncompleteCorpus`` with the number of missing persons unless ``strict`` is
off, and the report always lists, per case, how many persons are complete (every expected cell present), partial
(some cell missing) and absent. A record about a person who was not in the input, a repeated record id or a record
of another generator is an error, never ignored. Rows hold personal text and stay under ``data/private/``; the
report and the manifest hold counts and hashes."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from . import config
from .phish_gen import ATTACKS, VARIANTS

OUT_DIR = config.PRIVATE / "phish"
MANIFEST = "manifest.json"
CORPUS_VERSION = 1
FIELDS = ("id", "person", "case", "variant", "attack", "kind", "index", "generator", "label", "subject", "body")


class IncompleteCorpus(ValueError):
    """Some input persons have no valid message in the corpus."""


def expected_cells(negatives_per_positive: int) -> set[tuple[str, str, str, int]]:
    """The cells (variant, attack, kind, index) every person should have."""
    cells = set()
    for v in VARIANTS:
        for a in ATTACKS:
            cells.add((v, a, "positive", 0))
            for i in range(negatives_per_positive):
                cells.add((v, a, "negative", i))
    return cells


def assemble(persons: list[dict], records: list[dict], generator: str, *, negatives_per_positive: int = 1,
             strict: bool = True) -> dict:
    """``{"rows": [...], "report": {...}, "absent": [person ids], "partial": [person ids]}``.

    Failed records (``ok`` false) are counted by reason and never become rows."""
    input_ids = [p["person"] for p in persons]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("the input persons contain a repeated identifier")
    case_of = {p["person"]: p["case"] for p in persons}
    seen_ids: set[str] = set()
    rows: list[dict] = []
    failed: Counter = Counter()
    cells_by_person: dict[str, set] = defaultdict(set)
    for r in records:
        if r.get("generator") != generator:
            raise ValueError("a record belongs to another generator")
        if r["person"] not in case_of:
            raise ValueError("a record refers to a person who is not in the input")
        if r["case"] != case_of[r["person"]]:
            raise ValueError("a record has a case different from the one of its person")
        if r["id"] in seen_ids:
            raise ValueError("a record identifier is repeated")
        seen_ids.add(r["id"])
        if not r.get("ok"):
            failed[r.get("error", "unknown")] += 1
            continue
        rows.append({k: r[k] for k in FIELDS})
        cells_by_person[r["person"]].add((r["variant"], r["attack"], r["kind"], r["index"]))
    need = expected_cells(negatives_per_positive)
    absent = [p for p in input_ids if not cells_by_person.get(p)]
    partial = [p for p in input_ids if cells_by_person.get(p) and cells_by_person[p] != need]
    per_case: dict[str, dict] = {}
    for p in input_ids:
        c = per_case.setdefault(case_of[p], {"persons": 0, "complete": 0, "partial": 0, "absent": 0})
        c["persons"] += 1
        c["absent" if p in set(absent) else "partial" if p in set(partial) else "complete"] += 1
    report = {"generator": generator, "n_persons": len(input_ids), "n_absent": len(absent),
              "n_partial": len(partial), "n_complete": len(input_ids) - len(absent) - len(partial),
              "n_rows": len(rows), "n_failed": sum(failed.values()), "failed_by_reason": dict(failed),
              "n_positive": sum(r["label"] for r in rows), "n_negative": sum(1 - r["label"] for r in rows),
              "rows_by_variant": dict(Counter(r["variant"] for r in rows)), "per_case": per_case,
              "negatives_per_positive": negatives_per_positive}
    if absent and strict:
        raise IncompleteCorpus(f"{len(absent)} of {len(input_ids)} persons have no valid message")
    return {"rows": rows, "report": report, "absent": absent, "partial": partial}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(result: dict, name: str, out_dir: str | os.PathLike | None = None) -> dict:
    """Write ``<name>.jsonl`` and update the manifest. Returns the manifest entry (counts and hash)."""
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}.jsonl"
    tmp = target.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in result["rows"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, target)
    entry = {"corpus_version": CORPUS_VERSION, "sha256": _sha256(target), "report": result["report"]}
    mpath = out / MANIFEST
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    manifest[name] = entry
    mtmp = out / (MANIFEST + ".tmp")
    mtmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(mtmp, mpath)
    return entry


def load(name: str, out_dir: str | os.PathLike | None = None) -> list[dict]:
    """The rows of a written corpus; fails if the file does not match the manifest."""
    out = Path(out_dir) if out_dir else OUT_DIR
    mpath = out / MANIFEST
    entry = json.loads(mpath.read_text(encoding="utf-8")).get(name) if mpath.exists() else None
    path = out / f"{name}.jsonl"
    if not entry or not path.exists():
        raise FileNotFoundError(f"corpus {name!r} not written")
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"corpus {name!r}: the file does not match the manifest")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summary(report: dict) -> str:
    """Counts only."""
    lines = [f"generator {report['generator']}: persons {report['n_persons']} (complete {report['n_complete']}, "
             f"partial {report['n_partial']}, absent {report['n_absent']}), rows {report['n_rows']} "
             f"(positive {report['n_positive']}, negative {report['n_negative']}), failed {report['n_failed']}"]
    for case, c in sorted(report["per_case"].items()):
        lines.append(f"  {case}: persons {c['persons']}, complete {c['complete']}, partial {c['partial']}, "
                     f"absent {c['absent']}")
    return "\n".join(lines)
