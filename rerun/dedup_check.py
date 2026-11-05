"""Sample of merged pairs for the check by eye of the de-duplication (spec 4.5: merge precision on 100 pairs).

For every cluster of the confirmed de-duplication with at least two members, one pair of members is drawn at random;
the pairs of all cases are pooled and ``n`` of them are drawn, stratified by case in proportion to the number of
merged clusters with at least one pair per case that has any (deterministic for the seed). The reviewer reads the two
statements and answers ``same`` (yes / no): precision of the merge = share of "yes". The pairs hold personal text, so
the sheet and the key are written under ``data/private/`` only.

    python -m rerun.dedup_check --cases A,B,C --tag gloc --n 100 --out data/private/dedup_check
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from . import config

COLUMNS = ["pair_id", "case", "statement_a", "statement_b", "step_a", "step_b", "same", "notes"]


def _read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def merged_pairs(case: str, infer_rows, dedup_rows, seed: int = config.SEED) -> list:
    """One random pair of members for each cluster of size >= 2: dicts with the texts, steps and ids."""
    info = {}
    for r in infer_rows:
        for a in r["attributes"]:
            info[a["id"]] = (r["person"], r["step"], a["text"])
    rnd = random.Random(f"{seed}-{case}")
    out = []
    for row in dedup_rows:
        ids = [i for i in row["attr_ids"] if i in info]
        if len(ids) < 2:
            continue
        a, b = rnd.sample(ids, 2)
        out.append({"case": case, "person": info[a][0], "attr_a": a, "attr_b": b, "step_a": info[a][1],
                    "step_b": info[b][1], "statement_a": info[a][2], "statement_b": info[b][2],
                    "cluster_size": len(ids)})
    return out


def draw(pairs_by_case: dict, n: int = 100, seed: int = config.SEED) -> list:
    """``n`` pairs, shares proportional to each case's pairs (largest remainder), every non-empty case at least one."""
    total = sum(len(v) for v in pairs_by_case.values())
    if total <= n:
        return [p for c in sorted(pairs_by_case) for p in pairs_by_case[c]]
    cases = [c for c in sorted(pairs_by_case) if pairs_by_case[c]]
    quota = {c: max(1, int(n * len(pairs_by_case[c]) / total)) for c in cases}
    rest = n - sum(quota.values())
    order = sorted(cases, key=lambda c: -(n * len(pairs_by_case[c]) / total - int(n * len(pairs_by_case[c]) / total)))
    i = 0
    while rest != 0 and order:
        c = order[i % len(order)]
        if rest > 0 and quota[c] < len(pairs_by_case[c]):
            quota[c] += 1
            rest -= 1
        elif rest < 0 and quota[c] > 1:
            quota[c] -= 1
            rest += 1
        i += 1
        if i > 10 * len(order) * (abs(rest) + 1) + 1000:
            break
    out = []
    for c in cases:
        rnd = random.Random(f"{seed}-draw-{c}")
        out += rnd.sample(pairs_by_case[c], min(quota[c], len(pairs_by_case[c])))
    return out


def pair_id(p: dict, salt: str) -> str:
    return "P-" + hashlib.sha256(f"{salt}|{p['case']}|{p['attr_a']}|{p['attr_b']}".encode("utf-8")).hexdigest()[:10]


def write_sheet(sample: list, out_dir, salt: str, seed: int = config.SEED) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = [(pair_id(p, salt), p) for p in sample]
    random.Random(f"{seed}-order").shuffle(items)
    sheet, key = out / "dedup_check.csv", out / "dedup_check_key.jsonl"
    with open(sheet, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for pid, p in items:
            w.writerow({"pair_id": pid, "case": p["case"], "statement_a": p["statement_a"],
                        "statement_b": p["statement_b"], "step_a": p["step_a"], "step_b": p["step_b"]})
    with open(key, "w", encoding="utf-8") as f:
        for pid, p in items:
            f.write(json.dumps({"pair_id": pid, "case": p["case"], "person": p["person"], "attr_a": p["attr_a"],
                                "attr_b": p["attr_b"], "cluster_size": p["cluster_size"]}) + "\n")
    return {"sheet": sheet, "key": key, "n": len(items)}


def precision(sheet_csv) -> dict:
    """Merge precision from a filled sheet: share of "yes" among the answered pairs, with the per-case counts."""
    rows = list(csv.DictReader(open(sheet_csv, encoding="utf-8", newline="")))
    ans = [(r["case"], r["same"].strip().casefold()) for r in rows if r["same"].strip().casefold() in ("yes", "no")]
    out = {"n_pairs": len(rows), "n_answered": len(ans),
           "precision": (sum(1 for _, a in ans if a == "yes") / len(ans)) if ans else None, "by_case": {}}
    for c in sorted({c for c, _ in ans}):
        v = [a for cc, a in ans if cc == c]
        out["by_case"][c] = {"n": len(v), "precision": sum(1 for a in v if a == "yes") / len(v)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="comma-separated, e.g. A,B,C")
    ap.add_argument("--tag", default=None, help="stage directory suffix (stages_<tag>); default: stages")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=str(config.PRIVATE / "dedup_check"))
    ap.add_argument("--base", default=str(config.PRIVATE))
    ap.add_argument("--precision-of", default=None, help="a filled sheet: print the merge precision and stop")
    a = ap.parse_args(argv)
    if a.precision_of:
        print(json.dumps(precision(a.precision_of), indent=1))
        return
    base = Path(a.base)
    if not base.resolve().is_relative_to(config.PRIVATE.resolve()) or not Path(a.out).resolve().is_relative_to(
            config.PRIVATE.resolve()):
        raise SystemExit("the pairs hold personal text: base and out must be under data/private/")
    salt = (config.PRIVATE / "id_salt.txt").read_text(encoding="utf-8").strip() if (
        config.PRIVATE / "id_salt.txt").exists() else "dedup-check"
    by_case = {}
    for c in a.cases.split(","):
        d = base / f"case{c}" / (f"stages_{a.tag}" if a.tag else "stages")
        by_case[c] = merged_pairs(c, _read_jsonl(d / "infer.jsonl"), _read_jsonl(d / "dedup.jsonl"))
    sample = draw(by_case, a.n)
    res = write_sheet(sample, a.out, salt)
    print(json.dumps({"pairs_available": {c: len(v) for c, v in by_case.items()}, "drawn": res["n"],
                      "sheet": str(res["sheet"])}, indent=1))


if __name__ == "__main__":
    main()
