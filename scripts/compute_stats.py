#!/usr/bin/env python3
"""Rebuild results/ from the validation outputs of the private data.

For each case it loads the three per-judge audit files, consolidates them by
majority vote, and computes the global, per-step and per-judge metrics plus
the cumulative PES curve and the PES distribution. It also reads the
detection CSVs (classical / Transformer / LLM) if present. Only aggregate
numbers are written; no per-person rows and no names.

Expected private layout (configurable with --audit-root), per case <X>:

    <audit-root>/<X>/judges/<model>.json      # >=1 per-judge audit file
                     (user -> step -> attribute -> {plausible,coherent,
                                                    realistic,explicit,...})
    <audit-root>/<X>/detection/generated_classical.csv   # optional
    <audit-root>/<X>/detection/benchmark_<name>_classical.csv
    <audit-root>/<X>/detection/generated_llm.csv

Case F uses a single-step (100 %) auditor file with a list-of-attributes
schema; pass it as <X>/judges/*.json too and the loader adapts.

PES counting condition is selectable with --pes-condition
{as_reported, paper_text} (see src/metrics.py). The published tables use
'as_reported'. The default here is 'paper_text' (the definition written in
the paper); the authors will confirm which is canonical.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import metrics as M  # noqa: E402
from src.majority_vote import build_majority  # noqa: E402

CASES = ("A", "B", "C", "D", "E", "F", "G", "H")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


# Map percentage-style step labels (Case F) to the canonical step_N keys.
_STEP_ALIASES = {"25%": "step_1", "50%": "step_2", "75%": "step_3", "100%": "step_4"}


def _adapt_schema(data: dict) -> dict:
    """Normalise to user -> step_key -> attribute -> labels.

    Handles the Case F schema where each step maps to a LIST of
    {attributo, plausible, ...} records instead of a dict keyed by attribute,
    and where the step label is a percentage ("100%") rather than "step_4".
    """
    out: dict = {}
    for user, steps in data.items():
        out[user] = {}
        for step_key, attrs in steps.items():
            step_key = _STEP_ALIASES.get(step_key, step_key)
            if isinstance(attrs, list):
                out[user][step_key] = {
                    (rec.get("attributo") or rec.get("attribute") or f"attr_{i}"): rec
                    for i, rec in enumerate(attrs)
                }
            else:
                out[user][step_key] = attrs
    return out


PES_CONDITIONS = {
    "as_reported": M.PES_CONDITION_AS_REPORTED,
    "paper_text": M.PES_CONDITION_PAPER_TEXT,
}


def _pes_block(majority: dict) -> dict:
    """Cumulative PES curve and final-step distribution under both conditions."""
    block: dict = {}
    for name, cond in PES_CONDITIONS.items():
        cum = {s: [] for s in M.STEP_KEYS}
        dist: list[int] = []
        for steps in majority.values():
            curve = M.cumulative_pes(steps, cond)
            for i, s in enumerate(M.STEP_KEYS):
                if s in steps:
                    cum[s].append(curve[i])
            if any(s in steps for s in M.STEP_KEYS):
                dist.append(curve[-1])
        block[name] = {
            "cumulative_mean_per_step": {
                s: round(statistics.mean(cum[s]), 4) if cum[s] else None for s in M.STEP_KEYS
            },
            "final_distribution": (
                {"mean": round(statistics.mean(dist), 3),
                 "median": statistics.median(dist), "max": max(dist)}
                if dist else None
            ),
        }
    return block


def aggregate_case(judge_files: list[Path]) -> dict:
    """Majority-vote metrics for one case from its per-judge audit files."""
    judges = {p.stem: _adapt_schema(_load_json(p)) for p in judge_files}
    majority = build_majority(judges) if len(judges) > 1 else _adapt_schema(
        _load_json(judge_files[0])
    )

    flat: list[dict] = []
    per_step: dict[str, list[dict]] = {s: [] for s in M.STEP_KEYS}
    for steps in majority.values():
        for s in M.STEP_KEYS:
            evs = list(steps.get(s, {}).values())
            per_step[s].extend(evs)
            flat.extend(evs)

    g = M.global_metrics(flat)
    g["users"] = len(majority)
    per_step_out = {s: M.global_metrics(per_step[s]) for s in M.STEP_KEYS}
    per_judge = {}
    for name, jdata in judges.items():
        jflat = [e for steps in jdata.values() for attrs in steps.values() for e in attrs.values()]
        per_judge[name] = M.global_metrics(jflat)
    return {
        "global": g,
        "per_step": per_step_out,
        "per_judge": per_judge,
        "pes": _pes_block(majority),
    }


def _read_f1_csv(path: Path) -> dict[str, float]:
    """Read a {Model,...,F1,...} CSV into {model: F1}."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            model = (row.get("Model") or row.get("model") or "").strip()
            f1 = row.get("F1") or row.get("f1")
            if model and f1 not in (None, ""):
                try:
                    out[model] = round(float(f1), 4)
                except ValueError:
                    pass
    return out


def read_detection(case_dir: Path) -> dict:
    """Read the canonical detection CSVs for one case, if present.

    Expected files under <case>/detection/:
        generated_classical.csv         classical models on the generated corpus
        benchmark_CEAS_08_classical.csv  classical models on CEAS_08
        generated_llm.csv                LLMs (zero/few-shot) on the generated corpus
    """
    d = case_dir / "detection"
    det: dict = {}
    gen = _read_f1_csv(d / "generated_classical.csv")
    ceas = _read_f1_csv(d / "benchmark_CEAS_08_classical.csv")
    llm = _read_f1_csv(d / "generated_llm.csv")
    if gen:
        det["generated_classical_f1"] = gen
    if ceas:
        det["benchmark_ceas08_classical_f1"] = ceas
    if llm:
        det["generated_llm_f1"] = llm
    return det


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-root", default="data/private/audit",
                    help="root of the per-case validation outputs (private)")
    ap.add_argument("--out", default="results", help="output directory")
    args = ap.parse_args()

    root = Path(args.audit_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_res: dict[str, dict] = {}
    for case in CASES:
        jdir = root / case / "judges"
        files = sorted(jdir.glob("*.json")) if jdir.exists() else []
        if not files:
            print(f"[skip] Case {case}: no audit files under {jdir}")
            continue
        res = aggregate_case(files)
        det = read_detection(root / case)
        if det:
            res["detection"] = det
        (out / f"case{case}_metrics.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8"
        )
        all_res[case] = res
        g = res["global"]
        print(f"[ok]  Case {case}: {g['users']} profiles, {g['n_attributes']} attributes")

    _write_tidy_csvs(out, all_res)
    print("\nPES recorded under both conditions (as_reported, paper_text); "
          "the paper's definition (paper_text) is canonical. See caseX_metrics.json.")
    return 0


def _write_tidy_csvs(out: Path, all_res: dict[str, dict]) -> None:
    """Cross-case tidy CSVs for quick reading and for make_figures.py."""
    # RQ1 global
    with (out / "rq1_global.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "profiles", "attributes", "plausibility", "coherence",
                    "realism", "explicitness", "iqs", "dir", "hallucination"])
        for case, res in all_res.items():
            g = res["global"]
            w.writerow([case, g["users"], g["n_attributes"], g["plausibility"],
                        g["coherence"], g["realism"], g["explicitness"], g["iqs"],
                        g["dir"], g["hallucination"]])
    # RQ1 per-step (DIR, hallucination, IQS, explicitness)
    with (out / "rq1_per_step.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "step", "n_attributes", "dir", "hallucination", "iqs", "explicitness"])
        for case, res in all_res.items():
            for i, s in enumerate(M.STEP_KEYS, 1):
                ps = res["per_step"].get(s, {})
                if ps.get("n_attributes"):
                    w.writerow([case, i, ps["n_attributes"], ps["dir"], ps["hallucination"],
                                ps["iqs"], ps["explicitness"]])
    # PES per step (both conditions)
    with (out / "rq1_pes_per_step.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "step", "pes_paper_text", "pes_as_reported"])
        for case, res in all_res.items():
            pt = res["pes"]["paper_text"]["cumulative_mean_per_step"]
            ar = res["pes"]["as_reported"]["cumulative_mean_per_step"]
            for i, s in enumerate(M.STEP_KEYS, 1):
                if pt.get(s) is not None or ar.get(s) is not None:
                    w.writerow([case, i, pt.get(s), ar.get(s)])
    # RQ2 per-judge
    with (out / "rq2_judges.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "judge", "plausibility", "coherence", "realism",
                    "explicitness", "iqs", "dir", "hallucination"])
        for case, res in all_res.items():
            for judge, m in res["per_judge"].items():
                w.writerow([case, judge, m["plausibility"], m["coherence"], m["realism"],
                            m["explicitness"], m["iqs"], m["dir"], m["hallucination"]])
    # RQ4 detection: generated classical, CEAS benchmark, LLM
    def _detrows(key: str, fname: str, cols: tuple[str, str]) -> None:
        rows = []
        for case, res in all_res.items():
            det = res.get("detection", {}).get(key, {})
            for model, f1 in det.items():
                rows.append([case, model, f1])
        if rows:
            with (out / fname).open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["case", cols[0], cols[1]])
                w.writerows(rows)

    _detrows("generated_classical_f1", "rq4_generated_classical_f1.csv", ("model", "f1"))
    _detrows("benchmark_ceas08_classical_f1", "rq4_benchmark_ceas08_f1.csv", ("model", "f1"))
    _detrows("generated_llm_f1", "rq4_generated_llm_f1.csv", ("model_mode", "f1"))


if __name__ == "__main__":
    raise SystemExit(main())
