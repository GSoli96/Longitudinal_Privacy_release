"""Votes of the local judge with the small context vs the full context on the same pilot attributes (Case C).

Compares data/private/caseC/stages/local_judge_pilot_<tag>.jsonl (current run) with the copy made before the change
(..._ctx262k.jsonl); writes results/rerun/caseC/local_judge_ctx_check.json (statistics only).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rerun import config, judge  # noqa: E402
from rerun.run_case import _read_jsonl, stage_dir  # noqa: E402


def main():
    d = stage_dir("C")
    tag = config.LOCAL_JUDGE.replace(":", "_")
    old = {r["attr_id"]: r for r in _read_jsonl(d / f"local_judge_pilot_{tag}_ctx262k.jsonl")}
    new = {r["attr_id"]: r for r in _read_jsonl(d / f"local_judge_pilot_{tag}.jsonl")}
    ids = sorted(set(old) & set(new))
    same_all = sum(old[i]["local"] == new[i]["local"] for i in ids)
    per = {c: sum(old[i]["local"][k] == new[i]["local"][k] for i in ids) for k, c in enumerate(judge.CRITERIA)}
    sec_old = sum(old[i]["seconds"] for i in ids) / len(ids)
    sec_new = sum(new[i]["seconds"] for i in ids) / len(ids)
    out = {"n": len(ids), "identical_all_criteria": same_all, "identical_by_criterion": per,
           "seconds_mean_262k": round(sec_old, 1), "seconds_mean_small": round(sec_new, 1),
           "small_context": config.JUDGE_CTX_SMALL.get(config.LOCAL_JUDGE)}
    (config.RESULTS / "caseC" / "local_judge_ctx_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
