"""Cost and time per profile (spec §7.7, X16): aggregates only, no person id and no text.

Three sources, all already written by the pipeline:

* ``stages*/infer.jsonl``: one row per step with ``tokens_in``, ``tokens_out`` and ``seconds`` of the attacker (carried
  steps cost nothing), so the cost per profile of the inference is exact;
* ``stages*/run_summary_*.json``: wall time of each stage (judge, dedup, metrics, categories, control);
* ``llm/usage*.jsonl``: every non-cached call with model, tokens and seconds. The log does not say which case a call
  belongs to (Cases C, E, F, G share one file), so it is summed per model and per place (cloud or local server), never
  attributed to a case.

``python -m rerun.costs [--private DIR] [--summaries DIR ...] [--out FILE]``: ``--summaries`` adds stage directories that
live on another machine (copied here); a stage found twice is taken from the last directory given."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import config

STAGE_FILES = {"infer": "run_summary_infer.json", "judge-local": "run_summary_judge-local.json",
               "judge": "run_summary_judge.json", "dedup": "run_summary_dedup.json",
               "metrics": "run_summary_metrics.json", "categories": "run_summary_categories.json",
               "control": "run_summary_control.json"}


def _jsonl(path: Path):
    if not Path(path).exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def infer_cost(infer_path: Path) -> dict:
    """Per-profile cost of the attacker from ``infer.jsonl`` (steps that made a call; carried steps are free)."""
    per = defaultdict(lambda: {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "steps": 0})
    for r in _jsonl(infer_path):
        if r.get("carried"):
            continue
        p = per[r["person"]]
        p["tokens_in"] += int(r.get("tokens_in") or 0)
        p["tokens_out"] += int(r.get("tokens_out") or 0)
        p["seconds"] += float(r.get("seconds") or 0.0)
        p["steps"] += 1
    n = len(per)
    if not n:
        return {"persons": 0}
    tot = {k: sum(p[k] for p in per.values()) for k in ("tokens_in", "tokens_out", "seconds", "steps")}
    secs = sorted(p["seconds"] for p in per.values())
    return {"persons": n, "steps_with_calls": tot["steps"], "tokens_in": tot["tokens_in"],
            "tokens_out": tot["tokens_out"], "call_seconds": round(tot["seconds"], 1),
            "per_profile": {"tokens_in": round(tot["tokens_in"] / n, 1), "tokens_out": round(tot["tokens_out"] / n, 1),
                            "call_seconds_mean": round(tot["seconds"] / n, 1),
                            "call_seconds_median": round(secs[n // 2], 1)}}


def stage_times(stage_dir: Path) -> dict:
    """``{stage: {seconds_wall, ...}}`` from the run summaries in a stage directory."""
    out = {}
    for stage, name in STAGE_FILES.items():
        f = Path(stage_dir) / name
        if not f.exists():
            continue
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if stage == "control":      # roles are not timed; the inference and the judging of the control are
            parts = [s[k]["seconds_wall"] for k in ("control_infer", "control_judge")
                     if isinstance(s.get(k), dict) and "seconds_wall" in s[k]]
            body = {"seconds_wall": round(sum(parts), 1)} if parts else {}
        else:
            body = s.get(stage) if isinstance(s.get(stage), dict) else {}
        if "seconds_wall" in body:
            out[stage] = {"seconds_wall": body["seconds_wall"], "file_mtime": int(f.stat().st_mtime)}
    return out


def usage_by_model(usage_files) -> dict:
    """Non-cached calls per model: calls, ok calls, tokens, seconds, and whether the model runs on the local server."""
    acc = defaultdict(lambda: {"calls": 0, "ok": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0})
    for f in usage_files:
        for r in _jsonl(f):
            if r.get("cached"):
                continue
            a = acc[r.get("model", "?")]
            a["calls"] += 1
            a["ok"] += bool(r.get("ok"))
            a["tokens_in"] += int(r.get("tokens_in") or 0)
            a["tokens_out"] += int(r.get("tokens_out") or 0)
            a["seconds"] += float(r.get("seconds") or 0.0)
    return {m: {**a, "seconds": round(a["seconds"], 1), "place": "local" if config.is_local(m) else "cloud"}
            for m, a in sorted(acc.items())}


def collect(private: Path = config.PRIVATE, extra_stage_dirs=()) -> dict:
    private = Path(private)
    cases = {}
    for cdir in sorted(private.glob("case[A-Z]")):
        c = cdir.name[-1]
        entry = {"inference": infer_cost(cdir / "stages" / "infer.jsonl"), "stages": {}}
        for sd in [cdir / "stages", *sorted(cdir.glob("stages_*"))]:
            for stage, t in stage_times(sd).items():
                entry["stages"].setdefault(stage, {})[sd.name] = t
        cases[c] = entry
    for sd in extra_stage_dirs:
        sd = Path(sd)
        c = sd.parent.name[-1]
        cases.setdefault(c, {"inference": {"persons": 0}, "stages": {}})
        for stage, t in stage_times(sd).items():
            cases[c]["stages"].setdefault(stage, {})[f"{sd.name}@extra"] = t
    return {"cases": cases,
            "models": usage_by_model(sorted((private / "llm").glob("usage*.jsonl"))),
            "notes": ["per-profile inference cost is exact (infer.jsonl); stage wall times are as recorded in the run "
                      "summaries (parallel calls: wall time is not the sum of call seconds)",
                      "model totals come from the usage logs, which do not say which case a call belongs to"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--private", default=str(config.PRIVATE))
    ap.add_argument("--summaries", nargs="*", default=[], help="extra stage directories (copied from another machine)")
    ap.add_argument("--out", default=str(config.RESULTS / "costs.json"))
    a = ap.parse_args(argv)
    res = collect(Path(a.private), a.summaries)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written {out}: {len(res['cases'])} cases, {len(res['models'])} models")


if __name__ == "__main__":
    main()
