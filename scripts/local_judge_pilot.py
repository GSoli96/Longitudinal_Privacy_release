"""Pilot of the local judge: n attributes of Case C judged by the local model and compared with the cloud votes.

The sample (seed 42) comes from the Case C judge records, where gemma4, nemotron-3-ultra and minimax all voted. The
local judge sees the same prompt as the cloud judges (step text, other attributes, one attribute per call). Reports,
per criterion, agreement / Cohen's kappa / Gwet's AC1 of the local judge with each cloud judge and with the majority,
the cloud pairs on the same sample as reference, and the seconds per call. Votes go to data/private (ignored by git),
the summary (statistics only) to results/rerun/caseC/.

    .venv\\Scripts\\python.exe scripts\\local_judge_pilot.py --n 100
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rerun import config, corpus, judge, progress, stats  # noqa: E402
from rerun.llm import LLMClient, fetch_digests  # noqa: E402
from rerun.run_case import (_read_jsonl, budget_usable_attacker, sample_attributes, stage_dir,  # noqa: E402
                            step_texts)

CLOUD = ("gemma4:31b", "nemotron-3-ultra", "minimax-m3.1-flash")


def compare(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x in ("yes", "no") and y in ("yes", "no")]
    return {"n": len(pairs), "agreement": round(stats.percent_agreement(a, b), 3),
            "kappa": round(stats.cohen_kappa(a, b), 3),
            "ac1": round(stats.gwet_ac1([[x, y] for x, y in pairs], n_categories=2), 3) if pairs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.LOCAL_JUDGE)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--by-step", action="store_true", help="sample whole steps (prefix reuse)")
    ap.add_argument("--ctx", type=int, default=262144, help="context of a model other than config.LOCAL_JUDGE")
    ap.add_argument("--think", action="store_true", help="let the local judge reason before answering")
    a = ap.parse_args()
    if a.model != config.LOCAL_JUDGE:
        # another local candidate (e.g. qwen3.6:35b, pilot of 2026-10-23): same settings as the local judge
        config.LOCAL_MODELS = (*config.LOCAL_MODELS, a.model)
        config.JUDGE_THINK[a.model] = False
        config.NUM_CTX[a.model] = a.ctx
        config.JUDGE_BATCH_SIZE[a.model] = 1

    if a.think:
        config.JUDGE_THINK[a.model] = True

    d = stage_dir("C")
    rows_all = _read_jsonl(d / "judge.jsonl")
    if a.by_step:
        # whole steps in random order (seed 42) until n attributes: the attributes of a step are judged one after the
        # other, as in the full run, so the server can reuse the step text it has just read
        import random
        from collections import defaultdict
        steps_ = defaultdict(list)
        for r in sorted(rows_all, key=lambda r: r["attr_id"]):
            steps_[(r["person"], r["step"])].append(r)
        order = sorted(steps_)
        random.Random(42).shuffle(order)
        sample = []
        for k in order:
            if len(sample) >= a.n:
                break
            sample.extend(steps_[k])
    else:
        sample = sample_attributes(rows_all, a.n, seed=42)
    by_step = {(r["person"], r["step"]): r for r in _read_jsonl(d / "infer.jsonl") if not r["carried"]}
    persons = {p.id: p for p in corpus.load_case("C", workers=4)}
    usable = budget_usable_attacker()
    tag = a.model.replace(":", "_") + ("_think" if a.think else "")
    client = LLMClient([], cache_path=config.PRIVATE / "llm" / "local_judge_cache.jsonl",
                       usage_path=config.PRIVATE / "llm" / "local_judge_usage.jsonl",
                       digests=fetch_digests([a.model]), max_concurrency=1)
    out_path = d / f"local_judge_pilot_{tag}.jsonl"
    rows, secs = [], []
    with progress.Tracker(f"caseC-pilota-{tag}", "local judge", len(sample), unit="attributes", client=client,
                          next_step="comparison with the cloud judges") as tr:
        for i, j in enumerate(sample, 1):
            r = by_step[(j["person"], j["step"])]
            texts = step_texts(persons[j["person"]], j["step"], config.ATTACKER, usable)
            allattrs = [judge.JudgeAttr(id=x["id"], text=x["text"], evidence=x["evidence"], window=x["window"])
                        for x in r["attributes"]]
            me = next(x for x in allattrs if x.id == j["attr_id"])
            t0 = time.time()
            v = judge.judge_one(client, a.model, texts, me, context_attrs=allattrs)
            secs.append(time.time() - t0)
            rows.append({"attr_id": j["attr_id"], "cloud": j["votes"], "majority": j["majority"],
                         "local": [getattr(v, c) for c in judge.CRITERIA], "seconds": round(secs[-1], 1)})
            tr.advance()
            print(f"{i}/{len(sample)}  {secs[-1]:.0f} s  mean {sum(secs) / len(secs):.0f} s", flush=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {"model": a.model, "n": len(rows), "think": config.JUDGE_THINK.get(a.model),
               "invalid_local": sum(r["local"][0] == "invalid" for r in rows),
               "seconds_mean": round(sum(secs) / len(secs), 1), "seconds_median": round(sorted(secs)[len(secs) // 2], 1),
               "seconds_max": round(max(secs), 1), "by_criterion": {}}
    for i, c in enumerate(judge.CRITERIA):
        loc = [r["local"][i] for r in rows]
        res = {"vs_majority": compare(loc, [r["majority"][c] for r in rows])}
        for m in CLOUD:
            res[f"vs_{m}"] = compare(loc, [r["cloud"][m][i] for r in rows])
        for x, y in (("gemma4:31b", "nemotron-3-ultra"), ("gemma4:31b", "minimax-m3.1-flash"),
                     ("nemotron-3-ultra", "minimax-m3.1-flash")):
            res[f"ref_{x}_vs_{y}"] = compare([r["cloud"][x][i] for r in rows], [r["cloud"][y][i] for r in rows])
        res["yes_local"] = loc.count("yes")
        res["yes_majority"] = sum(r["majority"][c] == "yes" for r in rows)
        summary["by_criterion"][c] = res
    # attributes on which the local judge and gemma agree on all four criteria (the third judge is not needed there)
    summary["full_agreement_with"] = {m: round(sum(r["local"] == r["cloud"][m] for r in rows) / len(rows), 3)
                                      for m in CLOUD}
    # majority of gemma, ultra and the local judge vs the majority of gemma, ultra and minimax
    maj = [[("yes" if [r["cloud"]["gemma4:31b"][i], r["cloud"]["nemotron-3-ultra"][i], r["local"][i]].count("yes") >= 2
             else "no") for i in range(4)] for r in rows]
    summary["new_majority_equal"] = {c: round(sum(m[i] == r["majority"][c] for m, r in zip(maj, rows)) / len(rows), 3)
                                     for i, c in enumerate(judge.CRITERIA)}
    res_path = config.RESULTS / "caseC" / f"local_judge_pilot_{tag}.json"
    res_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
