"""Annotation of a stratified sample of attributes by three local LLM annotators (D99, 2026-10-29).

The annotators apply the four criteria of ``docs/codebook.md`` with the judges' prompt (``judge.build_prompt``), one
attribute per call and the same step text; they do not see the judges' votes. The agreement among them (Fleiss'
kappa, Cohen's kappa for each pair, Gwet's AC1) is a measure of how reproducible the criteria are for small models.

Two parts, so that the second can run on a machine that has neither the documents nor the corpus loader:
* ``build_items`` (needs the persons and the inference file of a case): draws the sample and writes self-contained
  items, one per attribute (step text, attribute, the other attributes of the step as context), with opaque ids;
* ``run_items`` (needs only the items file and a local Ollama server): one annotator over all the items, resumable.

Sampling: for each step (25/50/75/100 %) the same number of attributes, drawn at random with a fixed seed from the
steps that were run and fit in one window of ``config.ANNOTATION_MAX_PROMPT_TOKENS`` tokens. Attributes of longer or
windowed steps cannot be read by a small local model and are not sampled; their number is reported.
The items files contain personal data: they stay under ``data/private`` or on an authorised machine.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

from . import config, judge, stats

CRITERIA = judge.CRITERIA


def item_id(case: str, attr_id: str) -> str:
    return hashlib.sha256(f"item:{case}:{attr_id}".encode("utf-8")).hexdigest()[:12]


def _read_jsonl(path: Path) -> list:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_items(case: str, persons, infer_rows, step_texts_fn, *, n: int = config.ANNOTATION_ITEMS_PER_CASE,
                seed: int = config.SEED, max_prompt_tokens: int = config.ANNOTATION_MAX_PROMPT_TOKENS,
                cluster: int = 1):
    """-> (items, report). ``step_texts_fn(person, step_k)`` gives the windows of the step text.

    ``cluster``: attributes drawn from the same step text (1: every attribute independently). With more than one, the
    sampled steps are drawn at random and up to ``cluster`` of their attributes are taken, so that an annotator reads
    each text once and the server reuses its prefix (the cost of the run falls by about the same factor); the items
    of a text are then correlated, which the analysis has to take into account (cluster bootstrap by step)."""
    persons = {p.id: p for p in persons}
    rows = [r for r in infer_rows if not r["carried"] and r["attributes"]]
    strata = sorted({r["step"] for r in rows})
    cache: dict = {}

    def texts_of(r):
        key = (r["person"], r["step"])
        if key not in cache:
            cache[key] = step_texts_fn(persons[r["person"]], r["step"])
        return cache[key]

    def fits(r) -> bool:
        t = texts_of(r)
        return len(t) == 1 and len(t[0]) / config.TOKEN_CHARS_DEFAULT <= max_prompt_tokens

    quota = {k: n // len(strata) + (1 if i < n % len(strata) else 0) for i, k in enumerate(strata)} if strata else {}
    chosen: dict = {k: [] for k in strata}
    pools: dict = {}
    for k in strata:
        if cluster <= 1:
            cand = sorted(((r, a) for r in rows if r["step"] == k for a in r["attributes"]),
                          key=lambda ra: ra[1]["id"])
            random.Random(f"{seed}:{case}:{k}").shuffle(cand)
        else:
            rs = sorted((r for r in rows if r["step"] == k), key=lambda r: r["person"])
            rng = random.Random(f"{seed}:{case}:{k}:cluster")
            rng.shuffle(rs)
            cand = []
            for r in rs:
                attrs = sorted(r["attributes"], key=lambda a: a["id"])
                rng.shuffle(attrs)
                cand += [(r, a) for a in attrs[:cluster]]
        pools[k] = cand
    skipped = Counter()
    pos = {k: 0 for k in strata}

    def draw(k, want):
        while len(chosen[k]) < want and pos[k] < len(pools[k]):
            r, a = pools[k][pos[k]]
            pos[k] += 1
            if fits(r):
                chosen[k].append((r, a))
            else:
                skipped[k] += 1

    for k in strata:
        draw(k, quota[k])
    short = sum(quota[k] - len(chosen[k]) for k in strata)
    for k in strata:                       # a stratum without enough candidates: the others make up for it
        if short <= 0:
            break
        before = len(chosen[k])
        draw(k, quota[k] + short)
        short -= len(chosen[k]) - before
    items = []
    for k in strata:
        for r, a in chosen[k]:
            texts = texts_of(r)
            items.append({"item_id": item_id(case, a["id"]), "case": case, "step": k, "text": texts[0],
                          "attribute": a["text"],
                          "others": [x["text"] for x in r["attributes"] if x["id"] != a["id"]],
                          "_person": r["person"], "_attr_id": a["id"]})
    report = {"case": case, "items": len(items), "per_step": {str(k): len(chosen[k]) for k in strata},
              "quota": {str(k): quota[k] for k in strata},
              "skipped_too_long_or_windowed": {str(k): skipped[k] for k in strata},
              "seed": seed, "max_prompt_tokens": max_prompt_tokens, "cluster": cluster,
              "distinct_step_texts": len({(it["_person"], it["step"]) for it in items})}
    return items, report


def split_private(items: list) -> tuple[list, dict]:
    """Items to ship to another machine (no person or attribute id) and the private map item id -> origin."""
    shipped = [{k: v for k, v in it.items() if not k.startswith("_")} for it in items]
    origin = {it["item_id"]: {"person": it["_person"], "attr_id": it["_attr_id"]} for it in items}
    return shipped, origin


def run_items(items: list, model: str, client, out_path: Path, *, workers: int = 1, tracker=None,
              num_predict: int = config.ANNOTATION_NUM_PREDICT) -> dict:
    """One annotator over the items. Appends one line per item to ``out_path``; items with a valid vote already in it
    are skipped (resume). Items of the same step text are consecutive, so that the server reuses the prompt prefix."""
    from concurrent.futures import ThreadPoolExecutor
    out_path = Path(out_path)
    done = {r["item_id"] for r in _read_jsonl(out_path) if r["model"] == model and r["votes"] is not None}
    todo = sorted((it for it in items if it["item_id"] not in done),
                  key=lambda it: (hashlib.sha256(it["text"].encode("utf-8")).hexdigest(), it["item_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_c: Counter = Counter()
    t0 = time.time()
    lock = __import__("threading").Lock()

    def one(it):
        attr = judge.JudgeAttr(id=it["item_id"], text=it["attribute"])
        ctx = [attr] + [judge.JudgeAttr(id=f"o{i}", text=t) for i, t in enumerate(it["others"])]
        t1 = time.time()
        v = judge.judge_batch(client, model, [it["text"]], [attr], all_attrs=ctx, stats=stats_c,
                              num_ctx=config.ANNOTATION_NUM_CTX, num_predict=num_predict)[attr.id]
        votes = [getattr(v, c) for c in CRITERIA]
        valid = all(x in ("yes", "no") for x in votes)
        rec = {"item_id": it["item_id"], "model": model, "votes": votes if valid else None,
               "seconds": round(time.time() - t1, 1)}
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        if tracker is not None:
            tracker.advance()
        return valid

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(one, todo))
    return {"model": model, "items": len(items), "skipped_done": len(done), "run": len(todo),
            "invalid": results.count(False), "seconds_wall": round(time.time() - t0, 1),
            "judge_stats": dict(stats_c)}


def agreement(results: list, models=config.ANNOTATORS) -> dict:
    """Agreement of the annotators per criterion, on the items that every annotator answered with valid votes.
    -> {criterion: {n, fleiss_kappa, gwet_ac1, percent_all_equal, cohen_kappa: {"a|b": k}}}."""
    by: dict = {}
    for r in results:
        if r["votes"] is not None and r["model"] in models:
            by.setdefault(r["item_id"], {})[r["model"]] = r["votes"]
    units = [v for v in by.values() if all(m in v for m in models)]
    out = {"items_complete": len(units), "items_partial": len(by) - len(units), "criteria": {}}
    for i, c in enumerate(CRITERIA):
        rat = [[u[m][i] for m in models] for u in units]
        pairs = {f"{a}|{b}": stats.cohen_kappa([r[x] for r in rat], [r[y] for r in rat])
                 for (x, a) in enumerate(models) for (y, b) in enumerate(models) if x < y}
        out["criteria"][c] = {
            "n": len(rat), "fleiss_kappa": stats.fleiss_kappa(rat), "gwet_ac1": stats.gwet_ac1(rat, n_categories=2),
            "percent_all_equal": (sum(len(set(r)) == 1 for r in rat) / len(rat)) if rat else float("nan"),
            "cohen_kappa": pairs}
    return out


# ---- command line ----------------------------------------------------------------------------------------------------
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="draw the sample of a case (needs the corpus and the inference file)")
    b.add_argument("--case", required=True)
    b.add_argument("--stages", default=None, help="stage directory with infer.jsonl (default: the case's main one)")
    b.add_argument("--n", type=int, default=config.ANNOTATION_ITEMS_PER_CASE)
    b.add_argument("--cluster", type=int, default=1, help="attributes per step text (see build_items)")
    b.add_argument("--out-dir", default=None, help="default data/private/annotation/case<X>")
    r = sub.add_parser("run", help="one annotator over an items file (needs only a local Ollama server)")
    r.add_argument("--items", required=True)
    r.add_argument("--model", required=True, choices=list(config.ANNOTATORS))
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--limit", type=int, default=None)
    g = sub.add_parser("agreement", help="agreement of the annotators from the output files")
    g.add_argument("--results", nargs="+", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "build":
        from . import corpus, run_case
        out = Path(a.out_dir) if a.out_dir else config.PRIVATE / "annotation" / f"case{a.case}"
        out.mkdir(parents=True, exist_ok=True)
        persons = corpus.load_case(a.case, workers=2)
        d = Path(a.stages) if a.stages else run_case.stage_dir(a.case)
        usable = run_case.budget_usable_attacker()
        items, rep = build_items(a.case, persons, _read_jsonl(d / "infer.jsonl"),
                                 lambda p, k: run_case.step_texts(p, k, config.ATTACKER, usable), n=a.n,
                                 cluster=a.cluster)
        shipped, origin = split_private(items)
        (out / "items.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in shipped),
                                         encoding="utf-8")
        (out / "origin.json").write_text(json.dumps(origin), encoding="utf-8")
        (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep))
    elif a.cmd == "run":
        from .llm import LLMClient
        items = _read_jsonl(Path(a.items))[:a.limit]
        out = Path(a.out).resolve()
        # an answer cut at ANNOTATION_NUM_PREDICT is a small model repeating itself: invalid item at once, no doubling
        client = LLMClient([], cache_path=out.with_suffix(".cache.jsonl"), max_attempts=3, private_root=out.parent,
                           grow_truncated=False)
        print(json.dumps(run_items(items, a.model, client, Path(a.out), workers=a.workers)))
    else:
        res = [x for p in a.results for x in _read_jsonl(Path(p))]
        print(json.dumps(agreement(res), indent=1))


if __name__ == "__main__":
    main()
