"""Runner of one case: stages infer -> judge -> dedup -> metrics (plan T7+).

    py -3.11 -m rerun.run_case --case C --stage all --limit 3

* Every stage writes a JSONL file under ``data/private/case<X>/stages/`` (personal text: never committed) and
  returns a summary; the metrics stage writes aggregates only (no ids, no text) under ``results/rerun/``.
* Resume: all model calls go through ``LLMClient`` whose cache is the checkpoint; running a stage again after an
  interruption repeats no completed call (tested).
* Invariants are checked after each stage (persons in = persons out, one record per step, attributes judged =
  attributes inferred, same windows for the judges and the attacker) and the automatic alarms of the spec
  (invalid or failed outputs > 2 %, key errors > 5 %) are reported in the summary.
* A step identical to the previous one is carried forward, not re-run (``carried``); in the metrics the
  person keeps the results of the previous step.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import budget, config, corpus, dedup, infer, judge, metrics, progress, steps, stats, uniqueness


class Interrupted(RuntimeError):
    """Raised by the interruption hook (testing the resume)."""


def stage_dir(case: str, base=config.PRIVATE, tag: str | None = None) -> Path:
    d = Path(base) / f"case{case}" / ("stages" if not tag else f"stages_{tag}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_suffix(case: str, tag: str | None = None) -> str:
    """Suffix of the cache, usage and events files: the run's own for a tag in ``config.TAG_CACHE_SUFFIX`` (size study),
    else empty for the cases that share one cache and ``_case<X>`` for the others."""
    if tag in config.TAG_CACHE_SUFFIX:
        return config.TAG_CACHE_SUFFIX[tag]
    return "" if case in config.SHARED_CACHE_CASES else f"_case{case}"


def select_persons(persons, limit: int | None = None):
    """Deterministic selection: persons in id order, first ``limit``."""
    ps = sorted(persons, key=lambda p: p.id)
    return ps[:limit] if limit else ps


def sample_persons(persons, n: int, seed: int = config.PERSON_SAMPLE_SEED):
    """``n`` persons drawn at random (``seed``) from the persons in id order, returned in id order."""
    import random
    ps = sorted(persons, key=lambda p: p.id)
    return ps if len(ps) <= n else sorted(random.Random(seed).sample(ps, n), key=lambda p: p.id)


def attr_id(person_id: str, k: int, n: int) -> str:
    return f"{person_id}-s{k}-a{n:03d}"


def _write_jsonl(path: Path, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


class InterruptingClient:
    """Wraps a client and raises ``Interrupted`` after ``n`` calls that really went to the network (testing hook)."""

    def __init__(self, client, n: int):
        self._c, self._n = client, n

    def __getattr__(self, name):
        return getattr(self._c, name)

    def chat(self, *a, **k):
        before = self._c.stats["calls"]
        r = self._c.chat(*a, **k)
        if self._c.stats["calls"] > before:
            self._n -= self._c.stats["calls"] - before
            if self._n <= 0:
                raise Interrupted("interrupted by the test hook")
        return r


def _alarms(failed: int, total: int, client, *, kind: str = "failed") -> dict:
    """Rates against the thresholds of the spec. ``kind`` names the first rate: ``failed`` (inference steps) or
    ``invalid`` (judge votes). D58: a rate counts only with at least ``ALARM_MIN_CALLS`` observations, so a stage
    read back from the cache (a few network calls) never raises an alarm."""
    rate = failed / total if total else 0.0
    kr = client.key_error_rate() if hasattr(client, "key_error_rate") else 0.0
    calls = getattr(client, "stats", {}).get("calls", 0)
    return {f"{kind}_rate": round(rate, 4),
            f"{kind}_alarm": total >= config.ALARM_MIN_CALLS and rate > config.ALARM_INVALID_RATE,
            "key_error_rate": round(kr, 4),
            "key_error_alarm": calls >= config.ALARM_MIN_CALLS and kr > config.ALARM_KEY_ERROR_RATE}


def active_alarms(summary: dict) -> list[str]:
    """Names of the alarms a stage summary raises (D58): every ``*_alarm`` that is true, plus invariants that fail
    (a person or an attribute lost between stages) and the problems found by ``check_infer_invariants``."""
    out = [k for k, v in (summary.get("alarms") or {}).items() if k.endswith("_alarm") and v is True]
    if summary.get("invariants"):
        out.append("invariants")
    if summary.get("problems"):
        out.append("problems")
    return out


def persons_lost(previous_rows, next_rows) -> list[str]:
    """Persons that have rows in the previous stage and none in the next one (D58)."""
    have = {r["person"] for r in next_rows}
    return sorted({r["person"] for r in previous_rows} - have)


# ---- stage 1: inference --------------------------------------------------------------------------------------------
def copy_infer_subset(src: Path, dst: Path, person_ids) -> int:
    """Copy the inference records of the given persons (the same attributes, for a judge-only comparison)."""
    keep = set(person_ids)
    rows = [r for r in _read_jsonl(src) if r["person"] in keep]
    _write_jsonl(dst, rows)
    return len(rows)


def run_infer(persons, client, model: str, out_path: Path, *, usable_tokens: int, workers: int = 4,
              variant: str = infer.DEFAULT_VARIANT, tracker=None) -> dict:
    persons = list(persons)
    t0 = time.time()

    def one(p):
        try:
            return _one(p)
        finally:
            progress.tick(tracker)

    def _one(p):
        results = infer.infer_person(client, p, model, usable_tokens=usable_tokens, variant=variant)
        rows, prev_k = [], None
        for r in results:
            k = r.step.k
            row = {"person": p.id, "step": k, "variant": variant, "carried": r.carried, "n_docs": r.step.n_docs,
                   "chars": r.step.chars,
                   "dated_docs": r.step.dated_docs, "undated_docs": r.step.undated_docs, "ok": r.ok,
                   "error": r.error, "windows": r.windows, "chunked": r.chunked, "partial": r.partial,
                   "failed_windows": r.failed_windows, "structure_retries": r.structure_retries,
                   "empty_retries": r.empty_retries,
                   "evidence_exact": r.evidence_exact, "evidence_fuzzy": r.evidence_fuzzy,
                   "evidence_none": r.evidence_not_found, "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
                   "seconds": round(r.seconds, 2), "carried_from": prev_k if r.carried else None,
                   "attributes": [] if r.carried else
                   [{"id": attr_id(p.id, k, n), "text": a.text, "category_hint": a.category_hint,
                     "evidence": a.evidence, "window": a.window, "windows": list(a.windows),
                     "evidence_in_text": a.evidence_in_text, "evidence_match": a.evidence_match,
                     "evidence_ratio": a.evidence_ratio} for n, a in enumerate(r.attributes, 1)]}
            rows.append(row)
            prev_k = k
        return rows

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        per_person = list(ex.map(one, persons))
    rows = [r for rs in per_person for r in rs]
    _write_jsonl(out_path, rows)
    return summarize_infer(persons, rows, client, time.time() - t0)


def summarize_infer(persons, rows, client, seconds=0.0) -> dict:
    run = [r for r in rows if not r["carried"]]
    with_text = [p for p in rows if True]
    failed = sum(1 for r in run if not r["ok"])
    empty = sum(1 for r in run if r["ok"] and not r["attributes"])
    # empty lists: at the first attempt (retried or not) and still empty after the retry, per step
    empty_first = {k: sum(1 for r in run if r["step"] == k and r["ok"] and (r.get("empty_retries", 0) > 0 or not r["attributes"]))
                   for k in metrics.STEPS}
    empty_final = {k: sum(1 for r in run if r["step"] == k and r["ok"] and not r["attributes"]) for k in metrics.STEPS}
    s = {"persons_in": len(persons), "persons_out": len({r["person"] for r in rows}),
         "persons_with_steps": len({r["person"] for r in rows}),
         "steps_total": len(rows), "steps_run": len(run), "steps_carried": len(rows) - len(run),
         "steps_failed": failed, "steps_empty": empty, "empty_first_attempt_by_step": empty_first,
         "empty_final_by_step": empty_final, "steps_partial": sum(1 for r in run if r["partial"]),
         "steps_chunked": sum(1 for r in run if r["chunked"]), "windows_total": sum(r["windows"] for r in run),
         "attributes": sum(len(r["attributes"]) for r in run),
         "structure_retries": sum(r["structure_retries"] for r in run),
         "empty_retries": sum(r.get("empty_retries", 0) for r in run),
         "evidence_exact": sum(r["evidence_exact"] for r in run), "evidence_fuzzy": sum(r["evidence_fuzzy"] for r in run),
         "evidence_none": sum(r["evidence_none"] for r in run),
         "tokens_in": sum(r["tokens_in"] for r in run), "tokens_out": sum(r["tokens_out"] for r in run),
         "seconds_wall": round(seconds, 1), "client_stats": dict(getattr(client, "stats", {}))}
    s["alarms"] = _alarms(failed, len(run), client)
    return s


def check_infer_invariants(persons, rows) -> list[str]:
    """Problems found (empty list = fine): every person with text has one record per step, ids unique."""
    problems = []
    by = defaultdict(list)
    for r in rows:
        by[r["person"]].append(r["step"])
    for p in persons:
        expected = [s.k for s in steps.build_steps(p)]
        if sorted(by.get(p.id, [])) != sorted(expected):
            problems.append(f"{p.id}: steps {sorted(by.get(p.id, []))} != {sorted(expected)}")
    ids = [a["id"] for r in rows for a in r["attributes"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate attribute ids")
    return problems


# ---- stage 2: judges -------------------------------------------------------------------------------------------------
def step_texts(person, k: int, model: str, usable_tokens: int):
    """The windows of the step text for the judges: the same as the attacker's (see ``infer.split_step``)."""
    st = next(s for s in steps.build_steps(person) if s.k == k)
    return infer.split_step(st.text, usable_tokens, model)


def judge_rows_to_run(infer_path: Path) -> list:
    """Steps the judges see: run (not carried forward) and with at least one attribute."""
    return [r for r in _read_jsonl(infer_path) if not r["carried"] and r["attributes"]]


def run_judge(persons, infer_path: Path, client, judges, out_path: Path, *, attacker: str, usable_tokens: int,
              workers: int = 4, max_chars: int | None = None, texts_for=None, tracker=None,
              sequential: bool = False, full_panel: bool = False) -> dict:
    """``max_chars``: the judges read only the first ``max_chars`` characters of the step text (the old condition of
    Cases B, E, G, H, A used 2,000); the same prompt and judges otherwise.
    ``sequential``: ``judges`` is (first, second, tie-breaker) and the tie-breaker votes only where the first two
    differ, on the audit sample, or everywhere with ``full_panel`` (``judge.judge_step_sequential``); each record
    then has a ``panel`` field."""
    persons = {p.id: p for p in persons}
    rows = judge_rows_to_run(infer_path)
    t0 = time.time()
    stats_c = Counter()

    def one(r):
        try:
            return _one(r)
        finally:
            progress.tick(tracker)

    def _one(r):
        p = persons[r["person"]]
        if texts_for is not None:   # e.g. the control without documents: the judges read that text instead
            texts = texts_for(r["person"], r)
        else:
            texts = step_texts(p, r["step"], attacker, usable_tokens)
            assert len(texts) == r["windows"], "judge windows differ from the attacker's"
        if max_chars:
            texts = [texts[0][:max_chars]]
        attrs = [judge.JudgeAttr(id=a["id"], text=a["text"], evidence=a["evidence"], window=a["window"])
                 for a in r["attributes"]]
        if sequential:
            js = list(judges)
            res = judge.judge_step_sequential(client, js[:2], js[2], texts, attrs, full_panel=full_panel,
                                              stats=stats_c)
        else:
            res = judge.judge_step(client, judges, texts, attrs, stats=stats_c)
        return [{"person": r["person"], "step": r["step"], "attr_id": a.id,
                 "votes": {j: [getattr(v, c) for c in judge.CRITERIA] for j, v in res[a.id]["votes"].items()},
                 "majority": res[a.id]["majority"], **({"panel": res[a.id]["panel"]} if sequential else {})}
                for a in attrs]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        out = [x for rs in ex.map(one, rows) for x in rs]
    _write_jsonl(out_path, out)
    n_attr = sum(len(r["attributes"]) for r in rows)
    undec = sum(1 for o in out if any(v == "undecided" for v in o["majority"].values()))
    s = {"attributes_in": n_attr, "attributes_judged": len(out), "steps_judged": len(rows),
         "judge_text_chars": max_chars,
         "undecided_attributes": undec, "judge_stats": dict(stats_c), "seconds_wall": round(time.time() - t0, 1),
         "fenced": {m: n for m, n in getattr(client, "stats", {}).items() if str(m).startswith("fenced:")},
         "truncated": {m: n for m, n in getattr(client, "stats", {}).items() if str(m).startswith("truncated:")}}
    if sequential:
        s["panel"] = dict(Counter(o["panel"] for o in out))
    s["alarms"] = _alarms(stats_c["invalid_votes"], max(sum(len(o["votes"]) for o in out), 1), client, kind="invalid")
    calls_j = stats_c.get("calls", 0)
    s["alarms"].update({
        "failed_call_rate": round(stats_c.get("failed_calls", 0) / max(calls_j, 1), 4),
        "failed_call_alarm": calls_j >= config.ALARM_MIN_CALLS
        and stats_c.get("failed_calls", 0) / calls_j > config.ALARM_FAILED_CALL_RATE,
        "undecided_rate": round(undec / max(n_attr, 1), 4),
        "undecided_alarm": n_attr >= config.ALARM_MIN_CALLS and undec / n_attr > config.ALARM_UNDECIDED_RATE})
    s["invariants"] = [] if len(out) == n_attr else [f"judged {len(out)} != inferred {n_attr}"]
    return s


def run_judge_local(persons, infer_path: Path, client, model: str, *, usable_tokens: int,
                    attacker: str = config.ATTACKER, tracker=None) -> dict:
    """The local judge alone on every step, one step after the other: its votes go to the client cache, so that the
    judge stage later finds them there and only the cloud judges wait for the quota (no output file)."""
    persons = {p.id: p for p in persons}
    stats_c = Counter()
    t0 = time.time()
    rows = judge_rows_to_run(infer_path)
    texts_of = {id(r): step_texts(persons[r["person"]], r["step"], attacker, usable_tokens) for r in rows}
    # steps whose text fits the small context first: the server reloads the model when the context changes
    rows.sort(key=lambda r: max(len(t) for t in texts_of[id(r)]))
    for r in rows:
        texts = texts_of[id(r)]
        attrs = [judge.JudgeAttr(id=a["id"], text=a["text"], evidence=a["evidence"], window=a["window"])
                 for a in r["attributes"]]
        judge.judge_batch(client, model, texts, attrs, stats=stats_c)
        progress.tick(tracker)
    return {"model": model, "judge_stats": dict(stats_c), "seconds_wall": round(time.time() - t0, 1)}


# ---- batched vs one-by-one judging (pilot) ---------------------------------------------------------------------------
def sample_attributes(judge_rows, n: int, seed: int = 42) -> list:
    """A deterministic sample of ``n`` judged attributes (all of them if fewer)."""
    import random
    rows = sorted(judge_rows, key=lambda r: r["attr_id"])
    return rows if len(rows) <= n else random.Random(seed).sample(rows, n)


def run_batch_vs_single(persons, infer_path: Path, judge_path: Path, client, judges, out_path: Path, *, n: int = 300,
                        seed: int = 42, workers: int = 4, attacker: str = config.ATTACKER,
                        usable_tokens: int | None = None, threshold: float = 0.80) -> dict:
    """Judge ``n`` attributes one at a time (same text, same other attributes as in the batch) and compare with the
    votes of the batched run, per judge and criterion: percent agreement, Cohen's kappa, Gwet's AC1. Batching is
    accepted if the kappa of every judge and criterion is at least ``threshold`` (a kappa that is undefined because
    both sides are constant counts as accepted when the agreement is perfect)."""
    persons = {p.id: p for p in persons}
    usable_tokens = usable_tokens or budget_usable_attacker()
    infer_rows = _read_jsonl(infer_path)
    sample = sample_attributes(_read_jsonl(judge_path), n, seed)
    by_step = {(r["person"], r["step"]): r for r in infer_rows if not r["carried"]}
    judges = list(judges)
    t0 = time.time()

    def one(j):
        r = by_step[(j["person"], j["step"])]
        texts = step_texts(persons[j["person"]], j["step"], attacker, usable_tokens)
        allattrs = [judge.JudgeAttr(id=a["id"], text=a["text"], evidence=a["evidence"], window=a["window"])
                    for a in r["attributes"]]
        me = next(a for a in allattrs if a.id == j["attr_id"])
        votes = {}
        for m in judges:
            v = judge.judge_one(client, m, texts, me, context_attrs=allattrs)
            votes[m] = [getattr(v, c) for c in judge.CRITERIA]
        return {"attr_id": j["attr_id"], "batch": j["votes"], "single": votes}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = list(ex.map(one, sample))
    _write_jsonl(out_path, rows)
    by_judge, failing = {}, defaultdict(list)
    for m in judges:
        by_judge[m] = {}
        for i, c in enumerate(judge.CRITERIA):
            a = [r["batch"][m][i] if r["batch"][m][i] in ("yes", "no") else None for r in rows]
            b = [r["single"][m][i] if r["single"][m][i] in ("yes", "no") else None for r in rows]
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            kappa = stats.cohen_kappa(a, b)
            pa = stats.percent_agreement(a, b)
            res = {"n_valid_pairs": len(pairs), "percent_agreement": pa, "cohen_kappa": kappa,
                   "gwet_ac1": stats.gwet_ac1([[x, y] for x, y in pairs], n_categories=2) if pairs else float("nan"),
                   "yes_batch": sum(1 for x, _ in pairs if x == "yes"), "yes_single": sum(1 for _, y in pairs if y == "yes")}
            by_judge[m][c] = res
            ok = (kappa >= threshold) if not math.isnan(kappa) else (pa == 1.0)
            if not ok:
                failing[m].append(c)
    return {"n_attributes": len(rows), "by_judge": by_judge, "failing": dict(failing),
            "batching_accepted": not failing, "threshold": threshold, "seconds_wall": round(time.time() - t0, 1)}


def _agree(a, b) -> dict:
    pairs = [(x, y) for x, y in zip(a, b) if x in ("yes", "no") and y in ("yes", "no")]
    return {"n_valid_pairs": len(pairs), "percent_agreement": stats.percent_agreement(a, b),
            "cohen_kappa": stats.cohen_kappa(a, b),
            "gwet_ac1": stats.gwet_ac1([[x, y] for x, y in pairs], n_categories=2) if pairs else float("nan")}


def run_judge_retest(persons, infer_path: Path, judge_path: Path, bvs_path: Path, client, judges, out_path: Path, *,
                     n: int = 100, seed_offset: int = 100, workers: int = 4, attacker: str = config.ATTACKER,
                     usable_tokens: int | None = None) -> dict:
    """Noise floor of the judges: the same attributes judged again with another seed, in a batch and one at a time.
    Compares batch vs batch and single vs single (the first run of the single votes is in ``bvs_path``), so that the
    disagreement between batched and single judging can be read against the run-to-run noise."""
    persons = {p.id: p for p in persons}
    usable_tokens = usable_tokens or budget_usable_attacker()
    infer_rows = _read_jsonl(infer_path)
    first = _read_jsonl(bvs_path)[:n]
    by_step = {(r["person"], r["step"]): r for r in infer_rows if not r["carried"]}
    judge_by_id = {j["attr_id"]: j for j in _read_jsonl(judge_path)}
    judges = list(judges)
    t0 = time.time()

    def one(row):
        j = judge_by_id[row["attr_id"]]
        r = by_step[(j["person"], j["step"])]
        texts = step_texts(persons[j["person"]], j["step"], attacker, usable_tokens)
        allattrs = [judge.JudgeAttr(id=a["id"], text=a["text"], evidence=a["evidence"], window=a["window"])
                    for a in r["attributes"]]
        me = next(a for a in allattrs if a.id == j["attr_id"])
        single2, batch2 = {}, {}
        for m in judges:
            v = judge.judge_one(client, m, texts, me, context_attrs=allattrs, seed_offset=seed_offset)
            single2[m] = [getattr(v, c) for c in judge.CRITERIA]
            vb = judge.judge_batch(client, m, texts, allattrs, batch_size=judge.BATCH_SIZE,
                                   seed_offset=seed_offset)[me.id]
            batch2[m] = [getattr(vb, c) for c in judge.CRITERIA]
        return {"attr_id": row["attr_id"], "batch1": row["batch"], "single1": row["single"],
                "batch2": batch2, "single2": single2}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = list(ex.map(one, first))
    _write_jsonl(out_path, rows)
    by_judge = {}
    for m in judges:
        by_judge[m] = {}
        for i, c in enumerate(judge.CRITERIA):
            col = lambda key: [r[key][m][i] for r in rows]
            by_judge[m][c] = {"batch_vs_batch": _agree(col("batch1"), col("batch2")),
                              "single_vs_single": _agree(col("single1"), col("single2")),
                              "batch1_vs_single2": _agree(col("batch1"), col("single2")),
                              "batch2_vs_single1": _agree(col("batch2"), col("single1"))}
    return {"n_attributes": len(rows), "seed_offset": seed_offset, "by_judge": by_judge,
            "seconds_wall": round(time.time() - t0, 1)}


def run_batch_size_test(persons, infer_path: Path, judge_path: Path, bvs_path: Path, client, model: str,
                        out_path: Path, *, sizes=(10, 3), workers: int = 4, attacker: str = config.ATTACKER,
                        usable_tokens: int | None = None) -> dict:
    """One judge, the attributes of the batch-vs-single sample judged again in smaller batches (same text, same other
    attributes as context). A step with at most ``size`` attributes gives the same request as in the main run, so its
    votes come from the cache and are identical: only the larger steps change. Compared with the single votes."""
    persons = {p.id: p for p in persons}
    usable_tokens = usable_tokens or budget_usable_attacker()
    infer_rows = _read_jsonl(infer_path)
    first = _read_jsonl(bvs_path)
    by_step = {(r["person"], r["step"]): r for r in infer_rows if not r["carried"]}
    judge_by_id = {j["attr_id"]: j for j in _read_jsonl(judge_path)}
    t0 = time.time()

    def one(row):
        j = judge_by_id[row["attr_id"]]
        r = by_step[(j["person"], j["step"])]
        texts = step_texts(persons[j["person"]], j["step"], attacker, usable_tokens)
        allattrs = [judge.JudgeAttr(id=a["id"], text=a["text"], evidence=a["evidence"], window=a["window"])
                    for a in r["attributes"]]
        out = {"attr_id": row["attr_id"], "step_size": len(allattrs), "batch": row["batch"][model],
               "single": row["single"][model]}
        for k in sizes:
            v = judge.judge_batch(client, model, texts, allattrs, batch_size=k)[row["attr_id"]]
            out[f"b{k}"] = [getattr(v, c) for c in judge.CRITERIA]
        return out

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = list(ex.map(one, first))
    _write_jsonl(out_path, rows)
    res = {}
    for k in sizes:
        res[f"b{k}"] = {}
        for i, c in enumerate(judge.CRITERIA):
            col = lambda key: [r[key][i] for r in rows]
            res[f"b{k}"][c] = {"vs_single": _agree(col(f"b{k}"), col("single")),
                               "batch40_vs_single": _agree(col("batch"), col("single")),
                               "yes_b": sum(1 for x in col(f"b{k}") if x == "yes"),
                               "yes_single": sum(1 for x in col("single") if x == "yes"),
                               "yes_batch40": sum(1 for x in col("batch") if x == "yes")}
        res[f"b{k}"]["n_attributes_with_different_request"] = sum(1 for r in rows if r["step_size"] > k)
    return {"model": model, "n_attributes": len(rows), "by_size": res, "seconds_wall": round(time.time() - t0, 1)}


def run_concurrency_calibration(persons, infer_path: Path, client, model: str, out_path: Path, *,
                                levels=(16, 24, 32), calls_per_level: int = 120, attacker: str = config.ATTACKER,
                                usable_tokens: int | None = None, seed: int = 42) -> dict:
    """Real judging work of one model at several levels of calls in flight (one-attribute calls of disjoint steps):
    for each level the calls per minute, the median latency and the rate-limit and server-error events. The votes are
    cached, so the calls are not lost for the main run."""
    import random
    persons = {p.id: p for p in persons}
    usable_tokens = usable_tokens or budget_usable_attacker()
    rows = [r for r in _read_jsonl(infer_path) if not r["carried"] and r["attributes"]]
    random.Random(seed).shuffle(rows)
    out = {"model": model, "calls_per_level": calls_per_level, "levels": {}}
    pos = 0
    for level in levels:
        chosen, n = [], 0
        while pos < len(rows) and n < calls_per_level:
            chosen.append(rows[pos])
            n += len(rows[pos]["attributes"])
            pos += 1
        client.set_model_concurrency(model, level)
        before = Counter(client.stats)
        lat, t0 = [], time.time()

        class Timed:
            def __init__(self, c):
                self._c = c

            def chat(self, *a, **kw):
                res = self._c.chat(*a, **kw)
                if not res.cached:
                    lat.append(res.seconds)
                return res

            def __getattr__(self, k):
                return getattr(self._c, k)

        timed = Timed(client)

        def one_timed(r):
            texts = step_texts(persons[r["person"]], r["step"], attacker, usable_tokens)
            attrs = [judge.JudgeAttr(id=a["id"], text=a["text"], evidence=a["evidence"], window=a["window"])
                     for a in r["attributes"]]
            judge.judge_batch(timed, model, texts, attrs, batch_size=1)

        with ThreadPoolExecutor(max_workers=max(2, level // 4)) as ex:
            list(ex.map(one_timed, chosen))
        wall = time.time() - t0
        after = Counter(client.stats)
        d = lambda k: after[k] - before[k]
        lat_sorted = sorted(lat)
        med = lat_sorted[len(lat_sorted) // 2] if lat_sorted else float("nan")
        out["levels"][str(level)] = {"steps": len(chosen), "attributes": n, "calls_made": len(lat), "seconds_wall": round(wall, 1),
                                     "calls_per_min": round(len(lat) / wall * 60, 2) if wall else float("nan"),
                                     "latency_median": round(med, 1),
                                     "latency_mean": round(sum(lat) / len(lat), 1) if lat else float("nan"),
                                     "latency_p90": round(lat_sorted[int(0.9 * len(lat_sorted))], 1) if lat_sorted else float("nan"),
                                     "http_429": d("http_429"), "http_5xx": d("http_5xx"), "failed": d("failed"),
                                     "rate_429_share": round(d("http_429") / max(d("calls"), 1), 4)}
    _write_jsonl(out_path, [{"level": k, **v} for k, v in out["levels"].items()])
    return out


# ---- records for the metrics -----------------------------------------------------------------------------------------
def build_records(infer_rows, judge_rows, judge_name: str | None = None) -> list:
    """AttrRecord list with carry-forward: a carried step repeats the records of the previous step (``carried``).

    ``judge_name`` uses the votes of that judge alone instead of the majority."""
    lab = {}
    for j in judge_rows:
        if judge_name is None:
            lab[j["attr_id"]] = j["majority"]
        elif judge_name in j["votes"]:
            v = j["votes"][judge_name]
            lab[j["attr_id"]] = {c: (x if x in ("yes", "no") else "undecided") for c, x in zip(judge.CRITERIA, v)}
    recs, last = [], {}
    for r in sorted(infer_rows, key=lambda r: (r["person"], r["step"])):
        if not r["carried"]:
            cur = [metrics.AttrRecord(r["person"], r["step"], a["text"], lab.get(a["id"], {}), False, a["id"])
                   for a in r["attributes"] if a["id"] in lab]
            last[r["person"]] = cur
            recs += cur
        else:
            recs += [metrics.AttrRecord(c.person, r["step"], c.text, c.labels, True, c.attr_id)
                     for c in last.get(r["person"], [])]
    return recs


def _require_votes(judge_rows, judge_path) -> None:
    """A missing or empty vote file would give empty results without an error: stop instead."""
    if not judge_rows:
        raise SystemExit(f"no judge votes in {judge_path} (file missing or empty): run the judge stage first")


def _judges_in(judge_rows) -> list:
    """Judges that actually voted, in the order of first appearance (names differ between runs: cloud or ``-local``)."""
    seen = {}
    for j in judge_rows:
        for m in j["votes"]:
            seen.setdefault(m, None)
    return list(seen)


# ---- stage 3: de-duplication -----------------------------------------------------------------------------------------
def run_dedup(infer_path: Path, judge_path: Path, client, embed, out_path: Path, pairs_path: Path | None = None, *,
              workers: int = 4, tracker=None, model: str = config.DEDUP_MODEL) -> dict:
    infer_rows, judge_rows = _read_jsonl(infer_path), _read_jsonl(judge_path)
    _require_votes(judge_rows, judge_path)
    recs = build_records(infer_rows, judge_rows)
    by = defaultdict(list)
    for r in recs:
        by[r.person].append(r)
    stats_c = Counter()
    t0 = time.time()

    if tracker is not None:
        tracker.total = len(by)

    def one(person):
        try:
            clusters, info = dedup.confirmed_clusters(by[person], embed, client, model=model, stats=stats_c)
            return person, clusters, info
        finally:
            progress.tick(tracker)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        res = list(ex.map(one, sorted(by)))
    ids = {}
    for r in infer_rows:
        for a in r["attributes"]:
            ids[(r["person"], r["step"], a["text"])] = a["id"]
    rows, tot = [], Counter()
    for person, clusters, info in res:
        for k, cl in enumerate(clusters):
            rows.append({"person": person, "cluster": k, "attr_ids": [ids.get((r.person, r.step, r.text)) for r in cl],
                         "steps": [r.step for r in cl]})
        tot.update({k: v for k, v in info.items() if k not in ("largest_cluster",)})
        tot["largest_cluster"] = max(tot["largest_cluster"], info["largest_cluster"])
    _write_jsonl(out_path, rows)
    return {"persons": len(res), "clusters": len(rows), **{k: tot[k] for k in
            ("n_records", "n_candidates", "n_confirmed", "n_rejected", "n_invalid", "clusters_gt3", "largest_cluster")},
            "dedup_stats": dict(stats_c), "seconds_wall": round(time.time() - t0, 1)}


def load_clusters(dedup_path: Path, infer_rows, judge_rows) -> dict:
    """{person: [cluster of AttrRecord]} from the dedup file (same record objects as ``build_records``)."""
    recs = build_records(infer_rows, judge_rows)
    by_key = {}
    for r in infer_rows:
        for a in r["attributes"]:
            by_key[a["id"]] = (r["person"], r["step"], a["text"])
    lookup = {}
    for rec in recs:
        if not rec.carried:
            lookup[(rec.person, rec.step, rec.text)] = rec
    out = defaultdict(list)
    for row in _read_jsonl(dedup_path):
        cl = [lookup[by_key[i]] for i in row["attr_ids"] if i in by_key and by_key[i] in lookup]
        if cl:
            out[row["person"]].append(cl)
    return out, recs


def exact_clusters(person_records) -> list:
    """Clusters of attributes with the same normalised text (the 'exact' variant, sensitivity)."""
    g = {}
    for r in sorted((x for x in person_records if not x.carried), key=lambda x: x.step):
        g.setdefault(metrics._norm(r.text), []).append(r)
    return list(g.values())


# ---- stage 4: metrics ------------------------------------------------------------------------------------------------
def run_metrics(persons, infer_path: Path, judge_path: Path, dedup_path: Path, out_dir: Path, embed=None,
                judges=()) -> dict:
    persons_all = list(persons)
    infer_rows, judge_rows = _read_jsonl(infer_path), _read_jsonl(judge_path)
    # D54: the cohort of the PES and of the intervals is the persons with at least one step (a readable document);
    # one with steps and no attributes counts 0, one without text stays out of the mean but is counted below
    with_steps = {r["person"] for r in infer_rows}
    persons = [p for p in persons_all if p.id in with_steps]
    pids = [p.id for p in persons]
    _require_votes(judge_rows, judge_path)
    judges = _judges_in(judge_rows) if judges else ()   # the names in the votes, not the configured ones
    clusters_by_person, recs = load_clusters(dedup_path, infer_rows, judge_rows)
    step_chars = {p.id: {s.k: s.chars for s in steps.build_steps(p)} for p in persons}
    for p in persons:
        sc = step_chars[p.id]
        for k in metrics.STEPS:   # identical steps: the text up to the step is the previous one's
            sc.setdefault(k, sc.get(max([x for x in sc if x < k], default=None), 0))
    out = {"persons": len(persons_all), "persons_without_text": len(persons_all) - len(persons),
           "persons_in_cohort": len(persons), "steps": {}}
    out["by_step"] = {str(k): v for k, v in metrics.step_metrics(recs).items()}
    final = [r for r in recs if r.step == 100]
    out["label_combinations_final"] = metrics.label_combinations(final)
    out["label_combinations_all_steps"] = metrics.label_combinations([r for r in recs if not r.carried])
    out["novelty_exact"] = {str(k): v for k, v in metrics.novelty_exact(recs).items()}
    out["novelty_confirmed"] = {str(k): v for k, v in
                                metrics.novelty_from_clusters(recs, clusters_by_person).items()}
    ex_cl = {p: exact_clusters([r for r in recs if r.person == p]) for p in pids}
    out["pes"] = {"confirmed": _pes(recs, pids, embed, step_chars, clusters_by_person),
                  "exact": _pes(recs, pids, embed, step_chars, ex_cl)}
    if embed is not None:
        out["novelty_semantic"] = {str(t): {str(k): v for k, v in d.items()}
                                   for t, d in metrics.novelty_semantic(recs, embed).items()}
        for thr in (0.80, 0.85, 0.90):
            out["pes"][f"cosine_{thr:.2f}"] = {str(k): v for k, v in
                                               metrics.cohort_pes(recs, pids, embed, step_chars, threshold=thr).items()}
    per_judge = {}
    for j in judges:
        per_judge[j] = {str(k): v for k, v in metrics.step_metrics(build_records(infer_rows, judge_rows, j)).items()}
    out["by_judge"] = per_judge
    out["agreement"] = _agreement(judge_rows, list(judges))
    out["ci_final_step"] = _cis(recs, pids)
    out["reidentification"] = uniqueness.analyse(recs, embed)
    out["trend"] = _trends(recs, pids)
    out["carry_forward"] = {str(k): sum(1 for r in infer_rows if r["step"] == k and r["carried"]) for k in metrics.STEPS}
    out["evidence"] = {"exact": sum(r["evidence_exact"] for r in infer_rows if not r["carried"]),
                       "fuzzy": sum(r["evidence_fuzzy"] for r in infer_rows if not r["carried"]),
                       "none": sum(r["evidence_none"] for r in infer_rows if not r["carried"])}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "metrics.json").write_text(metrics.dumps_strict(out, indent=1), encoding="utf-8")
    return out


def _pes(recs, pids, embed, step_chars, clusters_by_person):
    by = defaultdict(list)
    for r in recs:
        by[r.person].append(r)
    cohort = metrics.cohort_pes(recs, pids, embed, step_chars, clusters_by_person={p: clusters_by_person.get(p, []) for p in pids})
    return {str(k): v for k, v in cohort.items()}


def _agreement(judge_rows, judges) -> dict:
    out = {}
    if len(judges) < 2:
        return out
    for i, c in enumerate(judge.CRITERIA):
        units = []
        for j in judge_rows:
            if j.get("panel") == "tie" or any(m not in j["votes"] for m in judges):
                continue   # the tie-breaker's records are the disagreements: not a sample for agreement
            v = [j["votes"][m][i] for m in judges]
            units.append([x if x in ("yes", "no") else None for x in v])
        valid = [u for u in units if all(x is not None for x in u)]
        out[c] = {"n_units": len(units), "n_valid": len(valid),
                  "fleiss": stats.fleiss_kappa(valid), "krippendorff": stats.krippendorff_alpha_nominal(units),
                  "gwet_ac1": stats.gwet_ac1(valid, n_categories=2), "percent_all_agree": (
                      sum(1 for u in valid if len(set(u)) == 1) / len(valid) if valid else float("nan"))}
    return out


def _trends(recs, pids) -> dict:
    """Trend of the metrics across the four steps: mean within-person OLS slope with a within-person permutation
    p-value, and the random-intercept model (spec 4.3). Carried copies stay in, as in ``step_metrics``; a person
    with fewer than two defined steps is left out of the slope."""
    out = {}
    for name in ("DIR", "IQS", "hallucination"):
        vals = metrics.per_person_step_values(recs, name, pids)
        rows = [(p, i, v) for p, row in vals.items() for i, v in enumerate(row) if v is not None]
        out[name] = {"permutation": stats.permutation_trend_test(vals), "mixed_model": stats.mixed_model_trend(rows)}
    return out


def _cis(recs, pids) -> dict:
    final = defaultdict(list)
    for r in recs:
        if r.step == 100:
            final[r.person].append(r)
    units = [final[p] for p in pids if final.get(p)]
    out = {}
    for name in ("DIR", "IQS", "hallucination"):
        out[name] = stats.cluster_bootstrap_ci(units, lambda us, n=name: metrics.basic_metrics(
            [r for u in us for r in u])[n], n_resamples=config.BOOTSTRAP_RESAMPLES)
    return out


# ---- command line ----------------------------------------------------------------------------------------------------
def load_keys_if_any(load_keys, load_key_types, path=None):
    """Keys and key types, or two empty lists when the keys file is absent: a machine without ``.secrets`` runs the
    local models only, and a cloud model then fails as ``no_valid_key`` and is counted, never silently skipped."""
    path = Path(path) if path else config.KEYS_PATH
    if not path.exists():
        print("note: no keys file, local models only", file=sys.stderr)
        return [], []
    return load_keys(path), load_key_types(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--stage", choices=["infer", "judge", "dedup", "metrics", "all", "batch_vs_single", "categories",
                                        "category-metrics", "control", "retest", "batchsize", "calibrate", "judge-local"],
                    default="all")
    ap.add_argument("--panel-scheme", choices=["sequential", "full"], default="sequential",
                    help="sequential: config.PANEL with the tie-breaker only where needed (D41); full: the three "
                         "judges of config.JUDGES on every attribute (runs up to 2026-10-23)")
    ap.add_argument("--judge-model", default=config.JUDGES[1], help="judge of the batchsize stage")
    ap.add_argument("--sizes", default="10,3", help="batch sizes of the batchsize stage")
    ap.add_argument("--model-concurrency", default=None,
                    help="calls in flight per model, e.g. nemotron-3-ultra=24,gemma4:31b=16 (others unlimited)")
    ap.add_argument("--levels", default="16,24,32", help="calibrate stage: calls in flight to try")
    ap.add_argument("--calls-per-level", type=int, default=120)
    ap.add_argument("--judge-workers", type=int, default=None, help="steps judged side by side (default: --workers)")
    ap.add_argument("--bvs-n", type=int, default=300, help="attributes for the batched vs single comparison")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extract-workers", type=int, default=8,
                    help="processes for text extraction and OCR (main PC: about half of the cores)")
    ap.add_argument("--interrupt-after-calls", type=int, default=None, help="testing hook for the resume")
    ap.add_argument("--tag", default=None, help="separate stage directory (pilot comparisons)")
    ap.add_argument("--variant", choices=list(infer.VARIANTS), default=infer.DEFAULT_VARIANT)
    ap.add_argument("--judge-chars", type=int, default=None, help="judges read only the first N characters of the step")
    ap.add_argument("--dedup-model", default=config.DEDUP_MODEL,
                    help="model that confirms the de-duplication pairs (e.g. gemma4:31b-local when the cloud is closed)")
    ap.add_argument("--infer-from", default=None, help="tag of another run whose inference records are reused")
    ap.add_argument("--ignore-alarms", action="store_true",
                    help="keep going after a stage raised an alarm (D58: by default the run stops with exit code 2)")
    ap.add_argument("--attacker", default=None,
                    help="attacker model of the run (default config.ATTACKER); a name ending in -local runs on the "
                         "local server with its own cache key, e.g. gpt-oss:20b-local for Cases A, D and H")
    ap.add_argument("--ablation", default=None,
                    help="condition of an ablation or of a mitigation policy (docs:N, order:P, years:K, last:K, "
                         "generalise; see rerun.ablation); the tag defaults to the condition's own")
    ap.add_argument("--sample", type=int, default=None,
                    help="work on N persons drawn at random with the fixed seed (the same ones in every condition)")
    ap.add_argument("--final-only", action="store_true",
                    help="only the step with all the documents (ablations compare the final step: a quarter of the cost)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="added to the attacker's seed (repetitions of the inference; judges are not touched)")
    a = ap.parse_args(argv)
    if a.attacker:
        config.ATTACKER = a.attacker
    config.ATTACKER_SEED_OFFSET = a.seed_offset
    if a.final_only:
        steps.FRACTIONS = (100,)

    from . import embed as embed_mod
    from .llm import LLMClient, fetch_digests, load_key_types, load_keys

    persons = select_persons(corpus.load_case(a.case, workers=a.extract_workers,
                                              progress_run=f"case{a.case}" + (f"-{a.tag}" if a.tag else "") + "-0-extract"),
                             a.limit)
    condition = {}
    if a.sample:
        persons = sample_persons(persons, a.sample)
        condition["sample"] = {"n": a.sample, "seed": config.PERSON_SAMPLE_SEED}
    if a.ablation:
        from . import ablation as ablation_mod
        a.tag = a.tag or ablation_mod.tag_for(a.ablation)
        persons, condition["ablation"] = ablation_mod.apply(persons, a.ablation)
    if a.seed_offset:
        condition["attacker_seed_offset"] = a.seed_offset
    if a.final_only:
        condition["final_step_only"] = True
    keys, key_types = load_keys_if_any(load_keys, load_key_types)
    models = (config.ATTACKER, *config.JUDGES, *config.PANEL, a.dedup_model)
    # Cases C, E, F, G keep the shared cache of the first runs; the other cases have their own files, so that
    # two cases can run side by side without two processes appending to the same file (main PC, 2026-10-23)
    sfx = cache_suffix(a.case, a.tag)
    client = LLMClient(keys, cache_path=config.PRIVATE / "llm" / f"cache{sfx}.jsonl",
                       usage_path=config.PRIVATE / "llm" / f"usage{sfx}.jsonl", digests=fetch_digests(models, keys),
                       events_path=config.PRIVATE / "llm" / f"events{sfx}.jsonl",
                       key_types=key_types, paid_models=config.PAID_MODELS,
                       paid_key_concurrency=config.PAID_KEY_CONCURRENCY,
                       model_concurrency=dict((kv.rsplit("=", 1)[0], int(kv.rsplit("=", 1)[1]))
                                              for kv in a.model_concurrency.split(",")) if a.model_concurrency
                                       else dict(config.MODEL_CONCURRENCY))
    api = InterruptingClient(client, a.interrupt_after_calls) if a.interrupt_after_calls else client
    d = stage_dir(a.case, tag=a.tag)
    if condition:       # which persons and which transformation this stage directory holds (ids are pseudonymous)
        (d / "condition.json").write_text(json.dumps({**condition, "persons": [p.id for p in persons]}, indent=1),
                                          encoding="utf-8")
    usable = budget_usable_attacker()
    summaries = {}

    def write_summary():
        name = f"run_summary{'_' + a.stage if a.stage not in ('all',) else ''}.json"
        (d / name).write_text(metrics.dumps_strict(summaries, indent=1), encoding="utf-8")

    def stop_on_alarm(stage: str):
        """D58: an alarm in a stage stops the run before the next one, unless --ignore-alarms."""
        found = active_alarms(summaries.get(stage, {}))
        if found and not a.ignore_alarms:
            summaries["stopped_by_alarm"] = {"stage": stage, "alarms": found}
            summaries["client_stats"] = dict(client.stats)
            write_summary()
            print(json.dumps(summaries["stopped_by_alarm"]))
            raise SystemExit(2)

    emb = embed_mod.Embedder(cache_path=config.PRIVATE / "llm" / "embed_cache.jsonl")
    if a.infer_from is not None:
        src = stage_dir(a.case, tag=None if a.infer_from == "main" else a.infer_from) / "infer.jsonl"
        copy_infer_subset(src, d / "infer.jsonl", [p.id for p in persons])
    run_name = f"case{a.case}" + (f"-{a.tag}" if a.tag else "")
    if a.stage in ("infer", "all") and a.infer_from is None:
        with progress.Tracker(f"{run_name}-1-infer", "inference", len(persons), unit="persons", client=client,
                              next_step="judge") as tr:
            summaries["infer"] = run_infer(persons, api, config.ATTACKER, d / "infer.jsonl", usable_tokens=usable,
                                           workers=a.workers, variant=a.variant, tracker=tr)
        summaries["infer"]["problems"] = check_infer_invariants(persons, _read_jsonl(d / "infer.jsonl"))
        stop_on_alarm("infer")
    sequential = a.panel_scheme == "sequential"
    panel = config.PANEL if sequential else config.JUDGES
    # judging and the later stages: all persons, or the sample of config.PERSON_SAMPLE (inference stays complete)
    infer_j, persons_j = d / "infer.jsonl", persons
    if a.case in config.PERSON_SAMPLE and not a.limit and a.stage in ("judge-local", "judge", "dedup", "metrics",
                                                                        "all", "categories", "category-metrics"):
        n = config.PERSON_SAMPLE[a.case]
        persons_j = sample_persons(persons, n)
        infer_j = d / f"infer_sample{n}.jsonl"
        copy_infer_subset(d / "infer.jsonl", infer_j, [p.id for p in persons_j])
        (d / f"sample{n}_persons.json").write_text(json.dumps([p.id for p in persons_j]), encoding="utf-8")
    if a.stage == "judge-local":
        with progress.Tracker(f"{run_name}-2a-judge-local", f"local judge ({config.LOCAL_JUDGE})",
                              len(judge_rows_to_run(infer_j)), unit="step", client=client,
                              next_step="cloud judges when the quota is back") as tr:
            summaries["judge-local"] = run_judge_local(persons_j, infer_j, api, config.LOCAL_JUDGE,
                                                       usable_tokens=usable, tracker=tr)
    if a.stage in ("judge", "all"):
        with progress.Tracker(f"{run_name}-2-judge", "judge", len(judge_rows_to_run(infer_j)),
                              unit="step", client=client, next_step="deduplication") as tr:
            summaries["judge"] = run_judge(persons_j, infer_j, api, panel, d / "judge.jsonl",
                                           attacker=config.ATTACKER, usable_tokens=usable,
                                           workers=a.judge_workers or a.workers, max_chars=a.judge_chars, tracker=tr,
                                           sequential=sequential,
                                           full_panel=sequential and a.case in config.FULL_PANEL_CASES)
        lost = persons_lost([r for r in _read_jsonl(infer_j) if r["attributes"]], _read_jsonl(d / "judge.jsonl"))
        if lost:
            summaries["judge"].setdefault("invariants", []).append(f"{len(lost)} persons lost between infer and judge")
        stop_on_alarm("judge")
    if a.stage in ("dedup", "all"):
        with progress.Tracker(f"{run_name}-3-dedup", "deduplication", 0, unit="persons", client=client,
                              next_step="metrics") as tr:
            summaries["dedup"] = run_dedup(infer_j, d / "judge.jsonl", api, emb, d / "dedup.jsonl",
                                           workers=a.workers, tracker=tr, model=a.dedup_model)
        lost = persons_lost(_read_jsonl(d / "judge.jsonl"), _read_jsonl(d / "dedup.jsonl"))
        if lost:
            summaries["dedup"] = {**summaries["dedup"],
                                  "invariants": [f"{len(lost)} persons lost between judge and dedup"]}
        stop_on_alarm("dedup")
    if a.stage in ("metrics", "all"):
        # a run limited to a few persons (smoke test, pilot) is not a result: it stays under data/private
        out_dir = (d / "metrics_limited" if (a.limit or a.tag) else config.RESULTS / f"case{a.case}")
        with progress.Tracker(f"{run_name}-4-metrics", "metrics", 1, unit="computation", client=client) as tr:
            run_metrics(persons_j, infer_j, d / "judge.jsonl", d / "dedup.jsonl", out_dir, emb, panel)
            tr.advance()
        summaries["metrics"] = {"written": str(out_dir / "metrics.json")}
    if a.stage == "categories":
        from . import categories as categories_mod
        summaries["categories"] = categories_mod.run_categories(infer_j, d / "categories.jsonl", api,
                                                                model=config.CATEGORY_MODEL, workers=a.workers)
    if a.stage == "category-metrics":     # no model call: judged records + categories.jsonl -> metrics_by_category.json
        from . import categories as categories_mod
        out_dir = (d / "metrics_limited" if (a.limit or a.tag) else config.RESULTS / f"case{a.case}")
        _, cat_recs = load_clusters(d / "dedup.jsonl", _read_jsonl(infer_j), _read_jsonl(d / "judge.jsonl"))
        res = categories_mod.write_category_metrics(cat_recs, d / "categories.jsonl", out_dir / "metrics_by_category.json")
        summaries["category_metrics"] = {"written": str(out_dir / "metrics_by_category.json"),
                                         "records": res["n_records"], "categories": sorted(res["categories"])}
    if a.stage == "control":
        from . import control as control_mod
        roles_path = config.PRIVATE / f"case{a.case}" / "roles.json"
        if roles_path.exists():
            roles = control_mod.load_roles(roles_path)
        else:
            roles, summaries["roles"] = control_mod.extract_roles(persons, api, model=config.CATEGORY_MODEL,
                                                                  workers=a.workers)
            control_mod.save_roles(roles, roles_path)
        summaries["control_infer"] = control_mod.run_control_infer(persons, roles, api, d / "control_infer.jsonl",
                                                                   model=config.ATTACKER, workers=a.workers)
        summaries["control_judge"] = control_mod.run_control_judge(persons, roles, d / "control_infer.jsonl", api,
                                                                   panel, d / "control_judge.jsonl",
                                                                   workers=a.workers, sequential=sequential,
                                                                   full_panel=sequential and a.case in config.FULL_PANEL_CASES)
        comp = control_mod.compare_with_documents(_read_jsonl(d / "infer.jsonl"), _read_jsonl(d / "judge.jsonl"),
                                                  _read_jsonl(d / "control_infer.jsonl"),
                                                  _read_jsonl(d / "control_judge.jsonl"))
        (d / "control_comparison.json").write_text(metrics.dumps_strict(comp, indent=1), encoding="utf-8")
        if not (a.limit or a.tag):      # the aggregates are a result: next to metrics.json, where rerun.report reads them
            res_dir = config.RESULTS / f"case{a.case}"
            res_dir.mkdir(parents=True, exist_ok=True)
            (res_dir / "control_comparison.json").write_text(metrics.dumps_strict(comp, indent=1), encoding="utf-8")
        summaries["control_comparison"] = comp
    if a.stage == "calibrate":
        summaries["calibrate"] = run_concurrency_calibration(persons, d / "infer.jsonl", api, a.judge_model,
                                                             d / "calibration.jsonl",
                                                             levels=tuple(int(x) for x in a.levels.split(",")),
                                                             calls_per_level=a.calls_per_level)
    if a.stage == "batchsize":
        summaries["batchsize"] = run_batch_size_test(persons, d / "infer.jsonl", d / "judge.jsonl",
                                                     d / "batch_vs_single.jsonl", api, a.judge_model, d / "batchsize.jsonl",
                                                     sizes=tuple(int(x) for x in a.sizes.split(",")), workers=a.workers)
    if a.stage == "retest":
        summaries["retest"] = run_judge_retest(persons, d / "infer.jsonl", d / "judge.jsonl",
                                               d / "batch_vs_single.jsonl", api, config.JUDGES,
                                               d / "retest.jsonl", n=a.bvs_n, workers=a.workers)
    if a.stage == "batch_vs_single":
        summaries["batch_vs_single"] = run_batch_vs_single(persons, d / "infer.jsonl", d / "judge.jsonl", api, config.JUDGES,
                                                           d / "batch_vs_single.jsonl", n=a.bvs_n, workers=a.workers)
    summaries["client_stats"] = dict(client.stats)
    (d / f"run_summary{'_' + a.stage if a.stage not in ('all',) else ''}.json").write_text(metrics.dumps_strict(summaries, indent=1), encoding="utf-8")
    print(metrics.dumps_strict(summaries, indent=1))


def budget_usable_attacker() -> int:
    return steps.usable_input_tokens(config.num_ctx_for(config.ATTACKER), output_reserve=config.ATTACKER_OUTPUT_RESERVE,
                                     prompt_overhead=config.ATTACKER_PROMPT_OVERHEAD)


if __name__ == "__main__":
    main()
