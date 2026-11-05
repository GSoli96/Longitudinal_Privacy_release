"""RQ4, the LLM detector alone on the generated corpora, for any local model (plan ``docs/plans/rq34.md`` §4, family C).

``rq4_run`` runs the classical detectors and, with ``--llm-model``, the LLM detector in the same process. This driver runs
only the LLM detector (zero-shot and few-shot, prompts and parsing of ``detect_llm``) so that it can go to a machine
with a small GPU, one corpus and one mode at a time, and be stopped and restarted:

* every finished (corpus, mode) pair is appended to ``<name>.parts.jsonl`` (scored records without any message text) and
  skipped on restart; a restart with a different model, option, mode, example set, seed or resample count ignores the file;
* inside a pair every call is also cached by the client (``<name>.llm_cache.jsonl``), so a stop in the middle of a pair
  loses no call.

The few-shot examples are the first three positives and negatives of the public benchmarks in the order of their ids
(``detect_llm.pick_examples``), as in ``rq4_run``. A machine without the benchmarks gets them as a file written by
``make-examples``. Metrics of a model are those of ``detect_metrics`` on the pooled records (all corpora, and each
message variant), plus the coverage of every pair (failed and unparseable answers are counted, never turned into
labels). The output holds counts and metrics only.

``run-bench`` evaluates the same detector on a fixed stratified sample of one public benchmark (CEAS_08: 500 positives
and 500 negatives, seed 42) with the same prompts, options, output files and checkpoint scheme (``bench_sample``,
``run_benchmark``). The few-shot examples are those of ``make-examples`` and are never part of the sample: the sample
is drawn from the benchmark without them. A message longer than ``--max-email-chars`` is cut before it goes into the
prompt (the context is only 4,096 tokens and the server would otherwise drop the start of the prompt, instruction
included); the number of cut messages is in the output."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from . import benchmarks, config, detect_llm, phish_corpus, rq4_run


def parse_options(items) -> dict:
    """``["num_gpu=99", "num_ctx=8192"]`` -> ``{"num_gpu": 99, "num_ctx": 8192}`` (ints, floats and booleans converted)."""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = v
    return out


def make_examples(bench_dir=None, *, cap=5000, seed=config.SEED, n_per_class=3) -> list[dict]:
    """The few-shot examples the full run would pick from the capped public benchmarks (no corpus text involved)."""
    bench = []
    for b in rq4_run.BENCH_NAMES:
        bench += rq4_run.to_records(benchmarks.load(b, bench_dir), benchmark=True)
    bench = rq4_run.cap_benchmarks(bench, cap, seed)
    return [{"id": e["id"], "y": e["y"], "subject": e["subject"], "body": e["body"]}
            for e in detect_llm.pick_examples(bench, n_per_class)]


def run(corpus_names, *, model, modes=detect_llm.MODES, corpus_dir=None, examples=None, options_extra=None, workers=2,
        out_dir, name, n_resamples=500, seed=config.SEED, client=None, loader=None, extra_fingerprint=None,
        meta=None) -> dict:
    """``loader(name) -> records`` replaces the load of a generated corpus (the benchmark sample uses it);
    ``extra_fingerprint`` joins the checkpoint fingerprint (a checkpoint of another sample is ignored); ``meta`` is
    copied into the result."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if client is None:
        from .llm import LLMClient
        client = LLMClient([], cache_path=out / f"{name}.llm_cache.jsonl", max_attempts=3, private_root=out,
                           grow_truncated=False)
    payload = {"model": model, "modes": list(modes), "options": options_extra or {}, "seed": seed,
               "n_resamples": n_resamples, "examples": [e["id"] for e in (examples or [])]}
    if extra_fingerprint is not None:
        payload["extra"] = extra_fingerprint
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    ckpt = out / f"{name}.parts.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("fingerprint") == fingerprint:
                    done[rec["key"]] = rec["part"]
    pooled = {m: [] for m in modes}
    coverage, n_messages, resumed, not_saved = {}, 0, 0, []
    for corpus in corpus_names:
        rows = loader(corpus) if loader else rq4_run.to_records(phish_corpus.load(corpus, corpus_dir))
        n_messages += len(rows)
        variant = {r["id"]: r.get("variant") for r in rows}
        for mode in modes:
            key = f"{corpus}|{mode}"
            if key in done:
                part, resumed = done[key], resumed + 1
            else:
                if mode == "few" and not examples:
                    raise ValueError("few-shot mode needs examples (--examples-file or --bench-dir)")
                res = detect_llm.evaluate(client, model, rows, mode, workers=workers, seed=seed,
                                          options_extra=options_extra, examples=examples)
                for rec in res["records"]:
                    rec["variant"] = variant.get(rec["id"])
                part = {"records": res["records"], "summary": res["summary"]}
                failed = sum(v for e, v in res["summary"]["errors"].items() if e.startswith("call_failed"))
                if failed:      # a pair with failed calls is not a finished pair: no checkpoint, a restart retries it
                    not_saved.append(key)
                else:
                    with open(ckpt, "a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps({"fingerprint": fingerprint, "key": key, "part": part}, sort_keys=True,
                                           default=str) + "\n")
                        f.flush()
            pooled[mode] += part["records"]
            coverage[key] = part["summary"]
    result = {"model": model, "modes": list(modes), "options": options_extra or {}, "corpora": list(corpus_names),
              "n_messages": n_messages, "n_resamples": n_resamples, "resumed_pairs": resumed, "coverage": coverage,
              "pairs_not_checkpointed": not_saved,
              "modes_result": {}, "alarms": []}
    if meta:
        result["meta"] = meta
    for mode in modes:
        recs = pooled[mode]
        parts = {"all": rq4_run._entry(recs, n_resamples, seed) if recs else {"n_test": 0}}
        for v in sorted({r["variant"] for r in recs if r.get("variant")}):
            sub = [r for r in recs if r.get("variant") == v]
            parts[f"variant:{v}"] = rq4_run._entry(sub, n_resamples, seed)
        errors = Counter()
        for k, s in coverage.items():
            if k.endswith(f"|{mode}"):
                errors.update(s["errors"])
        result["modes_result"][mode] = {"parts": parts, "errors": dict(errors),
                                        "n_scored": len(recs), "n_test": sum(s["n_test"] for k, s in coverage.items()
                                                                              if k.endswith(f"|{mode}"))}
        result["alarms"] += sorted(f"{mode}/{p}" for p, e in parts.items() if e.get("alarm"))
    (out / f"{name}.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return result


BENCH_SAMPLE_PER_CLASS = 500
MAX_EMAIL_CHARS = 6000   # subject plus body in the prompt; about 3% of CEAS_08 is longer (median 600 to 1,300 characters)


def bench_sample(rows, *, n_per_class: int = BENCH_SAMPLE_PER_CLASS, seed: int = config.SEED, exclude_ids=(),
                 exclude_texts=(), max_chars: int | None = MAX_EMAIL_CHARS) -> tuple[list[dict], dict]:
    """Stratified sample of a cleaned benchmark: ``n_per_class`` positives and as many negatives drawn with ``seed``
    from the rows in the order of their ``uid`` (one generator, positives first), without the rows in ``exclude_ids``
    or whose (subject, body) is in ``exclude_texts`` (the few-shot examples). -> (records for ``detect_llm``, info).
    A class with fewer rows raises ``ValueError``; nothing is padded. ``max_chars`` cuts subject plus body in the
    records (the body first); ``info['n_clipped']`` counts the messages cut."""
    skip_ids = set(exclude_ids)
    skip_texts = {(s.strip(), b.strip()) for s, b in exclude_texts}
    pool = [r for r in sorted(rows, key=lambda r: str(r["uid"]))
            if r["uid"] not in skip_ids and (r["subject"].strip(), r["body"].strip()) not in skip_texts]
    rng = random.Random(seed)
    picked = []
    for label in (1, 0):
        cls = [r for r in pool if int(r["label"]) == label]
        if len(cls) < n_per_class:
            raise ValueError(f"class {label} has {len(cls)} rows, fewer than {n_per_class}")
        picked += rng.sample(cls, n_per_class)
    clipped, cut = 0, []
    for r in picked:
        r = dict(r)
        if max_chars and len(r["subject"]) + len(r["body"]) > max_chars:
            clipped += 1
            r["subject"] = r["subject"][:max_chars]
            r["body"] = r["body"][:max(0, max_chars - len(r["subject"]))]
        cut.append(r)
    records = rq4_run.to_records(cut, benchmark=True)
    ids = "\n".join(sorted(x["id"] for x in records))
    info = {"n_sample": len(records), "n_positive": sum(x["y"] for x in records), "n_clipped": clipped,
            "n_pool_excluded": len(rows) - len(pool), "max_email_chars": max_chars, "seed": seed,
            "n_per_class": n_per_class, "sample_sha256": hashlib.sha256(ids.encode("utf-8")).hexdigest()[:16]}
    return records, info


def run_benchmark(benchmark: str, *, model, modes=detect_llm.MODES, bench_dir=None, examples=None, options_extra=None,
                  workers=2, out_dir, name, n_resamples=500, seed=config.SEED, n_per_class=BENCH_SAMPLE_PER_CLASS,
                  max_chars: int | None = MAX_EMAIL_CHARS, client=None) -> dict:
    """The LLM detector on the fixed sample of one public benchmark, with the output and the checkpoint of ``run``
    (corpus label ``<benchmark>_s<n>x2``). Also writes ``<name>.sample_ids.json`` (benchmark uids: public data).
    ``seed`` is the seed of the sample and of the model calls."""
    rows = benchmarks.load(benchmark, bench_dir)
    ex = examples or []
    records, info = bench_sample(rows, n_per_class=n_per_class, seed=seed, exclude_ids=[e["id"] for e in ex],
                                 exclude_texts=[(e.get("subject") or "", e.get("body") or "") for e in ex],
                                 max_chars=max_chars)
    info["benchmark"] = benchmark
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.sample_ids.json").write_text(
        json.dumps({**info, "ids": sorted(r["id"] for r in records)}, indent=1), encoding="utf-8")
    return run([f"{benchmark}_s{n_per_class}x2"], model=model, modes=modes, examples=examples,
               options_extra=options_extra, workers=workers, out_dir=out_dir, name=name, n_resamples=n_resamples,
               seed=seed, client=client, loader=lambda _name: records, extra_fingerprint={"sample": info},
               meta={"sample": info})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RQ4: the LLM detector alone (counts and metrics only in the output).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("run-bench", help="run the detector over a fixed stratified sample of one public benchmark")
    b.add_argument("--benchmark", default="CEAS_08", choices=list(benchmarks.BENCHMARKS))
    b.add_argument("--bench-dir", default=None, help="folder of the cleaned benchmarks (default data/private/benchmarks)")
    b.add_argument("--n-per-class", type=int, default=BENCH_SAMPLE_PER_CLASS)
    b.add_argument("--max-email-chars", type=int, default=MAX_EMAIL_CHARS, help="0 = no cut")
    b.add_argument("--model", required=True)
    b.add_argument("--modes", nargs="+", default=list(detect_llm.MODES), choices=list(detect_llm.MODES))
    b.add_argument("--examples-file", default=None, help="few-shot examples (default: those of make-examples)")
    b.add_argument("--bench-cap", type=int, default=5000, help="cap used to pick the examples when no file is given")
    b.add_argument("--option", action="append", default=[], help="model option, e.g. num_ctx=4096 (repeat)")
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--resamples", type=int, default=500)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--name", required=True)
    m = sub.add_parser("make-examples", help="write the few-shot examples from the public benchmarks")
    m.add_argument("--bench-dir", default=None)
    m.add_argument("--bench-cap", type=int, default=5000)
    m.add_argument("--out", required=True)
    r = sub.add_parser("run", help="run the detector over the corpora")
    r.add_argument("--corpus", nargs="+", required=True)
    r.add_argument("--corpus-dir", default=None)
    r.add_argument("--model", required=True)
    r.add_argument("--modes", nargs="+", default=list(detect_llm.MODES), choices=list(detect_llm.MODES))
    r.add_argument("--examples-file", default=None)
    r.add_argument("--bench-dir", default=None, help="to pick the few-shot examples when no file is given")
    r.add_argument("--bench-cap", type=int, default=5000)
    r.add_argument("--option", action="append", default=[], help="model option, e.g. num_gpu=99 (repeat)")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--resamples", type=int, default=500)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run-bench":
        # the examples are loaded in every case: they are kept out of the sample, so that zero-shot and few-shot
        # (and every model) see the same 1,000 messages
        examples = (json.loads(Path(a.examples_file).read_text(encoding="utf-8")) if a.examples_file
                    else make_examples(a.bench_dir, cap=a.bench_cap))
        res = run_benchmark(a.benchmark, model=a.model, modes=tuple(a.modes), bench_dir=a.bench_dir, examples=examples,
                            options_extra=parse_options(a.option), workers=a.workers, out_dir=a.out_dir, name=a.name,
                            n_resamples=a.resamples, n_per_class=a.n_per_class, max_chars=a.max_email_chars or None)
        print(json.dumps({"n_messages": res["n_messages"], "alarms": res["alarms"], "sample": res["meta"]["sample"],
                          "scored": {m_: r_["n_scored"] for m_, r_ in res["modes_result"].items()},
                          "errors": {m_: r_["errors"] for m_, r_ in res["modes_result"].items()},
                          "pairs_not_checkpointed": res["pairs_not_checkpointed"]}, sort_keys=True))
        return 0
    if a.cmd == "make-examples":
        ex = make_examples(a.bench_dir, cap=a.bench_cap)
        Path(a.out).write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"examples": len(ex), "positives": sum(e["y"] for e in ex)}))
        return 0
    examples = None
    if "few" in a.modes:
        if a.examples_file:
            examples = json.loads(Path(a.examples_file).read_text(encoding="utf-8"))
        else:
            examples = make_examples(a.bench_dir, cap=a.bench_cap)
    res = run(a.corpus, model=a.model, modes=tuple(a.modes), corpus_dir=a.corpus_dir, examples=examples,
              options_extra=parse_options(a.option), workers=a.workers, out_dir=a.out_dir, name=a.name,
              n_resamples=a.resamples)
    print(json.dumps({"n_messages": res["n_messages"], "alarms": res["alarms"],
                      "scored": {m_: r_["n_scored"] for m_, r_ in res["modes_result"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
