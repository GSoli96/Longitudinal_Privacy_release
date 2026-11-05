"""RQ4 driver: detectors on the generated corpus (plan ``docs/plans/rq34.md`` §4-5). Writes metrics and counts only.

Settings (``SETTINGS``):

* ``bench_to_generated`` (main): train on the public benchmarks, test on the generated corpus (all messages and
  each variant apart);
* ``person_split``: train and test on the generated corpus, no person on both sides;
* ``case_split``: train on all cases but one, test on the one left out, for each case;
* ``generator_split``: train on one generator, test on another, both ways (needs two generators in the corpus).

Detectors: the classical TF-IDF models, a style-only and a length-only baseline, the same classical models on the
texts cleaned by ``artefacts`` (``clean:<model>``), and the LLM detector (``llm:zero``, ``llm:few``; only in
``bench_to_generated``, where the few-shot examples come from the benchmarks). Metrics come from ``detect_metrics``
(AUC, AUPRC, F1, recall at 1% FPR; as generated, 1:10, 1:100; bootstrap over persons). A detector that cannot be fitted
is listed under ``skipped``, never scored. ``alarm`` is true when the F1 as generated is at least
``config.ALARM_DETECTION_F1``: the entry is kept and flagged, the run goes on, and the report says on which settings
it fired (open question 7 of the plan). Records in, rows of ``phish_corpus`` (``label``, ``subject``, ``body``,
``person``, ``case``, ``generator``, ``variant``); no text goes into the output."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from . import artefacts, benchmarks, config, detect_classic, detect_llm, detect_metrics, phish_corpus

SETTINGS = ("bench_to_generated", "person_split", "case_split", "generator_split")
OUT_DIR = config.ROOT / "results" / "rerun" / "rq4"
BENCH_NAMES = tuple(benchmarks.BENCHMARKS)


def to_records(rows, *, benchmark: bool = False) -> list[dict]:
    """Rows of a corpus -> detector records (``text``, ``y`` and the grouping keys)."""
    out = []
    for i, r in enumerate(rows):
        out.append({"id": r.get("id") or r.get("uid") or str(i), "text": detect_llm.message_text(r),
                    "y": int(r["label"]), "subject": r.get("subject", ""), "body": r.get("body", ""),
                    "person": r.get("uid") if benchmark else r.get("person"),
                    "case": r.get("benchmark") if benchmark else r.get("case"),
                    "generator": None if benchmark else r.get("generator"), "variant": r.get("variant"),
                    "attack": r.get("attack")})
    return out


def cap_benchmarks(records, cap_per_corpus: int | None, seed: int = config.SEED) -> list[dict]:
    """At most ``cap_per_corpus`` records per benchmark, drawn with a seed (all when ``None``)."""
    if not cap_per_corpus:
        return list(records)
    rng = random.Random(seed)
    by: dict = {}
    for r in records:
        by.setdefault(r["case"], []).append(r)
    out = []
    for name in sorted(by):
        rows = sorted(by[name], key=lambda r: str(r["id"]))
        out += rng.sample(rows, min(cap_per_corpus, len(rows)))
    return out


def _clean(records, organisation, domains):
    return [dict(r, text=detect_llm.message_text(c)) for r, c in
            zip(records, (artefacts.clean_record(r, organisation=organisation, domains=domains) for r in records))]


def _entry(scored, n_resamples, seed):
    summ = detect_metrics.summarise(scored, n_resamples=n_resamples, seed=seed)
    f1 = summ["as_generated"]["f1"]["estimate"]
    return {"n_test": len(scored), "n_positive": sum(r["y"] for r in scored), "metrics": summ,
            "alarm": bool(f1 == f1 and f1 >= config.ALARM_DETECTION_F1)}


def _run_models(train, test, models, name_prefix, n_resamples, seed):
    out, skipped = {}, {}
    res = detect_classic.evaluate(train, test, models)
    skipped.update({f"{name_prefix}{k}": v for k, v in res.pop("_skipped", {}).items()})
    for name, scored in res.items():
        out[f"{name_prefix}{name}"] = _entry(scored, n_resamples, seed)
    return out, skipped


def _baselines():
    return {"style_only": detect_classic.style_model(), "length_only": detect_classic.length_model()}


def evaluate_setting(setting: str, generated, bench, *, models=None, clean_args=((), ()), n_resamples=500,
                     seed=config.SEED, llm_client=None, llm_model=None, llm_workers=2,
                     done=None, on_part=None, only_parts=None) -> dict:
    """``{"parts": {part: {"detectors": {...}, "skipped": {...}, "n_train", "n_test"}}}`` for one setting.

    ``done`` maps ``"<setting>|<part>"`` to a part already computed (loaded from a checkpoint: it is not fitted again);
    ``on_part(key, part)`` is called as soon as a part is complete, so a stopped run can resume. ``only_parts`` limits
    the run to the named parts (e.g. ``all``, ``variant:none``): several machines can each take some, and their
    ``parts.jsonl`` files are concatenated before a final run that finds every part done."""
    models = detect_classic.classical_models() if models is None else models
    org, dom = clean_args
    parts: dict = {}

    def part(name, train, test, with_baselines=True, with_clean=True):
        key = f"{setting}|{name}"
        if only_parts and name not in only_parts and not (done and key in done):
            return
        if done and key in done:
            parts[name] = done[key]
            return
        p_models = {**models, **(_baselines() if with_baselines else {})}
        det, skipped = _run_models(train, test, p_models, "", n_resamples, seed)
        if with_clean:
            ctrain, ctest = _clean(train, org, dom), _clean(test, org, dom)
            cdet, cskip = _run_models(ctrain, ctest, {k: _fresh(models, k) for k in models}, "clean:", n_resamples,
                                      seed)
            det.update(cdet)
            skipped.update(cskip)
        parts[name] = {"detectors": det, "skipped": skipped, "n_train": len(train), "n_test": len(test)}
        if on_part:
            on_part(key, parts[name])

    if setting == "bench_to_generated":
        part("all", bench, generated)
        for v in sorted({r["variant"] for r in generated if r.get("variant")}):
            part(f"variant:{v}", bench, [r for r in generated if r.get("variant") == v], with_clean=False)
        if llm_client is not None and llm_model:
            for mode in detect_llm.MODES:
                res = detect_llm.evaluate(llm_client, llm_model, generated, mode, train=bench, workers=llm_workers,
                                          seed=seed)
                entry = _entry(res["records"], n_resamples, seed) if res["records"] else {"n_test": 0}
                parts["all"]["detectors"][f"llm:{mode}"] = {**entry, "coverage": res["summary"]}
    elif setting == "person_split":
        train, test = detect_classic.group_split(generated, "person", seed=seed)
        part("all", train, test)
    elif setting == "case_split":
        for c in sorted({r["case"] for r in generated}):
            train, test = detect_classic.holdout_by(generated, "case", [c])
            part(f"case:{c}", train, test, with_clean=False)
    elif setting == "generator_split":
        gens = sorted({r["generator"] for r in generated})
        for g in gens:
            others = [x for x in gens if x != g]
            train, test = detect_classic.holdout_by(generated, "generator", others)    # train = generator g
            if train and test:
                part(f"train:{g}->test:{','.join(others)}", train, test, with_clean=False)
    else:
        raise ValueError(f"setting must be one of {SETTINGS}")
    return {"setting": setting, "parts": parts}


def _fresh(models, key):
    from sklearn.base import clone
    return clone(models[key])


def artefact_report(generated, organisation, domains) -> dict:
    _, summ = artefacts.clean_records(to_records_for_clean(generated), organisation=organisation, domains=domains)
    return summ


def to_records_for_clean(rows):
    return [{"y": int(r["label"]), "subject": r.get("subject", ""), "body": r.get("body", "")} for r in rows]


def top_features_report(generated, seed=config.SEED, n=20) -> dict:
    """Heaviest terms of the logistic model on the person split (an artefact check; terms only, no message)."""
    train, test = detect_classic.group_split(generated, "person", seed=seed)
    model = detect_classic.classical_models().get("LogisticRegression") or next(iter(
        detect_classic.classical_models().values()))
    if len({r["y"] for r in train}) < 2:
        return {"skipped": "single_class"}
    model.fit([r["text"] for r in train], [r["y"] for r in train])
    return {"terms": detect_classic.top_features(model, n)}


def run(corpus_names, *, settings=SETTINGS, bench_cap=5000, n_resamples=500, organisation=("Ente",),
        domains=("ente",), llm_model=None, llm_workers=2, out_dir=None, name="rq4", seed=config.SEED,
        corpus_dir=None, bench_dir=None, client=None, only_parts=None) -> dict:
    generated_rows = [r for n in corpus_names for r in phish_corpus.load(n, corpus_dir)]
    generated = to_records(generated_rows)
    bench = []
    if "bench_to_generated" in settings:
        for b in BENCH_NAMES:
            bench += to_records(benchmarks.load(b, bench_dir), benchmark=True)
        bench = cap_benchmarks(bench, bench_cap, seed)
    if llm_model and client is None:
        from .llm import LLMClient
        out = Path(out_dir) if out_dir else OUT_DIR
        out.mkdir(parents=True, exist_ok=True)
        client = LLMClient([], cache_path=out / f"{name}.llm_cache.jsonl", max_attempts=3, private_root=out,
                           grow_truncated=False)
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    # checkpoint: every part of every setting is appended to ``<name>.parts.jsonl`` when complete; a restart with the
    # same corpora and settings of the run (same fingerprint) skips the parts already there
    fingerprint = hashlib.sha256(json.dumps(
        {"corpora": sorted(corpus_names), "n": len(generated), "bench_cap": bench_cap, "n_resamples": n_resamples,
         "seed": seed, "bench": len(bench), "org": list(organisation), "dom": list(domains)},
        sort_keys=True).encode("utf-8")).hexdigest()[:16]
    ckpt = out / f"{name}.parts.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("fingerprint") == fingerprint:
                    done[rec["key"]] = rec["part"]

    def save_part(key, part):
        with open(ckpt, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"fingerprint": fingerprint, "key": key, "part": part}, sort_keys=True,
                               default=str) + "\n")
            f.flush()

    result = {"corpora": list(corpus_names), "n_messages": len(generated),
              "classes": dict(Counter(r["y"] for r in generated)),
              "bench_cap": bench_cap, "n_resamples": n_resamples, "settings": {},
              "artefacts": artefact_report(generated_rows, organisation, domains),
              "top_features": top_features_report(generated, seed), "resumed_parts": len(done)}
    for s in settings:
        result["settings"][s] = evaluate_setting(s, generated, bench, clean_args=(organisation, domains),
                                                 n_resamples=n_resamples, seed=seed,
                                                 llm_client=client if s == "bench_to_generated" else None,
                                                 llm_model=llm_model, llm_workers=llm_workers,
                                                 done=done, on_part=save_part, only_parts=only_parts)
    alarms = sorted(f"{s}/{p}/{d}" for s, v in result["settings"].items() for p, pv in v["parts"].items()
                    for d, e in pv["detectors"].items() if e.get("alarm"))
    result["alarms"] = alarms
    (out / f"{name}.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RQ4: detectors on the generated corpus (metrics only in the output).")
    ap.add_argument("--corpus", nargs="+", required=True, help="names of corpora under data/private/phish")
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    ap.add_argument("--bench-cap", type=int, default=5000, help="max records per benchmark (0 = all)")
    ap.add_argument("--resamples", type=int, default=500)
    ap.add_argument("--organisation", nargs="*", default=["Ente"])
    ap.add_argument("--domains", nargs="*", default=["ente"])
    ap.add_argument("--llm-model", default=None, help="local model of the LLM detector (omit to skip family C)")
    ap.add_argument("--llm-workers", type=int, default=2)
    ap.add_argument("--name", default="rq4")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only-parts", nargs="*", default=None,
                    help="compute only these parts (all, variant:<v>, case:<c>, ...); merge the parts files afterwards")
    ap.add_argument("--corpus-dir", default=None, help="folder of the corpora and their manifest (default data/private/phish)")
    ap.add_argument("--bench-dir", default=None, help="folder of the cleaned benchmarks (default data/private/benchmarks)")
    a = ap.parse_args(argv)
    res = run(a.corpus, settings=tuple(a.settings), bench_cap=a.bench_cap or None, n_resamples=a.resamples,
              organisation=tuple(a.organisation), domains=tuple(a.domains), llm_model=a.llm_model,
              llm_workers=a.llm_workers, out_dir=a.out_dir, name=a.name, corpus_dir=a.corpus_dir,
              bench_dir=a.bench_dir, only_parts=a.only_parts)
    print(json.dumps({"n_messages": res["n_messages"], "alarms": res["alarms"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
