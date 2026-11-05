"""Five-fold cross-validation of the ten classical TF-IDF detectors on each public benchmark (paper, RQ4 heat map).

For every benchmark of ``benchmarks.BENCHMARKS``: stratified 5-fold cross-validation (seed 42) with the ten pipelines of
``detect_classic.classical_models()``; accuracy, precision, recall and F1 of the positive class (1 = phishing or spam)
per fold, then mean and standard deviation over the folds (population standard deviation, ``ddof=0``).

Choices that the output records:

* Nazario and Nigerian_Fraud have positives only (``benchmarks`` docstring). Their negatives are drawn, with seed 42,
  from the pooled negatives of the other benchmarks, as many as the positives. The output lists how many negatives come
  from which benchmark (counts only). The same rule applies to a benchmark with negatives only (none today).
* At most ``limit`` rows per corpus (default 20,000), drawn stratified by label with seed 42, after the negatives have
  been added. The limit and the original size are stored.
* The vectorisers are those of the paper (50,000 uni- and bigrams, ``min_df=2``), fitted inside each training fold.
* ``--workers`` runs (benchmark, model, fold) tasks in separate processes; the thread count of the forests and of SGD
  is set to 1 in that case. Every finished task is appended to ``<out>.parts.jsonl``; a restart with the same settings
  (same fingerprint) skips the tasks already there.

Outputs (``results/rerun/``): ``bench_cv.json`` (metrics, counts and settings, no text) and
``bench_cv_heatmap_rows.txt`` (rows ``Model & \\gcell{F1} & ... \\\\``, seven columns: the six benchmarks and the F1 of
the person split on the generated corpora, read from ``--person-split``; ``--`` when the file or the model is missing)."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import benchmarks, config, detect_classic, detect_llm

METRICS = ("accuracy", "precision", "recall", "f1")
FOLDS = 5
LIMIT = 20_000
OUT_DIR = config.RESULTS
DEFAULT_PERSON_SPLIT = (config.PRIVATE / "incoming_alienware_20261102_t2" / "lp_collect" / "dais"
                        / "rq4_final_classical_20261102-1144" / "final_person.json")
HEAVY_FIRST = ("Gradient Boosting", "AdaBoost", "Random Forest", "Extra Trees")


def repo_relative(path) -> str:
    """Path relative to the repository root, so that results never record a machine-specific absolute path."""
    p = Path(path).resolve()
    try:
        return p.relative_to(config.ROOT).as_posix()
    except ValueError:
        return p.name


def load_records(name: str, bench_dir=None) -> list[dict]:
    """``{id, text, y, benchmark}`` for the rows of one benchmark."""
    return [{"id": r["uid"], "text": detect_llm.message_text(r), "y": int(r["label"]), "benchmark": name}
            for r in benchmarks.load(name, bench_dir)]


def borrow(records_by_name: dict, name: str, label: int, n: int, seed: int = config.SEED) -> tuple[list, dict]:
    """``n`` records of class ``label`` drawn from the other benchmarks (uniform over the pooled rows, seed ``seed``).
    Returns the records and the count per source benchmark."""
    pool = sorted((r for other, rows in records_by_name.items() if other != name for r in rows if r["y"] == label),
                  key=lambda r: (r["benchmark"], str(r["id"])))
    if len(pool) < n:
        raise ValueError(f"not enough records of class {label} in the other benchmarks for {name}")
    picked = random.Random(seed).sample(pool, n)
    return picked, dict(sorted(Counter(r["benchmark"] for r in picked).items()))


def build_corpus(records_by_name: dict, name: str, seed: int = config.SEED) -> tuple[list, dict | None]:
    """Records of one benchmark; when a class is missing it is filled from the other benchmarks with as many rows
    as the class present. Returns ``(records, borrowed_counts_by_source or None)``."""
    own = list(records_by_name[name])
    n_pos = sum(r["y"] for r in own)
    n_neg = len(own) - n_pos
    if n_pos and n_neg:
        return own, None
    if not own:
        raise ValueError(f"benchmark {name} is empty")
    missing = 0 if n_pos else 1
    extra, counts = borrow(records_by_name, name, missing, len(own), seed)
    return own + extra, counts


def stratified_cap(records: list, limit: int | None, seed: int = config.SEED) -> list:
    """At most ``limit`` records, with the class proportions kept (rounded per class), drawn with ``seed``."""
    if not limit or len(records) <= limit:
        return list(records)
    rng = random.Random(seed)
    by = {0: [], 1: []}
    for r in sorted(records, key=lambda r: str(r["id"])):
        by[r["y"]].append(r)
    total = len(records)
    out = []
    quota = {y: int(round(limit * len(rows) / total)) for y, rows in by.items()}
    while sum(quota.values()) > limit:             # rounding can overshoot by one
        quota[max(quota, key=quota.get)] -= 1
    for y in (0, 1):
        out += rng.sample(by[y], min(quota[y], len(by[y])))
    return out


def fold_indices(y, k: int = FOLDS, seed: int = config.SEED) -> list:
    from sklearn.model_selection import StratifiedKFold
    return [(tr, te) for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y)]


def fold_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0))}


def summarise_folds(folds: list[dict]) -> dict:
    """``{metric: {mean, std}}`` over the folds (``ddof=0``), plus the list of folds."""
    out = {}
    for m in METRICS:
        v = np.array([f[m] for f in folds], dtype=float)
        out[m] = {"mean": float(v.mean()), "std": float(v.std())}
    out["folds"] = folds
    return out


def run_fold(model_name: str, train_texts, y_train, test_texts, y_test, single_thread: bool = True) -> dict:
    """Fit one fresh pipeline on the training texts and score the test texts."""
    model = detect_classic.classical_models()[model_name]
    if single_thread and "clf__n_jobs" in model.get_params():
        model.set_params(clf__n_jobs=1)
    model.fit(train_texts, y_train)
    return fold_metrics(y_test, model.predict(test_texts))


def _heavy_rank(model_name: str) -> int:
    return HEAVY_FIRST.index(model_name) if model_name in HEAVY_FIRST else len(HEAVY_FIRST)


def _task(key, model_name, train_texts, y_train, test_texts, y_test, single_thread):
    t0 = time.time()
    m = run_fold(model_name, train_texts, y_train, test_texts, y_test, single_thread)
    return key, m, round(time.time() - t0, 1)


def fmt_cell(x) -> str:
    return "--" if x is None or x != x else f"{x:.3f}"


def heatmap_rows(results: dict, bench_names, models, person_f1: dict | None = None) -> list[str]:
    """One LaTeX row per model: ``Name & \\gcell{F1 of each benchmark} & \\gcell{person split F1} \\\\``."""
    rows = []
    for m in models:
        cells = [fmt_cell(results[b][m]["f1"]["mean"]) if m in results.get(b, {}) else "--" for b in bench_names]
        cells.append(fmt_cell((person_f1 or {}).get(m)))
        rows.append(m + " & " + " & ".join(r"\gcell{%s}" % c if c != "--" else "--" for c in cells) + r" \\")
    return rows


def person_split_f1(path) -> dict:
    """F1 as generated (estimate) of each detector in the person split of an ``rq4_run`` result file; ``{}`` when
    the file is missing."""
    p = Path(path) if path else None
    if not p or not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    dets = d.get("settings", {}).get("person_split", {}).get("parts", {}).get("all", {}).get("detectors", {})
    out = {}
    for name, e in dets.items():
        est = e.get("metrics", {}).get("as_generated", {}).get("f1", {}).get("estimate")
        if est is not None:
            out[name] = float(est)
    return out


def run(*, limit=LIMIT, folds=FOLDS, seed=config.SEED, workers=1, out_dir=None, name="bench_cv", bench_dir=None,
        person_split=DEFAULT_PERSON_SPLIT, names=None, models=None) -> dict:
    t_start = time.time()
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    names = list(names or benchmarks.BENCHMARKS)
    by_name = {n: load_records(n, bench_dir) for n in benchmarks.BENCHMARKS}   # all six: they lend each other negatives
    model_names = list(models or detect_classic.classical_models())
    corpora, data = {}, {}
    for n in names:
        recs, borrowed = build_corpus(by_name, n, seed)
        capped = stratified_cap(recs, limit, seed)
        corpora[n] = {"n_before_cap": len(recs), "n_rows": len(capped), "n_positive": sum(r["y"] for r in capped),
                      "n_negative": len(capped) - sum(r["y"] for r in capped),
                      "borrowed_by_source": borrowed}
        data[n] = (capped, fold_indices([r["y"] for r in capped], folds, seed))
    fingerprint = hashlib.sha256(json.dumps(
        {"corpora": corpora, "folds": folds, "seed": seed, "limit": limit, "models": model_names},
        sort_keys=True).encode("utf-8")).hexdigest()[:16]
    ckpt = out / f"{name}.parts.jsonl"
    done: dict = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("fingerprint") == fingerprint:
                    done[rec["key"]] = rec["metrics"]
    tasks = []
    for n in names:
        recs, splits = data[n]
        texts, y = [r["text"] for r in recs], np.array([r["y"] for r in recs])
        for m in sorted(model_names, key=_heavy_rank):
            for i, (tr, te) in enumerate(splits):
                key = f"{n}|{m}|{i}"
                if key not in done:
                    tasks.append((key, m, [texts[j] for j in tr], y[tr].tolist(), [texts[j] for j in te],
                                  y[te].tolist(), workers > 1))
    resumed = len(done)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)  # noqa: E731
    log(f"{len(tasks)} tasks to run, {resumed} resumed, workers {workers}")

    def save(key, metrics, secs):
        done[key] = metrics
        with open(ckpt, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"fingerprint": fingerprint, "key": key, "metrics": metrics, "seconds": secs},
                               sort_keys=True) + "\n")

    if workers > 1 and tasks:
        from joblib import Parallel, delayed
        gen = Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(_task)(*t) for t in tasks)
        for k, (key, m, secs) in enumerate(gen, 1):
            save(key, m, secs)
            log(f"{k}/{len(tasks)} {key} f1={m['f1']:.3f} {secs}s")
    else:
        for k, t in enumerate(tasks, 1):
            key, m, secs = _task(*t)
            save(key, m, secs)
            log(f"{k}/{len(tasks)} {key} f1={m['f1']:.3f} {secs}s")
    results = {n: {m: summarise_folds([done[f"{n}|{m}|{i}"] for i in range(folds)]) for m in model_names}
               for n in names}
    pf1 = person_split_f1(person_split)
    result = {"settings": {"folds": folds, "seed": seed, "limit_per_corpus": limit, "models": model_names,
                           "stratified": True, "std": "population (ddof=0) over the folds",
                           "person_split_source": repo_relative(person_split) if pf1 else None,
                           "resumed_tasks": resumed},
              "corpora": corpora, "results": results, "person_split_f1": pf1,
              "elapsed_seconds": round(time.time() - t_start, 1)}
    (out / f"{name}.json").write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    flat = {n: {m: {"f1": results[n][m]["f1"]} for m in model_names} for n in names}
    lines = ["% columns: " + ", ".join(names) + ", generated corpora (person split); cells: mean F1 of 5-fold CV"]
    lines += heatmap_rows(flat, names, model_names, pf1)
    (out / f"{name}_heatmap_rows.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("done")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="5-fold CV of the ten classical detectors on the public benchmarks.")
    ap.add_argument("--limit", type=int, default=LIMIT, help="max rows per corpus (0 = all)")
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--name", default="bench_cv")
    ap.add_argument("--bench-dir", default=None)
    ap.add_argument("--person-split", default=str(DEFAULT_PERSON_SPLIT),
                    help="final_person.json of rq4_run for the last column of the heat map")
    ap.add_argument("--only", nargs="*", default=None, help="benchmarks to run (default all six)")
    a = ap.parse_args(argv)
    run(limit=a.limit or None, folds=a.folds, workers=a.workers, out_dir=a.out_dir, name=a.name,
        bench_dir=a.bench_dir, person_split=a.person_split, names=a.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
