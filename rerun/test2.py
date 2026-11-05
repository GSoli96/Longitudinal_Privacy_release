"""Baseline / Test 1 / Test 2 of the paper for the generated corpora (RQ4), with the rerun pipeline.

Four classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting; hyper-parameters of
``src.detection.get_classical_models``) on a TF-IDF with a fixed vocabulary: at most 5,000 unigrams, learned once on the
baseline training texts (``min_df=2``, sublinear tf, accents stripped) and never refitted.

* Baseline: the six public benchmarks (at most ``bench_cap`` rows per corpus, stratified draw with seed 42 as in
  ``rq4_run``), split 80/20 stratified by label (seed 42); trained on the 80%, scored on the 20%.
* Test 1: the baseline models applied to the generated messages (all of them, and the part that Test 2 uses as test).
* Test 2: the models retrained on baseline training + the training part of the generated messages and tested on the
  test part of the generated messages. The split is by person (``detect_classic.group_split``, 30% of the persons in
  the test, seed 42), so no person is on both sides; the retrained models are also scored on the baseline test.

Two sets of generated messages: ``F_v2`` (the case F v2 corpora of both generators, 384 messages) and ``all16`` (the 16
corpora). Metrics: accuracy, precision, recall, F1 of the phishing class, ROC AUC (nan when a class is missing).
Output ``results/rerun/test2.json``: counts and metrics only, no text. Models run in parallel (``--workers``)."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import benchmarks, config, detect_classic, phish_corpus, rq4_run

CORPUS_DIR = config.PRIVATE / "rq3_collect" / "corpus"
MODELS = ("Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting")
MAX_FEATURES = 5000
SETS = ("F_v2", "all16")
OUT_DIR = config.RESULTS


def corpus_names(corpus_dir=None) -> list[str]:
    d = Path(corpus_dir) if corpus_dir else CORPUS_DIR
    return sorted(json.loads((d / phish_corpus.MANIFEST).read_text(encoding="utf-8")))


def select_set(names: list[str], which: str) -> list[str]:
    if which == "all16":
        return list(names)
    if which == "F_v2":
        return [n for n in names if "_F_v2_" in n]
    raise ValueError(f"unknown set {which!r}; choose from {SETS}")


def fixed_vectorizer(train_texts, max_features: int = MAX_FEATURES):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1), sublinear_tf=True, strip_accents="unicode",
                        analyzer="word", min_df=2)
    return v.fit(train_texts)


def make_classifier(name: str, n_jobs: int | None = 1):
    clf = detect_classic.classical_models()[name].named_steps["clf"]
    if n_jobs is not None and "n_jobs" in clf.get_params():
        clf.set_params(n_jobs=n_jobs)
    return clf


def split_baseline(records, test_fraction: float = 0.2, seed: int = config.SEED) -> tuple[list, list]:
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(records))
    tr, te = train_test_split(idx, test_size=test_fraction, random_state=seed, stratify=[r["y"] for r in records])
    return [records[i] for i in sorted(tr)], [records[i] for i in sorted(te)]


def _scores(clf, X):
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(X), dtype=float)
    return np.asarray(clf.predict_proba(X)[:, 1], dtype=float)


def score(clf, X, y) -> dict:
    """Accuracy, precision, recall, F1 and AUC of ``clf`` on ``X``, ``y`` (1 = phishing)."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    y = np.asarray(y)
    pred = clf.predict(X)
    auc = float(roc_auc_score(y, _scores(clf, X))) if len(set(y.tolist())) == 2 else float("nan")
    return {"n": int(len(y)), "n_positive": int(y.sum()), "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "auc": auc}


def evaluate_model(name, X_btr, y_btr, X_bte, y_bte, sets: dict, n_jobs: int | None = 1) -> dict:
    """Baseline, Test 1 and Test 2 of one classifier. ``sets`` maps a set name to
    ``(X_train, y_train, X_test, y_test)`` of the generated messages (person split)."""
    from scipy.sparse import vstack
    base = make_classifier(name, n_jobs).fit(X_btr, y_btr)
    out = {"baseline": score(base, X_bte, y_bte), "sets": {}}
    for s, (X_gtr, y_gtr, X_gte, y_gte) in sets.items():
        X_all, y_all = vstack([X_gtr, X_gte]), list(y_gtr) + list(y_gte)
        retrained = make_classifier(name, n_jobs).fit(vstack([X_btr, X_gtr]), list(y_btr) + list(y_gtr))
        out["sets"][s] = {"test1_all_generated": score(base, X_all, y_all),
                          "test1_on_test2_split": score(base, X_gte, y_gte),
                          "test2": score(retrained, X_gte, y_gte),
                          "test2_model_on_baseline_test": score(retrained, X_bte, y_bte)}
    return out


def _worker(name, args):
    t0 = time.time()
    res = evaluate_model(name, *args)
    return name, res, round(time.time() - t0, 1)


def prepare(*, bench_cap, seed, corpus_dir, bench_dir, which_sets, max_features=MAX_FEATURES):
    """Vectorise everything once. Returns the shared matrices, the counts and the argument tuple for the workers."""
    bench = []
    for b in benchmarks.BENCHMARKS:
        bench += rq4_run.to_records(benchmarks.load(b, bench_dir), benchmark=True)
    bench = rq4_run.cap_benchmarks(bench, bench_cap, seed)
    b_tr, b_te = split_baseline(bench, 0.2, seed)
    vec = fixed_vectorizer([r["text"] for r in b_tr], max_features)
    X_btr, X_bte = vec.transform([r["text"] for r in b_tr]), vec.transform([r["text"] for r in b_te])
    corpus_dir = corpus_dir or CORPUS_DIR
    names = corpus_names(corpus_dir)
    sets, counts = {}, {"baseline": {"n_total": len(bench), "n_train": len(b_tr), "n_test": len(b_te),
                                     "n_train_positive": sum(r["y"] for r in b_tr),
                                     "n_test_positive": sum(r["y"] for r in b_te),
                                     "vocabulary_size": int(len(vec.vocabulary_)), "bench_cap": bench_cap},
                        "sets": {}}
    for s in which_sets:
        chosen = select_set(names, s)
        rows = [r for n in chosen for r in phish_corpus.load(n, corpus_dir)]
        recs = rq4_run.to_records(rows)
        train, test = detect_classic.group_split(recs, "person", seed=seed)
        sets[s] = (vec.transform([r["text"] for r in train]), [r["y"] for r in train],
                   vec.transform([r["text"] for r in test]), [r["y"] for r in test])
        counts["sets"][s] = {"corpora": chosen, "n_messages": len(recs), "n_persons": len({r["person"] for r in recs}),
                             "n_train": len(train), "n_test": len(test),
                             "n_train_persons": len({r["person"] for r in train}),
                             "n_test_persons": len({r["person"] for r in test}),
                             "n_test_positive": sum(r["y"] for r in test)}
    return (X_btr, [r["y"] for r in b_tr], X_bte, [r["y"] for r in b_te], sets), counts


def run(*, bench_cap=5000, seed=config.SEED, workers=1, out_dir=None, name="test2", corpus_dir=None, bench_dir=None,
        which_sets=SETS, models=MODELS) -> dict:
    t0 = time.time()
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    args, counts = prepare(bench_cap=bench_cap, seed=seed, corpus_dir=corpus_dir, bench_dir=bench_dir,
                           which_sets=which_sets)
    fingerprint = hashlib.sha256(json.dumps({"counts": counts, "seed": seed, "models": list(models)},
                                            sort_keys=True).encode("utf-8")).hexdigest()[:16]
    ckpt = out / f"{name}.parts.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("fingerprint") == fingerprint:
                    done[rec["model"]] = rec["result"]
    todo = [m for m in models if m not in done]
    n_jobs = 1 if workers > 1 else None
    full_args = (*args[:4], args[4], n_jobs)

    def save(model, res, secs):
        done[model] = res
        with open(ckpt, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"fingerprint": fingerprint, "model": model, "result": res, "seconds": secs},
                               sort_keys=True) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] {model} done in {secs}s", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] models to run: {todo}; resumed: {sorted(done)}", flush=True)
    if workers > 1 and todo:
        from joblib import Parallel, delayed
        for model, res, secs in Parallel(n_jobs=min(workers, len(todo)), return_as="generator_unordered")(
                delayed(_worker)(m, full_args) for m in todo):
            save(model, res, secs)
    else:
        for m in todo:
            save(*_worker(m, full_args))
    result = {"settings": {"seed": seed, "max_features": MAX_FEATURES, "ngram_range": [1, 1], "bench_cap": bench_cap,
                           "baseline_split": "80/20 stratified by label", "generated_split": "by person, 30% test",
                           "models": list(models), "resumed_models": len(models) - len(todo)},
              "counts": counts, "results": {m: done[m] for m in models},
              "elapsed_seconds": round(time.time() - t0, 1)}
    (out / f"{name}.json").write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Baseline / Test 1 / Test 2 on the generated corpora.")
    ap.add_argument("--bench-cap", type=int, default=5000, help="max rows per benchmark (0 = all)")
    ap.add_argument("--workers", type=int, default=1, help="models run in parallel")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--name", default="test2")
    ap.add_argument("--corpus-dir", default=None)
    ap.add_argument("--bench-dir", default=None)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    a = ap.parse_args(argv)
    run(bench_cap=a.bench_cap or None, workers=a.workers, out_dir=a.out_dir, name=a.name, corpus_dir=a.corpus_dir,
        bench_dir=a.bench_dir, which_sets=tuple(a.sets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
