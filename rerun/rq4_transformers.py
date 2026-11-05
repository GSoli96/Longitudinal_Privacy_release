"""RQ4: fine-tuned Transformer detectors and hard/soft voting ensembles, redone with the pipeline of the rerun.

What is computed
================
Two settings, each with a fixed train/test split (nothing is ever tuned or selected on a test set):

* ``generated_person_split``: the generated corpora (``phish_corpus``) split by ``detect_classic.group_split(..,
  "person", seed)``, the same call as the ``person_split`` of ``rq4_run`` (so the test persons are the same);
* ``ceas_08``: one fixed stratified sample of CEAS_08 (``data/private/benchmarks``): 10,000 train and 2,500 test
  messages, disjoint, class proportions as in the corpus, seed 42. Each message is its own bootstrap cluster (the
  benchmarks have no person).

Detectors: the six Transformers of ``src.detection.NLP_MODEL_NAMES`` fine-tuned (3 epochs, lr 2e-5, max_len 256,
seed 42, bf16 or fp16 when a GPU is present) and the ten classical TF-IDF classifiers of ``detect_classic``, all fitted
on the same data. Input text is ``detect_llm.message_text`` (subject and body), as for the other RQ4 detectors.

How the six ensemble members are chosen (no test data involved)
===============================================================
1. The train part of each setting is split by person (``group_split``, ``val_fraction``, default 0.10) into
   ``fit`` and ``val``. Every detector, Transformer or classical, is fitted on ``fit`` only and scores ``val`` and
   ``test``. So the reported Transformer and classical numbers come from models trained on 90% of the train
   (declared simplification: selecting on data the model has been fitted on would favour overfitting models).
2. The candidate pool is the 16 detectors (``--pool all``; ``--pool transformers`` keeps only the six Transformers,
   and then the ensemble contains all of them).
3. The six best candidates by F1 of their decisions on ``val`` (ties: higher AUC on ``val``, then name) are the
   members. The test set is never read at this step.
4. Hard voting: the member decisions are counted; the message is phishing when more than half of the members say so
   (a 3-3 tie goes to legitimate, as ``sklearn``'s ``VotingClassifier`` does). The ranking score of the hard
   ensemble is the share of members voting phishing (seven distinct values, so AUC and recall at 1% FPR are coarse).
   Soft voting: the mean of the members' phishing probabilities, phishing when it is at least 0.5. The classical
   models without ``predict_proba`` (Linear SVC) give ``1 / (1 + exp(-decision_function))``.
5. Metrics (``detect_metrics.summarise``: AUC, AUPRC, F1, recall at 1% FPR; as generated, 1:10, 1:100; percentile
   bootstrap over persons) for every detector and both ensembles on ``test``.

Checkpoint and output
=====================
After every detector (and after its metrics) one line is appended to ``<out-dir>/<name>.preds.jsonl`` (default
``data/private/rq4_transformers/``, git-ignored): the probability and the decision per message of ``val`` and
``test``, in the order of the split, plus the ids of the split; no text. A line carries a fingerprint of the
settings, the hyper-parameters and the ids of the split; a restart with the same fingerprint skips what is done, a
changed one recomputes it. Files from several machines can be concatenated before a final run. A detector that
fails (e.g. out of memory) is listed under ``skipped`` and retried at the next start. The summary goes to
``results/rerun/rq4_transformers.json``: counts and metrics with intervals only.

``trainer`` (argument of ``run``) is a function ``trainer(model_name, train_texts, train_y, eval_texts, **hp) ->
{eval_name: [probability of phishing, ...]}``. The default, ``hf_train_predict``, downloads the model from the
Hugging Face hub and trains it with a plain PyTorch loop; the tests inject a fake one."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

from . import benchmarks, config, detect_classic, detect_llm, detect_metrics, phish_corpus
from .rq4_run import to_records

SETTINGS = ("generated_person_split", "ceas_08")
CEAS_NAME = "CEAS_08"
OUT_DIR = config.PRIVATE / "rq4_transformers"
RESULTS_PATH = config.RESULTS / "rq4_transformers.json"
POOLS = ("all", "transformers")
HP = {"epochs": 3, "lr": 2e-5, "max_len": 256, "batch_size": 16, "seed": 42}
N_MEMBERS = 6


def transformer_names() -> tuple:
    from src import detection
    return tuple(detection.NLP_MODEL_NAMES)


# ---- data ---------------------------------------------------------------------------------------------------------
def stratified_sample(records, n_train: int, n_test: int, seed: int = config.SEED) -> tuple:
    """Disjoint stratified train/test samples: each class contributes in proportion to its share of ``records``.
    Deterministic given ``seed`` (records are ordered by id before the draw). Raises when there are too few."""
    need = n_train + n_test
    if need > len(records):
        raise ValueError(f"not enough records: {len(records)} < {need}")
    by: dict = {}
    for r in sorted(records, key=lambda r: str(r["id"])):
        by.setdefault(r["y"], []).append(r)
    rng = random.Random(seed)
    train, test = [], []
    for y in sorted(by):
        rows = by[y]
        rng.shuffle(rows)
        share = len(rows) / len(records)
        k_test, k_train = round(n_test * share), round(n_train * share)
        if k_test + k_train > len(rows):
            raise ValueError(f"class {y} has too few records")
        test += rows[:k_test]
        train += rows[k_test:k_test + k_train]
    return train, test


def split_setting(setting: str, *, generated=None, ceas=None, seed: int = config.SEED, val_fraction: float = 0.10,
                  n_train: int = 10_000, n_test: int = 2_500) -> dict:
    """``{"fit", "val", "test"}`` record lists of a setting (``val`` is taken from the train part, by person)."""
    if setting == "generated_person_split":
        train, test = detect_classic.group_split(generated, "person", seed=seed)
    elif setting == "ceas_08":
        train, test = stratified_sample(ceas, n_train, n_test, seed)
    else:
        raise ValueError(f"setting must be one of {SETTINGS}")
    fit, val = detect_classic.group_split(train, "person", test_fraction=val_fraction, seed=seed)
    return {"fit": fit, "val": val, "test": test}


# ---- checkpoint ---------------------------------------------------------------------------------------------------
def fingerprint(setting: str, data: dict, hp: dict, extra: dict | None = None) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"setting": setting, "hp": hp, "extra": extra or {}}, sort_keys=True).encode("utf-8"))
    for part in ("fit", "val", "test"):
        h.update(part.encode())
        h.update("\x1f".join(f"{r['id']}:{r['y']}" for r in data[part]).encode("utf-8"))
    return h.hexdigest()[:16]


def summary_fp(fp: str, n_resamples: int) -> str:
    """Fingerprint of the metric lines: the one of the predictions plus the number of bootstrap resamples."""
    return f"{fp}:r{n_resamples}"


def load_checkpoint(path: Path, fps_by_setting: dict) -> dict:
    """``{key: payload}`` of the lines whose fingerprint is one of the current ones of their setting."""
    done: dict = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("fingerprint") in fps_by_setting.get(rec.get("setting"), ()):
                done[rec["key"]] = rec["payload"]
    return done


def append_checkpoint(path: Path, setting: str, fp: str, key: str, payload: dict) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"fingerprint": fp, "setting": setting, "key": key, "payload": payload}, sort_keys=True,
                           default=str) + "\n")
        f.flush()


# ---- detectors ----------------------------------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -500, 500)))


def _proba(model, texts) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(texts)[:, 1], dtype=float)
    return _sigmoid(model.decision_function(texts))


def classical_predict(model, fit, eval_records: dict) -> dict:
    """Fit a classical pipeline on ``fit``; ``{eval_name: {"p": [...], "pred": [...]}}``."""
    model.fit([r["text"] for r in fit], [r["y"] for r in fit])
    out = {}
    for name, recs in eval_records.items():
        texts = [r["text"] for r in recs]
        out[name] = {"p": [float(v) for v in _proba(model, texts)], "pred": [int(v) for v in model.predict(texts)]}
    return out


def hf_train_predict(model_name: str, train_texts, train_y, eval_texts: dict, *, epochs: int = 3, lr: float = 2e-5,
                     max_len: int = 256, batch_size: int = 16, seed: int = 42, eval_batch_size: int = 64,
                     weight_decay: float = 0.01, warmup_ratio: float = 0.1) -> dict:
    """Fine-tune ``model_name`` (sequence classification, two labels) and return the probability of class 1 for each
    list in ``eval_texts``. AdamW, linear schedule with 10% warm-up, clipping at 1.0, shuffled batches. Mixed precision
    on a GPU: bf16 when supported, else fp16 with a gradient scaler; fp32 on the CPU. Imports torch lazily."""
    import gc

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype, scaler = None, None
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype, scaler = torch.float16, torch.amp.GradScaler("cuda")
    tok = AutoTokenizer.from_pretrained(model_name)
    # fp32 weights, whatever dtype the checkpoint is stored in (transformers 5 loads the stored dtype by default, and
    # AdamW on fp16 parameters gives NaN for mdeberta-v3-base); mixed precision comes from autocast only
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, dtype=torch.float32)
    except TypeError:                                                # transformers 4: the argument is torch_dtype
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, torch_dtype=torch.float32)
    model = model.to(device)

    def encode(texts):
        return tok(list(texts), truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]

    def batch_of(ids_list):
        b = tok.pad({"input_ids": ids_list}, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in b.items()}

    ids = encode(train_texts)
    y = torch.tensor(list(train_y), dtype=torch.long)
    n = len(ids)
    steps_per_epoch = math.ceil(n / batch_size)
    total = max(steps_per_epoch * epochs, 1)
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    groups = [{"params": [p for nm, p in model.named_parameters() if not any(t in nm for t in no_decay)],
               "weight_decay": weight_decay},
              {"params": [p for nm, p in model.named_parameters() if any(t in nm for t in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * total), total)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen).tolist()
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            batch = batch_of([ids[i] for i in idx])
            labels = y[idx].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(**batch, labels=labels).loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
    model.eval()
    out = {}
    for name, texts in eval_texts.items():
        enc = encode(texts)
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]))      # similar lengths together
        probs = [0.0] * len(enc)
        with torch.no_grad():
            for s in range(0, len(order), eval_batch_size):
                idx = order[s:s + eval_batch_size]
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(**batch_of([enc[i] for i in idx])).logits
                p = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
                for i, v in zip(idx, p):
                    probs[i] = float(v)
        out[name] = probs
    del model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def transformer_predict(trainer, model_name, fit, eval_records: dict, hp: dict) -> dict:
    """``{eval_name: {"p", "pred"}}`` from a trainer; the decision is ``p >= 0.5``."""
    probs = trainer(model_name, [r["text"] for r in fit], [r["y"] for r in fit],
                    {k: [r["text"] for r in v] for k, v in eval_records.items()}, **hp)
    out = {}
    for name, recs in eval_records.items():
        p = [float(v) for v in probs[name]]
        if len(p) != len(recs) or not all(math.isfinite(v) for v in p):
            raise ValueError(f"{model_name}: the trainer returned a wrong number of scores or a non-finite one")
        out[name] = {"p": p, "pred": [int(v >= 0.5) for v in p]}
    return out


# ---- ensembles ----------------------------------------------------------------------------------------------------
def val_scores(records, pred) -> dict:
    """F1 of the decisions and AUC of the probabilities on the validation records."""
    scored = [{"y": r["y"], "score": p, "pred": d} for r, p, d in zip(records, pred["p"], pred["pred"])]
    m = detect_metrics.metrics_on(scored)
    return {"f1": m["f1"], "auc_roc": m["auc_roc"]}


def select_members(val_metrics: dict, k: int = N_MEMBERS) -> list:
    """Names of the ``k`` best detectors by validation F1 (ties: AUC, then name). nan sorts last."""
    def key(name):
        f1, auc = val_metrics[name]["f1"], val_metrics[name]["auc_roc"]
        return (-(f1 if f1 == f1 else -1.0), -(auc if auc == auc else -1.0), name)
    return sorted(val_metrics, key=key)[:k]


def hard_vote(member_preds: list) -> dict:
    """``member_preds``: one ``{"p", "pred"}`` per member, aligned. Majority of decisions; a tie goes to 0."""
    votes = np.array([m["pred"] for m in member_preds], dtype=float)
    share = votes.mean(axis=0)
    return {"p": [float(v) for v in share], "pred": [int(v) for v in (share > 0.5)]}


def soft_vote(member_preds: list) -> dict:
    mean = np.array([m["p"] for m in member_preds], dtype=float).mean(axis=0)
    return {"p": [float(v) for v in mean], "pred": [int(v) for v in (mean >= 0.5)]}


# ---- metrics ------------------------------------------------------------------------------------------------------
def entry(test, pred, n_resamples: int, seed: int) -> dict:
    """Metrics with bootstrap intervals of one detector on ``test`` (counts and numbers only)."""
    scored = [{"y": r["y"], "score": p, "pred": d, "person": r.get("person")}
              for r, p, d in zip(test, pred["p"], pred["pred"])]
    summ = detect_metrics.summarise(scored, n_resamples=n_resamples, seed=seed)
    f1 = summ["as_generated"]["f1"]["estimate"]
    return {"n_test": len(test), "n_positive": sum(r["y"] for r in test), "metrics": summ,
            "alarm": bool(f1 == f1 and f1 >= config.ALARM_DETECTION_F1)}


# ---- one setting --------------------------------------------------------------------------------------------------
def run_setting(setting, data, *, transformers, classical, trainer, hp, ckpt: Path | None, done: dict, fp: str,
                n_resamples: int, seed: int, pool: str = "all", k: int = N_MEMBERS, log=lambda *_: None) -> dict:
    fit, val, test = data["fit"], data["val"], data["test"]
    evals = {"val": val, "test": test}
    result = {"n_fit": len(fit), "n_val": len(val), "n_test": len(test),
              "n_positive_test": sum(r["y"] for r in test), "detectors": {}, "skipped": {}, "pool": pool}
    if len({r["y"] for r in fit}) < 2 or not val or not test:
        result["skipped"] = {"all": "single_class_or_empty_split"}
        return result

    def save(key, payload, fingerprint_=None):
        if ckpt is not None:
            append_checkpoint(ckpt, setting, fingerprint_ or fp, key, payload)

    sfp = summary_fp(fp, n_resamples)

    if f"{setting}|split" not in done:
        save(f"{setting}|split", {"val_ids": [str(r["id"]) for r in val], "test_ids": [str(r["id"]) for r in test],
                                  "val_y": [r["y"] for r in val], "test_y": [r["y"] for r in test]})
    preds: dict = {}
    families: dict = {}
    work = [("classical", n, m) for n, m in classical.items()] + [("transformer", n, None) for n in transformers]
    for family, name, model in work:
        key = f"{setting}|pred|{name}"
        families[name] = family
        if key in done:
            preds[name] = done[key]
            log(f"[{setting}] {name}: from checkpoint")
            continue
        t0 = time.time()
        try:
            p = classical_predict(model, fit, evals) if family == "classical" else \
                transformer_predict(trainer, name, fit, evals, hp)
        except Exception as e:  # noqa: BLE001
            result["skipped"][name] = type(e).__name__
            log(f"[{setting}] {name}: skipped ({type(e).__name__})")
            continue
        p["seconds"] = round(time.time() - t0, 1)
        preds[name] = p
        save(key, p)
        log(f"[{setting}] {name}: done in {p['seconds']} s")

    def summary_of(name, p):
        skey = f"{setting}|summary|{name}"
        if skey in done:
            return done[skey]
        e = entry(test, p["test"], n_resamples, seed)
        save(skey, e, sfp)
        return e

    for name, p in preds.items():
        result["detectors"][name] = {"family": families[name], "seconds_fit_predict": p.get("seconds"),
                                     **summary_of(name, p)}
    # ensembles: candidates and members are chosen on val only
    cand = [n for n in preds if pool == "all" or families[n] == "transformer"]
    missing = [n for n in transformers if n not in preds]
    if missing:
        result["ensemble"] = {"skipped": "transformers_missing", "missing": missing}
        return result
    vm = {n: val_scores(val, preds[n]["val"]) for n in cand}
    members = select_members(vm, k) if pool == "all" else list(cand)
    if len(members) < 2:
        result["ensemble"] = {"skipped": "fewer_than_two_candidates"}
        return result
    ens = {"pool": pool, "selection_metric": "f1_on_validation", "members": members,
           "validation": {n: vm[n] for n in members}, "n_candidates": len(cand)}
    for label, fn in (("hard_voting", hard_vote), ("soft_voting", soft_vote)):
        member_test = [preds[n]["test"] for n in members]
        p = {"test": fn(member_test)}
        # the ensemble metrics depend on the members: the key includes them, so a change of members recomputes
        skey = f"{setting}|summary|{label}|" + "+".join(sorted(members))
        e = done.get(skey)
        if e is None:
            e = entry(test, p["test"], n_resamples, seed)
            save(skey, e, sfp)
        ens[label] = e
    result["ensemble"] = ens
    return result


# ---- run ----------------------------------------------------------------------------------------------------------
def run(corpus_names=(), *, settings=SETTINGS, models=None, classical=None, trainer=None, hp=None,
        n_resamples: int = 500, seed: int = config.SEED, val_fraction: float = 0.10, n_train: int = 10_000,
        n_test: int = 2_500, pool: str = "all", name: str = "rq4_transformers", out_dir=None, results_path=None,
        corpus_dir=None, bench_dir=None, generated=None, ceas=None, log=lambda *_: None) -> dict:
    """Run the settings and write the summary. ``generated`` / ``ceas`` (record lists) replace the files (tests)."""
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}")
    hp = {**HP, "seed": seed, **(hp or {})}
    trainer = trainer or hf_train_predict
    transformers = tuple(models) if models is not None else transformer_names()
    classical = detect_classic.classical_models() if classical is None else classical
    if "generated_person_split" in settings and generated is None:
        generated = to_records([r for n in corpus_names for r in phish_corpus.load(n, corpus_dir)])
    if "ceas_08" in settings and ceas is None:
        ceas = to_records(benchmarks.load(CEAS_NAME, bench_dir), benchmark=True)
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{name}.preds.jsonl"
    datas = {s: split_setting(s, generated=generated, ceas=ceas, seed=seed, val_fraction=val_fraction,
                              n_train=n_train, n_test=n_test) for s in settings}
    extra = {"val_fraction": val_fraction, "n_train": n_train, "n_test": n_test, "corpora": sorted(corpus_names)}
    fps = {s: fingerprint(s, datas[s], hp, extra) for s in settings}
    done = load_checkpoint(ckpt, {s: {fps[s], summary_fp(fps[s], n_resamples)} for s in settings})
    result = {"config": {"hp": hp, "val_fraction": val_fraction, "pool": pool, "n_resamples": n_resamples,
                         "seed": seed, "corpora": list(corpus_names), "models": list(transformers),
                         "ceas_sample": {"n_train": n_train, "n_test": n_test}},
              "resumed_items": len(done), "settings": {}}
    for s in settings:
        result["settings"][s] = run_setting(s, datas[s], transformers=transformers, classical=classical,
                                            trainer=trainer, hp=hp, ckpt=ckpt, done=done, fp=fps[s],
                                            n_resamples=n_resamples, seed=seed, pool=pool, log=log)
    result["alarms"] = sorted(f"{s}/{d}" for s, v in result["settings"].items()
                              for d, e in v.get("detectors", {}).items() if e.get("alarm"))
    path = Path(results_path) if results_path else RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RQ4: fine-tuned Transformers and voting ensembles "
                                             "(metrics only in the output; resumable).")
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    ap.add_argument("--corpus", nargs="*", default=[], help="corpora under data/private/phish (setting generated)")
    ap.add_argument("--models", nargs="*", default=None, help="Transformers to fine-tune (default: the six of "
                    "src.detection.NLP_MODEL_NAMES); the ensemble needs all of them")
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    ap.add_argument("--lr", type=float, default=HP["lr"])
    ap.add_argument("--max-len", type=int, default=HP["max_len"])
    ap.add_argument("--batch-size", type=int, default=HP["batch_size"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--val-fraction", type=float, default=0.10, help="share of the train persons kept as validation")
    ap.add_argument("--ceas-train", type=int, default=10_000)
    ap.add_argument("--ceas-test", type=int, default=2_500)
    ap.add_argument("--pool", choices=POOLS, default="all", help="candidates of the ensemble: all 16 detectors or "
                    "the six Transformers")
    ap.add_argument("--resamples", type=int, default=500)
    ap.add_argument("--name", default="rq4_transformers")
    ap.add_argument("--out-dir", default=None, help="folder of the checkpoint (default data/private/rq4_transformers)")
    ap.add_argument("--results", default=None, help="summary file (default results/rerun/rq4_transformers.json)")
    ap.add_argument("--corpus-dir", default=None)
    ap.add_argument("--bench-dir", default=None)
    a = ap.parse_args(argv)
    if "generated_person_split" in a.settings and not a.corpus:
        ap.error("--corpus is needed for the setting generated_person_split")
    hp = {"epochs": a.epochs, "lr": a.lr, "max_len": a.max_len, "batch_size": a.batch_size}
    res = run(a.corpus, settings=tuple(a.settings), models=a.models, hp=hp, n_resamples=a.resamples, seed=a.seed,
              val_fraction=a.val_fraction, n_train=a.ceas_train, n_test=a.ceas_test, pool=a.pool, name=a.name,
              out_dir=a.out_dir, results_path=a.results, corpus_dir=a.corpus_dir, bench_dir=a.bench_dir,
              log=lambda m: print(m, file=sys.stderr, flush=True))
    print(json.dumps({"settings": {s: {"n_test": v["n_test"], "skipped": v["skipped"]}
                                   for s, v in res["settings"].items()}, "alarms": res["alarms"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
