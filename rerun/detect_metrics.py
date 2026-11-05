"""Metrics of the phishing detectors (RQ4; plan ``docs/plans/rq34.md`` §4). Pure functions, numpy only, no I/O.

Records are dicts with ``y`` (1 = phishing, 0 = legitimate), ``score`` (higher = more likely phishing), ``pred``
(0/1 decision, optional) and ``person`` (the cluster of the bootstrap). Metrics:

* ``auc_roc``: probability that a positive scores above a negative (ties count one half).
* ``auprc``: average precision (step-wise area under the precision-recall curve, one step per distinct score).
* ``f1``: F1 of the decisions ``pred``.
* ``recall_at_fpr``: recall of the rule "positive if score > t", t the lowest threshold whose false-positive rate does
  not exceed ``fpr``. With ``n`` negatives the resolution is 1/n, so the value is meaningful on the pool of cases.
* Realistic class proportions (1:10, 1:100 phishing:legitimate): positives are weighted so that
  sum(w_pos) / sum(w_neg) = 1/k, negatives keep weight 1. AUC and recall at fixed FPR do not depend on the
  prevalence; AUPRC and F1 do.

A metric that is undefined (one class missing, no predicted positive and no positive) is nan, never 0.
``summarise`` gives each metric with a percentile bootstrap CI that resamples whole persons (``stats``)."""
from __future__ import annotations

import math

import numpy as np

from . import stats

NAN = float("nan")
PREVALENCES = (None, 10, 100)       # None = as generated; k = one phishing message per k legitimate ones
METRICS = ("auc_roc", "auprc", "f1", "recall_at_fpr")


def _arrays(y, s, w=None):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    return y, s, w


def weights_for_prevalence(y, k):
    """Weights 1 for negatives and ``n_neg / (k n_pos)`` for positives (k=None: all ones)."""
    y = np.asarray(y, dtype=float)
    n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
    if k is None or n_pos == 0 or n_neg == 0:
        return np.ones_like(y)
    return np.where(y == 1, n_neg / (k * n_pos), 1.0)


def auc_roc(y, s) -> float:
    y, s, _ = _arrays(y, s)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return NAN
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(allv.size)
    i = 0
    while i < allv.size:                   # average ranks over ties
        j = i
        while j + 1 < allv.size and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    rank_pos = ranks[np.argsort(order, kind="mergesort")][:pos.size].sum()
    return float((rank_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def auprc(y, s, w=None) -> float:
    y, s, w = _arrays(y, s, w)
    total_pos = float(w[y == 1].sum())
    if total_pos == 0 or not (y == 0).any():
        return NAN
    order = np.argsort(-s, kind="mergesort")
    y, s, w = y[order], s[order], w[order]
    tp = np.cumsum(w * (y == 1))
    fp = np.cumsum(w * (y == 0))
    last = np.r_[s[1:] != s[:-1], True]     # one point per distinct score
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / total_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def f1(y, pred, w=None) -> float:
    y, pred, w = _arrays(y, pred, w)
    tp = float(w[(y == 1) & (pred == 1)].sum())
    fp = float(w[(y == 0) & (pred == 1)].sum())
    fn = float(w[(y == 1) & (pred == 0)].sum())
    if tp + fp + fn == 0:
        return NAN
    return float(2 * tp / (2 * tp + fp + fn))


def recall_at_fpr(y, s, fpr: float = 0.01) -> float:
    y, s, _ = _arrays(y, s)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return NAN
    allowed = int(math.floor(fpr * neg.size + 1e-12))
    neg_desc = np.sort(neg)[::-1]
    t = neg_desc[allowed] if allowed < neg.size else -math.inf
    return float((pos > t).mean())


def metrics_on(records, prevalence=None, fpr: float = 0.01) -> dict:
    y = [r["y"] for r in records]
    s = [r["score"] for r in records]
    w = weights_for_prevalence(y, prevalence)
    out = {"auc_roc": auc_roc(y, s), "auprc": auprc(y, s, w), "recall_at_fpr": recall_at_fpr(y, s, fpr)}
    if all(r.get("pred") is not None for r in records) and records:
        out["f1"] = f1(y, [r["pred"] for r in records], w)
    else:
        out["f1"] = NAN
    return out


def _by_person(records) -> list:
    groups: dict = {}
    for r in records:
        groups.setdefault(r["person"], []).append(r)
    return list(groups.values())


def summarise(records, *, prevalences=PREVALENCES, fpr: float = 0.01, n_resamples: int = 2000, seed: int = 42) -> dict:
    """``{prevalence_label: {metric: {estimate, lo, hi, n_units, n_valid_resamples}}}``; labels "as_generated",
    "1:10", "1:100". Resamples whole persons; messages without a person id are each their own cluster."""
    records = [dict(r, person=r.get("person", ("_", i))) for i, r in enumerate(records)]
    units = _by_person(records)
    out = {}
    for k in prevalences:
        label = "as_generated" if k is None else f"1:{k}"
        out[label] = {}
        for m in METRICS:
            def stat(us, m=m, k=k):
                flat = [r for u in us for r in u]
                return metrics_on(flat, k, fpr)[m]
            out[label][m] = stats.cluster_bootstrap_ci(units, stat, n_resamples=n_resamples, seed=seed)
    return out
