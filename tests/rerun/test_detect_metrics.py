import math

import pytest

from rerun import detect_metrics as dm


def test_auc_known_values():
    assert dm.auc_roc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert dm.auc_roc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    # one of four pairs inverted
    assert dm.auc_roc([1, 1, 0, 0], [0.9, 0.3, 0.5, 0.1]) == pytest.approx(0.75)


def test_auc_ties_count_half_and_undefined_is_nan():
    assert dm.auc_roc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert math.isnan(dm.auc_roc([1, 1], [0.2, 0.3]))


def test_auprc_known_values():
    assert dm.auprc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    # ranking + - + -: precision 1 at recall 1/2, 2/3 at recall 1 -> 0.5 * 1 + 0.5 * 2/3
    assert dm.auprc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]) == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert math.isnan(dm.auprc([0, 0], [0.1, 0.2]))


def test_f1_known_values_and_nan():
    assert dm.f1([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.5)
    assert dm.f1([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert math.isnan(dm.f1([0, 0], [0, 0]))


def test_recall_at_fpr_resolution():
    y = [1] * 4 + [0] * 100
    s = [0.9, 0.8, 0.7, 0.2] + [0.1] * 98 + [0.5, 0.6]
    # 1% of 100 negatives = 1 false positive allowed: threshold = second highest negative (0.5)
    assert dm.recall_at_fpr(y, s, 0.01) == pytest.approx(0.75)
    # with 10 negatives the first false positive already exceeds 1%: threshold = highest negative
    y2 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    s2 = [0.9, 0.45, 0.5] + [0.1] * 9
    assert dm.recall_at_fpr(y2, s2, 0.01) == pytest.approx(0.5)


def test_prevalence_weights_lower_auprc_but_not_auc():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.6, 0.4, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.05, 0.02]
    w = dm.weights_for_prevalence(y, 10)
    assert sum(wi for wi, yi in zip(w, y) if yi == 1) / sum(wi for wi, yi in zip(w, y) if yi == 0) == pytest.approx(0.1)
    assert dm.auprc(y, s, w) < dm.auprc(y, s)
    assert dm.metrics_on([{"y": a, "score": b} for a, b in zip(y, s)], 10)["auc_roc"] == dm.auc_roc(y, s)


def test_summarise_clusters_by_person_and_is_deterministic():
    recs = []
    for p in range(10):
        recs.append({"person": p, "y": 1, "score": 0.9 - 0.01 * p, "pred": 1})
        recs.append({"person": p, "y": 0, "score": 0.1 + 0.01 * p, "pred": 0})
    a = dm.summarise(recs, n_resamples=50)
    b = dm.summarise(recs, n_resamples=50)
    assert a == b
    assert set(a) == {"as_generated", "1:10", "1:100"}
    est = a["as_generated"]["auc_roc"]
    assert est["estimate"] == 1.0 and est["n_units"] == 10
    assert a["as_generated"]["f1"]["estimate"] == 1.0


def test_missing_pred_gives_nan_f1():
    recs = [{"person": 0, "y": 1, "score": 0.9}, {"person": 1, "y": 0, "score": 0.1}]
    assert math.isnan(dm.metrics_on(recs)["f1"])
