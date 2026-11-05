"""Tests of rerun.stats: cluster bootstrap, permutation and mixed-model trend tests, agreement coefficients."""
import math
import random

import pytest

from rerun import stats


def mean(xs):
    return sum(xs) / len(xs)


def counts_to_ratings(table):
    """Fleiss counts table (units x categories) -> per-unit lists of labels."""
    return [[c for c, n in enumerate(row) for _ in range(n)] for row in table]


# ---- cluster bootstrap ---------------------------------------------------------------------------
def test_bootstrap_contains_mean_and_reports_counts():
    units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r = stats.cluster_bootstrap_ci(units, mean, n_resamples=500)
    assert r["estimate"] == pytest.approx(5.5)
    assert r["lo"] <= 5.5 <= r["hi"]
    assert r["n_units"] == 10
    assert r["n_valid_resamples"] == 500


def test_bootstrap_reproducible_and_seed_dependent():
    units = [1, 4, 2, 9, 3, 7, 5, 8]
    a = stats.cluster_bootstrap_ci(units, mean, n_resamples=300, seed=1)
    b = stats.cluster_bootstrap_ci(units, mean, n_resamples=300, seed=1)
    c = stats.cluster_bootstrap_ci(units, mean, n_resamples=300, seed=2)
    assert a == b
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"])


def test_bootstrap_narrows_with_more_units():
    few = list(range(10))
    many = list(range(10)) * 10
    w_few = stats.cluster_bootstrap_ci(few, mean, n_resamples=500)
    w_many = stats.cluster_bootstrap_ci(many, mean, n_resamples=500)
    assert (w_many["hi"] - w_many["lo"]) < (w_few["hi"] - w_few["lo"])


def test_bootstrap_degenerate_all_equal():
    r = stats.cluster_bootstrap_ci([3, 3, 3, 3], mean, n_resamples=100)
    assert r["lo"] == r["hi"] == r["estimate"] == 3


def test_bootstrap_empty_units():
    r = stats.cluster_bootstrap_ci([], mean)
    assert math.isnan(r["estimate"]) and math.isnan(r["lo"]) and math.isnan(r["hi"])
    assert r["n_units"] == 0


def test_bootstrap_ignores_nan_resamples():
    def stat(us):
        return float("nan") if us.count(0) == len(us) else mean(us)
    r = stats.cluster_bootstrap_ci([0, 0, 0, 1], stat, n_resamples=400)
    assert 0 < r["n_valid_resamples"] < 400
    assert not math.isnan(r["lo"])


# ---- permutation trend test ----------------------------------------------------------------------
def test_permutation_increasing_trend():
    data = {f"u{i}": [i * 0.1 + 1.0 * s for s in range(4)] for i in range(8)}
    r = stats.permutation_trend_test(data, n_permutations=2000)
    assert r["slope"] == pytest.approx(1.0)
    assert r["p_value"] < 0.01
    assert r["n_units"] == 8


def test_permutation_flat_constant():
    r = stats.permutation_trend_test({"a": [2.0] * 4, "b": [5.0] * 4}, n_permutations=200)
    assert r["slope"] == 0
    assert r["p_value"] == 1.0


def test_permutation_noise_not_significant():
    rng = random.Random(0)
    data = {f"u{i}": [rng.random() for _ in range(4)] for i in range(12)}
    r = stats.permutation_trend_test(data, n_permutations=2000, seed=42)
    assert r["p_value"] > 0.05


def test_permutation_ignores_units_with_single_valid_value():
    data = {"a": [1.0, 2.0, 3.0, 4.0], "b": [None, 9.0, None, None], "c": [None] * 4}
    r = stats.permutation_trend_test(data, n_permutations=100)
    assert r["n_units"] == 1
    assert r["slope"] == pytest.approx(1.0)


def test_permutation_missing_values_use_step_index():
    # values at steps 0 and 3 only: slope is (4 - 1) / 3
    r = stats.permutation_trend_test({"a": [1.0, None, None, 4.0]}, n_permutations=50)
    assert r["slope"] == pytest.approx(1.0)


def test_permutation_reproducible():
    data = {f"u{i}": [(i * s) % 3 for s in range(4)] for i in range(6)}
    assert (stats.permutation_trend_test(data, n_permutations=300, seed=7)
            == stats.permutation_trend_test(data, n_permutations=300, seed=7))


def test_permutation_no_valid_unit():
    r = stats.permutation_trend_test({"a": [None, 1.0, None, None]}, n_permutations=10)
    assert r["n_units"] == 0
    assert math.isnan(r["slope"]) and math.isnan(r["p_value"])


# ---- mixed model ---------------------------------------------------------------------------------
def test_mixed_model_recovers_slope():
    rng = random.Random(1)
    rows = []
    for u in range(10):
        offset = rng.gauss(0, 2)
        for s in range(4):
            rows.append((f"u{u}", s, 3.0 + offset + 0.5 * s + rng.gauss(0, 0.1)))
    r = stats.mixed_model_trend(rows)
    assert r["converged"] is True
    assert r["slope"] == pytest.approx(0.5, abs=0.1)
    assert r["p_value"] < 0.01
    assert r["n_obs"] == 40 and r["n_units"] == 10
    assert r["se"] > 0


def test_mixed_model_drops_missing_values():
    rows = [("a", 0, 1.0), ("a", 1, None), ("a", 2, float("nan")), ("b", 0, 2.0), ("b", 1, 3.0), ("c", 0, 1.0), ("c", 1, 2.0)]
    r = stats.mixed_model_trend(rows)
    assert r["n_obs"] == 5
    assert r["n_units"] == 3


@pytest.mark.parametrize("rows", [
    [],
    [("a", 0, 1.0)],
    [("a", 0, 1.0), ("a", 1, 2.0), ("a", 2, 3.0)],
    [("a", 0, None), ("b", 1, None)],
])
def test_mixed_model_too_little_data_does_not_raise(rows):
    r = stats.mixed_model_trend(rows)
    assert r["converged"] is False
    assert math.isnan(r["slope"]) and math.isnan(r["p_value"])


def test_mixed_model_constant_data_does_not_raise():
    rows = [(u, s, 1.0) for u in "abcd" for s in range(4)]
    r = stats.mixed_model_trend(rows)
    assert isinstance(r["converged"], bool)


# ---- percent agreement and Cohen's kappa ---------------------------------------------------------
def test_percent_agreement():
    assert stats.percent_agreement(["a", "b", "a", None], ["a", "a", "a", "b"]) == pytest.approx(2 / 3)
    assert math.isnan(stats.percent_agreement([None], ["a"]))
    with pytest.raises(ValueError):
        stats.percent_agreement(["a"], ["a", "b"])


def test_cohen_kappa_classic_example():
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert stats.cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_perfect_and_undefined():
    assert stats.cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    assert math.isnan(stats.cohen_kappa(["a", "a"], ["a", "a"]))
    with pytest.raises(ValueError):
        stats.cohen_kappa(["a"], [])


def test_cohen_kappa_ignores_none_pairs():
    assert stats.cohen_kappa(["a", "b", None, "a"], ["a", "b", "b", "a"]) == pytest.approx(1.0)


# ---- Fleiss' kappa -------------------------------------------------------------------------------
WIKI_FLEISS = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
               [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]


def test_fleiss_kappa_wikipedia_example():
    ratings = counts_to_ratings(WIKI_FLEISS)
    assert all(len(r) == 14 for r in ratings)
    assert round(stats.fleiss_kappa(ratings), 3) == 0.210


def test_fleiss_kappa_perfect_and_undefined():
    assert stats.fleiss_kappa([["a", "a", "a"], ["b", "b", "b"], ["a", "a", "a"]]) == pytest.approx(1.0)
    assert math.isnan(stats.fleiss_kappa([["a", "a"], ["a", "a"]]))
    assert math.isnan(stats.fleiss_kappa([]))


def test_fleiss_kappa_drops_irregular_units():
    regular = [["a", "a", "b"], ["b", "b", "b"], ["a", "a", "a"], ["a", "b", "b"]]
    noisy = regular + [["a", "b"], ["a", None, "b"]]
    assert stats.fleiss_kappa(noisy) == pytest.approx(stats.fleiss_kappa(regular))


# ---- Krippendorff's alpha ------------------------------------------------------------------------
def test_krippendorff_alpha_published_example():
    n = None
    obs = {
        "A": [1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n],
        "B": [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
        "C": [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n],
        "D": [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n],
    }
    ratings = [[obs[o][u] for o in "ABCD"] for u in range(12)]
    assert round(stats.krippendorff_alpha_nominal(ratings), 3) == 0.743


def test_krippendorff_alpha_perfect_ignores_single_value_units():
    ratings = [["a", "a", None], ["b", "b", "b"], [None, "a", None], ["a", None, "a"]]
    assert stats.krippendorff_alpha_nominal(ratings) == pytest.approx(1.0)


def test_krippendorff_alpha_undefined():
    assert math.isnan(stats.krippendorff_alpha_nominal([["a", "a"], ["a", "a"]]))
    assert math.isnan(stats.krippendorff_alpha_nominal([["a", None]]))


# ---- Gwet's AC1 ----------------------------------------------------------------------------------
def test_gwet_ac1_hand_computed():
    ratings = [["y", "y"]] * 40 + [["n", "n"]] * 40 + [["y", "n"]] * 10 + [["n", "y"]] * 10
    pa, pi_yes = 0.8, 0.5
    pe = pi_yes * (1 - pi_yes) + (1 - pi_yes) * pi_yes  # q = 2, 1 / (q - 1) = 1
    assert (pa - pe) / (1 - pe) == pytest.approx(0.6)
    assert stats.gwet_ac1(ratings) == pytest.approx(0.6)


def test_gwet_ac1_perfect_and_edge_cases():
    assert stats.gwet_ac1([["a", "a", "a"], ["b", "b", "b"]]) == pytest.approx(1.0)
    assert stats.gwet_ac1([["a", "a"], ["a", "a"]]) == pytest.approx(1.0)
    assert math.isnan(stats.gwet_ac1([]))
    assert math.isnan(stats.gwet_ac1([["a"], ["b"]]))


def test_gwet_ac1_resists_kappa_paradox():
    a = ["y"] * 90 + ["y"] * 5 + ["n"] * 5
    b = ["y"] * 90 + ["n"] * 5 + ["y"] * 5
    assert stats.gwet_ac1([[x, y] for x, y in zip(a, b)]) > stats.cohen_kappa(a, b)


def test_agreement_summary():
    a = ["y", "y", "n", "n", "y"]
    b = ["y", "n", "n", "n", "y"]
    s = stats.agreement_summary(a, b)
    assert set(s) == {"percent_agreement", "cohen_kappa", "gwet_ac1"}
    assert s["percent_agreement"] == pytest.approx(0.8)
    assert s["cohen_kappa"] == pytest.approx(stats.cohen_kappa(a, b))
    assert s["gwet_ac1"] == pytest.approx(stats.gwet_ac1([[x, y] for x, y in zip(a, b)]))


# ---- precision / recall --------------------------------------------------------------------------
def test_precision_recall_hand_computed():
    pred = ["a", "a", "b", "a", "b", None]
    truth = ["a", "b", "b", "a", "a", "a"]
    r = stats.precision_recall(pred, truth, "a")
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)


def test_precision_recall_undefined_and_errors():
    r = stats.precision_recall(["b", "b"], ["b", "b"], "a")
    assert math.isnan(r["precision"]) and math.isnan(r["recall"]) and math.isnan(r["f1"])
    r = stats.precision_recall(["b", "a"], ["a", "b"], "a")
    assert r["precision"] == 0 and r["recall"] == 0 and r["f1"] == 0
    with pytest.raises(ValueError):
        stats.precision_recall(["a"], [], "a")


def test_gwet_ac1_uses_the_number_of_possible_categories_when_given():
    ratings = [["a", "a"], ["b", "b"], ["a", "b"], ["a", "a"]]
    # pa = 0.75; pi_a = 0.625, pi_b = 0.375; q = 2: pe = 0.46875 -> 0.5294; q = 3: pe = 0.234375 -> 0.6735
    assert stats.gwet_ac1(ratings) == pytest.approx((0.75 - 0.46875) / (1 - 0.46875))
    assert stats.gwet_ac1(ratings, n_categories=3) == pytest.approx((0.75 - 0.234375) / (1 - 0.234375))
