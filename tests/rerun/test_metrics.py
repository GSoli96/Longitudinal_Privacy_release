"""Tests of rerun.metrics on toy data with hand-computed results (definitions of the paper, spec 4.5 and 10b.6)."""
import math

import numpy as np
import pytest

from rerun import metrics
from rerun.metrics import AttrRecord


def labels(bits):
    """'1110' -> P=yes C=yes R=yes E=no (order P C R E); 'u' marks an undecided criterion."""
    names = ("plausible", "coherent", "realistic", "explicit")
    return {n: {"1": "yes", "0": "no", "u": "undecided"}[b] for n, b in zip(names, bits)}


def rec(person, step, text, bits, carried=False):
    return AttrRecord(person=person, step=step, text=text, labels=labels(bits), carried=carried)


# four attributes: PCRE = 1110, 1101, 0000, 1010
BASE = [rec("p1", 100, "a1", "1110"), rec("p1", 100, "a2", "1101"), rec("p2", 100, "a3", "0000"),
        rec("p2", 100, "a4", "1010")]


# ---- rates, IQS, DIR, hallucination ----------------------------------------------------------------------
def test_rates_iqs_dir_hallucination_by_hand():
    m = metrics.basic_metrics(BASE)
    assert m["n_total"] == 4 and m["n_complete"] == 4 and m["n_undecided"] == 0
    assert m["P"] == pytest.approx(3 / 4) and m["C"] == pytest.approx(2 / 4)
    assert m["R"] == pytest.approx(2 / 4) and m["E"] == pytest.approx(1 / 4)
    assert m["IQS"] == pytest.approx((0.75 + 0.5 + 0.5 + 2 * 0.25) / 5)   # = 0.45
    assert m["DIR"] == pytest.approx(1 / 4)       # only a1: P, C, not E
    assert m["hallucination"] == pytest.approx(1 / 4)   # only a3: rejected on all four


def test_iqs_restricted_to_non_explicit_attributes():
    m = metrics.basic_metrics(BASE)
    # non-explicit: a1 (1110), a3 (0000), a4 (1010): P=2/3 C=1/3 R=2/3 E=0
    assert m["IQS_non_explicit"] == pytest.approx((2 / 3 + 1 / 3 + 2 / 3) / 5)
    assert m["n_non_explicit"] == 3


def test_undecided_attributes_are_excluded_and_counted():
    recs = BASE + [rec("p3", 100, "a5", "11u0"), rec("p3", 100, "a6", "uuuu")]
    m = metrics.basic_metrics(recs)
    assert m["n_total"] == 6 and m["n_complete"] == 4 and m["n_undecided"] == 2
    assert m["DIR"] == pytest.approx(1 / 4)       # denominators use the complete attributes only
    assert m["undecided_share"] == pytest.approx(2 / 6)


def test_empty_input_gives_nan_not_an_error():
    m = metrics.basic_metrics([])
    assert m["n_total"] == 0 and math.isnan(m["DIR"]) and math.isnan(m["IQS"])


def test_all_sixteen_label_combinations_are_reported():
    combos = metrics.label_combinations(BASE)
    assert len(combos) == 16 and set(combos) == {f"{i:04b}" for i in range(16)}
    assert sum(c["count"] for c in combos.values()) == 4
    assert combos["1110"]["count"] == 1 and combos["0000"]["count"] == 1 and combos["0100"]["count"] == 0
    assert combos["0000"]["share"] == pytest.approx(0.25)
    assert sum(c["share"] for c in combos.values()) == pytest.approx(1.0)


# ---- novelty ---------------------------------------------------------------------------------------------
def test_exact_novelty_per_step_by_hand():
    recs = [rec("p", 25, "x", "1110"), rec("p", 25, "y", "1110"),
            rec("p", 50, "Y ", "1110"), rec("p", 50, "z", "1110"),          # "Y " equals "y": old; z new
            rec("p", 75, "z", "1110"), rec("p", 75, "w", "1110"), rec("p", 75, "X", "1110")]  # w new
    nov = metrics.novelty_exact(recs)
    assert nov[25] is None                                   # first step: nothing to compare with
    assert nov[50] == pytest.approx(1 / 2)
    assert nov[75] == pytest.approx(1 / 3)


def test_novelty_is_pooled_over_persons_and_ignores_carried_steps():
    recs = [rec("p", 25, "x", "1110"), rec("q", 25, "x", "1110"),
            rec("p", 50, "y", "1110"), rec("q", 50, "x", "1110"),        # p new, q old (same person's own history)
            rec("p", 50, "y", "1110", carried=True)]                    # carried copy: not a new inference
    nov = metrics.novelty_exact(recs)
    assert nov[50] == pytest.approx(1 / 2)


class FakeEmbed:
    """Maps a text to a vector by prefix: texts starting with the same letter are close."""
    VEC = {"a": [1.0, 0.0, 0.0], "a2": [0.95, 0.312, 0.0], "a3": [0.8, 0.6, 0.0],
           "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]}

    def __call__(self, texts):
        out = np.array([self.VEC[t] for t in texts], dtype=float)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_semantic_novelty_thresholds():
    recs = [rec("p", 25, "a", "1110"), rec("p", 50, "a2", "1110"), rec("p", 50, "b", "1110")]
    # cos(a, a2) = 0.95 / norm = 0.95 / 1.0 ~ 0.95; cos(a, b) = 0
    n = metrics.novelty_semantic(recs, FakeEmbed(), thresholds=(0.80, 0.90, 0.99))
    assert n[0.80][50] == pytest.approx(1 / 2) and n[0.90][50] == pytest.approx(1 / 2)
    assert n[0.99][50] == pytest.approx(1.0)                 # a2 is not close enough at 0.99
    assert n[0.80][25] is None


# ---- semantic clusters ------------------------------------------------------------------------------------
def test_clusters_join_close_attributes_and_keep_far_ones_apart():
    recs = [rec("p", 25, "a", "1110"), rec("p", 50, "a2", "1110"), rec("p", 50, "b", "1110"),
            rec("p", 75, "c", "1110")]
    cl = metrics.semantic_clusters(recs, FakeEmbed(), threshold=0.85)
    assert sorted(len(c) for c in cl) == [1, 1, 2]
    assert metrics.semantic_clusters(recs, FakeEmbed(), threshold=0.99)[0] and len(
        metrics.semantic_clusters(recs, FakeEmbed(), threshold=0.99)) == 4


def test_clusters_compare_with_the_first_member_not_by_chaining():
    # a ~ a2 (0.95), a2 ~ a3 (>0.85) but a !~ a3 (0.8): a3 must NOT be chained into the cluster of a
    recs = [rec("p", 25, "a", "1110"), rec("p", 50, "a2", "1110"), rec("p", 75, "a3", "1110")]
    cl = metrics.semantic_clusters(recs, FakeEmbed(), threshold=0.85)
    assert sorted(len(c) for c in cl) == [1, 2]


# ---- PES ---------------------------------------------------------------------------------------------------
STEP_CHARS = {25: 1000, 50: 2000, 75: 3000, 100: 4000}


class PesEmbed:
    """a and a' are the same attribute; b and b' the same; c, d distinct."""
    VEC = {"A": [1, 0, 0, 0, 0], "A'": [1, 0.05, 0, 0, 0], "B": [0, 1, 0, 0, 0], "B'": [0, 1, 0.05, 0, 0],
           "C": [0, 0, 0, 1, 0], "D": [0, 0, 0, 0, 1]}

    def __call__(self, texts):
        out = np.array([self.VEC[t] for t in texts], dtype=float)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def pes_person():
    return [rec("p", 25, "A", "1110"), rec("p", 25, "B", "1101"),     # step 25: A valid, B explicit
            rec("p", 50, "A'", "1100"), rec("p", 50, "C", "1110"),    # step 50: A' now not realistic
            rec("p", 75, "D", "1110"), rec("p", 75, "B'", "1110"),    # step 75: B' valid and not explicit
            ]


def test_pes_uses_the_latest_occurrence_of_each_cluster_by_hand():
    pes = metrics.pes_by_step(pes_person(), PesEmbed(), STEP_CHARS, threshold=0.85)
    # step 25: A passes (1) -> 1 ; step 50: A's latest occurrence A' fails, C passes -> 1;
    # step 75: A' fails, B' passes (B failed earlier but B' is the latest occurrence), C, D pass -> 3; step 100: 3
    assert [pes[k]["pes"] for k in (25, 50, 75, 100)] == [1, 1, 3, 3]
    assert [pes[k]["per_1000_chars"] for k in (25, 50, 75, 100)] == pytest.approx([1.0, 0.5, 1.0, 0.75])


def test_pes_counts_each_cluster_once_even_with_many_paraphrases():
    recs = [rec("p", 25, "A", "1110"), rec("p", 25, "A'", "1110"), rec("p", 50, "A'", "1110")]
    pes = metrics.pes_by_step(recs, PesEmbed(), STEP_CHARS)
    assert pes[25]["pes"] == 1 and pes[100]["pes"] == 1


def test_pes_undecided_representative_does_not_pass_and_is_counted():
    recs = [rec("p", 25, "A", "11u0"), rec("p", 25, "C", "1110")]
    pes = metrics.pes_by_step(recs, PesEmbed(), STEP_CHARS)
    assert pes[25]["pes"] == 1 and pes[25]["undecided_clusters"] == 1


def test_cohort_pes_averages_over_all_persons_including_those_without_attributes():
    p1 = pes_person()
    p2 = [rec("q", 25, "C", "1110")]
    cohort = metrics.cohort_pes(p1 + p2, ["p", "q", "r"], PesEmbed(), STEP_CHARS)
    # step 100: p = 3, q = 1, r = 0 (no attributes at all) -> mean 4/3
    assert cohort[100]["mean"] == pytest.approx(4 / 3)
    assert cohort[100]["n_persons"] == 3 and cohort[100]["max"] == 3
    assert cohort[100]["median"] == 1


# ---- per-step metrics with carry-forward ------------------------------------------------------------------------
def test_step_metrics_keep_carried_persons_in_every_step():
    recs = [rec("p", 25, "A", "1110"), rec("p", 50, "A", "1110", carried=True),
            rec("q", 25, "C", "0000"), rec("q", 50, "D", "1110")]
    sm = metrics.step_metrics(recs)
    assert sm[25]["n_total"] == 2 and sm[50]["n_total"] == 2
    assert sm[50]["n_carried"] == 1 and sm[25]["n_carried"] == 0
    assert sm[25]["DIR"] == pytest.approx(1 / 2) and sm[50]["DIR"] == pytest.approx(1.0)
    assert sm[50]["n_persons"] == 2


def test_per_person_values_for_bootstrap_and_trend():
    recs = [rec("p", 25, "A", "1110"), rec("p", 25, "B", "0000"), rec("p", 50, "A", "1110")]
    per = metrics.per_person_step_values(recs, "DIR", persons=["p", "z"], steps=(25, 50))
    assert per["p"] == pytest.approx([0.5, 1.0])
    assert per["z"] == [None, None]                          # no attribute: undefined, not zero


# ---- clusters given by another de-duplication (confirmed by an LLM) -----------------------------------------------
def test_pes_with_given_clusters_keeps_unmerged_near_misses_apart():
    recs = [rec("p", 25, "A", "1110"), rec("p", 50, "A'", "1110")]
    merged_by_cosine = metrics.pes_by_step(recs, PesEmbed(), STEP_CHARS, threshold=0.85)
    assert merged_by_cosine[50]["pes"] == 1                              # cosine alone merges A and A'
    confirmed = [[recs[0]], [recs[1]]]                                   # the judge said: different facts
    given = metrics.pes_by_step(recs, PesEmbed(), STEP_CHARS, clusters=confirmed)
    assert given[50]["pes"] == 2 and given[25]["pes"] == 1


def test_cohort_pes_accepts_clusters_by_person():
    recs = [rec("p", 25, "A", "1110"), rec("p", 50, "A'", "1110"), rec("q", 25, "C", "1110")]
    cl = {"p": [[recs[0]], [recs[1]]], "q": [[recs[2]]]}
    cohort = metrics.cohort_pes(recs, ["p", "q"], PesEmbed(), STEP_CHARS, clusters_by_person=cl)
    assert cohort[100]["mean"] == pytest.approx((2 + 1) / 2)


def test_novelty_from_clusters_counts_a_member_of_an_older_cluster_as_old():
    x, y, x2 = rec("p", 25, "x", "1110"), rec("p", 50, "y", "1110"), rec("p", 75, "x'", "1110")
    nov = metrics.novelty_from_clusters([x, y, x2], {"p": [[x, x2], [y]]})
    assert nov[25] is None and nov[50] == pytest.approx(1.0) and nov[75] == pytest.approx(0.0)


def test_novelty_from_clusters_pooled_over_persons_and_ignores_carried():
    a1, a2 = rec("p", 25, "a", "1110"), rec("p", 50, "b", "1110")
    b1, b2 = rec("q", 25, "c", "1110"), rec("q", 50, "c'", "1110")
    carried = rec("p", 50, "a", "1110", carried=True)
    nov = metrics.novelty_from_clusters([a1, a2, b1, b2, carried], {"p": [[a1], [a2]], "q": [[b1, b2]]})
    assert nov[50] == pytest.approx(1 / 2)          # b new (p), c' old (q)


def test_dumps_strict_writes_null_for_nan_and_infinity_and_is_valid_json():
    import json
    text = metrics.dumps_strict({"a": float("nan"), "b": [1.5, float("inf")], "c": {"d": np.float64("nan"), "e": 2}})
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": None, "b": [1.5, None], "c": {"d": None, "e": 2}}
