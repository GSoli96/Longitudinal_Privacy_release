"""Tests of rerun.uniqueness: keys, k per attribute, combination uniqueness, semantic keys, dangerous population."""
import numpy as np
import pytest

from rerun import uniqueness
from rerun.metrics import AttrRecord

YES = {"plausible": "yes", "coherent": "yes", "realistic": "yes", "explicit": "no"}
EXPL = {"plausible": "yes", "coherent": "yes", "realistic": "yes", "explicit": "yes"}


def rec(p, t, labels=YES, step=100, carried=False):
    return AttrRecord(p, step, t, dict(labels), carried)


def test_exact_keys_normalise_case_and_punctuation():
    k = uniqueness.exact_keys([rec("a", "Ruolo: Dirigente."), rec("a", "ruolo dirigente"), rec("b", "x")])
    assert k["a"] == {"ruolo dirigente"} and k["b"] == {"x"}


def test_summary_counts_shared_and_unique_keys():
    keys = {"a": {"x", "y"}, "b": {"x", "z"}, "c": {"x"}}
    s = uniqueness.summarise(keys)
    assert s["n_distinct_keys"] == 3 and s["keys_k1"] == pytest.approx(2 / 3)     # y, z are held by one person
    assert s["persons_with_unique_key"] == pytest.approx(2 / 3)                    # c only has the shared x
    assert s["combination_uniqueness"]["1"] == pytest.approx(2 / 3)
    assert s["combination_uniqueness"]["2"] == pytest.approx(2 / 3)                # a:{x,y}, b:{x,z} are unique pairs


def test_identical_sets_are_found():
    keys = {"a": {"x", "y"}, "b": {"x", "y"}, "c": {"z"}}
    s = uniqueness.summarise(keys)
    # a and b share every key, so no subset of theirs is unique; c is exposed by its single key
    assert s["identical_sets"] == pytest.approx(2 / 3) and s["combination_uniqueness"]["2"] == pytest.approx(1 / 3)


def test_persons_without_attributes_are_counted_apart():
    s = uniqueness.summarise({"a": {"x"}, "b": set()})
    assert s["n_persons"] == 2 and s["n_persons_with_attributes"] == 1 and s["n_persons_without"] == 1


def test_too_many_combinations_give_none_instead_of_a_slow_run():
    keys = {"a": {str(i) for i in range(30)}, "b": {"z"}}
    s = uniqueness.summarise(keys, r_max=3, max_combos=100)
    assert s["combination_uniqueness"]["3"] is None and s["combination_uniqueness"]["1"] is not None


def test_semantic_keys_merge_close_texts_across_persons():
    vecs = {"gatto nero": [1.0, 0.0], "gatto scuro": [0.99, 0.14], "auto": [0.0, 1.0]}

    def embed(texts):
        v = np.array([vecs[t] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    k = uniqueness.semantic_keys([rec("a", "gatto nero"), rec("b", "gatto scuro"), rec("b", "auto")], embed, 0.9)
    assert k["a"] & k["b"] and len(k["b"]) == 2


def test_analyse_splits_dangerous_and_uses_final_step_only():
    recs = [rec("a", "x"), rec("a", "y", EXPL), rec("b", "x"), rec("b", "old", step=50), rec("c", "z", {})]
    out = uniqueness.analyse(recs)
    assert out["all"]["exact"]["n_distinct_keys"] == 3          # x, y, z (the step-50 record is left out)
    assert out["dangerous"]["exact"]["n_distinct_keys"] == 1     # only x is plausible, coherent, realistic, not explicit
    assert out["dangerous"]["exact"]["n_persons_in_case"] == 3
    assert out["all"]["exact"]["persons_with_unique_key"] == pytest.approx(2 / 3)    # a has y, c has z, b only x
