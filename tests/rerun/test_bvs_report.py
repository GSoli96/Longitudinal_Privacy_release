"""Tests of rerun.bvs_report: batched vs single judging analysis (2x2 tables, McNemar, paired DIR, positions)."""
import pytest

from rerun import bvs_report as br

CRIT = ("plausible", "coherent", "realistic", "explicit")
J = ("j1", "j2", "j3")


def row(attr_id, batch, single):
    """batch/single: dict judge -> 4-tuple of yes/no."""
    return {"attr_id": attr_id, "batch": {j: list(batch[j]) for j in J}, "single": {j: list(single[j]) for j in J}}


def same(v):
    return {j: v for j in J}


YES4, NO4 = ("yes",) * 4, ("no",) * 4
DANGEROUS = ("yes", "yes", "yes", "no")


# ---- McNemar ---------------------------------------------------------------------------------------------------
def test_mcnemar_exact_values():
    assert br.mcnemar(0, 0) == 1.0
    assert br.mcnemar(5, 5) == pytest.approx(1.0)
    assert br.mcnemar(0, 10) == pytest.approx(2 * 0.5 ** 10)       # 0.00195
    assert br.mcnemar(1, 9) == pytest.approx(0.02148, abs=1e-4)
    assert br.mcnemar(3, 9) == br.mcnemar(9, 3)


# ---- 2x2 tables ------------------------------------------------------------------------------------------------
def test_table_by_judge_and_criterion():
    rows = [row("a1", same(YES4), same(YES4)), row("a2", same(YES4), same(NO4)),
            row("a3", same(NO4), same(YES4)), row("a4", same(NO4), same(NO4)), row("a5", same(YES4), same(YES4))]
    t = br.agreement_tables(rows)
    r = t["j1"]["plausible"]
    assert r["table"] == {"yes_yes": 2, "yes_no": 1, "no_yes": 1, "no_no": 1}     # batch/single
    assert r["percent_agreement"] == pytest.approx(3 / 5) and r["n_valid_pairs"] == 5
    assert r["mcnemar_p"] == pytest.approx(1.0) and r["bias"] == pytest.approx(0.0)
    assert "cohen_kappa" in r and "gwet_ac1" in r


def test_bias_sign_is_batch_minus_single_yes_rate():
    rows = [row(f"a{i}", same(YES4), same(NO4)) for i in range(8)] + [row("b", same(YES4), same(YES4))]
    r = br.agreement_tables(rows)["j2"]["explicit"]
    assert r["bias"] == pytest.approx(8 / 9) and r["mcnemar_p"] == pytest.approx(2 / 256)


def test_invalid_votes_are_left_out_of_the_pairs():
    rows = [row("a1", same(YES4), same(YES4)), row("a2", {**same(YES4), "j1": ("invalid",) * 4}, same(NO4))]
    r = br.agreement_tables(rows)["j1"]["plausible"]
    assert r["n_valid_pairs"] == 1


# ---- majority and metrics --------------------------------------------------------------------------------------
def test_majority_labels_of_three_judges_for_batch_and_single():
    rows = [row("a1", {"j1": YES4, "j2": YES4, "j3": NO4}, {"j1": NO4, "j2": NO4, "j3": YES4})]
    b = br.majority_labels(rows, "batch")
    s = br.majority_labels(rows, "single")
    assert b["a1"]["plausible"] == "yes" and s["a1"]["plausible"] == "no"


def test_dir_explicit_share_and_iqs_for_batch_and_single_with_a_bootstrap_difference():
    person = {f"a{i}": f"p{i % 3}" for i in range(12)}
    rows = []
    for i in range(12):
        batch = same(DANGEROUS) if i < 9 else same(YES4)          # batch: 9 dangerous, 3 explicit
        single = same(DANGEROUS) if i < 6 else same(YES4)         # single: 6 dangerous, 6 explicit
        rows.append(row(f"a{i}", batch, single))
    out = br.compare_metrics(rows, person, n_resamples=300, seed=1)
    assert out["batch"]["DIR"] == pytest.approx(9 / 12) and out["single"]["DIR"] == pytest.approx(6 / 12)
    assert out["batch"]["E"] == pytest.approx(3 / 12) and out["single"]["E"] == pytest.approx(6 / 12)
    d = out["difference"]["DIR"]
    assert d["estimate"] == pytest.approx(0.25) and d["lo"] <= 0.25 <= d["hi"]
    assert out["difference"]["E"]["estimate"] == pytest.approx(-0.25)
    assert out["n_attributes"] == 12 and out["n_persons"] == 3
    assert br.batching_rule(out, br.agreement_tables(rows), delta=0.03)["accepted"] is False


def test_batching_rule_accepts_when_dir_difference_is_small_and_includes_zero_and_ac1_is_high():
    rows = [row(f"a{i}", same(DANGEROUS), same(DANGEROUS)) for i in range(40)]
    person = {f"a{i}": f"p{i % 8}" for i in range(40)}
    out = br.compare_metrics(rows, person, n_resamples=200, seed=2)
    rule = br.batching_rule(out, br.agreement_tables(rows), delta=0.03, ac1_min=0.80)
    assert rule["accepted"] is True and rule["dir_difference_ok"] and rule["explicit_ac1_ok"]


# ---- positions ----------------------------------------------------------------------------------------------------
def test_disagreement_by_position_in_the_step_first_vs_second_half():
    step_size = {"s1": 6}
    ids = [f"s1-a{n:03d}" for n in range(1, 7)]
    rows = []
    for k, aid in enumerate(ids):
        # disagreement on the plausible criterion only in the second half (attributes 4-6)
        batch = same(YES4)
        single = same(("no",) + YES4[1:]) if k >= 3 else same(YES4)
        rows.append(row(aid, batch, single))
    pos = br.disagreement_by_position(rows, size_of=lambda aid: 6)
    assert pos["plausible"]["first_half"]["rate"] == 0.0 and pos["plausible"]["second_half"]["rate"] == 1.0
    assert pos["plausible"]["first_half"]["n"] == 9 and pos["plausible"]["second_half"]["n"] == 9   # 3 attrs x 3 judges
    assert pos["explicit"]["second_half"]["rate"] == 0.0


def test_position_is_read_from_the_attribute_id():
    assert br.position_of("C-abc123-s75-a007") == 7
    assert br.position_of("weird") is None
