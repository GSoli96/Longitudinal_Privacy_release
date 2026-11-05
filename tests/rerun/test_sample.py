"""Tests of rerun.sample: stratified sample for the human validation and the annotation sheets (codebook section 8)."""
import csv
import json
import random
from collections import Counter

import pytest

from rerun import sample

STEPS = (25, 50, 75, 100)


def make_rows(case, n_per_cell=30, seed=1):
    """Judged attributes with known outcomes: for each step, validated dangerous (1110), explicit (1111) and other
    rejected (0110) attributes, plus a few undecided."""
    rnd = random.Random(seed)
    rows = []
    k = 0
    for step in STEPS:
        for outcome, bits in (("dangerous", "1110"), ("explicit", "1111"), ("rejected", "0100")):
            for _ in range(n_per_cell):
                k += 1
                rows.append(sample.Candidate(case=case, person=f"{case}-p{rnd.randint(1, 20)}", step=step,
                                             attr_id=f"{case}-a{k:04d}", text=f"attributo {k}", category=f"cat{k % 3}",
                                             evidence=f"prova {k}", majority=bits, votes={"j1": bits}))
        rows.append(sample.Candidate(case=case, person=f"{case}-p1", step=step, attr_id=f"{case}-u{step}",
                                     text=f"incerto {step}", category="x", evidence="e", majority="11u0", votes={}))
    return rows


def test_outcome_classes():
    assert sample.outcome("1110") == "dangerous"
    assert sample.outcome("1111") == "explicit" and sample.outcome("0101") == "explicit"
    assert sample.outcome("0100") == "rejected" and sample.outcome("1100") == "rejected" and sample.outcome("0000") == "rejected"
    assert sample.outcome("11u0") == "undecided" and sample.outcome("uuuu") == "undecided"


def test_sample_size_per_case_and_balanced_over_steps_and_outcomes():
    rows = make_rows("C")
    s = sample.draw_sample(rows, per_case=56, seed=7)
    assert len(s) == 56 and len({c.attr_id for c in s}) == 56
    by_step = Counter(c.step for c in s)
    assert set(by_step) == set(STEPS) and max(by_step.values()) - min(by_step.values()) <= 2
    by_out = Counter(sample.outcome(c.majority) for c in s)
    assert by_out["dangerous"] >= 14 and by_out["explicit"] >= 14 and by_out["rejected"] >= 14
    cells = Counter((c.step, sample.outcome(c.majority)) for c in s)
    assert max(cells.values()) - min(v for k, v in cells.items() if k[1] != "undecided") <= 3


def test_small_cells_are_filled_from_the_others_and_the_total_is_kept():
    rows = [r for r in make_rows("C") if not (r.step == 25 and sample.outcome(r.majority) == "dangerous")]
    rows += make_rows("C")[:0]
    s = sample.draw_sample(rows, per_case=40, seed=3)
    assert len(s) == 40
    # the empty cell (step 25, dangerous) is simply absent, the others share its quota


def test_fewer_candidates_than_requested_returns_them_all():
    rows = make_rows("C", n_per_cell=1)
    s = sample.draw_sample(rows, per_case=500, seed=1)
    assert len(s) == len([r for r in rows if sample.outcome(r.majority) != "undecided"])   # undecided is never sampled
    assert all(sample.outcome(c.majority) != "undecided" for c in s)


def test_sample_is_deterministic_and_changes_with_the_seed():
    rows = make_rows("C")
    a = [c.attr_id for c in sample.draw_sample(rows, per_case=30, seed=5)]
    b = [c.attr_id for c in sample.draw_sample(rows, per_case=30, seed=5)]
    c = [c.attr_id for c in sample.draw_sample(rows, per_case=30, seed=6)]
    assert a == b and a != c


def test_sample_is_drawn_per_case_from_all_cases_together():
    rows = make_rows("C") + make_rows("E", seed=2) + make_rows("F", seed=3)
    s = sample.draw_sample_all_cases(rows, per_case=20, seed=1)
    assert Counter(c.case for c in s) == {"C": 20, "E": 20, "F": 20}


def test_no_attribute_of_one_person_dominates_the_sample():
    rows = make_rows("C")
    s = sample.draw_sample(rows, per_case=56, seed=2, max_per_person=3)
    assert max(Counter(c.person for c in s).values()) <= 3 and len(s) == 56


def test_item_ids_are_opaque_stable_and_do_not_reveal_the_person_or_the_case_attribute_id():
    c = make_rows("C")[0]
    i1, i2 = sample.item_id(c, salt="s"), sample.item_id(c, salt="s")
    assert i1 == i2 and sample.item_id(c, salt="other") != i1
    assert c.person not in i1 and c.attr_id not in i1 and i1.startswith("I-")


def test_annotation_sheet_has_the_codebook_columns_and_hides_the_judges(tmp_path):
    rows = make_rows("C")
    s = sample.draw_sample(rows, per_case=12, seed=1)
    paths = sample.write_annotation_files(s, tmp_path, salt="s", annotators=("A1", "A2"), calibration=4,
                                          step_text=lambda c: f"testo del passo di {c.attr_id}")
    sheet = list(csv.DictReader(open(paths["sheet_A1"], encoding="utf-8")))
    assert list(sheet[0]) == ["item_id", "case", "step", "attribute", "category", "evidence", "documents",
                              "plausible", "coherent", "realistic", "explicit", "supported", "special_category",
                              "notes", "annotator", "seconds"]
    assert len(sheet) == 12 and all(r["annotator"] == "A1" for r in sheet)
    assert all(r["plausible"] == "" and r["seconds"] == "" for r in sheet)
    text = open(paths["sheet_A1"], encoding="utf-8").read()
    assert "1110" not in text and "j1" not in text and "C-p" not in text      # no judge labels, no person ids
    key = [json.loads(x) for x in open(paths["key"], encoding="utf-8")]
    assert len(key) == 12 and {"item_id", "attr_id", "person", "majority", "votes"} <= set(key[0])


def test_documents_are_written_one_file_per_item_in_the_documents_folder(tmp_path):
    s = sample.draw_sample(make_rows("C"), per_case=6, seed=1)
    paths = sample.write_annotation_files(s, tmp_path, salt="s", annotators=("A1", "A2"), calibration=2,
                                          step_text=lambda c: f"testo {c.attr_id}")
    sheet = list(csv.DictReader(open(paths["sheet_A1"], encoding="utf-8")))
    for r in sheet:
        f = tmp_path / r["documents"]
        assert f.exists() and f.read_text(encoding="utf-8").startswith("testo ")


def test_calibration_subset_is_shared_by_both_annotators_and_flagged(tmp_path):
    s = sample.draw_sample(make_rows("C"), per_case=20, seed=1)
    paths = sample.write_annotation_files(s, tmp_path, salt="s", annotators=("A1", "A2"), calibration=5,
                                          step_text=lambda c: "t")
    a1 = list(csv.DictReader(open(paths["sheet_A1"], encoding="utf-8")))
    a2 = list(csv.DictReader(open(paths["sheet_A2"], encoding="utf-8")))
    cal = [json.loads(x) for x in open(paths["calibration"], encoding="utf-8")]
    assert len(cal) == 5 and {c["item_id"] for c in cal} <= {r["item_id"] for r in a1} & {r["item_id"] for r in a2}
    # same items for both annotators, in an order that differs per annotator
    assert {r["item_id"] for r in a1} == {r["item_id"] for r in a2}


def test_files_are_written_only_under_the_given_private_directory(tmp_path):
    s = sample.draw_sample(make_rows("C"), per_case=6, seed=1)
    out = tmp_path / "ann"
    sample.write_annotation_files(s, out, salt="s", annotators=("A1",), calibration=1, step_text=lambda c: "t")
    assert all(p.is_relative_to(out) for p in out.rglob("*"))


def test_candidates_are_built_from_stage_files():
    infer = [{"person": "C-p1", "step": 50, "carried": False, "attributes": [
        {"id": "C-p1-s50-a001", "text": "ha un ruolo", "category_hint": "role", "evidence": "e"}]},
        {"person": "C-p1", "step": 75, "carried": True, "attributes": []}]
    judge = [{"person": "C-p1", "step": 50, "attr_id": "C-p1-s50-a001", "votes": {"j1": ["yes", "yes", "yes", "no"]},
              "majority": {"plausible": "yes", "coherent": "yes", "realistic": "yes", "explicit": "no"}}]
    cands = sample.candidates_from_stages("C", infer, judge)
    assert len(cands) == 1 and cands[0].majority == "1110" and cands[0].step == 50 and cands[0].category == "role"
