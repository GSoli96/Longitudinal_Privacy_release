"""Tests of rerun.dedup_check: one pair per cluster, stratified draw, sheet and precision."""
import csv

from rerun import dedup_check


def infer(person, n):
    return [{"person": person, "step": 100, "attributes": [{"id": f"{person}-{i}", "text": f"t {person} {i}"}
                                                          for i in range(n)]}]


def clusters(person, groups):
    return [{"person": person, "attr_ids": [f"{person}-{i}" for i in g]} for g in groups]


def test_one_pair_per_cluster_with_at_least_two_members():
    out = dedup_check.merged_pairs("X", infer("p", 5), clusters("p", [[0, 1, 2], [3], [4, 3]]))
    assert len(out) == 2 and {p["cluster_size"] for p in out} == {3, 2}
    assert all(p["attr_a"] != p["attr_b"] for p in out)


def test_draw_is_deterministic_stratified_and_keeps_small_cases():
    by = {"A": [{"case": "A", "k": i} for i in range(90)], "B": [{"case": "B", "k": i} for i in range(10)],
          "C": [{"case": "C", "k": 0}]}
    a, b = dedup_check.draw(by, 20), dedup_check.draw(by, 20)
    assert a == b and len(a) == 20
    cnt = {c: sum(1 for p in a if p["case"] == c) for c in "ABC"}
    assert cnt["C"] >= 1 and cnt["A"] > cnt["B"]
    assert len(dedup_check.draw(by, 500)) == 101        # fewer pairs than asked: all of them


def test_sheet_and_precision(tmp_path):
    pairs = dedup_check.merged_pairs("X", infer("p", 4), clusters("p", [[0, 1], [2, 3]]))
    res = dedup_check.write_sheet(pairs, tmp_path, "salt")
    rows = list(csv.DictReader(open(res["sheet"], encoding="utf-8", newline="")))
    assert len(rows) == 2 and rows[0]["pair_id"].startswith("P-") and "person" not in rows[0]
    rows[0]["same"], rows[1]["same"] = "yes", "no"
    with open(res["sheet"], "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dedup_check.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    p = dedup_check.precision(res["sheet"])
    assert p["precision"] == 0.5 and p["n_answered"] == 2 and p["by_case"]["X"]["n"] == 2
