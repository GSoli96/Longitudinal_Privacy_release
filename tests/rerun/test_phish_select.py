from rerun import phish_select as ps
from rerun.metrics import AttrRecord


def rec(person, text, step=100, p="yes", c="yes", r="yes", e="no", attr_id=None):
    return AttrRecord(person=person, step=step, text=text,
                      labels={"plausible": p, "coherent": c, "realistic": r, "explicit": e},
                      attr_id=attr_id or f"{person}-{text}")


def test_accepts_only_last_step_validated_attributes_and_counts_the_rest():
    recs = [rec("a", "x1"), rec("a", "x2", e="yes"), rec("a", "x3", p="no"), rec("a", "x4", r="no"),
            rec("a", "x5", c="undecided"), rec("a", "x6", step=75)]
    sel = ps.select_person(recs)
    assert [r.text for r in sel["inferred"]] == ["x1"]
    assert [r.text for r in sel["explicit"]] == ["x2"]
    k = sel["counts"]
    assert (k["accepted_inferred"], k["accepted_explicit"], k["rejected_plausible"], k["rejected_realistic"],
            k["undecided"]) == (1, 1, 1, 1, 1)
    assert sum(k.values()) == 5          # the step-75 record is not counted


def test_cap_is_applied_deterministically_and_counted():
    recs = [rec("a", f"attr {i}") for i in range(20)]
    s1 = ps.select_person(recs, cap=12, seed=7)
    s2 = ps.select_person(list(reversed(recs)), cap=12, seed=7)
    assert len(s1["inferred"]) == 12
    assert [r.text for r in s1["inferred"]] == [r.text for r in s2["inferred"]]
    assert s1["counts"]["beyond_cap_inferred"] == 8
    assert [r.text for r in ps.select_person(recs, cap=12, seed=8)["inferred"]] != [r.text for r in s1["inferred"]]


def test_duplicates_count_once_by_text_or_cluster():
    recs = [rec("a", "Vive a Roma", attr_id="1"), rec("a", "vive  a roma!", attr_id="2"), rec("a", "Altro", attr_id="3")]
    assert len(ps.select_person(recs)["inferred"]) == 2
    clustered = ps.select_person(recs, clusters={"1": 0, "2": 1, "3": 0})
    assert len(clustered["inferred"]) == 2 and clustered["counts"]["duplicate"] == 1


def test_every_input_person_is_in_the_output():
    recs = [rec("a", "x"), rec("zz", "y")]
    out = ps.select_all(recs, ["a", "b", "c"])
    assert set(out["by_person"]) == {"a", "b", "c"}
    s = out["summary"]
    assert s["n_persons"] == 3 and s["n_persons_without_attributes"] == 2
    assert s["records_of_unknown_persons"] == 1 and s["n_inferred"] == 1
    assert "text" not in str(s)
