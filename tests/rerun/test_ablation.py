"""Tests of rerun.ablation: document count, order, windows, generalisation, information kept, paired comparison."""
from datetime import date

import pytest

from rerun import ablation, steps
from rerun.corpus import Doc, Person


def person(pid="p1", dates=("2019-05-01", "2020-06-01", "2021-07-01", "2023-01-15", None), texts=None):
    docs = []
    for i, d in enumerate(dates):
        y = date.fromisoformat(d) if d else None
        docs.append(Doc(id=f"{pid}-d{i}", date=y, kind="pdf", text=(texts[i] if texts else f"documento {i} " * 10)))
    return Person(pid, "E", None, "ministry", docs)


def test_specs_are_parsed_and_bad_ones_rejected():
    assert ablation.parse("docs:4") == ("docs", 4) and ablation.parse("generalise") == ("generalise", None)
    for bad in ("docs", "docs:0", "docs:x", "unknown:1", "generalise:2"):
        with pytest.raises(ValueError):
            ablation.parse(bad)
    assert ablation.tag_for("docs:2") == "abl-docs2" and ablation.tag_for("years:3") == "mit-years3"


def test_first_n_documents_keep_the_order_and_short_persons_keep_all():
    p = person()
    out, info = ablation.apply([p, person("p2", dates=("2020-01-01",))], "docs:2")
    assert [d.id for d in out[0].docs] == ["p1-d0", "p1-d1"] and len(out[1].docs) == 1
    assert info["fewer_than_n"] == 1 and info["persons_out"] == 2


def test_steps_of_a_one_document_person_collapse_into_carried_steps():
    out, _ = ablation.apply([person()], "docs:1")
    st = steps.build_steps(out[0])
    assert [s.n_docs for s in st] == [1, 1, 1, 1] and [s.identical_to_prev for s in st] == [False, True, True, True]


def test_order_is_a_seeded_permutation_that_keeps_the_documents():
    p = person(dates=tuple(f"2020-0{i}-01" for i in range(1, 9)))
    a, _ = ablation.apply([p], "order:1")
    b, _ = ablation.apply([p], "order:1")
    c, _ = ablation.apply([p], "order:2")
    ids = lambda ps: [d.id for d in ps[0].docs]
    assert ids(a) == ids(b) and ids(a) != ids(c) or ids(a) != [d.id for d in p.docs]
    assert sorted(ids(a)) == sorted(d.id for d in p.docs)


def test_year_window_counts_back_from_the_latest_dated_document_and_drops_undated_ones():
    out, info = ablation.apply([person()], "years:3")
    assert [d.id for d in out[0].docs] == ["p1-d1", "p1-d2", "p1-d3"]    # 2020-06 is 2.6 years before 2023-01-15
    assert info["info_kept"]["documents_pooled"] == pytest.approx(3 / 5, abs=1e-3)
    out, _ = ablation.apply([person()], "years:1")
    assert [d.id for d in out[0].docs] == ["p1-d3"]


def test_person_without_dated_documents_is_excluded_and_counted():
    undated = person("p2", dates=(None, None))
    out, info = ablation.apply([person(), undated], "years:1")
    assert [p.id for p in out] == ["p1"] and info["persons_excluded_without_documents"] == 1
    assert info["persons_in"] == 2 and info["persons_out"] == 1


def test_last_k_documents_are_the_most_recent_ones():
    out, _ = ablation.apply([person()], "last:2")
    assert [d.id for d in out[0].docs] == ["p1-d3", "p1-d4"]


def test_generalisation_reduces_dates_amounts_and_identifiers():
    t = ("Nato il 12/03/1975. Compenso € 45.000,00 oltre a 1.200 euro. Contatto mario@example.org, 06 12345678. "
         "CF RSSMRA75C12H501X. Incarico dal 5 marzo 2021. IBAN IT60X0542811101000000123456.")
    g = ablation.generalise_text(t)
    assert "12/03/1975" not in g and "1975" in g and "2021" in g and "5 marzo" not in g
    assert "45.000" not in g and "10^4" in g and "10^3" in g
    assert "mario@example.org" not in g and "RSSMRA75C12H501X" not in g and "IT60X" not in g and "12345678" not in g


def test_generalisation_keeps_every_person_and_document():
    out, info = ablation.apply([person(texts=["Compenso € 5.000 il 01/02/2020"] * 5)], "generalise")
    assert len(out) == 1 and len(out[0].docs) == 5 and info["info_kept"]["documents_pooled"] == 1.0
    assert info["info_kept"]["characters_pooled"] <= 1.0


def test_info_kept_pools_and_averages_over_persons():
    a = person("a", dates=("2020-01-01", "2021-01-01"), texts=["x" * 100, "y" * 100])
    b = person("b", dates=("2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"), texts=["z" * 50] * 4)
    out, info = ablation.apply([a, b], "last:1")
    k = info["info_kept"]
    assert k["documents_pooled"] == pytest.approx(2 / 6, abs=1e-3)
    assert k["characters_pooled"] == pytest.approx(150 / 400, abs=1e-3)
    assert k["documents_mean_person"] == pytest.approx((0.5 + 0.25) / 2, abs=1e-3)


def test_comparison_is_the_paired_dir_reduction():
    def rows(pid, dirs):
        inf = [{"person": pid, "step": 100, "carried": False, "attributes": [
            {"id": f"{pid}-a{i}", "text": f"t{i}", "evidence": "", "category_hint": ""} for i in range(len(dirs))]}]
        jud = [{"person": pid, "step": 100, "attr_id": f"{pid}-a{i}", "votes": {},
                "majority": {"plausible": "yes", "coherent": "yes", "realistic": "yes", "explicit": "no" if d else "yes"}}
               for i, d in enumerate(dirs)]
        return inf, jud
    bi, bj, vi, vj = [], [], [], []
    for pid, b, v in (("p1", [1, 1, 1, 0], [1, 0, 0, 0]), ("p2", [1, 1], [1, 0])):
        i, j = rows(pid, b)
        bi += i
        bj += j
        i, j = rows(pid, v)
        vi += i
        vj += j
    out = ablation.compare(bi, bj, vi, vj, n_resamples=100)
    assert out["n_persons"] == 2
    assert out["DIR_baseline"]["mean"] == pytest.approx((0.75 + 1.0) / 2)
    assert out["DIR_variant"]["mean"] == pytest.approx((0.25 + 0.5) / 2)
    assert out["DIR_reduction"]["estimate"] == pytest.approx(((0.75 - 0.25) + (1.0 - 0.5)) / 2)


# ---- reports on stage directories ----------------------------------------------------------------------------------
import json  # noqa: E402


def write_run(base, tag, pid_dirs, persons=None, ablation=None):
    """One stage directory with infer.jsonl / judge.jsonl for the final step; ``pid_dirs``: {person: [1 dangerous | 0 explicit]}."""
    d = base / "caseX" / f"stages_{tag}"
    d.mkdir(parents=True)
    inf, jud = [], []
    for pid, dirs in pid_dirs.items():
        inf.append({"person": pid, "step": 100, "carried": False, "attributes": [
            {"id": f"{pid}-a{i}", "text": f"t{i}", "evidence": "", "category_hint": ""} for i in range(len(dirs))]})
        jud += [{"person": pid, "step": 100, "attr_id": f"{pid}-a{i}", "votes": {},
                 "majority": {"plausible": "yes", "coherent": "yes", "realistic": "yes", "explicit": "no" if x else "yes"}}
                for i, x in enumerate(dirs)]
    for name, rows in (("infer.jsonl", inf), ("judge.jsonl", jud)):
        (d / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    if persons is not None:
        (d / "condition.json").write_text(json.dumps({"persons": persons, "ablation": ablation or {}}), encoding="utf-8")


def test_report_compares_each_condition_with_the_baseline_on_its_own_persons(tmp_path):
    write_run(tmp_path, "gloc", {"p1": [1, 1, 1, 0], "p2": [1, 1], "p3": [0, 0]})
    write_run(tmp_path, "mit-last2", {"p1": [1, 0, 0, 0], "p2": [1, 0]}, persons=["p1", "p2"],
              ablation={"info_kept": {"documents_pooled": 0.4}, "persons_excluded_without_documents": 0})
    out = ablation.report("X", ["last:2", "docs:1"], "gloc", n_resamples=50, base=tmp_path)
    r = out["last:2"]
    assert r["n_persons"] == 2 and r["persons_run"] == 2          # p3 is not in the variant: not compared
    assert r["DIR_reduction"]["estimate"] == pytest.approx(((0.75 - 0.25) + (1.0 - 0.5)) / 2)
    assert r["info_kept"]["documents_pooled"] == 0.4
    assert out["docs:1"] == {"status": "not run"}


def test_variability_summarises_the_spread_between_repetitions(tmp_path):
    write_run(tmp_path, "gloc", {"p1": [1, 1, 0, 0], "p2": [1, 1]})
    write_run(tmp_path, "rep1", {"p1": [1, 0, 0, 0], "p2": [1, 1]})
    write_run(tmp_path, "rep2", {"p1": [1, 1, 1, 0], "p2": [1, 0]})
    v = ablation.variability("X", ["rep1", "rep2"], baseline_tag="gloc", base=tmp_path)
    assert v["persons_in_all_runs"] == 2 and v["runs"] == ["gloc", "rep1", "rep2"]
    assert v["DIR_by_run"] == {"gloc": 0.75, "rep1": 0.625, "rep2": 0.625}
    assert v["DIR_range"] == pytest.approx(0.125)
    assert v["mean_abs_person_difference_between_runs"] > 0


def test_variability_needs_two_runs(tmp_path):
    write_run(tmp_path, "gloc", {"p1": [1, 0]})
    assert ablation.variability("X", ["missing"], baseline_tag="gloc", base=tmp_path)["status"] == "fewer than two runs"


def test_full_condition_changes_nothing_and_has_its_own_tag():
    ps = [person()]
    out, info = ablation.apply(ps, "full")
    assert [d.id for d in out[0].docs] == [d.id for d in ps[0].docs] and ablation.tag_for("full") == "abl-full"
    assert info["info_kept"]["documents_pooled"] == 1.0


def test_final_only_leaves_a_single_step_with_all_the_documents(monkeypatch):
    monkeypatch.setattr(steps, "FRACTIONS", (100,))
    st = steps.build_steps(person())
    assert [s.k for s in st] == [100] and st[0].n_docs == 5
    assert [s.k for s in steps.build_steps(person(), fractions=(50, 100))] == [50, 100]      # explicit still wins
