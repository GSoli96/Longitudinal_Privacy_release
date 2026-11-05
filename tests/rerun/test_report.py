"""Tests of rerun.report: tables from metrics files, empty cells for missing blocks, LaTeX escaping."""
import csv
import json

from rerun import report


def metrics(with_extra=True):
    m = {"persons": 3, "persons_in_cohort": 3,
         "by_step": {s: {"n_total": 10, "P": 0.9, "C": 0.8, "R": 0.7, "E": 0.1, "IQS": 0.6, "DIR": 0.5,
                         "hallucination": 0.01} for s in report.STEPS},
         "ci_final_step": {"DIR": {"estimate": 0.5, "lo": 0.4, "hi": 0.6}},
         "pes": {"confirmed": {s: {"mean": 4.0, "mean_per_1000_chars": 0.5} for s in report.STEPS}},
         "novelty_confirmed": {"25": None, "50": 0.9, "75": 0.8, "100": 0.7}, "novelty_exact": {"25": None}}
    if with_extra:
        m["trend"] = {"DIR": {"permutation": {"slope": -0.01, "p_value": 0.2, "n_units": 3},
                              "mixed_model": {"slope": -0.01, "p_value": 0.3, "converged": True}}}
        m["reidentification"] = {"all": {"exact": {"n_distinct_keys": 20, "persons_with_unique_key": 1.0, "keys_k1": 0.9,
                                                  "identical_sets": 0.0,
                                                  "combination_uniqueness": {"1": 1.0, "2": 1.0, "3": None}}}}
    return m


def test_rq4_table_has_one_row_per_detector_and_empty_cells_for_missing_metrics(tmp_path):
    ci = lambda x: {"estimate": x, "lo": x - 0.1, "hi": x + 0.1}
    run = {"settings": {"bench_to_generated": {"parts": {"all": {"detectors": {
        "LR": {"n_test": 20, "n_positive": 10, "alarm": True,
               "metrics": {"as_generated": {"f1": ci(0.9), "auc_roc": ci(0.95), "auprc": ci(0.9),
                                            "recall_at_fpr": ci(0.5)}}},
        "llm:zero": {"n_test": 0}}}}}}}
    (tmp_path / "rq4").mkdir()
    (tmp_path / "rq4" / "t.json").write_text(json.dumps(run), encoding="utf-8")
    head, rows = report.rq4(tmp_path)
    by = {r[3]: dict(zip(head, r)) for r in rows}
    assert by["LR"]["F1"] == "0.900" and by["LR"]["alarm"] is True and by["LR"]["AUC"] == "0.950"
    assert by["llm:zero"]["F1"] == "" and by["llm:zero"]["n_test"] == 0
    assert report.write_all(tmp_path, tmp_path)["rows"]["rq4"] == 2


def test_tables_are_written_and_missing_blocks_stay_empty(tmp_path):
    for c, extra in (("A", True), ("B", False)):
        (tmp_path / f"case{c}").mkdir()
        (tmp_path / f"case{c}" / "metrics.json").write_text(json.dumps(metrics(extra)), encoding="utf-8")
    res = report.write_all(tmp_path, tmp_path)
    assert res["cases"] == ["A", "B"] and res["rows"]["rq2_steps"] == 8 and res["rows"]["trend"] == 1
    rows = list(csv.DictReader(open(tmp_path / "tables" / "rq1_final.csv", encoding="utf-8")))
    assert rows[0]["DIR"] == "0.500" and rows[0]["DIR_lo"] == "0.400"
    reid = list(csv.DictReader(open(tmp_path / "tables" / "reidentification.csv", encoding="utf-8")))
    a_all = [r for r in reid if r["case"] == "A" and r["population"] == "all"][0]
    b_all = [r for r in reid if r["case"] == "B" and r["population"] == "all"][0]
    assert a_all["unique_by_up_to_2_keys"] == "1.000" and a_all["unique_by_up_to_3_keys"] == ""
    assert b_all["persons_with_unique_key"] == ""
    assert "persons\\_with\\_unique\\_key" in (tmp_path / "latex" / "reidentification.tex").read_text(encoding="utf-8")


def test_category_and_control_tables_read_their_files_and_leave_missing_cells_empty(tmp_path):
    (tmp_path / "caseA").mkdir()
    (tmp_path / "caseA" / "metrics_by_category.json").write_text(json.dumps({
        "step": 100, "n_records": 10,
        "categories": {"economic": {"n": 4, "n_complete": 4, "DIR": 0.25, "IQS": 0.5, "hallucination": 0.0},
                       "_special": {"n_special": 1, "share": 0.1, "DIR": None, "IQS": None, "hallucination": None}}}),
        encoding="utf-8")
    (tmp_path / "caseA" / "control_comparison.json").write_text(json.dumps({
        "n_persons": 20, "persons_without_pair": 2, "DIR_documents": {"mean": 0.5}, "DIR_control": {"mean": 0.2},
        "DIR_difference": {"estimate": 0.3, "lo": 0.2, "hi": 0.4}, "IQS": {"documents": 0.6, "control": 0.4},
        "explicit_share": {"documents": 0.1}}), encoding="utf-8")
    head, rows = report.categories(tmp_path)
    by = {r[1]: dict(zip(head, r)) for r in rows}
    assert by["economic"]["share_of_attributes"] == "0.400" and by["economic"]["DIR"] == "0.250"
    assert by["_special"]["n"] == 1 and by["_special"]["share_of_attributes"] == "0.100" and by["_special"]["DIR"] == ""
    head, rows = report.control(tmp_path)
    row = dict(zip(head, rows[0]))
    assert row["difference"] == "0.300" and row["difference_lo"] == "0.200" and row["explicit_control"] == ""
    out = report.write_all(tmp_path, tmp_path)["rows"]
    assert out["categories"] == 2 and out["control"] == 1


def test_ablation_tables_read_the_ablation_report_and_leave_missing_cells_empty(tmp_path):
    cond = {"persons_run": 50, "n_persons": 48, "DIR_baseline": {"mean": 0.5}, "DIR_variant": {"mean": 0.3},
            "DIR_reduction": {"estimate": 0.2, "lo": 0.1, "hi": 0.3}, "IQS": {"baseline": 0.6, "variant": 0.5},
            "info_kept": {"characters_pooled": 0.4, "documents_pooled": 0.25}}
    res = {"case": "E", "baseline_tag": "gloc",
           "conditions": {"docs:1": cond, "years:1": {"status": "not run"}},
           "repetitions": {"runs": ["gloc", "rep1"], "persons_in_all_runs": 50, "DIR_by_run": {"gloc": 0.5, "rep1": 0.52},
                           "DIR_sd_between_runs": 0.014, "DIR_range": 0.02,
                           "mean_abs_person_difference_between_runs": 0.1}}
    (tmp_path / "ablation").mkdir()
    (tmp_path / "ablation" / "E.json").write_text(json.dumps(res), encoding="utf-8")
    head, rows = report.ablations(tmp_path)
    by = {r[1]: dict(zip(head, r)) for r in rows}
    assert by["docs:1"]["reduction"] == "0.200" and by["docs:1"]["documents_kept"] == "0.250"
    assert by["years:1"]["reduction"] == "" and by["years:1"]["status"] == "not run"
    head, rows = report.ablation_repetitions(tmp_path)
    assert rows[0][3] == "gloc=0.500; rep1=0.520" and rows[0][4] == "0.014"
    out = report.write_all(tmp_path, tmp_path)["rows"]
    assert out["ablations"] == 2 and out["ablation_repetitions"] == 1
