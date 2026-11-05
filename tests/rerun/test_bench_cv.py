import json

import pytest

from rerun import bench_cv, benchmarks

MODELS = ["Logistic Regression", "Multinomial Naive Bayes"]


def _rec(bench, i, y):
    word = "urgente password" if y else "riunione verbale"
    return {"id": f"{bench}{i}{y}", "text": f"{word} testo{i % 7} {bench}", "y": y, "benchmark": bench}


def by_name(n=20):
    out = {}
    for b in ("CEAS_08", "Enron", "Ling", "SpamAssasin"):
        out[b] = [_rec(b, i, i % 2) for i in range(n)]
    out["Nazario"] = [_rec("Nazario", i, 1) for i in range(8)]
    out["Nigerian_Fraud"] = [_rec("Nigerian_Fraud", i, 1) for i in range(6)]
    return out


def test_borrow_same_size_other_sources_and_deterministic():
    d = by_name()
    a, counts = bench_cv.borrow(d, "Nazario", 0, 8, seed=42)
    b, _ = bench_cv.borrow(d, "Nazario", 0, 8, seed=42)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len(a) == 8 and sum(counts.values()) == 8
    assert all(r["y"] == 0 and r["benchmark"] != "Nazario" for r in a)
    assert set(counts) <= {"CEAS_08", "Enron", "Ling", "SpamAssasin"}


def test_borrow_fails_when_pool_too_small():
    with pytest.raises(ValueError):
        bench_cv.borrow(by_name(4), "Nazario", 0, 50)


def test_build_corpus_fills_the_missing_class_only():
    d = by_name()
    recs, borrowed = bench_cv.build_corpus(d, "Nazario")
    assert len(recs) == 16 and sum(r["y"] for r in recs) == 8 and borrowed
    recs2, borrowed2 = bench_cv.build_corpus(d, "Enron")
    assert len(recs2) == 20 and borrowed2 is None
    only_neg = {**d, "Ling": [_rec("Ling", i, 0) for i in range(5)]}
    recs3, b3 = bench_cv.build_corpus(only_neg, "Ling")
    assert sum(r["y"] for r in recs3) == 5 and len(recs3) == 10 and b3


def test_stratified_cap_keeps_proportions_and_seed():
    recs = [_rec("X", i, 1 if i % 4 == 0 else 0) for i in range(400)]     # 25% positives
    a = bench_cv.stratified_cap(recs, 100)
    b = bench_cv.stratified_cap(recs, 100)
    assert len(a) == 100 and sum(r["y"] for r in a) == 25
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len(bench_cv.stratified_cap(recs, None)) == 400
    assert len(bench_cv.stratified_cap(recs, 1000)) == 400


def test_fold_indices_are_stratified_and_disjoint():
    y = [0, 1] * 25
    folds = bench_cv.fold_indices(y, 5, 42)
    assert len(folds) == 5
    seen = []
    for tr, te in folds:
        assert not set(tr) & set(te)
        assert sum(y[i] for i in te) == 5
        seen += list(te)
    assert sorted(seen) == list(range(50))


def test_summarise_folds_mean_and_std():
    folds = [{"accuracy": 1.0, "precision": 1.0, "recall": 0.5, "f1": 0.6},
             {"accuracy": 0.8, "precision": 1.0, "recall": 0.7, "f1": 0.8}]
    s = bench_cv.summarise_folds(folds)
    assert s["accuracy"]["mean"] == pytest.approx(0.9) and s["accuracy"]["std"] == pytest.approx(0.1)
    assert s["f1"]["mean"] == pytest.approx(0.7)
    assert s["folds"] == folds


def test_fold_metrics_zero_division_is_zero():
    m = bench_cv.fold_metrics([0, 0, 1], [0, 0, 0])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["accuracy"] == pytest.approx(2 / 3)


def test_heatmap_rows_format_and_missing_cell():
    res = {"A": {"LR": {"f1": {"mean": 0.9912}}}, "B": {"LR": {"f1": {"mean": 1.0}}}}
    rows = bench_cv.heatmap_rows(res, ["A", "B"], ["LR", "Other"], {"LR": 0.5})
    assert rows[0] == r"LR & \gcell{0.991} & \gcell{1.000} & \gcell{0.500} \\"
    assert rows[1] == r"Other & -- & -- & -- \\"


def test_person_split_f1_reads_estimates_and_tolerates_missing(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"settings": {"person_split": {"parts": {"all": {"detectors": {
        "LR": {"metrics": {"as_generated": {"f1": {"estimate": 0.97}}}}, "x": {}}}}}}}))
    assert bench_cv.person_split_f1(p) == {"LR": 0.97}
    assert bench_cv.person_split_f1(tmp_path / "none.json") == {}
    assert bench_cv.person_split_f1(None) == {}


def _write_bench_dir(tmp_path):
    out = tmp_path / "bench"
    out.mkdir()
    manifest = {"corpora": {}}
    for name, rows in by_name(30).items():
        path = out / f"{name}.jsonl"
        recs = [{"uid": r["id"], "benchmark": name, "subject": "s", "body": r["text"], "label": r["y"]} for r in rows]
        benchmarks._write_jsonl(path, recs)
        manifest["corpora"][name] = {"output_sha256": benchmarks.sha256_file(path)}
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def test_run_end_to_end_small_resumes_and_writes_outputs(tmp_path):
    bdir = _write_bench_dir(tmp_path)
    out = tmp_path / "out"
    res = bench_cv.run(limit=40, folds=5, out_dir=out, bench_dir=bdir, person_split=None, models=MODELS)
    assert set(res["results"]) == set(benchmarks.BENCHMARKS)
    naz = res["corpora"]["Nazario"]
    assert naz["n_positive"] == naz["n_negative"] and naz["borrowed_by_source"]
    assert res["corpora"]["Enron"]["borrowed_by_source"] is None
    assert res["corpora"]["CEAS_08"]["n_rows"] <= 40
    cell = res["results"]["Enron"]["Logistic Regression"]
    assert len(cell["folds"]) == 5 and 0 <= cell["f1"]["mean"] <= 1
    txt = (out / "bench_cv_heatmap_rows.txt").read_text(encoding="utf-8").splitlines()
    assert txt[0].startswith("%") and len(txt) == 3
    assert txt[1].count(r"\gcell") == 6 and txt[1].endswith(r"-- \\")
    assert json.loads((out / "bench_cv.json").read_text())["settings"]["folds"] == 5
    again = bench_cv.run(limit=40, folds=5, out_dir=out, bench_dir=bdir, person_split=None, models=MODELS)
    assert again["settings"]["resumed_tasks"] == 6 * 2 * 5
    assert again["results"] == res["results"]
