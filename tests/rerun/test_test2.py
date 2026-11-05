import json

import numpy as np
import pytest

from rerun import benchmarks, phish_corpus, test2

MODELS = ("Logistic Regression", "Decision Tree")


def _write_bench_dir(tmp_path):
    out = tmp_path / "bench"
    out.mkdir()
    manifest = {"corpora": {}}
    for name in benchmarks.BENCHMARKS:
        recs = []
        for i in range(40):
            y = 1 if name in ("Nazario", "Nigerian_Fraud") else i % 2
            body = ("urgente password conferma account subito" if y else "riunione verbale consiglio orario") \
                + f" w{i % 5}"
            recs.append({"uid": f"{name}{i}", "benchmark": name, "subject": "s", "body": body, "label": y})
        path = out / f"{name}.jsonl"
        benchmarks._write_jsonl(path, recs)
        manifest["corpora"][name] = {"output_sha256": benchmarks.sha256_file(path)}
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def _write_corpora(tmp_path, names=("rq3_A_gptoss", "rq3_F_v2_gptoss", "rq3_F_v2_llama31")):
    d = tmp_path / "corpus"
    for n in names:
        rows = []
        for p in range(10):
            for y in (0, 1):
                body = ("urgente conferma password account ente entro oggi" if y else "riunione verbale consiglio sala") \
                    + f" p{p}"
                rows.append({"id": f"{n}{p}{y}", "person": f"{n[4]}-{p}", "case": n[4], "variant": "none",
                             "attack": "BEC", "kind": "positive" if y else "negative", "index": 0,
                             "generator": "g", "label": y, "subject": "s", "body": body})
        phish_corpus.write({"rows": rows, "report": {}}, n, d)
    return d


def test_select_set():
    names = ["rq3_A_gptoss", "rq3_F_v2_gptoss", "rq3_F_v2_llama31", "rq3_G_gptoss"]
    assert test2.select_set(names, "F_v2") == ["rq3_F_v2_gptoss", "rq3_F_v2_llama31"]
    assert test2.select_set(names, "all16") == names
    with pytest.raises(ValueError):
        test2.select_set(names, "x")


def test_fixed_vectorizer_caps_vocabulary_and_uses_unigrams():
    texts = [f"parola{i % 30} comune altra{i % 40} comune" for i in range(200)]
    v = test2.fixed_vectorizer(texts, max_features=10)
    assert len(v.vocabulary_) == 10
    assert all(" " not in t for t in v.vocabulary_)


def test_split_baseline_stratified_and_disjoint():
    recs = [{"id": i, "text": "t", "y": 1 if i % 5 == 0 else 0} for i in range(100)]
    tr, te = test2.split_baseline(recs, 0.2, 42)
    assert len(te) == 20 and sum(r["y"] for r in te) == 4
    assert not {r["id"] for r in tr} & {r["id"] for r in te}
    assert test2.split_baseline(recs, 0.2, 42)[1] == te


def test_score_returns_nan_auc_with_one_class():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression().fit(np.array([[0.0], [1.0], [0.1], [0.9]]), [0, 1, 0, 1])
    s = test2.score(clf, np.array([[0.9], [1.0]]), [1, 1])
    assert s["recall"] == 1.0 and s["n_positive"] == 2 and s["auc"] != s["auc"]


def test_run_end_to_end_small(tmp_path):
    bdir, cdir, out = _write_bench_dir(tmp_path), _write_corpora(tmp_path), tmp_path / "out"
    res = test2.run(bench_cap=None, out_dir=out, corpus_dir=cdir, bench_dir=bdir, models=MODELS)
    c = res["counts"]
    assert c["baseline"]["n_total"] == 240 and c["baseline"]["n_train"] == 192 and c["baseline"]["n_test"] == 48
    assert c["sets"]["F_v2"]["n_messages"] == 40 and c["sets"]["all16"]["n_messages"] == 60
    s = c["sets"]["all16"]
    assert s["n_train"] + s["n_test"] == 60 and s["n_test_persons"] >= 1
    for m in MODELS:
        r = res["results"][m]
        assert 0 <= r["baseline"]["f1"] <= 1
        for k in ("test1_all_generated", "test1_on_test2_split", "test2", "test2_model_on_baseline_test"):
            assert 0 <= r["sets"]["F_v2"][k]["accuracy"] <= 1
        assert r["sets"]["all16"]["test1_all_generated"]["n"] == 60
        assert r["sets"]["all16"]["test2"]["n"] == s["n_test"]
    assert json.loads((out / "test2.json").read_text())["settings"]["max_features"] == 5000
    again = test2.run(bench_cap=None, out_dir=out, corpus_dir=cdir, bench_dir=bdir, models=MODELS)
    assert again["settings"]["resumed_models"] == 2 and again["results"] == res["results"]


def test_default_corpus_dir_is_the_collected_corpora(tmp_path, monkeypatch):
    cdir = _write_corpora(tmp_path)
    monkeypatch.setattr(test2, "CORPUS_DIR", cdir)
    args, counts = test2.prepare(bench_cap=None, seed=42, corpus_dir=None, bench_dir=_write_bench_dir(tmp_path),
                                 which_sets=("F_v2",))
    assert counts["sets"]["F_v2"]["n_messages"] == 40
