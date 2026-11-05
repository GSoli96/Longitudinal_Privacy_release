import json

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from rerun import rq4_run
from rerun.llm import LLMResult

PHISH = "Gentile {n}, urgente: confermi subito le credenziali su Ente entro oggi {n}"
LEGIT = "Gentile {n}, la riunione del consiglio è fissata per martedì alle dieci in sala grande {n}"


def models():
    return {"LR": Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=200))])}


def corpus_rows(n=12):
    rows = []
    for i in range(n):
        for label, pool in ((1, PHISH), (0, LEGIT)):
            for variant in ("none", "explicit"):
                rows.append({"id": f"{i}-{label}-{variant}", "person": f"p{i}", "case": "AB"[i % 2],
                             "generator": "g1" if i < n // 2 else "g2", "variant": variant, "attack": "BEC",
                             "kind": "positive" if label else "negative", "label": label,
                             "subject": f"s{i}", "body": pool.format(n=f"P{i}")})
    return rows


def bench_rows():
    rows = []
    for i in range(10):
        rows.append({"uid": f"b{i}p", "benchmark": "Enron" if i % 2 else "Ling", "subject": "x",
                     "body": "urgente confermi subito le credenziali password " + str(i), "label": 1})
        rows.append({"uid": f"b{i}l", "benchmark": "Enron" if i % 2 else "Ling", "subject": "y",
                     "body": "riunione del consiglio alle dieci verbale " + str(i), "label": 0})
    return rows


def test_to_records_maps_fields_and_keeps_text_out_of_ids():
    r = rq4_run.to_records(corpus_rows(1))[0]
    assert r["y"] in (0, 1) and r["text"].startswith("Oggetto: s0") and r["person"] == "p0"
    b = rq4_run.to_records(bench_rows(), benchmark=True)[0]
    assert b["case"] in ("Enron", "Ling") and b["generator"] is None and b["person"] == "b0p"


def test_cap_benchmarks_caps_per_corpus_and_is_deterministic():
    recs = rq4_run.to_records(bench_rows(), benchmark=True)
    capped = rq4_run.cap_benchmarks(recs, 3, seed=1)
    assert len(capped) == 6 and capped == rq4_run.cap_benchmarks(recs, 3, seed=1)
    assert len(rq4_run.cap_benchmarks(recs, None)) == len(recs)


def test_person_split_has_baselines_and_clean_variants_and_flags_alarm():
    gen = rq4_run.to_records(corpus_rows())
    out = rq4_run.evaluate_setting("person_split", gen, [], models=models(), n_resamples=20)
    dets = out["parts"]["all"]["detectors"]
    assert {"LR", "style_only", "length_only", "clean:LR"} <= set(dets)
    assert dets["LR"]["alarm"] is True                       # separable toy data: F1 = 1
    assert set(dets["LR"]["metrics"]) == {"as_generated", "1:10", "1:100"}


def test_case_split_makes_one_part_per_case():
    gen = rq4_run.to_records(corpus_rows())
    out = rq4_run.evaluate_setting("case_split", gen, [], models=models(), n_resamples=10)
    assert set(out["parts"]) == {"case:A", "case:B"}
    assert out["parts"]["case:A"]["n_test"] == sum(1 for r in gen if r["case"] == "A")


def test_generator_split_trains_on_one_generator_and_tests_on_the_other():
    gen = rq4_run.to_records(corpus_rows())
    out = rq4_run.evaluate_setting("generator_split", gen, [], models=models(), n_resamples=10)
    assert set(out["parts"]) == {"train:g1->test:g2", "train:g2->test:g1"}
    part = out["parts"]["train:g1->test:g2"]
    assert part["n_train"] == sum(1 for r in gen if r["generator"] == "g1")


def test_bench_to_generated_parts_and_llm_detector_with_coverage():
    gen = rq4_run.to_records(corpus_rows())
    bench = rq4_run.to_records(bench_rows(), benchmark=True)

    class Client:
        def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
            p = messages[0]["content"].rsplit("Email:", 1)[-1]
            if "P3" in p:
                return LLMResult(ok=False, error="http_500")
            return LLMResult(ok=True, text="1" if "urgente" in p else "0")

    out = rq4_run.evaluate_setting("bench_to_generated", gen, bench, models=models(), n_resamples=10,
                                   llm_client=Client(), llm_model="m")
    assert set(out["parts"]) == {"all", "variant:explicit", "variant:none"}
    d = out["parts"]["all"]["detectors"]
    assert "llm:zero" in d and "llm:few" in d
    cov = d["llm:zero"]["coverage"]
    assert cov["n_excluded"] == 4 and cov["errors"] == {"call_failed:http_500": 4}
    assert d["llm:zero"]["n_test"] == cov["n_scored"]


def test_unknown_setting_raises():
    with pytest.raises(ValueError):
        rq4_run.evaluate_setting("nope", [], [], models=models())


def test_run_writes_metrics_only_and_lists_alarms(tmp_path, monkeypatch):
    monkeypatch.setattr(rq4_run.phish_corpus, "load", lambda n, d=None: corpus_rows())
    monkeypatch.setattr(rq4_run.benchmarks, "load", lambda n, d=None: [r for r in bench_rows()
                                                                      if r["benchmark"] == n])
    monkeypatch.setattr(rq4_run.detect_classic, "classical_models", models)
    res = rq4_run.run(["c"], settings=("person_split",), n_resamples=10, out_dir=tmp_path, name="t")
    text = (tmp_path / "t.json").read_text(encoding="utf-8")
    assert "person_split/all/LR" in res["alarms"]
    assert "confermi subito" not in text and "riunione" not in text         # no message text in the output
    assert json.loads(text)["n_messages"] == len(corpus_rows())
    assert res["artefacts"]["by_class"]["1"]["organisation"] > 0


def test_run_resumes_from_the_parts_already_written_and_ignores_a_checkpoint_of_another_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(rq4_run.phish_corpus, "load", lambda n, d=None: corpus_rows())
    monkeypatch.setattr(rq4_run.benchmarks, "load", lambda n, d=None: [r for r in bench_rows()
                                                                      if r["benchmark"] == n])
    monkeypatch.setattr(rq4_run.detect_classic, "classical_models", models)
    calls = []
    real = rq4_run.detect_classic.evaluate
    monkeypatch.setattr(rq4_run.detect_classic, "evaluate", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = rq4_run.run(["c"], settings=("person_split", "case_split"), n_resamples=10, out_dir=tmp_path, name="t")
    n_first = len(calls)
    lines = (tmp_path / "t.parts.jsonl").read_text(encoding="utf-8").splitlines()
    assert n_first > 0 and len(lines) == 3 and first["resumed_parts"] == 0         # all + case:A + case:B
    # a stop after the first part: keep only that line, the rest must be computed again and the result must not change
    (tmp_path / "t.parts.jsonl").write_text(lines[0] + "\n", encoding="utf-8")
    del calls[:]
    again = rq4_run.run(["c"], settings=("person_split", "case_split"), n_resamples=10, out_dir=tmp_path, name="t")
    assert again["resumed_parts"] == 1 and 0 < len(calls) < n_first
    assert again["settings"] == first["settings"]
    assert len((tmp_path / "t.parts.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    # nothing left to compute: no model is fitted
    del calls[:]
    done = rq4_run.run(["c"], settings=("person_split", "case_split"), n_resamples=10, out_dir=tmp_path, name="t")
    assert calls == [] and done["resumed_parts"] == 3
    # another configuration (resamples) does not reuse the checkpoint
    other = rq4_run.run(["c"], settings=("person_split",), n_resamples=11, out_dir=tmp_path, name="t")
    assert other["resumed_parts"] == 0


def test_only_parts_computes_the_named_parts_and_a_later_full_run_finds_them_done(tmp_path, monkeypatch):
    monkeypatch.setattr(rq4_run.phish_corpus, "load", lambda n, d=None: corpus_rows())
    monkeypatch.setattr(rq4_run.detect_classic, "classical_models", models)
    part = rq4_run.run(["c"], settings=("case_split",), n_resamples=10, out_dir=tmp_path, name="t", only_parts=("case:A",))
    assert list(part["settings"]["case_split"]["parts"]) == ["case:A"]
    assert len((tmp_path / "t.parts.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    full = rq4_run.run(["c"], settings=("case_split",), n_resamples=10, out_dir=tmp_path, name="t")
    assert full["resumed_parts"] == 1 and sorted(full["settings"]["case_split"]["parts"]) == ["case:A", "case:B"]
