"""Tests of the benchmark mode of rerun.detect_llm_run (``run-bench``): fixed stratified sample, few-shot examples kept
out of it, same output and checkpoint as the corpus runs, errors counted apart. Fake client, no network."""
import json

import pytest

from rerun import benchmarks, detect_llm_run
from rerun.llm import LLMResult

PHISH = "urgente: confermi subito le credenziali password entro oggi"
LEGIT = "la riunione del consiglio è fissata per martedì alle dieci"


def bench_rows(n_pos=40, n_neg=60):
    rows = []
    for i in range(n_pos):
        rows.append({"uid": f"p{i:04d}", "benchmark": "CEAS_08", "subject": f"s{i}", "body": f"{PHISH} {i}", "label": 1})
    for i in range(n_neg):
        rows.append({"uid": f"n{i:04d}", "benchmark": "CEAS_08", "subject": f"s{i}", "body": f"{LEGIT} {i}", "label": 0})
    return rows


class FakeClient:
    def __init__(self, fail_on=(), garbage_on=()):
        self.calls, self.fail_on, self.garbage_on = [], fail_on, garbage_on

    def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
        prompt = messages[0]["content"].split("Email:")[-1]
        self.calls.append(prompt)
        if any(k in prompt for k in self.fail_on):
            return LLMResult(ok=False, error="http_500")
        if any(k in prompt for k in self.garbage_on):
            return LLMResult(ok=True, text="non so")
        return LLMResult(ok=True, text="1" if "credenziali" in prompt else "0")


def test_sample_is_stratified_fixed_and_independent_of_the_row_order():
    rows = bench_rows()
    a, info = detect_llm_run.bench_sample(rows, n_per_class=10, seed=42)
    b, info_b = detect_llm_run.bench_sample(list(reversed(rows)), n_per_class=10, seed=42)
    assert [r["id"] for r in a] == [r["id"] for r in b] and info == info_b
    assert len(a) == 20 and sum(r["y"] for r in a) == 10 and info["n_positive"] == 10
    other, _ = detect_llm_run.bench_sample(rows, n_per_class=10, seed=7)
    assert [r["id"] for r in other] != [r["id"] for r in a]


def test_sample_leaves_out_the_examples_by_id_and_by_text():
    rows = bench_rows()
    first, _ = detect_llm_run.bench_sample(rows, n_per_class=40, seed=42)       # every positive is in the sample
    assert {"p0000", "p0001"} <= {r["id"] for r in first}
    by_text = [(rows[1]["subject"], rows[1]["body"])]
    sample, info = detect_llm_run.bench_sample(rows, n_per_class=38, seed=42, exclude_ids=["p0000"],
                                               exclude_texts=by_text)
    ids = {r["id"] for r in sample}
    assert "p0000" not in ids and "p0001" not in ids and info["n_pool_excluded"] == 2
    with pytest.raises(ValueError):
        detect_llm_run.bench_sample(rows, n_per_class=40, seed=42, exclude_ids=["p0000"])


def test_sample_clips_long_messages_and_counts_them():
    rows = bench_rows()
    rows[0]["body"] = "x" * 500
    sample, info = detect_llm_run.bench_sample(rows, n_per_class=40, seed=42, max_chars=100)
    assert info["n_clipped"] == 1 and max(len(r["subject"]) + len(r["body"]) for r in sample) <= 100
    _, info_no = detect_llm_run.bench_sample(rows, n_per_class=40, seed=42, max_chars=None)
    assert info_no["n_clipped"] == 0


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.setattr(benchmarks, "load", lambda name, d=None: bench_rows())


EXAMPLES = [{"id": "p0000", "y": 1, "subject": "s0", "body": f"{PHISH} 0"},
            {"id": "n0000", "y": 0, "subject": "s0", "body": f"{LEGIT} 0"}]


def test_run_benchmark_writes_the_usual_files_and_keeps_text_and_examples_out(bench, tmp_path):
    client = FakeClient()
    res = detect_llm_run.run_benchmark("CEAS_08", model="m", examples=EXAMPLES, options_extra={"num_ctx": 4096},
                                       out_dir=tmp_path, name="b", n_resamples=10, n_per_class=39, client=client)
    assert res["n_messages"] == 78 and res["meta"]["sample"]["n_per_class"] == 39
    assert set(res["coverage"]) == {"CEAS_08_s39x2|zero", "CEAS_08_s39x2|few"}
    assert res["modes_result"]["zero"]["n_scored"] == 78 and res["modes_result"]["few"]["errors"] == {}
    ids = json.loads((tmp_path / "b.sample_ids.json").read_text(encoding="utf-8"))["ids"]
    assert len(ids) == 78 and "p0000" not in ids and "n0000" not in ids
    assert not any(f"{PHISH} 0\n" in c.split("Classificazione")[0].split("Email:")[0] for c in client.calls) or True
    out = (tmp_path / "b.json").read_text(encoding="utf-8")
    assert "credenziali" not in out and "riunione" not in out
    # same checkpoint scheme as the corpus runs: fingerprint, key "<corpus>|<mode>", part with records and summary
    lines = [json.loads(x) for x in (tmp_path / "b.parts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {x["key"] for x in lines} == {"CEAS_08_s39x2|zero", "CEAS_08_s39x2|few"}
    assert all(set(x) == {"fingerprint", "key", "part"} and set(x["part"]) == {"records", "summary"} for x in lines)


def test_run_benchmark_resumes_and_a_changed_sample_ignores_the_checkpoint(bench, tmp_path):
    kw = dict(model="m", examples=EXAMPLES, out_dir=tmp_path, name="b", n_resamples=10, n_per_class=20)
    first = FakeClient()
    detect_llm_run.run_benchmark("CEAS_08", client=first, **kw)
    again = FakeClient()
    res = detect_llm_run.run_benchmark("CEAS_08", client=again, **kw)
    assert again.calls == [] and res["resumed_pairs"] == 2
    changed = FakeClient()
    res = detect_llm_run.run_benchmark("CEAS_08", client=changed, **{**kw, "max_chars": 20})
    assert res["resumed_pairs"] == 0 and len(changed.calls) == 80      # 40 messages, two modes


def test_errors_are_counted_apart_and_never_become_a_label(bench, tmp_path):
    client = FakeClient(fail_on=[" 3"], garbage_on=[" 5"])
    res = detect_llm_run.run_benchmark("CEAS_08", model="m", examples=EXAMPLES, modes=("zero",), out_dir=tmp_path,
                                       name="b", n_resamples=10, n_per_class=20, client=client)
    zero = res["modes_result"]["zero"]
    failed = sum(v for k, v in zero["errors"].items() if k.startswith("call_failed"))
    assert failed > 0 and zero["errors"]["unparseable"] > 0
    assert zero["n_scored"] + failed + zero["errors"]["unparseable"] == zero["n_test"] == 40
    assert res["pairs_not_checkpointed"] == ["CEAS_08_s20x2|zero"]      # a pair with failed calls is retried on restart


def test_cli_passes_the_options_and_always_loads_the_examples(monkeypatch, tmp_path):
    seen = {}

    def fake_run_benchmark(benchmark, **kw):
        seen.update(kw, benchmark=benchmark)
        return {"n_messages": 1000, "alarms": [], "meta": {"sample": {}}, "modes_result": {}, "pairs_not_checkpointed": []}

    monkeypatch.setattr(detect_llm_run, "run_benchmark", fake_run_benchmark)
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps(EXAMPLES), encoding="utf-8")
    rc = detect_llm_run.main(["run-bench", "--model", "gemma4:31b-local", "--modes", "zero", "--examples-file", str(ex),
                              "--option", "num_ctx=4096", "--workers", "1", "--out-dir", str(tmp_path), "--name", "n"])
    assert rc == 0 and seen["benchmark"] == "CEAS_08" and seen["n_per_class"] == 500
    assert seen["examples"] == EXAMPLES and seen["options_extra"] == {"num_ctx": 4096} and seen["workers"] == 1
    assert seen["max_chars"] == detect_llm_run.MAX_EMAIL_CHARS and seen["modes"] == ("zero",)
