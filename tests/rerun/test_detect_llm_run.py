"""Tests of rerun.detect_llm_run: options parsing, checkpoint and resume per (corpus, mode), no text in the output."""
import json

import pytest

from rerun import detect_llm_run, phish_corpus
from rerun.llm import LLMResult

PHISH = "urgente: confermi subito le credenziali password entro oggi"
LEGIT = "la riunione del consiglio è fissata per martedì alle dieci"


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
        prompt = messages[0]["content"].split("Email:")[-1]
        self.calls.append((model, options))
        return LLMResult(ok=True, text="1" if "credenziali" in prompt else "0")


def rows(prefix, n=8):
    out = []
    for i in range(n):
        for label, text in ((1, PHISH), (0, LEGIT)):
            out.append({"id": f"{prefix}{i}-{label}", "person": f"p{i}", "case": "A", "generator": "g",
                        "variant": "none" if i % 2 else "explicit", "label": label, "subject": f"s{i}",
                        "body": f"{text} {i}"})
    return out


EXAMPLES = [{"id": "e1", "y": 1, "subject": "x", "body": PHISH}, {"id": "e2", "y": 0, "subject": "y", "body": LEGIT}]


@pytest.fixture(autouse=True)
def corpora(monkeypatch):
    monkeypatch.setattr(phish_corpus, "load", lambda n, d=None: rows(n))


def test_parse_options_converts_numbers_and_booleans():
    assert detect_llm_run.parse_options(["num_gpu=99", "temperature=0.5", "think=false", "x=abc"]) == {
        "num_gpu": 99, "temperature": 0.5, "think": False, "x": "abc"}


def test_run_scores_both_modes_passes_the_options_and_keeps_text_out_of_the_output(tmp_path):
    client = FakeClient()
    res = detect_llm_run.run(["c1", "c2"], model="m", examples=EXAMPLES, options_extra={"num_gpu": 99}, out_dir=tmp_path,
                             name="t", n_resamples=10, client=client)
    assert res["n_messages"] == 2 * 16 and res["modes_result"]["zero"]["n_scored"] == 32
    assert res["modes_result"]["few"]["parts"]["all"]["metrics"]["as_generated"]["f1"]["estimate"] == pytest.approx(1.0)
    assert "few/all" in res["alarms"] and "few/variant:none" in res["alarms"]
    assert all(o["num_gpu"] == 99 and o["num_predict"] == 16 for _, o in client.calls)
    text = (tmp_path / "t.json").read_text(encoding="utf-8")
    assert "credenziali" not in text and "riunione" not in text


def test_run_resumes_per_pair_and_ignores_a_checkpoint_of_another_configuration(tmp_path):
    first = FakeClient()
    detect_llm_run.run(["c1", "c2"], model="m", examples=EXAMPLES, out_dir=tmp_path, name="t", n_resamples=10, client=first)
    lines = (tmp_path / "t.parts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and len(first.calls) == 64
    (tmp_path / "t.parts.jsonl").write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")        # stopped before the last pair
    again = FakeClient()
    res = detect_llm_run.run(["c1", "c2"], model="m", examples=EXAMPLES, out_dir=tmp_path, name="t", n_resamples=10,
                             client=again)
    assert res["resumed_pairs"] == 3 and len(again.calls) == 16
    other = FakeClient()
    res = detect_llm_run.run(["c1", "c2"], model="m2", examples=EXAMPLES, out_dir=tmp_path, name="t", n_resamples=10,
                             client=other)
    assert res["resumed_pairs"] == 0 and len(other.calls) == 64


def test_few_shot_without_examples_is_refused(tmp_path):
    with pytest.raises(ValueError):
        detect_llm_run.run(["c1"], model="m", modes=("few",), out_dir=tmp_path, name="t", client=FakeClient())


def test_failed_and_unparseable_answers_are_counted_not_turned_into_labels(tmp_path):
    class Flaky(FakeClient):
        def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
            n = len(self.calls)
            self.calls.append((model, options))
            if n % 4 == 0:
                return LLMResult(ok=False, error="http_500")
            if n % 4 == 1:
                return LLMResult(ok=True, text="forse")
            return LLMResult(ok=True, text="1")
    res = detect_llm_run.run(["c1"], model="m", modes=("zero",), out_dir=tmp_path, name="t", n_resamples=10,
                             client=Flaky())
    cov = res["coverage"]["c1|zero"]
    assert cov["n_excluded"] == 8 and cov["n_scored"] == 8 and set(cov["errors"]) == {"call_failed:http_500", "unparseable"}


def test_a_pair_with_failed_calls_is_not_checkpointed_so_a_restart_retries_it(tmp_path):
    class Down(FakeClient):
        def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
            self.calls.append((model, options))
            return LLMResult(ok=False, error="no_valid_key")
    res = detect_llm_run.run(["c1"], model="m", modes=("zero",), out_dir=tmp_path, name="t", n_resamples=10, client=Down())
    assert res["pairs_not_checkpointed"] == ["c1|zero"] and not (tmp_path / "t.parts.jsonl").exists()
    ok = detect_llm_run.run(["c1"], model="m", modes=("zero",), out_dir=tmp_path, name="t", n_resamples=10,
                            client=FakeClient())
    assert ok["pairs_not_checkpointed"] == [] and len((tmp_path / "t.parts.jsonl").read_text().splitlines()) == 1


def test_small_detector_qwen25_is_a_local_model():
    from rerun import config
    assert config.is_local("qwen2.5:7b") and config.is_local("qwen2.5:7b-local")
