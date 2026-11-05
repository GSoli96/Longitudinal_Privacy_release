import json
import math

import pytest

from rerun import rubric
from rerun.llm import LLMResult


class FakeClient:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
        self.calls.append((model, messages[0]["content"], options, schema, think))
        a = self.answer(model, messages[0]["content"])
        if a == "FAIL":
            return LLMResult(ok=False, error="http_500")
        if isinstance(a, str):
            return LLMResult(ok=True, text=a)
        return LLMResult(ok=True, json=a)


def rec(i, variant="explicit_inferred", attack="BEC", person=None, kind="positive"):
    return {"id": f"r{i:02d}", "person": person or f"p{i}", "case": "E", "variant": variant, "attack": attack,
            "kind": kind, "label": 1 if kind == "positive" else 0, "subject": f"s{i}", "body": f"corpo {i}"}


def ok(specific=3, plausible=4, refs=1):
    return {"specific": specific, "plausible": plausible, "refs": refs}


@pytest.mark.parametrize("obj,reason", [({"specific": 0, "plausible": 3, "refs": 0}, "out_of_range_specific"),
                                        ({"specific": 3, "plausible": 6, "refs": 0}, "out_of_range_plausible"),
                                        ({"specific": 3, "plausible": 3, "refs": 3}, "out_of_range_refs"),
                                        ({"specific": 3, "plausible": 3}, "bad_refs"),
                                        ({"specific": "3", "plausible": 3, "refs": 1}, "bad_specific"),
                                        ({"specific": True, "plausible": 3, "refs": 1}, "bad_specific")])
def test_parse_rejects_invalid_answers(obj, reason):
    assert rubric.parse(LLMResult(ok=True, json=obj), 2) == (None, reason)


def test_parse_accepts_embedded_json_and_failed_call():
    s, why = rubric.parse(LLMResult(ok=True, text='ecco: {"specific": 2, "plausible": 5, "refs": 2}'), 2)
    assert s == {"specific": 2, "plausible": 5, "refs": 2} and why == ""
    assert rubric.parse(LLMResult(ok=False, error="timeout"), 2) == (None, "call_failed:timeout")
    assert rubric.parse(LLMResult(ok=True, text="boh"), 2) == (None, "not_json")


def test_prompt_lists_profile_and_does_not_reveal_the_variant():
    p = rubric.build_prompt(rec(1), ["fa il pendolare", "ha due figli"])
    assert "- fa il pendolare" in p and "- ha due figli" in p and "da 0 a 2" in p
    assert "explicit" not in p and "inferred" not in p


def test_rate_scores_positives_only_with_panel_mean_and_counts_failures():
    records = [rec(1), rec(2), rec(3, kind="negative")]
    def answer(model, prompt):
        if "corpo 2" in prompt and model == "b":
            return "FAIL"
        return ok(5 if model == "a" else 3)
    out = rubric.rate(FakeClient(answer), ("a", "b"), records, {"p1": ["x"], "p2": ["x"]})
    rows = {r["id"]: r for r in out["rows"]}
    assert set(rows) == {"r01", "r02"}
    assert rows["r01"]["score"] == 4.0 and rows["r01"]["n_valid"] == 2
    assert rows["r02"]["score"] == 5.0 and rows["r02"]["n_valid"] == 1
    assert out["summary"]["errors"] == {"b:call_failed:http_500": 1} and out["summary"]["n_unscored"] == 0


def test_message_with_no_valid_judge_is_left_out_not_zero():
    out = rubric.rate(FakeClient(lambda m, p: "FAIL"), ("a",), [rec(1)], {})
    assert out["rows"][0]["score"] is None and out["summary"]["n_unscored"] == 1
    assert rubric.summarise_by_variant(out["rows"])["explicit_inferred"]["n_units"] == 0


def test_summarise_by_variant_means_per_person():
    rows = [{"id": "a", "person": "p1", "variant": "none", "attack": "BEC", "score": 1.0},
            {"id": "b", "person": "p1", "variant": "none", "attack": "SpearPhishing", "score": 3.0},
            {"id": "c", "person": "p2", "variant": "none", "attack": "BEC", "score": 4.0},
            {"id": "d", "person": "p1", "variant": "full", "attack": "BEC", "score": 5.0}]
    s = rubric.summarise_by_variant(rows, n_resamples=50)
    assert s["none"]["estimate"] == 3.0 and s["none"]["n_units"] == 2 and s["full"]["estimate"] == 5.0
    assert not math.isnan(s["none"]["lo"])


def test_human_sample_is_stratified_seeded_and_ignores_unscored():
    rows = [{"id": f"r{i:02d}", "person": f"p{i}", "variant": v, "attack": a, "score": 1.0 if i != 0 else None}
            for i, (v, a) in enumerate([("none", "BEC")] * 6 + [("full", "BEC")] * 3)]
    s1 = rubric.human_sample(rows, n_per_cell=2, seed=1)
    assert s1 == rubric.human_sample(list(reversed(rows)), n_per_cell=2, seed=1)
    assert len(s1) == 4 and "r00" not in s1
    assert len([i for i in s1 if int(i[1:]) >= 6]) == 2
    assert len(rubric.human_sample(rows, n_per_cell=10)) == 8


def test_main_writes_scores_without_message_text(tmp_path, monkeypatch, capsys):
    from rerun import phish_corpus

    class Fake(FakeClient):
        def __init__(self, *a, **k):
            super().__init__(lambda m, p: ok(4, 4, 1))

    monkeypatch.setattr("rerun.llm.LLMClient", Fake)
    rows = [{**rec(i, variant=v), "generator": "g", "index": 0} for i, v in enumerate(["none", "explicit_inferred"])]
    monkeypatch.setattr(phish_corpus, "load", lambda n, d=None: rows)
    inp = tmp_path / "in.jsonl"
    inp.write_text("\n".join(json.dumps({"person": f"p{i}", "explicit": ["a"], "inferred": ["b"]}) for i in range(2)),
                   encoding="utf-8")
    assert rubric.main(["--corpus", "c", "--input", str(inp), "--panel", "m1", "m2", "--out-dir", str(tmp_path)]) == 0
    text = (tmp_path / "c.rubric.json").read_text(encoding="utf-8")
    assert "corpo" not in text and '"score": 4.0' in text
    assert json.loads(text)["summary"]["n_scored"] == 2
