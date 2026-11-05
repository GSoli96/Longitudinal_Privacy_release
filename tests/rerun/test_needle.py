"""Tests of rerun.needle (needle-in-a-haystack check of the usable context), with a fake client."""
import json

import pytest

from rerun import needle
from rerun.llm import LLMResult


class FakeClient:
    """Answers like a model that reads the whole prompt (or only its first/last part)."""

    def __init__(self, keep="all", tokens_in=None):
        self.keep, self.tokens_in, self.calls = keep, tokens_in, []

    def chat(self, model, messages, options=None, schema=None):
        self.calls.append({"model": model, "options": options, "schema": schema})
        prompt = messages[-1]["content"]
        if self.keep == "tail":           # silent truncation of the head
            prompt = prompt[-len(prompt) // 2:]
        if self.keep == "head":
            prompt = prompt[: len(prompt) // 2]
        codes = [c for c in needle.CODE_RE.findall(prompt)]
        return LLMResult(text=json.dumps({"codes": codes}), json={"codes": codes}, ok=True,
                         tokens_in=self.tokens_in if self.tokens_in is not None else len(prompt) // 4,
                         tokens_out=20, seconds=1.0, key_index=0)


TEXTS = [("Il dirigente ha ricevuto l'incarico presso la direzione generale. " * 40 + "\n") * 5] * 3


def test_haystack_has_three_codes_at_start_middle_end_and_target_length():
    hay, codes = needle.build_haystack(TEXTS, target_chars=60000, seed=1)
    assert len(codes) == 3 and len(set(codes)) == 3
    assert 59000 <= len(hay) <= 62000
    pos = [hay.index(c) for c in codes]
    assert pos[0] < 0.05 * len(hay) and 0.4 * len(hay) < pos[1] < 0.6 * len(hay) and pos[2] > 0.95 * len(hay)
    assert all(needle.CODE_RE.fullmatch(c) for c in codes)


def test_codes_are_deterministic_per_seed_and_differ_between_seeds():
    _, a = needle.build_haystack(TEXTS, 20000, seed=1)
    _, b = needle.build_haystack(TEXTS, 20000, seed=1)
    _, c = needle.build_haystack(TEXTS, 20000, seed=2)
    assert a == b and a != c


def test_full_reading_model_passes():
    r = needle.run_needle(FakeClient(), "m", TEXTS, target_tokens=15000, chars_per_token=4.0, num_ctx=32768, seed=3)
    assert r["found"] == [True, True, True] and r["all_found"] and r["truncated"] is False


@pytest.mark.parametrize("keep, expected", [("tail", [False, True, True]), ("head", [True, True, False])])
def test_silent_truncation_is_detected_by_missing_codes(keep, expected):
    r = needle.run_needle(FakeClient(keep=keep), "m", TEXTS, target_tokens=15000, chars_per_token=4.0,
                          num_ctx=32768, seed=3)
    assert not r["all_found"]
    assert r["found"][0] == expected[0] and r["found"][2] == expected[2]


def test_prompt_tokens_far_below_the_expected_length_flag_truncation():
    # the server read only 20 % of the prompt although the answer looks fine
    r = needle.run_needle(FakeClient(tokens_in=3000), "m", TEXTS, target_tokens=15000, chars_per_token=4.0,
                          num_ctx=32768, seed=3)
    assert r["truncated"] is True and r["prompt_tokens"] == 3000


def test_num_ctx_is_sent():
    fc = FakeClient()
    needle.run_needle(fc, "m", TEXTS, target_tokens=15000, chars_per_token=4.0, num_ctx=32768, seed=3)
    assert fc.calls[0]["options"]["num_ctx"] == 32768


def test_failed_call_is_reported_not_raised():
    class Bad:
        def chat(self, *a, **k):
            return LLMResult(ok=False, error="http_500")
    r = needle.run_needle(Bad(), "m", TEXTS, target_tokens=15000, chars_per_token=4.0, num_ctx=32768, seed=3)
    assert r["ok"] is False and r["error"] == "http_500" and r["all_found"] is False


def test_report_contains_no_haystack_text():
    r = needle.run_needle(FakeClient(), "m", TEXTS, target_tokens=15000, chars_per_token=4.0, num_ctx=32768, seed=3)
    assert "dirigente" not in json.dumps(r)


# ---- judge format check ---------------------------------------------------------------------------
def verdicts(n, over=None):
    over = over or {}
    out = []
    for i in range(1, n + 1):
        v = {"id": i, "plausible": "yes", "coherent": "yes", "realistic": "no", "explicit": "no"}
        v.update(over.get(i, {}))
        out.append(v)
    return {"verdicts": out}


class JudgeClient:
    def __init__(self, payload=None, text=None, ok=True, error=""):
        self.payload, self.text, self.ok, self.error = payload, text, ok, error
        self.last = None

    def chat(self, model, messages, options=None, schema=None):
        self.last = {"messages": messages, "options": options, "schema": schema}
        txt = self.text if self.text is not None else json.dumps(self.payload)
        js = self.payload if self.text is None else None
        return LLMResult(text=txt, json=js, ok=self.ok, error=self.error, tokens_in=5000, tokens_out=900, seconds=3.0)


def test_judge_format_valid_batch_of_40():
    c = JudgeClient(payload=verdicts(40))
    r = needle.run_judge_format(c, "m", "testo " * 500, n_attrs=40, num_ctx=32768)
    assert r["valid"] is True and r["n_returned"] == 40 and r["problems"] == []
    assert c.last["options"]["num_ctx"] == 32768
    assert c.last["messages"][-1]["content"].count("\n") > 40  # the 40 attributes are in the prompt


@pytest.mark.parametrize("payload, problem", [
    (verdicts(39), "wrong_count"),
    (verdicts(40, {5: {"plausible": "maybe"}}), "bad_value"),
    (verdicts(40, {3: {"id": 4}}), "bad_ids"),
    ({"verdicts": "no"}, "not_a_list"),
    ({"other": []}, "missing_key"),
])
def test_judge_format_problems_are_named(payload, problem):
    r = needle.run_judge_format(JudgeClient(payload=payload), "m", "testo " * 200, n_attrs=40, num_ctx=32768)
    assert r["valid"] is False and problem in r["problems"]


def test_judge_format_fenced_json_is_reported_as_a_distinct_failure():
    fenced = "```json\n" + json.dumps(verdicts(40)) + "\n```"
    r = needle.run_judge_format(JudgeClient(text=fenced, ok=False, error="invalid_json"), "m", "testo " * 200,
                                n_attrs=40, num_ctx=32768, raw_probe=lambda prompt: fenced)
    assert r["valid"] is False and r["error"] == "invalid_json"
    assert r["fenced_but_valid_inside"] is True


def test_judge_format_call_error_is_reported():
    r = needle.run_judge_format(JudgeClient(ok=False, error="http_500", payload={}), "m", "t " * 100,
                                n_attrs=40, num_ctx=32768)
    assert r["valid"] is False and r["error"] == "http_500"
