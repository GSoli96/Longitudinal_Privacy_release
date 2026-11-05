import math

import pytest

from rerun import detect_llm
from rerun.llm import LLMResult


class FakeClient:
    def __init__(self, answer=lambda prompt: "0"):
        self.answer = answer
        self.calls = []

    def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
        prompt = messages[0]["content"]
        self.calls.append((model, prompt, options, think))
        a = self.answer(prompt)
        if a == "FAIL":
            return LLMResult(ok=False, error="http_500")
        return LLMResult(ok=True, text=a)


def rec(i, y, body="corpo"):
    return {"id": f"r{i:02d}", "y": y, "subject": f"s{i}", "body": f"{body} {i}", "person": f"p{i % 3}",
            "case": "E", "generator": "g"}


@pytest.mark.parametrize("text,expected", [("1", 1), (" 0 ", 0), ('"1"', 1), ("0.", 0), ("1\n", 1),
                                          ("", None), (None, None), ("phishing", None), ("10", None),
                                          ("1 perché", None), ("2", None), ("0 1", None)])
def test_parse_label_accepts_only_a_bare_digit(text, expected):
    assert detect_llm.parse_label(text) == expected


def test_message_text_uses_subject_and_body_or_text():
    assert detect_llm.message_text({"subject": "A", "body": "B"}) == "Oggetto: A\n\nB"
    assert detect_llm.message_text({"text": "solo testo"}) == "solo testo"


def test_pick_examples_is_deterministic_and_balanced():
    train = [rec(i, i % 2) for i in range(20)][::-1]
    ex = detect_llm.pick_examples(train, 3)
    assert [e["y"] for e in ex] == [1, 1, 1, 0, 0, 0]
    assert [e["id"] for e in ex[:3]] == ["r01", "r03", "r05"]
    assert detect_llm.pick_examples(list(reversed(train)), 3) == ex


def test_zero_shot_prompt_is_the_paper_prompt():
    p = detect_llm.build_prompt(rec(1, 1), "zero")
    assert p.startswith("Classifica la seguente email come phishing (1) o legittima (0).")
    assert "Oggetto: s1" in p and p.endswith("Classificazione:")


def test_few_shot_needs_examples_and_truncates_them():
    with pytest.raises(ValueError):
        detect_llm.build_prompt(rec(1, 1), "few", [])
    long = {"id": "x", "y": 1, "subject": "S", "body": "z" * 1000}
    p = detect_llm.build_prompt(rec(1, 0), "few", [long])
    shot = p.split("Esempi:")[1].split("Email:")[1]
    assert shot.count("z") <= 300 and "Classificazione: 1" in p


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        detect_llm.build_prompt(rec(1, 1), "many")


def test_evaluate_scores_valid_answers_and_counts_errors_apart():
    test = [rec(i, i % 2) for i in range(6)]
    def answer(prompt):
        if "corpo 2" in prompt:
            return "FAIL"
        if "corpo 3" in prompt:
            return "forse"
        return "1" if "corpo 5" in prompt or "corpo 1" in prompt else "0"
    out = detect_llm.evaluate(FakeClient(answer), "m", test, "zero")
    s = out["summary"]
    assert (s["n_test"], s["n_scored"], s["n_excluded"]) == (6, 4, 2)
    assert s["errors"] == {"call_failed:http_500": 1, "unparseable": 1}
    assert math.isclose(s["coverage"], 4 / 6)
    assert {r["id"] for r in out["records"]} == {"r00", "r01", "r04", "r05"}
    assert all(r["score"] == float(r["pred"]) for r in out["records"])


def test_failed_calls_are_never_turned_into_a_label():
    test = [rec(i, 1) for i in range(4)]
    out = detect_llm.evaluate(FakeClient(lambda p: "FAIL"), "m", test, "zero")
    assert out["records"] == [] and out["summary"]["n_excluded"] == 4 and out["summary"]["coverage"] == 0


def test_call_options_are_deterministic_and_without_thinking():
    c = FakeClient()
    detect_llm.evaluate(c, "m", [rec(1, 0)], "zero", seed=7)
    _, _, options, think = c.calls[0]
    assert options == {"temperature": 0.0, "seed": 7, "num_predict": detect_llm.NUM_PREDICT} and think is False


def test_few_shot_uses_training_examples_only():
    train = [rec(i, i % 2, body="addestramento") for i in range(10)]
    test = [rec(20 + i, i % 2, body="prova") for i in range(2)]
    c = FakeClient()
    out = detect_llm.evaluate(c, "m", test, "few", train=train)
    assert out["summary"]["n_examples"] == 6
    assert all("addestramento" in call[1] for call in c.calls)
    assert all(call[1].count("Classificazione: ") == 6 for call in c.calls)


def test_empty_test_has_nan_coverage():
    out = detect_llm.evaluate(FakeClient(), "m", [], "zero")
    assert out["records"] == [] and math.isnan(out["summary"]["coverage"])
