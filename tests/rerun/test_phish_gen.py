import json

from rerun import phish_gen
from rerun.llm import LLMResult

PERSON = {"person": "p1", "case": "E", "display_name": "Nome Cognome", "explicit": ["ruolo dichiarato"],
          "inferred": ["attributo dedotto"]}
KW = dict(company_name="Ente X", company_domain="entex")


class FakeClient:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or (lambda prompt: None)

    def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
        prompt = messages[0]["content"]
        self.calls.append((model, prompt, options, schema))
        mode = self.fail(prompt)
        if mode == "error":
            return LLMResult(ok=False, error="http_500")
        if mode == "text":
            return LLMResult(ok=True, text="no json here")
        if mode == "empty":
            return LLMResult(ok=True, json={"subject": " ", "body": "x"})
        if mode == "embedded":
            return LLMResult(ok=True, text='ecco: {"subject": "S", "body": "B"}')
        return LLMResult(ok=True, json={"subject": " Oggetto ", "body": " Corpo "}, tokens_out=5)


def test_profile_per_variant():
    e, i = ["e1"], ["i1", "i2"]
    assert phish_gen.profile_for("none", e, i) == []
    assert phish_gen.profile_for("explicit", e, i) == ["e1"]
    assert phish_gen.profile_for("explicit_inferred", e, i) == ["e1", "i1", "i2"]


def test_base_context_is_in_every_variant_and_in_the_prompt():
    b = ["ruolo: funzionario", "ente: comune"]
    assert phish_gen.profile_for("none", ["e1"], ["i1"], b) == b
    assert phish_gen.profile_for("explicit", ["e1"], ["i1"], b) == b + ["e1"]
    assert phish_gen.profile_for("explicit_inferred", ["e1"], ["i1"], b) == b + ["e1", "i1"]
    c = FakeClient()
    phish_gen.run(c, "m", [{**PERSON, "base": b}], **KW)
    none_prompts = [p for _, p, _, _ in c.calls if "attributo dedotto" not in p and "ruolo dichiarato" not in p]
    assert none_prompts and all("- ruolo: funzionario" in p for p in none_prompts)


def test_none_variant_has_no_personal_attribute_in_prompt():
    p = phish_gen.build_prompt("positive", "BEC", display_name="N", profile=[], **KW)
    assert "nessuna informazione personale" in p and "- " + "ruolo" not in p
    q = phish_gen.build_prompt("positive", "SpearPhishing", display_name="N", profile=["ruolo x"], **KW)
    assert "- ruolo x" in q and "entex" in q and "Ente X" in q


def test_negative_prompt_is_legitimate_and_uses_profile():
    p = phish_gen.build_prompt("negative", "BEC", display_name="N", profile=["ruolo x"], **KW)
    assert "legittima" in p and "- ruolo x" in p and "credenziali" in p


def test_no_two_jobs_of_a_person_share_a_prompt():
    c = FakeClient()
    phish_gen.run(c, "m", [PERSON], **KW)
    prompts = [p for _, p, _, _ in c.calls]
    assert len(prompts) == 12 and len(set(prompts)) == 12


def test_generate_one_ok_and_labels():
    c = FakeClient()
    r = phish_gen.generate_one(c, "m", **{k: PERSON[k] for k in ("person", "case", "display_name", "explicit",
                                                                 "inferred")},
                               variant="explicit", attack="BEC", kind="positive", **KW)
    assert r["ok"] and r["subject"] == "Oggetto" and r["body"] == "Corpo" and r["label"] == 1
    n = phish_gen.generate_one(c, "m", **{k: PERSON[k] for k in ("person", "case", "display_name", "explicit",
                                                                 "inferred")},
                               variant="explicit", attack="BEC", kind="negative", **KW)
    assert n["label"] == 0 and n["id"] != r["id"]


def test_failures_are_kept_with_reason_never_turned_into_text():
    base = {k: PERSON[k] for k in ("person", "case", "display_name", "explicit", "inferred")}
    for mode, reason in (("error", "call_failed:http_500"), ("text", "not_json"), ("empty", "missing_subject_or_body")):
        r = phish_gen.generate_one(FakeClient(lambda p, m=mode: m), "m", **base, variant="none", attack="BEC",
                                   kind="positive", **KW)
        assert not r["ok"] and r["error"] == reason and "body" not in r
    r = phish_gen.generate_one(FakeClient(lambda p: "embedded"), "m", **base, variant="none", attack="BEC",
                               kind="positive", **KW)
    assert r["ok"] and r["subject"] == "S"


def test_options_pass_temperature_and_seed_plus_index():
    c = FakeClient()
    base = {k: PERSON[k] for k in ("person", "case", "display_name", "explicit", "inferred")}
    phish_gen.generate_one(c, "m", **base, variant="none", attack="BEC", kind="negative", index=2,
                           temperature=1.0, seed=10, **KW)
    _, _, options, schema = c.calls[0]
    assert options == {"temperature": 1.0, "seed": 12} and schema == phish_gen.SCHEMA


def test_jobs_and_run_cover_every_person_and_combination():
    persons = [PERSON, {**PERSON, "person": "p2"}]
    assert len(phish_gen.jobs_for_person(PERSON)) == 12
    assert len(phish_gen.jobs_for_person(PERSON, negatives_per_positive=3)) == 3 * 2 * 4
    out = phish_gen.run(FakeClient(), "m", persons, workers=3, **KW)
    s = out["summary"]
    assert s["n_persons"] == 2 and s["n_records"] == 24 and s["n_ok"] == 24 and s["n_failed"] == 0
    assert {r["person"] for r in out["records"]} == {"p1", "p2"}
    assert len({r["id"] for r in out["records"]}) == 24
    combos = {(r["variant"], r["attack"], r["kind"]) for r in out["records"] if r["person"] == "p1"}
    assert len(combos) == 12


def test_run_counts_failures_and_summary_has_no_text():
    c = FakeClient(lambda p: "error" if "legittima" in p else None)
    out = phish_gen.run(c, "m", [PERSON], **KW)
    s = out["summary"]
    assert s["n_failed"] == 6 and s["errors"] == {"call_failed:http_500": 6}
    assert "Oggetto" not in json.dumps(s) and "Nome Cognome" not in json.dumps(s)
    assert {r["person"] for r in out["records"]} == {"p1"}
