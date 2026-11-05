"""Tests of rerun.infer (inference with a fixed JSON schema), with a mocked client."""
import json

import pytest

from rerun import infer, steps
from rerun.llm import LLMResult


@pytest.fixture(autouse=True)
def _no_empty_retry_by_default(monkeypatch):
    from rerun import config
    monkeypatch.setattr(config, "ATTACKER_EMPTY_RETRIES", 0)


def answer(items):
    return json.dumps({"attributes": items})


def item(a, c="role", e="evidence text"):
    return {"attribute": a, "category": c, "evidence": e}


class Mock:
    """Scripted client: pops one result per call; results are dicts or LLMResult."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, model, messages, options=None, schema=None, think=None):
        self.calls.append({"model": model, "messages": messages, "options": options, "schema": schema,
                           "think": think})
        r = self.script.pop(0) if self.script else {"attributes": []}
        if isinstance(r, LLMResult):
            return r
        return LLMResult(text=json.dumps(r), json=r, ok=True, tokens_in=100, tokens_out=50, seconds=1.0)


def step_of(text, k=100, identical=False):
    return steps.Step(k=k, n_docs=1, text=text, identical_to_prev=identical, dated_docs=1, undated_docs=0,
                      chars=len(text))


TEXT = "[Document 1]\nIl dirigente coordina l'ufficio gare dal 2019. Ha un incarico di durata triennale."


def run(script, text=TEXT, **kw):
    m = Mock(script)
    kw.setdefault("usable_tokens", 100000)
    kw.setdefault("variant", "evidence")                    # the default variant is the one without evidence
    return infer.infer_step(m, step_of(text), "gpt-oss:20b", **kw), m


# ---- parsing ---------------------------------------------------------------------------------------
def test_valid_json_is_parsed_into_attributes():
    res, m = run([{"attributes": [item("Ha esperienza negli appalti", "competences", "coordina l'ufficio gare dal 2019"),
                                  item("Ha un contratto a termine", "role", "incarico di durata triennale")]}])
    assert res.ok and res.error == "" and not res.chunked and res.windows == 1
    assert [a.text for a in res.attributes] == ["Ha esperienza negli appalti", "Ha un contratto a termine"]
    a = res.attributes[0]
    assert a.category_hint == "competences" and a.evidence == "coordina l'ufficio gare dal 2019" and a.window == 0
    assert a.evidence_in_text is True and a.evidence_match == "exact" and a.evidence_ratio == 1.0


def test_empty_list_is_allowed():
    res, _ = run([{"attributes": []}])
    assert res.ok and res.attributes == []


def test_evidence_not_in_text_is_flagged_not_dropped():
    res, _ = run([{"attributes": [item("x", "c", "frase che non compare nel testo")]}])
    assert res.ok and len(res.attributes) == 1 and res.attributes[0].evidence_in_text is False
    assert res.attributes[0].evidence_match == "none" and res.evidence_not_found == 1


def test_evidence_match_ignores_case_spaces_and_quote_style():
    res, _ = run([{"attributes": [item("x", "c", "COORDINA  l\u2019ufficio gare")]}])
    a = res.attributes[0]
    assert a.evidence_in_text is True and a.evidence_match == "exact"


LONGTXT = ("[Document 1]\nIl dirigente generale ha ricevuto l'incarico di direzione dell'ufficio per la gestione "
           "delle procedure di gara e dei contratti pubblici con decreto del Ministro del 3 marzo 2019, "
           "per la durata di tre anni, con possibilita di rinnovo.\n" + "Altre informazioni di servizio. " * 30)


@pytest.mark.parametrize("evidence, match", [
    # exact, after normalisation
    ("ha ricevuto l'incarico di direzione dell'ufficio per la gestione delle procedure di gara", "exact"),
    # fuzzy: one typo and one dropped letter
    ("ha ricevuto l'incarico di direzzione dell'ufficio per la gestione delle procedure di gar", "fuzzy"),
    # fuzzy: a changed digit group
    ("con decreto del Ministro del 3 marzo 2018, per la durata di tre anni", "fuzzy"),
])
def test_evidence_exact_and_fuzzy_are_anchored(evidence, match):
    res, _ = run([{"attributes": [item("x", "c", evidence)]}], text=LONGTXT)
    a = res.attributes[0]
    assert a.evidence_in_text is True and a.evidence_match == match
    assert a.evidence_ratio >= 0.9
    assert (res.evidence_exact, res.evidence_fuzzy) == ((1, 0) if match == "exact" else (0, 1))


@pytest.mark.parametrize("evidence", [
    "il funzionario ha ottenuto una promozione a capo del servizio finanziario nel 2021",   # paraphrase / invented
    "lo stipendio annuo supera i centomila euro come indicato nella dichiarazione dei redditi",  # invented
    "Altre notizie sulla carriera passata dell'interessato in un diverso ente pubblico locale",  # plausible, absent
])
def test_paraphrased_or_invented_evidence_is_not_anchored_but_kept(evidence):
    res, _ = run([{"attributes": [item("x", "c", evidence)]}], text=LONGTXT)
    a = res.attributes[0]
    assert a.evidence_in_text is False and a.evidence_match == "none" and a.evidence_ratio < 0.9
    assert res.evidence_not_found == 1 and len(res.attributes) == 1


def test_evidence_longer_than_the_text_is_compared_as_a_whole():
    res, _ = run([{"attributes": [item("x", "c", "Il dirigente coordina l'ufficio gare dal 2019. " * 20)]}],
                 text="[Document 1]\nBreve.")
    assert res.attributes[0].evidence_match == "none"


def test_anchor_threshold_comes_from_config(monkeypatch):
    from rerun import config
    ev = "ha ricevuto l'incarico di direzzione dell'ufficio per la gestione delle procedure di gar"
    monkeypatch.setattr(config, "EVIDENCE_FUZZY_RATIO", 0.999)
    res, _ = run([{"attributes": [item("x", "c", ev)]}], text=LONGTXT)
    assert res.attributes[0].evidence_match == "none"


# ---- structure errors: retried with a correction, then failure ------------------------------------------
@pytest.mark.parametrize("bad", [
    {"items": []},                                   # wrong key
    {"attributes": "no"},                             # not a list
    {"attributes": [{"category": "c", "evidence": "e"}]},   # item without attribute
    {"attributes": [{"attribute": "", "category": "c", "evidence": "e"}]},   # empty attribute
    {"attributes": [{"attribute": 5, "category": "c", "evidence": "e"}]},    # not a string
    {"attributes": ["just a string"]},                # item not an object
    {"attributes": [{"attribute": "ok", "category": "c"}]},   # evidence missing
])
def test_bad_structure_is_retried_with_correction_then_succeeds(bad):
    res, m = run([bad, {"attributes": [item("valido")]}])
    assert res.ok and [a.text for a in res.attributes] == ["valido"]
    assert len(m.calls) == 2
    second = m.calls[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-1]["role"] == "user"
    assert "attributes" in second[-1]["content"]
    assert m.calls[1]["options"]["seed"] != m.calls[0]["options"].get("seed")
    assert res.structure_retries == 1


def test_bad_structure_never_parsed_into_attributes_after_retries():
    bad = {"attributes": [{"category": "c"}]}
    res, m = run([bad, bad, bad, bad], max_structure_retries=2)
    assert not res.ok and res.error == "invalid_structure" and res.attributes == []
    assert len(m.calls) == 3


def test_client_failure_is_a_failure_without_attributes():
    res, m = run([LLMResult(ok=False, error="invalid_json")])
    assert not res.ok and res.error == "invalid_json" and res.attributes == [] and len(m.calls) == 1


def test_prose_instead_of_json_is_a_failure_never_parsed_line_by_line():
    r = LLMResult(text="- Ha esperienza\n- Ha un contratto", json=None, ok=False, error="invalid_json")
    res, _ = run([r])
    assert not res.ok and res.attributes == []


# ---- windows and merging ---------------------------------------------------------------------------------
LONG = "\n".join(f"Riga {i}: il dirigente ha svolto attivita numero {i} presso l'ufficio." for i in range(400))


def test_long_text_is_split_in_windows_and_attributes_are_merged():
    res0, m0 = run([{"attributes": []}] * 20, text=LONG, usable_tokens=1500, chars_per_token=3.5)
    n = len(m0.calls)
    assert n > 1 and res0.chunked and res0.windows == n
    script = [{"attributes": [item("Ha un ruolo direttivo", "role", "dirigente"),
                              item(f"Attivita unica {i}", "x", "attivita")]} for i in range(n)]
    res, m = run(script, text=LONG, usable_tokens=1500, chars_per_token=3.5)
    texts = [a.text for a in res.attributes]
    assert texts.count("Ha un ruolo direttivo") == 1 and len(texts) == 1 + n
    first = next(a for a in res.attributes if a.text == "Ha un ruolo direttivo")
    assert first.window == 0 and first.windows == tuple(range(n))
    assert res.chunked and res.ok


def test_duplicates_are_merged_by_normalised_text_inside_one_window():
    res, _ = run([{"attributes": [item("Ha un ruolo DIRETTIVO."), item("  ha un ruolo direttivo "), item("Altro")]}])
    assert [a.text for a in res.attributes] == ["Ha un ruolo DIRETTIVO.", "Altro"]


def test_failure_of_one_window_marks_the_step_failed_and_keeps_partial_attributes():
    n = len(run([{"attributes": []}] * 20, text=LONG, usable_tokens=1500)[1].calls)
    assert n >= 3
    script = [{"attributes": [item("dal primo")]}, LLMResult(ok=False, error="invalid_json")] + \
             [{"attributes": [item(f"altro {i}")]} for i in range(n - 2)]
    res, _ = run(script, text=LONG, usable_tokens=1500)
    assert not res.ok and res.partial and res.failed_windows == [1]
    assert "dal primo" in [a.text for a in res.attributes]


def test_single_window_when_the_text_fits():
    res, m = run([{"attributes": []}], usable_tokens=100000)
    assert res.windows == 1 and not res.chunked and len(m.calls) == 1


# ---- the request ------------------------------------------------------------------------------------------
def test_prompt_keeps_the_original_instruction_and_embeds_the_text():
    p = infer.build_prompt("TESTO DI PROVA")
    assert "You are an analyst." in p
    assert "infer ONLY information that is NOT explicitly stated but can be deduced" in p
    assert "TESTO DI PROVA" in p and "----------------" in p
    assert "no need for valid JSON" not in p  # replaced by the schema request


def test_prompt_does_not_ask_for_special_category_data():
    p = infer.build_prompt("x").lower()
    for word in ("gdpr", "art. 9", "health", "political", "religio", "union", "sexual", "ethnic", "special categor"):
        assert word not in p


def test_request_has_schema_num_ctx_and_output_budget():
    res, m = run([{"attributes": []}], num_ctx=131072)
    c = m.calls[0]
    assert c["schema"] == infer.SCHEMA and c["options"]["num_ctx"] == 131072
    assert c["options"]["num_predict"] == infer.NUM_PREDICT
    assert c["messages"][-1]["role"] == "user"


def test_schema_requires_attribute_and_evidence():
    item_schema = infer.SCHEMA["properties"]["attributes"]["items"]
    assert set(item_schema["required"]) == {"attribute", "evidence"}


# ---- steps of a person: identical steps are not run again ---------------------------------------------------
def test_identical_steps_are_carried_forward_without_a_call():
    from rerun.corpus import Doc, Person
    docs = [Doc(id="X-d1", date=None, kind="pdf", text="Testo del primo documento. " * 10)]
    p = Person("X-1", "X", None, None, docs)
    m = Mock([{"attributes": [item("attributo uno")]}])
    out = infer.infer_person(m, p, "gpt-oss:20b", usable_tokens=100000)
    assert [r.step.k for r in out] == [25, 50, 75, 100]
    assert len(m.calls) == 1                           # only the first step runs
    assert [r.carried for r in out] == [False, True, True, True]
    assert all([a.text for a in r.attributes] == ["attributo uno"] for r in out)


def test_person_without_text_has_no_inference():
    from rerun.corpus import Person
    p = Person("X-2", "X", None, None, [], status="no_text")
    m = Mock([])
    assert infer.infer_person(m, p, "gpt-oss:20b", usable_tokens=100000) == []
    assert m.calls == []


# ---- fixed num_ctx per model ----------------------------------------------------------------------------
def test_num_ctx_is_fixed_per_model_not_chosen_per_step():
    from rerun import config
    from rerun.corpus import Doc, Person
    docs = [Doc(id=f"X-d{i}", date=None, kind="pdf", text=("Testo del documento numero %d. " % i) * (50 * i))
            for i in range(1, 5)]
    p = Person("X-3", "X", None, None, docs)
    m = Mock([{"attributes": []}] * 10)
    infer.infer_person(m, p, "gpt-oss:20b", usable_tokens=100000)
    used = {c["options"]["num_ctx"] for c in m.calls}
    assert used == {config.NUM_CTX["gpt-oss:20b"]} == {131072}


def test_fixed_num_ctx_for_models_and_fallback():
    from rerun import config
    assert config.num_ctx_for("llama3.2:3b") == 131072
    for judge in config.JUDGES:
        assert config.num_ctx_for(judge) == config.NUM_CTX[judge]
    assert config.num_ctx_for("unknown-model") == 131072


def test_attacker_uses_reasoning_effort_and_sampling_from_the_config():
    from rerun import config
    res, m = run([{"attributes": []}])
    c = m.calls[0]
    assert c["think"] == config.ATTACKER_THINK == "medium"
    assert c["options"]["temperature"] == 1.0 and c["options"]["top_p"] == 1.0
    assert c["options"]["seed"] == config.SEED


def test_seed_offset_of_a_repetition_shifts_only_the_attackers_seed(monkeypatch):
    from rerun import config
    monkeypatch.setattr(config, "ATTACKER_SEED_OFFSET", 1000)
    res, m = run([{"attributes": []}])
    assert m.calls[0]["options"]["seed"] == config.SEED + 1000


def test_retry_keeps_the_sampling_and_changes_only_the_seed():
    res, m = run([{"attributes": "no"}, {"attributes": []}])
    a, b = m.calls
    assert b["options"]["seed"] == a["options"]["seed"] + 1
    assert b["options"]["temperature"] == a["options"]["temperature"] and b["think"] == a["think"]


# ---- empty answers -----------------------------------------------------------------------------------------------
def test_an_empty_list_is_asked_again_once_with_a_new_seed_and_the_same_request(monkeypatch):
    from rerun import config
    monkeypatch.setattr(config, "ATTACKER_EMPTY_RETRIES", 1)
    res, m = run([{"attributes": []}, {"attributes": [item("trovato al secondo tentativo")]}])
    assert res.ok and [a.text for a in res.attributes] == ["trovato al secondo tentativo"]
    assert res.empty_retries == 1 and len(m.calls) == 2
    assert m.calls[1]["messages"] == m.calls[0]["messages"]                  # no correction message
    assert m.calls[1]["options"]["seed"] == m.calls[0]["options"]["seed"] + 1


def test_empty_twice_is_accepted_as_empty_and_counted(monkeypatch):
    from rerun import config
    monkeypatch.setattr(config, "ATTACKER_EMPTY_RETRIES", 1)
    res, m = run([{"attributes": []}, {"attributes": []}, {"attributes": [item("non deve essere chiesto")]}])
    assert res.ok and res.attributes == [] and res.empty_retries == 1 and len(m.calls) == 2


def test_empty_retries_can_be_disabled_or_raised(monkeypatch):
    from rerun import config
    monkeypatch.setattr(config, "ATTACKER_EMPTY_RETRIES", 0)
    res, m = run([{"attributes": []}])
    assert res.empty_retries == 0 and len(m.calls) == 1
    monkeypatch.setattr(config, "ATTACKER_EMPTY_RETRIES", 2)
    res, m = run([{"attributes": []}, {"attributes": []}, {"attributes": [item("terzo")]}])
    assert res.empty_retries == 2 and [a.text for a in res.attributes] == ["terzo"]


def test_empty_retry_and_structure_retry_use_distinct_seeds(monkeypatch):
    from rerun import config
    monkeypatch.setattr(config, "ATTACKER_EMPTY_RETRIES", 1)
    res, m = run([{"attributes": []}, {"attributes": "no"}, {"attributes": [item("ok")]}])
    seeds = [c["options"]["seed"] for c in m.calls]
    assert seeds == [config.SEED, config.SEED + 1, config.SEED + 2]
    assert res.empty_retries == 1 and res.structure_retries == 1


# ---- variant without the evidence field (pilot comparison, same schema minus "evidence") -----------------------------
def run_v(script, text=TEXT, variant="no_evidence", **kw):
    m = Mock(script)
    kw.setdefault("usable_tokens", 100000)
    return infer.infer_step(m, step_of(text), "gpt-oss:20b", variant=variant, **kw), m


def test_no_evidence_variant_has_a_schema_without_evidence():
    sch = infer.schema_for("no_evidence")
    props = sch["properties"]["attributes"]["items"]
    assert set(props["properties"]) == {"attribute", "category"} and props["required"] == ["attribute"]
    assert infer.schema_for("evidence") == infer.SCHEMA


def test_no_evidence_prompt_is_the_same_instruction_without_asking_for_evidence():
    a, b = infer.build_prompt("TESTO"), infer.build_prompt("TESTO", variant="no_evidence")
    assert "evidence" in a and "evidence" not in b.lower()
    assert "You are an analyst." in b and "infer ONLY information that is NOT explicitly stated" in b and "TESTO" in b
    assert infer.build_prompt("x", variant="evidence") == infer.build_prompt("x")


def test_no_evidence_items_are_valid_without_evidence_and_not_anchored():
    res, m = run_v([{"attributes": [{"attribute": "Ha un ruolo direttivo", "category": "role"},
                                    {"attribute": "Altro fatto"}]}])
    assert res.ok and [a.text for a in res.attributes] == ["Ha un ruolo direttivo", "Altro fatto"]
    assert all(a.evidence == "" and a.evidence_match == "n/a" and a.evidence_in_text is False for a in res.attributes)
    assert (res.evidence_exact, res.evidence_fuzzy, res.evidence_not_found) == (0, 0, 0)
    assert m.calls[0]["schema"] == infer.schema_for("no_evidence")


def test_no_evidence_variant_still_rejects_missing_attribute_and_retries():
    res, m = run_v([{"attributes": [{"category": "x"}]}, {"attributes": [{"attribute": "ok"}]}])
    assert res.ok and res.structure_retries == 1 and len(m.calls) == 2
    assert "evidence" not in m.calls[1]["messages"][-1]["content"].lower()


def test_infer_person_passes_the_variant_through():
    from rerun.corpus import Doc, Person
    p = Person("X-9", "X", None, None, [Doc(id="d", date=None, kind="pdf", text="Testo. " * 30)])
    m = Mock([{"attributes": [{"attribute": "a"}]}])
    out = infer.infer_person(m, p, "gpt-oss:20b", usable_tokens=100000, variant="no_evidence")
    assert out[0].attributes[0].evidence_match == "n/a" and m.calls[0]["schema"] == infer.schema_for("no_evidence")


def test_main_run_default_is_the_variant_without_evidence():
    assert infer.DEFAULT_VARIANT == "no_evidence"
    m = Mock([{"attributes": [{"attribute": "Ha un ruolo dirigenziale", "category": "role"}]}])
    res = infer.infer_step(m, step_of(TEXT), "gpt-oss:20b", usable_tokens=100000)
    assert res.ok and res.attributes[0].evidence == "" and "evidence" not in str(m.calls[0]["schema"])
