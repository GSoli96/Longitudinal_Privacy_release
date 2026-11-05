"""Tests of rerun.judge: batched and per-attribute judging, vote validation, majority."""
import itertools
import json
from collections import Counter

import pytest

from rerun import config, judge
from rerun.llm import LLMResult

C = judge.CRITERIA
YES = {c: "yes" for c in C}


def v(n, **over):
    d = {"id": n, **{c: "yes" for c in C}}
    d.update(over)
    return d


class Mock:
    """Scripted client: each call pops one scripted answer (dict -> valid JSON result, LLMResult as is)."""

    def __init__(self, script=None, default=None):
        self.script = list(script or [])
        self.default = default
        self.calls = []

    def chat(self, model, messages, options=None, schema=None):
        self.calls.append({"model": model, "messages": messages, "options": options, "schema": schema})
        if self.script:
            r = self.script.pop(0)
        elif self.default is not None:
            r = self.default(self.calls[-1])
        else:
            r = {"verdicts": []}
        if isinstance(r, LLMResult):
            return r
        return LLMResult(text=json.dumps(r), json=r, ok=True, tokens_in=100, tokens_out=50, seconds=1.0)


def echo_all(call):
    """Answers 'yes' to every attribute number that appears in the prompt."""
    import re
    prompt = call["messages"][-1]["content"]
    ids = [int(x) for x in re.findall(r"^(\d+)\. ", prompt.split("ATTRIBUTES TO JUDGE")[1], re.M)]
    return {"verdicts": [v(i) for i in ids]}


def attrs(n, window=0):
    return [judge.JudgeAttr(id=f"a{i:03d}", text=f"attributo numero {i}", evidence=f"prova {i}", window=window)
            for i in range(1, n + 1)]


# ---- majority ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("votes", list(itertools.product(("yes", "no"), repeat=3)))
def test_majority_truth_table_over_all_eight_patterns(votes):
    expected = "yes" if votes.count("yes") >= 2 else "no"
    vs = [judge.Vote(**{c: x for c in C}) for x in votes]
    m = judge.majority(vs)
    assert all(m[c] == expected for c in C)


def test_majority_needs_three_valid_votes_per_criterion():
    vs = [judge.Vote(plausible="yes", coherent="yes", realistic="no", explicit="no"),
          judge.Vote(plausible="yes", coherent="invalid", realistic="no", explicit="no"),
          judge.Vote(plausible="no", coherent="yes", realistic="no", explicit="yes")]
    m = judge.majority(vs)
    assert m == {"plausible": "yes", "coherent": "undecided", "realistic": "no", "explicit": "no"}


def test_two_votes_or_one_is_undecided_never_a_majority_of_fewer():
    assert judge.majority([judge.Vote(*["yes"] * 4)] * 2)["plausible"] == "undecided"
    assert judge.majority([])["plausible"] == "undecided"
    assert judge.majority([judge.Vote(*["yes"] * 4)] * 4)["plausible"] == "undecided"  # exactly three judges


def test_invalid_is_never_counted_as_no():
    vs = [judge.Vote(*["invalid"] * 4), judge.Vote(*["no"] * 4), judge.Vote(*["no"] * 4)]
    assert judge.majority(vs)["plausible"] == "undecided"  # two "no" must not win over an invalid vote


# ---- batch call ----------------------------------------------------------------------------------------
def test_batch_returns_votes_by_attribute_id():
    m = Mock([{"verdicts": [v(1), v(2, explicit="no", realistic="no"), v(3, plausible="no")]}])
    out = judge.judge_batch(m, "gemma4:31b", ["TESTO DEL PASSO"], attrs(3))
    assert set(out) == {"a001", "a002", "a003"}
    assert out["a002"] == judge.Vote(plausible="yes", coherent="yes", realistic="no", explicit="no")
    assert out["a003"].plausible == "no"


def test_prompt_has_text_numbered_attributes_and_no_evidence_for_single_window():
    m = Mock([{"verdicts": [v(1), v(2)]}])
    judge.judge_batch(m, "gemma4:31b", ["TESTO DEL PASSO"], attrs(2))
    p = m.calls[0]["messages"][-1]["content"]
    assert "TESTO DEL PASSO" in p and "1. attributo numero 1" in p and "2. attributo numero 2" in p
    assert "prova 1" not in p  # judges are blind to the attacker's evidence
    assert "html" not in p.lower()
    for crit in C:
        assert crit in p
    assert '"explicit" is "yes" only if the information is directly present' in p


def test_request_uses_schema_fixed_num_ctx_and_output_budget():
    m = Mock([{"verdicts": [v(1)]}])
    judge.judge_batch(m, "nemotron-3-ultra", ["T"], attrs(1))
    c = m.calls[0]
    assert c["schema"] == judge.SCHEMA
    assert c["options"]["num_ctx"] == config.num_ctx_for("nemotron-3-ultra")
    assert c["options"]["num_predict"] == judge.NUM_PREDICT == 32768


def test_batches_of_at_most_40_attributes_keep_global_ids():
    m = Mock(default=echo_all)
    out = judge.judge_batch(m, "gemma4:31b", ["T"], attrs(85))
    assert len(m.calls) == 3 and len(out) == 85
    assert out["a001"] and out["a041"] and out["a085"]
    sizes = [len(call["messages"][-1]["content"].split("ATTRIBUTES TO JUDGE")[1].split("\n")) for call in m.calls]
    assert sizes[0] > sizes[2]
    assert all(vote == judge.Vote(*["yes"] * 4) for vote in out.values())


# ---- invalid / missing votes ---------------------------------------------------------------------------
def test_missing_attribute_is_retried_alone_then_filled():
    m = Mock([{"verdicts": [v(1), v(3)]}, {"verdicts": [v(2, coherent="no")]}])
    stats = Counter()
    out = judge.judge_batch(m, "gemma4:31b", ["T"], attrs(3), stats=stats)
    assert out["a002"].coherent == "no" and out["a001"].coherent == "yes"
    assert len(m.calls) == 2
    retry_prompt = m.calls[1]["messages"][-1]["content"]
    assert "2. attributo numero 2" in retry_prompt and "1. attributo numero 1" not in retry_prompt
    assert m.calls[1]["options"]["seed"] != m.calls[0]["options"]["seed"]
    assert stats["retries"] == 1 and stats["invalid_votes"] == 0


@pytest.mark.parametrize("bad", ["maybe", "", None, 1, "YES!"])
def test_invalid_value_is_retried_then_marked_invalid_never_no(bad):
    bad_item = v(1, plausible=bad)
    m = Mock([{"verdicts": [bad_item]}] * 5)
    stats = Counter()
    out = judge.judge_batch(m, "gemma4:31b", ["T"], attrs(1), max_retries=2, stats=stats)
    assert len(m.calls) == 3
    assert out["a001"].plausible == "invalid"
    assert out["a001"].coherent == "invalid"  # an item with any invalid value is not trusted at all
    assert "no" not in {out["a001"].plausible, out["a001"].coherent}
    assert stats["invalid_votes"] == 1


def test_case_and_spaces_of_yes_no_are_normalised():
    m = Mock([{"verdicts": [v(1, plausible=" YES ", explicit="No")]}])
    out = judge.judge_batch(m, "gemma4:31b", ["T"], attrs(1))
    assert out["a001"].plausible == "yes" and out["a001"].explicit == "no"


def test_call_failure_is_retried_then_all_attributes_invalid():
    m = Mock([LLMResult(ok=False, error="invalid_json")] * 5)
    stats = Counter()
    out = judge.judge_batch(m, "minimax-m3.1-flash", ["T"], attrs(4), max_retries=2, stats=stats)
    assert len(m.calls) == 3
    assert all(vote == judge.Vote(*["invalid"] * 4) for vote in out.values()) and len(out) == 4
    assert stats["failed_calls"] == 3 and stats["invalid_votes"] == 4


def test_failure_then_success_gives_valid_votes():
    m = Mock([LLMResult(ok=False, error="http_500"), {"verdicts": [v(1)]}])
    out = judge.judge_batch(m, "gemma4:31b", ["T"], attrs(1))
    assert out["a001"] == judge.Vote(*["yes"] * 4)


def test_ids_outside_the_batch_and_duplicates_are_ignored():
    m = Mock([{"verdicts": [v(1), v(1, plausible="no"), v(99)]}])
    out = judge.judge_batch(m, "gemma4:31b", ["T"], attrs(1))
    assert out["a001"].plausible == "yes" and set(out) == {"a001"}


# ---- one attribute at a time ------------------------------------------------------------------------------
def test_judge_one_uses_the_same_format_for_a_single_attribute():
    m = Mock([{"verdicts": [v(1, explicit="no")]}])
    vote = judge.judge_one(m, "gemma4:31b", ["T"], attrs(1)[0])
    assert vote.explicit == "no"
    assert "1. attributo numero 1" in m.calls[0]["messages"][-1]["content"]


# ---- windows: explicitness is an OR over windows -------------------------------------------------------------
def test_explicit_is_yes_if_any_window_states_it_other_criteria_from_the_originating_window():
    a = attrs(2, window=0) + [judge.JudgeAttr(id="a900", text="attributo solo nella seconda finestra",
                                              evidence="prova 900", window=1)]
    # window 0 answers for the batch (attrs 1, 2, 3), then window 1
    w0 = {"verdicts": [v(1, explicit="no", plausible="no"), v(2, explicit="no"), v(3, explicit="no", coherent="no")]}
    w1 = {"verdicts": [v(1, explicit="yes", plausible="yes"), v(2, explicit="no"), v(3, explicit="yes", coherent="yes")]}
    m = Mock([w0, w1])
    out = judge.judge_batch(m, "gemma4:31b", ["finestra uno", "finestra due"], a)
    assert out["a001"].explicit == "yes"       # window 1 states it
    assert out["a001"].plausible == "no"       # other criteria from the originating window 0
    assert out["a002"].explicit == "no"        # no window states it
    assert out["a900"].explicit == "yes" and out["a900"].coherent == "yes"  # originating window is 1
    assert len(m.calls) == 2
    assert "finestra uno" in m.calls[0]["messages"][-1]["content"]
    assert "finestra due" in m.calls[1]["messages"][-1]["content"]


def test_evidence_is_shown_to_the_judge_only_when_the_text_is_windowed():
    m = Mock(default=echo_all)
    judge.judge_batch(m, "gemma4:31b", ["uno", "due"], attrs(1))
    assert "prova 1" in m.calls[0]["messages"][-1]["content"]


def test_explicit_invalid_in_a_window_without_a_yes_stays_invalid():
    w0 = {"verdicts": [v(1, explicit="no")]}
    bad = {"verdicts": [v(1, explicit="maybe")]}
    m = Mock([w0] + [bad] * 5)
    out = judge.judge_batch(m, "gemma4:31b", ["uno", "due"], attrs(1), max_retries=1)
    assert out["a001"].explicit == "invalid"


# ---- three judges in parallel -------------------------------------------------------------------------------
def test_judge_step_runs_every_judge_and_returns_votes_and_majority():
    answers = {"j1": v(1, plausible="yes"), "j2": v(1, plausible="yes"), "j3": v(1, plausible="no")}

    class Multi:
        def __init__(self):
            self.models = []

        def chat(self, model, messages, options=None, schema=None):
            self.models.append(model)
            r = {"verdicts": [answers[model]]}
            return LLMResult(text=json.dumps(r), json=r, ok=True)

    c = Multi()
    out = judge.judge_step(c, ["j1", "j2", "j3"], ["T"], attrs(1))
    assert sorted(c.models) == ["j1", "j2", "j3"]
    r = out["a001"]
    assert set(r["votes"]) == {"j1", "j2", "j3"} and r["majority"]["plausible"] == "yes"


def test_judge_step_with_one_failing_judge_is_undecided_for_its_attributes():
    class Mixed:
        def chat(self, model, messages, options=None, schema=None):
            if model == "j3":
                return LLMResult(ok=False, error="http_500")
            r = {"verdicts": [v(1)]}
            return LLMResult(text=json.dumps(r), json=r, ok=True)

    out = judge.judge_step(Mixed(), ["j1", "j2", "j3"], ["T"], attrs(1), max_retries=0)
    assert all(label == "undecided" for label in out["a001"]["majority"].values())
    assert out["a001"]["votes"]["j3"] == judge.Vote(*["invalid"] * 4)


# ---- prompt: single source of the criteria is the codebook -----------------------------------------------------
def test_prompt_definitions_come_from_the_codebook():
    from pathlib import Path
    codebook = (Path(__file__).resolve().parents[2] / "docs" / "codebook.md").read_text(encoding="utf-8")
    flat = " ".join(codebook.split())
    prompt = " ".join(judge.INSTRUCTION.split())
    phrases = [
        "Is the attribute credible for this person, given everything in the documents?",
        "it fits the role, career and context described",
        "Plausibility does not require proof.",
        "consistent with the documents and with the other attributes inferred for the same person at the same step",
        "If two documents contradict each other, judge against the more recent one",
        "concrete, specific fact about the person that could be true in the real world",
        "rather than a generic statement, a speculation about inner states, or boilerplate that applies to anyone in the role",
        "stated in the documents, as such or as an obvious rewording",
        "including paraphrases, translations, abbreviations or a direct combination of two adjacent fields of the same record",
        "reaching it requires an inference across facts, documents or time",
    ]
    for ph in phrases:
        assert ph in flat, f"not in the codebook any more: {ph}"
        assert ph in prompt, f"not in the judge prompt: {ph}"


def test_prompt_is_in_english_with_the_text_as_is_and_asks_only_yes_no():
    p = judge.build_prompt("Il dirigente ha ricevuto l'incarico.", [(1, "attributo", "")])
    assert "You are an expert auditor" in p and "Il dirigente ha ricevuto l'incarico." in p
    assert "cannot_tell" not in p  # that label is for the human annotators only
    assert "yes" in p and "no" in p


def test_other_attributes_of_the_step_are_listed_as_context_when_there_are_several_batches():
    m = Mock(default=echo_all)
    judge.judge_batch(m, "gemma4:31b", ["T"], attrs(45))
    first, second = (c["messages"][-1]["content"] for c in m.calls)
    assert "OTHER ATTRIBUTES OF THE SAME PERSON AT THIS STEP" in first
    assert "- attributo numero 45" in first and "- attributo numero 1\n" not in first   # 41-45 are the others
    assert "- attributo numero 1\n" in second and "- attributo numero 41" not in second.split("ATTRIBUTES TO JUDGE")[0]


def test_no_context_block_when_all_attributes_fit_in_one_call():
    m = Mock(default=echo_all)
    judge.judge_batch(m, "gemma4:31b", ["T"], attrs(5))
    assert "OTHER ATTRIBUTES" not in m.calls[0]["messages"][-1]["content"]


def test_retry_of_a_missing_attribute_lists_the_already_judged_ones_as_context():
    m = Mock([{"verdicts": [v(1), v(3)]}, {"verdicts": [v(2)]}])
    judge.judge_batch(m, "gemma4:31b", ["T"], attrs(3))
    retry = m.calls[1]["messages"][-1]["content"]
    assert "OTHER ATTRIBUTES" in retry and "- attributo numero 1" in retry and "- attributo numero 3" in retry


def test_judge_one_gets_the_same_text_and_the_other_attributes_as_a_batch():
    allattrs = attrs(5)
    mb = Mock(default=echo_all)
    judge.judge_batch(mb, "gemma4:31b", ["TESTO COMUNE"], allattrs[:1], all_attrs=allattrs)   # batch of one in a step of 5
    mo = Mock([{"verdicts": [v(1)]}])
    judge.judge_one(mo, "gemma4:31b", ["TESTO COMUNE"], allattrs[0], context_attrs=allattrs)
    pb, po = mb.calls[0]["messages"][-1]["content"], mo.calls[0]["messages"][-1]["content"]
    assert pb == po                                              # same request in every respect
    assert "TESTO COMUNE" in po and "OTHER ATTRIBUTES" in po
    for i in (2, 3, 4, 5):
        assert f"- attributo numero {i}" in po
    assert "- attributo numero 1\n" not in po                    # the judged attribute is not its own context


def test_judge_one_in_a_step_with_several_batches_matches_the_batch_context():
    allattrs = attrs(45)
    mb = Mock(default=echo_all)
    judge.judge_batch(mb, "gemma4:31b", ["T"], allattrs)           # batches of 40 and 5 over the full list
    mo = Mock([{"verdicts": [v(1)]}])
    judge.judge_one(mo, "gemma4:31b", ["T"], allattrs[44], context_attrs=allattrs)
    po = mo.calls[0]["messages"][-1]["content"]
    assert "- attributo numero 1\n" in po and "- attributo numero 44" in po and "- attributo numero 45\n" not in po


def test_judge_one_without_context_is_alone_and_has_no_context_block():
    mo = Mock([{"verdicts": [v(1)]}])
    judge.judge_one(mo, "gemma4:31b", ["T"], attrs(1)[0])
    assert "OTHER ATTRIBUTES" not in mo.calls[0]["messages"][-1]["content"]


def test_seed_offset_changes_the_seed_of_the_request_and_nothing_else():
    a = Mock([{"verdicts": [v(1)]}])
    b = Mock([{"verdicts": [v(1)]}])
    judge.judge_batch(a, "gemma4:31b", ["T"], attrs(1))
    judge.judge_batch(b, "gemma4:31b", ["T"], attrs(1), seed_offset=100)
    assert b.calls[0]["options"]["seed"] == a.calls[0]["options"]["seed"] + 100
    assert a.calls[0]["messages"] == b.calls[0]["messages"]


def test_judge_one_passes_the_seed_offset():
    m = Mock([{"verdicts": [v(1)]}])
    judge.judge_one(m, "gemma4:31b", ["T"], attrs(1)[0], seed_offset=7)
    assert m.calls[0]["options"]["seed"] == config.SEED + 7


def test_ultra_judges_one_attribute_per_call_with_the_others_as_context_and_the_other_judges_in_batches():
    ultra, gemma = "nemotron-3-ultra", "gemma4:31b"
    a = Mock([{"verdicts": [v(1)]}] * 3)
    judge.judge_batch(a, ultra, ["T"], attrs(3))
    assert len(a.calls) == 3
    assert all("attributo numero 2" in c["messages"][0]["content"] for c in a.calls)       # context of the step
    b = Mock([{"verdicts": [v(1), v(2), v(3)]}])
    judge.judge_batch(b, gemma, ["T"], attrs(3))
    assert len(b.calls) == 1
    c = Mock([{"verdicts": [v(1), v(2), v(3)]}])
    judge.judge_batch(c, ultra, ["T"], attrs(3), batch_size=40)                             # explicit size wins
    assert len(c.calls) == 1


def test_one_attribute_calls_of_a_step_run_side_by_side_and_keep_the_stats():
    import threading, time as _t
    live = {"now": 0, "max": 0}
    lock = threading.Lock()

    class Slow(Mock):
        def chat(self, model, messages, options=None, schema=None, think=None):
            with lock:
                live["now"] += 1
                live["max"] = max(live["max"], live["now"])
            _t.sleep(0.05)
            with lock:
                live["now"] -= 1
            return super().chat(model, messages, options, schema)

    m = Slow(default=echo_all)
    st = judge_counter = __import__("collections").Counter()
    out = judge.judge_batch(m, "nemotron-3-ultra", ["T"], attrs(6), stats=st)
    assert set(out) == {f"a{i:03d}" for i in range(1, 7)} and live["max"] > 1
    assert st["calls"] == 6
