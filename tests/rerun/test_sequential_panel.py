"""Sequential judging (D41): majority of two, tie-breaker only where the first pair differs, audit sample, affinity."""
import json
import re
from collections import Counter

from rerun import judge, llm
from rerun.llm import LLMResult
from rerun.run_case import _agreement, build_records

C = judge.CRITERIA


def vote(*xs):
    return judge.Vote(*xs)


Y, N, I = vote("yes", "yes", "yes", "yes"), vote("no", "no", "no", "no"), judge.INVALID


def test_majority_of_two_equals_the_plain_majority_with_three_valid_votes():
    import itertools
    for xs in itertools.product(("yes", "no"), repeat=3):
        vs = [judge.Vote(*[x] * 4) for x in xs]
        assert judge.majority_of_two(vs) == judge.majority(vs)


def test_majority_of_two_decides_on_two_agreeing_votes_and_needs_two_valid():
    assert judge.majority_of_two([Y, Y])["plausible"] == "yes"
    assert judge.majority_of_two([Y, Y, I])["plausible"] == "yes"     # an invalid third does not undo two votes
    assert judge.majority_of_two([Y, N, I])["plausible"] == "undecided"
    assert judge.majority_of_two([Y, N])["plausible"] == "undecided"


def test_tie_break_needed_on_any_difference_or_invalid_vote():
    assert not judge.needs_tie_break(Y, Y)
    assert judge.needs_tie_break(Y, vote("yes", "yes", "yes", "no"))
    assert judge.needs_tie_break(Y, I)


def test_audit_sample_is_fixed_and_close_to_the_share():
    ids = [f"X-{i:05d}" for i in range(20000)]
    first = [judge.in_audit(i) for i in ids]
    assert first == [judge.in_audit(i) for i in ids]
    assert 0.09 < sum(first) / len(ids) < 0.11


class PanelClient:
    """gemma says yes to everything; the local judge says no to the attributes listed in ``local_no``; the
    tie-breaker says no to everything. Records which model saw which attribute."""

    def __init__(self, local_no=()):
        self.local_no, self.seen = set(local_no), Counter()

    def chat(self, model, messages, options=None, schema=None, think=None):
        prompt = messages[-1]["content"]
        block = prompt.split("ATTRIBUTES TO JUDGE")[1]
        items = re.findall(r"^(\d+)\. (.+)$", block, re.M)
        out = []
        for n, text in items:
            self.seen[(model, text.strip())] += 1
            no = model == "T" or (model == "L" and text.strip() in self.local_no)
            out.append({"id": int(n), **{c: "no" if no else "yes" for c in C}})
        r = {"verdicts": out}
        return LLMResult(text=json.dumps(r), json=r, ok=True, tokens_in=1, tokens_out=1, seconds=0.1)


def attrs(n):
    return [judge.JudgeAttr(id=f"Z-p-s10-a{i:03d}", text=f"attributo {i}") for i in range(1, n + 1)]


def test_tie_breaker_only_where_the_pair_differs_or_in_the_audit():
    at = attrs(40)
    cl = PanelClient(local_no={"attributo 3"})
    out = judge.judge_step_sequential(cl, ("G", "L"), "T", ["testo"], at)
    for a in at:
        asked = cl.seen[("T", a.text)] > 0
        expected = a.text == "attributo 3" or judge.in_audit(a.id)
        assert asked == expected
        assert out[a.id]["panel"] == ("tie" if a.text == "attributo 3" else "audit" if judge.in_audit(a.id) else "pair")
    assert out[at[2].id]["majority"]["plausible"] == "no"     # gemma yes, local no, tie-breaker no
    assert out[at[0].id]["majority"]["plausible"] == "yes"    # pair agrees: an audit "no" cannot change it
    assert set(out[at[0].id]["votes"]) <= {"G", "L", "T"}


def test_full_panel_asks_the_tie_breaker_everywhere():
    at = attrs(5)
    cl = PanelClient()
    out = judge.judge_step_sequential(cl, ("G", "L"), "T", ["testo"], at, full_panel=True)
    assert all(cl.seen[("T", a.text)] == 1 for a in at)
    assert {o["panel"] for o in out.values()} == {"full"}


def test_build_records_skips_attributes_a_judge_did_not_vote_and_agreement_skips_ties():
    infer_rows = [{"person": "p", "step": 10, "carried": False,
                   "attributes": [{"id": "a1", "text": "x"}, {"id": "a2", "text": "y"}]}]
    yes = ["yes"] * 4
    judge_rows = [{"attr_id": "a1", "votes": {"G": yes, "L": yes}, "majority": dict.fromkeys(C, "yes"),
                   "panel": "pair"},
                  {"attr_id": "a2", "votes": {"G": yes, "L": ["no"] * 4, "T": yes},
                   "majority": dict.fromkeys(C, "yes"), "panel": "tie"}]
    assert [r.attr_id for r in build_records(infer_rows, judge_rows, "T")] == ["a2"]
    assert len(build_records(infer_rows, judge_rows)) == 2
    assert _agreement(judge_rows, ["G", "L", "T"])["plausible"]["n_units"] == 0


class OkTransport:
    def __init__(self):
        self.keys = []

    def __call__(self, url, headers, body, timeout):
        self.keys.append(headers["Authorization"].split(" ", 1)[1])
        return 200, {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}


def test_affinity_sends_the_calls_of_a_step_to_the_same_key(tmp_path):
    keys = [f"k{i}" for i in range(5)]
    t = OkTransport()
    c = llm.LLMClient(keys, transport=t, sleep=lambda s: None, private_root=tmp_path)
    for i in range(6):
        c.chat("gemma4:31b", [{"role": "user", "content": f"a{i}"}], schema=None, affinity=123)
    assert len(set(t.keys)) == 1
    for i in range(5):
        c.chat("gemma4:31b", [{"role": "user", "content": f"b{i}"}], schema=None)
    assert len(set(t.keys[6:])) == 5          # without affinity: round robin as before


def test_person_sample_is_fixed_by_the_seed_and_ignores_the_input_order():
    from types import SimpleNamespace
    from rerun.run_case import sample_persons
    ps = [SimpleNamespace(id=f"A-{i:03d}") for i in range(250)]
    a, b = sample_persons(ps, 120), sample_persons(list(reversed(ps)), 120)
    assert [p.id for p in a] == [p.id for p in b] and len(a) == 120
    assert [p.id for p in a] == sorted(p.id for p in a)
    assert sample_persons(ps[:50], 120) == sorted(ps[:50], key=lambda p: p.id)
