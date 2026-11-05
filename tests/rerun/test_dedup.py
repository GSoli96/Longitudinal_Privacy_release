"""Tests of rerun.dedup: candidate pairs by cosine, LLM confirmation, clusters = components of confirmed edges."""
import json
import re
from collections import Counter

import numpy as np
import pytest

from rerun import dedup
from rerun.llm import LLMResult
from rerun.metrics import AttrRecord


def rec(text, step=25, person="p", carried=False):
    return AttrRecord(person=person, step=step, text=text, labels={}, carried=carried)


class Embed:
    """Vectors given per text; unit length."""

    def __init__(self, vecs):
        self.vecs = vecs

    def __call__(self, texts):
        out = np.array([self.vecs[t] for t in texts], dtype=float)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def unit(cos):
    """A vector at the given cosine with [1, 0]."""
    return [cos, float(np.sqrt(1 - cos * cos))]


class Judge:
    """Mock of the confirmation model: decides by a function of the two texts."""

    def __init__(self, rule, garble=()):
        self.rule, self.calls, self.garble = rule, [], set(garble)

    def chat(self, model, messages, options=None, schema=None):
        self.calls.append({"model": model, "messages": messages, "options": options, "schema": schema})
        prompt = messages[-1]["content"]
        pairs = re.findall(r"^(\d+)\.\nA: (.*)\nB: (.*)$", prompt, re.M)
        out = []
        for n, a, b in pairs:
            if (a, b) in self.garble or (b, a) in self.garble:
                out.append({"id": int(n), "same": "maybe"})
            else:
                out.append({"id": int(n), "same": "yes" if self.rule(a, b) else "no"})
        r = {"verdicts": out}
        return LLMResult(text=json.dumps(r), json=r, ok=True)


A = "Ha un incarico dirigenziale a tempo determinato"
A2 = "Ha un incarico dirigenziale a tempo indeterminato"      # near-miss fact: cosine 0.96 with A
B = "Coordina un ufficio con piu di dieci dipendenti"
B2 = "Coordina un ufficio con meno di cinque dipendenti"      # near-miss fact
P = "Possiede competenze nella gestione delle gare d'appalto pubbliche"
P2 = "Ha esperienza nella gestione di appalti pubblici"       # paraphrase of P
Z = "Parla correntemente una lingua straniera"
VECS = {A: [1.0, 0.0, 0.0], A2: [0.96, 0.28, 0.0], B: [0.0, 1.0, 0.0], B2: [0.0, 0.86, 0.5103],
        P: [0.0, 0.0, 1.0], P2: [0.0, 0.1, 0.995], Z: [-1.0, 0.0, 0.0]}


def rule_paraphrase(a, b):
    return {frozenset((P, P2))}.__contains__(frozenset((a, b)))


# ---- candidates -------------------------------------------------------------------------------------------
def test_candidate_pairs_by_cosine_threshold_and_never_with_carried_copies():
    recs = [rec(A, 25), rec(A2, 50), rec(Z, 50), rec(A, 75, carried=True)]
    cands = dedup.candidate_pairs(recs, Embed(VECS), threshold=0.80)
    texts = {frozenset((recs_[i].text, recs_[j].text)) for recs_ in [dedup._ordered(recs)] for i, j, _ in cands}
    assert texts == {frozenset((A, A2))}
    assert all(c >= 0.80 for _, _, c in dedup.candidate_pairs(recs, Embed(VECS), threshold=0.80))
    assert dedup.candidate_pairs(recs, Embed(VECS), threshold=0.99) == []


# ---- confirmation --------------------------------------------------------------------------------------------
def test_near_miss_facts_are_candidates_but_not_merged():
    recs = [rec(A, 25), rec(A2, 50), rec(B, 25), rec(B2, 75)]
    j = Judge(rule_paraphrase)
    clusters, info = dedup.confirmed_clusters(recs, Embed(VECS), j, threshold=0.80)
    assert len(clusters) == 4 and all(len(c) == 1 for c in clusters)
    assert info["n_candidates"] >= 1 and info["n_confirmed"] == 0 and info["n_rejected"] == info["n_candidates"]


def test_a_confirmed_paraphrase_is_merged_and_kept_in_step_order():
    recs = [rec(P2, 50), rec(P, 25), rec(Z, 25)]
    clusters, info = dedup.confirmed_clusters(recs, Embed(VECS), Judge(rule_paraphrase), threshold=0.80)
    merged = [c for c in clusters if len(c) == 2][0]
    assert [r.step for r in merged] == [25, 50] and merged[0].text == P
    assert info["n_confirmed"] == 1 and len(clusters) == 2


def test_clusters_are_connected_components_of_confirmed_edges():
    v = {"a": unit(1.0), "b": unit(0.9), "c": unit(0.81), "d": [0.0, -1.0], "e": unit(0.5)}
    # a-b and b-c confirmed, a-c candidate rejected: still one cluster {a, b, c}
    rule = lambda x, y: frozenset((x, y)) in {frozenset(("a", "b")), frozenset(("b", "c"))}
    recs = [rec("a", 25), rec("b", 50), rec("c", 75), rec("d", 75), rec("e", 100)]
    clusters, info = dedup.confirmed_clusters(recs, Embed(v), Judge(rule), threshold=0.80)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 1, 3]
    big = max(clusters, key=len)
    assert {r.text for r in big} == {"a", "b", "c"}


def test_clusters_with_more_than_three_attributes_are_counted():
    v = {t: unit(1.0 - 0.01 * i) for i, t in enumerate("abcde")}
    recs = [rec(t, 25 + 25 * (i % 4)) for i, t in enumerate("abcde")]
    clusters, info = dedup.confirmed_clusters(recs, Embed(v), Judge(lambda x, y: True), threshold=0.80)
    assert len(clusters) == 1 and info["clusters_gt3"] == 1 and info["largest_cluster"] == 5


def test_invalid_answer_does_not_merge_and_is_counted_apart_from_rejections():
    recs = [rec(P, 25), rec(P2, 50)]
    j = Judge(rule_paraphrase, garble=[(P, P2)])
    clusters, info = dedup.confirmed_clusters(recs, Embed(VECS), j, threshold=0.80, max_retries=1)
    assert len(clusters) == 2
    assert info["n_invalid"] == 1 and info["n_confirmed"] == 0 and info["n_rejected"] == 0
    assert len(j.calls) == 2                      # first try + one retry of the unanswered pair


def test_call_failure_is_retried_then_pairs_stay_unmerged():
    class Down:
        def __init__(self):
            self.n = 0

        def chat(self, *a, **k):
            self.n += 1
            return LLMResult(ok=False, error="http_500")
    d = Down()
    clusters, info = dedup.confirmed_clusters([rec(P, 25), rec(P2, 50)], Embed(VECS), d, max_retries=2)
    assert len(clusters) == 2 and info["n_invalid"] == 1 and d.n == 3


def test_pairs_are_sent_in_batches_with_the_temperature_zero_model_and_a_schema():
    v = {f"t{i}": unit(0.95) if i else unit(1.0) for i in range(8)}
    recs = [rec(f"t{i}", 25 + 25 * (i % 4)) for i in range(8)]
    j = Judge(lambda a, b: False)
    dedup.confirmed_clusters(recs, Embed(v), j, batch_size=10)
    assert len(j.calls) >= 2 and all(c["model"] == "gemma4:31b" for c in j.calls)
    assert j.calls[0]["schema"] == dedup.SCHEMA and j.calls[0]["options"]["num_ctx"] == 262144
    sizes = [len(re.findall(r"^\d+\.\nA: ", c["messages"][-1]["content"], re.M)) for c in j.calls]
    assert max(sizes) <= 10 and sum(sizes) == 28          # all C(8,2) pairs are candidates


def test_prompt_states_the_criterion_and_shows_both_statements_without_the_person():
    j = Judge(rule_paraphrase)
    dedup.confirmed_clusters([rec(P, 25, person="SECRET-ID"), rec(P2, 50, person="SECRET-ID")], Embed(VECS), j)
    prompt = j.calls[0]["messages"][-1]["content"]
    assert "same fact about the person" in prompt and "same values" in prompt
    assert "contract type" in prompt and P in prompt and P2 in prompt
    assert "SECRET-ID" not in prompt


def test_nothing_to_compare_makes_no_call():
    j = Judge(lambda a, b: True)
    clusters, info = dedup.confirmed_clusters([rec(A, 25)], Embed(VECS), j)
    assert len(clusters) == 1 and j.calls == [] and info["n_candidates"] == 0
    assert dedup.confirmed_clusters([], Embed(VECS), j)[0] == []


def test_stats_counter_is_filled():
    st = Counter()
    dedup.confirmed_clusters([rec(P, 25), rec(P2, 50)], Embed(VECS), Judge(rule_paraphrase), stats=st)
    assert st["calls"] == 1 and st["pairs"] == 1
