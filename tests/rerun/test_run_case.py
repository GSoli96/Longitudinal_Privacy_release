"""Tests of rerun.run_case: stages, invariants, alarms, resume after an interruption (fake transport, real LLMClient)."""
import hashlib
import json
import re
from collections import Counter

import numpy as np
import pytest

from rerun import config, run_case
from rerun.corpus import Doc, Person
from rerun.llm import LLMClient

ATT, J1, J2, J3 = config.ATTACKER, *config.JUDGES


def fake_transport_factory(fail_attacker_on=None, explicit="no"):
    calls = Counter()

    def transport(url, headers, body, timeout):
        model = body["model"]
        prompt = body["messages"][-1]["content"]
        fmt = body.get("format") or {}
        props = fmt.get("properties", {})
        calls[model] += 1
        if model == ATT:
            if fail_attacker_on and fail_attacker_on in prompt:
                return 500, {}
            docs = re.findall(r"\[Document (\d+)\]\n(.*)", prompt)
            items = [{"attribute": f"Attributo dal documento {n}", "category": "role", "evidence": body_text[:25]}
                     for n, body_text in docs]
            if "ATTRIBUTO FALSO" in prompt:
                items.append({"attribute": "Attributo falso inventato", "category": "x", "evidence": "non nel testo"})
            return 200, {"message": {"content": json.dumps({"attributes": items})}, "prompt_eval_count": 10,
                         "eval_count": 5, "done_reason": "stop"}
        pairs = re.findall(r"^(\d+)\.\nA: (.*)\nB: (.*)$", prompt, re.M)
        if "same" in json.dumps(props):                         # de-duplication confirmation
            out = [{"id": int(n), "same": "yes" if a == b else "no"} for n, a, b in pairs]
        else:                                                    # judges
            block = prompt.split("ATTRIBUTES TO JUDGE")[1]
            out = []
            for n, text in re.findall(r"^(\d+)\. (.*)$", block, re.M):
                bad = "falso" in text
                out.append({"id": int(n), "plausible": "no" if bad else "yes", "coherent": "no" if bad else "yes",
                            "realistic": "no" if bad else "yes", "explicit": "no" if bad else explicit})
        return 200, {"message": {"content": json.dumps({"verdicts": out})}, "prompt_eval_count": 10,
                     "eval_count": 5, "done_reason": "stop"}
    return transport, calls


def make_client(tmp, transport):
    return LLMClient(keys=["k1", "k2"], transport=transport, cache_path=tmp / "cache.jsonl",
                     usage_path=tmp / "usage.jsonl", sleep=lambda s: None, private_root=tmp)


class FakeEmbed:
    def __call__(self, texts):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([(b - 127.5) for b in h[:16]])
        v = np.array(out, dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def person(pid, n_docs, extra=""):
    docs = [Doc(id=f"{pid}-d{i}", date=None, kind="pdf", text=f"Testo del documento numero {i} della persona {pid}. " * 4 + extra)
            for i in range(1, n_docs + 1)]
    return Person(pid, "X", None, None, docs)


PERSONS = lambda: [person("X-aaaaaa", 3), person("X-bbbbbb", 4)]


def run_all(tmp, transport, persons=None, variant="evidence", **kw):
    persons = persons or PERSONS()
    c = make_client(tmp, transport)
    d = tmp / "stages"
    d.mkdir(exist_ok=True)
    s = {"infer": run_case.run_infer(persons, c, ATT, d / "infer.jsonl", usable_tokens=100000, workers=1,
                                  variant=variant)}
    s["judge"] = run_case.run_judge(persons, d / "infer.jsonl", c, config.JUDGES, d / "judge.jsonl", attacker=ATT,
                                    usable_tokens=100000, workers=1)
    s["dedup"] = run_case.run_dedup(d / "infer.jsonl", d / "judge.jsonl", c, FakeEmbed(), d / "dedup.jsonl", workers=1)
    s["metrics"] = run_case.run_metrics(persons, d / "infer.jsonl", d / "judge.jsonl", d / "dedup.jsonl", tmp / "res",
                                        FakeEmbed(), config.JUDGES)
    return s, c, d


# ---- infer ------------------------------------------------------------------------------------------------------
def test_infer_writes_a_record_per_step_and_carries_identical_steps(tmp_path):
    t, calls = fake_transport_factory()
    s, c, d = run_all(tmp_path, t)
    rows = run_case._read_jsonl(d / "infer.jsonl")
    assert len(rows) == 8 and s["infer"]["steps_total"] == 8
    p1 = [r for r in rows if r["person"] == "X-aaaaaa"]
    assert [r["carried"] for r in p1] == [False, False, False, True]      # 3 docs: step 100 equals step 75
    assert p1[3]["attributes"] == [] and p1[3]["carried_from"] == 75
    assert s["infer"]["steps_run"] == 7 and s["infer"]["steps_carried"] == 1
    assert calls[ATT] == 7                                                 # one call per step actually run
    ids = [a["id"] for r in rows for a in r["attributes"]]
    assert len(ids) == len(set(ids)) and all(re.fullmatch(r"X-[a-z]{6}-s\d+-a\d{3}", i) for i in ids)


def test_infer_invariants_and_summary_counts(tmp_path):
    t, _ = fake_transport_factory()
    s, c, d = run_all(tmp_path, t)
    rows = run_case._read_jsonl(d / "infer.jsonl")
    assert run_case.check_infer_invariants(PERSONS(), rows) == []
    assert s["infer"]["persons_in"] == s["infer"]["persons_out"] == 2
    assert s["infer"]["evidence_exact"] == s["infer"]["attributes"] and s["infer"]["evidence_none"] == 0
    bad = [r for r in rows if r["person"] == "X-aaaaaa" and r["step"] == 50] and [r for r in rows if r["step"] != 50]
    assert run_case.check_infer_invariants(PERSONS(), bad) != []          # a missing step is detected


def test_alarms_need_the_minimum_number_of_observations_and_stop_the_stage():
    """D58: a rate above its threshold is an alarm only with ALARM_MIN_CALLS observations; the stage then stops."""
    class C:
        stats = {"calls": 500}

        def key_error_rate(self):
            return 0.0
    few = run_case._alarms(5, config.ALARM_MIN_CALLS - 1, C(), kind="invalid")
    many = run_case._alarms(5, config.ALARM_MIN_CALLS, C(), kind="invalid")
    assert few["invalid_alarm"] is False and many["invalid_alarm"] is True and many["invalid_rate"] == 0.05
    assert run_case.active_alarms({"alarms": many}) == ["invalid_alarm"]
    assert run_case.active_alarms({"alarms": few}) == []
    assert run_case.active_alarms({"alarms": few, "invariants": ["x"], "problems": ["y"]}) == ["invariants", "problems"]
    # a client that made only a few network calls (stage read back from the cache) raises no key-error alarm
    class Cached:
        stats = {"calls": 3}

        def key_error_rate(self):
            return 0.667
    assert run_case._alarms(0, 1000, Cached())["key_error_alarm"] is False


def test_persons_lost_between_stages_are_listed():
    prev = [{"person": "X-aaaaaa"}, {"person": "X-bbbbbb"}, {"person": "X-bbbbbb"}]
    assert run_case.persons_lost(prev, [{"person": "X-bbbbbb"}]) == ["X-aaaaaa"]
    assert run_case.persons_lost(prev, prev) == []


def test_failed_step_is_counted_and_raises_the_alarm_never_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ALARM_MIN_CALLS", 1)
    t, _ = fake_transport_factory(fail_attacker_on="ZZZ")
    ps = [person("X-aaaaaa", 3, extra="ZZZ"), person("X-bbbbbb", 3)]
    s, c, d = run_all(tmp_path, t, ps)
    assert s["infer"]["steps_failed"] > 0 and s["infer"]["alarms"]["failed_alarm"] is True
    rows = run_case._read_jsonl(d / "infer.jsonl")
    assert all(r["error"] for r in rows if r["person"] == "X-aaaaaa" and not r["carried"])
    assert s["infer"]["persons_out"] == 2                                  # persons are never dropped


# ---- resume ------------------------------------------------------------------------------------------------------
def test_resume_after_interruption_repeats_no_completed_call(tmp_path):
    t0, calls0 = fake_transport_factory()
    run_all(tmp_path / "base", t0) if (tmp_path / "base").mkdir() is None else None
    baseline = sum(calls0.values())

    t, calls = fake_transport_factory()
    tmp = tmp_path / "interrupted"
    tmp.mkdir()
    c = make_client(tmp, t)
    d = tmp / "stages"
    d.mkdir()
    api = run_case.InterruptingClient(c, 4)
    with pytest.raises(run_case.Interrupted):
        run_case.run_infer(PERSONS(), api, ATT, d / "infer.jsonl", usable_tokens=100000, workers=1)
    first = sum(calls.values())
    assert first == 4
    # new process: new client, same cache; the whole pipeline now completes
    c2 = make_client(tmp, t)
    run_case.run_infer(PERSONS(), c2, ATT, d / "infer.jsonl", usable_tokens=100000, workers=1)
    run_case.run_judge(PERSONS(), d / "infer.jsonl", c2, config.JUDGES, d / "judge.jsonl", attacker=ATT,
                       usable_tokens=100000, workers=1)
    run_case.run_dedup(d / "infer.jsonl", d / "judge.jsonl", c2, FakeEmbed(), d / "dedup.jsonl", workers=1)
    assert sum(calls.values()) == baseline                                  # no call was made twice


# ---- judge -----------------------------------------------------------------------------------------------------
def test_judge_stage_judges_every_inferred_attribute_with_three_judges(tmp_path):
    t, calls = fake_transport_factory()
    s, c, d = run_all(tmp_path, t)
    inf = run_case._read_jsonl(d / "infer.jsonl")
    jr = run_case._read_jsonl(d / "judge.jsonl")
    n_attr = sum(len(r["attributes"]) for r in inf)
    assert len(jr) == n_attr == s["judge"]["attributes_in"] == s["judge"]["attributes_judged"]
    assert s["judge"]["invariants"] == []
    assert set(jr[0]["votes"]) == {J1, J2, J3} and all(len(v) == 4 for v in jr[0]["votes"].values())
    assert jr[0]["majority"] == {"plausible": "yes", "coherent": "yes", "realistic": "yes", "explicit": "no"}
    assert calls[J2] >= s["judge"]["steps_judged"] and calls[J3] >= s["judge"]["steps_judged"]


def test_judges_read_the_same_windows_as_the_attacker_when_the_step_is_chunked(tmp_path):
    t, calls = fake_transport_factory()
    long = person("X-cccccc", 1, extra="parola " * 4000)
    c = make_client(tmp_path, t)
    d = tmp_path / "s"
    d.mkdir()
    s = run_case.run_infer([long], c, ATT, d / "infer.jsonl", usable_tokens=2000, workers=1)
    assert s["steps_chunked"] >= 1 and s["windows_total"] > s["steps_run"]
    sj = run_case.run_judge([long], d / "infer.jsonl", c, config.JUDGES, d / "judge.jsonl", attacker=ATT,
                            usable_tokens=2000, workers=1)
    assert sj["invariants"] == []                                          # window counts match (asserted inside)


# ---- dedup and metrics --------------------------------------------------------------------------------------------
def test_dedup_merges_confirmed_repeats_across_steps(tmp_path):
    t, _ = fake_transport_factory()
    s, c, d = run_all(tmp_path, t)
    # "Attributo dal documento 1" appears in every step of a person: one cluster with several members
    assert s["dedup"]["n_confirmed"] > 0 and s["dedup"]["n_invalid"] == 0
    rows = run_case._read_jsonl(d / "dedup.jsonl")
    sizes = sorted(len(r["attr_ids"]) for r in rows)
    assert sizes[-1] >= 3 and s["dedup"]["largest_cluster"] >= 3


def test_metrics_by_step_with_carry_forward_novelty_and_pes(tmp_path):
    t, _ = fake_transport_factory()
    s, c, d = run_all(tmp_path, t)
    m = s["metrics"]
    assert set(m["by_step"]) == {"25", "50", "75", "100"}
    assert m["by_step"]["100"]["n_persons"] == 2 and m["by_step"]["100"]["n_carried"] == 3
    assert m["by_step"]["25"]["DIR"] == pytest.approx(1.0)                  # all inferred, never explicit
    assert m["carry_forward"] == {"25": 0, "50": 0, "75": 0, "100": 1}
    assert m["novelty_exact"]["25"] is None and m["novelty_exact"]["50"] == pytest.approx(0.5)
    assert m["novelty_confirmed"]["50"] == pytest.approx(0.5)
    assert m["pes"]["confirmed"]["100"]["mean"] == pytest.approx(3.5)       # 3 and 4 distinct attributes
    assert m["pes"]["exact"]["100"]["mean"] == pytest.approx(3.5)


def test_hallucinated_attribute_is_rejected_by_all_and_counted(tmp_path):
    t, _ = fake_transport_factory()
    ps = [person("X-aaaaaa", 2, extra="ATTRIBUTO FALSO")]
    s, c, d = run_all(tmp_path, t, ps)
    combos = s["metrics"]["label_combinations_final"]
    assert combos["0000"]["count"] == 1 and s["metrics"]["by_step"]["100"]["hallucination"] > 0
    assert s["metrics"]["pes"]["confirmed"]["100"]["mean"] == pytest.approx(2)  # the false one does not count


def test_agreement_and_per_judge_metrics_are_reported(tmp_path):
    t, _ = fake_transport_factory()
    s, c, d = run_all(tmp_path, t)
    m = s["metrics"]
    assert set(m["by_judge"]) == {J1, J2, J3} and m["by_judge"][J1]["100"]["n_total"] > 0
    assert set(m["agreement"]) == set(("plausible", "coherent", "realistic", "explicit"))


def test_metrics_file_has_aggregates_only_no_ids_or_text(tmp_path):
    t, _ = fake_transport_factory()
    s, c, d = run_all(tmp_path, t)
    text = (tmp_path / "res" / "metrics.json").read_text(encoding="utf-8")
    assert "X-aaaaaa" not in text and "Attributo dal documento" not in text and "Testo del documento" not in text


def test_metrics_report_reidentification_and_trend_blocks(tmp_path):
    t, _ = fake_transport_factory()
    s, c, d = run_all(tmp_path, t)
    m = s["metrics"]
    assert set(m["reidentification"]) == {"step", "all", "dangerous"} and "exact" in m["reidentification"]["all"]
    assert set(m["trend"]) == {"DIR", "IQS", "hallucination"}
    assert set(m["trend"]["DIR"]) == {"permutation", "mixed_model"} and "slope" in m["trend"]["DIR"]["permutation"]


def test_select_persons_is_deterministic_and_limited():
    ps = [person(f"X-{c}", 2) for c in "dcba"]
    assert [p.id for p in run_case.select_persons(ps, 2)] == ["X-a", "X-b"]
    assert [p.id for p in run_case.select_persons(ps)] == ["X-a", "X-b", "X-c", "X-d"]


# ---- pilot variants: stage directories, judge text truncation, evidence-free attacker, subset of another run ---------
def test_stage_dir_is_separate_per_tag(tmp_path):
    a = run_case.stage_dir("C", base=tmp_path)
    b = run_case.stage_dir("C", base=tmp_path, tag="noev")
    assert a != b and a.name == "stages" and b.name == "stages_noev" and a.exists() and b.exists()


def test_judges_can_read_only_the_first_n_characters_of_the_step(tmp_path):
    seen = []
    t, calls = fake_transport_factory()

    def spy(url, headers, body, timeout):
        if body["model"] in (J1, J2, J3) and "same" not in json.dumps((body.get("format") or {}).get("properties", {})):
            prompt = body["messages"][-1]["content"]
            seen.append(len(prompt.split("ORIGINAL TEXT:")[1].split("ATTRIBUTES TO JUDGE")[0]))
        return t(url, headers, body, timeout)

    persons = [person("X-aaaaaa", 4)]
    c = make_client(tmp_path, spy)
    d = tmp_path / "s"
    d.mkdir()
    run_case.run_infer(persons, c, ATT, d / "infer.jsonl", usable_tokens=100000, workers=1)
    long_text_len = len(" ".join(x.text for x in persons[0].docs))
    run_case.run_judge(persons, d / "infer.jsonl", c, config.JUDGES, d / "judge_full.jsonl", attacker=ATT,
                       usable_tokens=100000, workers=1)
    full = max(seen)
    seen.clear()
    c2 = make_client(tmp_path / "c2" if (tmp_path / "c2").mkdir() is None else tmp_path, spy)
    s = run_case.run_judge(persons, d / "infer.jsonl", c2, config.JUDGES, d / "judge_cut.jsonl", attacker=ATT,
                           usable_tokens=100000, workers=1, max_chars=200)
    assert max(seen) < full and max(seen) < 200 + 400            # the text block is cut (plus fixed separators)
    assert s["attributes_judged"] == s["attributes_in"] and s["invariants"] == []
    assert s["judge_text_chars"] == 200


def test_infer_stage_can_run_the_evidence_free_variant(tmp_path):
    seen = []
    t, calls = fake_transport_factory()

    def spy(url, headers, body, timeout):
        if body["model"] == ATT:
            seen.append("evidence" in json.dumps(body.get("format")))
        return t(url, headers, body, timeout)

    c = make_client(tmp_path, spy)
    d = tmp_path / "s"
    d.mkdir()
    s = run_case.run_infer(PERSONS(), c, ATT, d / "infer.jsonl", usable_tokens=100000, workers=1, variant="no_evidence")
    rows = run_case._read_jsonl(d / "infer.jsonl")
    assert not any(seen) and rows[0]["variant"] == "no_evidence"
    assert all(a["evidence_match"] == "n/a" for r in rows for a in r["attributes"])
    assert s["attributes"] > 0 and s["evidence_exact"] == 0


def test_subset_of_another_run_keeps_only_the_given_persons(tmp_path):
    t, _ = fake_transport_factory()
    s, c, d = run_all(tmp_path, t)
    out = tmp_path / "sub.jsonl"
    n = run_case.copy_infer_subset(d / "infer.jsonl", out, ["X-bbbbbb"])
    rows = run_case._read_jsonl(out)
    assert n == len(rows) and {r["person"] for r in rows} == {"X-bbbbbb"} and len(rows) == 4


# ---- batched vs one-by-one judging (pilot, spec 10b.1: kappa >= 0.80 for every criterion) ------------------------------
def test_batch_vs_single_agrees_perfectly_when_the_judge_is_consistent(tmp_path):
    t, calls = fake_transport_factory()
    ps = [person("X-aaaaaa", 4), person("X-bbbbbb", 4)]
    s, c, d = run_all(tmp_path, t, ps)
    before = calls[J1]
    out = run_case.run_batch_vs_single(ps, d / "infer.jsonl", d / "judge.jsonl", c, config.JUDGES, d / "bvs.jsonl",
                                       n=6, seed=1, workers=1)
    assert out["n_attributes"] == 6 and calls[J1] > before                   # single calls were really made (ultra is already single)
    for j in config.JUDGES:
        for crit in ("plausible", "coherent", "realistic", "explicit"):
            r = out["by_judge"][j][crit]
            assert r["percent_agreement"] == 1.0 and r["n_valid_pairs"] == 6
    assert out["batching_accepted"] is True


def test_batch_vs_single_rejects_batching_when_a_criterion_disagrees(tmp_path):
    t, calls = fake_transport_factory()
    ps = [person("X-aaaaaa", 4), person("X-bbbbbb", 4)]
    s, c, d = run_all(tmp_path, t, ps)

    def flip(url, headers, body, timeout):
        st, payload = t(url, headers, body, timeout)
        if body["model"] == J1 and "ORIGINAL TEXT" in body["messages"][-1]["content"] and \
                len(re.findall(r"^\d+\. ", body["messages"][-1]["content"].split("ATTRIBUTES TO JUDGE")[1], re.M)) == 1:
            m = json.loads(payload["message"]["content"])
            for v in m["verdicts"]:
                v["explicit"] = "yes"                                     # the single-attribute judge always says explicit
            payload["message"]["content"] = json.dumps(m)
        return st, payload

    c2 = make_client(tmp_path / "c2" if (tmp_path / "c2").mkdir() is None else tmp_path, flip)
    out = run_case.run_batch_vs_single(ps, d / "infer.jsonl", d / "judge.jsonl", c2, config.JUDGES, d / "bvs2.jsonl",
                                       n=8, seed=1, workers=1)
    assert out["by_judge"][J1]["explicit"]["percent_agreement"] < 1.0
    assert out["batching_accepted"] is False and J1 in out["failing"] and "explicit" in out["failing"][J1]


def test_batch_vs_single_reports_gwet_ac1_and_does_not_fail_on_constant_labels(tmp_path):
    t, _ = fake_transport_factory()
    ps = [person("X-aaaaaa", 4)]
    s, c, d = run_all(tmp_path, t, ps)
    out = run_case.run_batch_vs_single(ps, d / "infer.jsonl", d / "judge.jsonl", c, config.JUDGES, d / "bvs.jsonl",
                                       n=4, seed=1, workers=1)
    r = out["by_judge"][J1]["realistic"]
    assert "gwet_ac1" in r and "cohen_kappa" in r                      # kappa is nan with constant labels, AC1 is 1.0
    assert r["gwet_ac1"] == pytest.approx(1.0)
    assert out["batching_accepted"] is True                              # a nan kappa with perfect agreement is accepted


def test_batch_vs_single_sample_is_deterministic_and_limited_to_available_attributes(tmp_path):
    t, _ = fake_transport_factory()
    ps = [person("X-aaaaaa", 4)]
    s, c, d = run_all(tmp_path, t, ps)
    a = run_case.run_batch_vs_single(ps, d / "infer.jsonl", d / "judge.jsonl", c, config.JUDGES, d / "bvs.jsonl",
                                     n=1000, seed=3, workers=1)
    assert a["n_attributes"] == s["judge"]["attributes_judged"]         # cannot sample more than exist
    b = run_case.sample_attributes(run_case._read_jsonl(d / "judge.jsonl"), 3, seed=5)
    b2 = run_case.sample_attributes(run_case._read_jsonl(d / "judge.jsonl"), 3, seed=5)
    assert [x["attr_id"] for x in b] == [x["attr_id"] for x in b2] and len(b) == 3


# ---- run-to-run noise of the judges (same request, another seed) ------------------------------------------------------
def test_retest_reports_noise_floor_for_batch_and_single_votes(tmp_path):
    t, calls = fake_transport_factory()
    ps = [person("X-aaaaaa", 4), person("X-bbbbbb", 4)]
    s, c, d = run_all(tmp_path, t, ps)
    bvs = run_case.run_batch_vs_single(ps, d / "infer.jsonl", d / "judge.jsonl", c, config.JUDGES, d / "bvs.jsonl",
                                       n=6, seed=1, workers=1)
    before = sum(calls.values())
    out = run_case.run_judge_retest(ps, d / "infer.jsonl", d / "judge.jsonl", d / "bvs.jsonl", c, config.JUDGES,
                                    d / "retest.jsonl", n=4, seed_offset=100, workers=1)
    assert sum(calls.values()) > before                                  # new seed: new requests, not cache hits
    assert out["n_attributes"] == 4
    r = out["by_judge"][J1]["explicit"]
    assert set(r) >= {"batch_vs_batch", "single_vs_single"} and r["batch_vs_batch"]["percent_agreement"] == 1.0
    assert r["single_vs_single"]["percent_agreement"] == 1.0


def test_batch_size_test_reuses_cache_for_small_steps_and_issues_new_requests_for_larger_ones(tmp_path):
    t, calls = fake_transport_factory()
    ps = [person("X-aaaaaa", 4), person("X-bbbbbb", 4)]
    s, c, d = run_all(tmp_path, t, ps)
    run_case.run_batch_vs_single(ps, d / "infer.jsonl", d / "judge.jsonl", c, config.JUDGES, d / "bvs.jsonl",
                                 n=6, seed=1, workers=1)
    before = sum(calls.values())
    out = run_case.run_batch_size_test(ps, d / "infer.jsonl", d / "judge.jsonl", d / "bvs.jsonl", c, config.JUDGES[1],
                                       d / "bs.jsonl", sizes=(10, 1), workers=1)
    assert out["model"] == config.JUDGES[1] and out["n_attributes"] == 6
    big = out["by_size"]["b10"]
    assert big["n_attributes_with_different_request"] == 0
    assert big["explicit"]["vs_single"]["percent_agreement"] == 1.0           # same request as the main run
    assert out["by_size"]["b1"]["n_attributes_with_different_request"] > 0
    assert sum(calls.values()) > before                                       # size 1 needs new requests


def test_concurrency_calibration_runs_real_work_per_level_and_reports_latency_and_events(tmp_path):
    t, calls = fake_transport_factory()
    ps = [person(f"X-{c * 6}", 4) for c in "abcd"]
    s, c, d = run_all(tmp_path, t, ps)
    before = calls[J1]
    out = run_case.run_concurrency_calibration(ps, d / "infer.jsonl", c, J1, d / "cal.jsonl", levels=(2, 3),
                                               calls_per_level=4)
    assert set(out["levels"]) == {"2", "3"} and calls[J1] > before        # one-attribute calls were really made
    for lv in out["levels"].values():
        assert lv["calls_made"] >= 4 and lv["http_429"] == 0 and lv["failed"] == 0
        assert lv["latency_median"] == lv["latency_median"]                 # not NaN
    assert out["levels"]["2"]["steps"] >= 1 and out["levels"]["3"]["steps"] >= 1


def test_dedup_and_metrics_stop_when_the_vote_file_is_missing_or_empty(tmp_path):
    for stage in (lambda p: run_case.run_dedup(tmp_path / "i.jsonl", p, None, None, tmp_path / "d.jsonl"),
                  lambda p: run_case.run_metrics([], tmp_path / "i.jsonl", p, tmp_path / "d.jsonl", tmp_path)):
        with pytest.raises(SystemExit):
            stage(tmp_path / "missing.jsonl")
        (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(SystemExit):
            stage(tmp_path / "empty.jsonl")


def test_judges_are_read_from_the_votes_so_local_names_are_not_lost():
    rows = [{"votes": {"a-local": ["yes"], "b": ["no"]}}, {"votes": {"a-local": ["yes"], "b": ["no"], "c-local": ["yes"]}}]
    assert run_case._judges_in(rows) == ["a-local", "b", "c-local"]


def test_pes_cohort_leaves_out_persons_without_text_but_counts_them(tmp_path):
    """D54: persons with no step are not in the PES mean; metrics.json reports total, without text and in the cohort."""
    t, _ = fake_transport_factory()
    s, c, d = run_all(tmp_path, t)
    ghost = person("X-cccccc", 0)                          # a person whose only file has no readable text
    res = run_case.run_metrics([*PERSONS(), ghost], d / "infer.jsonl", d / "judge.jsonl", d / "dedup.jsonl",
                               tmp_path / "res2", FakeEmbed(), config.JUDGES)
    assert res["persons"] == 3 and res["persons_without_text"] == 1 and res["persons_in_cohort"] == 2
    assert res["pes"]["confirmed"]["100"]["n_persons"] == 2
    assert res["pes"]["confirmed"] == s["metrics"]["pes"]["confirmed"]       # the mean does not move


def test_missing_keys_file_gives_empty_keys_and_never_reads_it(tmp_path, capsys):
    def boom(path):
        raise AssertionError("the keys file must not be read when it does not exist")
    assert run_case.load_keys_if_any(boom, boom, tmp_path / "absent.json") == ([], [])
    assert "no keys file" in capsys.readouterr().err
    p = tmp_path / "keys.json"
    p.write_text("{}", encoding="utf-8")
    assert run_case.load_keys_if_any(lambda q: ["k"], lambda q: ["free"], p) == (["k"], ["free"])
