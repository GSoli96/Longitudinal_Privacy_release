"""Annotation by local LLM annotators (rerun.annotate): sampling, resume and agreement; fixtures only."""
import json
from types import SimpleNamespace

from rerun import annotate, config
from rerun.llm import LLMResult

ANN = config.ANNOTATORS


def rows(n_persons=6, per_step=5):
    out = []
    for p in range(n_persons):
        for k in (25, 50, 75, 100):
            out.append({"person": f"p{p}", "step": k, "carried": False, "windows": 1,
                        "attributes": [{"id": f"p{p}-{k}-{i}", "text": f"attr {p} {k} {i}", "evidence": "", "window": 0}
                                       for i in range(per_step)]})
    return out


def texts(person, k):
    long_ = person.id == "p0"           # one person whose texts are too long for the annotators
    return ["x" * (200000 if long_ else 300 + k)]


PEOPLE = [SimpleNamespace(id=f"p{i}") for i in range(6)]


def test_sample_is_stratified_seeded_and_skips_texts_that_do_not_fit():
    items, rep = annotate.build_items("C", PEOPLE, rows(), texts, n=20, seed=7)
    assert len(items) == 20 and rep["per_step"] == {"25": 5, "50": 5, "75": 5, "100": 5}
    assert all(it["_person"] != "p0" for it in items)
    again, _ = annotate.build_items("C", PEOPLE, rows(), texts, n=20, seed=7)
    assert [i["item_id"] for i in items] == [i["item_id"] for i in again]
    assert len({i["item_id"] for i in items}) == 20
    other, _ = annotate.build_items("C", PEOPLE, rows(), texts, n=20, seed=8)
    assert [i["item_id"] for i in items] != [i["item_id"] for i in other]
    assert sum(rep["skipped_too_long_or_windowed"].values()) >= 0


def test_a_short_stratum_is_made_up_by_the_others_and_the_shipped_items_hold_no_ids():
    few = [r for r in rows() if not (r["step"] == 25 and r["person"] != "p1")]   # step 25: only the 5 attributes of p1
    items, rep = annotate.build_items("C", PEOPLE, few, texts, n=24, seed=1)
    assert len(items) == 24 and rep["per_step"]["25"] == 5
    shipped, origin = annotate.split_private(items)
    assert all("_person" not in s and "_attr_id" not in s for s in shipped)
    assert set(origin) == {s["item_id"] for s in shipped}
    assert all(len(s["others"]) == 4 for s in shipped)


class Scripted:
    def __init__(self, vote_for):
        self.vote_for = vote_for
        self.calls = 0

    def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
        self.calls += 1
        v = self.vote_for(model)
        r = {"verdicts": [{"id": 1, **dict(zip(annotate.CRITERIA, v))}]}
        return LLMResult(text=json.dumps(r), json=r, ok=True)


def test_run_items_resumes_and_marks_invalid_answers(tmp_path):
    items, _ = annotate.build_items("C", PEOPLE, rows(), texts, n=8, seed=3)
    shipped, _ = annotate.split_private(items)
    out = tmp_path / "ann.jsonl"
    cl = Scripted(lambda m: ["yes", "no", "yes", "no"])
    s = annotate.run_items(shipped, ANN[1], cl, out)
    assert s["run"] == 8 and s["invalid"] == 0 and cl.calls == 8
    s2 = annotate.run_items(shipped, ANN[1], cl, out)
    assert s2["run"] == 0 and s2["skipped_done"] == 8 and cl.calls == 8
    bad = Scripted(lambda m: ["maybe", "no", "yes", "no"])
    s3 = annotate.run_items(shipped[:2], ANN[0], bad, out)
    assert s3["invalid"] == 2
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    assert sum(r["votes"] is None for r in recs) == 2


def test_agreement_uses_only_complete_items():
    res = []
    for i in range(10):
        for j, m in enumerate(ANN):
            v = "yes" if (i % 2 == 0) else "no"
            if m == ANN[2] and i == 9:
                continue                      # the third annotator did not answer item 9
            if m == ANN[1] and i < 2:
                v = "no" if v == "yes" else "yes"
            res.append({"item_id": f"i{i}", "model": m, "votes": [v, "yes", "yes", "yes"], "seconds": 1})
    a = annotate.agreement(res)
    assert a["items_complete"] == 9 and a["items_partial"] == 1
    c = a["criteria"]["plausible"]
    assert c["n"] == 9 and 0 < c["fleiss_kappa"] < 1 and abs(c["percent_all_equal"] - 7 / 9) < 1e-9
    assert set(c["cohen_kappa"]) == {f"{ANN[0]}|{ANN[1]}", f"{ANN[0]}|{ANN[2]}", f"{ANN[1]}|{ANN[2]}"}


def test_cli_run_does_not_double_the_output_limit(monkeypatch, tmp_path):
    seen = {}

    class FakeClient:
        def __init__(self, keys, **kw):
            seen.update(kw)

    monkeypatch.setattr("rerun.llm.LLMClient", FakeClient)
    monkeypatch.setattr(annotate, "run_items", lambda *a, **k: {"run": 0})
    items = tmp_path / "items.jsonl"
    items.write_text("")
    annotate.main(["run", "--items", str(items), "--model", ANN[1], "--out", str(tmp_path / "o.jsonl")])
    assert seen["grow_truncated"] is False


def test_a_truncated_answer_fails_at_once_without_growing_the_limit(tmp_path):
    from rerun.llm import LLMClient
    sent = []

    def transport(url, headers, body, timeout):
        sent.append(body["options"]["num_predict"])
        return 200, {"message": {"content": '{"verdicts": [{"id": 1,'}, "done_reason": "length",
                     "prompt_eval_count": 10, "eval_count": body["options"]["num_predict"]}

    kw = dict(transport=transport, private_root=tmp_path, sleep=lambda s: None, max_attempts=3)
    r = LLMClient([], grow_truncated=False, **kw).chat(ANN[1], [{"role": "user", "content": "x"}],
                                                       options={"num_predict": 512}, schema={"type": "object"})
    assert r.error == "truncated" and sent == [512]
    sent.clear()
    LLMClient([], **kw).chat(ANN[1], [{"role": "user", "content": "x"}],
                             options={"num_predict": 512}, schema={"type": "object"})
    assert sent[:3] == [512, 1024, 2048]            # the default keeps doubling (reasoning judges)


def test_cluster_takes_several_attributes_of_the_same_step_text():
    items, rep = annotate.build_items("C", PEOPLE, rows(per_step=5), texts, n=20, seed=7, cluster=5)
    assert len(items) == 20 and rep["cluster"] == 5
    assert rep["distinct_step_texts"] == 4          # 5 attributes of one text per step
    items1, rep1 = annotate.build_items("C", PEOPLE, rows(per_step=5), texts, n=20, seed=7)
    assert rep1["distinct_step_texts"] > 4
