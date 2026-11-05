"""Tests of rerun.categories (post-hoc classification with a fixed taxonomy) and the by-category metrics."""
import json
import re
from collections import Counter

import pytest

from rerun import categories, config, metrics
from rerun.llm import LLMResult


class Mock:
    def __init__(self, script=None, rule=None):
        self.script, self.rule, self.calls = list(script or []), rule, []

    def chat(self, model, messages, options=None, schema=None, think=None):
        self.calls.append({"model": model, "messages": messages, "options": options, "schema": schema, "think": think})
        if self.script:
            r = self.script.pop(0)
        else:
            prompt = messages[-1]["content"]
            ids = [int(x) for x in re.findall(r"^(\d+)\. ", prompt.split("ATTRIBUTES")[1], re.M)]
            texts = dict(re.findall(r"^(\d+)\. (.*)$", prompt.split("ATTRIBUTES")[1], re.M))
            r = {"classifications": [dict(id=i, **(self.rule(texts[str(i)]) if self.rule else
                                                  {"category": "role_career", "special": "no"})) for i in ids]}
        if isinstance(r, LLMResult):
            return r
        return LLMResult(text=json.dumps(r), json=r, ok=True)


def attrs(n):
    return [(f"a{i:03d}", f"attributo numero {i}") for i in range(1, n + 1)]


def test_taxonomy_is_the_fixed_one_of_the_spec_plus_other():
    assert categories.TAXONOMY == ("role_career", "organisation_network", "competences", "mobility_location",
                                   "economic", "special_categories", "other")


def test_classification_maps_ids_to_category_and_special_flag():
    m = Mock([{"classifications": [{"id": 1, "category": "economic", "special": "no"},
                                   {"id": 2, "category": "special_categories", "special": "yes"}]}])
    out = categories.classify_batch(m, "gemma4:31b", attrs(2))
    assert out["a001"] == categories.Classification("economic", False)
    assert out["a002"] == categories.Classification("special_categories", True)


def test_request_uses_schema_fixed_context_and_temperature_zero_model():
    m = Mock()
    categories.classify_batch(m, "gemma4:31b", attrs(3))
    c = m.calls[0]
    assert c["schema"] == categories.SCHEMA and c["options"]["num_ctx"] == min(config.num_ctx_for("gemma4:31b"), categories.NUM_CTX)
    assert c["model"] == "gemma4:31b" and "temperature" not in c["options"]   # the client default (0) applies


def test_context_is_capped_and_think_follows_the_judge_setting():
    m = Mock()
    categories.classify_batch(m, config.LOCAL_JUDGE, attrs(2))
    c = m.calls[0]
    assert c["options"]["num_ctx"] <= categories.NUM_CTX and c["think"] is False


def test_prompt_lists_the_taxonomy_and_only_the_attribute_text():
    m = Mock()
    categories.classify_batch(m, "gemma4:31b", [("a1", "Ha un ruolo direttivo")])
    p = m.calls[0]["messages"][-1]["content"]
    for t in categories.TAXONOMY:
        assert t in p
    assert "Ha un ruolo direttivo" in p and "GDPR" in p and "article 9" in p.lower()
    assert "evidence" not in p.lower() and "documents" not in p.lower().split("attributes")[1]


def test_batches_of_at_most_40_attributes():
    m = Mock()
    out = categories.classify_batch(m, "gemma4:31b", attrs(85))
    assert len(m.calls) == 3 and len(out) == 85


@pytest.mark.parametrize("bad", [{"id": 1, "category": "astrology", "special": "no"},
                                 {"id": 1, "category": "economic", "special": "maybe"},
                                 {"id": 1, "category": "economic"}, {"category": "economic", "special": "no"}])
def test_invalid_item_is_retried_then_unclassified_never_a_default_category(bad):
    m = Mock([{"classifications": [bad]}] * 5)
    stats = Counter()
    out = categories.classify_batch(m, "gemma4:31b", attrs(1), max_retries=2, stats=stats)
    assert out["a001"] == categories.UNCLASSIFIED and len(m.calls) == 3 and stats["unclassified"] == 1


def test_missing_attribute_is_retried_alone():
    m = Mock([{"classifications": [{"id": 1, "category": "economic", "special": "no"}]},
              {"classifications": [{"id": 2, "category": "competences", "special": "no"}]}])
    out = categories.classify_batch(m, "gemma4:31b", attrs(2))
    assert out["a002"].category == "competences" and len(m.calls) == 2
    assert "2. attributo numero 2" in m.calls[1]["messages"][-1]["content"]
    assert "1. attributo numero 1" not in m.calls[1]["messages"][-1]["content"]


def test_call_failure_is_retried_then_unclassified():
    m = Mock([LLMResult(ok=False, error="http_500")] * 5)
    out = categories.classify_batch(m, "gemma4:31b", attrs(2), max_retries=1)
    assert all(v == categories.UNCLASSIFIED for v in out.values()) and len(m.calls) == 2


def test_special_flag_and_category_are_independent_and_both_reported():
    m = Mock([{"classifications": [{"id": 1, "category": "organisation_network", "special": "yes"}]}])
    out = categories.classify_batch(m, "gemma4:31b", attrs(1))
    assert out["a001"].special is True and out["a001"].category == "organisation_network"
    assert categories.is_special(out["a001"]) is True
    assert categories.is_special(categories.Classification("special_categories", False)) is True
    assert categories.is_special(categories.Classification("economic", False)) is False


def test_run_categories_writes_one_line_per_attribute_and_summarises(tmp_path):
    infer = [{"person": "p1", "step": 50, "carried": False, "attributes": [
        {"id": "p1-s50-a001", "text": "uno", "category_hint": "x", "evidence": "e"},
        {"id": "p1-s50-a002", "text": "due", "category_hint": "x", "evidence": "e"}]},
        {"person": "p1", "step": 75, "carried": True, "attributes": []}]
    ip = tmp_path / "infer.jsonl"
    ip.write_text("\n".join(json.dumps(r) for r in infer), encoding="utf-8")
    m = Mock(rule=lambda t: {"category": "economic" if t == "uno" else "competences", "special": "no"})
    s = categories.run_categories(ip, tmp_path / "cat.jsonl", m, model="gemma4:31b", workers=1)
    rows = [json.loads(x) for x in (tmp_path / "cat.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["attr_id"]: r["category"] for r in rows} == {"p1-s50-a001": "economic", "p1-s50-a002": "competences"}
    assert s["attributes"] == 2 and s["unclassified"] == 0 and s["by_category"] == {"economic": 1, "competences": 1}


# ---- metrics by category ------------------------------------------------------------------------------------------
def rec(aid, bits, step=100):
    names = ("plausible", "coherent", "realistic", "explicit")
    lab = {n: {"1": "yes", "0": "no"}[b] for n, b in zip(names, bits)}
    return metrics.AttrRecord(person="p", step=step, text=aid, labels=lab, attr_id=aid)


def test_dir_by_category_and_special_share():
    recs = [rec("a1", "1110"), rec("a2", "1111"), rec("a3", "1110"), rec("a4", "0000")]
    cats = {"a1": categories.Classification("economic", False), "a2": categories.Classification("economic", False),
            "a3": categories.Classification("role_career", False), "a4": categories.Classification("economic", True)}
    out = categories.metrics_by_category(recs, cats)
    assert out["economic"]["n"] == 3 and out["economic"]["DIR"] == pytest.approx(1 / 3)
    assert out["role_career"]["DIR"] == pytest.approx(1.0) and out["role_career"]["n"] == 1
    assert out["_special"]["n_special"] == 1 and out["_special"]["share"] == pytest.approx(1 / 4)
    assert out["_special"]["DIR"] == pytest.approx(0.0)


def test_write_category_metrics_reads_the_categories_file_and_keeps_only_the_final_step(tmp_path):
    cp = tmp_path / "categories.jsonl"
    cp.write_text("\n".join(json.dumps(r) for r in (
        {"attr_id": "a1", "category": "economic", "special": False},
        {"attr_id": "a2", "category": "role_career", "special": True})), encoding="utf-8")
    recs = [rec("a1", "1110"), rec("a2", "1111"), rec("a3", "1110"), rec("a1", "0000", step=50)]
    out = categories.write_category_metrics(recs, cp, tmp_path / "out" / "metrics_by_category.json")
    saved = json.loads((tmp_path / "out" / "metrics_by_category.json").read_text(encoding="utf-8"))
    assert out["n_records"] == 3 and saved["step"] == 100 and saved["n_records"] == 3
    assert saved["categories"]["economic"]["n"] == 1 and saved["categories"]["_unclassified"]["n"] == 1
    assert saved["categories"]["_special"]["n_special"] == 1


def test_load_categories_maps_ids_to_classifications(tmp_path):
    cp = tmp_path / "categories.jsonl"
    cp.write_text(json.dumps({"attr_id": "a1", "category": "economic", "special": True}) + "\n", encoding="utf-8")
    assert categories.load_categories(cp) == {"a1": categories.Classification("economic", True)}


def test_metrics_by_category_counts_unclassified_apart():
    recs = [rec("a1", "1110"), rec("a2", "1110")]
    out = categories.metrics_by_category(recs, {"a1": categories.Classification("economic", False)})
    assert out["_unclassified"]["n"] == 1 and out["economic"]["n"] == 1
