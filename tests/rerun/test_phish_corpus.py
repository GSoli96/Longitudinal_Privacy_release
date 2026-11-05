import json

import pytest

from rerun import phish_corpus, phish_gen
from rerun.llm import LLMResult

KW = dict(company_name="Ente X", company_domain="entex")
PERSONS = [{"person": "p1", "case": "E", "display_name": "N1", "explicit": ["e"], "inferred": ["i"]},
           {"person": "p2", "case": "F", "display_name": "N2", "explicit": [], "inferred": ["i"]}]


class Client:
    def __init__(self, fail=lambda prompt, name: False):
        self.fail = fail

    def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
        prompt = messages[0]["content"]
        name = "N1" if "N1" in prompt else "N2"
        if self.fail(prompt, name):
            return LLMResult(ok=False, error="http_500")
        return LLMResult(ok=True, json={"subject": "S " + name, "body": "B " + name})


def _records(client=None, persons=PERSONS):
    return phish_gen.run(client or Client(), "gen", persons, **KW)["records"]


def test_complete_corpus_has_every_person_and_cell():
    out = phish_corpus.assemble(PERSONS, _records(), "gen")
    r = out["report"]
    assert r["n_persons"] == 2 and r["n_complete"] == 2 and r["n_absent"] == 0 and r["n_partial"] == 0
    assert r["n_rows"] == 24 and r["n_positive"] == 12 and r["n_negative"] == 12
    assert r["per_case"] == {"E": {"persons": 1, "complete": 1, "partial": 0, "absent": 0},
                             "F": {"persons": 1, "complete": 1, "partial": 0, "absent": 0}}
    assert {row["person"] for row in out["rows"]} == {"p1", "p2"}


def test_absent_person_raises_when_strict_and_is_reported_otherwise():
    recs = _records(Client(lambda p, n: n == "N2"))
    with pytest.raises(phish_corpus.IncompleteCorpus):
        phish_corpus.assemble(PERSONS, recs, "gen")
    out = phish_corpus.assemble(PERSONS, recs, "gen", strict=False)
    assert out["absent"] == ["p2"] and out["report"]["n_absent"] == 1
    assert out["report"]["failed_by_reason"] == {"call_failed:http_500": 12}
    assert all(row["person"] == "p1" for row in out["rows"])


def test_partial_person_is_flagged_and_failed_never_becomes_a_row():
    recs = _records(Client(lambda p, n: n == "N1" and "legittima" in p))
    out = phish_corpus.assemble(PERSONS, recs, "gen")
    assert out["partial"] == ["p1"] and out["report"]["n_partial"] == 1 and out["report"]["n_complete"] == 1
    assert out["report"]["n_rows"] == 18 and out["report"]["n_failed"] == 6


def test_inconsistent_records_are_errors():
    recs = _records()
    with pytest.raises(ValueError):
        phish_corpus.assemble(PERSONS, recs, "other")
    with pytest.raises(ValueError):
        phish_corpus.assemble(PERSONS[:1], recs, "gen")
    with pytest.raises(ValueError):
        phish_corpus.assemble(PERSONS, recs + [recs[0]], "gen")
    with pytest.raises(ValueError):
        phish_corpus.assemble(PERSONS + [PERSONS[0]], recs, "gen")
    bad = [dict(recs[0], case="Z")] + recs[1:]
    with pytest.raises(ValueError):
        phish_corpus.assemble(PERSONS, bad, "gen")


def test_more_negatives_change_the_expected_cells():
    recs = phish_gen.run(Client(), "gen", PERSONS, negatives_per_positive=2, **KW)["records"]
    assert phish_corpus.assemble(PERSONS, recs, "gen", negatives_per_positive=2)["report"]["n_partial"] == 0
    assert phish_corpus.assemble(PERSONS, recs, "gen", negatives_per_positive=1)["report"]["n_partial"] == 2


def test_write_load_and_tamper_detection(tmp_path):
    out = phish_corpus.assemble(PERSONS, _records(), "gen")
    entry = phish_corpus.write(out, "primary", tmp_path)
    assert entry["report"]["n_rows"] == 24
    assert len(phish_corpus.load("primary", tmp_path)) == 24
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "S N1" not in json.dumps(manifest)
    with open(tmp_path / "primary.jsonl", "a", encoding="utf-8") as f:
        f.write("{}\n")
    with pytest.raises(ValueError):
        phish_corpus.load("primary", tmp_path)
    with pytest.raises(FileNotFoundError):
        phish_corpus.load("missing", tmp_path)


def test_summary_has_counts_only():
    out = phish_corpus.assemble(PERSONS, _records(), "gen")
    text = phish_corpus.summary(out["report"])
    assert "persons 2 (complete 2" in text and "S N1" not in text and "p1" not in text
