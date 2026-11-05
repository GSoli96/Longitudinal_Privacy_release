import json

from rerun import phish_run
from rerun.llm import LLMResult
from rerun.metrics import AttrRecord


def rec(person, text, explicit="no", step=100, plausible="yes"):
    return AttrRecord(person, step, text, {"plausible": plausible, "coherent": "yes", "realistic": "yes",
                                           "explicit": explicit}, False, f"{person}-{text}")


RECORDS = [rec("p1", "dedotto uno"), rec("p1", "dichiarato uno", explicit="yes"), rec("p1", "scartato", plausible="no"),
           rec("p1", "vecchio passo", step=50), rec("p2", "dedotto due"), rec("p3", "altro", plausible="no")]


class Client:
    def __init__(self):
        self.calls = 0

    def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
        self.calls += 1
        return LLMResult(ok=True, json={"subject": "S", "body": "B"})


def test_synthetic_name_is_stable_and_not_the_identifier():
    assert phish_run.synthetic_name("p1") == phish_run.synthetic_name("p1")
    assert "p1" not in phish_run.synthetic_name("p1") and " " in phish_run.synthetic_name("p1")
    assert len({phish_run.synthetic_name(f"x{i}") for i in range(50)}) > 20


def test_display_names_are_unique_within_a_file():
    recs = [rec(f"q{i}", "dedotto") for i in range(200)]
    persons = phish_run.persons_from_records(recs, "E")["persons"]
    names = [p["display_name"] for p in persons]
    assert len(persons) == 200 and len(set(names)) == 200
    assert names == [p["display_name"] for p in phish_run.persons_from_records(recs, "E")["persons"]]


def test_select_sample_is_seeded_sorted_and_whole_when_none():
    ids = [f"p{i}" for i in range(20)]
    a = phish_run.select_sample(ids, 5, seed=1)
    assert a == phish_run.select_sample(reversed(ids), 5, seed=1) and a == sorted(a) and len(a) == 5
    assert a != phish_run.select_sample(ids, 5, seed=2)
    assert phish_run.select_sample(ids, None) == sorted(ids) == phish_run.select_sample(ids, 99)


def test_persons_from_records_splits_and_keeps_every_sampled_person():
    meta = {"p1": {"role": "funzionario", "org_type": "comune"}}
    res = phish_run.persons_from_records(RECORDS, "E", meta=meta)
    by = {p["person"]: p for p in res["persons"]}
    assert set(by) == {"p1", "p2", "p3"}
    assert by["p1"]["inferred"] == ["dedotto uno"] and by["p1"]["explicit"] == ["dichiarato uno"]
    assert by["p1"]["base"] == ["Ruolo: funzionario", "Tipo di ente: comune"] and by["p2"]["base"] == []
    assert by["p3"]["inferred"] == [] and by["p3"]["explicit"] == []
    assert res["summary"]["n_sampled"] == 3 and res["summary"]["n_persons_without_attributes"] == 1
    assert all(p["case"] == "E" and p["display_name"] for p in res["persons"])
    assert len(phish_run.persons_from_records(RECORDS, "E", n=2)["persons"]) == 2


def test_prepare_then_generate_round_trip(tmp_path):
    summary = phish_run.prepare("E", "in", records=RECORDS, out_dir=tmp_path)
    assert summary["n_sampled"] == 3
    persons = phish_run._read_jsonl(tmp_path / "in.jsonl")
    c = Client()
    out = phish_run.generate(persons, "gen", "run1", client=c, out_dir=tmp_path)
    assert c.calls == 3 * 12 and out["corpus"]["n_complete"] == 3 and out["corpus"]["n_rows"] == 36
    assert (tmp_path / "run1.records.jsonl").exists() and (tmp_path / "run1.jsonl").exists()
    saved = json.loads((tmp_path / "run1.summary.json").read_text(encoding="utf-8"))
    assert saved["corpus"]["n_failed"] == 0 and "dedotto uno" not in json.dumps(saved)
    assert len(phish_run._read_jsonl(tmp_path / "run1.jsonl")) == 36


def test_cli_prepare_and_generate_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(phish_run, "load_stage_records", lambda case, stages=None: RECORDS)
    monkeypatch.setattr(phish_run, "OUT_DIR", tmp_path)
    assert phish_run.main(["prepare", "--case", "E", "--name", "in", "--n", "2"]) == 0
    assert len(phish_run._read_jsonl(tmp_path / "in.jsonl")) == 2
    monkeypatch.setattr(phish_run, "generate", lambda persons, *a, **k: {"n": len(persons)})
    assert phish_run.main(["generate", "--input", str(tmp_path / "in.jsonl"), "--model", "m", "--name", "r",
                           "--limit-persons", "1"]) == 0
