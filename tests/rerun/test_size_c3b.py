"""Tests of the size study with the 3 B attacker (rerun.size_c3b, rerun.local_panel, the config entries): routing and
settings of the model, no reasoning field, a stage folder and a cache of their own, safeguards on the tag, the guard of
the local panel, and a dry run on the real persons of Case C when they are on the machine. Fake client, no network."""
import hashlib
import json
from pathlib import Path

import pytest

from rerun import config, infer, llm, local_panel, run_case, size_c3b
from rerun.llm import LLMClient, LLMResult

ATT = config.SIZE_ATTACKER_3B


# ---- config -------------------------------------------------------------------------------------------------------
def test_the_3b_alias_is_local_and_has_the_windows_of_the_main_attacker():
    assert ATT == "llama3.2:3b-local" and config.is_local(ATT)
    assert not config.is_local("llama3.2:3b")                       # the plain name keeps its old meaning
    assert config.num_ctx_for(ATT) == config.num_ctx_for("gpt-oss:20b") == 131072
    assert config.TOKEN_CHARS[ATT] == config.TOKEN_CHARS["llama3.2:3b"] == 2.9
    assert config.MODEL_CTX[ATT] == 131072
    assert config.MODEL_CONCURRENCY[ATT] == config.MODEL_CONCURRENCY_START[ATT] == 1


def test_think_is_left_out_for_the_3b_only():
    assert config.attacker_think_for(ATT) is None
    assert config.attacker_think_for("gpt-oss:20b") == config.attacker_think_for("gpt-oss:20b-local") == "medium"
    assert config.attacker_think_for("anything-else") == config.ATTACKER_THINK


class Recorder:
    def __init__(self):
        self.calls = []

    def chat(self, model, messages, options=None, schema=None, think=None, affinity=None):
        self.calls.append({"model": model, "options": options, "think": think, "schema": schema})
        return LLMResult(ok=True, text="{}", json={"attributes": [{"attribute": "x", "category": "c"}]})


def one_step():
    from rerun.corpus import Doc, Person
    p = Person("C-aaaaaa", "C", None, None, [Doc(id="C-aaaaaa-d01", date=None, kind="pdf", text="Testo del documento. " * 20)])
    return p


def test_the_attacker_request_has_the_main_sampling_and_no_think_field():
    c = Recorder()
    infer.infer_person(c, one_step(), ATT, usable_tokens=100000)
    call = c.calls[0]
    assert call["think"] is None
    assert call["options"]["num_ctx"] == 131072 and call["options"]["temperature"] == 1.0
    assert call["options"]["top_p"] == 1.0 and call["options"]["seed"] == config.SEED
    c2 = Recorder()
    infer.infer_person(c2, one_step(), "gpt-oss:20b-local", usable_tokens=100000)
    assert c2.calls[0]["think"] == "medium"                         # the other attackers are unchanged


def test_no_think_in_the_http_body_of_the_3b_and_local_url():
    sent = []

    def transport(url, headers, body, timeout):
        sent.append((url, body))
        return 200, {"message": {"content": '{"attributes": []}'}, "done_reason": "stop"}

    c = LLMClient([], transport=transport)
    c.chat(ATT, [{"role": "user", "content": "x"}], {"num_ctx": 131072}, infer.SCHEMA_NO_EVIDENCE,
           think=config.attacker_think_for(ATT))
    assert "think" not in sent[0][1] and sent[0][0].startswith(config.LOCAL_URL)


# ---- folders and caches ------------------------------------------------------------------------------------------------
def test_cache_suffix_is_the_runs_own_for_the_size_tag_and_unchanged_otherwise():
    assert run_case.cache_suffix("C") == "" and run_case.cache_suffix("C", "gloc") == ""
    assert run_case.cache_suffix("A") == "_caseA" and run_case.cache_suffix("A", "gloc") == "_caseA"
    assert run_case.cache_suffix("C", config.SIZE_TAG_3B) == "_caseC_size3b"


def make_private(tmp_path):
    """A data/private with the folders and files of the existing runs of Case C."""
    base = tmp_path / "private"
    for name in ("stages", "stages_gloc", "stages_ref", "stages_noev", "stages_evidence", "stages_trunc2000"):
        d = base / "caseC" / name
        d.mkdir(parents=True)
        (d / "infer.jsonl").write_text(f"existing {name}", encoding="utf-8")
        (d / "judge.jsonl").write_text(f"votes {name}", encoding="utf-8")
    return base


def snapshot(base):
    return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(base).rglob("*") if p.is_file()}


def test_the_tag_does_not_touch_existing_folders(tmp_path):
    base = make_private(tmp_path)
    before, dirs = snapshot(base), size_c3b.existing_stage_dirs(base)
    d = run_case.stage_dir("C", base=base, tag=config.SIZE_TAG_3B)
    (d / "infer.jsonl").write_text("new", encoding="utf-8")
    assert d.name == "stages_size-llama3.2-3b" and d.name not in dirs
    assert size_c3b.existing_stage_dirs(base) == sorted([*dirs, d.name])
    after = snapshot(base)
    assert {k: v for k, v in after.items() if "size-llama3.2-3b" not in k} == before


def test_check_tag_creates_marks_and_refuses_other_runs(tmp_path):
    base = make_private(tmp_path)
    with pytest.raises(size_c3b.TagError):
        size_c3b.check_tag(config.SIZE_TAG_3B, base=base)               # no folder yet and not the infer mode
    d = size_c3b.check_tag(config.SIZE_TAG_3B, create=True, base=base)
    assert (d / size_c3b.MARKER).exists()
    (d / "infer.jsonl").write_text("x", encoding="utf-8")
    assert size_c3b.check_tag(config.SIZE_TAG_3B, base=base) == d       # resume is allowed
    assert size_c3b.check_tag(config.SIZE_TAG_3B, create=True, base=base) == d
    for bad in ("gloc", "stages", "size-", "size-x/y", "", "size-a b"):
        with pytest.raises(size_c3b.TagError):
            size_c3b.check_tag(bad, create=True, base=base)
    # a folder with files and no marker (made by someone else) is never reused
    other = base / "caseC" / "stages_size-other"
    other.mkdir()
    (other / "infer.jsonl").write_text("not ours", encoding="utf-8")
    with pytest.raises(size_c3b.TagError):
        size_c3b.check_tag("size-other", create=True, base=base)
    # a marker of another attacker is refused too
    (other / size_c3b.MARKER).write_text(json.dumps({"tag": "size-other", "attacker": "gpt-oss:20b-local"}), encoding="utf-8")
    with pytest.raises(size_c3b.TagError):
        size_c3b.check_tag("size-other", base=base)


def test_argv_of_every_mode_uses_the_tag_and_the_attacker_and_no_cloud_model():
    for mode in ("infer", "pass-gemma", "pass-qwen", "pass-third", "judge", "dedup", "metrics"):
        argv = size_c3b.build_argv(mode)
        assert argv[argv.index("--tag") + 1] == "size-llama3.2-3b" and argv[argv.index("--attacker") + 1] == ATT
        assert argv[argv.index("--case") + 1] == "C" and "--infer-from" not in argv
    assert size_c3b.build_argv("infer", workers=2)[-1].endswith(f"{ATT}=2")
    assert size_c3b.build_argv("judge")[size_c3b.build_argv("judge").index("--panel-scheme") + 1] == "sequential"
    assert size_c3b.build_argv("dedup")[size_c3b.build_argv("dedup").index("--dedup-model") + 1] == "gemma4:31b-local"
    assert "--limit" in size_c3b.build_argv("infer", limit=3)
    with pytest.raises(ValueError):
        size_c3b.build_argv("export")


def test_export_copies_only_the_metrics_file(tmp_path):
    base, res = make_private(tmp_path), tmp_path / "results"
    d = size_c3b.check_tag(config.SIZE_TAG_3B, create=True, base=base)
    with pytest.raises(size_c3b.TagError):
        size_c3b.export(base=base, results=res)
    (d / "metrics_limited").mkdir()
    (d / "metrics_limited" / "metrics.json").write_text('{"persons": 95}', encoding="utf-8")
    out = size_c3b.export(base=base, results=res)
    assert out == res / "caseC" / "size_llama3.2-3b_metrics.json" and json.loads(out.read_text()) == {"persons": 95}
    assert not (res / "caseC" / "metrics.json").exists()


# ---- local panel -----------------------------------------------------------------------------------------------------
@pytest.fixture
def panel():
    original_chat = llm.LLMClient.chat
    state = (dict(config.NUM_CTX), dict(config.JUDGE_THINK), config.PANEL, llm.NUM_PREDICT_CAP)
    yield local_panel.setup()
    local_panel.teardown()
    assert llm.LLMClient.chat is original_chat
    assert (dict(config.NUM_CTX), dict(config.JUDGE_THINK), config.PANEL, llm.NUM_PREDICT_CAP) == state


def test_panel_settings_and_guard(panel, tmp_path):
    assert panel == ("gemma4:31b-local", "qwen3.5:9b", "nemotron-3-nano:30b-local")
    assert config.JUDGE_THINK["gemma4:31b-local"] is False and config.JUDGE_THINK["nemotron-3-nano:30b-local"] is False
    assert config.JUDGE_BATCH_SIZE["gemma4:31b-local"] == 1 and llm.NUM_PREDICT_CAP == 8192
    seen = []

    def transport(url, headers, body, timeout):
        seen.append(body)
        return 200, {"message": {"content": "ok"}, "done_reason": "stop"}

    c = LLMClient([], transport=transport, cache_path=tmp_path / "c.jsonl", private_root=tmp_path)
    msg = [{"role": "user", "content": "x"}]
    big = c.chat("gemma4:31b-local", msg, {"num_ctx": 262144})
    assert not big.ok and big.error == "too_long_for_local" and seen == [] and c.stats["too_long_for_local"] == 1
    assert c.chat("gemma4:31b-local", msg, {"num_ctx": 65536}).ok
    third = c.chat("nemotron-3-nano:30b-local", msg, {"num_ctx": 65536, "num_predict": 32768})
    assert third.ok and seen[-1]["options"]["num_predict"] == 8192
    assert c.chat("qwen3.5:9b", msg, {"num_ctx": 262144}).ok            # the qwen judge keeps the full context


# ---- dry run on the real persons of Case C ---------------------------------------------------------------------------------
def test_dry_run_is_skipped_without_the_data(tmp_path):
    assert size_c3b.dry_run(base=tmp_path, raw=tmp_path)["status"] == "skipped"


@pytest.mark.skipif(not (config.PRIVATE / "caseC" / "text").is_dir() or not (config.RAW / "caseC").is_dir(),
                    reason="Case C data not on this machine")
def test_dry_run_on_real_persons_leaves_the_existing_folders_alone():
    res = size_c3b.dry_run(limit=4)
    assert res["status"] == "ok" and res["persons"] == 4 and res["steps"] > 0
    assert res["real_files_unchanged"] and not res["tag_dir_created_in_real_data"]
    assert res["existing_stage_dirs_before"] == res["existing_stage_dirs_after"]
    assert res["tag_dir_name"] not in res["existing_stage_dirs_before"]
    assert res["all_calls_to_local_server"] and res["think_field_sent_to_attacker"] == 0
    assert res["attacker_num_ctx"] == [131072] and res["undecided"] == 0
    assert res["models_called"][ATT] == res["steps_run"] and res["windows_total"] == res["steps_run"]
    assert res["panel_scheme"] == {"full": res["attributes"]}
    assert set(res["models_called"]) == {ATT, "gemma4:31b-local", "qwen3.5:9b", "nemotron-3-nano:30b-local"}


def test_shards_partition_the_persons():
    from rerun import size_c3b
    persons = [f"p{i:02d}" for i in range(95)]
    parts = [size_c3b.shard_persons(persons, k, 4) for k in range(4)]
    assert sorted(x for p in parts for x in p) == persons          # a partition: nothing lost, nothing twice
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1
    assert size_c3b.parse_shard("2/5") == (2, 5)
    for bad in ("5/5", "-1/3", "a/b", "1", "0/0"):
        try:
            size_c3b.parse_shard(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_a_shard_judges_only_the_rows_of_its_persons(tmp_path, monkeypatch):
    """The inference file holds every person: a shard must not look up the persons of the other shards (it raised a
    KeyError on the machine), and the shards together must cover every row once."""
    from types import SimpleNamespace
    ids = [f"C-{i:02d}" for i in range(10)]
    path = tmp_path / "infer.jsonl"
    path.write_text("\n".join(json.dumps({"person": i, "step": 1, "carried": False, "attributes": [{"id": i}]})
                              for i in ids) + "\n", encoding="utf-8")
    persons = [SimpleNamespace(id=i) for i in ids]
    seen = []
    original_rows = run_case.judge_rows_to_run
    for k in range(3):                       # each shard starts from the unpatched functions
        monkeypatch.setattr(run_case, "select_persons", lambda p, limit=None: sorted(p, key=lambda x: x.id))
        monkeypatch.setattr(run_case, "judge_rows_to_run", original_rows)
        keep = size_c3b.apply_shard(k, 3)
        chosen = run_case.select_persons(persons)
        rows = run_case.judge_rows_to_run(path)
        assert {r["person"] for r in rows} == keep == {p.id for p in chosen}
        seen += [r["person"] for r in rows]
    assert sorted(seen) == ids
