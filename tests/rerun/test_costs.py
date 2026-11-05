"""Tests of rerun.costs: per-profile cost of the inference, stage times, per-model usage, and no personal content."""
import json

import pytest

from rerun import costs


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_inference_cost_is_per_profile_and_carried_steps_are_free(tmp_path):
    f = tmp_path / "infer.jsonl"
    write_jsonl(f, [
        {"person": "p1", "step": 25, "carried": False, "tokens_in": 100, "tokens_out": 10, "seconds": 4.0},
        {"person": "p1", "step": 50, "carried": False, "tokens_in": 200, "tokens_out": 30, "seconds": 6.0},
        {"person": "p1", "step": 75, "carried": True, "tokens_in": 999, "tokens_out": 999, "seconds": 999.0},
        {"person": "p2", "step": 25, "carried": False, "tokens_in": 60, "tokens_out": 20, "seconds": 2.0},
    ])
    c = costs.infer_cost(f)
    assert c["persons"] == 2 and c["steps_with_calls"] == 3
    assert c["tokens_in"] == 360 and c["tokens_out"] == 60 and c["call_seconds"] == 12.0
    assert c["per_profile"]["tokens_in"] == 180.0 and c["per_profile"]["call_seconds_mean"] == 6.0


def test_missing_inference_file_gives_zero_persons(tmp_path):
    assert costs.infer_cost(tmp_path / "nothing.jsonl") == {"persons": 0}


def test_stage_times_read_the_run_summaries_and_sum_the_control_parts(tmp_path):
    (tmp_path / "run_summary_infer.json").write_text(json.dumps({"infer": {"seconds_wall": 12.5}}), encoding="utf-8")
    (tmp_path / "run_summary_judge.json").write_text(json.dumps({"judge": {"seconds_wall": 30.0}}), encoding="utf-8")
    (tmp_path / "run_summary_control.json").write_text(json.dumps(
        {"control_infer": {"seconds_wall": 10.0}, "control_judge": {"seconds_wall": 5.5}}), encoding="utf-8")
    t = costs.stage_times(tmp_path)
    assert t["infer"]["seconds_wall"] == 12.5 and t["judge"]["seconds_wall"] == 30.0
    assert t["control"]["seconds_wall"] == 15.5 and "dedup" not in t


def test_usage_counts_only_calls_that_were_made_and_tells_local_from_cloud(tmp_path):
    f = tmp_path / "usage.jsonl"
    write_jsonl(f, [
        {"model": "gpt-oss:20b", "ok": True, "tokens_in": 10, "tokens_out": 5, "seconds": 1.0, "cached": False},
        {"model": "gpt-oss:20b", "ok": False, "tokens_in": 10, "tokens_out": 0, "seconds": 2.0, "cached": False},
        {"model": "gpt-oss:20b", "ok": True, "tokens_in": 99, "tokens_out": 99, "seconds": 9.0, "cached": True},
        {"model": "gemma4:31b-local", "ok": True, "tokens_in": 7, "tokens_out": 3, "seconds": 0.5, "cached": False},
    ])
    u = costs.usage_by_model([f])
    assert u["gpt-oss:20b"] == {"calls": 2, "ok": 1, "tokens_in": 20, "tokens_out": 5, "seconds": 3.0, "place": "cloud"}
    assert u["gemma4:31b-local"]["place"] == "local" and u["gemma4:31b-local"]["calls"] == 1


def test_collect_holds_aggregates_only(tmp_path):
    priv = tmp_path / "private"
    write_jsonl(priv / "caseX" / "stages" / "infer.jsonl",
                [{"person": "secret-id-123", "step": 100, "carried": False, "tokens_in": 1, "tokens_out": 1,
                  "seconds": 1.0, "attributes": [{"text": "personal text"}]}])
    write_jsonl(priv / "llm" / "usage.jsonl", [{"model": "gpt-oss:20b", "ok": True, "tokens_in": 1, "tokens_out": 1,
                                                 "seconds": 1.0, "cached": False}])
    res = costs.collect(priv)
    dumped = json.dumps(res)
    assert "secret-id-123" not in dumped and "personal text" not in dumped
    assert res["cases"]["X"]["inference"]["persons"] == 1 and "gpt-oss:20b" in res["models"]


def test_extra_stage_directories_are_added(tmp_path):
    priv = tmp_path / "private"
    (priv / "llm").mkdir(parents=True)
    extra = tmp_path / "from_other_pc" / "caseA" / "stages_gloc"
    extra.mkdir(parents=True)
    (extra / "run_summary_judge.json").write_text(json.dumps({"judge": {"seconds_wall": 7.0}}), encoding="utf-8")
    res = costs.collect(priv, [extra])
    assert res["cases"]["A"]["stages"]["judge"]["stages_gloc@extra"]["seconds_wall"] == 7.0
