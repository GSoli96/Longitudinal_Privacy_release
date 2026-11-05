"""Tests of rerun.llm: cache, checkpoint/resume, key rotation, retries, no key leakage."""
import json

import pytest

from rerun import config
from rerun.llm import LLMClient, cache_key

KEYS = ["sk-test-AAAA1111", "sk-test-BBBB2222", "sk-test-CCCC3333"]
MSG = [{"role": "user", "content": "hello"}]


def ok_payload(text="answer", tin=7, tout=3, done="stop"):
    return {"message": {"content": text}, "prompt_eval_count": tin, "eval_count": tout, "done_reason": done}


class FakeTransport:
    """Scripted transport: pops (status, payload) per call, records the requests."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": body})
        if not self.script:
            return 200, ok_payload()
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    """sleep() advances the time, so cool-downs and rate limits are deterministic."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s

    def __call__(self):
        return self.now


def make(tmp_path, script=(), keys=KEYS, **kw):
    t = FakeTransport(script)
    clock = FakeClock()
    c = LLMClient(keys=keys, transport=t, cache_path=tmp_path / "cache.jsonl",
                  usage_path=tmp_path / "usage.jsonl", sleep=clock.sleep, clock=clock,
                  backoff_base=2.0, private_root=tmp_path, **kw)
    return c, t, clock.sleeps


def test_cache_key_depends_on_every_input():
    base = cache_key("m", MSG, {"a": 1}, None)
    assert base == cache_key("m", MSG, {"a": 1}, None)
    assert base != cache_key("m2", MSG, {"a": 1}, None)
    assert base != cache_key("m", MSG + MSG, {"a": 1}, None)
    assert base != cache_key("m", MSG, {"a": 2}, None)
    assert base != cache_key("m", MSG, {"a": 1}, {"type": "object"})
    # key order of dictionaries must not matter
    assert cache_key("m", MSG, {"a": 1, "b": 2}, None) == cache_key("m", MSG, {"b": 2, "a": 1}, None)


def test_second_call_is_cached_and_identical(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload("same text"))])
    r1 = c.chat("gpt-oss:20b", MSG, {"num_ctx": 4096})
    r2 = c.chat("gpt-oss:20b", MSG, {"num_ctx": 4096})
    assert r1.ok and not r1.cached
    assert r2.ok and r2.cached
    assert r2.text == r1.text == "same text"
    assert len(t.calls) == 1


def test_defaults_and_options_reach_the_request(tmp_path):
    c, t, _ = make(tmp_path)
    c.chat("gpt-oss:20b", MSG, {"num_ctx": 8192})
    body = t.calls[0]["body"]
    assert body["model"] == "gpt-oss:20b" and body["stream"] is False
    assert body["options"]["temperature"] == 0
    assert body["options"]["seed"] == config.SEED
    assert body["options"]["num_ctx"] == 8192


def test_routing_local_vs_cloud(tmp_path):
    c, t, _ = make(tmp_path)
    c.chat("bge-m3", MSG)            # the only local model left (cloud-only rule, 2026-10-23)
    c.chat("gpt-oss:20b", MSG)
    local, cloud = t.calls
    assert not config.is_local("llama3.2:3b")   # no generative model runs locally any more
    assert local["url"].startswith(config.LOCAL_URL) and "Authorization" not in local["headers"]
    assert cloud["url"].startswith(config.CLOUD_URL)
    assert cloud["headers"]["Authorization"].startswith("Bearer ")


def test_429_backs_off_and_moves_to_next_key(tmp_path):
    c, t, sleeps = make(tmp_path, [(429, None), (200, ok_payload())])
    r = c.chat("gpt-oss:20b", MSG)
    assert r.ok and r.key_index == 1
    assert len(t.calls) == 2
    assert t.calls[0]["headers"]["Authorization"] != t.calls[1]["headers"]["Authorization"]
    # key 0 is cooling: the next fresh call must not use it
    r2 = c.chat("gpt-oss:20b", [{"role": "user", "content": "other"}])
    assert r2.key_index != 0
    assert c.stats["http_429"] == 1


def test_all_keys_cooling_sleeps_until_one_is_free(tmp_path):
    script = [(429, None)] * 3 + [(200, ok_payload())]
    c, t, sleeps = make(tmp_path, script)
    r = c.chat("gpt-oss:20b", MSG)
    assert r.ok
    assert sleeps and all(s > 0 for s in sleeps)


def test_failures_set_error_and_stop_after_max_attempts(tmp_path):
    c, t, _ = make(tmp_path, [(500, None)] * 10, max_attempts=4)
    r = c.chat("gpt-oss:20b", MSG)
    assert not r.ok and r.error and r.text == ""
    assert len(t.calls) == 4


def test_network_exception_is_a_retryable_failure(tmp_path):
    c, t, _ = make(tmp_path, [ConnectionError("boom"), (200, ok_payload())])
    r = c.chat("gpt-oss:20b", MSG)
    assert r.ok and len(t.calls) == 2
    assert c.stats["network_error"] == 1


def test_401_disables_key_permanently(tmp_path):
    c, t, _ = make(tmp_path, [(401, None), (200, ok_payload())])
    r = c.chat("gpt-oss:20b", MSG)
    assert r.ok and r.key_index == 1
    for i in range(4):
        rr = c.chat("gpt-oss:20b", [{"role": "user", "content": f"q{i}"}])
        assert rr.key_index != 0
    assert c.stats["http_401"] == 1


def test_no_valid_key_left_fails_cleanly(tmp_path):
    c, t, _ = make(tmp_path, [(401, None)], keys=["only-key"], max_attempts=3)
    r = c.chat("gpt-oss:20b", MSG)
    assert not r.ok and "no_valid_key" in r.error


def test_failed_calls_are_not_cached(tmp_path):
    c, t, _ = make(tmp_path, [(500, None)] * 2, max_attempts=2)
    assert not c.chat("gpt-oss:20b", MSG).ok
    r = c.chat("gpt-oss:20b", MSG)  # script exhausted -> default 200
    assert r.ok and not r.cached


def test_checkpoint_resume_does_not_recall_completed_calls(tmp_path):
    c1, t1, _ = make(tmp_path)
    prompts = [[{"role": "user", "content": f"p{i}"}] for i in range(5)]
    first = [c1.chat("gpt-oss:20b", p).text for p in prompts]
    c2, t2, _ = make(tmp_path)  # new process: same cache file
    second = [c2.chat("gpt-oss:20b", p) for p in prompts]
    assert len(t2.calls) == 0
    assert all(r.cached for r in second)
    assert [r.text for r in second] == first


def test_resume_after_interruption_calls_only_missing(tmp_path):
    c1, t1, _ = make(tmp_path)
    prompts = [[{"role": "user", "content": f"p{i}"}] for i in range(6)]
    for p in prompts[:3]:  # "interrupted" after three calls
        c1.chat("gpt-oss:20b", p)
    c2, t2, _ = make(tmp_path)
    for p in prompts:
        c2.chat("gpt-oss:20b", p)
    assert len(t2.calls) == 3


def test_truncated_last_cache_line_is_ignored(tmp_path):
    c1, _, _ = make(tmp_path)
    c1.chat("gpt-oss:20b", MSG)
    with open(tmp_path / "cache.jsonl", "a", encoding="utf-8") as f:
        f.write('{"key": "abc", "text": "tru')  # killed mid-write
    c2, t2, _ = make(tmp_path)
    assert c2.chat("gpt-oss:20b", MSG).cached
    assert len(t2.calls) == 0


def test_no_key_in_any_file_or_error(tmp_path):
    script = [(429, None), (500, None), ConnectionError("failed with " + KEYS[0]), (200, ok_payload())]
    c, t, _ = make(tmp_path, script)
    r = c.chat("gpt-oss:20b", MSG)
    assert r.ok
    c.chat("gpt-oss:20b", MSG)
    for name in ("cache.jsonl", "usage.jsonl"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert not any(k in text for k in KEYS), name
    # an error message that contains a key must be redacted
    other = tmp_path / "other"
    other.mkdir()
    c2, _, _ = make(other, [ConnectionError("failed with " + KEYS[1])] * 3, max_attempts=3)
    r2 = c2.chat("gpt-oss:20b", [{"role": "user", "content": "z"}])
    assert not r2.ok and not any(k in r2.error for k in KEYS)


def test_usage_log_has_key_index_not_key(tmp_path):
    c, t, _ = make(tmp_path)
    c.chat("gpt-oss:20b", MSG)
    rec = json.loads((tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["key_index"] == 0
    assert rec["model"] == "gpt-oss:20b" and rec["date"] and rec["ok"] is True
    assert "key" not in rec or not any(k in str(rec) for k in KEYS)


def test_schema_valid_json_is_parsed(tmp_path):
    payload = ok_payload(json.dumps({"attributes": []}))
    c, t, _ = make(tmp_path, [(200, payload)])
    r = c.chat("gpt-oss:20b", MSG, schema={"type": "object"})
    assert r.ok and r.json == {"attributes": []}
    assert t.calls[0]["body"]["format"] == {"type": "object"}


@pytest.mark.parametrize("bad", ['Sure! {"a": 1}', '{"a": 1', "", "[]  trailing",
                                 'Ecco:\n```json\n{"a": 1}\n```',          # prose before the fence
                                 '```json\n{"a": 1}\n```\nSpero sia utile.',  # prose after the fence
                                 '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```',  # two blocks
                                 '```json\n{"a": 1\n```'])                  # truncated JSON inside the fence
def test_schema_invalid_json_is_retried_then_failure(tmp_path, bad):
    c, t, _ = make(tmp_path, [(200, ok_payload(bad))] * 5, invalid_json_retries=2)
    r = c.chat("gpt-oss:20b", MSG, schema={"type": "object"})
    assert not r.ok and r.error == "invalid_json" and r.json is None
    assert len(t.calls) == 3  # first try + 2 retries
    assert c.stats["invalid_json"] == 1


def test_schema_json_must_be_an_object_or_array(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload("42"))] * 5, invalid_json_retries=0)
    r = c.chat("gpt-oss:20b", MSG, schema={"type": "object"})
    assert not r.ok and r.error == "invalid_json"


def test_rate_limit_spaces_calls_on_same_key(tmp_path):
    c, t, sleeps = make(tmp_path, keys=["k0"], min_interval=1.0)
    c.chat("gpt-oss:20b", [{"role": "user", "content": "a"}])
    assert sleeps == []
    c.chat("gpt-oss:20b", [{"role": "user", "content": "b"}])
    assert sleeps == [pytest.approx(1.0)]


def test_stats_error_rates(tmp_path):
    c, t, _ = make(tmp_path, [(429, None), (200, ok_payload())] + [(200, ok_payload())] * 3)
    for i in range(4):
        c.chat("gpt-oss:20b", [{"role": "user", "content": f"s{i}"}])
    assert c.stats["calls"] >= 4
    assert 0 < c.key_error_rate() < 1


def test_concurrency_per_key_is_limited(tmp_path):
    import threading
    import time as _time

    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def slow(url, headers, body, timeout):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        _time.sleep(0.03)
        with lock:
            state["now"] -= 1
        return 200, ok_payload()

    c = LLMClient(keys=["only"], transport=slow, cache_path=tmp_path / "c.jsonl",
                  max_concurrency=2, private_root=tmp_path)
    threads = [threading.Thread(target=c.chat, args=("gpt-oss:20b", [{"role": "user", "content": str(i)}]))
               for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["peak"] <= 2
    assert c.stats["calls"] == 8


def test_load_keys_returns_only_key_strings(tmp_path):
    from rerun.llm import load_keys
    p = tmp_path / "keys.json"
    p.write_text(json.dumps({"CLOUD_API_KEYS": [
        {"key": "k-one", "name": "n1", "type": "free"}, {"key": "k-two", "name": "n2", "type": "PRO"}]}),
        encoding="utf-8")
    assert load_keys(p) == ["k-one", "k-two"]


def test_repr_does_not_expose_keys(tmp_path):
    c, _, _ = make(tmp_path)
    assert not any(k in repr(c) for k in KEYS)


def test_invalid_json_retry_is_a_different_request_and_counts_retries(tmp_path):
    good = json.dumps({"a": 1})
    c, t, _ = make(tmp_path, [(200, ok_payload("Sure! {bad")), (200, ok_payload(good))])
    r = c.chat("gpt-oss:20b", MSG, schema={"type": "object"})
    assert r.ok and r.json == {"a": 1} and r.retries == 1
    first, second = t.calls[0]["body"], t.calls[1]["body"]
    assert first != second
    assert second["messages"][:-2] == MSG
    assert second["messages"][-2] == {"role": "assistant", "content": "Sure! {bad"}
    assert second["messages"][-1]["role"] == "user"
    assert "valid JSON" in second["messages"][-1]["content"]
    assert second["options"]["seed"] == config.SEED + 1
    assert first["options"]["seed"] == config.SEED


def test_retried_answer_is_cached_under_the_original_request(tmp_path):
    good = json.dumps({"a": 1})
    c, t, _ = make(tmp_path, [(200, ok_payload("oops")), (200, ok_payload(good))])
    c.chat("gpt-oss:20b", MSG, schema={"type": "object"})
    r2 = c.chat("gpt-oss:20b", MSG, schema={"type": "object"})
    assert r2.cached and r2.json == {"a": 1} and len(t.calls) == 2


def test_rate_limits_do_not_consume_max_attempts(tmp_path):
    script = [(429, None)] * 8 + [(200, ok_payload())]
    c, t, _ = make(tmp_path, script, max_attempts=2)
    r = c.chat("gpt-oss:20b", MSG)
    assert r.ok and len(t.calls) == 9


def test_rate_limit_wait_is_capped_then_rate_limited(tmp_path):
    c, t, clock = make(tmp_path, [(429, None)] * 200, rate_limit_max_wait=10.0)
    r = c.chat("gpt-oss:20b", MSG)
    assert not r.ok and r.error == "rate_limited"
    assert sum(clock) <= 10.0 + 1e-9


def test_401_does_not_consume_max_attempts(tmp_path):
    c, t, _ = make(tmp_path, [(401, None), (401, None), (200, ok_payload())], max_attempts=1)
    r = c.chat("gpt-oss:20b", MSG)
    assert r.ok and r.key_index == 2


def test_stats_increments_are_thread_safe(tmp_path):
    import threading
    c, _, _ = make(tmp_path)

    def bump():
        for _ in range(2000):
            c._inc("calls")

    ts = [threading.Thread(target=bump) for _ in range(8)]
    [x.start() for x in ts]
    [x.join() for x in ts]
    assert c.stats["calls"] == 16000


def test_cache_and_usage_must_live_under_private_root(tmp_path):
    inside = tmp_path / "private"
    inside.mkdir()
    LLMClient(keys=["k"], cache_path=inside / "c.jsonl", usage_path=inside / "u.jsonl",
              private_root=inside)
    with pytest.raises(ValueError):
        LLMClient(keys=["k"], cache_path=tmp_path / "c.jsonl", private_root=inside)
    with pytest.raises(ValueError):
        LLMClient(keys=["k"], usage_path=tmp_path / "elsewhere" / "u.jsonl", private_root=inside)


@pytest.mark.parametrize("text", ['```json\n{"a": 1}\n```', '```\n{"a": 1}\n```', '  ```json\n{"a": 1}\n```  \n',
                                  '```JSON {"a": 1} ```'])
def test_a_single_fence_around_valid_json_is_accepted_and_counted(tmp_path, text):
    c, t, _ = make(tmp_path, [(200, ok_payload(text))])
    r = c.chat("gemma4:31b", MSG, schema={"type": "object"})
    assert r.ok and r.json == {"a": 1} and r.fenced is True and r.retries == 0
    assert c.stats["fenced"] == 1 and c.stats["fenced:gemma4:31b"] == 1
    assert len(t.calls) == 1  # no retry: the answer is valid


def test_plain_json_is_not_counted_as_fenced(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload('{"a": 1}'))])
    r = c.chat("gpt-oss:20b", MSG, schema={"type": "object"})
    assert r.ok and r.fenced is False and c.stats["fenced"] == 0


def test_fenced_flag_survives_the_cache(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload('```json\n{"a": 1}\n```'))])
    c.chat("gemma4:31b", MSG, schema={"type": "object"})
    c2, t2, _ = make(tmp_path)
    r = c2.chat("gemma4:31b", MSG, schema={"type": "object"})
    assert r.cached and r.fenced is True and r.json == {"a": 1} and len(t2.calls) == 0


def test_fence_without_schema_is_left_alone(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload('```json\n{"a": 1}\n```'))])
    r = c.chat("gemma4:31b", MSG)
    assert r.ok and r.text.startswith("```") and r.json is None and r.fenced is False


# ---- truncation by the output limit (done_reason == "length") ---------------------------------------
def test_length_truncation_is_retried_with_doubled_num_predict(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload("par", done="length")), (200, ok_payload("complete answer"))])
    r = c.chat("gpt-oss:20b", MSG, {"num_predict": 4096})
    assert r.ok and r.text == "complete answer" and r.truncations == 1
    assert t.calls[0]["body"]["options"]["num_predict"] == 4096
    assert t.calls[1]["body"]["options"]["num_predict"] == 8192
    assert c.stats["truncated"] == 1 and c.stats["truncated:gpt-oss:20b"] == 1
    assert c.stats["invalid_json"] == 0


def test_truncated_json_is_not_counted_as_invalid_json(tmp_path):
    good = json.dumps({"a": 1})
    c, t, _ = make(tmp_path, [(200, ok_payload('{"a": [1, 2', done="length")), (200, ok_payload(good))])
    r = c.chat("gemma4:31b", MSG, {"num_predict": 8192}, schema={"type": "object"})
    assert r.ok and r.json == {"a": 1} and r.truncations == 1 and r.retries == 0
    assert c.stats["invalid_json"] == 0 and c.stats["truncated:gemma4:31b"] == 1


def test_doubling_stops_at_32768_then_fails_as_truncated(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload("x", done="length"))] * 10)
    r = c.chat("gpt-oss:20b", MSG, {"num_predict": 4096})
    assert not r.ok and r.error == "truncated"
    sent = [call["body"]["options"]["num_predict"] for call in t.calls]
    assert sent == [4096, 8192, 16384, 32768]
    assert c.stats["truncated"] == 4 and c.stats["failed"] == 1


def test_length_at_the_cap_from_the_start_fails_without_retry(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload("x", done="length"))] * 3)
    r = c.chat("gpt-oss:20b", MSG, {"num_predict": 32768})
    assert not r.ok and r.error == "truncated" and len(t.calls) == 1


def test_no_num_predict_given_starts_doubling_from_16384(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload("x", done="length")), (200, ok_payload("ok"))])
    r = c.chat("gpt-oss:20b", MSG)
    assert r.ok and t.calls[1]["body"]["options"]["num_predict"] == 16384


def test_truncation_retry_is_cached_under_the_original_request(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload("x", done="length")), (200, ok_payload("done"))])
    c.chat("gpt-oss:20b", MSG, {"num_predict": 4096})
    r2 = c.chat("gpt-oss:20b", MSG, {"num_predict": 4096})
    assert r2.cached and r2.text == "done" and r2.truncations == 1 and len(t.calls) == 2


def test_truncation_does_not_consume_max_attempts_or_json_retries(tmp_path):
    good = json.dumps({"a": 1})
    script = [(200, ok_payload("{", done="length")), (200, ok_payload("{", done="length")), (200, ok_payload(good))]
    c, t, _ = make(tmp_path, script, max_attempts=1, invalid_json_retries=0)
    r = c.chat("gpt-oss:20b", MSG, {"num_predict": 4096}, schema={"type": "object"})
    assert r.ok and r.truncations == 2


def test_normal_stop_is_not_a_truncation(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload("fine"))])
    r = c.chat("gpt-oss:20b", MSG, {"num_predict": 4096})
    assert r.ok and r.truncations == 0 and c.stats["truncated"] == 0


# ---- think (reasoning effort) --------------------------------------------------------------------------------
def test_think_is_sent_as_a_top_level_field_and_not_when_absent(tmp_path):
    c, t, _ = make(tmp_path, [(200, ok_payload("a")), (200, ok_payload("b"))])
    c.chat("gpt-oss:20b", MSG, {"temperature": 1.0, "top_p": 1.0}, think="medium")
    c.chat("gpt-oss:20b", [{"role": "user", "content": "other"}])
    assert t.calls[0]["body"]["think"] == "medium" and "think" not in t.calls[1]["body"]
    assert t.calls[0]["body"]["options"]["temperature"] == 1.0 and t.calls[0]["body"]["options"]["top_p"] == 1.0


def test_think_changes_the_cache_key_but_absent_think_keeps_the_old_key():
    base = cache_key("m", MSG, {"a": 1}, None)
    assert base == cache_key("m", MSG, {"a": 1}, None, None)
    assert cache_key("m", MSG, {"a": 1}, None, "low") != cache_key("m", MSG, {"a": 1}, None, "medium")
    assert cache_key("m", MSG, {"a": 1}, None, "low") != base


def test_same_request_with_different_think_is_not_a_cache_hit(tmp_path):
    c, t, _ = make(tmp_path)
    c.chat("gpt-oss:20b", MSG, think="low")
    r = c.chat("gpt-oss:20b", MSG, think="high")
    assert not r.cached and len(t.calls) == 2
    assert c.chat("gpt-oss:20b", MSG, think="low").cached


# ---- rate-limit events log -------------------------------------------------------------------------------------------
def make_ev(tmp_path, script):
    t = FakeTransport(script)
    clock = FakeClock()
    c = LLMClient(keys=KEYS, transport=t, cache_path=tmp_path / "cache.jsonl", usage_path=tmp_path / "usage.jsonl",
                  events_path=tmp_path / "events.jsonl", sleep=clock.sleep, clock=clock, backoff_base=2.0,
                  private_root=tmp_path)
    return c, t


def test_429_401_and_5xx_are_logged_as_events_with_the_key_index_only(tmp_path):
    c, t = make_ev(tmp_path, [(429, None), (401, None), (500, None), (200, ok_payload())])
    c.chat("gpt-oss:20b", MSG)
    lines = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["kind"] for e in lines] == ["http_429", "http_401", "http_5xx"]
    assert all(e["model"] == "gpt-oss:20b" and isinstance(e["key_index"], int) and e["date"] for e in lines)
    text = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert not any(k in text for k in KEYS)


def test_no_events_file_when_nothing_goes_wrong(tmp_path):
    c, t = make_ev(tmp_path, [(200, ok_payload())])
    c.chat("gpt-oss:20b", MSG)
    assert not (tmp_path / "events.jsonl").exists()


def test_events_path_must_be_under_the_private_root(tmp_path):
    inside = tmp_path / "p"
    inside.mkdir()
    with pytest.raises(ValueError):
        LLMClient(keys=["k"], events_path=tmp_path / "events.jsonl", private_root=inside)


def test_model_concurrency_limits_the_calls_in_flight_of_that_model_only():
    import threading, time as _t
    from rerun.llm import LLMClient
    live = {"now": 0, "max": 0}
    lock = threading.Lock()

    def transport(url, headers, body, timeout):
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        _t.sleep(0.05)
        with lock:
            live["now"] -= 1
        return 200, {"message": {"content": "{}"}, "prompt_eval_count": 1, "eval_count": 1, "done_reason": "stop"}

    c = LLMClient(keys=[f"k{i}" for i in range(8)], transport=transport, sleep=lambda s: None, max_concurrency=4,
                  model_concurrency={"limited-model": 2})
    ts = [threading.Thread(target=lambda i=i: c.chat("limited-model", [{"role": "user", "content": str(i)}])) for i in range(10)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert live["max"] <= 2
    live["max"] = 0
    ts = [threading.Thread(target=lambda i=i: c.chat("free-model", [{"role": "user", "content": str(i)}])) for i in range(10)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert live["max"] > 2
