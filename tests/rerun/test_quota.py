from rerun import llm


def test_quota_kind_reads_the_server_message():
    assert llm.quota_kind({"error": "you (acct) have reached your monthly usage limit"}) == "monthly"
    assert llm.quota_kind({"error": "You reached the Free usage limit. Pro is $20/month"}) == "window"
    assert llm.quota_kind({"error": "too many requests"}) is None
    assert llm.quota_kind(None) is None


class Transport:
    """Key 0: monthly cap; key 1: free window cap; key 2: answers."""

    def __init__(self, keys):
        self.keys, self.calls = keys, []

    def __call__(self, url, headers, body, timeout):
        k = headers["Authorization"].split(" ", 1)[1]
        self.calls.append(k)
        if k == self.keys[0]:
            return 429, {"error": "you (x) have reached your monthly usage limit"}
        if k == self.keys[1]:
            return 429, {"error": "You reached the Free usage limit."}
        return 200, {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}


def test_monthly_key_is_dropped_window_key_rests_and_the_limit_is_not_shrunk(tmp_path):
    keys = ["k0", "k1", "k2"]
    t = Transport(keys)
    now = [0.0]
    c = llm.LLMClient(keys, transport=t, sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0],
                      private_root=tmp_path, model_concurrency={"gpt-oss:20b": 4})
    for _ in range(5):
        assert c.chat("gpt-oss:20b", [{"role": "user", "content": "x"}], schema=None).ok
    assert t.calls.count("k0") == 1                     # monthly: never tried again
    assert t.calls.count("k1") == 1                     # window: resting for QUOTA_COOLDOWN
    assert c.stats["quota_monthly"] == 1 and c.stats["quota_window"] == 1
    assert c.model_limits()["gpt-oss:20b"] == 4         # quota answers do not halve the calls in flight


class PlanTransport:
    """Ollama Cloud after 2026-10-23: a paid model answers 402 on free keys; the paid key answers everything."""

    def __init__(self, paid_key):
        self.paid_key, self.calls = paid_key, []

    def __call__(self, url, headers, body, timeout):
        k = headers["Authorization"].split(" ", 1)[1]
        self.calls.append((body["model"], k))
        if body["model"] == "minimax-m3.1-flash" and k != self.paid_key:
            return 402, {"error": "This model is not in the Free plan."}
        return 200, {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}


def test_paid_models_use_only_the_paid_keys_and_free_models_spare_them(tmp_path):
    keys = ["k0", "k1", "k2"]
    t = PlanTransport("k0")
    c = llm.LLMClient(keys, transport=t, sleep=lambda s: None, private_root=tmp_path,
                      key_types=["PRO", "free", "free"], paid_models=("minimax-m3.1-flash",))
    for i in range(4):
        assert c.chat("minimax-m3.1-flash", [{"role": "user", "content": f"x{i}"}], schema=None).ok
        assert c.chat("gemma4:31b", [{"role": "user", "content": f"y{i}"}], schema=None).ok
    assert {k for m, k in t.calls if m == "minimax-m3.1-flash"} == {"k0"}
    assert {k for m, k in t.calls if m == "gemma4:31b"} == {"k1", "k2"}


def test_a_paid_model_without_a_paid_key_fails_at_once(tmp_path):
    t = PlanTransport("none")
    c = llm.LLMClient(["k1"], transport=t, sleep=lambda s: None, private_root=tmp_path, key_types=["free"],
                      paid_models=("minimax-m3.1-flash",))
    r = c.chat("minimax-m3.1-flash", [{"role": "user", "content": "x"}], schema=None)
    assert not r.ok and r.error == "no_valid_key" and t.calls == []


def test_load_key_types_reads_the_type_field(tmp_path):
    p = tmp_path / "k.json"
    p.write_text('{"CLOUD_API_KEYS": [{"key": "a", "type": "PRO"}, {"key": "b", "type": "free"}, {"key": "c"}]}',
                 encoding="utf-8")
    assert llm.load_key_types(p) == ["pro", "free", "free"]
