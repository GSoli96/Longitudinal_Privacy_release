"""Core setting of the judges (decision of 2026-10-20): every judge sees one attribute per call."""
import pytest

from rerun import config, judge

pytestmark = pytest.mark.real_judge_config


def test_every_core_judge_judges_one_attribute_per_call():
    assert all(config.JUDGE_BATCH_SIZE.get(m) == 1 for m in config.JUDGES)


def test_every_core_judge_has_a_concurrency_limit():
    assert set(config.MODEL_CONCURRENCY) >= set(config.JUDGES)


def test_judge_batch_makes_one_call_per_attribute_for_every_judge():
    class M:
        def __init__(self):
            self.n = 0

        def chat(self, model, messages, options=None, schema=None, think=None):
            from rerun.llm import LLMResult
            import json
            self.n += 1
            r = {"verdicts": [{"id": 1, "plausible": "yes", "coherent": "yes", "realistic": "yes", "explicit": "no"}]}
            return LLMResult(text=json.dumps(r), json=r, ok=True)

    attrs = [judge.JudgeAttr(id=f"a{i}", text=f"attributo {i}", evidence="", window=0) for i in range(4)]
    for m in config.JUDGES:
        c = M()
        judge.judge_batch(c, m, ["T"], attrs)
        assert c.n == 4


def test_local_attacker_alias_has_the_settings_of_the_cloud_attacker():
    """gpt-oss:20b-local (Cases A, D, H) must cut steps and judge windows exactly like gpt-oss:20b."""
    from rerun import config
    cloud, local = "gpt-oss:20b", "gpt-oss:20b-local"
    assert config.is_local(local) and not config.is_local(cloud)
    assert config.num_ctx_for(local) == config.num_ctx_for(cloud)
    assert config.TOKEN_CHARS[local] == config.TOKEN_CHARS[cloud]
    assert config.MODEL_CTX[local] == config.MODEL_CTX[cloud]
