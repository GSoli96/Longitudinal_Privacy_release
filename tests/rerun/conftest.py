"""The batching mechanics are tested with only nemotron-3-ultra judging one attribute per call; the real setting
(all three judges one at a time) is checked in test_config_core_judging.py."""
import pytest

from rerun import config


@pytest.fixture(autouse=True)
def _batched_judges_except_ultra(monkeypatch, request):
    if "real_judge_config" in request.keywords:
        return
    monkeypatch.setattr(config, "JUDGE_BATCH_SIZE", {"nemotron-3-ultra": 1})
