"""Context budget per model: how many steps and persons exceed the usable context.

``models`` maps a model name to ``dict(ctx, chars_per_token, output_reserve, prompt_overhead)``
or, for a judge, ``dict(ctx, chars_per_token, batch_attrs)`` (reserves from the batch length).
The usable input of a model is ``ctx - output_reserve - prompt_overhead`` tokens; a step whose
estimated tokens exceed it must be read in windows (``steps.windows``). Identical steps are not
run, so they are not counted. The report holds counts only, never person identifiers.
"""
from __future__ import annotations

from . import steps as st

JUDGE_BATCH_MAX = 40          # attributes per judge call
JUDGE_OUT_TOKENS_PER_ATTR = 60
JUDGE_OUT_MIN = 2048
JUDGE_REASONING_ALLOWANCE = 30368  # reasoning judges (nemotron, minimax) spend 4,000-16,000+ tokens thinking; gemma4 none;
# 2,400 (40 verdicts) + 30,368 = 32,768 = the output limit of the judge calls
JUDGE_NUM_PREDICT = 16384
JUDGE_INSTRUCTION_TOKENS = 800
JUDGE_ATTR_TOKENS = 80        # attribute + evidence in the prompt (to be measured in T5)


def judge_batch_reserves(n_attrs: int) -> tuple[int, int]:
    """(output reserve, prompt overhead) in tokens for a judge batch of ``n_attrs`` attributes."""
    n = max(1, min(n_attrs, JUDGE_BATCH_MAX))
    out = max(JUDGE_OUT_MIN, JUDGE_OUT_TOKENS_PER_ATTR * n) + JUDGE_REASONING_ALLOWANCE
    return out, JUDGE_INSTRUCTION_TOKENS + JUDGE_ATTR_TOKENS * n


def exceedance_report(persons, models: dict) -> dict:
    out = {}
    for name, m in models.items():
        if "batch_attrs" in m:  # judges: reserves from the real batch length
            out_res, overhead = judge_batch_reserves(m["batch_attrs"])
        else:
            out_res, overhead = m["output_reserve"], m["prompt_overhead"]
        usable = st.usable_input_tokens(m["ctx"], output_reserve=out_res, prompt_overhead=overhead)
        cpt = m["chars_per_token"]
        r = dict(usable_tokens=usable, persons=len(persons), persons_over=0, steps_run=0, steps_over=0,
                 max_step_tokens=0, windows_total=0, max_windows=0)
        for p in persons:
            over_any = False
            for s in st.build_steps(p):
                if s.identical_to_prev:
                    continue
                r["steps_run"] += 1
                tokens = st.estimate_tokens(s.text, cpt)
                r["max_step_tokens"] = max(r["max_step_tokens"], tokens)
                if tokens > usable:
                    r["steps_over"] += 1
                    over_any = True
                    n_win = len(st.window_spans(s.text, usable, cpt))
                else:
                    n_win = 1
                r["windows_total"] += n_win
                r["max_windows"] = max(r["max_windows"], n_win)
            r["persons_over"] += over_any
        out[name] = r
    return out
