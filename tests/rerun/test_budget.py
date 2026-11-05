"""Tests of rerun.budget: which steps/profiles exceed a model's usable context."""
from rerun import budget, steps
from rerun.corpus import Doc, Person


def person(pid, sizes):
    docs = [Doc(id=f"{pid}-d{i}", date=None, kind="pdf", text="x" * n) for i, n in enumerate(sizes, 1)]
    return Person(pid, "X", None, None, docs, "ok" if any(sizes) else "no_text")


MODELS = {"small": dict(ctx=10000, chars_per_token=4.0, output_reserve=1000, prompt_overhead=500),
          "big": dict(ctx=1000000, chars_per_token=4.0, output_reserve=1000, prompt_overhead=500)}


def test_usable_tokens_and_exceedance_counts():
    # usable(small) = 10000 - 1000 - 500 = 8500 tokens = 34000 chars
    short = person("a", [4000, 4000, 4000, 4000])        # 16 k chars + markers < 34 k
    long_ = person("b", [20000, 20000, 20000, 20000])    # 25 % = 20 k ok, 50 % = 40 k over, 100 % = 80 k over
    rep = budget.exceedance_report([short, long_], MODELS)
    s = rep["small"]
    assert s["persons"] == 2 and s["persons_over"] == 1
    assert s["steps_run"] == 8 and s["steps_over"] == 3          # steps 50, 75, 100 of "b"
    assert rep["big"]["persons_over"] == 0 and rep["big"]["steps_over"] == 0


def test_identical_steps_are_not_counted_as_runs():
    p = person("c", [30000])  # one document: steps 50/75/100 are identical to 25
    rep = budget.exceedance_report([p], MODELS)
    assert rep["small"]["steps_run"] == 1 and rep["small"]["steps_over"] == 0


def test_windows_needed_and_max_tokens():
    p = person("d", [60000, 60000])  # 100 %: ~120 k chars = ~30 k tokens > 8.5 k
    rep = budget.exceedance_report([p], MODELS)["small"]
    assert rep["max_step_tokens"] >= 30000
    assert rep["windows_total"] > rep["steps_run"]  # the over-sized steps need several windows
    assert rep["max_windows"] >= 4


def test_no_text_persons_are_counted_but_have_no_steps():
    p = person("e", [0, 0])
    rep = budget.exceedance_report([p], MODELS)["small"]
    assert rep["persons"] == 1 and rep["steps_run"] == 0 and rep["persons_over"] == 0


def test_report_has_no_person_identifiers():
    rep = budget.exceedance_report([person("secret-id", [100])], MODELS)
    assert "secret-id" not in str(rep)


def test_judge_reserves_follow_the_batch_length():
    assert budget.judge_batch_reserves(1) == (2048 + budget.JUDGE_REASONING_ALLOWANCE, 880)
    assert budget.judge_batch_reserves(40) == (2400 + budget.JUDGE_REASONING_ALLOWANCE, 800 + 80 * 40)
    assert budget.judge_batch_reserves(100) == budget.judge_batch_reserves(40)  # batches of at most 40


def test_exceedance_uses_judge_reserves_when_batch_attrs_given():
    p = person("f", [30000])
    judge = {"j": dict(ctx=80000, chars_per_token=4.0, batch_attrs=40)}
    fixed = {"j": dict(ctx=80000, chars_per_token=4.0, output_reserve=2400 + budget.JUDGE_REASONING_ALLOWANCE, prompt_overhead=4000)}
    assert budget.exceedance_report([p], judge)["j"]["usable_tokens"] ==         budget.exceedance_report([p], fixed)["j"]["usable_tokens"] == 80000 - 2400 - budget.JUDGE_REASONING_ALLOWANCE - 4000
