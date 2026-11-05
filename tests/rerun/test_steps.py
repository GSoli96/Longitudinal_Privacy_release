"""Tests of rerun.steps: exposure steps, identical-step detection, windows, token budget."""
import random
from datetime import date

import pytest

from rerun import steps
from rerun.corpus import Doc, Person


def person(texts, dates=None, status="ok", pid="X-000001"):
    dates = dates or [None] * len(texts)
    docs = [Doc(id=f"{pid}-d{i + 1:02d}", date=d, kind="pdf", text=t,
                date_kind="full" if d else None) for i, (t, d) in enumerate(zip(texts, dates))]
    return Person(pid, "X", None, None, docs, status)


def n_docs(st):
    return [s.n_docs for s in st]


# ---- steps ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("n, expected", [
    (8, [2, 4, 6, 8]),
    (4, [1, 2, 3, 4]),
    (3, [1, 2, 3, 3]),
    (2, [1, 1, 2, 2]),
    (1, [1, 1, 1, 1]),
    (5, [2, 3, 4, 5]),
    (10, [3, 5, 8, 10]),
])
def test_steps_by_documents_25_50_75_100(n, expected):
    p = person([f"testo del documento {i}" for i in range(n)])
    st = steps.build_steps(p)
    assert [s.k for s in st] == [25, 50, 75, 100]
    assert n_docs(st) == expected


def test_each_step_is_a_prefix_of_the_next_and_follows_document_order():
    p = person(["alfa " * 5, "beta " * 5, "gamma " * 5, "delta " * 5])
    st = steps.build_steps(p)
    for a, b in zip(st, st[1:]):
        assert b.text.startswith(a.text)
    assert st[0].text.index("alfa") < st[-1].text.index("delta")
    assert st[-1].text.index("beta") < st[-1].text.index("gamma")


def test_documents_are_separated_by_markers_without_personal_metadata():
    p = person(["uno", "due"], dates=[date(2020, 1, 1), date(2021, 5, 5)])
    full = steps.build_steps(p)[-1].text
    assert "[Document 1]" in full and "[Document 2]" in full
    assert "2020" not in full and "2021" not in full  # dates are not added to the text
    assert full.index("[Document 1]") < full.index("uno") < full.index("[Document 2]") < full.index("due")


def test_identical_steps_are_flagged_and_first_is_not():
    p = person(["a b c", "d e f", "g h i"])  # steps: 1,2,3,3 documents
    st = steps.build_steps(p)
    assert [s.identical_to_prev for s in st] == [False, False, False, True]
    p1 = person(["solo uno"])
    assert [s.identical_to_prev for s in steps.build_steps(p1)] == [False, True, True, True]


def test_empty_documents_do_not_change_text_but_count_as_documents():
    p = person(["testo uno", "", "testo tre", ""])
    st = steps.build_steps(p)
    assert n_docs(st) == [1, 2, 3, 4]
    assert [s.identical_to_prev for s in st] == [False, True, False, True]
    assert "[Document 3]" in st[2].text and "[Document 2]" not in st[2].text  # original numbering kept


def test_no_text_person_has_no_steps():
    p = person(["", ""], status="no_text")
    assert steps.build_steps(p) == []


def test_dated_and_undated_documents_are_counted_per_step():
    p = person(["a", "b", "c", "d"], dates=[date(2020, 1, 1), date(2021, 1, 1), None, None])
    st = steps.build_steps(p)
    assert [(s.dated_docs, s.undated_docs) for s in st] == [(1, 0), (2, 0), (2, 1), (2, 2)]


def test_step_reports_cumulative_chars():
    p = person(["a" * 100, "b" * 200])
    st = steps.build_steps(p)
    assert st[-1].chars == len(st[-1].text) and st[0].chars < st[-1].chars


# ---- token budget and windows --------------------------------------------------------------------
def test_estimate_tokens_is_conservative_rounding_up():
    assert steps.estimate_tokens("a" * 35, 3.5) == 10
    assert steps.estimate_tokens("a" * 36, 3.5) == 11
    assert steps.estimate_tokens("", 3.5) == 0


def test_usable_input_tokens_subtracts_reserves():
    assert steps.usable_input_tokens(131072, output_reserve=8192, prompt_overhead=1500) == 131072 - 8192 - 1500
    with pytest.raises(ValueError):
        steps.usable_input_tokens(1000, output_reserve=900, prompt_overhead=500)


@pytest.mark.parametrize("needed, mx, expected", [
    (5000, 131072, 8192), (8192, 131072, 8192), (8193, 131072, 16384),
    (100000, 131072, 131072), (200000, 131072, 131072), (300000, 1048576, 524288),
])
def test_choose_num_ctx_is_a_power_of_two_capped_at_the_model_maximum(needed, mx, expected):
    assert steps.choose_num_ctx(needed, mx) == expected


def make_text(n_lines=400, seed=1):
    rnd = random.Random(seed)
    words = ["dirigente", "incarico", "direzione", "ufficio", "decreto", "titolare", "funzioni", "area"]
    return "\n".join(" ".join(rnd.choice(words) for _ in range(rnd.randint(6, 14))) + f" riga{i}"
                     for i in range(n_lines))


def test_short_text_gives_one_window():
    t = make_text(5)
    assert steps.window_spans(t, ctx_tokens=10000) == [(0, len(t))]
    assert steps.windows(t, ctx_tokens=10000) == [t]


def test_windows_cover_the_whole_text_and_overlap_about_ten_percent():
    t = make_text(500)
    ctx = 1200  # tokens -> 4200 chars at 3.5
    spans = steps.window_spans(t, ctx_tokens=ctx, chars_per_token=3.5)
    assert len(spans) > 2
    assert spans[0][0] == 0 and spans[-1][1] == len(t)
    wchars = int(ctx * 3.5)
    covered = 0
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert b - a <= wchars                       # never above the budget
        assert c < b                                  # overlap, no gap
        assert b - c >= 0.07 * wchars                 # about 10 % (line snapping allowed)
        assert c > a                                  # progress
        covered = max(covered, b)
    assert covered >= spans[-1][0]
    # every character belongs to some window
    marks = [False] * len(t)
    for a, b in spans:
        for i in range(a, b):
            marks[i] = True
    assert all(marks)


def test_windows_cut_at_line_boundaries_when_possible():
    t = make_text(500)
    for a, b in steps.window_spans(t, ctx_tokens=1200)[:-1]:
        assert t[b - 1] == "\n" or t[b] == "\n" or b == len(t) or "\n" not in t[a:b]


def test_text_without_newlines_is_still_windowed():
    t = "parola " * 3000
    spans = steps.window_spans(t, ctx_tokens=1000)
    assert spans[0][0] == 0 and spans[-1][1] == len(t) and len(spans) > 1
    assert all(b - a <= int(1000 * 3.5) for a, b in spans)


def test_windows_text_equals_slices():
    t = make_text(300)
    assert steps.windows(t, 1000) == [t[a:b] for a, b in steps.window_spans(t, 1000)]


def test_too_small_budget_is_rejected():
    with pytest.raises(ValueError):
        steps.window_spans("abc " * 100, ctx_tokens=10)
