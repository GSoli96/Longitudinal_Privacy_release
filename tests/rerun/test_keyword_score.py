import json

import pytest

from rerun import keyword_score as ks
from rerun import phish_corpus

KW = ("incarico", "ufficio", "responsabile", "dichiarazione", "data", "funzione")


def test_keywords_read_from_the_source_file():
    assert ks.load_keywords() == KW


def test_keywords_missing_assignment_raises(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("A = 1\n")
    with pytest.raises(ValueError):
        ks.load_keywords(p)


def test_band_thresholds_match_the_old_rule():
    assert [ks.band(n) for n in range(0, 7)] == ["low", "low", "medium", "medium", "high", "high", "high"]


def test_substring_matches_inside_words_but_whole_word_does_not():
    text = "Il database dell'Ufficio, dichiarazione3"
    assert set(ks.present_substring(text, KW)) == {"ufficio", "dichiarazione", "data"}
    assert set(ks.present_word(text, KW)) == {"ufficio"}
    assert set(ks.present_word("Data: 3 maggio", KW)) == {"data"}


def test_summarise_counts_shares_and_keyword_prevalence():
    texts = ["incarico ufficio responsabile data", "ufficio data", "ciao", "ciao ufficio"]
    s = ks.summarise(texts, KW)
    assert s["n_messages"] == 4
    assert s["substring"]["counts"] == {"high": 1, "medium": 1, "low": 2}
    assert s["substring"]["shares"]["high"] == 0.25
    assert s["word"]["messages_with_keyword"]["ufficio"] == 3
    assert s["substring"]["mean_keywords_present"] == pytest.approx((4 + 2 + 0 + 1) / 4)


def test_summarise_empty_has_no_shares():
    s = ks.summarise([], KW)
    assert s["n_messages"] == 0 and s["word"]["shares"]["low"] is None


def test_run_selects_positive_explicit_inferred_only(tmp_path):
    d = tmp_path / "corpus"
    for g in ("gptoss", "llama31"):
        rows = []
        for i, (label, variant, body) in enumerate([
                (1, "explicit_inferred", "incarico ufficio responsabile dichiarazione"),
                (1, "explicit_inferred", "ciao"),
                (1, "none", "incarico ufficio responsabile dichiarazione"),
                (0, "explicit_inferred", "incarico ufficio responsabile dichiarazione")]):
            rows.append({"id": f"{g}{i}", "person": f"A-{i}", "case": "A", "variant": variant, "attack": "BEC",
                         "kind": "positive" if label else "negative", "index": 0, "generator": g, "label": label,
                         "subject": "s", "body": body})
        phish_corpus.write({"rows": rows, "report": {}}, f"rq3_A_{g}", d)
    out = tmp_path / "out"
    res = ks.run(cases=("A",), corpus_dir=d, out_dir=out)
    e = res["corpora"]["A/gptoss"]
    assert e["n_messages"] == 2 and e["substring"]["counts"] == {"high": 1, "medium": 0, "low": 1}
    assert e["by_attack"]["BEC"]["n_messages"] == 2
    assert set(res["corpora"]) == {"A/gptoss", "A/llama31"}
    assert json.loads((out / "keyword_score.json").read_text())["variant"] == "explicit_inferred"
