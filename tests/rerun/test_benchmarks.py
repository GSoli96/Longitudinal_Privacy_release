import csv

import pytest

from rerun import benchmarks

HEAD = ["sender", "receiver", "date", "subject", "body", "label", "urls"]


def _write(path, rows, head=HEAD):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def _src(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    row = lambda s, b, lab: ["a@x", "b@y", "d", s, b, lab, ""]  # noqa: E731
    _write(src / "CEAS_08.csv", [row("s1", "  hello ", "1"), row("s1", "hello", "1"), row("s2", "", "0"),
                                 row("s3", "ciao", "x"), row("s4", "other", "0")])
    _write(src / "Enron.csv", [["s", "b1", "0"], ["s", "b2", "1"]], head=["subject", "body", "label"])
    _write(src / "Ling.csv", [["s", "b1", "0"]], head=["subject", "body", "label"])
    _write(src / "Nazario.csv", [row("n", "phish", "1")])
    _write(src / "Nigerian_Fraud.csv", [row("n", "fraud", "1")])
    _write(src / "SpamAssasin.csv", [row("n", "spam", "1"), row("n", "ham", "0")])
    return src


def test_clean_counts_and_strip(tmp_path):
    src = _src(tmp_path)
    rows, c = benchmarks.clean_csv("CEAS_08", src / "CEAS_08.csv")
    assert c["read"] == 5 and c["kept"] == 2 and c["kept_positive"] == 1
    assert c["dropped_duplicate"] == 1 and c["dropped_empty_body"] == 1 and c["dropped_label"] == 1
    assert rows[0]["body"] == "hello" and rows[0]["label"] == 1
    assert len({r["uid"] for r in rows}) == 2


def test_uid_depends_on_corpus():
    assert benchmarks._uid("A", "s", "b") != benchmarks._uid("B", "s", "b")
    assert benchmarks._uid("A", "s", "b") == benchmarks._uid("A", "s", "b")


def test_prepare_builds_then_reuses(tmp_path):
    src, out = _src(tmp_path), tmp_path / "out"
    m1 = benchmarks.prepare(src, out)
    assert set(m1["status"].values()) == {"built"}
    assert set(m1["corpora"]) == set(benchmarks.BENCHMARKS)
    m2 = benchmarks.prepare(src, out)
    assert set(m2["status"].values()) == {"reused"}
    assert len(benchmarks.load("Enron", out)) == 2


def test_prepare_rebuilds_when_source_changes(tmp_path):
    src, out = _src(tmp_path), tmp_path / "out"
    benchmarks.prepare(src, out)
    _write(src / "Ling.csv", [["s", "b1", "0"], ["s", "b9", "1"]], head=["subject", "body", "label"])
    m = benchmarks.prepare(src, out)
    assert m["status"]["Ling"] == "built" and m["status"]["Enron"] == "reused"
    assert len(benchmarks.load("Ling", out)) == 2


def test_prepare_rebuilds_on_new_clean_version(tmp_path, monkeypatch):
    src, out = _src(tmp_path), tmp_path / "out"
    benchmarks.prepare(src, out, names=["Ling"])
    monkeypatch.setattr(benchmarks, "CLEAN_VERSION", benchmarks.CLEAN_VERSION + 1)
    assert benchmarks.prepare(src, out, names=["Ling"])["status"]["Ling"] == "built"


def test_load_detects_modified_file(tmp_path):
    src, out = _src(tmp_path), tmp_path / "out"
    benchmarks.prepare(src, out)
    with open(out / "Ling.jsonl", "a", encoding="utf-8") as f:
        f.write("{}\n")
    with pytest.raises(ValueError):
        benchmarks.load("Ling", out)


def test_load_before_prepare_and_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        benchmarks.load("Ling", tmp_path / "none")
    with pytest.raises(FileNotFoundError):
        benchmarks.prepare(tmp_path / "empty-not-there", tmp_path / "o", names=["Ling"])
    with pytest.raises(KeyError):
        benchmarks.prepare(_src(tmp_path), tmp_path / "o2", names=["Nope"])


def test_source_dir_from_environment(tmp_path, monkeypatch):
    monkeypatch.delenv(benchmarks.SOURCE_ENV, raising=False)
    with pytest.raises(FileNotFoundError):
        benchmarks.source_dir()
    monkeypatch.setenv(benchmarks.SOURCE_ENV, str(tmp_path))
    assert benchmarks.source_dir() == tmp_path


def test_summary_has_counts_only(tmp_path):
    m = benchmarks.prepare(_src(tmp_path), tmp_path / "out")
    text = benchmarks.summary(m)
    assert "CEAS_08: read 5, kept 2" in text and "hello" not in text
