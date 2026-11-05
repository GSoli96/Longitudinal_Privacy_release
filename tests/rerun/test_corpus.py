"""Tests of rerun.corpus on synthetic fixtures (no personal data)."""
import json
import re
from datetime import date

import pytest

from rerun import corpus, extract

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TEXT_A = ("Il dirigente ha ricevuto l'incarico presso la Direzione generale il 3 marzo 2019 "
          "con decreto del Ministro e relativa dichiarazione di insussistenza di cause di inconferibilita. ")
TEXT_B = ("Curriculum vitae: laurea in giurisprudenza conseguita nel 2004, master di secondo livello "
          "nel 2007, esperienza presso la Direzione territoriale dal 2010 al 2015. ")


def pdf(path, pages):
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as p:
        for t in pages:
            fig = plt.figure(figsize=(8.27, 11.69))
            if t:
                fig.text(0.05, 0.9, t, fontsize=7, wrap=True)
            p.savefig(fig)
            plt.close(fig)


def fake_ocr(image):
    return "", 0.0


def load(case, tmp_path, **kw):
    return corpus.load_case_with_report(case, raw_dir=tmp_path / "raw", private_dir=tmp_path / "private",
                                        ocr_page=fake_ocr, vlm_page=None, **kw)


# ---- file-tree cases (C, E) -----------------------------------------------------------------------
def make_tree(tmp_path, case="C"):
    root = tmp_path / "raw" / f"case{case}"
    pdf(root / "Mario Rossi" / "Incarico_2019.pdf", [TEXT_A * 2])
    pdf(root / "Mario Rossi" / "cv.pdf", [TEXT_B * 2, TEXT_B])
    pdf(root / "Anna Bianchi" / "dich.pdf", [TEXT_A])
    pdf(root / "Luisa Verdi" / "scan.pdf", [""])  # no text layer, OCR returns nothing -> no_text
    (root / "Anna Bianchi" / "note.xlsx").write_bytes(b"PK")  # spreadsheet, excluded by default
    return root


def test_persons_documents_and_ids(tmp_path):
    make_tree(tmp_path)
    persons, rep = load("C", tmp_path, include_spreadsheets=False)
    assert len(persons) == 3 and len({p.id for p in persons}) == 3
    assert all(re.fullmatch(r"C-[0-9a-f]{6,}", p.id) for p in persons)
    assert all(p.case == "C" for p in persons)
    by_docs = {len(p.docs): p for p in persons}
    assert set(by_docs) == {2, 1}
    assert rep["persons"] == 3 and rep["docs"] == 4


def test_no_text_person_is_kept_with_status(tmp_path):
    make_tree(tmp_path)
    persons, rep = load("C", tmp_path)
    no_text = [p for p in persons if p.status == "no_text"]
    assert len(no_text) == 1 and rep["no_text"] == 1
    assert sum(len(d.text) for d in no_text[0].docs) == 0
    assert len(persons) == 3  # never dropped


def test_spreadsheets_can_be_excluded_and_counted(tmp_path):
    make_tree(tmp_path)
    persons, rep = load("C", tmp_path, include_spreadsheets=False)
    assert rep["spreadsheets_excluded"] == 1
    assert all(d.kind != "spreadsheet" for p in persons for d in p.docs)


def test_spreadsheets_are_included_by_default_and_reported(tmp_path):
    import openpyxl
    root = make_tree(tmp_path)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Anno", "Voce", "Importo"])
    ws.append([2022, "Stipendio", 51000])
    wb.save(root / "Anna Bianchi" / "note.xlsx")  # replaces the fake file
    big = openpyxl.Workbook()
    big.active.append(["Voce", "Valore"])
    for i in range(300):
        big.active.append([f"v{i}", i])
    big.save(root / "Mario Rossi" / "big.xlsx")
    persons, rep = load("C", tmp_path)
    ss = [d for p in persons for d in p.docs if d.kind == "spreadsheet"]
    assert len(ss) == 1 and "Voce: Stipendio" in ss[0].text
    assert rep["spreadsheets_included"] == 1 and rep["spreadsheet_persons"] == 1
    assert rep["spreadsheet_chars"] == len(ss[0].text) > 0
    assert rep["spreadsheets_flagged"] == {"too_many_rows": 1}


def test_dates_and_order_are_extracted(tmp_path):
    make_tree(tmp_path)
    persons, rep = load("C", tmp_path)
    docs = [d for p in persons for d in p.docs]
    dated = [d for d in docs if d.date]
    assert any(d.date == date(2019, 3, 3) and d.date_kind == "full" for d in dated)
    assert rep["dates"]["full"] >= 1
    assert sum(rep["dates"].values()) == len(docs)


def test_person_ids_are_opaque_and_map_is_private(tmp_path):
    make_tree(tmp_path)
    persons, _ = load("C", tmp_path)
    assert not any("Rossi" in p.id or "Mario" in p.id for p in persons)
    m = json.loads((tmp_path / "private" / "caseC" / "persons_map.json").read_text(encoding="utf-8"))
    assert set(m.values()) == {p.id for p in persons}
    assert any("Rossi" in k for k in m)  # the map (private) keeps the names
    assert not any("Rossi" in d.id for p in persons for d in p.docs)


def test_ids_are_stable_across_runs_and_depend_on_the_private_salt(tmp_path):
    make_tree(tmp_path)
    ids1 = [p.id for p in load("C", tmp_path)[0]]
    ids2 = [p.id for p in load("C", tmp_path)[0]]
    assert ids1 == ids2
    (tmp_path / "raw" / "caseC" / "Zeno Neri").mkdir()
    pdf(tmp_path / "raw" / "caseC" / "Zeno Neri" / "x.pdf", [TEXT_A])
    ids3 = [p.id for p in load("C", tmp_path)[0]]
    assert set(ids1) <= set(ids3) and len(ids3) == len(ids1) + 1  # adding a person does not renumber
    (tmp_path / "private" / "id_salt.txt").write_text("another salt", encoding="utf-8")
    ids4 = [p.id for p in load("C", tmp_path)[0]]
    assert not set(ids4) & set(ids3)


def test_accented_names_load(tmp_path):
    root = tmp_path / "raw" / "caseC"
    pdf(root / "José Pérez" / "Atto_n°_5_–_Ufficio.pdf", [TEXT_A])
    persons, _ = load("C", tmp_path)
    assert len(persons) == 1 and persons[0].docs[0].text


def test_hash_suffixed_collisions_similar_variants_keep_one(tmp_path):
    root = tmp_path / "raw" / "caseE"
    pdf(root / "Pers A" / "Doc_Uno__aaaa1111.pdf", [TEXT_A * 2])
    pdf(root / "Pers A" / "doc_uno__bbbb2222.pdf", [TEXT_A * 2])  # same document downloaded twice
    pdf(root / "Pers A" / "Altro__cccc3333.pdf", [TEXT_B * 2])
    pdf(root / "Pers A" / "altro__dddd4444.pdf", [TEXT_A * 2 + " completamente diverso " * 20])
    persons, rep = load("E", tmp_path)
    names = [d.id for d in persons[0].docs]
    assert len(persons[0].docs) == 3  # first pair collapsed, second pair kept distinct
    c = rep["collisions"]
    assert c["groups"] == 2 and c["similar_kept_one"] == 1 and c["different_kept_both"] == 1


def test_unreadable_pdf_is_reported_not_fatal(tmp_path):
    root = tmp_path / "raw" / "caseC"
    pdf(root / "Pers A" / "ok.pdf", [TEXT_A])
    (root / "Pers A").mkdir(exist_ok=True)
    (root / "Pers A" / "bad.pdf").write_bytes(b"")
    persons, rep = load("C", tmp_path)
    assert rep["unreadable_files"] == 1
    assert len(persons[0].docs) == 2 and any(d.error for d in persons[0].docs)


# ---- case G ----------------------------------------------------------------------------------------
def make_g(tmp_path, second_has_name=True):
    raw = tmp_path / "raw" / "caseG"
    named = "Dichiarazione di Mario Rossi. " + TEXT_A
    named_b = "Curriculum di Rossi Mario. " + TEXT_B
    pdf(raw / "docs" / "Sezione_Uno" / "Rossi_Mario" / "a.pdf", [named])
    pdf(raw / "docs" / "Sezione_Uno" / "Bianchi_Anna" / "b.pdf", [TEXT_B + " Anna Bianchi"])
    pdf(raw / "docs" / "Sezione_Due" / "Verdi_Luisa" / "c.pdf", [TEXT_A + " Luisa Verdi"])
    pdf(raw / "docs" / "Albo_Docenti" / "elenco.pdf", [TEXT_B])               # section-level files = unit
    pdf(raw / "docs" / "Sezione_Due" / "Rossi_Mario" / "d.pdf", [named_b if second_has_name else TEXT_B])
    (raw / "profiles").mkdir(parents=True)
    for n in ("Sezione_Uno__Rossi_Mario", "Sezione_Uno__Bianchi_Anna", "Sezione_Due__Verdi_Luisa",
              "Sezione_Due__Rossi_Mario", "Albo_Docenti"):
        (raw / "profiles" / f"{n}.html").write_text("<html><h1>x</h1></html>", encoding="utf-8")
    return raw


def test_case_g_units_excluded_and_counted_people_merged(tmp_path):
    make_g(tmp_path)
    persons, rep = load("G", tmp_path)
    assert len(persons) == 3
    assert rep["excluded_units"] == 1
    assert rep["profiles_html"] == 5 and rep["merged_people"] == 1 and rep["merge_doubtful_kept_separate"] == 0
    rossi = [p for p in persons if len(p.docs) == 2]
    assert len(rossi) == 1 and rossi[0].meta["sections"] == 2
    m = json.loads((tmp_path / "private" / "caseG" / "merged_people.json").read_text(encoding="utf-8"))
    assert "Rossi_Mario" in m["merged"]


def test_case_g_same_name_without_textual_support_stays_separate(tmp_path):
    make_g(tmp_path, second_has_name=False)
    persons, rep = load("G", tmp_path)
    assert len(persons) == 4  # the two Rossi pages are not merged
    assert rep["merged_people"] == 0 and rep["merge_doubtful_kept_separate"] == 1
    m = json.loads((tmp_path / "private" / "caseG" / "merged_people.json").read_text(encoding="utf-8"))
    assert "Rossi_Mario" in m["kept_separate_doubtful"]


# ---- case F ----------------------------------------------------------------------------------------
def test_case_f_reads_texts_from_json(tmp_path):
    raw = tmp_path / "raw" / "caseF"
    raw.mkdir(parents=True)
    data = {"Persona Uno": {"score": "7", "documenti": [
                {"filename": "cv_2018.pdf", "tipo": "CV", "testo": TEXT_B * 3},
                {"filename": "dich.pdf", "tipo": "Dichiarazione", "testo": TEXT_A}]},
            "Persona Due": {"score": "3", "documenti": [{"filename": "x.pdf", "tipo": "CV", "testo": ""}]}}
    (raw / "database_ai_ready.json").write_text(json.dumps(data), encoding="utf-8")
    persons, rep = load("F", tmp_path)
    assert len(persons) == 2 and rep["persons"] == 2
    assert {p.status for p in persons} == {"ok", "no_text"}
    p1 = next(p for p in persons if p.status == "ok")
    assert [d.kind for d in p1.docs] == ["CV", "Dichiarazione"]


# ---- role and org type from private side files -----------------------------------------------------
def test_org_type_comes_from_private_case_file(tmp_path):
    make_tree(tmp_path)
    (tmp_path / "private" / "caseC").mkdir(parents=True)
    (tmp_path / "private" / "caseC" / "case.json").write_text(json.dumps({"org_type": "ministry"}), encoding="utf-8")
    persons, _ = load("C", tmp_path)
    assert {p.org_type for p in persons} == {"ministry"}


def test_unknown_case_is_a_clear_error(tmp_path):
    with pytest.raises(KeyError):
        load("Z", tmp_path)


def test_all_case_labels_have_a_source_interface():
    assert set(corpus.SOURCES) >= {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert set(corpus.MISSING_CASES) == set()   # H arrived (D49); a missing case gives a clear message, not an empty corpus
    assert not set(corpus.SOURCES) & set(corpus.MISSING_CASES)


def test_docs_remember_source_and_non_text_layer_pages(tmp_path):
    root = tmp_path / "raw" / "caseC"
    pdf(root / "Pers A" / "mix.pdf", [TEXT_A * 2, ""])  # second page has no text layer
    from PIL import Image
    persons, rep = corpus.load_case_with_report(
        "C", raw_dir=tmp_path / "raw", private_dir=tmp_path / "private",
        ocr_page=lambda im: ("ocr " * 60, 90.0), vlm_page=None)
    d = persons[0].docs[0]
    assert d.source.endswith("mix.pdf")
    assert d.page_methods == {2: "blank"}  # the second page is a white sheet
    assert rep["pages_by_method"] == {"text_layer": 1, "blank": 1}


def test_assign_ids_same_length_per_case_and_extends_on_collision():
    keys = [f"person {i}" for i in range(60)]
    ids = corpus.assign_ids("H", keys, "salt", lengths=(1, 8))  # 1 hex digit cannot hold 60 persons
    assert len(set(ids.values())) == 60
    assert {len(v) for v in ids.values()} == {len("H-") + 8}
    ids6 = corpus.assign_ids("H", keys[:5], "salt")
    assert {len(v) for v in ids6.values()} == {len("H-") + 6}
    with pytest.raises(AssertionError):
        corpus.assign_ids("H", keys, "salt", lengths=(1, 1))


def test_documents_are_chronological_undated_last_in_original_order(tmp_path):
    root = tmp_path / "raw" / "caseC" / "Pers A"
    pdf(root / "a_late.pdf", ["Roma, 3 marzo 2021. " + TEXT_B * 2])
    pdf(root / "b_undated.pdf", ["senza alcuna indicazione temporale per questo testo di prova abbastanza lungo " * 3])
    pdf(root / "c_early.pdf", ["Roma, 5 maggio 2017. " + TEXT_B * 2])
    pdf(root / "d_undated.pdf", ["altro testo di prova privo di date ma abbastanza lungo da avere un livello di testo " * 3])
    persons, rep = load("C", tmp_path)
    names = [d.filename for d in persons[0].docs]
    assert names == ["c_early.pdf", "a_late.pdf", "b_undated.pdf", "d_undated.pdf"]
    assert rep["dates"]["none"] == 2


def test_case_g_keeps_section_order_not_dates(tmp_path):
    raw = tmp_path / "raw" / "caseG" / "docs"
    pdf(raw / "Sezione_A" / "Rossi_Mario" / "x.pdf", ["Roma, 3 marzo 2021. Mario Rossi " + TEXT_B * 2])
    pdf(raw / "Sezione_B" / "Rossi_Mario" / "y.pdf", ["Roma, 5 maggio 2017. Mario Rossi " + TEXT_B * 2])
    persons, rep = load("G", tmp_path)
    assert len(persons) == 1
    assert [d.filename for d in persons[0].docs] == ["x.pdf", "y.pdf"]  # section order, although y is older


def test_vision_model_is_off_by_default_pages_stay_tesseract_low(tmp_path):
    root = tmp_path / "raw" / "caseC"
    pdf(root / "Pers A" / "scan.pdf", [TEXT_A * 2, ""])
    called = []
    persons, rep = corpus.load_case_with_report(
        "C", raw_dir=tmp_path / "raw", private_dir=tmp_path / "private",
        ocr_page=lambda im: ("poco", 20.0))  # vlm_page not given: the default is OFF
    assert called == []
    d = persons[0].docs[0]
    assert d.page_methods in ({2: "tesseract_low"}, {2: "blank"})
