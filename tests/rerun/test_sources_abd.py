"""Sources of Cases A, B and D (main-PC phase, 2026-10-23): fixtures only, no personal data."""
import shutil
from datetime import date

import pytest

from rerun import corpus

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

TEXT = ("Il dirigente ha ricevuto l'incarico presso la Direzione generale il 3 marzo 2019 con decreto "
        "e relativa dichiarazione di insussistenza di cause di inconferibilita. ")


def pdf(path, pages):
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as p:
        for t in pages:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.05, 0.9, t, fontsize=7, wrap=True)
            p.savefig(fig)
            plt.close(fig)


def load(case, tmp_path):
    return corpus.load_case_with_report(case, raw_dir=tmp_path / "raw", private_dir=tmp_path / "private",
                                        ocr_page=lambda im: ("", 0.0))


def html_page(path, body, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"<html><body><nav>menu</nav><h1>Profilo</h1><p>{body}</p><script>x=1</script></body></html>"
                     .encode(encoding))


def test_case_a_profile_pages_are_documents_and_profile_only_persons_exist(tmp_path):
    """D50: the saved profile pages are documents like the PDFs; a person with a page and no PDF is a person."""
    root = tmp_path / "raw" / "caseA"
    pdf(root / "downloads" / "ROSSI_MARIO" / "cv.pdf", [TEXT * 3])
    pdf(root / "downloads" / "BIANCHI_ANNA" / "dich.pdf", [TEXT * 2])
    for n in ("ROSSI_MARIO", "BIANCHI_ANNA", "VERDI_LUCA"):
        html_page(root / "profiles" / f"{n}.html", TEXT)
    persons, rep = load("A", tmp_path)
    assert len(persons) == 3 and all(p.status == "ok" for p in persons)
    assert rep["profiles_html"] == 3 and rep["persons_from_profile_only"] == 1
    kinds = sorted(sorted(d.kind for d in p.docs) for p in persons)
    assert kinds == [["html"], ["html", "pdf"], ["html", "pdf"]]
    page = next(d for p in persons for d in p.docs if d.kind == "html")
    assert "menu" not in page.text and "x=1" not in page.text and "Direzione generale" in page.text


def test_case_a_folder_names_come_from_layout_json_and_old_pages_are_cp1252(tmp_path):
    root = tmp_path / "raw" / "caseA"
    pdf(root / "dl" / "ROSSI_MARIO" / "cv.pdf", [TEXT])
    html_page(root / "pp" / "ROSSI_MARIO.html", "L’incarico presso l’ufficio. " + TEXT, encoding="cp1252")
    lay = tmp_path / "private" / "caseA" / "layout.json"
    lay.parent.mkdir(parents=True)
    lay.write_text('{"downloads": "dl", "profiles": "pp"}', encoding="utf-8")
    persons, rep = load("A", tmp_path)
    assert len(persons) == 1 and rep["profiles_html"] == 1
    page = next(d for d in persons[0].docs if d.kind == "html")
    assert "�" not in page.text and "L’incarico" in page.text


def test_case_b_groups_by_name_uses_filename_dates_and_excludes_unsplit(tmp_path):
    sep = tmp_path / "raw" / "caseB" / "Documenti" / "REDOWN" / "Curriculum" / "PDF_SEPARATI"
    pdf(sep / "ROSSI_MARIO_2021-05-04.pdf", [TEXT + " Firmato il 1 gennaio 2023."])
    pdf(sep / "ROSSI_MARIO_2019-02-01.pdf", [TEXT])
    pdf(sep / "ROSSI_MARIO_2019-02-01_2.pdf", [TEXT + " altra versione"])
    pdf(sep / "BIANCHI_ANNA_2020-10-10.pdf", [TEXT])
    pdf(sep.parent / "Curriculum_tutti.pdf", [TEXT * 4])          # multi-person original: excluded
    persons, rep = load("B", tmp_path)
    assert len(persons) == 2
    rossi = max(persons, key=lambda p: len(p.docs))
    assert [d.date for d in rossi.docs] == [date(2019, 2, 1), date(2019, 2, 1), date(2021, 5, 4)]
    assert {d.date_kind for d in rossi.docs} == {"filename"}      # not the 2023 date of the text
    assert rep["excluded_unsplit_pdfs"] == 1 and rep["unmatched_file_names"] == 0


def test_case_d_pooled_acts_go_only_to_the_named_person_profiles_excluded(tmp_path):
    ats = tmp_path / "raw" / "caseD" / "docs"
    pdf(ats / "ROSSI_MARIO" / "cv.pdf", [TEXT])
    pdf(ats / "BIANCHI_ANNA" / "cv.pdf", [TEXT + " Altra sede di servizio."])   # not byte-identical to the other cv (D53)
    pdf(ats / "_determinazioni" / "det1.pdf", [TEXT + " incarico conferito a Mario Rossi, dirigente.",
                                                "Pagina senza nomi: elenco delle spese del servizio."])
    pdf(ats / "_determinazioni" / "det2.pdf", [TEXT + " incarico conferito a Mario Rossini."])  # not a match
    pdf(ats / "_determinazioni" / "det3.pdf", [TEXT + " Rossi Mario e Bianchi Anna, commissione."])
    pdf(ats / "_unmatched" / "x.pdf", [TEXT])
    (ats / "doc").mkdir(parents=True)
    (ats / "doc" / "ROSSI_MARIO.html").write_text("<html>p</html>", encoding="utf-8")
    persons, rep = load("D", tmp_path)
    assert len(persons) == 2                                      # doc/, _unmatched, pooled are not persons
    n_docs = sorted(len(p.docs) for p in persons)
    assert n_docs == [2, 3]                                       # Rossi: cv + det1 + det3; Bianchi: cv + det3
    assert rep["pooled_acts"] == 3 and rep["pooled_acts_assigned"] == 2
    assert rep["pooled_acts_in_several_profiles"] == 1 and rep["pooled_assignments"] == 3
    assert rep["pooled_pages_assigned"] == 3                     # one page of det1, one of det3 twice
    rossi = max(persons, key=lambda p: len(p.docs))
    pooled_texts = [d.text for d in rossi.docs if d.kind == "pdf_pages"]
    assert len(pooled_texts) == 2
    assert all("elenco delle spese" not in t for t in pooled_texts)   # the page without the name stays out


def test_case_d_error_pages_saved_as_pdf_are_not_documents(tmp_path):
    """D51: a portal error page stored with the .pdf extension is counted in ``not_pdf``; the person stays."""
    ats = tmp_path / "raw" / "caseD" / "docs"
    pdf(ats / "ROSSI_MARIO" / "cv.pdf", [TEXT])
    (ats / "VERDI_LUCA").mkdir(parents=True)
    (ats / "VERDI_LUCA" / "scheda.pdf").write_bytes(b"<html><body>Pagina non trovata</body></html>")
    persons, rep = load("D", tmp_path)
    assert len(persons) == 2
    verdi = next(p for p in persons if not p.docs)
    assert verdi.status == "no_text"
    assert rep["not_pdf"] == 1 and rep["persons_only_not_pdf"] == 1 and rep["no_text"] == 1


def test_case_d_a_file_in_several_person_folders_gives_each_only_their_pages(tmp_path):
    """D53: the same file (same hash) in the folders of two persons is a cumulative act."""
    ats = tmp_path / "raw" / "caseD" / "docs"
    act = ats / "ROSSI_MARIO" / "atto.pdf"
    pdf(act, [TEXT + " incarico conferito a Mario Rossi.", TEXT + " incarico conferito a Anna Bianchi.",
              "Pagina senza nomi."])
    (ats / "BIANCHI_ANNA").mkdir(parents=True)
    shutil.copy(act, ats / "BIANCHI_ANNA" / "atto.pdf")
    pdf(ats / "BIANCHI_ANNA" / "cv.pdf", [TEXT])
    persons, rep = load("D", tmp_path)
    assert rep["shared_files"] == 1 and rep["shared_copies"] == 2 and rep["shared_persons"] == 2
    assert rep["shared_pages_assigned"] == 2
    texts = {len(p.docs): [d.text for d in p.docs if d.kind == "pdf_pages"] for p in persons}
    for p in persons:
        shared = [d.text for d in p.docs if d.kind == "pdf_pages"]
        assert len(shared) == 1 and "Pagina senza nomi" not in shared[0]
    names = sorted("Rossi" in d.text or "Bianchi" in d.text for p in persons for d in p.docs
                   if d.kind == "pdf_pages")
    assert names == [True, True] and texts


def test_case_d_profile_page_without_folder_makes_a_person_when_the_acts_name_them(tmp_path):
    """D52: a profile page with no folder + the pooled pages that name the person; none: excluded and counted."""
    ats = tmp_path / "raw" / "caseD" / "docs"
    pdf(ats / "ROSSI_MARIO" / "cv.pdf", [TEXT])
    pdf(ats / "_determinazioni" / "det.pdf", [TEXT + " incarico conferito a Paolo Neri, dirigente."])
    html_page(ats / "doc" / "NERI_PAOLO.html", TEXT + " Profilo del dirigente.")
    html_page(ats / "doc" / "BLU_GINO.html", TEXT)
    html_page(ats / "doc" / "ROSSI_MARIO.html", TEXT)             # has a folder: not created, not a document
    persons, rep = load("D", tmp_path)
    assert len(persons) == 2
    assert rep["profile_pages_without_folder"] == 2
    assert rep["persons_created_from_acts"] == 1 and rep["persons_excluded_no_acts"] == 1
    neri = next(p for p in persons if any(d.kind == "html" for d in p.docs))
    assert sorted(d.kind for d in neri.docs) == ["html", "pdf_pages"]


def test_case_h_is_read_like_a_pdf_folders_plus_one_profile_page_per_person(tmp_path):
    """D49: H has up to ten PDFs per person and one profile page for every person."""
    root = tmp_path / "raw" / "caseH"
    pdf(root / "downloads" / "P1" / "a.pdf", [TEXT * 2])
    pdf(root / "downloads" / "P1" / "b.pdf", [TEXT * 3])
    pdf(root / "downloads" / "P2" / "a.pdf", [TEXT * 2])
    for n in ("P1", "P2"):
        html_page(root / "profiles" / f"{n}.html", TEXT)
    (root / "excluded_units").mkdir()
    html_page(root / "excluded_units" / "index.html", "not a person")     # outside profiles: never read
    persons, rep = load("H", tmp_path)
    assert len(persons) == 2 and rep["profiles_html"] == 2 and rep["persons_from_profile_only"] == 0
    assert sorted(len(p.docs) for p in persons) == [2, 3]
    assert not corpus.MISSING_CASES
