"""Tests of rerun.extract: page-level method choice (text layer / tesseract / vlm), caching,
blank pages, unreadable files, dates and HTML cleaning. OCR and VLM are injected."""
import io
import json
from datetime import date

import pytest

from rerun import extract
from rerun.extract import Page, extract_pdf, page_method_summary

try:  # synthetic PDFs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    pytest.skip("matplotlib/Pillow missing", allow_module_level=True)


LONG = "Il dirigente ha ricevuto l'incarico presso la Direzione generale il 3 marzo 2019. " * 3


def make_text_pdf(path, pages):
    with PdfPages(path) as pdf:
        for txt in pages:
            fig = plt.figure(figsize=(8.27, 11.69))
            if txt:
                fig.text(0.05, 0.9, txt, fontsize=8, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)


def make_image_pdf(path, pages, blank=()):
    imgs = []
    for i, txt in enumerate(pages):
        img = Image.new("RGB", (1240, 1754), "white")
        if txt and i not in blank:
            d = ImageDraw.Draw(img)
            try:
                f = ImageFont.truetype("arial.ttf", 34)
            except OSError:
                f = ImageFont.load_default()
            y = 100
            for line in txt.split("|"):
                d.text((80, y), line, fill="black", font=f)
                y += 60
        imgs.append(img)
    imgs[0].save(path, "PDF", resolution=150.0, save_all=True, append_images=imgs[1:])


class Fakes:
    def __init__(self, ocr=("x" * 300, 90.0), vlm="trascrizione letterale della pagina " * 5):
        self.ocr_result, self.vlm_text = ocr, vlm
        self.ocr_calls = self.vlm_calls = 0

    def ocr(self, image):
        self.ocr_calls += 1
        return self.ocr_result

    def vlm(self, image):
        self.vlm_calls += 1
        return self.vlm_text


def run(path, fk, cache=None, **kw):
    return extract_pdf(path, ocr_page=fk.ocr, vlm_page=fk.vlm, cache_dir=cache,
                       render=lambda p, n: Image.new("RGB", (200, 200), "gray"), **kw)


def test_text_layer_pages_do_not_call_ocr(tmp_path):
    p = tmp_path / "a.pdf"
    make_text_pdf(p, [LONG, LONG])
    fk = Fakes()
    r = run(p, fk)
    assert [pg.method for pg in r.pages] == ["text_layer", "text_layer"]
    assert fk.ocr_calls == 0 and fk.vlm_calls == 0
    assert "Direzione generale" in r.text and r.error is None


def test_page_without_text_layer_goes_to_tesseract(tmp_path):
    p = tmp_path / "a.pdf"
    make_text_pdf(p, [LONG, ""])  # second page empty -> not blank image in fake render (gray)
    fk = Fakes(ocr=("ocr text " * 40, 91.0))
    r = run(p, fk)
    assert [pg.method for pg in r.pages] == ["text_layer", "tesseract"]
    assert r.pages[1].conf == 91.0 and fk.vlm_calls == 0


@pytest.mark.parametrize("ocr", [("short", 95.0), ("x " * 200, 40.0), ("", 0.0)])
def test_tesseract_failure_goes_to_vlm(tmp_path, ocr):
    p = tmp_path / "a.pdf"
    make_text_pdf(p, [""])
    fk = Fakes(ocr=ocr)
    r = run(p, fk)
    assert r.pages[0].method == "vlm" and fk.vlm_calls == 1
    assert r.pages[0].text.startswith("trascrizione")
    assert r.pages[0].tesseract_chars == len(ocr[0].strip())


def test_vlm_disabled_keeps_low_quality_tesseract_flagged(tmp_path):
    p = tmp_path / "a.pdf"
    make_text_pdf(p, [""])
    fk = Fakes(ocr=("poco", 30.0))
    r = extract_pdf(p, ocr_page=fk.ocr, vlm_page=None,
                    render=lambda a, b: Image.new("RGB", (200, 200), "gray"))
    assert r.pages[0].method == "tesseract_low" and r.pages[0].text == "poco"


def test_vlm_empty_answer_marks_page_failed_not_text(tmp_path):
    p = tmp_path / "a.pdf"
    make_text_pdf(p, [""])
    fk = Fakes(ocr=("", 0.0), vlm="")
    r = run(p, fk)
    assert r.pages[0].method == "failed" and r.pages[0].text == ""


def test_ocr_exception_does_not_abort_the_document(tmp_path):
    p = tmp_path / "a.pdf"
    make_text_pdf(p, [LONG, "", LONG])

    def boom(image):
        raise RuntimeError("tesseract timed out")

    fk = Fakes()
    r = extract_pdf(p, ocr_page=boom, vlm_page=fk.vlm,
                    render=lambda a, b: Image.new("RGB", (200, 200), "gray"))
    assert [pg.method for pg in r.pages] == ["text_layer", "vlm", "text_layer"]
    assert r.pages[1].conf == 0.0 and fk.vlm_calls == 1
    r2 = extract_pdf(p, ocr_page=boom, vlm_page=None,
                     render=lambda a, b: Image.new("RGB", (200, 200), "gray"))
    assert r2.pages[1].method == "tesseract_low" and r2.pages[1].text == ""


def test_vlm_exception_marks_page_failed(tmp_path):
    p = tmp_path / "a.pdf"
    make_text_pdf(p, [""])

    def vboom(image):
        raise ConnectionError("model unavailable")

    fk = Fakes(ocr=("", 0.0))
    r = extract_pdf(p, ocr_page=fk.ocr, vlm_page=vboom,
                    render=lambda a, b: Image.new("RGB", (200, 200), "gray"))
    assert r.pages[0].method == "failed"


def test_blank_page_is_skipped_without_ocr_or_vlm(tmp_path):
    p = tmp_path / "a.pdf"
    make_text_pdf(p, [""])
    fk = Fakes()
    r = extract_pdf(p, ocr_page=fk.ocr, vlm_page=fk.vlm,
                    render=lambda a, b: Image.new("RGB", (200, 200), "white"))
    assert r.pages[0].method == "blank" and fk.ocr_calls == 0 and fk.vlm_calls == 0


def test_unreadable_file_is_reported_not_raised(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"")
    r = extract_pdf(p, ocr_page=Fakes().ocr)
    assert r.pages == [] and r.error and r.text == ""
    p2 = tmp_path / "bad2.pdf"
    p2.write_bytes(b"%PDF-1.4 not really")
    assert extract_pdf(p2, ocr_page=Fakes().ocr).error


def test_cache_hit_gives_same_result_without_new_calls(tmp_path):
    p = tmp_path / "a.pdf"
    make_text_pdf(p, [LONG, ""])
    cache = tmp_path / "cache"
    fk = Fakes()
    r1 = run(p, fk, cache)
    calls = fk.ocr_calls
    r2 = run(p, fk, cache)
    assert fk.ocr_calls == calls
    assert r2.text == r1.text and [x.method for x in r2.pages] == [x.method for x in r1.pages]
    assert len(list(cache.glob("*.json"))) == 1


def test_concurrent_extraction_of_identical_files_does_not_collide(tmp_path):
    """Identical PDFs in different folders are extracted by different workers at once."""
    from concurrent.futures import ThreadPoolExecutor
    files = []
    for i in range(12):
        f = tmp_path / f"d{i}" / "same.pdf"
        f.parent.mkdir()
        make_text_pdf(f, [LONG, LONG])
        files.append(f)
    cache = tmp_path / "cache"
    fk = Fakes()
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda f: extract_pdf(f, ocr_page=fk.ocr, cache_dir=cache), files))
    assert all(r.error is None and r.text for r in res)
    assert len({r.text for r in res}) == 1
    assert len(list(cache.glob("*.tmp"))) == 0


def test_cache_key_changes_with_file_content(tmp_path):
    cache = tmp_path / "cache"
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    make_text_pdf(a, [LONG])
    make_text_pdf(b, [LONG + " altro"])
    fk = Fakes()
    run(a, fk, cache)
    run(b, fk, cache)
    assert len(list(cache.glob("*.json"))) == 2


def test_summary_counts_methods(tmp_path):
    p = tmp_path / "a.pdf"
    make_text_pdf(p, [LONG, "", ""])
    fk = Fakes(ocr=("ok " * 80, 90.0))
    r = run(p, fk)
    s = page_method_summary([r])
    assert s == {"text_layer": 1, "tesseract": 2}


@pytest.mark.skipif(not extract.tools_available(), reason="Tesseract/Poppler not installed")
def test_real_tesseract_reads_italian_scan(tmp_path):
    p = tmp_path / "scan.pdf"
    make_image_pdf(p, ["Direzione generale per le politiche di sviluppo|Incarico dirigenziale conferito il 3 marzo 2019|"
                       "Ufficio del personale e della formazione"])
    r = extract_pdf(p, vlm_page=None)
    assert r.pages[0].method in ("tesseract", "tesseract_low")
    low = r.text.lower()
    assert "direzione" in low and "incarico" in low and "2019" in low
    assert r.pages[0].conf is not None and r.pages[0].conf > 60


@pytest.mark.skipif(not extract.tools_available(), reason="Tesseract/Poppler not installed")
def test_real_blank_scan_page_is_blank(tmp_path):
    p = tmp_path / "scan.pdf"
    make_image_pdf(p, ["Pagina con testo di prova abbastanza lungo per l'OCR|seconda riga di prova", ""])
    r = extract_pdf(p, vlm_page=None)
    assert r.pages[1].method == "blank"


# ---- dates -------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("Roma, 3 marzo 2019", date(2019, 3, 3)),
    ("conferito con decreto del 16 novembre 2016 n. 25", date(2016, 11, 16)),
    ("in data 05/07/2021 il dirigente", date(2021, 7, 5)),
    ("dal 1° gennaio 2020 al 31/12/2022", date(2022, 12, 31)),   # latest wins
    ("anno 2018", date(2018, 1, 1)),
    ("ai sensi della legge 6 novembre 2012, n. 190. Roma, 3 marzo 2019", date(2019, 3, 3)),
    ("D.Lgs. 8 aprile 2013 n. 39 e d.p.r. 16/04/2013 - firmato il 12/05/2021", date(2021, 5, 12)),
    ("decreto legislativo 14 marzo 2013, n. 33; Roma 10/01/2024", date(2024, 1, 10)),
    ("incarico dal 2021 fino al 31/12/2029, firmato il 15 gennaio 2022", date(2022, 1, 15)),  # future end date ignored
    ("laurea nel 2004, master nel 2007, ultimo incarico 2019", date(2019, 1, 1)),
])
def test_extract_date_formats(text, expected):
    assert extract.extract_date(text, today=date(2026, 10, 20)) == expected


def test_extract_date_only_statutory_dates_falls_back_to_none_or_year():
    t = "ai sensi della legge 6 novembre 2012, n. 190"
    assert extract.extract_date(t, today=date(2026, 10, 20)) is None


def test_extract_date_none_when_absent_or_implausible():
    assert extract.extract_date("nessuna data qui") is None
    assert extract.extract_date("riferimento 1899") is None
    assert extract.extract_date("tel 0812 3456 2999") is None


def test_extract_date_prefers_filename_year_over_none():
    assert extract.extract_date("senza date", filename="CV_2017.pdf") == date(2017, 1, 1)


# ---- html --------------------------------------------------------------------
def test_clean_html_text_removes_navigation_and_markup():
    html = """<html><head><style>.a{}</style><script>var x=1;</script></head><body>
    <nav>Home | Contatti | Mappa del sito</nav><header>Portale</header>
    <main><h1>Nome Cognome</h1><p>Dirigente <b>responsabile</b> dell'ufficio.</p></main>
    <footer>Privacy cookie</footer></body></html>"""
    t = extract.clean_html_text(html)
    assert "Dirigente responsabile dell'ufficio." in t and "Nome Cognome" in t
    for junk in ("Home", "Contatti", "Privacy", "var x", ".a{}", "<"):
        assert junk not in t


# ---- spreadsheets --------------------------------------------------------------------------------
def make_xlsx(path, sheets):
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    wb.save(path)


def test_spreadsheet_rows_become_header_value_lines(tmp_path):
    p = tmp_path / "a.xlsx"
    make_xlsx(p, {"Foglio1": [["Anno", "Voce", "Importo"], [2022, "Stipendio", 51000.5], [None, None, None],
                              [2023, "Retribuzione di risultato", 12000]],
                  "Foglio2": [["Ente", "Ruolo"], ["Direzione X", "Dirigente"]]})
    text, reason = extract.spreadsheet_to_text(p)
    assert reason is None
    lines = text.splitlines()
    assert lines[0].endswith("Foglio1") or lines[0].startswith("Foglio1")
    assert "Anno: 2022; Voce: Stipendio; Importo: 51000.5" in lines
    assert "Anno: 2023; Voce: Retribuzione di risultato; Importo: 12000" in lines
    assert "Ente: Direzione X; Ruolo: Dirigente" in lines
    assert lines.index("Anno: 2022; Voce: Stipendio; Importo: 51000.5") < lines.index("Ente: Direzione X; Ruolo: Dirigente")
    assert not any(not ln.strip() for ln in lines)  # empty rows skipped


def test_spreadsheet_with_many_rows_is_excluded(tmp_path):
    p = tmp_path / "big.xlsx"
    make_xlsx(p, {"S": [["Voce", "Valore"]] + [[f"v{i}", i] for i in range(250)]})
    text, reason = extract.spreadsheet_to_text(p)
    assert text == "" and reason == "too_many_rows"


def test_spreadsheet_listing_several_people_is_excluded(tmp_path):
    p = tmp_path / "multi.xlsx"
    make_xlsx(p, {"S": [["Cognome", "Nome", "Importo"], ["Rossi", "Mario", 1], ["Bianchi", "Anna", 2]]})
    text, reason = extract.spreadsheet_to_text(p)
    assert text == "" and reason == "multiple_people"


def test_spreadsheet_one_person_in_name_columns_is_kept(tmp_path):
    p = tmp_path / "one.xlsx"
    make_xlsx(p, {"S": [["Cognome", "Nome", "Anno", "Importo"], ["Rossi", "Mario", 2021, 5], ["Rossi", "Mario", 2022, 6]]})
    text, reason = extract.spreadsheet_to_text(p)
    assert reason is None and "Cognome: Rossi; Nome: Mario; Anno: 2022; Importo: 6" in text


def test_spreadsheet_unreadable_reports_reason(tmp_path):
    p = tmp_path / "bad.xlsx"
    p.write_bytes(b"PK not an xlsx")
    text, reason = extract.spreadsheet_to_text(p)
    assert text == "" and reason.startswith("read_error")
