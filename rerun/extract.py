"""Text extraction from documents: PDF pages (text layer / Tesseract / local vision model),
document dates, HTML cleaning.

Page-level method (plan T2, decided 2026-10-20):

* ``text_layer``  the PDF text layer has at least ``MIN_TEXT_LAYER_CHARS`` characters;
* ``blank``       no text layer and the rendered page is an empty white sheet (nothing to read);
* ``tesseract``   OCR gives at least ``OCR_MIN_CHARS`` characters with a mean word
                  confidence of at least ``OCR_MIN_CONF``;
* ``vlm``         Tesseract failed that test and the local vision model transcribed the page
                  verbatim (temperature 0); the page image never leaves the PC;
* ``tesseract_low`` Tesseract failed and the vision model is not enabled: the OCR text is kept
                  and flagged;
* ``failed``      nothing could be read (the vision model returned no text).

Results are cached per file (sha256 of the bytes) under the case's private directory.
OCR and the vision model are injectable so that the logic is tested without them.
"""
from __future__ import annotations

import base64
import glob
import hashlib
import io
import json
import os
import re
import shutil
import threading
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

MIN_TEXT_LAYER_CHARS = 30
OCR_MIN_CHARS = 100
OCR_MIN_CONF = 60.0
OCR_DPI = 300
OCR_LANG = "ita"
OCR_TIMEOUT = 180  # seconds per page
VLM_MODEL = "gemma3:12b"
VLM_MAX_SIDE = 1800
CACHE_VERSION = "v1"

VLM_PROMPT = (
    "Transcribe verbatim all the text visible in this page image (the language is Italian). "
    "Keep the line breaks. Do not correct, translate, summarise, explain or complete anything, "
    "and do not add comments. If the page contains no readable text, answer exactly: EMPTY"
)


# ---- tools -----------------------------------------------------------------------------------
def tesseract_cmd() -> str | None:
    env = os.environ.get("TESSERACT_CMD")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return default if os.path.isfile(default) else None


def tessdata_dir() -> str | None:
    cands = [os.environ.get("TESSDATA_DIR"), str(Path.home() / "tessdata"),
             r"C:\Program Files\Tesseract-OCR\tessdata"]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, OCR_LANG + ".traineddata")):
            return c
    return None


def poppler_path() -> str | None:
    env = os.environ.get("POPPLER_PATH")
    if env and os.path.isdir(env):
        return env
    found = shutil.which("pdftoppm")
    if found:
        return os.path.dirname(found)
    pat = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages",
                       "*Poppler*", "**", "pdftoppm.exe")
    hits = glob.glob(pat, recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    # MiKTeX ships the Poppler tools (main PC, 2026-10-23)
    mik = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "MiKTeX", "miktex", "bin", "x64")
    return mik if os.path.isfile(os.path.join(mik, "pdftoppm.exe")) else None


def tools_available() -> bool:
    return bool(tesseract_cmd() and tessdata_dir() and poppler_path())


def _setup_tesseract():
    import pytesseract
    cmd, data = tesseract_cmd(), tessdata_dir()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    if data:  # TESSDATA_PREFIX avoids the quoting problem of --tessdata-dir on Windows
        os.environ["TESSDATA_PREFIX"] = data
    return pytesseract


# ---- result types ----------------------------------------------------------------------------
@dataclass
class Page:
    n: int
    method: str
    text: str = ""
    conf: float | None = None
    layer_chars: int = 0
    tesseract_chars: int | None = None


@dataclass
class DocText:
    pages: list[Page] = field(default_factory=list)
    error: str | None = None
    sha: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(p.text.strip() for p in self.pages if p.text.strip())

    def to_json(self) -> dict:
        return {"pages": [asdict(p) for p in self.pages], "error": self.error, "sha": self.sha}

    @classmethod
    def from_json(cls, d: dict) -> "DocText":
        return cls(pages=[Page(**p) for p in d["pages"]], error=d.get("error"), sha=d.get("sha", ""))


def page_method_summary(docs) -> dict:
    c = Counter(p.method for d in docs for p in d.pages)
    return dict(c)


# ---- default OCR / render / VLM ---------------------------------------------------------------
def _fix_orientation(image, pytesseract):
    try:
        osd = pytesseract.image_to_osd(image, timeout=60)
        rot = int(re.search(r"Rotate: (\d+)", osd).group(1))
        if rot:
            return image.rotate(-rot, expand=True)
    except Exception:  # noqa: BLE001 - OSD fails on pages with little text: keep the image
        pass
    return image


def ocr_page_default(image, lang=None) -> tuple[str, float]:
    """Tesseract on one page image -> (text, mean word confidence)."""
    pytesseract = _setup_tesseract()
    img = _fix_orientation(image, pytesseract)
    d = pytesseract.image_to_data(img, lang=lang or OCR_LANG, output_type=pytesseract.Output.DICT,
                                  timeout=OCR_TIMEOUT)
    lines, confs, order = {}, [], []
    for i, word in enumerate(d["text"]):
        if not word.strip():
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(word)
        c = float(d["conf"][i])
        if c >= 0:
            confs.append(c)
    text = "\n".join(" ".join(lines[k]) for k in order)
    return text, (sum(confs) / len(confs) if confs else 0.0)


def render_default(path, n, dpi=OCR_DPI):
    from pdf2image import convert_from_path
    imgs = convert_from_path(str(path), dpi, first_page=n, last_page=n, poppler_path=poppler_path())
    return imgs[0]


def is_blank(image) -> bool:
    from PIL import ImageStat
    g = image.convert("L")
    st = ImageStat.Stat(g)
    return st.mean[0] > 240 and st.stddev[0] < 5


def make_vlm_page(client, model=VLM_MODEL):
    """Page transcriber on top of the LLM client (local model, temperature 0)."""
    def vlm_page(image) -> str:
        img = image.convert("RGB")
        if max(img.size) > VLM_MAX_SIDE:
            img.thumbnail((VLM_MAX_SIDE, VLM_MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        r = client.chat(model, [{"role": "user", "content": VLM_PROMPT, "images": [b64]}],
                        {"num_predict": 4096})
        text = (r.text or "").strip() if r.ok else ""
        return "" if text.upper() == "EMPTY" else text
    return vlm_page


# ---- PDF ---------------------------------------------------------------------------------------
def _long(path) -> str:
    p = os.path.abspath(str(path))
    return "\\\\?\\" + p if os.name == "nt" and not p.startswith("\\\\?\\") else p


def _read_bytes(path) -> bytes:
    with open(_long(path), "rb") as f:
        return f.read()


def extract_pdf(path, *, ocr_page=None, vlm_page=None, render=None, cache_dir=None,
                dpi=OCR_DPI) -> DocText:
    """Extract the text of a PDF page by page (see module docstring for the methods)."""
    ocr_page = ocr_page or ocr_page_default
    render = render or (lambda p, n: render_default(p, n, dpi))
    try:
        data = _read_bytes(path)
    except OSError as e:
        return DocText(error=f"read_error: {type(e).__name__}")
    sha = hashlib.sha256(data).hexdigest()
    cache_file = None
    if cache_dir:
        tag = f"{CACHE_VERSION}-d{dpi}-{'vlm' if vlm_page else 'novlm'}"
        cache_file = Path(cache_dir) / f"{sha}-{tag}.json"
        if cache_file.exists():
            try:
                return DocText.from_json(json.loads(cache_file.read_text(encoding="utf-8")))
            except (ValueError, KeyError, TypeError):
                pass
    result = _extract_bytes(path, data, sha, ocr_page, vlm_page, render)
    if cache_file and result.error is None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # unique temporary name: identical PDFs (same sha) are processed by several workers
        tmp = cache_file.with_suffix(f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(result.to_json(), ensure_ascii=False), encoding="utf-8")
        try:
            tmp.replace(cache_file)
        except OSError:  # another worker finished the same file first: its copy is identical
            tmp.unlink(missing_ok=True)
    return result


def _extract_bytes(path, data, sha, ocr_page, vlm_page, render) -> DocText:
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        n_pages = len(reader.pages)
    except Exception as e:  # noqa: BLE001 - corrupt, empty or encrypted file
        return DocText(error=f"pdf_error: {type(e).__name__}", sha=sha)
    out = DocText(sha=sha)
    for i in range(n_pages):
        n = i + 1
        try:
            layer = (reader.pages[i].extract_text() or "").strip()
        except Exception:  # noqa: BLE001
            layer = ""
        if len(layer) >= MIN_TEXT_LAYER_CHARS:
            out.pages.append(Page(n, "text_layer", layer, layer_chars=len(layer)))
            continue
        try:
            image = render(path, n)
        except Exception as e:  # noqa: BLE001 - cannot render: nothing more to try
            out.pages.append(Page(n, "failed", layer, layer_chars=len(layer)))
            out.error = out.error or f"render_error: {type(e).__name__}"
            continue
        if is_blank(image):
            out.pages.append(Page(n, "blank", "", layer_chars=len(layer)))
            continue
        try:
            text, conf = ocr_page(image)
        except Exception:  # noqa: BLE001 - OCR crash or timeout: the page falls to the next method
            text, conf = "", 0.0
        text = text.strip()
        if len(text) >= OCR_MIN_CHARS and conf >= OCR_MIN_CONF:
            out.pages.append(Page(n, "tesseract", text, conf, len(layer), len(text)))
            continue
        if vlm_page is None:
            out.pages.append(Page(n, "tesseract_low", text, conf, len(layer), len(text)))
            continue
        try:
            v = (vlm_page(image) or "").strip()
        except Exception:  # noqa: BLE001 - model unavailable: the page stays unread
            v = ""
        out.pages.append(Page(n, "vlm" if v else "failed", v, conf, len(layer), len(text)))
    return out


# ---- spreadsheets ------------------------------------------------------------------------------
SHEET_MAX_ROWS = 200
_NAME_HEADERS = ("nome", "cognome", "nominativo", "name", "surname", "dipendente")


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    if hasattr(v, "isoformat"):
        return v.isoformat()[:10]
    return str(v).strip()


def _sheet_rows(path):
    """[(sheet name, [[cell strings]])] for .xlsx / .xls, or raises."""
    low = str(path).lower()
    if low.endswith((".xlsx", ".xlsm")):
        import openpyxl
        wb = openpyxl.load_workbook(_long(path), read_only=True, data_only=True)
        return [(ws.title, [[_cell(c) for c in row] for row in ws.iter_rows(values_only=True)])
                for ws in wb.worksheets]
    import xlrd
    wb = xlrd.open_workbook(_long(path))
    return [(sh.name, [[_cell(c.value) for c in sh.row(r)] for r in range(sh.nrows)]) for sh in wb.sheets()]


def spreadsheet_to_text(path) -> tuple[str, str | None]:
    """Spreadsheet -> text, one line per row as "header: value; ..." (sheets in order, empty
    rows skipped). Returns ("", reason) when the file must not enter a person's profile:
    ``too_many_rows`` (more than SHEET_MAX_ROWS rows) or ``multiple_people`` (a name/surname
    column with more than one distinct value: a list of people, not a personal file)."""
    try:
        sheets = _sheet_rows(path)
    except Exception as e:  # noqa: BLE001
        return "", f"read_error: {type(e).__name__}"
    lines = []
    total = 0
    for name, rows in sheets:
        rows = [r for r in rows if any(c for c in r)]
        total += len(rows)
        if total > SHEET_MAX_ROWS:
            return "", "too_many_rows"
        if not rows:
            continue
        header = [h or f"col{i + 1}" for i, h in enumerate(rows[0])]
        lines.append(name)
        name_cols = [i for i, h in enumerate(header) if h.casefold().strip() in _NAME_HEADERS]
        for i in name_cols:
            vals = {r[i].casefold() for r in rows[1:] if i < len(r) and r[i]}
            if len(vals) > 1:
                return "", "multiple_people"
        for r in rows[1:]:
            cells = [f"{header[i] if i < len(header) else f'col{i + 1}'}: {c}" for i, c in enumerate(r) if c]
            if cells:
                lines.append("; ".join(cells))
    return "\n".join(lines), None


# ---- dates -------------------------------------------------------------------------------------
_MONTHS = {"gennaio": 1, "gen": 1, "febbraio": 2, "feb": 2, "marzo": 3, "mar": 3, "aprile": 4,
           "apr": 4, "maggio": 5, "mag": 5, "giugno": 6, "giu": 6, "luglio": 7, "lug": 7,
           "agosto": 8, "ago": 8, "settembre": 9, "set": 9, "sett": 9, "ottobre": 10, "ott": 10,
           "novembre": 11, "nov": 11, "dicembre": 12, "dic": 12}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_YEAR_MIN = 1990
_RE_NUM = re.compile(r"(?<!\d)(0?[1-9]|[12]\d|3[01])\s*[/.\-]\s*(0?[1-9]|1[0-2])\s*[/.\-]\s*((?:19|20)\d{2})(?!\d)")
_RE_ISO = re.compile(r"(?<!\d)((?:19|20)\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?!\d)")
_RE_TXT = re.compile(r"(?<!\d)(0?[1-9]|[12]\d|3[01])\s*[°ºo]?\s+(" + _MONTH_RE + r")\.?\s+((?:19|20)\d{2})(?!\d)",
                     re.IGNORECASE)
_RE_YEAR = re.compile(r"(?<![\d.,/-])((?:19|20)\d{2})(?![\d/-])")


def _mk(y, m, d, today):
    try:
        if _YEAR_MIN <= y <= today.year:
            dt = date(y, m, d)
            return dt if dt <= today else None
    except ValueError:
        pass
    return None


# statutory acts are cited with their own date (e.g. "legge 6 novembre 2012, n. 190"): that
# date is not the date of the document that cites it
_LEGAL = re.compile(r"(legge|l\.|d\.?\s?lgs\.?|decreto[\s-]+legislativo|decreto[\s-]+legge|d\.?l\.|"
                    r"d\.?\s?p\.?\s?r\.?|dpr|regio decreto|r\.d\.)\s*(n\.?\s*\d+[\s,/]*)?(del|dell')?\s*$",
                    re.IGNORECASE)


def _is_legal_citation(text, start):
    return bool(_LEGAL.search(text[max(0, start - 40):start]))


def find_date(text: str, filename: str | None = None, today: date | None = None):
    """Document date as (date, kind) with kind in {'full', 'year', 'filename'}, or None.

    Among the complete dates of the text, those inside citations of statutes (legge, D.Lgs.,
    D.P.R.) are discarded and the latest one not after ``today`` wins (the signing or issue
    date follows the references to laws, and future end dates of a mandate are ignored). If
    there is none, the latest plausible year of the text; then a year in the file name."""
    today = today or date.today()
    cands, spans = [], []
    for rx, build in (
        (_RE_TXT, lambda m: _mk(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)), today)),
        (_RE_NUM, lambda m: _mk(int(m.group(3)), int(m.group(2)), int(m.group(1)), today)),
        (_RE_ISO, lambda m: _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)), today)),
    ):
        for m in rx.finditer(text):
            spans.append((m.start(), m.end()))  # the year of a full date is not a bare year
            d = build(m)
            if d and not _is_legal_citation(text, m.start()):
                cands.append(d)
    if cands:
        return max(cands), "full"
    years = [d for d in (_mk(int(m.group(1)), 1, 1, today) for m in _RE_YEAR.finditer(text)
                         if not any(a <= m.start() < b for a, b in spans)) if d]
    if years:
        return max(years), "year"
    if filename:
        fy = [d for d in (_mk(int(m.group(1)), 1, 1, today) for m in _RE_YEAR.finditer(filename)) if d]
        if fy:
            return max(fy), "filename"
    return None


def extract_date(text: str, filename: str | None = None, today: date | None = None):
    r = find_date(text, filename, today)
    return r[0] if r else None


# ---- HTML --------------------------------------------------------------------------------------
_DROP = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe",
         "svg", "button", "select"]
_BLOCK = ["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
          "table", "ul", "ol", "dd", "dt", "blockquote"]


def clean_html_text(html: str) -> str:
    """Visible text of the content, without navigation, scripts, styles or markup."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(_DROP):
        t.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for t in soup.find_all(_BLOCK):
        t.append("\n")
    for td in soup.find_all(["td", "th"]):
        td.append(" ")
    text = soup.get_text("")
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.splitlines()]
    out, blank = [], False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank and out:
            out.append("")
            blank = True
    return "\n".join(out).strip()
