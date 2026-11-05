"""Corpus: per-case loaders -> ``Person(id, case, role, org_type, docs[])``.

Every case is read through a *source* with the same interface (``SOURCES``), so that the
cases whose documents are not available yet (A, B, D, H: ``MISSING_CASES``) only need a
source to be added.

Rules (plan T2, spec section 4):
* the text comes from the documents only, re-extracted from the files (never from the
  enriched HTML pages of the old pipeline), full length;
* persons without any usable text are kept with ``status="no_text"``, never dropped;
* person ids are opaque (``C001``...); the map id -> source name stays in
  ``data/private/<case>/persons_map.json``;
* organisation type comes from ``data/private/<case>/case.json`` (never in tracked files);
* spreadsheets are excluded and counted unless ``include_spreadsheets`` is set;
* files whose names differ only by case (Case E, ``__<hash>`` suffix) are compared by
  text: near-identical variants collapse to one document, different ones stay distinct.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import statistics
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from . import config, extract

MISSING_CASES: dict = {}   # H was here until its archive arrived (D49, 2026-10-29); the mechanism stays for new cases
SIMILAR_RATIO = 0.97
SPREADSHEET_EXT = (".xlsx", ".xls", ".xlsm", ".ods")
_HASH_SUFFIX = re.compile(r"__[0-9a-f]{8}(?=\.[^.]+$|$)")


# ---- data model --------------------------------------------------------------------------------
@dataclass
class Doc:
    id: str
    date: date | None
    kind: str
    text: str
    date_kind: str | None = None
    pages: int = 0
    methods: dict = field(default_factory=dict)
    error: str | None = None
    filename: str = ""
    source: str = ""  # path of the file (personal data: only in memory and in data/private)
    page_methods: dict = field(default_factory=dict)  # page number -> method, pages not read from a text layer


@dataclass
class Person:
    id: str
    case: str
    role: str | None
    org_type: str | None
    docs: list[Doc]
    status: str = "ok"
    meta: dict = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return sum(len(d.text) for d in self.docs)


@dataclass
class RawFile:
    path: Path | None
    filename: str
    kind: str = "pdf"
    text: str | None = None  # for sources that already hold the text (Case F)


@dataclass
class RawPerson:
    key: str
    files: list[RawFile]
    role: str | None = None
    meta: dict = field(default_factory=dict)


# ---- helpers -----------------------------------------------------------------------------------
def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t.replace("\x00", ""))
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", t)
    return t.strip()


def _walk(root: Path):
    base = extract._long(root)
    for d, _, fs in os.walk(base):
        for f in sorted(fs):
            yield Path(d[4:] if d.startswith("\\\\?\\") else d) / f


def _subdirs(root: Path):
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold())


def _strip_hash(name: str) -> str:
    return _HASH_SUFFIX.sub("", name)


def _collision_key(name: str) -> str:
    return unicodedata.normalize("NFC", _strip_hash(name)).casefold()


def _similar(a: str, b: str) -> bool:
    a, b = re.sub(r"\s+", " ", a).strip(), re.sub(r"\s+", " ", b).strip()
    if not a or not b:
        return a == b
    if a == b:
        return True
    if abs(len(a) - len(b)) / max(len(a), len(b)) > 0.1:
        return False
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= SIMILAR_RATIO


def _read_layout(private_case_dir: Path) -> dict:
    """Folder names of a case that identify its organisation are not in the code: they are read from
    ``data/private/case<X>/layout.json`` (keys: ``downloads``, ``profiles``, ``docs``). Absent: neutral defaults."""
    f = Path(private_case_dir) / "layout.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _html_text(path: Path) -> str:
    """Visible text of a saved portal page (D50). UTF-8 first; the older pages are cp1252."""
    data = Path(path).read_bytes()
    try:
        html = data.decode("utf-8")
    except UnicodeDecodeError:
        html = data.decode("cp1252", errors="replace")
    return norm_text(extract.clean_html_text(html))


def _is_pdf(path: Path) -> bool:
    """A saved error page of a portal is often stored with the ``.pdf`` extension (D51)."""
    with open(extract._long(Path(path)), "rb") as fh:
        return b"%PDF-" in fh.read(1024)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(extract._long(Path(path)), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- sources -----------------------------------------------------------------------------------
class FileTreeSource:
    """``root/<person>/**/*.pdf`` (Cases C and E)."""

    def __init__(self, case, raw_dir, layout=None):
        self.case = case
        self.root = Path(raw_dir) / f"case{case}"
        self.report: dict = {}

    def people(self):
        out = []
        for pdir in _subdirs(self.root):
            files = []
            for f in _walk(pdir):
                low = f.name.lower()
                if low.endswith(".pdf"):
                    files.append(RawFile(f, f.name, "pdf"))
                elif low.endswith(SPREADSHEET_EXT):
                    files.append(RawFile(f, f.name, "spreadsheet"))
            out.append(RawPerson(pdir.name, files))
        return out


class CaseFSource:
    """Texts already inside ``database_ai_ready.json`` (16 persons)."""

    def __init__(self, case, raw_dir, layout=None):
        self.case = case
        self.path = Path(raw_dir) / f"case{case}" / "database_ai_ready.json"
        self.report: dict = {}

    def people(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        out = []
        for name in sorted(data, key=str.casefold):
            files = [RawFile(None, d.get("filename", ""), d.get("tipo") or "doc", d.get("testo") or "")
                     for d in data[name].get("documenti", [])]
            out.append(RawPerson(name, files, meta={"score": data[name].get("score")}))
        return out


class CaseGSource:
    """``docs/SECTION/ENTITY/**.pdf`` (profiles: one html page per entity or section).

    One RawPerson per (section, entity); the loader merges the same entity found in several
    sections when the texts support it. Files placed directly in a section, and the profile
    pages of sections, are units of the portal, not persons: they are excluded and counted."""

    def __init__(self, case, raw_dir, layout=None):
        self.case = case
        self.root = Path(raw_dir) / f"case{case}"
        self.report: dict = {}

    def people(self):
        down = self.root / "docs"
        prof = self.root / "profiles"
        html_names = {p.stem for p in prof.glob("*.html")} if prof.exists() else set()
        out, units, person_pages = [], set(), set()
        for sec in _subdirs(down):
            if any(f.is_file() for f in sec.iterdir()):
                units.add(sec.name)
            for ent in _subdirs(sec):
                person_pages.add(f"{sec.name}__{ent.name}")
                files = [RawFile(f, f.name, "pdf") for f in _walk(ent) if f.name.lower().endswith(".pdf")]
                out.append(RawPerson(f"{sec.name}__{ent.name}", files,
                                     meta={"entity": ent.name, "section": sec.name}))
        unit_pages = (html_names - person_pages) | units
        self.report = {"profiles_html": len(html_names), "excluded_units": len(unit_pages)}
        return out


class SubTreeSource(FileTreeSource):
    """``root/<downloads>/<person>/**/*.pdf`` plus ``root/<profiles>/<person>.html`` (Case A).

    D50 (2026-10-25, applied 2026-10-29): the saved profile pages are documents like the PDFs, read with
    ``extract.clean_html_text``; a person with a profile page and no PDF is a person (38 of 288 in the
    original archive). The two folder names come from ``layout.json`` of the case (keys ``downloads``,
    ``profiles``), never from the code."""

    def __init__(self, case, raw_dir, layout=None):
        super().__init__(case, raw_dir)
        lay = layout or {}
        self.profiles = self.root / lay.get("profiles", "profiles")
        self.root = self.root / lay.get("downloads", "downloads")

    def people(self):
        out = super().people()
        pages = {p.stem: p for p in sorted(self.profiles.glob("*.html"))} if self.profiles.exists() else {}
        by_key = {rp.key.casefold(): rp for rp in out}
        profile_only = 0
        for stem, path in pages.items():
            rp = by_key.get(stem.casefold())
            if rp is None:
                rp = RawPerson(stem, [])
                out.append(rp)
                by_key[stem.casefold()] = rp
                profile_only += 1
            rp.files.append(RawFile(path, path.name, "html", _html_text(path)))
        out.sort(key=lambda rp: rp.key.casefold())
        self.report = {"profiles_html": len(pages), "persons_from_profile_only": profile_only}
        return out


class CaseBSource:
    """``.../PDF_SEPARATI/NAME_YYYY-MM-DD[_n].pdf`` grouped by NAME (Case B).

    The other PDFs of the download folder are the multi-person originals before the student's
    splitting script: excluded and counted. The date in the file name (version of the CV) is
    used before the dates found in the text (``filename_dates_first``)."""

    PATTERN = re.compile(r"^(.+?)_(\d{4}-\d{2}-\d{2})(?:_\d+)?\.pdf$", re.IGNORECASE)
    filename_dates_first = True

    def __init__(self, case, raw_dir, layout=None):
        self.case = case
        self.root = Path(raw_dir) / f"case{case}"
        self.report: dict = {}

    def people(self):
        seps =[p for p in self.root.rglob("PDF_SEPARATI") if p.is_dir()]
        if len(seps) != 1:
            raise FileNotFoundError(f"Case B: expected one PDF_SEPARATI folder, found {len(seps)}")
        sep = seps[0]
        groups: dict[str, list[RawFile]] = defaultdict(list)
        unmatched = 0
        for f in _walk(sep):
            m = self.PATTERN.match(f.name)
            if m and f.parent == sep:
                groups[m.group(1)].append(RawFile(f, f.name, "pdf"))
            elif f.name.lower().endswith(".pdf"):
                unmatched += 1
        all_pdfs = sum(1 for f in _walk(self.root) if f.name.lower().endswith(".pdf"))
        in_sep = sum(len(v) for v in groups.values()) + unmatched
        self.report = {"excluded_unsplit_pdfs": all_pdfs - in_sep, "unmatched_file_names": unmatched}
        return [RawPerson(k, sorted(v, key=lambda r: r.filename)) for k, v in sorted(groups.items(),
                                                                                       key=lambda kv: kv[0].casefold())]


class CaseDSource:
    """``<docs>/<person>/**/*.pdf`` (Case D; the folder name comes from ``layout.json``, key ``docs``).

    From the acts in the pooled folders (``_determinazioni``, ``_bandi``) a person receives only the pages
    where all of the person's name tokens appear as whole words (decisions 2026-10-20 and 2026-10-23: whole
    acts of 67,000-272,000 characters made profiles of up to 1.2 million characters); ``_unmatched`` is
    excluded. Needs ``cache_dir`` (set by the loader) to read the pooled acts. Since 2026-10-29:
    * D51: a file with the ``.pdf`` extension that is not a PDF (a saved portal error page) is not a document: it
      is counted in ``not_pdf``; a person left without files stays, with no text;
    * D52: a profile page (key ``profiles``, default ``doc``) without a person folder makes a person, made of
      that page and of the pooled pages that name them; without any such page the person is excluded and counted;
    * D53: a file found in the folders of several persons (same hash) is treated as a pooled act: each of them
      receives only the pages with their name."""

    POOLED = ("_determinazioni", "_bandi")

    def __init__(self, case, raw_dir, layout=None):
        lay = layout or {}
        self.case = case
        self.root = Path(raw_dir) / f"case{case}" / lay.get("docs", "docs")
        self.profiles_name = lay.get("profiles", "doc")
        self.excluded = (self.profiles_name, "_unmatched")
        self.report: dict = {}
        self.cache_dir: Path | None = None

    @staticmethod
    def _tokens(key: str) -> list[str]:
        return [_plain(t) for t in re.split(r"[_\s\-']+", key) if len(_plain(t)) >= 2]

    @staticmethod
    def _has_name(tokens: list[str], plain_text: str) -> bool:
        return bool(tokens) and all(re.search(rf"\b{re.escape(t)}\b", plain_text) for t in tokens)

    def people(self):
        persons, pooled = [], []
        not_pdf, had_pdf_name = 0, set()
        for d in _subdirs(self.root):
            if d.name in self.excluded:
                continue
            pdfs = []
            for f in _walk(d):
                if not f.name.lower().endswith(".pdf"):
                    continue
                had_pdf_name.add(d.name)
                if _is_pdf(f):
                    pdfs.append(RawFile(f, f.name, "pdf"))
                else:
                    not_pdf += 1
            if d.name in self.POOLED:
                pooled.extend(pdfs)
            else:
                persons.append(RawPerson(d.name, pdfs))
        only_non_pdf = sum(1 for rp in persons if not rp.files and rp.key in had_pdf_name)

        # D53: the same file in the folders of several persons is a pooled act for each of them
        by_hash = defaultdict(list)          # sha256 -> [(person index, RawFile)]
        for i, rp in enumerate(persons):
            for f in rp.files:
                by_hash[_sha256(f.path)].append((i, f))
        shared_files: dict[int, list] = defaultdict(list)    # person index -> its copies (sha, RawFile)
        shared_shas = set()
        for sha, items in by_hash.items():
            if len({i for i, _ in items}) > 1:
                shared_shas.add(sha)
                for i, f in items:
                    shared_files[i].append((sha, f))
        for i, copies in shared_files.items():
            drop = {id(f) for _, f in copies}
            persons[i].files = [f for f in persons[i].files if id(f) not in drop]

        pages = {}   # sha256 or path -> [(original page text, plain page text)]

        def pages_of(key, path):
            if key not in pages:
                dt = extract.extract_pdf(path, vlm_page=None, cache_dir=self.cache_dir)
                pages[key] = [(p.text, _plain(p.text or "")) for p in dt.pages if (p.text or "").strip()]
            return pages[key]

        given, n_pages, chars = Counter(), 0, 0
        shared_given, shared_pages = Counter(), 0

        def add_pages(rp, toks, sources, tally, count_shared):
            nonlocal n_pages, chars, shared_pages
            found = 0
            for key, f in sources:
                hit = [orig for orig, plain in pages_of(key, f.path) if self._has_name(toks, plain)]
                if hit:
                    text = "\n\n".join(h.strip() for h in hit)
                    rp.files.append(RawFile(f.path, f.filename, "pdf_pages", text))
                    tally[key] += 1
                    found += 1
                    if count_shared:
                        shared_pages += len(hit)
                    else:
                        n_pages += len(hit)
                        chars += len(text)
            return found

        pooled_sources = [(f.path, f) for f in pooled]
        for i, rp in enumerate(persons):
            toks = self._tokens(rp.key)
            add_pages(rp, toks, pooled_sources, given, False)
            add_pages(rp, toks, shared_files.get(i, []), shared_given, True)

        # D52: profile pages without a person folder
        names = {rp.key.casefold() for rp in persons}
        profile_pages = sorted(self.root.joinpath(self.profiles_name).glob("*.html")) \
            if self.root.joinpath(self.profiles_name).exists() else []
        created, excluded_no_acts = 0, 0
        for page in profile_pages:
            if page.stem.casefold() in names:
                continue
            rp = RawPerson(page.stem, [RawFile(page, page.name, "html", _html_text(page))])
            if add_pages(rp, self._tokens(rp.key), pooled_sources, given, False):
                persons.append(rp)
                created += 1
            else:
                excluded_no_acts += 1
        self.report = {"pooled_acts": len(pooled), "pooled_acts_assigned": sum(1 for f in pooled if given[f.path]),
                       "pooled_acts_in_several_profiles": sum(1 for f in pooled if given[f.path] > 1),
                       "pooled_assignments": sum(given.values()), "pooled_pages_assigned": n_pages,
                       "pooled_chars_assigned": chars,
                       "not_pdf": not_pdf, "persons_only_not_pdf": only_non_pdf,
                       "shared_files": len(shared_shas), "shared_copies": sum(len(v) for v in shared_files.values()),
                       "shared_persons": len(shared_files), "shared_pages_assigned": shared_pages,
                       "profile_pages_without_folder": created + excluded_no_acts,
                       "persons_created_from_acts": created, "persons_excluded_no_acts": excluded_no_acts}
        return persons


SOURCES = {
    "A": SubTreeSource,
    "B": CaseBSource,
    "C": FileTreeSource,
    "D": CaseDSource,
    "E": FileTreeSource,
    "F": CaseFSource,
    "G": CaseGSource,
    "H": SubTreeSource,   # D49 (2026-10-29): same shape as A, ``downloads/<person>/*.pdf`` and ``profiles/<person>.html``
}


# ---- extraction -------------------------------------------------------------------------------
def _extract_task(args):
    path, cache_dir = args
    return extract.extract_pdf(path, vlm_page=None, cache_dir=cache_dir)


def _extract_all(files, *, cache_dir, ocr_page, vlm_page, workers, tracker=None):
    """PDF files -> DocText, in order. Workers run the text layer and Tesseract; the pages
    that Tesseract fails are re-done in this process through the vision model."""
    results = {}
    if ocr_page is None and workers > 1 and files:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for f, r in zip(files, ex.map(_extract_task, [(f, cache_dir) for f in files], chunksize=4)):
                results[f] = r
                if tracker is not None:
                    tracker.advance()
    else:
        for f in files:
            results[f] = extract.extract_pdf(f, ocr_page=ocr_page, vlm_page=None, cache_dir=cache_dir)
            if tracker is not None:
                tracker.advance()
    if vlm_page is not None:
        for f, r in list(results.items()):
            if any(p.method == "tesseract_low" for p in r.pages):
                results[f] = extract.extract_pdf(f, ocr_page=ocr_page, vlm_page=vlm_page, cache_dir=cache_dir)
    return results


_RE_NAME_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _filename_date(filename: str):
    m = _RE_NAME_DATE.search(filename or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _doc_from(f: RawFile, idx: int, pid: str, dt: extract.DocText | None, filename_first: bool = False) -> Doc:
    if f.kind == "spreadsheet":
        t, reason = extract.spreadsheet_to_text(f.path)
        text, pages, methods, err = norm_text(t), 0, {}, reason
    elif dt is not None:
        text, pages, methods, err = norm_text(dt.text), len(dt.pages), dict(Counter(p.method for p in dt.pages)), dt.error
    else:
        text, pages, methods, err = norm_text(f.text or ""), 0, {}, None
    fd = _filename_date(f.filename) if filename_first else None
    found = (fd, "filename") if fd else extract.find_date(text, f.filename)
    pm = ({p.n: p.method for p in dt.pages if p.method != "text_layer"}
          if dt is not None and f.kind != "spreadsheet" else {})
    return Doc(id=f"{pid}-d{idx:02d}", date=found[0] if found else None, kind=f.kind, text=text,
               date_kind=found[1] if found else None, pages=pages, methods=methods, error=err,
               filename=f.filename, source=str(f.path) if f.path else "", page_methods=pm)


def _collapse_collisions(docs: list[Doc], stats: Counter) -> list[Doc]:
    groups: dict[str, list[Doc]] = defaultdict(list)
    for d in docs:
        groups[_collision_key(d.filename)].append(d)
    drop = set()
    for key, ds in groups.items():
        if len(ds) < 2 or not any(_HASH_SUFFIX.search(x.filename) for x in ds):
            continue
        stats["groups"] += 1
        keep = [ds[0]]
        for d in ds[1:]:
            if any(_similar(d.text, k.text) for k in keep):
                drop.add(id(d))
            else:
                keep.append(d)
        if len(keep) == 1:
            stats["similar_kept_one"] += 1
        else:
            stats["different_kept_both"] += 1
    return [d for d in docs if id(d) not in drop]


def _order(docs: list[Doc], chronological: bool = True) -> list[Doc]:
    """Chronological by document date; undated documents go last in their original order
    (the sort is stable). Case G keeps the order of the portal sections."""
    if not chronological:
        return list(docs)
    return sorted(docs, key=lambda d: (d.date is None, d.date or date.max))


# ---- ids and merging ---------------------------------------------------------------------------
def _salt(private_dir: Path) -> str:
    f = Path(private_dir) / "id_salt.txt"
    if not f.exists():
        Path(private_dir).mkdir(parents=True, exist_ok=True)
        f.write_text(secrets.token_hex(16), encoding="utf-8")
    return f.read_text(encoding="utf-8").strip()


def assign_ids(case: str, keys, salt: str, lengths=(6, 8, 10)) -> dict:
    """Stable pseudonymous ids for a case: case + the first N hex digits of the salted hash of
    the source key. N is the same for every person of the case: 6, or the next length if two
    persons would collide. The salt (data/private/id_salt.txt) is the only way to regenerate
    the same ids: never delete it."""
    keys = list(keys)
    for n in lengths:
        ids = {k: f"{case}-" + hashlib.sha256(f"{salt}|{case}|{k}".encode("utf-8")).hexdigest()[:n]
               for k in keys}
        if len(set(ids.values())) == len(keys):
            return ids
    raise AssertionError(f"case {case}: ids still collide with {lengths[-1]} hex digits")


def _plain(t: str) -> str:
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii").casefold()


def _name_in_texts(entity: str, docs: list) -> bool:
    """True if the entity's name (at least two of its name tokens) appears in the texts."""
    toks = [_plain(t) for t in entity.replace("-", "_").split("_") if len(_plain(t)) >= 3]
    if not toks:
        return False
    text = _plain(" ".join(d.text for d in docs))
    need = min(2, len(toks))
    return sum(1 for t in toks if t in text) >= need


# ---- public ------------------------------------------------------------------------------------
def load_case_with_report(case, *, raw_dir=config.RAW, private_dir=config.PRIVATE,
                          ocr_page=None, vlm_page=None, workers=1, include_spreadsheets=True,
                          progress_run: str | None = None):
    """``progress_run``: name of the dashboard entry of the text extraction (None: no entry, as in tests)."""
    if case in MISSING_CASES:
        raise FileNotFoundError(f"Case {case}: {MISSING_CASES[case]}")
    priv = Path(private_dir) / f"case{case}"
    priv.mkdir(parents=True, exist_ok=True)
    src = SOURCES[case](case, raw_dir, _read_layout(priv))
    org_type = None
    cj = priv / "case.json"
    if cj.exists():
        org_type = json.loads(cj.read_text(encoding="utf-8")).get("org_type")

    if hasattr(src, "cache_dir"):
        src.cache_dir = priv / "text"
    raw_people = src.people()
    filename_first = getattr(src, "filename_dates_first", False)
    pdfs = list(dict.fromkeys(f.path for rp in raw_people for f in rp.files if f.kind == "pdf" and f.path))
    if progress_run:
        from . import progress
        with progress.Tracker(progress_run, "text extraction", len(pdfs), unit="PDF",
                              next_step="inference") as tr:
            extracted = _extract_all(pdfs, cache_dir=priv / "text", ocr_page=ocr_page, vlm_page=vlm_page,
                                     workers=workers, tracker=tr)
    else:
        extracted = _extract_all(pdfs, cache_dir=priv / "text", ocr_page=ocr_page, vlm_page=vlm_page,
                                 workers=workers)

    rep = Counter()
    flagged = Counter()
    items = []  # dicts: key, role, meta, docs, sheet (bool)
    for rp in raw_people:
        docs, sheet = [], False
        for j, f in enumerate(rp.files, 1):
            if f.kind == "spreadsheet" and not include_spreadsheets:
                rep["spreadsheets_excluded"] += 1
                continue
            d = _doc_from(f, j, "tmp", extracted.get(f.path) if f.path else None, filename_first)
            if f.kind == "spreadsheet" and d.error in ("too_many_rows", "multiple_people"):
                flagged[d.error] += 1  # not a personal file: kept out of the profile
                continue
            if f.kind == "spreadsheet" and not d.error:
                rep["spreadsheets_included"] += 1
                rep["spreadsheet_chars"] += len(d.text)
                sheet = True
            docs.append(d)
        items.append({"key": rp.key, "role": rp.role, "meta": dict(rp.meta), "docs": docs, "sheet": sheet})

    # same entity in several sections: one person if the name is in the texts of every section
    merged_log, doubtful_log = {}, {}
    by_entity = defaultdict(list)
    for it in items:
        if it["meta"].get("entity"):
            by_entity[it["meta"]["entity"]].append(it)
    for ent, group in by_entity.items():
        if len(group) < 2:
            continue
        if all(_name_in_texts(ent, g["docs"]) for g in group):
            merged = {"key": ent, "role": None, "docs": [d for g in group for d in g["docs"]],
                      "sheet": any(g["sheet"] for g in group),
                      "meta": {"entity": ent, "sections": len(group)}}
            items = [it for it in items if it not in group] + [merged]
            merged_log[ent] = [g["meta"].get("section") for g in group]
        else:
            doubtful_log[ent] = [g["meta"].get("section") for g in group]
    items.sort(key=lambda it: it["key"].casefold())

    persons = []
    salt = _salt(priv.parent)
    pmap = assign_ids(case, [it["key"] for it in items], salt)
    collisions = Counter()
    with_sheet = 0
    for it in items:
        pid = pmap[it["key"]]
        docs = _order(_collapse_collisions(it["docs"], collisions), chronological=(case != "G"))
        for k, d in enumerate(docs, 1):
            d.id = f"{pid}-d{k:02d}"
        status = "ok" if sum(len(d.text) for d in docs) > 0 else "no_text"
        persons.append(Person(pid, case, it["role"], org_type, docs, status, it["meta"]))
        rep["unreadable_files"] += sum(1 for d in docs if d.error)
        rep["no_text"] += status == "no_text"
        with_sheet += any(d.kind == "spreadsheet" for d in docs)

    (priv / "persons_map.json").write_text(json.dumps(pmap, ensure_ascii=False, indent=1), encoding="utf-8")
    if merged_log or doubtful_log:
        (priv / "merged_people.json").write_text(
            json.dumps({"merged": merged_log, "kept_separate_doubtful": doubtful_log}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    alldocs = [d for p in persons for d in p.docs]
    per_person = [p.chars for p in persons]
    per_doc = [len(d.text) for d in alldocs]
    pages = Counter()
    for d in alldocs:
        pages.update(d.methods)

    def spread(xs):
        return {"min": min(xs), "median": statistics.median(xs), "max": max(xs)} if xs else {}

    report = {
        "case": case, "persons": len(persons), "docs": len(alldocs),
        "chars_per_person": spread(per_person), "chars_per_doc": spread(per_doc),
        "no_text": rep["no_text"], "unreadable_files": rep["unreadable_files"],
        "spreadsheets_excluded": rep["spreadsheets_excluded"],
        "spreadsheets_included": rep["spreadsheets_included"],
        "spreadsheet_persons": with_sheet, "spreadsheet_chars": rep["spreadsheet_chars"],
        "spreadsheets_flagged": dict(flagged),
        "pages_by_method": dict(pages),
        "dates": {k: sum(1 for d in alldocs if (d.date_kind or "none") == k) for k in ("full", "year", "filename", "none")},
        "collisions": dict(collisions) if collisions else {"groups": 0, "similar_kept_one": 0, "different_kept_both": 0},
        **src.report,
    }
    if case == "G":
        report["merged_people"] = len(merged_log)
        report["merge_doubtful_kept_separate"] = len(doubtful_log)
    return persons, report


def load_case(case, **kw) -> list[Person]:
    return load_case_with_report(case, **kw)[0]
