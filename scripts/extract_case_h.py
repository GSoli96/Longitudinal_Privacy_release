"""Unpack the archive of Case H into ``data/raw/caseH`` with neutral folder names (D49, 2026-10-29).

The archive has two top-level folders: one with ``<person>/*.pdf`` and one with a folder of ``<person>.html`` pages
(plus a few files next to it). They become ``downloads/<person>/*.pdf`` and ``profiles/<person>.html``, the layout of
Case A, so that ``corpus.SubTreeSource`` reads it without a ``layout.json``. Files that are not in those two places are
kept in ``excluded_units/`` and never read. The person folder names are personal data: they stay in ``data/raw``
(git-ignored); this script prints only counts.

    python -I scripts/extract_case_h.py <archive.zip> [--dest data/raw/caseH]

Checks: no path leaves the destination, every ``.pdf`` starts with ``%PDF-`` (a saved error page with a ``.pdf``
extension is counted, not dropped, since the loader decides), nothing is overwritten (``downloads``, ``profiles`` and
``excluded_units`` must not exist yet; the archive itself may sit in the destination)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def _safe(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or any(":" in part for part in p.parts):
        raise ValueError("unsafe path in archive")
    return p


def extract(archive: Path, dest: Path) -> dict:
    names = ("downloads", "profiles", "excluded_units")
    if any((dest / x).exists() for x in names):
        raise FileExistsError("already extracted: extraction never overwrites")
    z = zipfile.ZipFile(archive)
    files = [i for i in z.infolist() if not i.is_dir()]
    by_top: dict[str, list] = {}
    for i in files:
        by_top.setdefault(_safe(i.filename).parts[0], []).append(i)
    pdf_tops = [t for t, v in by_top.items() if any(i.filename.lower().endswith(".pdf") for i in v)]
    html_tops = [t for t, v in by_top.items() if t not in pdf_tops and any(i.filename.lower().endswith(".html") for i in v)]
    if len(pdf_tops) != 1 or len(html_tops) != 1 or len(by_top) != 2:
        raise ValueError(f"expected one folder of PDFs and one of pages, found {len(pdf_tops)} and {len(html_tops)}")
    # the folder of pages: the sub-folder that holds the person pages (the biggest one)
    sub = Counter(_safe(i.filename).parts[1] for i in by_top[html_tops[0]] if len(_safe(i.filename).parts) >= 3)
    page_dir = sub.most_common(1)[0][0] if sub else None
    n = Counter()
    tmp = dest / ".partial_extract"
    if tmp.exists():
        raise FileExistsError("a partial extraction exists: remove it by hand after looking at it")
    for i in files:
        parts = _safe(i.filename).parts
        if parts[0] == pdf_tops[0] and len(parts) >= 3:
            target = tmp / "downloads" / Path(*parts[1:]); n["pdf_files"] += 1
        elif parts[0] == html_tops[0] and len(parts) == 3 and parts[1] == page_dir:
            target = tmp / "profiles" / parts[2]; n["profile_pages"] += 1
        else:
            target = tmp / "excluded_units" / Path(*parts); n["excluded_files"] += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(i) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        if parts[0] == pdf_tops[0] and parts[-1].lower().endswith(".pdf"):
            with open(target, "rb") as fh:
                if b"%PDF-" not in fh.read(1024):
                    n["pdf_without_header"] += 1
    persons = [p for p in (tmp / "downloads").iterdir() if p.is_dir()]
    pages = {p.stem for p in (tmp / "profiles").glob("*.html")}
    n["person_folders"] = len(persons)
    n["pages_without_folder"] = len(pages - {p.name for p in persons})
    n["folders_without_page"] = len({p.name for p in persons} - pages)
    for x in names:
        if (tmp / x).exists():
            (tmp / x).rename(dest / x)
    tmp.rmdir()
    return dict(n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive")
    ap.add_argument("--dest", default=str(ROOT / "data" / "raw" / "caseH"))
    a = ap.parse_args(argv)
    print(json.dumps(extract(Path(a.archive), Path(a.dest))))


if __name__ == "__main__":
    sys.exit(main())
