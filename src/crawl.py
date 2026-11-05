"""Stage 1 - Public knowledge extraction (crawling).

Walks a set of local HTML index pages of a transparency portal, follows the
document links they contain and downloads the attached files (typically
PDFs), grouping them per individual.

The crawl TARGET is a configuration placeholder (`base_url`, `html_index_dir`).
Point it only at portals you are authorised to access. The pipeline uses
exclusively information that administrations publish under mandatory-
transparency obligations; it does not bypass any authentication or access
control.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .config import CaseConfig


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "research-crawler/1.0"})
    return s


def extract_links(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Return (url, filename) for the downloadable links inside tables."""
    links: list[tuple[str, str]] = []
    for a in soup.select("table a[href]"):
        href = a.get("href", "")
        if not href or href.startswith("#"):
            continue
        url = href if href.startswith("http") else urljoin(base_url, href)
        if "nomefile=" in url:
            filename = url.split("nomefile=")[-1]
        else:
            filename = a.get_text(strip=True).replace(" ", "_") + ".bin"
        links.append((url, filename))
    return links


def download(session: requests.Session, url: str, dest: Path) -> None:
    if dest.exists():
        return
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)


def crawl(cfg: CaseConfig) -> None:
    session = _session()
    index_dir = Path(cfg.html_index_dir)
    out_root = Path(cfg.downloads_dir)
    for html_path in sorted(index_dir.glob("*.html")):
        person = html_path.stem  # anonymous identifier derived from the index
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        first = next(
            (
                a["href"] if a["href"].startswith("http") else urljoin(cfg.base_url, a["href"])
                for a in soup.select("a[href]")
                if a.get("href") and not a["href"].startswith("#")
            ),
            None,
        )
        if not first:
            continue
        resp = session.get(first, timeout=30)
        resp.raise_for_status()
        for url, filename in extract_links(BeautifulSoup(resp.text, "html.parser"), cfg.base_url):
            download(session, url, out_root / person / filename)
