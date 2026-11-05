"""scripts/extract_case_h.py on a toy archive (fixtures only)."""
import importlib.util
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("extract_case_h", Path(__file__).resolve().parents[2] / "scripts" / "extract_case_h.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def make_zip(path, extra=()):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("TOPPDF/Pers1/a.pdf", b"%PDF-1.4 a")
        z.writestr("TOPPDF/Pers1/b.pdf", b"%PDF-1.4 b")
        z.writestr("TOPPDF/Pers2/a.pdf", b"<html>error page saved as pdf</html>")
        z.writestr("TOPHTML/pages/Pers1.html", "<p>x</p>")
        z.writestr("TOPHTML/pages/Pers2.html", "<p>y</p>")
        z.writestr("TOPHTML/index.html", "<p>unit</p>")
        for name, data in extra:
            z.writestr(name, data)


def test_extraction_gives_the_layout_of_case_a_and_counts(tmp_path):
    make_zip(tmp_path / "x.zip")
    dest = tmp_path / "caseH"
    dest.mkdir()
    (dest / "x.zip").write_bytes((tmp_path / "x.zip").read_bytes())      # the archive sits in the destination
    rep = mod.extract(dest / "x.zip", dest)
    assert rep == {"pdf_files": 3, "profile_pages": 2, "excluded_files": 1, "pdf_without_header": 1,
                   "person_folders": 2, "pages_without_folder": 0, "folders_without_page": 0}
    assert (dest / "downloads" / "Pers1" / "b.pdf").exists() and (dest / "profiles" / "Pers2.html").exists()
    assert (dest / "excluded_units" / "TOPHTML" / "index.html").exists() and not (dest / ".partial_extract").exists()
    with pytest.raises(FileExistsError):
        mod.extract(dest / "x.zip", dest)                                # never overwrites


def test_a_path_that_leaves_the_destination_is_refused(tmp_path):
    make_zip(tmp_path / "bad.zip", extra=[("TOPPDF/../../evil.pdf", b"%PDF-")])
    with pytest.raises(ValueError):
        mod.extract(tmp_path / "bad.zip", tmp_path / "out")
    assert not (tmp_path / "evil.pdf").exists()
