"""Public e-mail benchmarks for the RQ4 detectors (plan ``docs/plans/rq34.md`` §2 and §8 step 4).

Six corpora (CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud, SpamAssasin) are read from a source folder, cleaned and
written once to ``data/private/benchmarks/`` as one JSONL file per corpus plus a manifest. A later call reuses the
files when the SHA-256 of the source CSV and ``CLEAN_VERSION`` are unchanged; nothing is read twice. Cleaning is
deliberately minimal: whitespace at the ends is stripped, rows without a body or with a label other than 0/1 are
dropped, and exact repeats of (subject, body) count once. The manifest keeps counts only, no text.

Label convention: 1 = phishing or spam, 0 = legitimate. Nazario and Nigerian_Fraud contain positives only; the
detectors take their negatives from the other corpora."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from . import config

CLEAN_VERSION = 1
BENCHMARKS = {
    "CEAS_08": "CEAS_08.csv",
    "Enron": "Enron.csv",
    "Ling": "Ling.csv",
    "Nazario": "Nazario.csv",
    "Nigerian_Fraud": "Nigerian_Fraud.csv",
    "SpamAssasin": "SpamAssasin.csv",  # the file name has a single "s" in the middle
}
SOURCE_ENV = "LP_BENCHMARK_DIR"
OUT_DIR = config.PRIVATE / "benchmarks"
MANIFEST = "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_dir(explicit: str | os.PathLike | None = None) -> Path:
    """The folder with the original CSVs: argument first, then the environment variable ``LP_BENCHMARK_DIR``."""
    p = explicit or os.environ.get(SOURCE_ENV)
    if not p:
        raise FileNotFoundError(f"no benchmark folder: pass it or set {SOURCE_ENV}")
    return Path(p)


def _uid(name: str, subject: str, body: str) -> str:
    return hashlib.sha256(f"{name}\x1f{subject}\x1f{body}".encode("utf-8")).hexdigest()[:16]


def clean_csv(name: str, path: Path) -> tuple[list[dict], Counter]:
    """Rows ``{uid, benchmark, subject, body, label}`` of one corpus and the counters of what was dropped."""
    counts: Counter = Counter()
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            counts["read"] += 1
            body = (rec.get("body") or "").strip()
            subject = (rec.get("subject") or "").strip()
            label = (rec.get("label") or "").strip()
            if label not in ("0", "1"):
                counts["dropped_label"] += 1
                continue
            if not body:
                counts["dropped_empty_body"] += 1
                continue
            key = (subject, body)
            if key in seen:
                counts["dropped_duplicate"] += 1
                continue
            seen.add(key)
            rows.append({"uid": _uid(name, subject, body), "benchmark": name, "subject": subject, "body": body,
                         "label": int(label)})
    counts["kept"] = len(rows)
    counts["kept_positive"] = sum(r["label"] for r in rows)
    return rows, counts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_manifest(out_dir: Path) -> dict:
    p = out_dir / MANIFEST
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def prepare(src: str | os.PathLike | None = None, out_dir: str | os.PathLike | None = None,
            names: list[str] | None = None) -> dict:
    """Clean the corpora once; reuse the files whose source hash and cleaning version did not change.
    Returns the manifest (counts, hashes and a ``status`` per corpus: ``built`` or ``reused``)."""
    src_dir = source_dir(src)
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(out)
    entries = dict(manifest.get("corpora", {}))
    status = {}
    for name in names or list(BENCHMARKS):
        if name not in BENCHMARKS:
            raise KeyError(f"unknown benchmark {name!r}")
        csv_path = src_dir / BENCHMARKS[name]
        if not csv_path.exists():
            raise FileNotFoundError(f"{BENCHMARKS[name]} not found in the benchmark folder")
        digest = sha256_file(csv_path)
        prev = entries.get(name)
        target = out / f"{name}.jsonl"
        if prev and prev.get("source_sha256") == digest and prev.get("clean_version") == CLEAN_VERSION \
                and target.exists():
            status[name] = "reused"
            continue
        rows, counts = clean_csv(name, csv_path)
        _write_jsonl(target, rows)
        entries[name] = {"source_file": BENCHMARKS[name], "source_sha256": digest, "clean_version": CLEAN_VERSION,
                         "counts": dict(counts), "output_sha256": sha256_file(target)}
        status[name] = "built"
    result = {"clean_version": CLEAN_VERSION, "corpora": entries}
    tmp = out / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out / MANIFEST)
    return {**result, "status": status}


def load(name: str, out_dir: str | os.PathLike | None = None) -> list[dict]:
    """The cleaned rows of one corpus. Fails if ``prepare`` has not run or if the file changed since."""
    out = Path(out_dir) if out_dir else OUT_DIR
    entry = _read_manifest(out).get("corpora", {}).get(name)
    path = out / f"{name}.jsonl"
    if not entry or not path.exists():
        raise FileNotFoundError(f"benchmark {name!r} not prepared: run rerun.benchmarks.prepare first")
    if sha256_file(path) != entry["output_sha256"]:
        raise ValueError(f"benchmark {name!r}: the file does not match the manifest")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summary(manifest: dict) -> str:
    """One line per corpus with counts only."""
    lines = []
    for name, e in sorted(manifest.get("corpora", {}).items()):
        c = e["counts"]
        lines.append(f"{name}: read {c.get('read', 0)}, kept {c.get('kept', 0)} (positives {c.get('kept_positive', 0)}),"
                     f" dropped empty body {c.get('dropped_empty_body', 0)}, duplicates {c.get('dropped_duplicate', 0)},"
                     f" bad label {c.get('dropped_label', 0)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Clean the public e-mail benchmarks once and reuse them.")
    ap.add_argument("--src", help=f"folder with the original CSVs (default: ${SOURCE_ENV})")
    ap.add_argument("--out", help="output folder (default: data/private/benchmarks)")
    args = ap.parse_args(argv)
    manifest = prepare(args.src, args.out)
    print(summary(manifest))
    print("status:", manifest["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
