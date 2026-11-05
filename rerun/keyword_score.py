"""Keyword personalisation score of the old paper, applied to the generated phishing messages (RQ4 side analysis).

The old paper scored each individual by how many of the six words of ``PERSONALISATION_KEYWORDS``
(``src/inference.py``) appear in the attacker's scenario: band ``high`` with at least 4 words, ``medium`` with at least
2, ``low`` otherwise. The scenario of the old pipeline does not exist in the rerun, so the same rule is applied to the
generated phishing messages (label 1, variant ``explicit_inferred``) of cases A and E, for each generator.

Two matchings are reported: ``substring`` (the original rule: the lower-cased word occurs anywhere in the lower-cased
text, so "data" also matches "database") and ``word`` (whole word, case-insensitive). The text is subject and body, as
the detectors see it. Output ``results/rerun/keyword_score.json``: counts and shares only, no text.

The keywords are read from ``src/inference.py`` without importing it (the module needs the Ollama client)."""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

from . import config, detect_llm, phish_corpus

CORPUS_DIR = config.PRIVATE / "rq3_collect" / "corpus"
OUT_DIR = config.RESULTS
CASES = ("A", "E")
GENERATORS = ("gptoss", "llama31")
VARIANT = "explicit_inferred"
BANDS = ("high", "medium", "low")
INFERENCE_PY = config.ROOT / "src" / "inference.py"


def load_keywords(path=None) -> tuple[str, ...]:
    """The tuple ``PERSONALISATION_KEYWORDS`` assigned in ``src/inference.py``, read with ``ast``."""
    tree = ast.parse(Path(path or INFERENCE_PY).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PERSONALISATION_KEYWORDS"
                                                for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise ValueError("PERSONALISATION_KEYWORDS not found")


def present_substring(text: str, keywords) -> list[str]:
    low = (text or "").lower()
    return [k for k in keywords if k in low]


def present_word(text: str, keywords) -> list[str]:
    return [k for k in keywords if re.search(r"\b" + re.escape(k) + r"\b", text or "", re.IGNORECASE)]


def band(n_hits: int) -> str:
    """Same thresholds as ``src.inference.personalisation_score``."""
    return "high" if n_hits >= 4 else "medium" if n_hits >= 2 else "low"


def summarise(texts, keywords) -> dict:
    """Number of texts and, for each matching, the count and share per band, the mean number of words found and
    how many texts contain each keyword."""
    out = {"n_messages": len(texts)}
    for label, fn in (("substring", present_substring), ("word", present_word)):
        found = [fn(t, keywords) for t in texts]
        bands = Counter(band(len(f)) for f in found)
        n = len(texts)
        out[label] = {"counts": {b: bands.get(b, 0) for b in BANDS},
                      "shares": {b: (bands.get(b, 0) / n if n else None) for b in BANDS},
                      "mean_keywords_present": (sum(len(f) for f in found) / n if n else None),
                      "messages_with_keyword": {k: sum(k in f for f in found) for k in keywords}}
    return out


def select(rows, variant: str = VARIANT) -> list[dict]:
    """Phishing (label 1) rows of one variant."""
    return [r for r in rows if int(r["label"]) == 1 and r.get("variant") == variant]


def run(*, cases=CASES, generators=GENERATORS, variant=VARIANT, corpus_dir=None, out_dir=None,
        name="keyword_score", keywords=None) -> dict:
    kws = tuple(keywords or load_keywords())
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    res = {"keywords": list(kws), "variant": variant, "label": 1,
           "rule": {"high": ">=4 keywords", "medium": ">=2", "low": "otherwise"}, "corpora": {}}
    for c in cases:
        for g in generators:
            corpus = f"rq3_{c}_{g}"
            rows = select(phish_corpus.load(corpus, corpus_dir or CORPUS_DIR), variant)
            texts = [detect_llm.message_text(r) for r in rows]
            entry = summarise(texts, kws)
            entry["by_attack"] = {a: summarise([detect_llm.message_text(r) for r in rows if r["attack"] == a], kws)
                                  for a in sorted({r["attack"] for r in rows})}
            entry["generator"] = rows[0]["generator"] if rows else None
            res["corpora"][f"{c}/{g}"] = entry
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Keyword personalisation score on the generated phishing messages.")
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS))
    ap.add_argument("--corpus-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    res = run(cases=tuple(a.cases), generators=tuple(a.generators), corpus_dir=a.corpus_dir, out_dir=a.out_dir)
    for k, e in res["corpora"].items():
        print(k, e["n_messages"], "substring", e["substring"]["counts"], "word", e["word"]["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
