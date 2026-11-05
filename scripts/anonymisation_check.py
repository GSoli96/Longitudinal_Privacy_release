#!/usr/bin/env python3
"""Scan files for personal data and administration identifiers before release.

The scanner is a safety net, not a proof of anonymity: a clean run means that
none of the patterns below matched, nothing more. Every data file must also go
through the manual sample review described in docs/ANONYMISATION.md.

What it flags
-------------
high    EMAIL, PHONE_IT, CODICE_FISCALE, IBAN, SECRET, FORBIDDEN_TERM,
        KNOWN_NAME, MODEL_FILE
medium  URL, DOMAIN, PARTITA_IVA, DOB_CONTEXT, ADDRESS, HONORIFIC_NAME,
        UNSCANNED_DOCUMENT
review  NER_PER / NER_ORG / NER_LOC (only with --ner), NAME_LIKE (only with
        --name-heuristics), IMAGE

File names and directory names are scanned as well as file contents. JSON and
notebook files are parsed, so dictionary keys are checked too.

Local, never-committed inputs
-----------------------------
--forbidden-file FILE  one term per line (names, acronyms, domains, cities of
                       the eight administrations, programme codes, ...).
                       Lines starting with '#' are comments. Matching ignores
                       case, accents and separators (space, _, -, ., ').
                       Terms of 6+ characters are also matched inside longer
                       tokens (e.g. inside a file name written in CamelCase).
--names-file FILE      one real person name per line, built locally from the
                       source data. Matched in the given order and with the
                       surname first ("Mario Rossi" also matches "ROSSI_MARIO").
Both files must stay outside the repository or be listed in .gitignore.

Output never shows matched text unless --reveal is given; previews are masked
(first letter of each word kept). Do not paste --reveal output anywhere public.

Exit status: 0 = no high/medium findings, 1 = findings, 2 = usage error.
With --strict, review-level findings also make the run fail.

Examples
--------
  python scripts/anonymisation_check.py data/ results/ \
      --forbidden-file ../private/forbidden_terms.txt \
      --names-file ../private/known_names.txt --ner
  python scripts/anonymisation_check.py . --summary-only
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
TEXT_EXT = {
    ".txt", ".md", ".csv", ".tsv", ".py", ".tex", ".bib", ".html", ".htm",
    ".xml", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".sh", ".ps1", ".r",
    ".sql", ".log", ".env", ".example", ".gitignore", ".cff", ".rst", "",
}
JSON_EXT = {".json", ".ipynb", ".jsonl", ".ndjson"}
MODEL_EXT = {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".bin", ".safetensors",
             ".h5", ".onnx", ".ckpt", ".npy", ".npz"}
DOC_EXT = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".odt", ".ods", ".p7m",
           ".zip", ".7z", ".tar", ".gz", ".rar", ".eml", ".msg"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".tif", ".tiff", ".bmp",
             ".webp", ".eps"}
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "venv311", "node_modules",
             ".mypy_cache", ".pytest_cache"}

# hosts that may legitimately appear in a public research repository
DEFAULT_ALLOWED_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "github.com", "raw.githubusercontent.com",
    "zenodo.org", "doi.org", "dx.doi.org", "arxiv.org", "huggingface.co",
    "kaggle.com", "www.kaggle.com", "ollama.com", "pypi.org", "python.org",
    "www.python.org", "docs.python.org", "spacy.io", "creativecommons.org",
    "opensource.org", "spdx.org", "example.org", "example.com", "example.net",
    "scikit-learn.org", "pytorch.org", "orcid.org", "example",
    # sources cited in docs/ANONYMISATION.md (none is one of the eight administrations)
    "eur-lex.europa.eu", "edpb.europa.eu", "ec.europa.eu", "curia.europa.eu",
    "normattiva.it", "garanteprivacy.it", "anticorruzione.it", "codiceappalti.it",
    "agendadigitale.eu", "nvlpubs.nist.gov", "csrc.nist.gov", "aclanthology.org",
    "academic.oup.com", "ico.org.uk", "help.zenodo.org", "presidio.dataprivacystack.org",
    "microsoft.github.io", "hunton.com", "cliffordchance.com",
}
# placeholders and pseudonyms produced by the transformation rules
DEFAULT_ALLOW_PATTERNS = [
    r"Case[A-H]-[POE]\d{3,5}(?:-S\d-A\d{1,3})?",
    r"\[(?:PERSON|NAME|SURNAME|ORG|ADMINISTRATION|UNIT|ROLE|LOCATION|CITY|"
    r"ADDRESS|EMAIL|PHONE|URL|DATE|YEAR|ID|PROGRAMME|LAW|UNIVERSITY|AMOUNT)"
    r"(?:_\d+)?\]",
    r"[\w.+-]+@example\.(?:org|com|net)",
    r"[\w.+-]+@[\w.-]+\.example",  # reserved TLD used for surrogate addresses (RFC 2606)
]
GENERIC_SALUTATION_WORDS = {
    "cliente", "clienti", "collega", "colleghi", "utente", "utenti", "dipendente",
    "dipendenti", "dirigente", "dirigenti", "responsabile", "direttore",
    "direttrice", "signore", "signora", "signori", "tutti", "tutte", "team",
    "customer", "user", "users", "sir", "madam", "colleague", "colleagues",
    "staff", "personale", "amministrazione", "ufficio", "settore", "servizio",
    "sottoscritto", "sottoscritta", "candidato", "candidata", "funzionario",
    "funzionaria", "referente", "segreteria", "beneficiario", "titolare",
    "dottore", "dottoressa", "professore", "professoressa", "ingegnere", "avvocato",
}

# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------
RE_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}(?![\w-])"
    r"|\b[\w.+-]+\s*[\[(]\s*(?:at|chiocciola)\s*[\])]\s*[\w-]+(?:\s*[\[(]\s*(?:dot|punto)\s*[\])]\s*[\w-]+)+",
    re.IGNORECASE,
)
RE_PHONE = re.compile(
    r"(?<![\w.,/:-])(?:(?:\+|00)\s?39[\s.\-/]?)?(?:3\d{2}|0\d{1,4})(?:[\s.\-/]?\d){5,8}(?![\w]|[.,]\d)"
)
RE_CF = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{6}[0-9LMNPQRSTUVlmnpqrstuv]{2}[ABCDEHLMPRSTabcdehlmprst]"
    r"[0-9LMNPQRSTUVlmnpqrstuv]{2}[A-Za-z][0-9LMNPQRSTUVlmnpqrstuv]{3}[A-Za-z](?![A-Za-z0-9])"
)
RE_IBAN = re.compile(r"\bIT\s?\d{2}\s?[A-Z](?:\s?[0-9A-Z]){22}\b", re.IGNORECASE)
RE_PIVA = re.compile(
    r"(?i)\b(?:p\.?\s?iva|partita\s+iva|vat(?:\s+(?:no|number))?|c\.?\s?f\.?|codice\s+fiscale)"
    r"\s*[:.\-]?\s*(?:IT)?\s?(\d{11})\b"
)
RE_DATE = re.compile(r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}")
RE_URL = re.compile(r"\b(?:https?|ftp)://[^\s<>\"'`)\]}]+|\bwww\.[^\s<>\"'`)\]}]+", re.IGNORECASE)
RE_DOMAIN = re.compile(
    r"(?<![\w@/.-])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:gov\.it|it|eu)(?![\w-])",
    re.IGNORECASE,
)
RE_DOB = re.compile(
    r"(?i)\b(?:data\s+di\s+nascita|luogo\s+di\s+nascita|nat[oa]\s+(?:a|il|in|nel)\b|"
    r"date\s+of\s+birth|place\s+of\s+birth|born\s+(?:on|in)\b|d\.o\.b\.?)"
)
RE_ADDRESS = re.compile(
    r"(?:\b(?i:via|viale|v\.le|piazza|p\.zza|piazzale|corso|c\.so|largo|vicolo|contrada|"
    r"c\.da|localit[aà]|loc\.|strada|lungomare|lungotevere|salita|traversa)\s+"
    r"(?:[A-ZÀ-Ý][\w'’àèéìòù.]*\s+){0,5}?[A-ZÀ-Ý0-9][\w'’àèéìòù.]*,?\s*(?:n\.?\s*)?\d{1,4}[a-zA-Z]?\b)"
    r"|(?:\b\d{5}\s+[A-ZÀ-Ý][a-zà-ù]+(?:\s+[A-ZÀ-Ý][a-zà-ù]+)?\s*\(?[A-Z]{2}\)?(?![\w]))"
)
RE_HONORIFIC = re.compile(
    r"(?<![\w])(?:Dott\.?(?:ssa)?|Dr\.?|Prof\.?(?:ssa)?|Ing\.|Avv\.|Arch\.|Geom\.|Rag\.|"
    r"Sig\.?(?:ra|na)?|On\.|Gentile|Gent\.(?:mo|ma|le)?|Gentilissim[oa]|Egregi[oa]|Egr\.|"
    r"Car[oa]|Mr\.?|Mrs\.?|Ms\.?|Dear)\s+"
    r"(?:(?:Dott\.?(?:ssa)?|Dr\.?|Prof\.?(?:ssa)?|Ing\.|Avv\.|Sig\.?(?:ra)?)\s+)?"
    r"([A-ZÀ-Ý][\w'’àèéìòù]+(?:\s+[A-ZÀ-Ý][\w'’àèéìòù]+){0,3})"
)
RE_NAME_LIKE = re.compile(
    r"\b[A-ZÀ-Ý]{3,20}(?:['’][A-ZÀ-Ý]{2,20})?(?:[ _][A-ZÀ-Ý]{3,20}){1,3}\b"
    r"|\b[A-ZÀ-Ý][a-zà-ù]{2,20}_[A-ZÀ-Ý][a-zà-ù]{2,20}(?:_[A-ZÀ-Ý][a-zà-ù]{2,20}){0,2}\b"
)
RE_SECRETS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9_\-.=]{20,}"),
    re.compile(r"\b(?:ghp|gho|ghs|github_pat|glpat|hf|xox[abp])_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|apikey|access[_-]?token|auth[_-]?token|secret(?:[_-]?key)?|"
        r"password|passwd|token)\b[\"']?\s*[:=]\s*[\"']([^\"'\s]{12,})[\"']"
    ),
]
# Absolute paths of a user's home folder (Windows, Linux, macOS): they reveal the machine and the user name.
RE_LOCAL_PATH = re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+[^\\/\s\"'<>]+|(?<![\w.])/home/[a-z_][\w.-]*|(?<![\w.:])/Users/[A-Za-z][^/\s\"'<>]*")
SECRET_PLACEHOLDER = re.compile(
    r"(?i)sostituire|your|tua|xxx|example|placeholder|changeme|<|>|\.\.\.|environ|getenv|\{|\}"
)

CF_ODD = {
    **dict(zip(string.digits, [1, 0, 5, 7, 9, 13, 15, 17, 19, 21])),
    **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
               [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14,
                16, 10, 22, 25, 24, 23])),
}


def cf_checksum_ok(cf: str) -> bool:
    cf = cf.upper()
    total = 0
    for i, ch in enumerate(cf[:15]):
        if i % 2 == 0:  # positions 1, 3, 5, ... (1-based) are "odd"
            total += CF_ODD[ch]
        else:
            total += int(ch) if ch.isdigit() else ord(ch) - 65
    return chr(65 + total % 26) == cf[15]


def piva_checksum_ok(num: str) -> bool:
    if len(num) != 11 or not num.isdigit() or num == "0" * 11:
        return False
    s = 0
    for i, ch in enumerate(num[:10]):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return (10 - s % 10) % 10 == int(num[10])


def phone_ok(match: str) -> bool:
    digits = re.sub(r"\D", "", match)
    if digits.startswith("0039"):
        digits = digits[4:]
    elif match.strip().startswith("+39") or (digits.startswith("39") and len(digits) >= 11):
        digits = digits[2:]
    if digits.startswith("3"):
        return 9 <= len(digits) <= 10
    if digits.startswith("0"):
        return 6 <= len(digits) <= 11
    return False


# --------------------------------------------------------------------------
# term matching (forbidden terms, known names)
# --------------------------------------------------------------------------
def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def tokens_of(text: str) -> list[str]:
    text = fold(text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)       # camelCase
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)      # ABCDef -> ABC Def
    return re.findall(r"[a-z0-9]+", text.lower())


class TermIndex:
    """Exact n-gram lookup on normalised tokens; long single tokens also as substrings."""

    def __init__(self) -> None:
        self.by_len: dict[int, set[tuple[str, ...]]] = collections.defaultdict(set)
        self.substrings: set[str] = set()

    def add(self, term: str, both_orders: bool = False) -> None:
        toks = tuple(tokens_of(term))
        if not toks:
            return
        self.by_len[len(toks)].add(toks)
        if both_orders and len(toks) >= 2:
            self.by_len[len(toks)].add(toks[1:] + toks[:1])    # surname(s) first
            self.by_len[len(toks)].add(toks[-1:] + toks[:-1])
        compact = "".join(toks)
        if len(compact) >= 6 and not both_orders:
            self.substrings.add(compact)

    def __bool__(self) -> bool:
        return bool(self.by_len) or bool(self.substrings)

    def find(self, text: str) -> list[str]:
        toks = tokens_of(text)
        hits = []
        for n, terms in self.by_len.items():
            for i in range(len(toks) - n + 1):
                gram = tuple(toks[i:i + n])
                if gram in terms:
                    hits.append(" ".join(gram))
        if self.substrings:
            for t in toks:
                for s in self.substrings:
                    if s in t and t != s:
                        hits.append(t)
        return hits


def load_terms(paths: list[str], extra: list[str], both_orders: bool) -> TermIndex:
    idx = TermIndex()
    for p in paths or []:
        with open(p, encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    idx.add(line, both_orders)
    for t in extra or []:
        idx.add(t, both_orders)
    return idx


# --------------------------------------------------------------------------
# findings
# --------------------------------------------------------------------------
SEVERITY = {
    "EMAIL": "high", "PHONE_IT": "high", "CODICE_FISCALE": "high", "IBAN": "high",
    "SECRET": "high", "FORBIDDEN_TERM": "high", "KNOWN_NAME": "high", "MODEL_FILE": "high",
    "URL": "medium", "DOMAIN": "medium", "PARTITA_IVA": "medium", "DOB_CONTEXT": "medium",
    "ADDRESS": "medium", "HONORIFIC_NAME": "medium", "UNSCANNED_DOCUMENT": "medium", "LOCAL_PATH": "medium",
    "NER_PER": "review", "NER_ORG": "review", "NER_LOC": "review", "NAME_LIKE": "review",
    "IMAGE": "review",
}


def mask(s: str) -> str:
    s = s.replace("\n", " ")[:80]
    return re.sub(r"(?<=[^\W\d_])[^\W\d_]+", lambda m: "*" * len(m.group()), s).translate(
        str.maketrans(string.digits, "#" * 10))


@dataclass
class Finding:
    detector: str
    location: str
    text: str


@dataclass
class FileReport:
    path: str
    findings: list[Finding] = field(default_factory=list)


class Scanner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.allowed_hosts = set(DEFAULT_ALLOWED_HOSTS)
        self.allow_res = [re.compile(p) for p in DEFAULT_ALLOW_PATTERNS]
        for p in args.allow_file or []:
            with open(p, encoding="utf-8-sig") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("host:"):
                        self.allowed_hosts.add(line[5:].strip().lower())
                    else:
                        self.allow_res.append(re.compile(line))
        self.forbidden = load_terms(args.forbidden_file, args.forbid, both_orders=False)
        self.names = load_terms(args.names_file, [], both_orders=True)
        self.nlp = None
        if args.ner:
            try:
                import spacy  # noqa: WPS433
            except ImportError:
                sys.exit("--ner needs spaCy and an Italian model (it_core_news_lg or it_core_news_sm)")
            for model in (args.ner_model, "it_core_news_lg", "it_core_news_sm"):
                if not model:
                    continue
                try:
                    self.nlp = spacy.load(model, exclude=["parser", "lemmatizer", "morphologizer",
                                                          "tagger", "attribute_ruler"])
                    break
                except OSError:
                    continue
            if self.nlp is None:
                sys.exit("--ner: no Italian spaCy model found (it_core_news_lg / it_core_news_sm)")
            self.nlp.max_length = 2_000_000
            self.ner_labels = {lab.strip().upper() for lab in args.ner_labels.split(",")}

    # -- helpers -----------------------------------------------------------
    def allowed(self, s: str) -> bool:
        return any(r.fullmatch(s.strip()) for r in self.allow_res)

    def host_allowed(self, url: str) -> bool:
        host = re.sub(r"^(?:https?|ftp)://", "", url, flags=re.I).split("/")[0].split(":")[0].lower()
        return host in self.allowed_hosts or any(host.endswith("." + h) for h in self.allowed_hosts)

    def scan_text(self, text: str, location: str, out: list[Finding], use_ner: bool = True) -> None:
        if not text:
            return
        spans: list[tuple[int, int]] = []

        def add(det: str, m_text: str, start: int = -1, end: int = -1) -> None:
            if self.allowed(m_text):
                return
            out.append(Finding(det, location, m_text))
            if start >= 0:
                spans.append((start, end))

        for m in RE_EMAIL.finditer(text):
            add("EMAIL", m.group(), m.start(), m.end())
        for m in RE_LOCAL_PATH.finditer(text):
            add("LOCAL_PATH", m.group(), m.start(), m.end())
        for m in RE_URL.finditer(text):
            if not self.host_allowed(m.group()):
                add("URL", m.group(), m.start(), m.end())
            else:
                spans.append((m.start(), m.end()))
        for m in RE_DOMAIN.finditer(text):
            if any(a <= m.start() < b for a, b in spans):
                continue
            if not self.host_allowed(m.group()):
                add("DOMAIN", m.group())
        for m in RE_CF.finditer(text):
            ok = cf_checksum_ok(m.group())
            if ok or m.group().isupper():  # OCR-damaged codes are usually upper case
                add("CODICE_FISCALE", m.group() + ("" if ok else " (checksum fails)"))
        for m in RE_IBAN.finditer(text):
            add("IBAN", m.group())
        for m in RE_PIVA.finditer(text):
            if piva_checksum_ok(m.group(1)):
                add("PARTITA_IVA", m.group(), m.start(), m.end())
        for m in RE_PHONE.finditer(text):
            if (phone_ok(m.group()) and not RE_DATE.fullmatch(m.group())
                    and not any(a <= m.start() < b for a, b in spans)):
                add("PHONE_IT", m.group())
        for m in RE_DOB.finditer(text):
            add("DOB_CONTEXT", text[max(0, m.start() - 10): m.end() + 25])
        for m in RE_ADDRESS.finditer(text):
            add("ADDRESS", m.group())
        for m in RE_HONORIFIC.finditer(text):
            first = m.group(1).split()[0].lower()
            if first not in GENERIC_SALUTATION_WORDS:
                add("HONORIFIC_NAME", m.group())
        secret_spans: list[tuple[int, int]] = []
        for rx in RE_SECRETS:
            for m in rx.finditer(text):
                val = m.group(1) if rx.groups else m.group()
                if SECRET_PLACEHOLDER.search(val):
                    continue
                if any(a < m.end() and m.start() < b for a, b in secret_spans):
                    continue
                secret_spans.append((m.start(), m.end()))
                add("SECRET", m.group())
        if self.forbidden:
            for hit in self.forbidden.find(text):
                add("FORBIDDEN_TERM", hit)
        if self.names:
            for hit in self.names.find(text):
                add("KNOWN_NAME", hit)
        if self.args.name_heuristics:
            for m in RE_NAME_LIKE.finditer(text):
                add("NAME_LIKE", m.group())
        if use_ner and self.nlp is not None and len(text) >= 3:
            doc = self.nlp(text[: self.args.ner_max_chars])
            for ent in doc.ents:
                lab = {"PER": "NER_PER", "ORG": "NER_ORG", "LOC": "NER_LOC"}.get(ent.label_)
                if lab and ent.label_ in self.ner_labels and not self.allowed(ent.text):
                    out.append(Finding(lab, location, ent.text))

    # -- file types --------------------------------------------------------
    def scan_json_obj(self, obj, pointer: str, out: list[Finding], budget: list[int]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                ks = str(k)
                self.scan_text(ks, f"json:{pointer}/<key>", out, use_ner=self._ner_budget(ks, budget))
                self.scan_json_obj(v, f"{pointer}/{mask(ks)[:24]}", out, budget)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self.scan_json_obj(v, f"{pointer}[{i}]", out, budget)
        elif isinstance(obj, str):
            self.scan_text(obj, f"json:{pointer}", out, use_ner=self._ner_budget(obj, budget))

    def _ner_budget(self, s: str, budget: list[int]) -> bool:
        if self.nlp is None or budget[0] <= 0:
            return False
        budget[0] -= len(s)
        return True

    def scan_file(self, path: str, rel: str) -> FileReport:
        rep = FileReport(rel)
        self.scan_text(rel, "<path>", rep.findings, use_ner=False)
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path).lower()
        if name.startswith(".env") and name != ".env.example":
            rep.findings.append(Finding("SECRET", "<path>", "environment file"))
        if ext in MODEL_EXT:
            rep.findings.append(Finding("MODEL_FILE", "<path>",
                                        "serialized model/array: may memorise training text"))
            return rep
        if ext in DOC_EXT:
            rep.findings.append(Finding("UNSCANNED_DOCUMENT", "<path>",
                                        "binary document: convert to text and rescan, or review by hand"))
            return rep
        if ext in IMAGE_EXT:
            rep.findings.append(Finding("IMAGE", "<path>", "image: check by eye (labels, legends, axes)"))
            return rep
        if name == ".ds_store":
            rep.findings.append(Finding("UNSCANNED_DOCUMENT", "<path>",
                                        ".DS_Store stores file names: remove it"))
            return rep
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as fh:
                raw = fh.read()
        except OSError as exc:
            rep.findings.append(Finding("UNSCANNED_DOCUMENT", "<path>", f"unreadable: {exc}"))
            return rep
        budget = [self.args.ner_max_chars]
        if ext in JSON_EXT:
            parsed = False
            if ext in {".jsonl", ".ndjson"}:
                try:
                    for i, line in enumerate(raw.splitlines(), 1):
                        if line.strip():
                            self.scan_json_obj(json.loads(line), f"L{i}", rep.findings, budget)
                    parsed = True
                except json.JSONDecodeError:
                    rep.findings.clear()
                    self.scan_text(rel, "<path>", rep.findings, use_ner=False)
            else:
                try:
                    self.scan_json_obj(json.loads(raw), "", rep.findings, budget)
                    parsed = True
                except json.JSONDecodeError:
                    pass
            if parsed:
                return rep
        if ext not in TEXT_EXT and ext not in JSON_EXT and "\x00" in raw[:4096]:
            rep.findings.append(Finding("UNSCANNED_DOCUMENT", "<path>", "unknown binary format"))
            return rep
        for i, line in enumerate(raw.splitlines(), 1):
            self.scan_text(line, f"L{i}", rep.findings, use_ner=self._ner_budget(line, budget))
        return rep


def iter_files(roots: list[str]):
    """Yield (full path or None for a directory, path shown and scanned).

    Only the part below each root is scanned as a path, so the location of the
    checkout on the local disk does not produce findings."""
    for root in roots:
        root = os.path.normpath(root)
        if os.path.isfile(root):
            yield root, os.path.basename(root)
            continue
        base = os.path.basename(os.path.abspath(root))
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            rel_dir = os.path.relpath(dirpath, root)
            for d in dirnames:  # directory names are data too
                yield None, os.path.normpath(os.path.join(base, rel_dir, d)).replace(os.sep, "/") + "/"
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                yield full, os.path.normpath(os.path.join(base, rel_dir, fn)).replace(os.sep, "/")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="files or directories to scan")
    ap.add_argument("--forbidden-file", action="append", help="local list of forbidden terms (repeatable)")
    ap.add_argument("--forbid", action="append", help="one forbidden term on the command line (repeatable)")
    ap.add_argument("--names-file", action="append", help="local list of real person names (repeatable)")
    ap.add_argument("--allow-file", action="append",
                    help="regexes of allowed strings, one per line; 'host:name' lines allow a URL host")
    ap.add_argument("--ner", action="store_true", help="also run spaCy Italian NER (slow)")
    ap.add_argument("--ner-model", default="", help="spaCy model name (default: it_core_news_lg, then _sm)")
    ap.add_argument("--ner-labels", default="PER", help="comma-separated NER labels to report (PER,ORG,LOC)")
    ap.add_argument("--ner-max-chars", type=int, default=300_000, help="NER budget per file, in characters")
    ap.add_argument("--name-heuristics", action="store_true",
                    help="flag ALL-CAPS word pairs and Name_Surname tokens (noisy)")
    ap.add_argument("--reveal", action="store_true", help="print matched text unmasked (local use only)")
    ap.add_argument("--summary-only", action="store_true", help="print counts only")
    ap.add_argument("--max-per-file", type=int, default=15, help="findings printed per file")
    ap.add_argument("--json-out", help="write a JSON report (masked unless --reveal)")
    ap.add_argument("--strict", action="store_true", help="review-level findings also fail the run")
    args = ap.parse_args(argv)

    for p in args.paths:
        if not os.path.exists(p):
            print(f"error: {p} does not exist", file=sys.stderr)
            return 2
    scanner = Scanner(args)
    totals: collections.Counter = collections.Counter()
    by_sev: collections.Counter = collections.Counter()
    reports = []
    n_files = 0
    for full, rel in iter_files(args.paths):
        if full is None:  # directory name only
            rep = FileReport(rel)
            scanner.scan_text(rel, "<path>", rep.findings, use_ner=False)
        else:
            n_files += 1
            rep = scanner.scan_file(full, rel)
        if not rep.findings:
            continue
        reports.append(rep)
        counts = collections.Counter(f.detector for f in rep.findings)
        totals.update(counts)
        for det, c in counts.items():
            by_sev[SEVERITY[det]] += c
        if not args.summary_only:
            path_flagged = any(f.location == "<path>" and SEVERITY[f.detector] != "review"
                               and f.detector not in {"MODEL_FILE", "UNSCANNED_DOCUMENT"}
                               for f in rep.findings)
            shown = mask(rep.path) if path_flagged and not args.reveal else rep.path
            print(f"== {shown}  [" + ", ".join(f"{d}={c}" for d, c in counts.most_common()) + "]")
            for f in rep.findings[: args.max_per_file]:
                txt = f.text if args.reveal else mask(f.text)
                print(f"   {f.location[:60]:<60} {f.detector:<18} {txt}")
            if len(rep.findings) > args.max_per_file:
                print(f"   ... {len(rep.findings) - args.max_per_file} more")
    print("\nSUMMARY")
    print(f"  files scanned: {n_files}; paths with findings: {len(reports)}")
    for det, c in sorted(totals.items(), key=lambda x: (["high", "medium", "review"].index(SEVERITY[x[0]]), x[0])):
        print(f"  {SEVERITY[det]:<7} {det:<18} {c}")
    if args.json_out:
        data = {
            "files_scanned": n_files,
            "totals": dict(totals),
            "files": [
                {"path": r.path if args.reveal else mask(r.path),
                 "counts": dict(collections.Counter(f.detector for f in r.findings))}
                for r in reports
            ],
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
    failed = by_sev["high"] + by_sev["medium"] + (by_sev["review"] if args.strict else 0)
    print("RESULT:", "FAIL" if failed else "PASS",
          f"(high={by_sev['high']}, medium={by_sev['medium']}, review={by_sev['review']})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
