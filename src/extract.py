"""Stage 2 - Text extraction from downloaded documents.

Extracts text from each PDF with pypdf and falls back to Tesseract OCR for
scanned files, then assembles one per-person document-level knowledge base.
A lightweight keyword heuristic classifies each document and extracts the
years it mentions, which supports the longitudinal ordering of Stage 3.

External tools required: Poppler (for pdf2image) and Tesseract OCR.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pdf2image import convert_from_path
from pypdf import PdfReader
import pytesseract

logger = logging.getLogger(__name__)

# Italian document-type keywords (the corpora are Italian public records).
_DOC_TYPES = [
    ("Contract", r"\b(contratto|assunzione|lettera di impegno)\b"),
    ("Payslip", r"\b(cedolino|busta paga|retribuzione|prospetto paga)\b"),
    ("Curriculum Vitae", r"\b(curriculum|cv|esperienze lavorative)\b"),
    ("Income certification", r"\b(certificazione|cud|cu|redditi)\b"),
]


def ocr_pdf(pdf_path: str | Path, dpi: int = 600) -> str:
    text = ""
    try:
        for page in convert_from_path(str(pdf_path), dpi):
            text += pytesseract.image_to_string(page) + "\n"
    except Exception as exc:  # noqa: BLE001
        logger.error("OCR failed for %s: %s", pdf_path, exc)
    return text


def extract_pdf_text(pdf_path: str | Path, ocr_dpi: int = 600) -> str:
    """Native text extraction with OCR fallback for scanned PDFs."""
    text = ""
    try:
        for page in PdfReader(str(pdf_path)).pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Text extraction failed for %s: %s", pdf_path, exc)
    if not text.strip():
        logger.info("No embedded text in %s; falling back to OCR", pdf_path)
        text = ocr_pdf(pdf_path, ocr_dpi)
    return text


def classify_document(text: str) -> str:
    low = text.lower()
    for label, pattern in _DOC_TYPES:
        if re.search(pattern, low):
            return label
    return "Other/Unclassified"


def extract_years(text: str) -> list[int]:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    return sorted({int(y) for y in years})
