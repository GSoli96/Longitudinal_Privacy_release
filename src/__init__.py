"""Longitudinal privacy inference pipeline.

Stages (paper, Section 3):
  1. crawl       - public knowledge extraction
  2. extract     - text extraction (PDF + OCR)
  3. inference   - longitudinal attribute inference
  4. validation  - multi-judge attribute validation
  5. phishing    - phishing scenario simulation
  6. detection   - phishing detection

Supporting modules: config, llm_client, majority_vote, metrics.
"""

__all__ = [
    "config",
    "llm_client",
    "crawl",
    "extract",
    "inference",
    "validation",
    "majority_vote",
    "metrics",
    "phishing",
    "detection",
]
