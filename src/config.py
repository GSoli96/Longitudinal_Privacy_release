"""Run configuration for one case.

A case is one public administration. All case-specific choices live in a
`CaseConfig` and are loaded from a YAML file in `configs/` (one per case,
shipped with placeholders). Nothing here names a real administration,
portal, person or key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class CaseConfig:
    # Identity (anonymous): internal letter and generic organisation type.
    case_id: str = "X"
    org_type: str = "public administration"

    # Stage 1 — crawling. Targets are configuration placeholders; set them
    # to the transparency portal(s) you are authorised to crawl.
    base_url: str = "https://example.org"      # placeholder
    html_index_dir: str = "data/private/index"  # local HTML index pages
    downloads_dir: str = "data/private/downloads"

    # Stage 2 — extraction.
    docs_dir: str = "data/private/docs"
    ocr_dpi: int = 600

    # Stage 3 — inference.
    inference_model: str = "gpt-oss:20b"
    exposure_steps: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    max_chars: int = 12_000         # characters of context per inference call
    carry_history: bool = False     # True only for Case F (cumulative inference)

    # Stage 4 — validation.
    judge_models: tuple[str, ...] = (
        "gemma3:12b",
        "gemma4:31b",
        "gpt-oss:120b",
    )

    # Stage 5 — phishing generation.
    generation_models: tuple[str, ...] = ("gemma3:12b", "gemma4:31b")
    max_attrs_in_prompt: int = 12
    # Organisation strings that appear in the generated messages. These are
    # placeholders; set them for the organisation you are authorised to model.
    company_name: str = "PLACEHOLDER_ORG"
    company_domain: str = "placeholder.example"

    # Stage 6 — detection.
    benchmarks: tuple[str, ...] = (
        "CEAS_08", "Enron", "Ling", "Nazario", "Nigerian_Fraud", "SpamAssassin",
    )
    legit_corpus: str = "data/private/legit_it.json"
    nlp_epochs: int = 3
    llm_sample_size: int = 200

    # I/O
    output_dir: str = "outputs"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CaseConfig":
        if yaml is None:
            raise ImportError("Install PyYAML:  pip install pyyaml")
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for key in ("exposure_steps", "judge_models", "generation_models", "benchmarks"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: v for k, v in data.items() if k not in known}
        data = {k: v for k, v in data.items() if k in known}
        cfg = cls(**data)
        cfg.extra.update(extra)
        return cfg


def ollama_env() -> dict[str, str]:
    """Endpoint settings read from the environment (never hard-coded)."""
    return {
        "host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        "api_key": os.environ.get("OLLAMA_API_KEY", ""),
    }
