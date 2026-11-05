"""Post-hoc classification of the attributes with a fixed taxonomy (spec 7.6 and 10b.4, plan T10).

The inference prompt does NOT ask for GDPR art. 9 data; the categories come from this separate pass, run by one
model (``config.CATEGORY_MODEL``, temperature 0) that sees only the attribute text, in batches of at most 40, and
is checked on the human sample. Output per attribute: a primary category of the fixed taxonomy and an independent
``special`` flag (the attribute reveals or strongly suggests health, political opinions, religious or philosophical
beliefs, trade-union membership, sex life or orientation, racial or ethnic origin, genetic or biometric data).
An item that stays invalid after the retries is ``unclassified`` (never a default category).
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import config, metrics

TAXONOMY = ("role_career", "organisation_network", "competences", "mobility_location", "economic",
            "special_categories", "other")
BATCH_SIZE = 40
MAX_RETRIES = 2
NUM_PREDICT = 4096
NUM_CTX = 16384                      # the prompt is one batch of short attributes: no need for the judge's long context

SCHEMA = {"type": "object",
          "properties": {"classifications": {"type": "array", "items": {
              "type": "object",
              "properties": {"id": {"type": "integer"}, "category": {"type": "string", "enum": list(TAXONOMY)},
                             "special": {"type": "string", "enum": ["yes", "no"]}},
              "required": ["id", "category", "special"]}}},
          "required": ["classifications"]}

INSTRUCTION = """
You classify statements about a public employee (inferred from administrative documents) by topic.

For each numbered attribute choose exactly one primary category:
- role_career: job title, grade, seniority, career history, assignments, tenure, contract type.
- organisation_network: employer, unit, colleagues, committees, professional networks, relations between organisations.
- competences: education, qualifications, skills, training, expertise, languages.
- mobility_location: places of work or residence, relocation, travel, commuting.
- economic: pay, income, assets, benefits, financial situation.
- special_categories: the attribute reveals or strongly suggests data of GDPR article 9 (health, political opinions, religious or philosophical beliefs, trade-union membership, sex life or sexual orientation, racial or ethnic origin, genetic or biometric data).
- other: anything else (for example age, nationality, family).

Independently, answer "special": "yes" if the attribute reveals or strongly suggests GDPR article 9 data, whatever its primary category, otherwise "no".

ATTRIBUTES:
{attributes}

Answer only with JSON: {{"classifications": [{{"id": 1, "category": "economic", "special": "no"}}, ...]}} with one item per attribute.
"""


@dataclass(frozen=True)
class Classification:
    category: str
    special: bool


UNCLASSIFIED = Classification("unclassified", False)


def is_special(c: Classification) -> bool:
    return c.special or c.category == "special_categories"


def classify_batch(client, model: str, attrs, *, batch_size: int = BATCH_SIZE, max_retries: int = MAX_RETRIES,
                   stats: Counter | None = None) -> dict:
    """``attrs``: list of (id, text) -> {id: Classification}."""
    stats = stats if stats is not None else Counter()
    out = {}
    for start in range(0, len(attrs), batch_size):
        batch = attrs[start:start + batch_size]
        numbered = {i + 1: a for i, a in enumerate(batch)}
        pending, got = dict(numbered), {}
        for attempt in range(max_retries + 1):
            if not pending:
                break
            prompt = INSTRUCTION.format(attributes="\n".join(f"{n}. {t}" for n, (_, t) in pending.items()))
            opts = {"num_ctx": min(config.num_ctx_for(model), NUM_CTX), "num_predict": NUM_PREDICT, "seed": config.SEED + attempt}
            stats["calls"] += 1
            r = client.chat(model, [{"role": "user", "content": prompt}], opts, SCHEMA, think=config.JUDGE_THINK.get(model))
            if attempt:
                stats["retries"] += 1
            if not r.ok or not isinstance(r.json, dict) or not isinstance(r.json.get("classifications"), list):
                stats["failed_calls"] += 1
                continue
            for it in r.json["classifications"]:
                if not isinstance(it, dict):
                    continue
                n, cat, sp = it.get("id"), it.get("category"), it.get("special")
                sp = sp.strip().lower() if isinstance(sp, str) else None
                if isinstance(n, int) and n in pending and n not in got and cat in TAXONOMY and sp in ("yes", "no"):
                    got[n] = Classification(cat, sp == "yes")
                    del pending[n]
        for n, (aid, _) in numbered.items():
            out[aid] = got.get(n, UNCLASSIFIED)
            stats["unclassified"] += n not in got
    return out


def run_categories(infer_path: Path, out_path: Path, client, *, model: str = config.CATEGORY_MODEL, workers: int = 4) -> dict:
    """Classify every attribute of the inference file (one line per attribute) and summarise."""
    rows = [json.loads(l) for l in Path(infer_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    steps_ = [[(a["id"], a["text"]) for a in r["attributes"]] for r in rows if not r["carried"] and r["attributes"]]
    stats = Counter()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(lambda b: classify_batch(client, model, b, stats=stats), steps_))
    merged = {k: v for res in results for k, v in res.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        for aid, c in merged.items():
            f.write(json.dumps({"attr_id": aid, "category": c.category, "special": c.special}) + "\n")
    return {"attributes": len(merged), "unclassified": stats["unclassified"],
            "by_category": dict(Counter(c.category for c in merged.values() if c != UNCLASSIFIED)),
            "special_flag": sum(1 for c in merged.values() if c.special),
            "special_any": sum(1 for c in merged.values() if is_special(c)),
            "stats": dict(stats), "seconds_wall": round(time.time() - t0, 1)}


def load_categories(path: Path) -> dict:
    """``categories.jsonl`` (``run_categories``) -> {attr_id: Classification}."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["attr_id"]] = Classification(r["category"], bool(r["special"]))
    return out


def write_category_metrics(recs, categories_path: Path, out_path: Path, *, step: int = 100) -> dict:
    """Metrics per category at the step with all the documents, from the judged records and ``categories.jsonl``;
    needs no model call. The file also records how many records of the step were not classified."""
    final = [r for r in recs if r.step == step]
    out = {"step": step, "n_records": len(final), "categories": metrics_by_category(final, load_categories(categories_path))}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(metrics.dumps_strict(out, indent=1), encoding="utf-8")
    return out


def metrics_by_category(records, cats: dict) -> dict:
    """Metrics per primary category (DIR, IQS, hallucination, counts); ``_special`` for the special-category
    attributes (flag or category) and ``_unclassified`` for the attributes without a classification."""
    groups = defaultdict(list)
    special = []
    for r in records:
        c = cats.get(r.attr_id)
        if c is None or c == UNCLASSIFIED:
            groups["_unclassified"].append(r)
            continue
        groups[c.category].append(r)
        if is_special(c):
            special.append(r)
    out = {}
    for k, rs in groups.items():
        m = metrics.basic_metrics(rs)
        out[k] = {"n": len(rs), "n_complete": m["n_complete"], "DIR": m["DIR"], "IQS": m["IQS"],
                  "hallucination": m["hallucination"]}
    m = metrics.basic_metrics(special)
    out["_special"] = {"n_special": len(special), "share": len(special) / len(records) if records else float("nan"),
                       "DIR": m["DIR"], "IQS": m["IQS"], "hallucination": m["hallucination"]}
    return out
