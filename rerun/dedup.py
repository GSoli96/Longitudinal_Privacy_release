"""Two-phase semantic de-duplication of the attributes of a person (main variant for novelty and PES).

1. Candidates: pairs of attributes of the same person whose bge-m3 cosine is at least
   ``config.DEDUP_CANDIDATE_COSINE`` (0.80). Cosine alone does not separate paraphrases from near-miss facts
   (a fixed-term and a permanent contract score 0.96), so it only proposes pairs.
2. Confirmation: an LLM (``config.DEDUP_MODEL``, temperature 0) decides, for each candidate pair, whether the
   two statements express the same fact about the person with the same values (dates, numbers, contract type,
   role). Only confirmed pairs are merged; a pair whose answer stays missing or invalid after the retries is
   not merged and is counted apart (``n_invalid``).
Clusters are the connected components of the confirmed edges (so A-B and B-C confirmed give {A, B, C}); the
number of clusters with more than three attributes and the largest cluster are reported for review.
Carried copies of a step are not clustered (they repeat an earlier attribute).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from . import config

NUM_PREDICT = 4096
BATCH_SIZE = 20
MAX_RETRIES = 2

SCHEMA = {"type": "object",
          "properties": {"verdicts": {"type": "array", "items": {
              "type": "object",
              "properties": {"id": {"type": "integer"}, "same": {"type": "string", "enum": ["yes", "no"]}},
              "required": ["id", "same"]}}},
          "required": ["verdicts"]}

INSTRUCTION = """
You compare pairs of statements about the same person, inferred from public administrative documents.

For each numbered pair decide whether the two statements express the same fact about the person with the same values (dates, numbers, quantities, contract type, role, place, organisation).
- Answer "yes" only if one statement is a paraphrase of the other.
- Answer "no" if the facts are different, if any value differs, or if one statement adds or removes a detail that changes what is claimed.

PAIRS:
{pairs}

Answer only with JSON: {{"verdicts": [{{"id": 1, "same": "yes"}}, ...]}} with one item per pair.
"""


def _ordered(records) -> list:
    """The records that are clustered: carried copies out, in (step, original order)."""
    recs = [(i, r) for i, r in enumerate(records) if not r.carried]
    return [r for _, r in sorted(recs, key=lambda t: (t[1].step, t[0]))]


def candidate_pairs(records, embed, threshold: float = config.DEDUP_CANDIDATE_COSINE) -> list[tuple[int, int, float]]:
    """(i, j, cosine) with i < j, indices into ``_ordered(records)``, cosine >= threshold."""
    recs = _ordered(records)
    if len(recs) < 2:
        return []
    vec = np.asarray(embed([r.text for r in recs]))
    cos = vec @ vec.T
    iu, ju = np.triu_indices(len(recs), k=1)
    keep = cos[iu, ju] >= threshold
    return [(int(i), int(j), float(cos[i, j])) for i, j in zip(iu[keep], ju[keep])]


def _build_prompt(pairs: list[tuple[str, str]]) -> str:
    return INSTRUCTION.format(pairs="\n".join(f"{n}.\nA: {a}\nB: {b}" for n, (a, b) in enumerate(pairs, 1)))


def confirm_pairs(client, model: str, pairs: list[tuple[str, str]], *, batch_size: int = BATCH_SIZE,
                  max_retries: int = MAX_RETRIES, stats: Counter | None = None) -> list[str]:
    """For each (a, b): "yes", "no" or "invalid" (answer missing or not yes/no after the retries)."""
    stats = stats if stats is not None else Counter()
    out = ["invalid"] * len(pairs)
    for start in range(0, len(pairs), batch_size):
        pending = list(range(start, min(start + batch_size, len(pairs))))
        for attempt in range(max_retries + 1):
            if not pending:
                break
            prompt = _build_prompt([pairs[k] for k in pending])
            opts = {"num_ctx": config.num_ctx_for(model), "num_predict": NUM_PREDICT, "seed": config.SEED + attempt}
            stats["calls"] += 1
            r = client.chat(model, [{"role": "user", "content": prompt}], opts, SCHEMA)
            if attempt:
                stats["retries"] += 1
            if not r.ok or not isinstance(r.json, dict) or not isinstance(r.json.get("verdicts"), list):
                stats["failed_calls"] += 1
                continue
            seen, answered = set(), set()
            for it in r.json["verdicts"]:
                n = it.get("id") if isinstance(it, dict) else None
                same = it.get("same") if isinstance(it, dict) else None
                same = same.strip().lower() if isinstance(same, str) else None
                if isinstance(n, int) and 1 <= n <= len(pending) and n not in seen and same in ("yes", "no"):
                    seen.add(n)
                    out[pending[n - 1]] = same
                    answered.add(pending[n - 1])
            pending = [k for k in pending if k not in answered]
    stats["pairs"] += len(pairs)
    return out


def confirmed_clusters(records, embed, client, *, model: str = config.DEDUP_MODEL,
                       threshold: float = config.DEDUP_CANDIDATE_COSINE, batch_size: int = BATCH_SIZE,
                       max_retries: int = MAX_RETRIES, stats: Counter | None = None):
    """-> (clusters, info). Each cluster is a list of records in (step, order); clusters in order of first member."""
    recs = _ordered(records)
    cands = candidate_pairs(records, embed, threshold)
    answers = confirm_pairs(client, model, [(recs[i].text, recs[j].text) for i, j, _ in cands],
                            batch_size=batch_size, max_retries=max_retries, stats=stats) if cands else []
    parent = list(range(len(recs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j, _), ans in zip(cands, answers):
        if ans == "yes":
            parent[find(i)] = find(j)
    groups: dict[int, list] = {}
    for i, r in enumerate(recs):
        groups.setdefault(find(i), []).append(r)
    clusters = sorted(groups.values(), key=lambda c: recs.index(c[0]))
    info = {"n_records": len(recs), "n_candidates": len(cands), "n_confirmed": answers.count("yes"),
            "n_rejected": answers.count("no"), "n_invalid": answers.count("invalid"),
            "n_clusters": len(clusters), "clusters_gt3": sum(1 for c in clusters if len(c) > 3),
            "largest_cluster": max((len(c) for c in clusters), default=0)}
    return clusters, info
