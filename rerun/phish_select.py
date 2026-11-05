"""Selection of the attributes that feed the phishing generation (RQ3; plan ``docs/plans/rq34.md`` §2 and §8 step 1).

One filter for every case: the attributes of the last step (100% of the documents) whose majority labels are
plausible = coherent = realistic = yes. They are split by the fourth label: ``inferred`` (explicit = no) and
``explicit`` (explicit = yes; the generation of the explicit-only variant uses a separate extraction, the
attributes listed here document what the judges accepted). At most ``MAX_ATTRIBUTES`` per list and per person, taken
with a seeded shuffle per person (seed ``"seed:person"``), so that the choice depends neither on the order of the
file nor on the model's wording. Duplicates (same semantic cluster if given, else same normalised text) count once.

Every input person is in the output, also with empty lists; the counters say why attributes were left out. Counts
only: no text goes into the summary."""
from __future__ import annotations

import random
from collections import Counter

from . import config, metrics

MAX_ATTRIBUTES = 12
LAST_STEP = 100
ACCEPT = ("plausible", "coherent", "realistic")


def _verdict(rec) -> str:
    """"accepted_inferred", "accepted_explicit" or the reason for the rejection."""
    labels = rec.labels
    if any(labels.get(c) not in ("yes", "no") for c in metrics.CRITERIA):
        return "undecided"
    for c in ACCEPT:
        if labels[c] == "no":
            return "rejected_" + c
    return "accepted_explicit" if labels["explicit"] == "yes" else "accepted_inferred"


def select_person(records, *, cap: int = MAX_ATTRIBUTES, seed: int = config.SEED, clusters: dict | None = None,
                  step: int = LAST_STEP) -> dict:
    """``{"inferred": [rec, ...], "explicit": [rec, ...], "counts": Counter}`` for the records of one person."""
    counts: Counter = Counter()
    pools = {"inferred": [], "explicit": []}
    seen = set()
    person = records[0].person if records else ""
    for rec in records:
        if rec.step != step:
            continue
        v = _verdict(rec)
        counts[v] += 1
        if not v.startswith("accepted"):
            continue
        key = (clusters or {}).get(rec.attr_id) if clusters else None
        key = key if key is not None else metrics._norm(rec.text)
        kind = "explicit" if v == "accepted_explicit" else "inferred"
        if (kind, key) in seen:
            counts["duplicate"] += 1
            continue
        seen.add((kind, key))
        pools[kind].append(rec)
    out = {"counts": counts}
    for kind, pool in pools.items():
        pool = sorted(pool, key=lambda r: (r.attr_id, metrics._norm(r.text)))
        random.Random(f"{seed}:{person}:{kind}").shuffle(pool)
        if len(pool) > cap:
            counts["beyond_cap_" + kind] += len(pool) - cap
        out[kind] = pool[:cap]
    return out


def select_all(records, persons, *, cap: int = MAX_ATTRIBUTES, seed: int = config.SEED,
               clusters_by_person: dict | None = None, step: int = LAST_STEP) -> dict:
    """``{"by_person": {person: select_person(...)}, "summary": {...}}``. Persons without records are kept."""
    by_p: dict = {p: [] for p in persons}
    extra = 0
    for r in records:
        if r.person in by_p:
            by_p[r.person].append(r)
        else:
            extra += 1
    selected = {p: select_person(rs, cap=cap, seed=seed, step=step,
                                 clusters=(clusters_by_person or {}).get(p)) for p, rs in by_p.items()}
    total: Counter = Counter()
    for sel in selected.values():
        total.update(sel["counts"])
    summary = {
        "n_persons": len(by_p),
        "n_persons_with_inferred": sum(1 for s in selected.values() if s["inferred"]),
        "n_persons_with_explicit": sum(1 for s in selected.values() if s["explicit"]),
        "n_persons_without_attributes": sum(1 for s in selected.values() if not s["inferred"] and not s["explicit"]),
        "n_inferred": sum(len(s["inferred"]) for s in selected.values()),
        "n_explicit": sum(len(s["explicit"]) for s in selected.values()),
        "records_of_unknown_persons": extra,
        "counts": dict(total),
        "cap": cap, "step": step, "seed": seed,
    }
    return {"by_person": selected, "summary": summary}
