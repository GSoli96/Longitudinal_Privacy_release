"""Re-identification check: how unique are the sets of inferred attributes (spec 7.8, plan task 14).

The question is whether the attribute sets of a case could point to one person if they were released. For each
person the final-step attributes (carried copies included: they are the person's attributes at that step) are
reduced to keys; two attributes of two persons share a key if their normalised texts are equal ("exact") or, with an
embedder, if they fall in the same greedy cluster at a cosine threshold ("semantic"). Then, per population:

* ``k_attribute``: for each distinct key, the number of persons that carry it; share of keys held by one person
  (``keys_k1``), by fewer than 5 (``keys_k_lt5``);
* ``person_min_k``: for each person the smallest k among their keys; share of persons with a key held only by them
  (``persons_with_unique_key``) and with min k below 5;
* ``combination_uniqueness``: for r = 1, 2, 3, the share of persons that have at least one set of at most r keys that no
  other person carries in full (the usual k-anonymity reading, at the level of attribute sets);
* ``identical_sets``: persons whose whole set of keys equals the set of another person.

Two populations are reported: ``all`` (every attribute) and ``dangerous`` (plausible, coherent and realistic, not
explicit, i.e. the attributes counted by the DIR). Assumption to confirm with the authors: the check treats the
attribute texts, not the documents, as the data that would be released, and counts a person as exposed when some
combination of at most three of their attributes is unique in the case. It says nothing about outside knowledge
of an intruder.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from . import metrics

R_MAX = 3
MAX_COMBOS = 5_000_000


def _final(records, step: int = 100):
    return [r for r in records if r.step == step]


def _dangerous(r) -> bool:
    b = metrics._bits(r)
    return b is not None and bool(b[0] and b[1] and not b[3])


def exact_keys(records) -> dict:
    """{person: set of normalised texts}."""
    out = defaultdict(set)
    for r in records:
        k = metrics._norm(r.text)
        if k:
            out[r.person].add(k)
    return out


def semantic_keys(records, embed, threshold: float = 0.85, block: int = 2048) -> dict:
    """{person: set of cluster ids}: greedy clusters over the distinct normalised texts of the whole case, in order
    of first appearance (sorted texts, so that the result does not depend on the input order)."""
    texts = sorted({metrics._norm(r.text) for r in records} - {""})
    if not texts:
        return {}
    vec = np.asarray(embed(texts), dtype=np.float32)
    reps = np.zeros_like(vec)
    n_reps, cluster_of = 0, {}
    for i, t in enumerate(texts):
        if n_reps:
            best, best_cos = -1, -1.0
            for s in range(0, n_reps, block):
                sims = reps[s:min(s + block, n_reps)] @ vec[i]
                j = int(np.argmax(sims))
                if float(sims[j]) > best_cos:
                    best, best_cos = s + j, float(sims[j])
            if best_cos >= threshold:
                cluster_of[t] = best
                continue
        reps[n_reps] = vec[i]
        cluster_of[t] = n_reps
        n_reps += 1
    out = defaultdict(set)
    for r in records:
        k = metrics._norm(r.text)
        if k:
            out[r.person].add(cluster_of[k])
    return out


def summarise(keys_by_person: dict, r_max: int = R_MAX, max_combos: int = MAX_COMBOS) -> dict:
    """Uniqueness measures for {person: set of keys} (see the module docstring)."""
    persons = [p for p, ks in keys_by_person.items() if ks]
    n_without = len(keys_by_person) - len(persons)
    out = {"n_persons": len(keys_by_person), "n_persons_with_attributes": len(persons), "n_persons_without": n_without}
    if not persons:
        return out
    holders = Counter(k for p in persons for k in keys_by_person[p])
    n_keys = len(holders)
    out["n_distinct_keys"] = n_keys
    out["keys_k1"] = sum(1 for v in holders.values() if v == 1) / n_keys
    out["keys_k_lt5"] = sum(1 for v in holders.values() if v < 5) / n_keys
    out["k_attribute_median"] = float(np.median(list(holders.values())))
    mins = {p: min(holders[k] for k in keys_by_person[p]) for p in persons}
    out["persons_with_unique_key"] = sum(1 for v in mins.values() if v == 1) / len(persons)
    out["persons_min_k_lt5"] = sum(1 for v in mins.values() if v < 5) / len(persons)
    sets = Counter(frozenset(keys_by_person[p]) for p in persons)
    out["identical_sets"] = sum(1 for p in persons if sets[frozenset(keys_by_person[p])] > 1) / len(persons)
    comb, exposed = {}, set()
    for r in range(1, r_max + 1):
        total = sum(_ncomb(len(keys_by_person[p]), r) for p in persons)
        if total > max_combos:
            comb[str(r)] = None
            continue
        count = Counter()
        for p in persons:
            for c in combinations(sorted(keys_by_person[p]), r):
                count[c] += 1
        for p in persons:
            if p not in exposed and any(count[c] == 1 for c in combinations(sorted(keys_by_person[p]), r)):
                exposed.add(p)
        comb[str(r)] = len(exposed) / len(persons)
    out["combination_uniqueness"] = comb
    return out


def _ncomb(n: int, r: int) -> int:
    from math import comb
    return comb(n, r)


def analyse(records, embed=None, thresholds=(0.85,), step: int = 100) -> dict:
    """Re-identification summary of the final step of a case: exact keys and, with ``embed``, semantic keys."""
    fin = _final(records, step)
    out = {"step": step}
    for name, recs in (("all", fin), ("dangerous", [r for r in fin if _dangerous(r)])):
        block = {"exact": summarise(exact_keys(recs))}
        # persons without any attribute of the population still count in the denominator of the case
        n_case = len({r.person for r in fin})
        block["exact"]["n_persons_in_case"] = n_case
        if embed is not None:
            for thr in thresholds:
                s = summarise(semantic_keys(recs, embed, thr))
                s["n_persons_in_case"] = n_case
                block[f"cosine_{thr:.2f}"] = s
        out[name] = block
    return out
