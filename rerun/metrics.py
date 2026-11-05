"""Metrics of the rerun (definitions of the paper, spec 4.5 and 10b.6).

Per attribute, the majority vote over the three judges gives four labels P, C, R, E (plausible,
coherent, realistic, explicit). An attribute whose four labels are not all decided (one criterion
``undecided``) is left out of the rates and counted separately (``n_undecided``): every metric is
computed on the complete attributes.

* IQS = (P + C + R + 2 E) / 5 with P, C, R, E the rates; ``IQS_non_explicit`` is the same formula on
  the attributes with E = 0.
* DIR = share of attributes with P = 1, C = 1, E = 0 (realism not required).
* Hallucination = share of attributes rejected on all four criteria; the distribution of all 16 label
  combinations is reported by ``label_combinations`` (keys are the bits P C R E).
* Novelty at step t = share of the attributes of step t not already produced at the steps before, by
  person; exact (normalised string: case, spaces) and semantic (bge-m3 cosine below a threshold). The
  first step has no novelty (``None``). Carried steps (identical to the previous one, not re-run) are not
  new inferences and are left out.
* PES: the attributes of a person are grouped in semantic clusters (cosine >= threshold with the first
  member of the cluster, so there is no chaining); a cluster counts if its labels at its latest occurrence
  up to the step are P = C = R = 1 and E = 0 (the most informed text decides: an attribute that was
  explicit or unrealistic at an early step and is validated later counts, and the reverse). Cumulative PES
  at step s = number of clusters first seen at or before s that pass at their latest occurrence <= s; also
  per 1,000 characters of the text up to that step. A cluster whose representative has an undecided label
  does not count and is reported (``undecided_clusters``).
"""
from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

CRITERIA = ("plausible", "coherent", "realistic", "explicit")
STEPS = (25, 50, 75, 100)
NAN = float("nan")


def _nan_to_none(x):
    if isinstance(x, (float, np.floating)):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_nan_to_none(v) for v in x]
    return x


def dumps_strict(obj, indent=None) -> str:
    """JSON text that any parser reads: nan and infinity (no data, empty denominator) become null. ``json.dumps``
    would write the bare words NaN/Infinity, which are not JSON."""
    return json.dumps(_nan_to_none(obj), indent=indent, default=str, allow_nan=False)


@dataclass
class AttrRecord:
    person: str
    step: int
    text: str
    labels: dict = field(default_factory=dict)   # criterion -> "yes" | "no" | "undecided"
    carried: bool = False
    attr_id: str = ""                            # id of the attribute (links to the categories and the sample)


def _bits(r: AttrRecord):
    """(P, C, R, E) as 0/1, or None if any label is not decided."""
    out = []
    for c in CRITERIA:
        v = r.labels.get(c)
        if v not in ("yes", "no"):
            return None
        out.append(1 if v == "yes" else 0)
    return tuple(out)


def _norm(t: str) -> str:
    return re.sub(r"[\W_]+", " ", t.casefold()).strip()


def _mean(xs):
    return sum(xs) / len(xs) if xs else NAN


def _iqs(bits):
    n = len(bits)
    if not n:
        return NAN
    p, c, r, e = (sum(b[i] for b in bits) / n for i in range(4))
    return (p + c + r + 2 * e) / 5


# ---- rates ----------------------------------------------------------------------------------------------------
def basic_metrics(records) -> dict:
    recs = list(records)
    bits = [b for b in (_bits(r) for r in recs) if b is not None]
    n, nc = len(recs), len(bits)
    out = {"n_total": n, "n_complete": nc, "n_undecided": n - nc,
           "undecided_share": (n - nc) / n if n else NAN}
    for i, k in enumerate("PCRE"):
        out[k] = _mean([b[i] for b in bits])
    out["IQS"] = _iqs(bits)
    out["DIR"] = _mean([1 if (b[0] and b[1] and not b[3]) else 0 for b in bits])
    out["hallucination"] = _mean([1 if not any(b) else 0 for b in bits])
    non_expl = [b for b in bits if not b[3]]
    out["n_non_explicit"] = len(non_expl)
    out["IQS_non_explicit"] = _iqs(non_expl)
    return out


def label_combinations(records) -> dict:
    """Count and share of the 16 combinations of (P, C, R, E) among the complete attributes."""
    bits = [b for b in (_bits(r) for r in records) if b is not None]
    out = {f"{i:04b}": {"count": 0, "share": 0.0} for i in range(16)}
    for b in bits:
        out["".join(map(str, b))]["count"] += 1
    for v in out.values():
        v["share"] = v["count"] / len(bits) if bits else 0.0
    return out


def step_metrics(records) -> dict:
    """Metrics per step; carried copies stay in (a person keeps the results of the previous step)."""
    by = defaultdict(list)
    for r in records:
        by[r.step].append(r)
    out = {}
    for s in sorted(by):
        m = basic_metrics(by[s])
        m["n_carried"] = sum(1 for r in by[s] if r.carried)
        m["n_persons"] = len({r.person for r in by[s]})
        out[s] = m
    return out


_METRIC_KEYS = {"DIR", "IQS", "hallucination", "P", "C", "R", "E", "IQS_non_explicit"}


def per_person_step_values(records, metric: str, persons, steps=STEPS) -> dict:
    """{person: [value per step]} of one metric; ``None`` where the person has no complete attribute at
    the step (undefined, not zero). Input of the person-level bootstrap and of the trend tests."""
    assert metric in _METRIC_KEYS, metric
    by = defaultdict(list)
    for r in records:
        by[(r.person, r.step)].append(r)
    out = {}
    for p in persons:
        row = []
        for s in steps:
            m = basic_metrics(by.get((p, s), []))
            v = m[metric]
            row.append(None if (m["n_complete"] == 0 or (isinstance(v, float) and math.isnan(v))) else v)
        out[p] = row
    return out


# ---- novelty ---------------------------------------------------------------------------------------------------
def _by_person_step(records, include_carried=False):
    out = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.carried and not include_carried:
            continue
        out[r.person][r.step].append(r)
    return out


def novelty_exact(records, include_carried: bool = False) -> dict:
    data = _by_person_step(records, include_carried)
    steps = sorted({s for p in data.values() for s in p})
    new, tot = defaultdict(int), defaultdict(int)
    for p in data.values():
        seen = set()
        for s in sorted(p):
            texts = [_norm(r.text) for r in p[s]]
            tot[s] += len(texts)
            new[s] += sum(1 for t in texts if t not in seen)
            seen.update(texts)
    return {s: (None if s == steps[0] else (new[s] / tot[s] if tot[s] else None)) for s in steps}


def novelty_semantic(records, embed, thresholds=(0.80, 0.85, 0.90), include_carried: bool = False) -> dict:
    """{threshold: {step: novelty}}: an attribute is new if its cosine to every attribute of the earlier
    steps of the same person is below the threshold. ``embed`` maps a list of texts to unit vectors."""
    data = _by_person_step(records, include_carried)
    steps = sorted({s for p in data.values() for s in p})
    out = {}
    for thr in thresholds:
        new, tot = defaultdict(int), defaultdict(int)
        for p in data.values():
            ss = sorted(p)
            texts = [r.text for s in ss for r in p[s]]
            vec = np.asarray(embed(texts)) if texts else np.zeros((0, 1))
            idx = 0
            earlier = []
            for s in ss:
                n_s = len(p[s])
                cur = vec[idx:idx + n_s]
                idx += n_s
                tot[s] += n_s
                if not earlier:
                    new[s] += n_s
                else:
                    prev = np.vstack(earlier)
                    new[s] += int(np.sum(np.max(cur @ prev.T, axis=1) < thr))
                earlier.append(cur)
        out[thr] = {s: (None if s == steps[0] else (new[s] / tot[s] if tot[s] else None)) for s in steps}
    return out


# ---- semantic clusters and PES ---------------------------------------------------------------------------------------
def semantic_clusters(person_records, embed, threshold: float = 0.85) -> list:
    """Greedy clusters of the attributes of one person, in (step, order): an attribute joins the cluster
    whose first member is the most similar if the cosine reaches the threshold, otherwise starts a new one.
    Carried copies are not clustered (they repeat an earlier attribute)."""
    recs = [r for r in person_records if not r.carried]
    recs = sorted(enumerate(recs), key=lambda t: (t[1].step, t[0]))
    recs = [r for _, r in recs]
    if not recs:
        return []
    vec = np.asarray(embed([r.text for r in recs]))
    clusters, reps = [], []
    for i, r in enumerate(recs):
        best, best_cos = None, -1.0
        for k, ri in enumerate(reps):
            c = float(vec[i] @ vec[ri])
            if c > best_cos:
                best, best_cos = k, c
        if best is not None and best_cos >= threshold:
            clusters[best].append(r)
        else:
            clusters.append([r])
            reps.append(i)
    return clusters


def novelty_from_clusters(records, clusters_by_person: dict) -> dict:
    """{step: novelty}: an attribute of step t is new if its cluster has no member at an earlier step.
    ``clusters_by_person`` maps a person to its clusters (lists of records), e.g. the confirmed ones of
    ``rerun.dedup``; carried copies are not in the clusters and are left out here too."""
    first_step = {}
    for p, clusters in clusters_by_person.items():
        for cl in clusters:
            m = min(r.step for r in cl)
            for r in cl:
                first_step[id(r)] = m
    new, tot = defaultdict(int), defaultdict(int)
    steps = sorted({r.step for r in records if not r.carried})
    for r in records:
        if r.carried or id(r) not in first_step:
            continue
        tot[r.step] += 1
        new[r.step] += int(first_step[id(r)] == r.step)
    return {s: (None if s == steps[0] else (new[s] / tot[s] if tot[s] else None)) for s in steps}


def pes_by_step(person_records, embed, step_chars, threshold: float = 0.85, steps=STEPS, clusters=None) -> dict:
    """Cumulative PES of one person at each step, and per 1,000 characters of the text up to that step.

    ``clusters`` (optional) are the clusters of the person from another de-duplication (the confirmed ones);
    by default they come from the embedding cosine alone (``semantic_clusters``)."""
    if clusters is None:
        clusters = semantic_clusters(person_records, embed, threshold)
    out = {}
    for s in steps:
        passing = undecided = n_cl = 0
        for cl in clusters:
            upto = [r for r in cl if r.step <= s]
            if not upto:
                continue
            n_cl += 1
            last = max(r.step for r in upto)
            rep = next(r for r in upto if r.step == last)   # the first of the latest occurrences
            b = _bits(rep)
            if b is None:
                undecided += 1
            elif b == (1, 1, 1, 0):
                passing += 1
        chars = step_chars.get(s, 0) if step_chars else 0
        out[s] = {"pes": passing, "per_1000_chars": (passing / (chars / 1000) if chars else NAN),
                  "undecided_clusters": undecided, "n_clusters": n_cl}
    return out


def cohort_pes(records, persons, embed, step_chars, threshold: float = 0.85, steps=STEPS,
               clusters_by_person: dict | None = None) -> dict:
    """Cohort PES per step over all ``persons`` (a person without attributes counts 0).

    ``step_chars`` is a dict step -> characters (same for everyone) or person -> that dict."""
    by = defaultdict(list)
    for r in records:
        by[r.person].append(r)
    per = {}
    for p in persons:
        sc = step_chars.get(p) if step_chars and isinstance(next(iter(step_chars.values())), dict) else step_chars
        per[p] = pes_by_step(by.get(p, []), embed, sc, threshold, steps,
                             clusters=None if clusters_by_person is None else clusters_by_person.get(p, []))
    out = {}
    for s in steps:
        vals = [per[p][s]["pes"] for p in persons]
        rates = [per[p][s]["per_1000_chars"] for p in persons if not math.isnan(per[p][s]["per_1000_chars"])]
        out[s] = {"mean": _mean(vals), "median": statistics.median(vals) if vals else NAN,
                  "max": max(vals) if vals else NAN, "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                  "n_persons": len(vals), "mean_per_1000_chars": _mean(rates),
                  "undecided_clusters": sum(per[p][s]["undecided_clusters"] for p in persons)}
    return out
