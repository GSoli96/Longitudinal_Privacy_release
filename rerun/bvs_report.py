"""Analysis of the batched vs one-by-one judging of the pilot (plan T8, spec 10b.1).

Input: the rows of ``run_case.run_batch_vs_single`` (per attribute and judge, the votes of the batched run and of the
single-attribute run). Output: per judge and criterion the 2x2 table (rows = batch, columns = single), percent
agreement, Cohen's kappa, Gwet's AC1 and the exact McNemar test; DIR, explicit share (E) and IQS with the majority of
the three judges for both ways of judging, with the difference and a person-level cluster bootstrap interval; where
the disagreements are (criterion, judge, position of the attribute in its step). Invalid votes are left out.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from scipy.stats import binom

from . import judge, metrics, stats

CRITERIA = judge.CRITERIA


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b (batch yes, single no) and c (batch no, single yes)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * float(binom.cdf(min(b, c), n, 0.5)))


def position_of(attr_id: str):
    m = re.search(r"-a(\d+)$", attr_id)
    return int(m.group(1)) if m else None


def _yn(x):
    return x if x in ("yes", "no") else None


def agreement_tables(rows) -> dict:
    judges = list(rows[0]["batch"]) if rows else []
    out = {}
    for j in judges:
        out[j] = {}
        for i, c in enumerate(CRITERIA):
            a = [_yn(r["batch"][j][i]) for r in rows]
            b = [_yn(r["single"][j][i]) for r in rows]
            pairs = [(x, y) for x, y in zip(a, b) if x and y]
            t = {"yes_yes": 0, "yes_no": 0, "no_yes": 0, "no_no": 0}
            for x, y in pairs:
                t[f"{x}_{y}"] += 1
            n = len(pairs)
            out[j][c] = {"n_valid_pairs": n, "table": t,
                         "percent_agreement": (t["yes_yes"] + t["no_no"]) / n if n else float("nan"),
                         "cohen_kappa": stats.cohen_kappa(a, b),
                         "gwet_ac1": stats.gwet_ac1([[x, y] for x, y in pairs], n_categories=2) if pairs else float("nan"),
                         "mcnemar_p": mcnemar(t["yes_no"], t["no_yes"]),
                         "bias": ((t["yes_yes"] + t["yes_no"]) - (t["yes_yes"] + t["no_yes"])) / n if n else float("nan")}
    return out


def majority_labels(rows, which: str) -> dict:
    """{attribute id: {criterion: yes/no/undecided}} with the majority of the judges of ``which`` (batch or single)."""
    out = {}
    for r in rows:
        votes = [judge.Vote(*r[which][j]) for j in r[which]]
        out[r["attr_id"]] = judge.majority(votes)
    return out


def _records(rows, which, person_of):
    labs = majority_labels(rows, which)
    return [metrics.AttrRecord(person_of[r["attr_id"]], 100, r["attr_id"], labs[r["attr_id"]], False, r["attr_id"])
            for r in rows]


def compare_metrics(rows, person_of: dict, *, n_resamples: int = 2000, seed: int = 42) -> dict:
    """DIR, E and IQS with batched and single votes (majority of the judges) and their difference (batch - single)."""
    rb, rs = _records(rows, "batch", person_of), _records(rows, "single", person_of)
    mb, ms = metrics.basic_metrics(rb), metrics.basic_metrics(rs)
    by_person = defaultdict(list)
    for a, b in zip(rb, rs):
        by_person[a.person].append((a, b))
    units = list(by_person.values())

    def diff(key):
        def stat(us):
            x = metrics.basic_metrics([p[0] for u in us for p in u])[key]
            y = metrics.basic_metrics([p[1] for u in us for p in u])[key]
            return x - y
        return stats.cluster_bootstrap_ci(units, stat, n_resamples=n_resamples, seed=seed)

    pick = lambda m: {k: m[k] for k in ("DIR", "E", "IQS", "P", "C", "R", "n_complete", "n_undecided")}
    return {"n_attributes": len(rows), "n_persons": len(units), "batch": pick(mb), "single": pick(ms),
            "difference": {k: diff(k) for k in ("DIR", "E", "IQS")}}


def batching_rule(compare: dict, tables: dict, *, delta: float = 0.03, ac1_min: float = 0.80) -> dict:
    """Rule proposed by the main session: batching is accepted if the DIR difference has an interval that includes 0
    and |difference| < delta, and the explicit criterion (which decides the DIR) has AC1 >= ac1_min for every judge."""
    d = compare["difference"]["DIR"]
    dir_ok = (d["lo"] <= 0 <= d["hi"]) and abs(d["estimate"]) < delta
    ac1 = {j: t["explicit"]["gwet_ac1"] for j, t in tables.items()}
    expl_ok = all(v >= ac1_min for v in ac1.values()) if ac1 else False
    return {"dir_difference_ok": dir_ok, "explicit_ac1_ok": expl_ok, "explicit_ac1": ac1,
            "accepted": bool(dir_ok and expl_ok)}


def disagreement_by_position(rows, size_of) -> dict:
    """Disagreement rate between batch and single, per criterion, for the first and the second half of the step's
    attribute list (pooled over judges; ``size_of(attr_id)`` is the number of attributes of the step) and for the
    positions 1-3, 4-6 and 7 or more."""
    cnt = {c: {k: [0, 0] for k in ("first_half", "second_half", "pos_1_3", "pos_4_6", "pos_7_plus")} for c in CRITERIA}
    for r in rows:
        pos, size = position_of(r["attr_id"]), size_of(r["attr_id"])
        if pos is None or not size:
            continue
        half = "first_half" if pos <= math.ceil(size / 2) else "second_half"
        band = "pos_1_3" if pos <= 3 else "pos_4_6" if pos <= 6 else "pos_7_plus"
        for j in r["batch"]:
            for i, c in enumerate(CRITERIA):
                a, b = _yn(r["batch"][j][i]), _yn(r["single"][j][i])
                if a and b:
                    for k in (half, band):
                        cnt[c][k][0] += a != b
                        cnt[c][k][1] += 1
    return {c: {k: {"rate": (d / n if n else float("nan")), "n": n} for k, (d, n) in v.items()} for c, v in cnt.items()}
