"""Tables of the results of the rerun from ``results/rerun/case*/metrics.json`` (CSV and LaTeX, aggregates only).

    python -m rerun.report [--base results/rerun] [--out results/rerun]

Writes ``<out>/tables/<name>.csv`` and ``<out>/latex/<name>.tex`` (booktabs, no caption: the paper sets it) for

* ``rq1_final``: per case, persons, attributes and P, C, R, E, IQS, DIR (95% CI) and hallucination at the 100% step;
* ``rq2_steps``: per case and step, DIR, IQS, PES (confirmed de-duplication) and novelty;
* ``trend``: per case and metric, mean within-person slope per step with the permutation p-value and the mixed model;
* ``reidentification``: per case, share of persons with a key held by one person and uniqueness of combinations;
* ``categories``: per case and category, DIR, IQS and hallucination at the final step (``metrics_by_category.json``);
* ``control``: per case, the paired DIR with the documents and without them (``control_comparison.json``);
* ``rq4``: per run, setting, part and detector, the metrics as generated of ``rerun.rq4_run`` (``<base>/rq4/*.json``);
* ``ablations`` and ``ablation_repetitions``: per case and condition, the paired DIR reduction against the baseline
  and the spread between repeated runs, from ``rerun.ablation`` (``<base>/ablation/*.json``).

A block missing from a metrics file (older runs) gives empty cells, never a made-up value.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from . import config

STEPS = ("25", "50", "75", "100")


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def load(base) -> dict:
    out = {}
    for p in sorted(Path(base).glob("case*/metrics.json")):
        out[p.parent.name[4:]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def rq1_final(ms: dict):
    head = ["case", "persons", "attributes", "P", "C", "R", "E", "IQS", "DIR", "DIR_lo", "DIR_hi", "hallucination"]
    rows = []
    for c, m in ms.items():
        f = _get(m, "by_step", "100") or {}
        ci = _get(m, "ci_final_step", "DIR") or {}
        rows.append([c, m.get("persons_in_cohort", m.get("persons")), f.get("n_total"), *(_f(f.get(k)) for k in "PCRE"),
                     _f(f.get("IQS")), _f(f.get("DIR")), _f(ci.get("lo")), _f(ci.get("hi")), _f(f.get("hallucination"))])
    return head, rows


def rq2_steps(ms: dict):
    head = ["case", "step", "DIR", "IQS", "PES_mean", "PES_per_1000_chars", "novelty_confirmed", "novelty_exact"]
    rows = []
    for c, m in ms.items():
        for s in STEPS:
            b = _get(m, "by_step", s) or {}
            pes = _get(m, "pes", "confirmed", s) or {}
            rows.append([c, s, _f(b.get("DIR")), _f(b.get("IQS")), _f(pes.get("mean"), 2),
                         _f(pes.get("mean_per_1000_chars")), _f(_get(m, "novelty_confirmed", s)),
                         _f(_get(m, "novelty_exact", s))])
    return head, rows


def trend(ms: dict):
    head = ["case", "metric", "slope_per_step", "p_permutation", "n_units", "mixed_slope", "mixed_p", "mixed_converged"]
    rows = []
    for c, m in ms.items():
        for name, t in (m.get("trend") or {}).items():
            p, mm = t.get("permutation") or {}, t.get("mixed_model") or {}
            rows.append([c, name, _f(p.get("slope"), 4), _f(p.get("p_value"), 4), p.get("n_units", ""),
                         _f(mm.get("slope"), 4), _f(mm.get("p_value"), 4), mm.get("converged", "")])
    return head, rows


def reidentification(ms: dict):
    head = ["case", "population", "keys", "persons_with_unique_key", "keys_held_by_one_person",
            "unique_by_up_to_2_keys", "unique_by_up_to_3_keys", "identical_sets"]
    rows = []
    for c, m in ms.items():
        for pop in ("all", "dangerous"):
            e = _get(m, "reidentification", pop, "exact") or {}
            comb = e.get("combination_uniqueness") or {}
            rows.append([c, pop, e.get("n_distinct_keys", ""), _f(e.get("persons_with_unique_key")), _f(e.get("keys_k1")),
                         _f(comb.get("2")), _f(comb.get("3")), _f(e.get("identical_sets"))])
    return head, rows


TABLES = {"rq1_final": rq1_final, "rq2_steps": rq2_steps, "trend": trend, "reidentification": reidentification}


def rq4(base):
    """One row per run, setting, part and detector from ``<base>/rq4/*.json`` (``rerun.rq4_run``): F1, AUC, AUPRC
    and recall at 1% FPR as generated with the CI of the F1, plus the alarm flag and the number of messages."""
    head = ["run", "setting", "part", "detector", "n_test", "n_positive", "AUC", "AUPRC", "F1", "F1_lo", "F1_hi",
            "recall_at_1pct_FPR", "alarm"]
    rows = []
    for p in sorted(Path(base).glob("rq4/*.json")):
        res = json.loads(p.read_text(encoding="utf-8"))
        for s, sv in (res.get("settings") or {}).items():
            for part, pv in (sv.get("parts") or {}).items():
                for det, e in (pv.get("detectors") or {}).items():
                    g = _get(e, "metrics", "as_generated") or {}
                    f1 = g.get("f1") or {}
                    rows.append([p.stem, s, part, det, e.get("n_test", ""), e.get("n_positive", ""),
                                 _f((g.get("auc_roc") or {}).get("estimate")), _f((g.get("auprc") or {}).get("estimate")),
                                 _f(f1.get("estimate")), _f(f1.get("lo")), _f(f1.get("hi")),
                                 _f((g.get("recall_at_fpr") or {}).get("estimate")), e.get("alarm", "")])
    return head, rows


def categories(base):
    """One row per case and category from ``<base>/case*/metrics_by_category.json`` (``--stage category-metrics``):
    attributes, DIR, IQS and hallucination at the final step; ``_special`` is the GDPR art. 9 subset with its share of
    all the attributes of the step, ``_unclassified`` the attributes without a classification."""
    head = ["case", "category", "n", "n_complete", "share_of_attributes", "DIR", "IQS", "hallucination"]
    rows = []
    for p in sorted(Path(base).glob("case*/metrics_by_category.json")):
        res = json.loads(p.read_text(encoding="utf-8"))
        cats = res.get("categories") or {}
        for k in sorted(cats):
            c = cats[k]
            n = c.get("n", c.get("n_special", ""))
            share = c.get("share") if k == "_special" else (n / res["n_records"] if res.get("n_records") and n != "" else None)
            rows.append([p.parent.name[4:], k, n, c.get("n_complete", ""), _f(share), _f(c.get("DIR")), _f(c.get("IQS")),
                         _f(c.get("hallucination"))])
    return head, rows


def control(base):
    """One row per case from ``<base>/case*/control_comparison.json`` (``--stage control``): the paired DIR with the
    documents and without them (role and organisation type only) and the difference with its CI."""
    head = ["case", "n_persons", "persons_without_pair", "DIR_documents", "DIR_control", "difference", "difference_lo",
            "difference_hi", "IQS_documents", "IQS_control", "explicit_documents", "explicit_control"]
    rows = []
    for p in sorted(Path(base).glob("case*/control_comparison.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        diff = c.get("DIR_difference") or {}
        rows.append([p.parent.name[4:], c.get("n_persons", ""), c.get("persons_without_pair", ""),
                     _f(_get(c, "DIR_documents", "mean")), _f(_get(c, "DIR_control", "mean")), _f(diff.get("estimate")),
                     _f(diff.get("lo")), _f(diff.get("hi")), _f(_get(c, "IQS", "documents")), _f(_get(c, "IQS", "control")),
                     _f(_get(c, "explicit_share", "documents")), _f(_get(c, "explicit_share", "control"))])
    return head, rows


def ablations(base):
    """One row per case and condition from ``<base>/ablation/*.json`` (``rerun.ablation``): DIR of the baseline and of
    the variant on the same persons, the paired reduction with its CI, the IQS, and the information kept."""
    head = ["case", "condition", "persons_run", "n_persons", "DIR_baseline", "DIR_variant", "reduction", "reduction_lo",
            "reduction_hi", "IQS_baseline", "IQS_variant", "characters_kept", "documents_kept", "status"]
    rows = []
    for p in sorted(Path(base).glob("ablation/*.json")):
        res = json.loads(p.read_text(encoding="utf-8"))
        for spec, c in (res.get("conditions") or {}).items():
            red = c.get("DIR_reduction") or {}
            kept = c.get("info_kept") or {}
            rows.append([res.get("case", p.stem), spec, c.get("persons_run", ""), c.get("n_persons", ""),
                         _f(_get(c, "DIR_baseline", "mean")), _f(_get(c, "DIR_variant", "mean")), _f(red.get("estimate")),
                         _f(red.get("lo")), _f(red.get("hi")), _f(_get(c, "IQS", "baseline")), _f(_get(c, "IQS", "variant")),
                         _f(kept.get("characters_pooled")), _f(kept.get("documents_pooled")), c.get("status", "")])
    return head, rows


def ablation_repetitions(base):
    """One row per case from ``<base>/ablation/*.json``: the final-step DIR of each repeated run, its spread between
    runs and the mean absolute per-person difference between runs."""
    head = ["case", "runs", "persons_in_all_runs", "DIR_by_run", "DIR_sd_between_runs", "DIR_range",
            "mean_abs_person_difference", "status"]
    rows = []
    for p in sorted(Path(base).glob("ablation/*.json")):
        res = json.loads(p.read_text(encoding="utf-8"))
        v = res.get("repetitions")
        if not v:
            continue
        by_run = "; ".join(f"{t}={_f(x)}" for t, x in (v.get("DIR_by_run") or {}).items())
        rows.append([res.get("case", p.stem), " ".join(v.get("runs") or []), v.get("persons_in_all_runs", ""), by_run,
                     _f(v.get("DIR_sd_between_runs")), _f(v.get("DIR_range")),
                     _f(v.get("mean_abs_person_difference_between_runs")), v.get("status", "")])
    return head, rows


def to_tex(head, rows) -> str:
    esc = lambda s: str(s).replace("_", r"\_")
    lines = [r"\begin{tabular}{" + "l" * 2 + "r" * (len(head) - 2) + "}", r"\toprule",
             " & ".join(esc(h) for h in head) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(x) for x in r) + r" \\" for r in rows]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def write_all(base, out) -> dict:
    ms = load(base)
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "latex").mkdir(parents=True, exist_ok=True)
    written = {}
    jobs = [(name, lambda fn=fn: fn(ms)) for name, fn in TABLES.items()] + [("categories", lambda: categories(base)),
                                                                          ("control", lambda: control(base)),
                                                                          ("rq4", lambda: rq4(base)),
                                                                          ("ablations", lambda: ablations(base)),
                                                                          ("ablation_repetitions", lambda: ablation_repetitions(base))]
    for name, job in jobs:
        head, rows = job()
        with open(out / "tables" / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(rows)
        (out / "latex" / f"{name}.tex").write_text(to_tex(head, rows), encoding="utf-8")
        written[name] = len(rows)
    return {"cases": sorted(ms), "rows": written}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(config.ROOT / "results" / "rerun"))
    ap.add_argument("--out", default=str(config.ROOT / "results" / "rerun"))
    a = ap.parse_args(argv)
    print(json.dumps(write_all(a.base, a.out), indent=1))


if __name__ == "__main__":
    main()
