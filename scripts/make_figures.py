#!/usr/bin/env python3
"""Rebuild every figure in figures/ from the aggregated results/ files only.

The figures never touch the private data; they read only the CSV/JSON in
results/. Each figure is saved as both PDF and PNG. Cases keep a consistent
colour and marker throughout, and the line figures use distinct markers and
line styles so they remain readable in grayscale.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

CASES = ("A", "B", "C", "D", "E", "F", "G", "H")

# Consistent per-case colour + marker + line style (grayscale-safe).
STYLE = {
    "A": ("#1f77b4", "o", "-"),
    "B": ("#d62728", "s", "--"),
    "C": ("#2ca02c", "^", "-."),
    "D": ("#ff7f0e", "D", ":"),
    "E": ("#9467bd", "p", "-"),
    "F": ("#000000", "*", "--"),
    "G": ("#17becf", "o", "-."),
    "H": ("#8c564b", "x", ":"),
}


def _read_csv(name: str) -> list[dict]:
    path = RES / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _by_case(rows: list[dict], key: str, val: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for r in rows:
        out.setdefault(r["case"], {})[int(r[key])] = float(r[val])
    return out


def fig_dir_per_step() -> None:
    data = _by_case(_read_csv("rq1_per_step.csv"), "step", "dir")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in CASES:
        if c not in data:
            continue
        col, mk, ls = STYLE[c]
        xs = sorted(data[c])
        ax.plot(xs, [data[c][x] for x in xs], marker=mk, color=col, linestyle=ls,
                label=f"Case {c}", linewidth=1.6, markersize=7)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["S1 (25%)", "S2 (50%)", "S3 (75%)", "S4 (100%)"])
    ax.set_xlabel("Exposure step")
    ax.set_ylabel("Dangerous Inference Rate")
    ax.set_ylim(0.55, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.set_title("DIR per exposure step (majority vote)")
    _save(fig, "rq1_dir_per_step")


def fig_pes_per_step() -> None:
    rows = _read_csv("rq1_pes_per_step.csv")
    data = _by_case(rows, "step", "pes_paper_text")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in CASES:
        pts = {s: v for s, v in data.get(c, {}).items()}
        if len(pts) < 2:  # F has only the final step
            continue
        col, mk, ls = STYLE[c]
        xs = sorted(pts)
        ax.plot(xs, [pts[x] for x in xs], marker=mk, color=col, linestyle=ls,
                label=f"Case {c}", linewidth=1.6, markersize=7)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["S1 (25%)", "S2 (50%)", "S3 (75%)", "S4 (100%)"])
    ax.set_xlabel("Exposure step")
    ax.set_ylabel("Cumulative PES per person")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.set_title("Cumulative PES per step (paper definition: P,C,R and not E)")
    _save(fig, "rq1_pes_per_step")


def fig_quality_profile() -> None:
    rows = _read_csv("rq1_global.csv")
    metrics = [("plausibility", "Plaus."), ("coherence", "Coh."), ("realism", "Real."),
               ("explicitness", "Expl."), ("iqs", "IQS"), ("hallucination", "Hall.")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n = len(metrics)
    width = 0.1
    for i, r in enumerate(rows):
        c = r["case"]
        col = STYLE[c][0]
        xs = [j + i * width for j in range(n)]
        vals = [float(r[m]) if r[m] not in ("", None) else 0.0 for m, _ in metrics]
        ax.bar(xs, vals, width=width, color=col, label=f"Case {c}", edgecolor="white", linewidth=0.3)
    ax.set_xticks([j + width * (len(rows) - 1) / 2 for j in range(n)])
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=8, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Per-case quality profile (majority vote)")
    _save(fig, "rq1_quality_profile")


def fig_judge_spread() -> None:
    judges = _read_csv("rq2_judges.csv")
    glob = {r["case"]: r for r in _read_csv("rq1_global.csv")}
    cases = [c for c in CASES if any(r["case"] == c for r in judges)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in ((axes[0], "dir", "DIR"), (axes[1], "hallucination", "Hallucination")):
        for xi, c in enumerate(cases):
            ys = [float(r[metric]) for r in judges if r["case"] == c]
            ax.scatter([xi] * len(ys), ys, color=STYLE[c][0], marker="o", s=40,
                       alpha=0.8, zorder=3)
            if c in glob and glob[c][metric] not in ("", None):
                ax.scatter([xi], [float(glob[c][metric])], color="black", marker="_",
                           s=400, linewidths=2.2, zorder=4)
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels([f"{c}" for c in cases])
        ax.set_xlabel("Case")
        ax.set_ylabel(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(f"{title}: individual judges (dots) vs majority (bar)")
    _save(fig, "rq2_judge_spread")


def fig_detection() -> None:
    gen = _read_csv("rq4_generated_classical_f1.csv")
    ceas = _read_csv("rq4_benchmark_ceas08_f1.csv")
    if not gen and not ceas:
        return
    cases = [c for c in CASES if any(r["case"] == c for r in gen + ceas)]

    def _minmax(rows: list[dict]) -> dict[str, tuple[float, float]]:
        out: dict[str, list[float]] = {}
        for r in rows:
            out.setdefault(r["case"], []).append(float(r["f1"]))
        return {c: (min(v), max(v)) for c, v in out.items()}

    gmm, cmm = _minmax(gen), _minmax(ceas)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = range(len(cases))
    w = 0.38
    for i, c in enumerate(cases):
        if c in cmm:
            lo, hi = cmm[c]
            ax.bar(i - w / 2, hi - lo + 1e-4, bottom=lo, width=w, color="#999999",
                   label="CEAS_08 (classical, min-max)" if i == 0 else None)
        if c in gmm:
            lo, hi = gmm[c]
            ax.bar(i + w / 2, hi - lo + 1e-4, bottom=lo, width=w, color=STYLE[c][0],
                   label="Generated (classical, min-max)" if i == 0 else None)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"Case {c}" for c in cases], rotation=0)
    ax.set_ylabel("F1 (classical classifiers)")
    ax.set_ylim(0.8, 1.005)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Detection F1: external benchmark (CEAS_08) vs generated corpus")
    _save(fig, "rq4_detection_f1")


def fig_llm_detector() -> None:
    rows = _read_csv("rq4_generated_llm_f1.csv")
    if not rows:
        return
    cases = [c for c in CASES if any(r["case"] == c for r in rows)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for xi, c in enumerate(cases):
        ys = [float(r["f1"]) for r in rows if r["case"] == c]
        ax.scatter([xi] * len(ys), ys, color=STYLE[c][0], marker="o", s=35, alpha=0.8)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([f"Case {c}" for c in cases])
    ax.set_xlabel("Case")
    ax.set_ylabel("F1 on generated corpus")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("LLM detectors (zero-/few-shot) F1 per case")
    _save(fig, "rq4_llm_detector_f1")


def fig_pes_distribution() -> None:
    cases, means, maxes = [], [], []
    for c in CASES:
        p = RES / f"case{c}_metrics.json"
        if not p.exists():
            continue
        dist = json.loads(p.read_text(encoding="utf-8"))["pes"]["paper_text"]["final_distribution"]
        if not dist:
            continue
        cases.append(c)
        means.append(dist["mean"])
        maxes.append(dist["max"])
    if not cases:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = range(len(cases))
    ax.bar(list(xs), means, color=[STYLE[c][0] for c in cases], label="mean")
    ax.scatter(list(xs), maxes, color="black", marker="v", zorder=3, label="max")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"Case {c}" for c in cases])
    ax.set_ylabel("Final-step PES per person")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("PES per person: cohort mean and maximum (paper definition)")
    _save(fig, "pes_distribution")


def main() -> int:
    fig_dir_per_step()
    fig_pes_per_step()
    fig_quality_profile()
    fig_judge_spread()
    fig_detection()
    fig_llm_detector()
    fig_pes_distribution()
    print(f"Figures written to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
