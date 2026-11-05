# Figures

Each figure is saved as PDF (vector) and PNG (200 dpi) and is rebuilt from
`results/` only:

```bash
python scripts/make_figures.py
```

Cases keep a consistent colour and marker across figures, and the line plots
use distinct markers and line styles so they stay readable in grayscale.

| File | Caption |
|------|---------|
| `rq1_dir_per_step` | Dangerous Inference Rate per exposure step (25/50/75/100 %), one line per case (majority vote). The DIR is high and nearly flat across steps in every case; Case F appears only at the final step. |
| `rq1_pes_per_step` | Cumulative Privacy Exposure Score per person per exposure step, paper definition (plausible, coherent, realistic and not explicit). Exposure grows almost linearly in every case. |
| `rq1_quality_profile` | Per-case quality profile: plausibility, coherence, realism, explicitness, IQS and hallucination rate (global majority vote). |
| `rq2_judge_spread` | Spread of the three individual judges (coloured dots) against the majority vote (black bar), for DIR and for the hallucination rate. Shows how far a single-judge estimate can drift from the consensus. |
| `rq4_detection_f1` | Detection F1 of the classical classifiers: external benchmark (CEAS_08) vs the generated corpus, as a min-max range per case. The generated corpus is separated almost perfectly; the benchmark is strong but imperfect. |
| `rq4_llm_detector_f1` | F1 of the LLMs used directly as zero-/few-shot detectors on the generated corpus, per case. Mostly near-perfect, with occasional outright failures. |
| `pes_distribution` | Per-person PES at the final step: cohort mean (bar) and maximum (marker) per case, paper definition. The distributions are right-skewed. |

Notes:

- `rq1_pes_per_step` and `pes_distribution` use the paper's PES definition.
  The alternative (originally published) counting is in
  `results/rq1_pes_per_step.csv` and the per-case JSON if a comparison plot
  is needed.
- Case F is omitted from the per-step line plots (its validation was applied
  at the final step only) and reflects the earlier run present in the
  repository (see `results/CONSISTENCY.md`).
