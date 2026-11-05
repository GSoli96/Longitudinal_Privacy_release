# Consistency check: paper vs source

Every number in `results/` is recomputed from the original per-case
validation and detection outputs and compared with the corresponding value
in the paper. The recomputation uses `scripts/compute_stats.py` (majority
vote over the three judges, then the metric definitions in `src/metrics.py`).

Verdict in one line: **RQ1 and RQ2 reproduce the paper exactly for Cases B,
C, D, E, G, H; Case A matches within rounding with a small attribute-count
difference; Case F in the repository is a different, earlier run than the
one reported; RQ4 detection reproduces the paper except for the Case F
scenario table.** Two substantive discrepancies (Case A count, Case F) and
one metric-definition correction (PES) are detailed below.

## RQ1 global metrics (`tab:rq1-global`, `results/rq1_global.csv`)

| Case | Plaus/Coh/Real/Expl | IQS | DIR | Hall | Attributes | PES (S4) | Result |
|------|---------------------|-----|-----|------|------------|----------|--------|
| A | 0.831 / 0.750 / (0.786) / 0.008 | (0.476) | 0.735 | 0.155 | 11,528 vs **11,543** | (28.2) | match within rounding; see note A |
| B | 0.935 / 0.898 / 0.897 / 0.150 | 0.606 | 0.744 | 0.060 | 13,839 | 40.17 | exact |
| C | 0.856 / 0.847 / 0.891 / 0.215 | 0.605 | 0.620 | 0.101 | 4,277 | 27.83 | exact |
| D | 0.740 / 0.706 / 0.671 / 0.097 | 0.462 | 0.603 | 0.250 | 9,727 | 34.11 | exact |
| E | 0.885 / 0.849 / 0.842 / 0.073 | 0.544 | 0.773 | 0.105 | 18,051 | 50.24 | exact |
| F | 0.967 / 0.954 / 0.947 / 0.066 | 0.600 | 0.887 | 0.033 | 151 vs **117** | — | **differs**; see note F |
| G | 0.862 / 0.831 / 0.827 / 0.073 | 0.533 | 0.754 | 0.127 | 4,993 | 45.27 | exact |
| H | 0.928 / 0.917 / 0.904 / 0.160 | 0.614 | 0.753 | 0.064 | 25,165 | 39.38 | exact |

Values in parentheses are computed here but are reported as "--" in the
paper (A's realism, IQS and PES). Per-step DIR and hallucination
(`results/rq1_per_step.csv`) and the per-judge table
(`results/rq2_judges.csv`) were checked the same way and match the paper's
Appendix A and `tab:rq2-judges` for all cases that report them.

### Note A — attribute count (11,528 vs 11,543)

The paper's Case A per-step counts (2,632 / 2,900 / 2,955 / 3,056; total
11,543) come from the step-level metrics file produced at inference time
(`outputs/metrics/llama3.json`). The number of **distinct audited**
attributes reconstructed here from the three judges' files is 2,632 / 2,893
/ 2,955 / 3,048 = **11,528** (a 0.13 % difference, from a handful of
attributes that appear in the inference log but not in all judge files).
Every Case A rate (plausibility 0.831, coherence 0.750, explicitness 0.008,
DIR 0.735, hallucination 0.155) matches regardless.

### Note F — repository contains an earlier Case F run

The Case F validation file in the repository has **16 profiles and 151
distinct attributes at the final step** (12 at 100 %, 4 at 75 %), giving
plausibility 0.967, coherence 0.954, realism 0.947, explicitness 0.066, IQS
0.600, DIR 0.887, hallucination 0.033. The paper reports **18 profiles and
117 attributes** with plausibility 0.983, coherence 0.957, realism 0.966,
explicitness 0.026, IQS 0.592, DIR 0.932, hallucination 0.017. The two do
not reconcile: the released artefacts are an earlier run than the one in the
paper. **Reported as-is, not reconciled** — the authors should either
restore the exact run used in the paper or update the paper to the released
run.

## PES definition (corrected)

The paper defines PES as the per-person count of **fully validated dangerous**
attributes: `plausible and coherent and realistic and NOT explicit`,
summed over the four exposure steps and averaged over the cohort. The
default in `src/metrics.py` is this definition (`PES_CONDITION_PAPER_TEXT`),
and `results/` uses it. Recomputed cumulative means per step
(`results/rq1_pes_per_step.csv`, column `pes_paper_text`):

| Case | S1 | S2 | S3 | S4 |
|------|----|----|----|----|
| B | 9.96 | 19.98 | 29.97 | 40.17 |
| C | 6.54 | 13.26 | 20.56 | 27.83 |
| D | 8.31 | 17.01 | 25.69 | 34.11 |
| E | 12.26 | 25.06 | 37.68 | 50.24 |
| G | 11.07 | 22.00 | 33.24 | 45.27 |
| H | 9.03 | 18.86 | 29.02 | 39.38 |

These match the paper's updated `tab:rq1-global` and `fig:rq1-pes`.

**Correction recorded.** The first version of the paper reported PES values
that were produced by the original analysis code
(`audit_analysis.py`, line 190 and its per-case copies), which counted
attributes with `plausible and coherent and realistic and explicit == yes`.
That count equals the number of explicit, fully-validated attributes, not
the non-explicit dangerous attributes of the written definition. The
originally published (now corrected) final-step values were:

| Case | B | C | D | E | G | H |
|------|---|---|---|---|---|---|
| old PES (explicit count) | 8.38 | 9.68 | 5.96 | 4.97 | 4.58 | 8.60 |

Both conditions are reproduced here for audit: each `caseX_metrics.json`
records `pes.paper_text` and `pes.as_reported`, and
`results/rq1_pes_per_step.csv` lists both columns. The `as_reported` column
reproduces the old values exactly.

## RQ4 detection

- **Classical on CEAS_08** (`results/rq4_benchmark_ceas08_f1.csv`): matches
  the paper's `tab:rq4-ceas`. Example (Case B): Linear SVC F1 = 0.9978,
  AdaBoost = 0.9753.
- **Classical on the generated corpus**
  (`results/rq4_generated_classical_f1.csv`): F1 = 1.000 for every model in
  every case except where the paper also reports a lower value (Case G
  Decision Tree / Gradient Boosting 0.984), matching `tab:rq4-generated` and
  `fig:caseA-heatmap`.
- **LLM detectors** (`results/rq4_generated_llm_f1.csv`): match `tab:rq4-llm`,
  including the near-perfect large models and the failures (`gpt-oss:20b`
  collapsing on Case H, `deepseek-v3.2` near chance on Cases E and H).
- **Case F scenario table** (`tab:caseF-rq4`): the repository's Case F
  detection file reports different F1 values than the paper (e.g. Logistic
  Regression baseline 0.980 vs 0.900; Decision Tree Test 1 = 0.984 vs the
  paper's collapse to 0.105). This is the same earlier-run discrepancy as
  note F and is **reported, not reconciled**.

## Problems found in the source (summary)

1. **Case F**: the released validation and detection artefacts are an
   earlier run (16 profiles / 151 attributes / DIR 0.887) than the paper's
   (18 / 117 / 0.932). Needs the authors' decision.
2. **Case A**: 11,528 audited attributes vs 11,543 reported from the
   inference log (0.13 %); rates unaffected.
3. **PES**: original code counted explicit attributes, not the written
   definition; corrected here and in the paper.
4. Hard-coded API keys were present in two source scripts; they were removed
   and replaced by environment variables (never copied into this release).
