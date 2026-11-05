# Rerun design (approved scope, 2026-10-20)

Supersedes the scope of `docs/RERUN_PLAN.md` (kept for history). Decisions taken
with the corresponding author on 2026-10-20.

## 1. Intent

Re-run every experiment of the paper with one corrected, harmonised pipeline and
deliver results that replace the old ones in the paper (no old/new comparison:
the old results are known to be wrong). The author will then rewrite and review
the paper with the new results.

Why: the code review found defects that invalidate the old numbers (truncated
inputs, judges reading raw HTML, output lines counted as attributes, cumulative
double counting in PES, single-judge filtering for the generation stage,
shortcut features and leakage in detection, unseeded runs, stale caches) and
design confounds (different models, judges and filters per case).

## 2. Goals

1. Corrected pipeline (section 4), one configuration for all cases (section 5).
2. Results for RQ1-RQ4 for every available case, plus the additions in section 7.
3. Deliverables (section 9): CSV/JSON numbers, a summary per RQ, LaTeX tables
   and figures in the paper's style, a methods note.
4. Every step verified (section 8) before the next one starts.

Non-goals: comparison with the old results; new crawling of portals (unless the
author decides it for Case D); agentic or retrieval-augmented attackers; any
sending of generated messages.

Anti-goals (failure even if it runs): numbers that cannot be traced to logged
calls; personal data in tracked files; results driven by artefacts (formatting,
source differences) rather than content; silent failures counted as answers.

## 3. Cases and data

| Case | Source data | Status |
|---|---|---|
| A, B, C, E, H | original repository (Case E: read from git objects, both variants of case-colliding names kept) | available |
| F | texts inside the case JSON | available |
| G | archive from the student, `data/raw/caseG/Archivio.zip` (MURIN 80 html, MURDOWN 550 pdf); exclude `__MACOSX` | available |
| D | requested from the student | pending; if it does not arrive, ask the author (re-crawl or drop) |

Only natural persons are analysed (Case G has organisational units among its
80 profiles: exclude them and report the count).
The mapping case -> original folder is in the paper repository
(`TO_CONTINUE.md`), never in this repository.

## 4. Pipeline corrections

1. Text: content of the documents only (no portal navigation), full length;
   OCR with the Italian language model. `num_ctx` set explicitly; texts longer
   than the context are split into overlapping windows and the attributes merged
   (profiles marked `chunked`).
2. Exposure steps 25/50/75/100%: by document date where dates exist (C, E, F; see the per-case coverage in RERUN_LOG.md);
   otherwise by document order, and the axis is described as amount of text.
   A step whose text equals the previous one is not re-run (marked `identical`).
3. Inference output: JSON with a fixed schema; one attribute per item with
   `attribute` and `category` (decided by the author on 2026-10-20 after the pilot:
   asking for an `evidence` sentence made the attacker produce 35 % fewer and more
   explicit attributes, DIR -0.10 on 30 persons of Case C; the variant with
   `evidence` is kept as a sensitivity analysis on those 30 persons). Invalid JSON is
   retried, then logged as failure; never parsed line by line.
4. Judges: the same cleaned text as the inference model; JSON answers validated;
   one attribute per call for all three judges, with the other attributes of the step
   given as context. Decision of 2026-10-20 after the pilot on Case C: batches of up to
   40 were not accepted (nemotron-3-ultra marks more attributes as explicit in a batch
   than alone, AC1 .69, also with batches of 10 and 5; on the final variant the mixed
   configuration kept the DIR only because the opposite errors of gemma4 and minimax
   cancelled out: explicit AC1 .73 and .78 on 300 attributes). Missing or invalid
   votes are retried, then counted separately.
   Majority vote only when all three votes are valid; otherwise `undecided`
   (reported).
5. Metrics: DIR (plausible, coherent, not explicit); hallucination as defined in
   the paper, with the distribution of all 16 label combinations reported;
   novelty exact and semantic. Semantic deduplication (decided 2026-10-20, after
   bge-m3 failed to separate paraphrases from close but different facts on short
   Italian attributes): candidate pairs of the same person with cosine >= 0.80,
   then confirmation by an LLM judge (`gemma4:31b`, temperature 0: same fact with
   the same values?); clusters = connected components of confirmed pairs. This is
   the main variant; exact strings and embedding-only clustering at 0.80/0.85/0.90
   are reported as sensitivity; merge precision checked by eye on 100 pairs in the pilot; PES = number of distinct (semantic dedup) attributes
   per person that are plausible, coherent, realistic and not explicit, also
   normalised per 1,000 characters of input. IQS: also report a variant
   restricted to non-explicit attributes.
6. Reproducibility: fixed seeds (also torch/transformers), model tags/digests and
   call dates logged, every answer stored in the cache; 3 repetitions on a subset to
   measure variability. Temperature 0 for non-reasoning models. Reasoning models
   (decided 2026-10-20 after a grid on real steps: at temperature 0 greedy decoding
   loops in the reasoning trace and truncates, e.g. medium 2/4, high 4/4 truncated):
   attacker `gpt-oss:20b` with reasoning effort `medium`, temperature 1.0, top_p 1.0,
   seed 42 (0/4 truncated); truncation rate monitored in the pilot with the 2 % alarm,
   fallback `low`. Judges: temperature 0, output cap 32,768 from the first call
   (their reasoning is long, not looping). Cloud answers are NOT reproducible even
   with a fixed seed (four identical calls gave four different texts): the stored
   cache is the record, and run-to-run variance is measured with 3 complete runs on
   50 persons of Case E and 30 of Case C, and folded into the reported intervals.
   An empty attribute list from the attacker is retried once with a new seed
   (counted and reported).
7. Statistics: 95% CIs by person-level cluster bootstrap; trend test across
   steps (mixed model with person as random effect, or permutation test);
   inter-judge agreement (Fleiss kappa, Krippendorff alpha).

## 5. Configuration (same for all cases)

- Attacker model (inference and message generation): `gpt-oss:20b`.
- Judges: `gemma4:31b` (Google), `nemotron-3-ultra` (NVIDIA), `minimax-m3.1-flash`
  (MiniMax): no judge from the attacker's family. Changed on 2026-10-20 from
  `deepseek-v4.1-flash` and `mistral-large-3`, which are not available on the free
  plan of the 32 keys (HTTP 402). Fallback if one is too slow or unstable in the
  pilot: `nemotron-3-super`; any change is logged.
- Model-size ablation (one case, same judges and prompts): `llama3.2:3b` (local),
  `gpt-oss:20b`, `gpt-oss:120b`.
- LLM detectors for RQ4: chosen in the RQ4 step, all different from the
  generator.
- Embeddings: `bge-m3` (local). Local GPU: RTX 5090.

## 6. RQ3/RQ4 (generation and detection)

Generation (RQ3) is part of the measurement only: messages are produced for the
detection benchmark and never sent. The generation step uses the attributes
selected by the majority vote, with one filter for all cases.

Detection (RQ4) is redesigned:
- negative class generated by the same model under the same conditions
  (legitimate institutional messages), so that both classes share source and style;
- main setting: detectors trained only on the public benchmarks and tested on
  the generated set; secondary settings: cross-case and cross-generator splits;
- grouping by person (no person in both train and test); ensemble members chosen
  without looking at the test set;
- artefact controls: remove organisation names, fixed urgency formulas,
  signatures, placeholders; length-only and style-only baselines; report the
  top-weighted features;
- realistic base rates (1:10, 1:100) with AUPRC and recall at 1% FPR, besides F1;
- full messages to the detectors (no truncation); detector errors and
  unparseable outputs counted separately, never mapped to a label;
- public benchmarks run once and reused for every case.
- Limit to state: benchmarks are English, generated messages Italian.

## 7. Additions

1. Human validation: stratified sample of 400-500 attributes (50-60 per case, by
   step and by majority outcome), two author-annotators, a third adjudicates.
   `docs/codebook.md` defines the criteria, answer format, examples and edge
   cases (partial or paraphrased disclosure, obvious inference, contradictory
   documents, vague attributes). Calibration round on 30-50 attributes first.
   Metrics: Cohen kappa and percent agreement per criterion, Gwet AC1, Fleiss
   kappa / Krippendorff alpha among judges, Cohen kappa judge-vs-human,
   precision/recall of the majority vote per criterion and for the DIR
   condition, bootstrap CIs. Annotation is done by the authors after the rerun;
   the pipeline prepares the sample and an annotation sheet.
2. Three message variants per person (no personal attributes, explicit-only,
   explicit + inferred): detection rate and a personalisation rubric (LLM panel
   plus human check on a sample).
3. Mitigation: minimisation policies applied to the documents before inference
   (field generalisation, last-k-years window, k documents per person); DIR/PES
   reduction against the share of information kept.
4. Ablations on one case: model size; number of documents (1, 2, 4, 8);
   chronological vs shuffled order (Case E; see 10b.1).
5. Control without documents: the model receives only the role and the type of
   organisation (no name, no documents); same judges.
6. Attribute categories, including GDPR Art. 9 special categories; DIR per category.
7. Cost and time per profile (tokens, wall time, hardware).
8. Re-identification check (uniqueness of attribute sets) before any data release.

## 8. Verification at every step

- After each task and subtask the executing session re-reads its work, runs unit
  tests (parsers, majority vote, metrics on toy data with known results) and
  invariant checks (persons in = persons out, no lost attributes, call counts).
- Cross-review: the main session pulls branch `rerun`, reviews code and result
  summaries, and gives the go-ahead before the next step.
- `RERUN_LOG.md` is updated at every step: done (with checks), next, open
  issues, decisions.
- An error found later sends the work back to the faulty step; everything
  downstream is redone and logged.
- Automatic alarms stop a stage: empty or invalid outputs above 2%, any F1 >= 0.99,
  persons lost between stages, key quota errors above 5%.

## 9. Deliverables

- `results/rerun/` (aggregates only): CSV/JSON per case and RQ, `SUMMARY_RQ1..4.md`.
- `results/rerun/latex/`: tables and pgfplots figures in the paper's style.
- `docs/METHODS_NOTE.md`: what changed in the method, for Methodology and Threats.
- Raw outputs in `data/private/` (git-ignored), never committed.

## 10. Engineering

- Keys from `.secrets/ollama_keys.json`, round-robin with rate limiting and
  back-off; usage logged per key index only; keys never printed or committed.
- Cache keyed by hash of (model, prompt, options, input); JSONL checkpoint per
  call; resumable.
- Parallel calls across keys; judges in parallel; local GPU for gemma/embeddings/
  Transformers.
- Work on branch `rerun` of this repository; push after each verified step.

## 10b. Resolutions after the builder check (calls of the main session, 2026-10-20)

1. Budget and order. No deadline given; the run proceeds in priority order so
   that a quota shortfall cuts only the least important parts:
   (a) RQ1-RQ2 core on all available cases; (b) RQ3-RQ4 core; (c) control without
   documents (all cases) and three message variants (all cases); (d) ablations,
   mitigation and repetitions on one case only: Case E (changed from Case B on
   2026-10-20 because the documents of B are not available; E has full dates on
   99.3 % of its documents and a median span of 2.9 years per person);
   mitigation uses three policies with two values of k each.
   Batched judging is accepted if, on 300 attributes, Cohen kappa against
   per-attribute judging is >= 0.80 for every criterion; otherwise judging stays
   per attribute for the core run and batching is not used. Outcome (pilot, Case C):
   not accepted, the core judges one attribute per call (see item 4 of section 3).
2. Cross-generator split: one second generator from a different family (not a
   judge, not a detector), chosen and logged in the RQ4 step, on all cases.
3. Message content: attributes = majority-validated, plausible, coherent and
   realistic, at most 12, taken from the 100 % step. Variant "explicit-only" uses
   a separate extraction call of the attacker model that lists facts stated in the
   documents (also reported as an extraction baseline); variant "inferred" adds
   the non-explicit validated attributes. Negative class: legitimate institutional
   messages generated for the same person with the same name and the same
   attributes, so both classes are equally personalised.
4. Categories and special categories (ethical choice, conservative default; the
   author may change it): the inference prompt does NOT ask for GDPR Art. 9 data.
   Categories come from a separate post-hoc classification pass with a fixed
   taxonomy (role and career, organisation and network, competences, mobility and
   location, economic, special categories), run by one judge model and checked on
   the human sample.
5. Chunked profiles: judges receive the full text whenever it fits their context;
   otherwise explicitness is evaluated against every window (explicit if any window
   states it) and the other criteria against the originating window plus the
   evidence. The number of profiles in this situation is reported.
6. PES: attributes are grouped into semantic clusters per person; a cluster counts
   if its majority labels at its latest occurrence (most informed text) are
   plausible, coherent, realistic and not explicit. Cumulative PES at step s =
   clusters first seen at or before s that pass at their latest occurrence <= s.
   Normalisation per 1,000 characters of the text up to that step.
7. Handshake: at the end of a step the executor pushes, writes `REVIEW <step>` in
   `RERUN_LOG.md` and messages the main session; the main session answers with
   `GO <step>` (message and log). While waiting, the executor may run stages that
   do not depend on the step under review. Progress reports to the author come
   from the main session.
8. Case F follows the common design (independent steps, common models); its
   history-carrying variant is not kept, and its description in the paper changes.

## 11. When to interrupt the author

Only for decisions that are the author's: Case D data not arriving; a result
that contradicts a claim of the paper; quota exhausted on all keys; any ethical
or legal question. Otherwise the two sessions resolve issues themselves.
Progress reports to the author every 30 minutes.
