# Methods note for the rerun (draft, updated 2026-11-03: sections 1 partly corrected; sections 2 to 6 still to update)

Internal note for the authors. It lists where the rerun departs from the plan in `docs/plans/` and from the
paper as submitted, so that each departure can be declared in the text. Final counts are filled in from
`results/rerun/` when the runs end; nothing here is a result. Tool versions (Ollama builds) are in the internal
execution log only and stay out of the paper.

## 1. Inference and judging

- **Judging panel (corrected 2026-11-03).** The panel and the procedure differ across cases. For cases A, B, D, E and H
  the panel is sequential: the first two judges vote on every attribute; the third votes only where the first two
  differ on at least one criterion (or a vote is invalid) and on a hash-fixed audit sample of 10% of the attributes
  (`config.AUDIT_SHARE`, chosen by the SHA-256 of "audit:" plus the attribute id). Where the third judge does not vote,
  the label is the common value of the first two. For cases C, F and G all three judges vote on every attribute. A
  criterion is undecided when fewer than two valid and equal votes exist (`majority_of_two` in the sequential scheme;
  the module docstring that says "all three valid" describes the non-sequential scheme only). Judges: A, B, D and H
  use gemma4:31b-local, qwen3.5:9b and nemotron-3-nano:30b-local (all local); C, F and G use gemma4:31b, qwen3.5:9b and
  nemotron-3-ultra; case E has the same names as C, F and G, with 1,257 gemma votes and 874 nemotron votes filled by a
  local fallback (counts from the run summaries, 2026-11-03). The earlier statement that only D is judged locally is
  wrong. The ablations on case E use the local panel (gemma4:31b, qwen3.5:9b, nemotron-3-nano as tie-breaker).
  Agreement for A, B, D, E and H is computed on the audit items only (already pair-agreed, so biased upward and not
  comparable with C, F and G, which use all attributes); results per judge of the third judge rest on non-random
  subsets (ties and audit).
- **Local attacker.** Ablations and the local reruns use `gpt-oss:20b-local`, whose weights digest differs from the
  cloud model used for the main runs. Ablations run the final step only, on a sample of 50 persons.
- **Failed calls, invalid votes and undecided attributes (corrected 2026-11-03).** An invalid vote is never turned into
  "no". All invalid votes of B, D and H have one cause: the local-judge guard (`local_panel.guard`) refuses any call to
  a "-local" judge whose context would exceed 65,536 tokens and returns `too_long_for_local`; the model never sees the
  prompt. The rule is deterministic: prompts longer than 174,182 characters (step text plus the attribute plus the other
  attributes of the step) need the 262,144 context and are refused; it predicted every invalid attribute with no false
  positive (checked on B, D and H). qwen3.5:9b is not guarded and answered those prompts. Failed calls are exactly three
  times the invalid votes (three attempts per attribute). Counts: B 178 undecided attribute rows of 8,515 (366 invalid
  votes, 1,098 failed calls, 17 steps of 12 persons), D 79 of 2,546 (3.1%; 158 invalid votes, 474 failed calls, 10
  steps of 6 persons), H 96 of 10,811 (192 invalid votes, 576 failed calls, 1.9%, 8 steps of 6 persons). Earlier drafts
  said 366 attributes (4.3%) for B and 2.6% failed calls for H: both were wrong (366 counts votes). The undecided rows
  fall in steps 50 to 100 (B 8/39/131 at steps 50/75/100; D 43/36 at 75/100; H 22/74), always in whole steps, and leave
  the rates, so they can bias the DIR of the late steps of B, D and H. A targeted repeat of only these 353 attributes is
  in progress (2026-11-03); update this bullet with its result.
- **Number of persons (verified 2026-11-03).** A 288 (plan: 250), B 248, C 95, D 158 (148 with text, 148 judged), E 265,
  F 16 (plan: 18), G 49 (31 organisational units excluded), H 467. Draft files that show 120 persons for A and B are
  wrong.
- **Steps carried forward.** A step whose text is identical to the previous one is not re-run. Persons carried at steps
  50/75/100: A 268/38/275, B 29/19/43, C 0/0/37, D 117/73/129, E 2/0/12, F 0/0/3, G 3/1/14, H 284/0/426. For most
  persons in A, D and H steps 25 = 50 and 75 = 100, so there are two distinct exposures.
- **Temperature and sampling.** The temperature differs from the original runs where the local runtime required it.
  `repeat_penalty` is 1.1 for phi4-mini, to avoid empty outputs.
- **Run control.** `run_steps.py` stops on return code 2 (changed stop criterion). The first item of a long text pays a
  fixed cost, visible in the timings.
- **Control without documents (decision of 2026-11-02).** The attacker `gpt-oss:20b-local` ran at about 6.5 steps per
  minute on case A (1,152 steps), so all persons of the eight cases would have taken over 24 hours on one GPU. The
  control is therefore run on the sample of 50 persons per case (seed 42, the same persons as RQ3 and the ablations),
  paired with the judged inference with documents. The specification asks for all cases, not for all persons.

## 2. Human validation (D99)

- The plan asked for 400-500 attributes with human annotators. The rerun uses 100 items per case, annotated by three
  LLM annotators (cases B, C, E, F, G done; A, D, H pending). deepseek-r1:7b ran with thinking off and covers only
  part of the items. Three items of the verylong text 6 (case G) are invalid and excluded.
- Cases A and D (annotation of 2026-11-01/02): the outputs of an annotator come from more than one machine. For
  llama3.1:8b on A, 6 items were annotated on a CPU workstation and the other 94 on the laptop 4070; on D, 1 item
  came from a second CPU workstation and 99 from the laptop. phi4-mini ran only on the laptop, deepseek-r1:7b on the
  CPU of the 2-GPU machine. The partial files were merged by item id, never overwriting an item already present, and
  the source hashes are kept. Valid items: llama3.1:8b 97/100 (A) and 94/100 (D), phi4-mini 61/100 (A) and 66/100 (D).
  Invalid items are those whose answer was cut or unparsable and are excluded from the agreement, as in the other cases.
  Case H (sample built on 2026-11-02 with the same seed): llama3.1:8b 91/100 and phi4-mini 74/100 valid, both on the laptop.
  deepseek-r1:7b valid 44/100 on A and 49/100 on D (CPU of the 2-GPU machine) and 42/100 on H (laptop GPU, same recipe);
  its share of valid items is low in every case (B 17/85, C 43/100, E 29/90, F 26/85, G 23/70 on the laptop), so the
  agreement uses only the items that all three answered (A 28, D 39, H 29 of 100).
- A sheet for human annotators is prepared separately and does not replace the LLM annotations.

## 3. Ablations (case E)

- Conditions: documents 1/2/4/8, chronological versus random order, last k documents (2, 4), years window (1, 3),
  generalisation, three repetitions of the full condition (seed offsets 0, 1000, 2000), case C on 30 persons times 3.
- **Dropped rows (decision of 2026-11-01).** The model-size rows (nemotron-3-nano:30b and qwen3.5:9b as attackers) and
  the three repetitions of case C on 30 persons were removed from the queue because the judging time (about 7 hours
  per row on one GPU, 63 to 70 hours per queue) did not fit. The remaining 15 rows run in priority order. The paper
  must state that model size and the repetition on case C were not run. The rows are kept in
  `queue_rimosse_20261101.txt` on the 2-GPU machine and can be restored.
- **Model size (not run).** The plan was three local attackers instead of five (gpt-oss:20b, nemotron-3-nano:30b,
  qwen3.5:9b) on 50 persons, with thinking off for nemotron and qwen and nemotron also acting as judge.
- The defaults for mitigation (k years 1 and 3, last k documents 2 and 4, generalisation) are choices of the
  execution session and can be changed.

## 4. RQ3 and RQ4

- **Sample.** 50 persons per case, drawn with seed 42 (F has 16, G has 49). Primary generator `gpt-oss:20b`,
  secondary `llama3.1:8b` on the same persons.
- **Synthetic content.** Recipient names are synthetic and deterministic; organisation and domain are placeholders;
  the "none" variant keeps only role and organisation type.
- **Corrections.** A first version of case F had duplicate negatives; it was discarded and replaced by F v2
  (commits `15aa750`, `a759e15`). Three persons of F have no explicit attributes; the analytic choice is open.
- **Hardware.** The secondary generator runs on CPUs and small GPUs, so truncations depend on the runtime build.
- **Benchmarks.** Public benchmarks are in English and the generated messages in Italian; the transfer setting
  mixes languages and is reported as such.
- **LLM detector.** The detector returns a label only, so its AUC equals balanced accuracy. It is used zero-shot and
  few-shot (judge `gemma4:31b-local`).
- **Personalisation rubric.** Criteria "specific" (1-5), "plausible" (1-5) and reference counts are the execution
  session's choice, to be confirmed.
- **Alarm.** A detector whose F1 as generated reaches `config.ALARM_DETECTION_F1` is kept and flagged.

## 5. Distribution of the work

Runs were split over six machines (two RTX 5000 Ada GPUs, an RTX 5090, a laptop 4070, a laptop 3060 and two CPU-only
workstations). The cache key does not depend on the machine, so partial caches are merged by concatenation (later rows
win) and the stage is rerun on the merging machine, which must issue zero new calls. Raw data for cases E and C were
copied to the 2-GPU machine and checked against four manifests (file count and SHA-256 of the sorted listing).

## 6. Still to fill in

Final counts per case, the ablation table, the agreement figures, costs, and the list of items excluded for length.
