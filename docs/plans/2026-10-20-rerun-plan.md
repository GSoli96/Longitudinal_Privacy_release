# Rerun Implementation Plan

> For agentic workers: execute task by task on branch `rerun`; steps use checkboxes.
> After every task: tests green, invariants checked, `RERUN_LOG.md` updated with
> `REVIEW <task>`, push, message the main session, wait for `GO <task>` before
> starting any task that depends on it (independent tasks may proceed).

**Goal:** re-run all experiments with one corrected, harmonised pipeline and
produce paper-ready results.

**Architecture:** a new package `rerun/` (next to `src/`) with one module per
stage, all LLM traffic through one client with cache, checkpoint and key
rotation; stage outputs as JSONL under `data/private/<case>/`; aggregates under
`results/rerun/`.

**Tech stack:** Python 3.11 (`py -3.11`), requests, pydantic, numpy, pandas,
scipy, statsmodels, scikit-learn, torch (cu132), transformers, pytest; Ollama
local (gemma3:12b, llama3.2:3b, bge-m3) and Ollama Cloud.

**Spec:** `docs/specs/2026-10-20-rerun-design.md` (read sections 1-11 and 10b).

## Global constraints

- Configuration (spec §5): attacker `gpt-oss:20b`; judges `gemma4:31b`,
  `deepseek-v4.1-flash`, `mistral-large-3`; size ablation `llama3.2:3b`,
  `gpt-oss:20b`, `gpt-oss:120b`; embeddings `bge-m3`.
- Temperature 0, seed 42 everywhere; model tag, digest and date logged per call.
- Keys only from `.secrets/ollama_keys.json`; never print, log or commit keys
  or key names; usage logged by key index.
- Personal data only in `data/private/` and `data/raw/` (git-ignored). No
  person, entity, student or portal names in tracked files.
- Case labels A-H only; mapping case -> source folder read from the paper
  repository `TO_CONTINUE.md`, never written here.
- Alarms (spec §8) stop the stage: invalid/empty outputs > 2 %, any detection
  F1 >= 0.99, persons lost between stages, key errors > 5 %.
- No generated message is ever sent anywhere.

## Review focus

1. Persons with zero usable documents or empty OCR: kept in counts as
   `no_text`, never silently dropped (test in Task 2).
2. Model refusals, prose around JSON, code fences, partial JSON: retried, then
   counted as failures, never parsed into attributes or votes (Tasks 4, 5).
3. Accented and case-colliding file names (Case E) and Windows paths: every
   variant read from git objects, none overwritten (Task 2).
4. Texts longer than the model context: windows with overlap, attribute merge,
   explicitness against all windows (Tasks 3, 5).
5. Interruption, 429 and network errors mid-run: resume without duplicate or
   missing calls; cache hits identical to fresh calls (Task 1).

## File structure

```
rerun/
  config.py        models, paths, thresholds (values above)
  llm.py           client: key pool, rate limit, retry, cache, checkpoint, usage log
  corpus.py        per-case loaders -> Person(id, case, role, org_type, docs[])
  steps.py         exposure steps, identical-step detection, windows
  infer.py         inference with JSON schema
  judge.py         batched and per-attribute judging, vote validation, majority
  metrics.py       DIR, hallucination, 16 combos, IQS variants, novelty, PES
  stats.py         cluster bootstrap, trend test, agreement coefficients
  categories.py    post-hoc category classification (taxonomy spec §10b.4)
  sample.py        human-validation sample and annotation sheet
  run_case.py      CLI: py -3.11 -m rerun.run_case --case C --stage infer|judge|metrics
tests/rerun/       unit tests with fixtures (no personal data)
RERUN_LOG.md       progress log (spec §8)
```

### Task 0: Setup and data staging
- [ ] Create branch `rerun`, `RERUN_LOG.md` (sections: Done, Next, Open issues, Decisions), `tests/rerun/`.
- [ ] Extract `data/raw/caseG/Archivio.zip` into `data/raw/caseG/` excluding `__MACOSX/`; check MURIN = 80 html, MURDOWN = 550 pdf.
- [ ] Case E: export documents from git objects into `data/raw/caseE/` with a short hash suffix on case-colliding names; check file count equals the blob count in HEAD.
- [ ] Commit, `REVIEW T0`.

### Task 1: LLM client (`rerun/llm.py`)
Interfaces: `chat(model: str, messages: list[dict], options: dict, schema: dict | None) -> LLMResult` with fields `text, json, ok, error, tokens_in, tokens_out, seconds, key_index, cached`.
- [ ] Tests: cache key = sha256(model, messages, options, schema) and a second call is `cached=True` with identical text; 429 triggers back-off and next key; after N failures `ok=False, error` set; checkpoint JSONL resumes without re-calling completed keys; no key string appears in logs (grep test).
- [ ] Implement; local vs cloud routing by model name; `num_ctx` passed in options; concurrency limit per key.
- [ ] Commit, `REVIEW T1`.

### Task 2: Corpus (`rerun/corpus.py`)
Interfaces: `load_case(case: str) -> list[Person]`; `Person.docs: list[Doc(id, date | None, kind, text)]`.
- [ ] Tests on fixtures: navigation and markup removed; OCR language `ita`; persons with no text kept as `no_text`; organisational units excluded from Case G with a count; accented and hash-suffixed names loaded.
- [ ] OCR fallback (decided 2026-10-20): a page fails Tesseract if it yields < 100 characters or mean word confidence < 60. Failed pages are transcribed by the local vision model `gemma3:12b` (verbatim transcription instruction, no correction or completion, temperature 0); the document never leaves the PC. Record per page the method (`text_layer`, `tesseract`, `vlm`). Quality check: on 50 pages readable by both, character error rate of the VLM against Tesseract; 10 VLM pages checked by eye on the new PC by the executing session (page image against transcription; documents never leave the PC), with the outcome per page in the log. Report the number of VLM pages per case (for the paper).
- [ ] Implement one loader per case, reusing the original extraction logic of that case where it is correct (read via `git show` from the original repository).
- [ ] Report per case: persons, docs, chars (min/median/max), excluded units, `no_text`.
- [ ] Commit, `REVIEW T2`.

### Task 3: Steps and windows (`rerun/steps.py`)
Interfaces: `build_steps(p: Person) -> list[Step(k, text, identical_to_prev)]`; `windows(text, ctx_tokens) -> list[str]`.
- [ ] Tests: date order when dates exist, else document order; 25/50/75/100 % by documents; identical step flagged; windows overlap 10 % and cover the whole text.
- [ ] Implement; commit, `REVIEW T3`.

### Task 4: Inference (`rerun/infer.py`)
Interfaces: `infer(step: Step, model: str) -> list[Attribute(text, category_hint, evidence, window)]`.
- [ ] Tests with a mocked client: valid JSON parsed; fenced/prosy JSON retried then failure; duplicate attributes across windows merged; empty list allowed.
- [ ] Prompt: the original inference instruction of the paper (non-explicit attributes), output schema `{attributes: [{attribute, category, evidence}]}`; no request for special-category data (spec §10b.4).
- [ ] Commit, `REVIEW T4`.

### Task 5: Judges (`rerun/judge.py`)
Interfaces: `judge_batch(step, attrs, model) -> dict[attr_id, Vote]`, `judge_one(step, attr, model) -> Vote`, `majority(votes: list[Vote]) -> Label | "undecided"`.
- [ ] Tests: majority only with 3 valid votes (truth table over all 8 yes/no patterns per criterion); invalid vote -> retry -> `invalid`, never `no`; explicitness OR over windows (spec §10b.5).
- [ ] Implement; judges run in parallel; commit, `REVIEW T5`.

### Task 6: Metrics and statistics (`rerun/metrics.py`, `rerun/stats.py`)
- [ ] Tests on toy data with hand-computed results: DIR; hallucination and the 16-combination table; IQS and IQS restricted to non-explicit; novelty exact and semantic (thresholds 0.80/0.85/0.90, bge-m3); PES by semantic clusters per spec §10b.6, cumulative and per 1,000 characters; cluster bootstrap CI by person (2,000 resamples); trend test; Cohen, Fleiss, Krippendorff, Gwet AC1.
- [ ] Implement; commit, `REVIEW T6`.

### Task 7: Smoke test (Case C, 3 persons)
- [ ] Run infer -> judge -> metrics; interrupt once mid-judging and resume; compare call counts with expected; `REVIEW T7` with the log summary.

### Task 8: Pilot (Case C, all persons)
- [ ] Full run; on 300 attributes run both batched and per-attribute judging; Cohen kappa per criterion; batching kept only if all >= 0.80 (spec §10b.1).
- [ ] Measure judge speed; replace a judge that is too slow by another family and log it.
- [ ] `REVIEW T8` with timings, failures, chunked/identical counts, metrics with CIs.

### Task 9: Core RQ1-RQ2, all available cases
- [ ] Run cases in order C, F, G, B, A, E, H (then D when its data arrive); `REVIEW T9-<case>` after each.

### Task 10: Categories and human-validation sample
- [ ] `categories.py`: taxonomy of spec §10b.4, one judge model, JSON output; tests.
- [ ] `sample.py`: stratified 50-60 attributes per case by step and majority outcome; annotation sheet with the columns of `docs/codebook.md` §8; calibration subset of 40.
- [ ] `REVIEW T10`.

### Task 11: Control without documents (all cases)
- [ ] Inference with only role and organisation type; same judges; DIR difference with CIs. `REVIEW T11`.

### Task 12: Case B experiments
- [ ] Size ablation (3 models), document count (1, 2, 4, 8), chronological vs shuffled (3 permutations), 3 complete runs on 50 persons of E and 30 of C (run-to-run variance), mitigation (3 policies x 2 values of k) with information-kept measure. `REVIEW T12`.

### Task 13: RQ3/RQ4
- [ ] Before coding, the executor writes a short sub-plan (`docs/plans/rq34.md`) adapting the original generation and detection code to the common configuration and to spec §6 and §10b.2-3; the main session reviews it (`GO T13-plan`). Messages stay inside the benchmark and are never sent.

### Task 14: Re-identification check and cost table
- [ ] Uniqueness of attribute sets per case; tokens, time and hardware per profile. `REVIEW T14`.

### Task 15: Deliverables
- [ ] `results/rerun/` CSV/JSON, `SUMMARY_RQ1..4.md`, `latex/` tables and pgfplots figures in the paper's style, `docs/METHODS_NOTE.md`; anonymisation scan (`scripts/anonymisation_check.py`) clean. `REVIEW T15`.
