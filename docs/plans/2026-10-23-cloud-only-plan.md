# Rerun plan, phase 2: cloud models only, on the main PC (2026-10-23)

Supersedes the execution part of `2026-10-20-rerun-plan.md` (method unchanged:
`docs/specs/2026-10-20-rerun-design.md`, including section 10b and the pilot
decisions). Decided with the author on 2026-10-23 after the advisor's request.

## What changes
- Every model that produces a result runs on Ollama Cloud. Allowed locally: OCR
  (Tesseract, not a language model; texts already extracted), `bge-m3` embeddings
  for the dedup candidates (author's decision), training of the RQ4 Transformer
  detectors on this PC (author's decision).
- Size ablation in the cloud: `gpt-oss:20b` vs `gpt-oss:120b`, and
  `nemotron-3-nano:30b` / `nemotron-3-super` / `nemotron-3-ultra` (ultra is also a
  judge: declare it).
- Execution on the main PC; the new PC is retired. Its data was moved to
  `Longitudinal_Privacy_data/_locale/Directory_AltroPC/` (git-ignored, archive only).
- Two standing rules: a progress dashboard on screen for every experiment; before
  every task, check the running load and size the parallelism.

## What is reused (verified 2026-10-23)
- Code on branch `rerun` (7fe551d, 410 tests), extracted texts and OCR of C, E,
  F, G, the Case D archive, the id salt, 39,113 cached cloud answers (attacker and
  three judges), 8,200 cached `bge-m3` embeddings.
- Complete: C (reference pilot), F, G. Case E: inference complete, judging about
  half done (in cache), metrics never computed. Zips: no integrity error; one
  Case G PDF missing in the extracted folder, present in the zip.

## Phases (each ends with tests, invariants, log entry, commit)
0. Setup on the main PC: copy `data/raw`, `data/private` from the archive into the
   working repo (robocopy, counts and sizes checked), recover the missing G PDF,
   check `.secrets/`; check out `rerun`; Python environment; Ollama update and
   `bge-m3`; full test suite.
1. Dashboard: the runner writes `data/private/progress/<run>.json` (stage,
   done/total, rate, ETA, errors, 429); `rerun/dashboard.py` shows the bars in a
   separate console window that refreshes itself; text bars in chat at each check.
2. Adaptive concurrency: per model, halve the calls in flight when 429 exceed 5 %
   in a one-minute window, add one when clean; start low (ultra 8, gemma 6,
   minimax 8, attacker 8). Re-run C, F, G from the cache as a check (no new calls
   expected, identical metrics).
3. Case E: resume judging from the cache, dedup, metrics, REVIEW.
4. Case D: loader (pooled acts by name), full run.
5. Additions: categories and roles (T10-T11 code ready), control without
   documents on all cases, Case E experiments on a subset (size ablation with the
   five cloud models, documents 1/2/4/8, chronological vs shuffled, mitigation,
   3 repetitions), judge-noise repeats.
6. RQ3/RQ4 sub-plan (generation in the cloud; classical detectors on CPU; LLM
   detectors in the cloud; Transformers trained here after installing PyTorch with
   CUDA, with the author's ok).
7. Cases A, B, H when the student projects are in `data/raw/caseX/project/`.
8. Human-validation sample over all cases, deliverables (CSV, summaries, LaTeX).

## Parallelism (evaluated 2026-10-23)
- Machine: CPU idle, 39 GB RAM free, GPU free; another Claude session and its
  Python processes are running, so local work (embeddings, OCR, tests) stays at
  half the cores.
- The limit is the cloud quota, not the PC: adaptive limiter above, one runner
  process at a time per model group (the 32 keys are shared).
- Subagents: read-only checks in parallel (at most 5 per wave), never two writers
  on the same files.
