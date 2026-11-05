# Rerun plan (2026-10-20)

Protocol for re-running the experiments of Cases A-H after the code review of
2026-10-19. Goal: same pipeline as the original runs, with only the defects
below fixed, so that old and new numbers can be compared case by case.

## 1. What changes, and nothing else

| # | Original behaviour | Rerun |
|---|---|---|
| R1 | Inference prompt keeps only the first 12,000 characters of the cumulative text | Full cumulative text of the step. Set `num_ctx` explicitly. If the text exceeds the model context, split it into overlapping windows, infer per window, take the union of the attributes, and log the profile as `chunked`. |
| R2 | Judges see the first 2,000 characters of the profile page | Judges see the same text the inference model saw at that step (same chunking rule). |
| R3 | Novelty compares exact strings | Report both: exact-string novelty (as before) and semantic novelty: an attribute is new if its cosine similarity to every earlier attribute of the same person is below 0.85, with multilingual embeddings (`bge-m3`, local, via Ollama). Also compute it at 0.80 and 0.90 (sensitivity). |
| R4 | PES counted P=C=R=yes and E=yes | PES counts P=C=R=yes and E=no (paper definition). |
| R5 | Case E: on Windows, 34 groups of file names differing only in letter case overwrite each other on disk | Read Case E documents from git objects (`git cat-file`/`git archive`), not from the working tree; give colliding names a short hash suffix so every variant is kept. Check in the original code whether both variants were used, and do the same. |

Everything else stays as in the original code of each case: models, prompts,
temperature, exposure steps (25/50/75/100 %), judge set, majority vote,
attribute filters for phishing, generators, detectors, benchmarks. Where the
original code of a case differs from the others, keep the difference and write
it down.

## 2. Engineering requirements

- Keys: read from `.secrets/ollama_keys.json`; never print, log or commit a
  key or its name. Rotate keys round-robin; on 429/5xx/connection errors back
  off and move to the next key. Log usage per key *index* only.
- Checkpointing: every LLM call is appended to a JSONL file keyed by
  (case, person, step, stage, model, window); a restart skips completed keys.
- Raw outputs contain personal data: they go under `data/private/`
  (git-ignored) and are never committed. Only aggregated metrics go to
  `results/`, with `_rerun` in the file name; the original results stay.
- Record for every case: model versions/digests, `num_ctx`, number of chunked
  profiles, wall time, number of calls, failed calls.

## 3. Order of work

1. Implement R1-R5 on a branch `rerun`, reusing the original scripts of each
   case as the reference.
2. Smoke test: Case C, 3 profiles, all stages through majority vote. Check the
   checkpoint/resume by interrupting and restarting once.
3. Pilot: full Case C (inference, validation, metrics). Report timings, the
   number of chunked profiles, and the new vs old values of DIR, IQS,
   hallucination, PES, exact and semantic novelty.
4. **Stop and report.** The main session reviews the pilot before the other
   cases start.
5. Full runs: A, B, C, E, F, H (Case F keeps its history-carrying design),
   then phishing generation and detection for each case.
6. Cases D and G need a new crawl (raw documents are not in the repository):
   not started without an explicit decision of the user.
