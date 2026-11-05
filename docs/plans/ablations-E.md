# Sub-plan for the ablations on Case E (DRAFT, to be approved by the authors)

Status: draft of 30 October 2026. Nothing has been run, and the code has nothing yet for ablations, countermeasure and permutations (only the no-documents control remains: `rerun/control.py`, stage `--stage control`, with the panel fixed in commit `9061660`). It derives from `docs/specs/2026-10-20-rerun-design.md` (lines 94-99, 112-113, 155-161, 203-206), from the plans of 20 and 23 October and from `RERUN_LOG.md`. Where the specification is silent I write "to be decided" and list it among the questions. The numbers are orders of magnitude derived from the files, not measurements.

## 1. What the specification asks for

All experiments are on Case E (265 persons, 4,040 documents, dates on 99.3% of the documents, median span of 2.9 years per person; main inference: 1,046 steps run, 10,333 attributes).

1. Number of documents: 1, 2, 4, 8.
2. Chronological versus random order, with 3 permutations.
3. Countermeasure: three minimisation policies applied before inference (generalisation of fields, window of the last k years, k documents per person), each with two values of k, measuring the reduction in DIR and PES relative to the share of information retained.
4. Three complete repetitions on 50 persons of E and 30 of C, to measure the variability between runs, because cloud responses are not reproducible even with a fixed seed.
5. Size ablation: in the cloud plan of 23 October there are five models (`config.SIZE_ABLATION`: gpt-oss:20b and 120b, nemotron-3-nano:30b, super, ultra), which replace the three of the specification.

## 2. How to add them to the code

The steps are cumulative prefixes built by `steps.build_steps(person)` with the first `ceil(k·n/100)` documents. Judges and metrics rebuild the steps by themselves (`step_texts`, `check_infer_invariants`, `run_metrics`), so every variant must live in the `Person` object, otherwise the judges would read a text different from the attacker's. Proposal: a module `rerun/ablation.py` with pure functions that take a list of `Person` and return a transformed one (`dataclasses.replace` on the documents): random order with `random.Random(f"{seed}:{person.id}:{perm}")`, last k years, first k documents. For the number of documents, an indication read by `build_steps` is needed (for example in `Person.meta`), so that the 25/50/75/100 labels remain as positions and `metrics.STEPS` does not change; it must be verified that `pes_by_step` and `cohort_pes` accept missing steps.

Output goes under `data/private/caseE/stages_<tag>` with tags such as `abl-docs`, `abl-order-p1`, `mit-<policy>-k<k>`, `rep1`, `size-<model>`; with a tag the metrics go to `metrics_limited` and do not overwrite `results/rerun/caseE`. An argument `--sample N` is needed that applies `sample_persons` already from the inference stage (today the sample applies only from the judges onwards) and saves the list. For the repetitions a `--seed-offset` for the attacker is needed, because with the same text and seed the cache would return the same response with zero variance; `judge_batch` already has `seed_offset`, but `run_judge` does not expose it. Steps that coincide with those of the main run cost nothing thanks to the cache. For size, the path is ready with `--attacker`, but `NUM_CTX`, `TOKEN_CHARS` and `MODEL_CONCURRENCY` have no entries for the new models, and `ATTACKER_THINK="medium"` with temperature 1.0 is tuned for gpt-oss:20b.

## 3. Cost

For the 50-person experiments alone (size, order, mitigation at 100%, documents, repetitions of E and C) the files indicate roughly 28,000-37,000 attributes to be judged, to which dedup and retries must be added. The bottleneck is the judge: the local panel of B took 10.4 hours for 8,902 judgements of gemma4:31b-local. At that pace 30,000 attributes require days of GPU time, not hours. The plan should therefore be scaled down or split across several machines, and it is best to decide this after measuring the real speed on the Alienware.

## 4. Open questions for the authors

1. Number of documents: the first n in chronological order, the last n, or a sample? Is only the "all n documents" step measured, or also 25/50/75/100% of those n? Are persons with fewer than n documents (minimum 2 in E) excluded, kept, or counted separately?
2. Do the 50 persons of E (and the 30 of C) apply only to the repetitions, or also to size, documents, order and mitigation? Or all 265?
3. How are they chosen: random sample with seed 42, first by id, stratified by number of documents or span? Are the 30 of C those of the pilot?
4. Random order: three different orderings per person, with a seed per person? What to do with the 21 undated documents and with persons who have 2 documents (only 2 permutations)?
5. Are the three policies those listed in §7.3? What are the two values of k for each? How is field generalisation realised on text extracted from PDFs without structured fields (rules, substitutions, a model that rewrites, and which one)?
6. Do the "3 complete runs" include the main run? In the repetitions, do only the attacker's seeds change, or also the judges'? Are dedup and metrics redone?
7. How is the information retained measured (characters, documents, explicit attributes, extracted facts), and at which step is the reduction evaluated?
8. Judges: local panel as in the main run of E, or cloud panel? Sequential or full scheme? With different panels the comparison is not homogeneous.
9. Size: do the five models replace the three of the specification? Same sampling and reasoning parameters for all? If nemotron-3-ultra is also a judge, should it be removed from the panel for that condition?
10. Metrics and tests: DIR and PES at 100%, or the curves per step? Paired difference per person with bootstrap, as in `compare_with_documents`?
11. Is the no-documents control part of this batch? It is already implemented.

Note: Task 12 of the plan of 20 October is still titled "Case B experiments" but is about E and C; specification §10b.1 has already moved everything to E. It is a leftover of the text, not a contradiction of substance.
