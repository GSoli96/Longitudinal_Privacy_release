# data/

This directory is intentionally almost empty. The pipeline's inputs are
personal data from public-sector transparency portals and are **not**
redistributed here.

## What will go here

- `data/private/` (git-ignored): the raw and intermediate inputs used
  locally to run the pipeline and to rebuild `results/`:
  - `data/private/<CASE>/downloads/` — crawled documents (never committed);
  - `data/private/<CASE>/docs/` — per-person text knowledge base;
  - `data/private/audit/<CASE>/judges/*.json` — the three judges' audit files
    that `scripts/compute_stats.py` consolidates;
  - `data/private/audit/<CASE>/detection/*.csv` — the detection outputs;
  - `data/private/legit_it.json` — the legitimate-e-mail corpus for the
    detectors (Italian e-mails of **undocumented provenance**, to be
    confirmed by the authors; not described as human-written or real).

- `data/` (committed) will receive an **anonymised** benchmark only if and
  when `docs/ANONYMISATION.md` explicitly allows it and states what may be
  released and how. That file is being prepared separately.

## Rules

- Do not commit raw documents, per-person rows, person names, portal URLs,
  administration names or API keys. `.gitignore` already blocks
  `data/raw/`, `data/private/`, `.env` and common artefact folders.
- Read `docs/ANONYMISATION.md` before adding anything to the committed part
  of `data/`. If it is not present yet, add nothing.
