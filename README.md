# Longitudinal Privacy Inference from Public-Sector Data

Code and aggregated results accompanying the paper *Longitudinal Privacy
Inference from Public-Sector Data* (paper link: **to be added**).

The pipeline measures how far a large language model can reconstruct
sensitive personal attributes from the administrative records that public
administrations publish under mandatory-transparency rules, how much that
exposure grows as more documents are combined, and whether the inferred
profiles are enough to drive targeted phishing that automatic detectors can
still recognise.

All administrations are anonymised. They appear only as **Case A** to
**Case H** with a generic organisation type (regional government, ministry,
central agency, healthcare agency). The repository contains no entity names,
person names, portal URLs or API keys, and no raw per-person documents.

## Pipeline

The pipeline follows Section 3 of the paper and is organised one module per
stage, under `src/`:

| Stage | Module | Paper section | What it does |
|------|--------|---------------|--------------|
| 1. Public knowledge extraction | `src/crawl.py` | 3.2 | Crawls a transparency portal (target is a configuration placeholder) and downloads the public documents per individual. |
| 2. Text extraction | `src/extract.py` | 3.2 | Extracts text from PDFs with pypdf, falls back to Tesseract OCR, classifies each document and records the years it mentions. |
| 3. Longitudinal inference | `src/inference.py` | 3.3 | Exposes the per-person corpus to an LLM at 25/50/75/100 % and asks for attributes that are *not* explicitly stated. |
| 4. Attribute validation | `src/validation.py`, `src/majority_vote.py` | 3.4 | Three independent LLM judges score each attribute (plausible, coherent, realistic, explicit); labels are merged by majority vote. |
| 5. Phishing simulation | `src/phishing.py` | 3.5 | Generates one BEC and one Spear Phishing message per profile from the validated attributes, for detection only. |
| 6. Detection | `src/detection.py` | 4.6 | Classical TF-IDF classifiers, fine-tuned Transformers, LLM zero/few-shot classifiers and voting ensembles, on the generated corpus and on six public benchmarks. |

Metrics live in `src/metrics.py` (see **Metrics** below).

## Directory layout

```
README.md              this file
requirements.txt       Python dependencies
pytest.ini             test configuration
CITATION.cff           citation metadata
RERUN_LOG.md           chronological log of the rerun (decisions, incidents, numbers)
src/                   first-version pipeline, one module per stage, plus metrics
rerun/                 the rerun pipeline of the paper (inference, judging, de-duplication,
                       metrics, ablations, RQ3/RQ4 detection), one module per task;
                       run_case.py is the per-case runner, dashboard.py the progress view
configs/               one YAML per case (A-H) with placeholders, and example.env
scripts/
  compute_stats.py            rebuild results/ from the private validation outputs
  make_figures.py             rebuild figures/ from results/ only
  anonymisation_check.py      scan files for direct identifiers (see docs/ANONYMISATION.md)
  local_judge_panel.py        run the panel of local judges
  local_judge_pilot.py        pilot of the local judges
  precompute_embeddings.py    precompute the embeddings used by the semantic de-duplication
  key_status.py               report the state of the (cloud) API keys, without printing them
  check_local_context.py      check the context window of the local models
  extract_case_h.py, export_git_dir.py   helpers
tests/rerun/           unit and integration tests (pytest)
results/               aggregated numbers only (no per-person rows)
  README.md            provenance of every results file
  CONSISTENCY.md       every paper value vs the value found in the source
  rerun/case<A-H>/     metrics.json and metrics_by_category.json of the final rerun
  rerun/ablation/      ablation results; rerun/*.json: benchmarks and side studies
figures/               plots as PDF + PNG (captions in figures/README.md)
docs/                  ANONYMISATION.md (release policy), METHODS_NOTE.md, RERUN_PLAN.md,
                       codebook.md, specs/ (design of the rerun), plans/ (work plans),
                       anonymisation_allow.txt (narrow allow list of the scanner)
data/                  where private inputs go; see data/README.md (empty in the release)
```

Language note: the repository is written in English. The LLM prompts, the
regular expressions applied to the Italian e-mails and the test fixtures are
in Italian, because they are part of the method and of the data language.

## Installation

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Case G only:
python -m spacy download it_core_news_lg
```

External tools: **Poppler** (for `pdf2image`) and **Tesseract OCR** (for
`pytesseract`). The Transformer detectors need `torch`/`transformers`; they
are optional and skipped if not installed.

## Configuration

Model names, exposure steps, judges and generators are set per case in
`configs/caseX.yaml`. Crawl targets and organisation strings are
placeholders; set them only for data you are authorised to process.

Endpoints and keys are read from the environment, never hard-coded. Copy
`configs/example.env` to a local `.env` (git-ignored) and set:

```
OLLAMA_HOST=http://localhost:11434      # or an Ollama-Cloud-compatible host
OLLAMA_API_KEY=                          # bearer token for a cloud endpoint
```

The original experiment scripts contained hard-coded API keys; those have
been removed. Never commit a key.

## Running

The stages are meant to run in order, driven by a case config:

```python
from src.config import CaseConfig
from src.llm_client import make_client
cfg = CaseConfig.from_yaml("configs/caseC.yaml")
client = make_client()               # reads OLLAMA_HOST / OLLAMA_API_KEY
# stage 1-2: crawl + extract  -> per-person text
# stage 3:  src.inference.run_longitudinal(client, cfg, record, cache_dir)
# stage 4:  src.validation.audit_attribute(...) over the 3 judges,
#           then src.majority_vote.build_majority(...)
# stage 5:  src.phishing.generate_message(...)
# stage 6:  src.detection.get_classical_models(), .metrics(...)
```

### Running one case of the rerun

```bash
python -m rerun.run_case --case C --stage all --limit 3
```

Stages are `infer`, `judge`, `dedup` and `metrics` (or `all`). Every stage
writes JSONL files under `data/private/case<X>/stages/` (personal text, never
committed); the metrics stage writes aggregates only under `results/rerun/`.
All model calls go through a cache that acts as the checkpoint, so an
interrupted run resumes without repeating completed calls. See
`python -m rerun.run_case --help` and `docs/specs/2026-10-20-rerun-design.md`.

### Tests and anonymisation check

```bash
python -m pytest -q
python scripts/anonymisation_check.py <paths> --allow-file docs/anonymisation_allow.txt
```

The scan must end with `RESULT: PASS` (see `docs/ANONYMISATION.md`, Section 7).

### Regenerating results and figures

`results/` is rebuilt from the private validation outputs; `figures/` is
rebuilt from `results/` alone:

```bash
python scripts/compute_stats.py --audit-root data/private/audit --out results
python scripts/make_figures.py
```

`compute_stats.py` expects, per case `<X>`, the three per-judge audit files
under `data/private/audit/<X>/judges/*.json` and, optionally, detection
CSVs under `data/private/audit/<X>/detection/` (see the script's docstring
for the exact file names). It writes only aggregate numbers.

## Metrics (as implemented)

For each attribute the majority vote yields four binary labels: plausible
(P), coherent (C), realistic (R), explicit (E). Over a set of attributes:

- **IQS** (Inference Quality Score) `= (P + C + R + 2E) / 5`, averaged over
  attributes. Explicitness is weighted twice because an explicitly supported
  attribute is maximally reliable.
- **DIR** (Dangerous Inference Rate) `= share with P and C and not E`
  (realism not required).
- **Hallucination rate** `= share rejected on all four criteria`.
- **Novelty** at step *t* `= share of attributes not seen at steps 1..t-1`.
  Attributes are compared as **exact strings** (no semantic de-duplication).
- **PES** (Privacy Exposure Score): per person, the count of **fully
  validated dangerous** attributes accumulated over the four steps, with the
  cohort mean reported. The paper's definition is `P and C and R and not E`
  and is the default in `src/metrics.py` (`PES_CONDITION_PAPER_TEXT`).

  Note on PES: the first version of the paper reported PES computed by the
  original code with `P and C and R and E` (i.e. counting explicit,
  fully-validated attributes); those values were later corrected to the
  definition above. Both conditions are kept selectable so the correction is
  auditable, and both are recorded in `results/`. See
  `results/CONSISTENCY.md`.

### Implementation details that affect the numbers

These are honest limitations of the experimental code, kept as-is so the
released numbers are reproducible:

- The inference prompt keeps at most the **first 12,000 characters** of the
  per-person text (`raw_text[:12000]`). For long profiles the later exposure
  steps can therefore receive the same (truncated) input.
- Each judge sees the **first 2,000 characters** of the profile text
  (`html_text[:2000]`); Case C passes a larger window (up to 120,000).
- Inference steps are **independent** model calls, except in **Case F**,
  whose prompts carry the history of earlier inferences (cumulative).
- The majority rule is a simple majority of the judges that responded
  (`votes > responders/2`); Cases A and C use "at least 2 of 3", and Case F
  uses "at least 2, or unanimity when fewer responded". These differ only
  marginally.
- Coverage is not always complete: for example, in Case B one judge
  (`gemma4`) returned verdicts for 168 of the 248 individuals.

## Cases

Eight Italian public administrations of four kinds. Each is described only
by what matters for the analysis. The number of persons is the one in
`results/rerun/case<X>/metrics.json` (final rerun).

| Case | Type | Persons |
|------|------|--------:|
| A | regional government | 288 |
| B | regional government (dated CV snapshots) | 248 |
| C | ministry | 95 |
| D | regional healthcare agency | 158 |
| E | central agency | 265 |
| F | central agency (local models only) | 16 |
| G | ministry | 49 |
| H | provincial healthcare agency | 467 |

Models, judges and exposure steps of each case are in `configs/` and, for
the rerun, in `rerun/config.py`; the paper describes them.

## Ethics and anonymisation

- The data analysed are personal data that Italian public administrations
  are legally required to publish. No authentication, paywall or access
  control was bypassed.
- The administrations are **not named** anywhere in this repository. Raw
  documents and per-person data are **not** redistributed.
- Generated phishing messages were produced **only** as input to the
  detection experiments and were **never sent**.
- Before adding any data to `data/`, read `docs/ANONYMISATION.md`.
- The legitimate-e-mail corpus used by the detectors is a set of Italian
  e-mails of **undocumented provenance** (to be confirmed by the authors);
  it is not described as human-written or real.

## Citation

See `CITATION.cff`. A formal reference and DOI will be added on publication.

## License

MIT, see `LICENSE`. It covers the code, the aggregated results and the figures.

## History

Commit hashes and the branch `rerun` cited in `RERUN_LOG.md` and in `docs/` refer to the development history, which is not published: the public repository starts from a single commit.
