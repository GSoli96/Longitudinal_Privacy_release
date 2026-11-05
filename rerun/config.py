"""Models, paths and thresholds of the rerun (spec section 5, plan global constraints)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS_PATH = ROOT / ".secrets" / "ollama_keys.json"
RAW = ROOT / "data" / "raw"
PRIVATE = ROOT / "data" / "private"
RESULTS = ROOT / "results" / "rerun"

# Models. Judges changed on 2026-10-20 from deepseek-v4.1-flash and mistral-large-3 (HTTP 402: not in
# the free plan of the 32 keys) to three families different from the attacker's.
ATTACKER = "gpt-oss:20b"
JUDGES = ("gemma4:31b", "nemotron-3-ultra", "minimax-m3.1-flash")   # panel of the runs up to 2026-10-23 (C, F, G)
JUDGE_FALLBACK = "nemotron-3-super"
# Attributes per judge call when it is not the default of ``judge.BATCH_SIZE`` (40). Pilot, Case C, 300 attributes:
# nemotron-3-ultra marks more attributes as explicit in a batch than alone (AC1 .69, McNemar p<.0001), also with
# batches of 10 and 5; with the final variant (no evidence field) gemma4 shows it too (AC1 .73) and minimax is at .78.
# Decision of 2026-10-20: all three judges one attribute per call, the other attributes of the step as context.
JUDGE_BATCH_SIZE = {"gemma4:31b": 1, "nemotron-3-ultra": 1, "minimax-m3.1-flash": 1, "qwen3.5:9b": 1}
# Calls in flight per model in the core runs. Calibration on Case C (150 one-attribute calls per level, ultra):
# 16 -> 25.6 calls/min, 24 -> 44.6, 32 -> 34.2, no 429 at any level; 24 gives the most calls per minute.
MODEL_CONCURRENCY = {"nemotron-3-ultra": 24, "gemma4:31b": 16, "minimax-m3.1-flash": 24, "gpt-oss:20b": 16,
                     "qwen3.5:9b": 1,   # local judge: one call at a time on this PC
                     "gpt-oss:20b-local": 2}   # local attacker (Cases A, D, H): the server's OLLAMA_NUM_PARALLEL decides
# Main PC (2026-10-23): the limits above are ceilings of an adaptive limit (rerun.llm.AdaptiveLimit) that starts
# here, halves on a burst of 429 and grows by one while calls are clean (Case E on the other PC: ~54,000 429 events
# at the ceilings).
SHARED_CACHE_CASES = ("C", "E", "F", "G")   # use data/private/llm/cache.jsonl; other cases: cache_case<X>.jsonl
MODEL_CONCURRENCY_START ={"nemotron-3-ultra": 8, "gemma4:31b": 6, "minimax-m3.1-flash": 8, "gpt-oss:20b": 8,
                           "qwen3.5:9b": 1, "gpt-oss:20b-local": 2}
# Since about 04:00 on 2026-10-23 minimax is no longer in the free plan (HTTP 402). Author's decision: one Pro account;
# the paid models use only the keys of type PRO in the keys file, the free models only the others (spares the credit).
PAID_MODELS = ("minimax-m3.1-flash",)
PAID_KEY_CONCURRENCY = 3   # calls in flight on one paid key: the Pro plan allows 3 concurrent requests (ollama.com/pricing)
# Cloud-only rule (advisor, 2026-10-23): every model that produces a result runs on Ollama Cloud; size ablation with
# two cloud families (gpt-oss 20B/120B; nemotron 30B/120B/550B; nemotron-3-ultra is also a judge).
SIZE_ABLATION = ("gpt-oss:20b", "gpt-oss:120b", "nemotron-3-nano:30b", "nemotron-3-super", "nemotron-3-ultra")
EMBEDDER = "bge-m3"
LOCAL_MODELS = ("bge-m3",)   # auxiliary embeddings for the dedup candidates (author's decision); nothing else local
# Local judge (author's decision, 2026-10-23): with minimax paid and no budget, the third judge runs on this PC.
# A Qwen model: a family different from the attacker and from the other two judges. qwen3.6:35b (MoE) took 66 s per
# call on this PC (91% on CPU at a 262,144 context; pilot of 2026-10-23, 28 attributes); qwen3.5:9b fits in the 8 GB
# GPU (author's choice). The attributes of a step are judged one after the other, so that the server reuses the
# step text already read (same prompt prefix).
LOCAL_JUDGE = "qwen3.5:9b"
LOCAL_MODELS = (*LOCAL_MODELS, LOCAL_JUDGE)
# Reasoning per judge when it is not the model's default (None: the field is not sent). The local judge answers
# without reasoning: with it, one call takes minutes on this PC.
JUDGE_THINK = {LOCAL_JUDGE: False}
# Annotators (decision 2026-10-29, D99): three small local models of different families annotate a stratified sample of
# attributes with the four criteria of the codebook; Fleiss' kappa and Cohen's kappa measure their agreement. They run on
# the CPU of the spare machines, so they answer without reasoning and with a short output limit.
ANNOTATORS = ("llama3.1:8b", "phi4-mini", "deepseek-r1:7b")
ANNOTATION_ITEMS_PER_CASE = 100
ANNOTATION_MAX_PROMPT_TOKENS = 28000   # items whose text needs more (or several windows) are not sampled; counted
ANNOTATION_NUM_CTX = 32768
ANNOTATION_NUM_PREDICT = 512
LOCAL_MODELS = (*LOCAL_MODELS, *ANNOTATORS)
# small detectors of RQ4 (family C, extra model size): they run on a local server only, never on the cloud route
SMALL_DETECTORS = ("qwen2.5:7b",)
LOCAL_MODELS = (*LOCAL_MODELS, *SMALL_DETECTORS)
JUDGE_THINK = {**JUDGE_THINK, "deepseek-r1:7b": False}
JUDGE_BATCH_SIZE = {**JUDGE_BATCH_SIZE, **{m: 1 for m in ANNOTATORS}}
# Smaller context for the local judge when the prompt fits (2026-10-23): at 262,144 tokens two thirds of the model run
# on the CPU (21 s per call); 95% of the judge prompts are below 32,768 tokens. Accepted only if the votes equal those
# of the full context on the pilot attributes (results/rerun/caseC/local_judge_ctx_check.json).
# Check of 2026-10-23: identical on 95 of 103 attributes (realistic 96, explicit 101), 11.4 s vs 21.3 s per call;
# not identical, so the full context is kept (rule agreed with the author). Re-enable with {LOCAL_JUDGE: 32768}.
JUDGE_CTX_SMALL = {}
# Judging scheme from 2026-10-23 (author's decision, no budget; D41): three judges, majority of two. The first pair
# votes on every attribute; the tie-breaker (the judge that costs most quota) only where the pair does not give the
# same valid vote on all four criteria, plus an audit share of attributes (fixed by a hash of the attribute id) on
# which all three always vote, for the agreement statistics. Cases whose panel is in the cache (C, F, G) are run
# with the full panel.
PANEL = ("gemma4:31b", LOCAL_JUDGE, "nemotron-3-ultra")
FIRST_PAIR = PANEL[:2]
TIE_BREAKER = PANEL[2]
AUDIT_SHARE = 0.10
FULL_PANEL_CASES = ("C", "F", "G")
# Persons judged per case when only a random sample (seed 42) is judged; the inference stays complete either way.
# Empty since D48 (2026-10-25, applied 2026-10-29): A and B are judged on all persons. The sample of 120 (D41) is a
# fallback if the judging cannot finish: PERSON_SAMPLE = {"A": 120, "B": 120}.
PERSON_SAMPLE: dict = {}
PERSON_SAMPLE_SEED = 42

# Determinism
SEED = 42
ATTACKER_SEED_OFFSET = 0              # repetitions of the inference (--seed-offset); the judges keep their seeds
TEMPERATURE = 0
DEFAULT_OPTIONS = {"temperature": TEMPERATURE, "seed": SEED}

# Endpoints
LOCAL_URL = os.environ.get("LP_LOCAL_URL", "http://localhost:11434")  # the variable is for a second server on another port
CLOUD_URL = "https://ollama.com"

# Thresholds
NOVELTY_THRESHOLDS = (0.80, 0.85, 0.90)
NOVELTY_MAIN = 0.85
WINDOW_OVERLAP = 0.10
BOOTSTRAP_RESAMPLES = 2000
ALARM_INVALID_RATE = 0.02
ALARM_KEY_ERROR_RATE = 0.05
ALARM_FAILED_CALL_RATE = 0.05    # judge calls that failed (e.g. prompt too long for the local context), D91
ALARM_UNDECIDED_RATE = 0.10      # attributes left without a majority because every call failed, D91
ALARM_MIN_CALLS = 100            # D58: below this many observations a rate is not an alarm (cache re-reads)
ALARM_DETECTION_F1 = 0.99


def is_local(model: str) -> bool:
    return model in LOCAL_MODELS or model.endswith("-local")


# Characters per token on Italian document text, measured with each model's own tokenizer
# (prompt token counts reported by the server, slope between two prompt lengths, 12 samples,
# 2026-10-20): the minimum observed, rounded down, so that budgets are conservative.
# median / minimum: gpt-oss:20b 3.61 / 3.12; gemma4:31b 3.40 / 2.75; llama3.2:3b 3.25 / 2.94.
TOKEN_CHARS = {"gpt-oss:20b": 3.1, "gpt-oss:20b-local": 3.1, "gemma4:31b": 2.7, "llama3.2:3b": 2.9}
TOKEN_CHARS_DEFAULT = 2.7   # unmeasured model: the most conservative measured value

# Maximum context read from the server (/api/show)
MODEL_CTX = {"gpt-oss:20b": 131072, "gpt-oss:20b-local": 131072, "gpt-oss:120b": 131072,"gemma4:31b": 262144, "llama3.2:3b": 131072,
             "nemotron-3-super": 262144, "nemotron-3-ultra": 262144, "nemotron-3-nano:30b": 262144,
             "minimax-m3.1-flash": 1000000, "qwen3.5:9b": 262144}

# Reserves (tokens). Judges: from the batch length (rerun.budget.judge_batch_reserves).
ATTACKER_OUTPUT_RESERVE = 16384
ATTACKER_PROMPT_OVERHEAD = 1500


# Fixed context window per model (decision 2026-10-20): the same value for every step and window of a
# person and of the whole run, so that settings never differ between steps (reproducibility); in the
# cloud it does not change the cost. Judges: the largest value verified with the needle test.
NUM_CTX = {"gpt-oss:20b": 131072, "gpt-oss:20b-local": 131072, "llama3.2:3b": 131072,"gemma4:31b": 262144,
           "nemotron-3-ultra": 262144, "nemotron-3-super": 262144, "minimax-m3.1-flash": 262144,
           "qwen3.5:9b": 262144, **{m: 32768 for m in ANNOTATORS}}


def num_ctx_for(model: str) -> int:
    return NUM_CTX.get(model) or min(MODEL_CTX.get(model, 131072), 131072)


# Evidence anchoring: the cited evidence is looked up in the step text after normalisation; if not exact,
# a fuzzy match counts when the similarity of the best aligned window of the same length reaches this ratio.
EVIDENCE_FUZZY_RATIO = 0.90

JUDGE_NUM_PREDICT = 32768  # judges: reasoning + JSON of a batch of 40. minimax-m3.1-flash reasons for 9,000-16,400+ tokens
# (cut at 16,384 in 6 of 10 runs, not a repetition cycle); the client cap is 32,768 (decision 2026-10-20)

# Two-phase semantic de-duplication: candidates by cosine, confirmed by an LLM (main variant of novelty and PES)
DEDUP_MODEL = "gemma4:31b"
DEDUP_CANDIDATE_COSINE = 0.80


# Attacker sampling (decision 2026-10-20, main session): gpt-oss:20b at temperature 0 falls into repetition
# cycles in its reasoning (medium: 2 of 4 steps cut at 16,384 tokens, high: 4 of 4; 8-gram repetition share
# 0.8-0.96, tail of the reasoning already seen before) and returns no answer. With temperature 1.0 and top_p 1.0
# (the recommended setting) and reasoning effort "medium" no step was cut (0 of 4, 17-34 s). Fixed seed 42, so a
# run is reproducible where the server honours the seed; the saved cache is the source of truth. Fallback: "low".
ATTACKER_THINK = "medium"
ATTACKER_OPTIONS = {"temperature": 1.0, "top_p": 1.0}

# An empty list of attributes on a long text is almost always spurious (observed: 1 of 8 identical runs of a
# 12,000-character step, and the low reasoning level on a 35,000-character text): the step is asked again once with a
# new seed (no correction message); if the answer is empty again it is accepted as empty. Counted per step.
ATTACKER_EMPTY_RETRIES = 1

# Post-hoc classification of the attributes (categories, special categories): one model, temperature 0.
CATEGORY_MODEL = "gemma4:31b"


# ---- Size study, Case C with a 3 B attacker (2026-11-04) -------------------------------------------------------------
# Replaces the claim "a 3 B model on basic hardware". The attacker is the local build of llama3.2:3b, served under the
# alias ``llama3.2:3b-local`` (``ollama cp llama3.2:3b llama3.2:3b-local``; the suffix routes it to the local server and
# gives it its own cache key, like gpt-oss:20b-local; the plain name ``llama3.2:3b`` stays "not local" on purpose).
# Same prompt, windows (131,072 tokens as the 20 B attacker), sampling and seed as the main runs; what changes is that
# the model does not reason, so no ``think`` field is sent (Ollama rejects it for models without thinking).
# characters per token: the measurement of llama3.2:3b (2026-10-20, minimum observed, see TOKEN_CHARS above).
SIZE_ATTACKER_3B = "llama3.2:3b-local"
SIZE_TAG_3B = "size-llama3.2-3b"
TOKEN_CHARS[SIZE_ATTACKER_3B] = TOKEN_CHARS["llama3.2:3b"]
MODEL_CTX[SIZE_ATTACKER_3B] = MODEL_CTX["llama3.2:3b"]
NUM_CTX[SIZE_ATTACKER_3B] = NUM_CTX["llama3.2:3b"]
# one call at a time: with 131,072 tokens one slot of the server holds about 17 GB (RERUN_LOG, 2026-10-20), so a second
# slot does not fit a 24 GB card
MODEL_CONCURRENCY[SIZE_ATTACKER_3B] = 1
MODEL_CONCURRENCY_START[SIZE_ATTACKER_3B] = 1
# Reasoning field per attacker when it differs from ``ATTACKER_THINK``; None = the field is not sent.
ATTACKER_THINK_BY_MODEL = {SIZE_ATTACKER_3B: None}
# Cache file of a tagged run, instead of the one of the case (suffix of cache/usage/events): the size study does not
# write into the shared cache of Case C, and its files can be copied and merged on their own.
TAG_CACHE_SUFFIX = {SIZE_TAG_3B: "_caseC_size3b"}


def attacker_think_for(model: str):
    """Reasoning effort sent to the attacker ``model``: ``ATTACKER_THINK`` unless the model has its own entry."""
    return ATTACKER_THINK_BY_MODEL[model] if model in ATTACKER_THINK_BY_MODEL else ATTACKER_THINK
