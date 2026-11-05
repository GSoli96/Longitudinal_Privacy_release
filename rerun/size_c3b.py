"""Size study: Case C with a 3 B attacker (llama3.2:3b-local), inference and local-panel judging in a stage folder of
its own (``data/private/caseC/stages_size-llama3.2-3b``).

    python -m rerun.size_c3b dry-run                  # whole chain with a fake client, nothing real is touched
    python -m rerun.size_c3b infer                    # 95 persons, 4 steps, attacker llama3.2:3b-local
    python -m rerun.size_c3b pass-gemma | pass-qwen | pass-third    # one judge over every attribute (optional split)
    python -m rerun.size_c3b judge                    # panel gemma4:31b-local, qwen3.5:9b, nemotron-3-nano:30b-local
    python -m rerun.size_c3b dedup                    # pair confirmation with gemma4:31b-local, bge-m3 embeddings
    python -m rerun.size_c3b metrics                  # metrics.json in <stage folder>/metrics_limited
    python -m rerun.size_c3b export                   # copy it to results/rerun/caseC/size_llama3.2-3b_metrics.json

Every mode except ``dry-run`` calls ``rerun.run_case.main`` with the tag, with the attacker and without cloud keys
(none is loaded: no call can leave the machine). The safeguards, all in ``check_tag``:

* the tag must start with ``size-`` and cannot be one of the folders that exist (``stages``, ``stages_gloc``, ...);
* ``infer`` creates the stage folder and writes ``size_run.json`` in it; the other modes refuse to run without that
  file, and ``infer`` refuses a folder that has files but no marker or another attacker, so a folder made by another
  run is never reused or overwritten;
* the model cache is ``data/private/llm/cache_caseC_size3b.jsonl`` (``config.TAG_CACHE_SUFFIX``), not the shared cache
  of Case C. ``--part NAME`` adds ``_NAME`` to the file names, so that two passes on one machine do not append to the
  same file; the parts are joined with ``cat`` before ``judge`` (see docs of the run).

``--limit N`` and ``--tag size-...-smoke`` give a small trial in a folder of its own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from . import config, local_panel, run_case

CASE = "C"
ATTACKER = config.SIZE_ATTACKER_3B
TAG = config.SIZE_TAG_3B
RESERVED_TAGS = ("evidence", "noev", "ref", "trunc2000", "gloc")
MARKER = "size_run.json"
MODES = ("dry-run", "infer", "pass-gemma", "pass-qwen", "pass-third", "judge", "dedup", "metrics", "export")
DEDUP_MODEL = local_panel.GEMMA
DEDUP_NUM_CTX = 8192
EXPORT_NAME = "size_llama3.2-3b_metrics.json"


class TagError(RuntimeError):
    """The tag or its folder cannot be used safely."""


def stage_path(tag: str, base=config.PRIVATE, case: str = CASE) -> Path:
    """Where ``run_case.stage_dir`` puts the stages of ``tag`` (the folder is not created)."""
    return Path(base) / f"case{case}" / f"stages_{tag}"


def existing_stage_dirs(base=config.PRIVATE, case: str = CASE) -> list[str]:
    d = Path(base) / f"case{case}"
    return sorted(p.name for p in d.glob("stages*") if p.is_dir()) if d.exists() else []


def check_tag(tag: str, *, create: bool = False, attacker: str = ATTACKER, base=config.PRIVATE) -> Path:
    """The stage folder of ``tag`` if it is safe to use, else ``TagError``. With ``create`` (mode ``infer``) a new or
    empty folder is made and marked; without it the folder must already carry the marker of this attacker."""
    if not re.fullmatch(r"size-[A-Za-z0-9._-]+", tag or ""):
        raise TagError(f"tag {tag!r}: it must start with 'size-' and use letters, digits, '.', '_' and '-'")
    if tag in RESERVED_TAGS:      # cannot happen after the prefix check; kept as a statement of what is protected
        raise TagError(f"tag {tag!r} is the name of an existing run")
    d = stage_path(tag, base)
    marker = d / MARKER
    has_files = d.exists() and any(d.iterdir())
    if has_files:
        info = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
        if info is None or info.get("attacker") != attacker or info.get("tag") != tag:
            raise TagError(f"{d.name} exists and was not made by this run (no marker, or another attacker): not used")
        return d
    if not create:
        raise TagError(f"{d.name} does not exist: run the 'infer' mode first")
    d.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"tag": tag, "attacker": attacker, "case": CASE}, indent=1), encoding="utf-8")
    return d


def cache_suffix_for(tag: str, part: str | None = None) -> str:
    base = config.TAG_CACHE_SUFFIX.get(tag, f"_case{CASE}_{tag}")
    return base + (f"_{part}" if part else "")


def parse_shard(spec: str) -> tuple[int, int]:
    """``"K/N"`` -> (K, N) with 0 <= K < N."""
    try:
        k, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise ValueError(f"--shard must look like K/N, got {spec!r}") from None
    if n < 1 or not 0 <= k < n:
        raise ValueError(f"--shard K/N needs 0 <= K < N, got {spec!r}")
    return k, n


def shard_persons(persons: list, k: int, n: int) -> list:
    """Every N-th person of the (already id-sorted) list, starting at K. The N shards partition the list."""
    return list(persons)[k::n]


def apply_shard(k: int, n: int) -> set:
    """Restrict ``run_case`` to shard K of N: the persons it selects and the rows of the inference file that the judge
    passes read (that file holds every person; a pass judges only the persons of its shard). Returns the set of person
    ids kept, filled when the persons are selected."""
    select, rows = run_case.select_persons, run_case.judge_rows_to_run
    keep: set = set()

    def select_shard(persons, limit=None):
        chosen = shard_persons(select(persons, limit), k, n)
        keep.update(p.id for p in chosen)
        return chosen

    run_case.select_persons = select_shard
    run_case.judge_rows_to_run = lambda path: [r for r in rows(path) if r["person"] in keep]
    return keep


def build_argv(mode: str, *, tag: str = TAG, workers: int = 1, limit: int | None = None,
               extra: list[str] | None = None) -> list[str]:
    """Arguments of ``run_case.main`` for a mode (see the module docstring)."""
    common = ["--case", CASE, "--tag", tag, "--attacker", ATTACKER, "--extract-workers", "4"]
    if limit:
        common += ["--limit", str(limit)]
    conc = f"{local_panel.GEMMA}=1,{local_panel.THIRD}=1,{local_panel.QWEN}=1,{ATTACKER}={max(1, workers)}"
    if mode == "infer":
        argv = common + ["--stage", "infer", "--workers", str(max(1, workers)), "--model-concurrency", conc]
    elif mode in ("pass-gemma", "pass-qwen", "pass-third"):
        argv = common + ["--stage", "judge-local", "--workers", "1", "--model-concurrency", conc]
    elif mode == "judge":
        argv = common + ["--stage", "judge", "--panel-scheme", "sequential", "--workers", "1",
                         "--model-concurrency", conc]
    elif mode == "dedup":
        argv = common + ["--stage", "dedup", "--dedup-model", DEDUP_MODEL, "--workers", "8"]
    elif mode == "metrics":
        argv = common + ["--stage", "metrics", "--workers", "1"]
    else:
        raise ValueError(f"mode {mode!r} does not call run_case")
    return argv + list(extra or [])


def export(tag: str = TAG, *, base=config.PRIVATE, results=config.RESULTS) -> Path:
    """Copy the aggregate metrics of the tagged run to ``results/rerun/caseC/``. The file has no ids and no text."""
    src = check_tag(tag, base=base) / "metrics_limited" / "metrics.json"
    if not src.exists():
        raise TagError(f"{src.name} not found: run the 'metrics' mode first")
    dst = Path(results) / f"case{CASE}" / (EXPORT_NAME if tag == TAG else f"{tag}_metrics.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


# ---- dry run ---------------------------------------------------------------------------------------------------------
class FakeTransport:
    """A server that answers like the three kinds of model, and records what it was asked (no network)."""

    def __init__(self):
        self.calls = Counter()
        self.urls = Counter()
        self.think_sent = Counter()
        self.num_ctx = {}

    def __call__(self, url, headers, body, timeout):
        model, prompt = body["model"], body["messages"][-1]["content"]
        props = (body.get("format") or {}).get("properties", {})
        self.calls[model] += 1
        self.urls[(model, url.split("/api")[0])] += 1
        if "think" in body:
            self.think_sent[model] += 1
        self.num_ctx.setdefault(model, set()).add(body["options"].get("num_ctx"))
        if "attributes" in props:                                    # the attacker
            h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
            items = [{"attribute": f"Attributo di prova {h} numero {n}", "category": "role"} for n in (1, 2)]
            content = {"attributes": items}
        elif "same" in json.dumps(props):                            # de-duplication
            content = {"verdicts": [{"id": int(n), "same": "yes" if a == b else "no"}
                                    for n, a, b in re.findall(r"^(\d+)\.\nA: (.*)\nB: (.*)$", prompt, re.M)]}
        else:                                                        # judges
            block = prompt.split("ATTRIBUTES TO JUDGE")[1]
            content = {"verdicts": [{"id": int(n), "plausible": "yes", "coherent": "yes", "realistic": "yes",
                                     "explicit": "no"} for n, _ in re.findall(r"^(\d+)\. (.*)$", block, re.M)]}
        return 200, {"message": {"content": json.dumps(content)}, "prompt_eval_count": 10, "eval_count": 5,
                     "done_reason": "stop"}


class _FakeEmbed:
    def __call__(self, texts):
        import numpy as np
        out = [[b - 127.5 for b in hashlib.sha256(t.encode("utf-8")).digest()[:16]] for t in texts]
        v = np.array(out, dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _listing(base: Path) -> dict:
    """Name, size and modification time of every file under ``caseC`` and of the files of ``llm`` that belong to a size
    run (no content). Other files of ``llm`` are left out: a job running on the same machine may be appending to them."""
    out = {}
    for root, deep in ((Path(base) / f"case{CASE}", True), (Path(base) / "llm", False)):
        if not root.exists():
            continue
        for p in (root.rglob("*") if deep else root.glob("*")):
            if p.is_file() and (deep or "size" in p.name):
                st = p.stat()
                out[str(p.relative_to(base))] = (st.st_size, st.st_mtime_ns)
    return out


def dry_run(*, base=config.PRIVATE, raw=config.RAW, limit: int | None = None) -> dict:
    """The whole chain (infer, judge with the local panel, dedup, metrics) on the real persons of Case C with a fake
    server, in a temporary copy of the text cache. Reports what the real folders look like before and after: it must be
    the same, and the tagged folder must not appear there."""
    from . import corpus

    base = Path(base)
    if not (base / f"case{CASE}" / "text").is_dir() or not (Path(raw) / f"case{CASE}").is_dir():
        return {"status": "skipped", "reason": "text cache or raw documents of Case C not on this machine"}
    before = _listing(base)
    dirs_before = existing_stage_dirs(base)
    with tempfile.TemporaryDirectory() as t:
        priv = Path(t) / "private"
        (priv / f"case{CASE}").mkdir(parents=True)
        shutil.copytree(base / f"case{CASE}" / "text", priv / f"case{CASE}" / "text")
        for name in ("case.json", "layout.json"):
            if (base / f"case{CASE}" / name).exists():
                shutil.copyfile(base / f"case{CASE}" / name, priv / f"case{CASE}" / name)
        if (base / "id_salt.txt").exists():
            shutil.copyfile(base / "id_salt.txt", priv / "id_salt.txt")
        persons = run_case.select_persons(corpus.load_case(CASE, raw_dir=raw, private_dir=priv, workers=1), limit)
        ids_match = None
        if (base / f"case{CASE}" / "persons_map.json").exists() and not limit:
            real = json.loads((base / f"case{CASE}" / "persons_map.json").read_text(encoding="utf-8"))
            ids_match = sorted(real.values()) == sorted(p.id for p in persons)
        d = run_case.stage_dir(CASE, base=priv, tag=TAG)
        transport = FakeTransport()
        from .llm import LLMClient
        saved_attacker = config.ATTACKER
        config.ATTACKER = ATTACKER
        try:
            panel = local_panel.setup()
            client = LLMClient([], transport=transport, cache_path=priv / "llm" / "cache.jsonl",
                               usage_path=priv / "llm" / "usage.jsonl", private_root=priv, sleep=lambda s: None,
                               model_concurrency={m: 1 for m in (ATTACKER, *panel)})
            usable = run_case.budget_usable_attacker()
            summary = {"infer": run_case.run_infer(persons, client, ATTACKER, d / "infer.jsonl", usable_tokens=usable,
                                                   workers=1)}
            summary["judge"] = run_case.run_judge(persons, d / "infer.jsonl", client, panel, d / "judge.jsonl",
                                                  attacker=ATTACKER, usable_tokens=usable, workers=1, sequential=True,
                                                  full_panel=True)
            summary["dedup"] = run_case.run_dedup(d / "infer.jsonl", d / "judge.jsonl", client, _FakeEmbed(),
                                                  d / "dedup.jsonl", workers=1, model=DEDUP_MODEL)
            metrics = run_case.run_metrics(persons, d / "infer.jsonl", d / "judge.jsonl", d / "dedup.jsonl",
                                           d / "metrics_limited", _FakeEmbed(), panel)
        finally:
            local_panel.teardown()
            config.ATTACKER = saved_attacker
        files = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
        rows = run_case._read_jsonl(d / "infer.jsonl")
    after = _listing(base)
    return {"status": "ok", "persons": len(persons), "ids_match_persons_map": ids_match,
            "steps": len(rows), "steps_run": summary["infer"]["steps_run"], "windows_total": summary["infer"]["windows_total"],
            "attributes": summary["infer"]["attributes"], "judge_calls": sum(
                n for m, n in transport.calls.items() if m in panel), "undecided": summary["judge"]["undecided_attributes"],
            "panel_scheme": summary["judge"].get("panel"), "persons_in_metrics": metrics["persons_in_cohort"],
            "stage_files": files, "tag_dir_name": d.name,
            "models_called": dict(transport.calls),
            "all_calls_to_local_server": all(u == config.LOCAL_URL for (_, u) in transport.urls),
            "think_field_sent_to_attacker": transport.think_sent.get(ATTACKER, 0),
            "attacker_num_ctx": sorted(transport.num_ctx.get(ATTACKER, [])),
            "existing_stage_dirs_before": dirs_before, "existing_stage_dirs_after": existing_stage_dirs(base),
            "tag_dir_created_in_real_data": stage_path(TAG, base).exists(),
            "real_files_unchanged": before == after}


# ---- command line ----------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Case C with the 3 B attacker: stages in a folder of their own.")
    ap.add_argument("mode", choices=MODES)
    ap.add_argument("--tag", default=TAG, help=f"stage tag, must start with 'size-' (default {TAG})")
    ap.add_argument("--workers", type=int, default=1, help="inference calls side by side (1 fits a 24 GB card)")
    ap.add_argument("--limit", type=int, default=None, help="first N persons only (use another --tag)")
    ap.add_argument("--part", default=None, help="adds _PART to the cache file names (two passes on one machine)")
    ap.add_argument("--ignore-alarms", action="store_true", help="passed to run_case")
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="pass-* modes only: judge the persons number K, K+N, K+2N, ... of the id-sorted list (K from 0). "
                         "Run each K on its own machine with its own --part, concatenate the caches, then run 'judge'")
    a = ap.parse_args(argv)
    if a.mode == "dry-run":
        res = dry_run(limit=a.limit)
        print(json.dumps(res, indent=1))
        ok = res["status"] == "skipped" or (res["real_files_unchanged"] and not res["tag_dir_created_in_real_data"]
                                            and res["all_calls_to_local_server"])
        return 0 if ok else 1
    try:
        if a.mode == "export":
            print(export(a.tag))
            return 0
        check_tag(a.tag, create=(a.mode == "infer"))
    except TagError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    config.TAG_CACHE_SUFFIX[a.tag] = cache_suffix_for(a.tag, a.part)
    if a.shard:
        if not a.mode.startswith("pass-"):
            print("error: --shard is only for the pass-* modes", file=sys.stderr)
            return 2
        try:
            k, n = parse_shard(a.shard)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        apply_shard(k, n)
    # no cloud key is read: every model of the run is local
    from . import llm
    llm.load_keys = lambda *x, **k: []
    llm.load_key_types = lambda *x, **k: None
    if a.mode in ("pass-gemma", "pass-qwen", "pass-third", "judge"):
        local_panel.setup()
        config.LOCAL_JUDGE = {"pass-gemma": local_panel.GEMMA, "pass-third": local_panel.THIRD}.get(a.mode,
                                                                                                  config.LOCAL_JUDGE)
    elif a.mode == "dedup":
        # the prompts are a few hundred tokens: a small window keeps the 31 B model on the GPU of a 24 GB card, where the
        # default window of 131072 tokens does not fit and every call takes minutes; temperature 0, same answers
        config.NUM_CTX[DEDUP_MODEL] = DEDUP_NUM_CTX
    elif a.mode == "metrics":
        config.PANEL = (local_panel.GEMMA, local_panel.QWEN, local_panel.THIRD)
    extra = ["--ignore-alarms"] if a.ignore_alarms else []
    run_case.main(build_argv(a.mode, tag=a.tag, workers=a.workers, limit=a.limit, extra=extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
