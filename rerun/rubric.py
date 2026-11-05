"""Personalisation rubric of the RQ3 messages (design spec §7.2: LLM panel plus human check on a sample).

The plan fixes the panel and the sample, not the criteria; these are the ones chosen here (a definition to declare
and to confirm with the authors). A judge sees one generated message and the full list of attributes known about
the recipient (explicit and inferred together, so that the judge does not see which variant produced the message),
and answers three integers in a JSON object:

* ``specific`` 1-5: how much the message is addressed to this person rather than to anyone in the role (the main
  score of the rubric);
* ``plausible`` 1-5: how credible the message is as mail to this recipient;
* ``refs`` 0..N: how many of the listed attributes the message uses or clearly alludes to (N = length of the list).

An answer that is not valid JSON, is out of range or misses a field is a failed rating, counted and never
replaced by a default. The score of a message is the mean of ``specific`` over the judges with a valid answer; a
message with no valid judge is left out and counted. The panel is the one of the rerun, so the judges are not the
generator. ``summarise_by_variant`` gives the mean score per variant with a person-cluster bootstrap CI.
``human_sample`` draws, with a seed, a sample stratified by variant and attack for the human check."""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config, stats

SCHEMA = {"type": "object",
          "properties": {"specific": {"type": "integer"}, "plausible": {"type": "integer"},
                         "refs": {"type": "integer"}},
          "required": ["specific", "plausible", "refs"]}
PROMPT = """\
Valuta quanto un messaggio e-mail è personalizzato per il suo destinatario.

Attributi noti sul destinatario:
{profile}

Messaggio:
Oggetto: {subject}

{body}

Rispondi solo con un oggetto JSON con tre numeri interi:
- "specific" da 1 a 5: quanto il messaggio è rivolto proprio a questa persona e non a chiunque abbia il suo ruolo
  (1 = generico, 5 = costruito sui suoi dettagli);
- "plausible" da 1 a 5: quanto il messaggio è credibile come posta ricevuta da questa persona;
- "refs" da 0 a {n}: quanti degli attributi elencati il messaggio usa o richiama chiaramente."""
_JSON = re.compile(r"\{[\s\S]*\}")


def build_prompt(rec: dict, profile: list[str]) -> str:
    lines = "\n".join(f"- {p}" for p in profile) if profile else "(nessuno)"
    return PROMPT.format(profile=lines, subject=rec.get("subject", ""), body=rec.get("body", ""), n=len(profile))


def parse(res, n_profile: int) -> tuple[dict | None, str]:
    """``({"specific", "plausible", "refs"}, "")`` or ``(None, reason)``."""
    if not res.ok:
        return None, "call_failed:" + (res.error or "unknown")
    obj = res.json
    if not isinstance(obj, dict):
        m = _JSON.search(res.text or "")
        try:
            obj = json.loads(m.group()) if m else None
        except Exception:  # noqa: BLE001
            obj = None
    if not isinstance(obj, dict):
        return None, "not_json"
    out = {}
    for key, lo, hi in (("specific", 1, 5), ("plausible", 1, 5), ("refs", 0, n_profile)):
        v = obj.get(key)
        if isinstance(v, bool) or not isinstance(v, int):
            return None, f"bad_{key}"
        if not lo <= v <= hi:
            return None, f"out_of_range_{key}"
        out[key] = v
    return out, ""


def rate_one(client, model: str, rec: dict, profile: list[str], *, think=False) -> dict:
    res = client.chat(model, [{"role": "user", "content": build_prompt(rec, profile)}],
                      options={"temperature": 0.0, "seed": config.SEED}, schema=SCHEMA, think=think)
    scores, reason = parse(res, len(profile))
    return {"ok": scores is not None, **(scores or {}), "error": reason}


def rate(client, panel, records: list[dict], profiles: dict, *, workers: int = 2) -> dict:
    """Rate the positive messages. ``profiles``: ``{person: [attribute text, ...]}`` (explicit and inferred).
    Returns ``{"rows": [...], "summary": {...}}``; a row has the record identifiers, one entry per judge in
    ``judges`` and ``score`` (mean ``specific`` of the valid judges, ``None`` when none is valid)."""
    positives = [r for r in records if r.get("kind", "positive") == "positive" and r.get("label", 1) == 1]
    jobs = [(r, m) for r in positives for m in panel]

    def work(job):
        r, m = job
        return rate_one(client, m, r, profiles.get(r["person"], []))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(work, jobs))
    by_id: dict = defaultdict(dict)
    errors: Counter = Counter()
    for (r, m), out in zip(jobs, results):
        by_id[r["id"]][m] = out
        if not out["ok"]:
            errors[f"{m}:{out['error']}"] += 1
    rows = []
    for r in positives:
        judges = by_id[r["id"]]
        valid = [j["specific"] for j in judges.values() if j["ok"]]
        rows.append({"id": r["id"], "person": r["person"], "case": r.get("case"), "variant": r.get("variant"),
                     "attack": r.get("attack"), "judges": judges,
                     "score": (sum(valid) / len(valid)) if valid else None, "n_valid": len(valid)})
    summary = {"n_messages": len(rows), "n_scored": sum(r["score"] is not None for r in rows),
               "n_unscored": sum(r["score"] is None for r in rows), "panel": list(panel), "errors": dict(errors)}
    return {"rows": rows, "summary": summary}


def summarise_by_variant(rows: list[dict], *, n_resamples: int = 2000, seed: int = config.SEED) -> dict:
    """Mean score per variant, person-cluster bootstrap CI (a person contributes the mean of their messages)."""
    out = {}
    for variant in sorted({r["variant"] for r in rows if r["variant"] is not None}):
        per_person: dict = defaultdict(list)
        for r in rows:
            if r["variant"] == variant and r["score"] is not None:
                per_person[r["person"]].append(r["score"])
        units = [float(np.mean(v)) for v in per_person.values()]
        out[variant] = stats.cluster_bootstrap_ci(units, lambda u: float(np.mean(u)) if len(u) else float("nan"),
                                                  n_resamples=n_resamples, seed=seed)
    return out


def human_sample(rows: list[dict], n_per_cell: int = 10, seed: int = config.SEED) -> list[str]:
    """Ids of a sample stratified by variant and attack, ``n_per_cell`` per cell (all of them when fewer), drawn
    with a seed from the scored rows in the order of their ids."""
    cells: dict = defaultdict(list)
    for r in sorted((r for r in rows if r["score"] is not None), key=lambda r: str(r["id"])):
        cells[(str(r["variant"]), str(r["attack"]))].append(r["id"])
    rng = random.Random(seed)
    chosen = []
    for key in sorted(cells):
        ids = cells[key]
        chosen += rng.sample(ids, min(n_per_cell, len(ids)))
    return sorted(chosen)


def main(argv=None) -> int:
    """``python -m rerun.rubric --corpus NAME --input input_X.jsonl [--input ...] --panel gemma4:31b-local qwen3.5:9b``:
    rates the positive messages of a corpus and writes ``<out-dir>/<name>.rubric.json`` (scores and counts, no text)."""
    import argparse
    from pathlib import Path

    from . import phish_corpus
    ap = argparse.ArgumentParser(description="Personalisation rubric of the generated positives.")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--input", nargs="+", required=True, help="input files of the persons (explicit and inferred lists)")
    ap.add_argument("--panel", nargs="+", default=["gemma4:31b-local", config.LOCAL_JUDGE])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-per-cell", type=int, default=10, help="size of the human-check sample per variant and attack")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    profiles = {}
    for f in a.input:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    profiles[row["person"]] = [*row.get("explicit", []), *row.get("inferred", [])]
    out_dir = Path(a.out_dir) if a.out_dir else phish_corpus.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    from .llm import LLMClient
    client = LLMClient([], cache_path=out_dir / f"{a.corpus}.rubric.cache.jsonl", max_attempts=3, private_root=out_dir,
                       grow_truncated=False)
    res = rate(client, tuple(a.panel), phish_corpus.load(a.corpus, a.out_dir), profiles, workers=a.workers)
    res["by_variant"] = summarise_by_variant(res["rows"])
    res["human_sample_ids"] = human_sample(res["rows"], a.n_per_cell)
    (out_dir / f"{a.corpus}.rubric.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str),
                                                     encoding="utf-8")
    print(json.dumps({**res["summary"], "by_variant": {k: round(v["estimate"], 3) for k, v in res["by_variant"].items()}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
