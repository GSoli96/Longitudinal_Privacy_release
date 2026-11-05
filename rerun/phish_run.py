"""Input and launch of the RQ3/RQ4 generation (plan ``docs/plans/rq34.md`` §3 and §8 steps 1-3 put together).

``prepare``: from the infer and judge files of a case, pick a sample of persons (seeded, sorted), select their
accepted attributes with ``phish_select`` and write one JSONL file of persons under ``data/private/phish/``.
``generate``: read such a file and write, for one generator on a local Ollama server, the records, the corpus of
``phish_corpus`` and a summary of counts. It needs no keys and can run on any machine that has the repository code,
Python 3.9 or later and the model. The cache of ``LLMClient`` is the record of every answer: a run that is stopped
and started again repeats no call that was answered.

The data hold no names, so the recipient's display name is synthetic: a first name and a surname chosen from short
lists by a hash of the person identifier. It is stable for a person and it is not a real person's name on purpose.
Role and type of the organisation, if a ``--meta`` file gives them, go into the ``base`` context of every variant.
Organisation name and domain are placeholders until the authors give the real ones (open question of the plan).
Everything written here holds personal text and stays under ``data/private/``; the printed summary holds counts."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from . import config, phish_corpus, phish_gen, phish_select

OUT_DIR = config.PRIVATE / "phish"
DEFAULT_COMPANY_NAME = "Ente"
DEFAULT_COMPANY_DOMAIN = "ente"
FIRST_NAMES = ("Marco", "Giulia", "Luca", "Francesca", "Andrea", "Chiara", "Matteo", "Sara", "Davide", "Elena",
               "Stefano", "Martina", "Paolo", "Valentina", "Alessandro", "Federica", "Simone", "Silvia", "Roberto",
               "Laura", "Fabio", "Monica", "Daniele", "Paola")
LAST_NAMES = ("Rossi", "Ferrari", "Russo", "Bianchi", "Romano", "Colombo", "Ricci", "Marino", "Greco", "Bruno",
              "Gallo", "Conti", "De Luca", "Costa", "Giordano", "Mancini", "Rizzo", "Lombardi", "Moretti", "Barbieri",
              "Fontana", "Santoro", "Mariani", "Rinaldi")


def synthetic_name(person_id: str) -> str:
    h = hashlib.sha256(f"name:{person_id}".encode("utf-8")).digest()
    return f"{FIRST_NAMES[h[0] % len(FIRST_NAMES)]} {LAST_NAMES[h[1] % len(LAST_NAMES)]}"


def select_sample(ids, n: int | None, seed: int = config.SEED) -> list[str]:
    """All the identifiers, sorted, or ``n`` of them drawn with a seeded generator and sorted."""
    ids = sorted(set(ids))
    if n is None or n >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, n))


def base_lines(meta: dict | None) -> list[str]:
    out = []
    if meta and meta.get("role"):
        out.append(f"Ruolo: {meta['role']}")
    if meta and meta.get("org_type"):
        out.append(f"Tipo di ente: {meta['org_type']}")
    return out


def persons_from_records(records, case: str, *, n: int | None = None, seed: int = config.SEED,
                         meta: dict | None = None, cap: int = phish_select.MAX_ATTRIBUTES) -> dict:
    """``{"persons": [...], "summary": {...}}``: the sampled persons with their accepted attributes."""
    chosen = select_sample([r.person for r in records], n, seed)
    sel = phish_select.select_all(records, chosen, cap=cap, seed=seed)
    persons = []
    used: set[str] = set()
    for p in chosen:
        s = sel["by_person"][p]
        name, k = synthetic_name(p), 0
        while name in used:  # two persons with one name would get the same prompt in the "none" variant
            k += 1
            name = synthetic_name(f"{p}#{k}")
        used.add(name)
        persons.append({"person": p, "case": case, "display_name": name,
                        "explicit": [r.text for r in s["explicit"]], "inferred": [r.text for r in s["inferred"]],
                        "base": base_lines((meta or {}).get(p))})
    return {"persons": persons, "summary": {**sel["summary"], "n_sampled": len(chosen), "sample_seed": seed}}


def load_stage_records(case: str, stages: str | Path | None = None) -> list:
    """The records of a case from its infer and judge files (majority labels, carry-forward as in the metrics)."""
    from . import run_case
    d = Path(stages) if stages else run_case.stage_dir(case)
    return run_case.build_records(_read_jsonl(d / "infer.jsonl"), _read_jsonl(d / "judge.jsonl"))


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_meta(path: str | Path | None) -> dict:
    """Optional JSONL ``{person, role, org_type}`` -> ``{person: {role, org_type}}``."""
    if not path:
        return {}
    return {r["person"]: r for r in _read_jsonl(Path(path))}


def prepare(case: str, name: str, *, n: int | None = None, seed: int = config.SEED, stages=None, meta_path=None,
            cap: int = phish_select.MAX_ATTRIBUTES, out_dir=None, records=None) -> dict:
    out = Path(out_dir) if out_dir else OUT_DIR
    recs = records if records is not None else load_stage_records(case, stages)
    res = persons_from_records(recs, case, n=n, seed=seed, meta=read_meta(meta_path), cap=cap)
    _write_jsonl(out / f"{name}.jsonl", res["persons"])
    return res["summary"]


def generate(persons: list[dict], model: str, name: str, *, client=None, out_dir=None,
             company_name: str = DEFAULT_COMPANY_NAME, company_domain: str = DEFAULT_COMPANY_DOMAIN,
             temperature: float = phish_gen.DEFAULT_TEMPERATURE, seed: int = config.SEED,
             negatives_per_positive: int = phish_gen.DEFAULT_NEGATIVES_PER_POSITIVE, workers: int = 2) -> dict:
    """Generate, assemble and write ``<name>.records.jsonl``, ``<name>.jsonl`` (corpus), ``<name>.summary.json``."""
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    if client is None:
        from .llm import LLMClient
        client = LLMClient([], cache_path=out / f"{name}.cache.jsonl", max_attempts=3, private_root=out,
                           grow_truncated=False)
    gen = phish_gen.run(client, model, persons, company_name=company_name, company_domain=company_domain,
                        temperature=temperature, seed=seed, negatives_per_positive=negatives_per_positive,
                        workers=workers)
    _write_jsonl(out / f"{name}.records.jsonl", gen["records"])
    assembled = phish_corpus.assemble(persons, gen["records"], model, negatives_per_positive=negatives_per_positive,
                                      strict=False)
    entry = phish_corpus.write(assembled, name, out)
    summary = {"generation": gen["summary"], "corpus": assembled["report"], "corpus_sha256": entry["sha256"]}
    (out / f"{name}.summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare and run the RQ3/RQ4 message generation.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="sample the persons of a case and write the input file")
    p.add_argument("--case", required=True)
    p.add_argument("--name", required=True, help="name of the input file under data/private/phish")
    p.add_argument("--n", type=int, default=None, help="sample size (default: all the persons)")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--stages", default=None, help="folder with infer.jsonl and judge.jsonl (default: the case folder)")
    p.add_argument("--meta", default=None, help="optional JSONL with person, role, org_type")
    g = sub.add_parser("generate", help="generate the messages of an input file with a local model")
    g.add_argument("--input", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--name", required=True, help="name of the outputs under the output folder")
    g.add_argument("--out-dir", default=None, help="default data/private/phish")
    g.add_argument("--company-name", default=DEFAULT_COMPANY_NAME)
    g.add_argument("--company-domain", default=DEFAULT_COMPANY_DOMAIN)
    g.add_argument("--temperature", type=float, default=phish_gen.DEFAULT_TEMPERATURE)
    g.add_argument("--seed", type=int, default=config.SEED)
    g.add_argument("--negatives", type=int, default=phish_gen.DEFAULT_NEGATIVES_PER_POSITIVE)
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--limit-persons", type=int, default=None, help="first k persons of the file (speed measurement)")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(json.dumps(prepare(a.case, a.name, n=a.n, seed=a.seed, stages=a.stages, meta_path=a.meta),
                         sort_keys=True))
    else:
        persons = _read_jsonl(Path(a.input))[:a.limit_persons]
        print(json.dumps(generate(persons, a.model, a.name, out_dir=a.out_dir, company_name=a.company_name,
                                  company_domain=a.company_domain, temperature=a.temperature, seed=a.seed,
                                  negatives_per_positive=a.negatives, workers=a.workers), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
