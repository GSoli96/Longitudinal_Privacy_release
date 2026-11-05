"""Embeddings (local bge-m3) of the attributes of cases whose inference is done, ahead of the de-duplication.

The vectors go to the shared embedding cache (data/private/llm/embed_cache.jsonl): the dedup and metrics stages
then find them there. Uses no cloud quota. Prints counts only.

    .venv\\Scripts\\python.exe scripts\\precompute_embeddings.py --cases D A B
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rerun import config, progress  # noqa: E402
from rerun.embed import Embedder  # noqa: E402
from rerun.run_case import _read_jsonl, stage_dir  # noqa: E402

CHUNK = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["D", "A", "B"])
    a = ap.parse_args()
    emb = Embedder(cache_path=config.PRIVATE / "llm" / "embed_cache.jsonl")
    for case in a.cases:
        rows = _read_jsonl(stage_dir(case) / "infer.jsonl")
        texts = sorted({x["text"] for r in rows if not r["carried"] for x in r["attributes"]})
        chunks = [texts[i:i + CHUNK] for i in range(0, len(texts), CHUNK)]
        with progress.Tracker(f"case{case}-embedding", "embedding (bge-m3)", len(texts), unit="attributes",
                              next_step="deduplication after the judges") as tr:
            for ch in chunks:
                emb(ch)
                tr.advance(len(ch))
        print(f"case {case}: {len(texts)} distinct attributes", flush=True)


if __name__ == "__main__":
    main()
