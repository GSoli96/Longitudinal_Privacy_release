"""Export a directory of a git repository from its objects (no checkout).

Needed when file names cannot exist side by side on the local file system
(names differing only by case on Windows, ':' in names): every blob is read
from the object store and nothing is overwritten.

Names that collide when compared case-insensitively get the short blob hash
before the extension (all variants of the group, so no variant is privileged).

    py -3.11 -I scripts/export_git_dir.py <git_repo> <tree_dir> <out_dir>

Writes <out_dir>/MANIFEST.tsv (original path, exported path, sha, size).
The output directory must be git-ignored (personal data).
"""
import os
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import PurePosixPath


def list_blobs(repo, tree_dir):
    raw = subprocess.run(
        ["git", "-C", repo, "ls-tree", "-r", "-z", "--long", "HEAD", "--", tree_dir],
        capture_output=True, check=True).stdout
    rows = []
    for rec in raw.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, sha, size = meta.split()
        if typ == b"blob":
            # paths with bytes that are not valid UTF-8 are exported with U+FFFD; the group
            # check below fails loudly if that makes two names collide
            rows.append((sha.decode(), int(size), path.decode("utf-8", errors="replace")))
    return rows


def main(repo, tree_dir, out_dir):
    prefix = tree_dir.rstrip("/") + "/"
    rows = list_blobs(repo, tree_dir)
    groups = defaultdict(list)
    for sha, size, path in rows:
        rel = path[len(prefix):]
        groups[unicodedata.normalize("NFC", rel).lower()].append((sha, size, rel))
    plan = []
    for items in groups.values():
        for sha, size, rel in items:
            if len(items) > 1:
                p = PurePosixPath(rel)
                rel_out = str(p.with_name(f"{p.stem}__{sha[:8]}{p.suffix}"))
            else:
                rel_out = rel
            plan.append((sha, size, prefix + rel, rel_out))
    assert len({r[3].lower() for r in plan}) == len(plan), "exported names still collide"

    proc = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    manifest = []
    for sha, size, orig, rel_out in plan:
        proc.stdin.write(sha.encode() + b"\n")
        proc.stdin.flush()
        head = proc.stdout.readline().split()
        assert head[0].decode() == sha and int(head[2]) == size, (head, sha)
        data = proc.stdout.read(size)
        proc.stdout.read(1)
        dest = os.path.abspath(os.path.join(out_dir, *rel_out.split("/")))
        dest = "\\\\?\\" + dest if os.name == "nt" else dest
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        manifest.append((orig, rel_out, sha, size))
    proc.stdin.close()
    proc.wait()
    with open(os.path.join(out_dir, "MANIFEST.tsv"), "w", encoding="utf-8") as f:
        f.write("original_path\texported_path\tsha\tsize\n")
        for row in manifest:
            f.write("\t".join(map(str, row)) + "\n")
    collided = sum(1 for g in groups.values() if len(g) > 1)
    print(f"blobs in HEAD: {len(rows)} | exported: {len(manifest)} | "
          f"collision groups: {collided}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
