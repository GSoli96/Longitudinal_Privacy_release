"""Tests of rerun.embed: batching, cache (cache hits identical to fresh calls), normalisation, failures."""
import hashlib

import numpy as np
import pytest

from rerun import embed


def vec_of(text, dim=8):
    h = hashlib.sha256(text.encode()).digest()
    return [(b - 127.5) / 127.5 for b in h[:dim]]


class Transport:
    def __init__(self, fail=0, status=200):
        self.calls, self.fail, self.status = [], fail, status

    def __call__(self, url, body, timeout):
        self.calls.append(body)
        if self.fail:
            self.fail -= 1
            return 500, {}
        return self.status, {"embeddings": [vec_of(t) for t in body["input"]]}


def make(tmp_path, **kw):
    t = kw.pop("transport", None) or Transport()
    e = embed.Embedder(cache_path=tmp_path / "emb.jsonl", transport=t, private_root=tmp_path, sleep=lambda s: None, **kw)
    return e, t


def test_vectors_are_unit_length_and_in_input_order(tmp_path):
    e, _ = make(tmp_path)
    v = e(["uno", "due", "tre"])
    assert v.shape == (3, 8)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)
    single = e(["due"])
    assert np.allclose(v[1], single[0])


def test_batches_and_duplicates_are_sent_once(tmp_path):
    e, t = make(tmp_path, batch_size=2)
    e(["a", "b", "c", "a", "b"])
    sent = [x for call in t.calls for x in call["input"]]
    assert sorted(sent) == ["a", "b", "c"] and len(t.calls) == 2 and all(len(c["input"]) <= 2 for c in t.calls)
    assert t.calls[0]["model"] == "bge-m3"


def test_cache_hit_is_identical_to_the_fresh_vector_and_makes_no_call(tmp_path):
    e1, t1 = make(tmp_path)
    fresh = e1(["alpha", "beta"])
    e2, t2 = make(tmp_path)                    # new process, same cache file
    cached = e2(["alpha", "beta"])
    assert t2.calls == []
    assert np.array_equal(fresh, cached)       # exactly equal, not just close


def test_only_missing_texts_are_requested(tmp_path):
    e1, _ = make(tmp_path)
    e1(["alpha"])
    e2, t2 = make(tmp_path)
    e2(["alpha", "gamma"])
    assert [x for c in t2.calls for x in c["input"]] == ["gamma"]


def test_failed_batch_is_retried_then_raises_never_returns_zeros(tmp_path):
    e, t = make(tmp_path, transport=Transport(fail=1))
    assert e(["x"]).shape == (1, 8) and len(t.calls) == 2
    e2, _ = make(tmp_path / "two" if (tmp_path / "two").mkdir() is None else tmp_path, transport=Transport(fail=10),
                 max_attempts=3)
    with pytest.raises(RuntimeError):
        e2(["never cached text"])


def test_wrong_number_of_embeddings_is_an_error(tmp_path):
    class Bad:
        def __call__(self, url, body, timeout):
            return 200, {"embeddings": [[1.0, 0.0]]}
    e, _ = make(tmp_path, transport=Bad())
    with pytest.raises(RuntimeError):
        e(["a", "b"])


def test_empty_input_gives_an_empty_array(tmp_path):
    e, t = make(tmp_path)
    assert e([]).shape[0] == 0 and t.calls == []


def test_cache_must_live_under_the_private_root(tmp_path):
    inside = tmp_path / "private"
    inside.mkdir()
    embed.Embedder(cache_path=inside / "e.jsonl", transport=Transport(), private_root=inside)
    with pytest.raises(ValueError):
        embed.Embedder(cache_path=tmp_path / "e.jsonl", transport=Transport(), private_root=inside)


def test_truncated_last_cache_line_is_ignored(tmp_path):
    e1, _ = make(tmp_path)
    e1(["alpha"])
    with open(tmp_path / "emb.jsonl", "a", encoding="utf-8") as f:
        f.write('{"key": "abc", "vec": "AAAA')
    e2, t2 = make(tmp_path)
    e2(["alpha"])
    assert t2.calls == []
