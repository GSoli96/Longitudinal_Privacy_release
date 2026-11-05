"""Tests of rq4_transformers: no model download, no network. The Transformer trainer is a fake that is injected."""
import csv
import hashlib
import json

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from rerun import benchmarks
from rerun import rq4_transformers as rt

NAMES = ("m1", "m2", "m3", "m4", "m5", "m6")
CANARY = "SEGRETO-DA-NON-SCRIVERE"
PHISH = ["Gentile {n}, urgente: confermi subito le credenziali su http://x.example/{n} entro oggi",
         "Buongiorno {n}, la sua password scade oggi, acceda subito a http://y.example/{n} per confermare"]
LEGIT = ["Gentile {n}, la riunione del consiglio è fissata per martedì prossimo alle dieci in sala grande",
         "Buongiorno {n}, in allegato il verbale della seduta precedente e l'ordine del giorno"]


def _records(n_persons=30, per=2, secret=False):
    recs = []
    for i in range(n_persons):
        for j in range(per):
            for y, pool in ((1, PHISH), (0, LEGIT)):
                text = pool[(i + j) % 2].format(n=f"P{i}x{j}") + (f" {CANARY}" if secret else "")
                recs.append({"id": f"r{i}_{j}_{y}", "text": text, "y": y, "person": f"p{i}", "case": "A",
                             "generator": "g"})
    return recs


def _classical():
    return {"LR": Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=200))]),
            "NB": Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())]),
            "SVC": Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])}


class FakeTrainer:
    """Probability from a keyword of the text, with a model-dependent share of wrong answers. Counts its calls."""

    def __init__(self, fail=()):
        self.calls, self.fail, self.seen = [], set(fail), []

    def __call__(self, model_name, train_texts, train_y, eval_texts, **hp):
        if model_name in self.fail:
            raise RuntimeError("out of memory")
        self.calls.append(model_name)
        self.seen.append((len(train_texts), dict(hp)))
        bad = int(model_name[1:]) if model_name[1:].isdigit() else 0          # m1 is the best, m6 the worst
        out = {}
        for name, texts in eval_texts.items():
            ps = []
            for t in texts:
                h = int(hashlib.sha256((model_name + t).encode()).hexdigest(), 16) % 100
                phish = "credenziali" in t or "password" in t
                phish = (not phish) if h < 3 * bad else phish
                ps.append(0.9 if phish else 0.1)
            out[name] = ps
        return out


def _run(tmp_path, trainer, **kw):
    kw.setdefault("generated", _records())
    kw.setdefault("ceas", _records(20, 3))
    kw.setdefault("n_train", 60)
    kw.setdefault("n_test", 20)
    return rt.run(settings=kw.pop("settings", rt.SETTINGS), models=NAMES, classical=_classical(), trainer=trainer,
                  n_resamples=20, out_dir=tmp_path / "ck", results_path=tmp_path / "res.json", **kw)


# ---- data ---------------------------------------------------------------------------------------------------------
def test_stratified_sample_sizes_disjoint_and_proportional():
    recs = [{"id": str(i), "y": int(i % 5 == 0)} for i in range(1000)]     # 20% positives
    tr, te = rt.stratified_sample(recs, 400, 100, seed=42)
    assert len(tr) == 400 and len(te) == 100
    assert not {r["id"] for r in tr} & {r["id"] for r in te}
    assert sum(r["y"] for r in tr) == 80 and sum(r["y"] for r in te) == 20
    assert (tr, te) == rt.stratified_sample(list(reversed(recs)), 400, 100, seed=42)
    assert tr != rt.stratified_sample(recs, 400, 100, seed=1)[0]
    with pytest.raises(ValueError):
        rt.stratified_sample(recs, 900, 200)


def test_split_by_person_has_no_overlap():
    d = rt.split_setting("generated_person_split", generated=_records(), val_fraction=0.2)
    persons = {k: {r["person"] for r in v} for k, v in d.items()}
    assert not persons["fit"] & persons["val"] and not persons["fit"] & persons["test"]
    assert not persons["val"] & persons["test"]
    assert persons["fit"] and persons["val"] and persons["test"]
    with pytest.raises(ValueError):
        rt.split_setting("nonsense")


# ---- voting -------------------------------------------------------------------------------------------------------
def test_select_members_by_validation_f1_then_auc_then_name():
    vm = {"a": {"f1": 0.9, "auc_roc": 0.9}, "b": {"f1": 0.9, "auc_roc": 0.95}, "c": {"f1": float("nan"), "auc_roc": 1.0},
          "d": {"f1": 0.5, "auc_roc": 0.5}, "e": {"f1": 0.9, "auc_roc": 0.9}}
    assert rt.select_members(vm, 4) == ["b", "a", "e", "d"]
    assert rt.select_members(vm, 5)[-1] == "c"


def test_hard_vote_majority_and_tie_goes_to_legitimate():
    members = [{"p": [0.9, 0.9, 0.1], "pred": [1, 1, 0]}, {"p": [0.9, 0.1, 0.1], "pred": [1, 0, 0]}]
    h = rt.hard_vote(members)
    assert h["pred"] == [1, 0, 0] and h["p"] == [1.0, 0.5, 0.0]
    three = members + [{"p": [0.9, 0.9, 0.1], "pred": [1, 1, 0]}]
    assert rt.hard_vote(three)["pred"] == [1, 1, 0]


def test_soft_vote_averages_probabilities():
    members = [{"p": [0.9, 0.2], "pred": [1, 0]}, {"p": [0.4, 0.2], "pred": [0, 0]}]
    s = rt.soft_vote(members)
    assert s["p"] == pytest.approx([0.65, 0.2]) and s["pred"] == [1, 0]


def test_linear_svc_probability_is_sigmoid_of_decision():
    recs = _records()
    out = rt.classical_predict(_classical()["SVC"], recs, {"t": recs})
    assert all(0.0 <= p <= 1.0 for p in out["t"]["p"])
    assert [int(p >= 0.5) for p in out["t"]["p"]] == out["t"]["pred"]


# ---- run ----------------------------------------------------------------------------------------------------------
def test_run_end_to_end_with_fake_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "hf_train_predict", lambda *a, **k: pytest.fail("the real trainer was called"))
    tr = FakeTrainer()
    res = _run(tmp_path, tr)
    assert set(res["settings"]) == set(rt.SETTINGS)
    for s in rt.SETTINGS:
        v = res["settings"][s]
        assert set(v["detectors"]) == set(NAMES) | {"LR", "NB", "SVC"} and not v["skipped"]
        ens = v["ensemble"]
        assert len(ens["members"]) == 6 and set(ens["validation"]) == set(ens["members"])
        for label in ("hard_voting", "soft_voting"):
            m = ens[label]["metrics"]["as_generated"]
            assert set(m) == {"auc_roc", "auprc", "f1", "recall_at_fpr"}
            assert {"estimate", "lo", "hi"} <= set(m["f1"])
        assert v["n_fit"] + v["n_val"] + v["n_test"] > 0
    assert json.loads((tmp_path / "res.json").read_text(encoding="utf-8"))["settings"].keys() == res["settings"].keys()
    # hyper-parameters reach the trainer; the fit part is smaller than the train part
    assert all(hp["epochs"] == 3 and hp["lr"] == 2e-5 and hp["max_len"] == 256 and hp["seed"] == 42
               for _, hp in tr.seen)
    # the best fake models are chosen as members, the worst one is out of a pool of nine only if beaten
    members = res["settings"]["generated_person_split"]["ensemble"]["members"]
    assert "m1" in members


def test_checkpoint_has_scores_but_no_text_and_summary_has_no_text(tmp_path):
    _run(tmp_path, FakeTrainer(), generated=_records(secret=True), ceas=_records(20, 3, secret=True))
    ck = (tmp_path / "ck" / "rq4_transformers.preds.jsonl").read_text(encoding="utf-8")
    res = (tmp_path / "res.json").read_text(encoding="utf-8")
    for blob in (ck, res):
        assert CANARY not in blob and "credenziali" not in blob and "riunione" not in blob
    lines = [json.loads(x) for x in ck.splitlines()]
    preds = [x for x in lines if "|pred|" in x["key"]]
    assert len(preds) == 2 * 9
    split = next(x for x in lines if x["key"].endswith("|split"))["payload"]
    one = next(x for x in preds if x["setting"] == "generated_person_split")["payload"]
    assert len(one["test"]["p"]) == len(split["test_ids"]) and len(one["val"]["pred"]) == len(split["val_ids"])


def test_members_do_not_depend_on_test_labels(tmp_path):
    base = _run(tmp_path / "a", FakeTrainer(), settings=("generated_person_split",))
    flipped = _records()
    data = rt.split_setting("generated_person_split", generated=flipped)
    test_ids = {r["id"] for r in data["test"]}
    for r in flipped:
        if r["id"] in test_ids:
            r["y"] = 1 - r["y"]
    other = _run(tmp_path / "b", FakeTrainer(), settings=("generated_person_split",), generated=flipped)
    s = "generated_person_split"
    assert base["settings"][s]["ensemble"]["members"] == other["settings"][s]["ensemble"]["members"]
    assert base["settings"][s]["ensemble"]["validation"] == other["settings"][s]["ensemble"]["validation"]


def test_resume_skips_everything_done(tmp_path):
    first = FakeTrainer()
    r1 = _run(tmp_path, first)
    ck = tmp_path / "ck" / "rq4_transformers.preds.jsonl"
    n_lines = len(ck.read_text(encoding="utf-8").splitlines())

    class Boom(FakeTrainer):
        def __call__(self, *a, **k):
            raise AssertionError("trainer called on resume")

    r2 = _run(tmp_path, Boom())
    assert len(ck.read_text(encoding="utf-8").splitlines()) == n_lines
    assert r2["resumed_items"] > 0
    assert r2["settings"] == r1["settings"]


def test_resume_after_a_failed_model_trains_only_that_one(tmp_path):
    r1 = _run(tmp_path, FakeTrainer(fail={"m3"}), settings=("generated_person_split",))
    v = r1["settings"]["generated_person_split"]
    assert v["skipped"] == {"m3": "RuntimeError"} and "m3" not in v["detectors"]
    assert v["ensemble"]["skipped"] == "transformers_missing"
    second = FakeTrainer()
    r2 = _run(tmp_path, second, settings=("generated_person_split",))
    assert second.calls == ["m3"]
    v2 = r2["settings"]["generated_person_split"]
    assert not v2["skipped"] and len(v2["ensemble"]["members"]) == 6


def test_changed_hyper_parameters_or_data_refit_but_more_resamples_do_not(tmp_path):
    _run(tmp_path, FakeTrainer(), settings=("generated_person_split",))
    t2 = FakeTrainer()
    _run(tmp_path, t2, settings=("generated_person_split",), hp={"epochs": 1})
    assert sorted(t2.calls) == sorted(NAMES)
    t3 = FakeTrainer()
    kw = dict(settings=("generated_person_split",), generated=_records())
    rt.run(models=NAMES, classical=_classical(), trainer=t3, n_resamples=30, out_dir=tmp_path / "ck",
           results_path=tmp_path / "res.json", hp={"epochs": 1}, n_train=60, n_test=20, **kw)
    assert t3.calls == []
    t4 = FakeTrainer()
    _run(tmp_path, t4, settings=("generated_person_split",), generated=_records(25))
    assert sorted(t4.calls) == sorted(NAMES)


def test_pool_transformers_uses_the_six_models_only(tmp_path):
    res = _run(tmp_path, FakeTrainer(), settings=("ceas_08",), pool="transformers")
    ens = res["settings"]["ceas_08"]["ensemble"]
    assert sorted(ens["members"]) == sorted(NAMES) and ens["pool"] == "transformers"
    with pytest.raises(ValueError):
        _run(tmp_path, FakeTrainer(), pool="bad")


def test_single_class_train_is_reported_not_scored(tmp_path):
    recs = [dict(r, y=0) for r in _records()]
    res = _run(tmp_path, FakeTrainer(), settings=("generated_person_split",), generated=recs)
    assert res["settings"]["generated_person_split"]["skipped"] == {"all": "single_class_or_empty_split"}


# ---- files --------------------------------------------------------------------------------------------------------
def _csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "body", "label"])
        w.writerows(rows)


def test_ceas_loads_from_prepared_benchmark_and_corpus_from_manifest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rows = [[f"s{i}", (PHISH if i % 3 == 0 else LEGIT)[i % 2].format(n=f"n{i}"), str(int(i % 3 == 0))]
            for i in range(120)]
    _csv(src / "CEAS_08.csv", rows)
    for n in ("Enron", "Ling", "Nazario", "Nigerian_Fraud", "SpamAssasin"):
        _csv(src / benchmarks.BENCHMARKS[n], [["s", "b", "0"]])
    bench = tmp_path / "bench"
    benchmarks.prepare(src, bench)
    phish = tmp_path / "phish"
    phish.mkdir()
    gen_rows = [{"id": r["id"], "person": r["person"], "case": "A", "variant": "none", "attack": "a",
                 "kind": "x", "index": 0, "generator": "g", "label": r["y"], "subject": "s", "body": r["text"]}
                for r in _records()]
    p = phish / "toy.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in gen_rows) + "\n", encoding="utf-8")
    (phish / "manifest.json").write_text(json.dumps({"toy": {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}}),
                                         encoding="utf-8")
    res = rt.run(["toy"], models=NAMES, classical=_classical(), trainer=FakeTrainer(), n_resamples=10, n_train=60,
                 n_test=20, out_dir=tmp_path / "ck", results_path=tmp_path / "res.json", corpus_dir=phish,
                 bench_dir=bench)
    assert res["settings"]["ceas_08"]["n_test"] == 20
    assert res["settings"]["ceas_08"]["n_fit"] + res["settings"]["ceas_08"]["n_val"] == 60
    assert res["settings"]["generated_person_split"]["n_test"] > 0
    assert res["config"]["corpora"] == ["toy"]


def test_cli_needs_a_corpus_for_the_generated_setting(capsys):
    with pytest.raises(SystemExit):
        rt.main(["--settings", "generated_person_split"])
    assert "--corpus" in capsys.readouterr().err


def test_default_hyper_parameters_and_models_are_the_ones_of_the_study():
    assert rt.HP == {"epochs": 3, "lr": 2e-5, "max_len": 256, "batch_size": 16, "seed": 42}
    assert len(rt.transformer_names()) == 6


def test_hf_loop_trains_a_tiny_local_model_without_network(tmp_path, monkeypatch):
    """The real training loop on a 2-layer random BERT built in a temp folder (CPU, a few seconds, no download)."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    words = sorted({w.lower().strip(",.:") for t in PHISH + LEGIT for w in t.split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    d = tmp_path / "tiny"
    d.mkdir()
    (d / "vocab.txt").write_text(chr(10).join(vocab), encoding="utf-8")
    tok = transformers.BertTokenizerFast(vocab_file=str(d / "vocab.txt"))
    tok.save_pretrained(d)
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=32, max_position_embeddings=64)
    transformers.BertModel(cfg).save_pretrained(d)
    cfg.save_pretrained(d)
    recs = _records(20, 1)
    texts, ys = [r["text"] for r in recs], [r["y"] for r in recs]
    out = rt.hf_train_predict(str(d), texts, ys, {"a": texts[:6], "b": texts[:3]}, epochs=2, lr=1e-2, max_len=32,
                              batch_size=8, seed=1)
    assert len(out["a"]) == 6 and len(out["b"]) == 3 and all(0.0 <= p <= 1.0 for p in out["a"])
    again = rt.hf_train_predict(str(d), texts, ys, {"a": texts[:6]}, epochs=2, lr=1e-2, max_len=32, batch_size=8,
                                seed=1)
    assert again["a"] == pytest.approx(out["a"])
