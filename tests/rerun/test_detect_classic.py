import pytest

from rerun import detect_classic as dc
from rerun import detect_metrics as dm

PHISH = ["Gentile {n}, urgente: confermi subito le credenziali su http://x.example/{n} entro oggi",
         "Buongiorno {n}, la sua password scade oggi, acceda subito a http://y.example/{n} per confermare"]
LEGIT = ["Gentile {n}, la riunione del consiglio è fissata per martedì prossimo alle dieci in sala grande",
         "Buongiorno {n}, in allegato il verbale della seduta precedente e l'ordine del giorno"]


def _records():
    recs = []
    for i in range(20):
        for y, pool in ((1, PHISH), (0, LEGIT)):
            recs.append({"text": pool[i % 2].format(n=f"P{i}"), "y": y, "person": f"p{i}",
                         "case": "ABCD"[i % 4], "generator": "g1" if i < 10 else "g2"})
    return recs


def test_group_split_is_disjoint_and_deterministic():
    recs = _records()
    tr, te = dc.group_split(recs, "person", seed=1)
    assert tr and te and not {r["person"] for r in tr} & {r["person"] for r in te}
    assert (tr, te) == dc.group_split(recs, "person", seed=1)


def test_holdout_by_case_and_generator():
    recs = _records()
    tr, te = dc.holdout_by(recs, "generator", ["g2"])
    assert {r["generator"] for r in te} == {"g2"} and {r["generator"] for r in tr} == {"g1"}


def test_style_features_see_url_urgency_and_length():
    f = dc.style_features("Urgente! Vai su http://a.example subito")
    assert f[0] == len("Urgente! Vai su http://a.example subito")
    assert f[9] == 1.0 and f[10] >= 2.0
    assert dc.style_features("")[0] == 0.0


def test_evaluate_fast_models_end_to_end():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    recs = _records()
    tr, te = dc.group_split(recs, "person", seed=3)
    models = {"lr": Pipeline([("tfidf", TfidfVectorizer(min_df=1)), ("clf", LogisticRegression(max_iter=500))]),
              "style": dc.style_model(), "length": dc.length_model()}
    out = dc.evaluate(tr, te, models)
    assert set(out) == {"lr", "style", "length"}
    assert len(out["lr"]) == len(te)
    assert dm.auc_roc([r["y"] for r in out["lr"]], [r["score"] for r in out["lr"]]) > 0.9
    assert dc.top_features(models["lr"], 5)[0][1] > 0
    assert dc.top_features(models["style"]) == []


def test_single_class_training_is_skipped_not_scored():
    recs = [r for r in _records() if r["y"] == 1]
    out = dc.evaluate(recs[:10], recs[10:], {"style": dc.style_model()})
    assert list(out) == ["_skipped"]


def test_ten_paper_models_are_available():
    assert len(dc.classical_models()) == 10
