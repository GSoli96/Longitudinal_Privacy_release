"""Classical TF-IDF detectors of the RQ4 stage (plan ``docs/plans/rq34.md`` §4, family A). CPU only.

The ten pipelines are those of the paper (``src.detection.get_classical_models``) so that the rerun and the
published setting differ only in the data. Records are dicts with ``text``, ``y`` (1 = phishing) and the grouping
keys ``person``, ``case`` and ``generator``.

* ``holdout_by`` / ``group_split``: train and test never share a group (person, case or generator).
* ``style_features`` / ``style_model`` / ``length_model``: detectors that see only style (length, punctuation, URLs,
  fixed urgency formulae) or only the length, to separate style from content (an artefact check).
* ``evaluate``: fits every model on the training records and returns, per model, scored test records
  ``{y, score, pred, person, case, generator}`` for ``detect_metrics``. Nothing is tuned on the test records.
* ``top_features``: the n terms with the largest weight of a linear model (artefact check)."""
from __future__ import annotations

import random
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

URGENCY = ("urgent", "immediat", "subito", "entro", "scadenz", "asap", "deadline", "oggi", "today", "now")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)


def classical_models() -> dict:
    from src import detection
    return detection.get_classical_models()


def style_features(text: str) -> list:
    """Length, punctuation, capitals, digits, URLs and urgency formulae; no word identity."""
    text = text or ""
    n = max(len(text), 1)
    words = text.split()
    low = text.lower()
    return [float(len(text)), float(len(words)), len(text) / max(len(words), 1),
            text.count("!") / n, text.count("?") / n, text.count(",") / n, text.count(".") / n,
            sum(c.isupper() for c in text) / n, sum(c.isdigit() for c in text) / n,
            float(len(_URL.findall(text))), float(sum(low.count(u) for u in URGENCY)), float(text.count("\n"))]


def _style_matrix(texts):
    return np.array([style_features(t) for t in texts], dtype=float)


def _length_matrix(texts):
    return np.array([[float(len(t or ""))] for t in texts], dtype=float)


def style_model() -> Pipeline:
    return Pipeline([("feat", FunctionTransformer(_style_matrix)), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000))])


def length_model() -> Pipeline:
    return Pipeline([("feat", FunctionTransformer(_length_matrix)), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000))])


def holdout_by(records, key: str, test_values) -> tuple:
    """Train = records whose ``key`` is not in ``test_values``; test = the others."""
    test_values = set(test_values)
    return ([r for r in records if r[key] not in test_values], [r for r in records if r[key] in test_values])


def group_split(records, key: str, *, test_fraction: float = 0.3, seed: int = 42) -> tuple:
    """Random split with disjoint groups (at least one group on each side when there are two or more)."""
    groups = sorted({r[key] for r in records}, key=str)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_test = min(max(1, round(len(groups) * test_fraction)), max(len(groups) - 1, 0))
    return holdout_by(records, key, groups[:n_test])


def _scores(model, texts) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(texts), dtype=float)
    return np.asarray(model.predict_proba(texts)[:, 1], dtype=float)


def evaluate(train, test, models: dict | None = None) -> dict:
    """``{model_name: [scored test record, ...]}``. A model that cannot be fitted (one class in the training
    records) is skipped and listed under the key ``"_skipped"`` with its error type, never scored as 0."""
    models = classical_models() if models is None else models
    y_train = [r["y"] for r in train]
    out, skipped = {}, {}
    if len(set(y_train)) < 2 or not test:
        return {"_skipped": {name: "single_class_or_empty" for name in models}}
    x_train = [r["text"] for r in train]
    x_test = [r["text"] for r in test]
    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
            s = _scores(model, x_test)
            p = model.predict(x_test)
        except Exception as e:  # noqa: BLE001
            skipped[name] = type(e).__name__
            continue
        out[name] = [{"y": r["y"], "score": float(si), "pred": int(pi), "person": r.get("person"),
                      "case": r.get("case"), "generator": r.get("generator")}
                     for r, si, pi in zip(test, s, p)]
    if skipped:
        out["_skipped"] = skipped
    return out


def top_features(model: Pipeline, n: int = 20) -> list:
    """The n terms with the largest positive weight (towards phishing) of a fitted linear TF-IDF pipeline;
    ``[]`` when the classifier has no coefficients."""
    clf = model.named_steps.get("clf")
    tfidf = model.named_steps.get("tfidf")
    coef = getattr(clf, "coef_", None)
    if coef is None or tfidf is None:
        return []
    w = np.asarray(coef).reshape(-1)
    names = tfidf.get_feature_names_out()
    order = np.argsort(-w)[:n]
    return [(str(names[i]), float(w[i])) for i in order]
