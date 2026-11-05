"""Stage 6 - Phishing detection.

Detectors are evaluated on a balanced (1:1) corpus of generated phishing
and legitimate Italian e-mails, and on six public phishing benchmarks
(CEAS_08, Enron, Ling, Nazario, Nigerian Fraud, SpamAssassin).

Four families are supported:
  A) ten classical TF-IDF classifiers with 5-fold stratified CV;
  B) fine-tuned multilingual / Italian Transformers (optional, needs torch);
  C) LLMs used directly as zero-shot / few-shot classifiers (via Ollama);
  D) hard- and soft-voting ensembles over the trained models.

This module provides the model definitions and the metric helper; the full
training loop with checkpointing lives in the original experiment scripts
and is driven here through `get_classical_models()` and `metrics()`.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Transformer detectors used in the study (Section B).
NLP_MODEL_NAMES = (
    "bert-base-multilingual-cased",
    "distilbert-base-multilingual-cased",
    "xlm-roberta-base",
    "microsoft/mdeberta-v3-base",
    "dbmdz/bert-base-italian-cased",
    "dbmdz/electra-base-italian-xxl-cased-discriminator",
)

# LLM zero-shot / few-shot prompts (verbatim, Italian).
ZERO_SHOT = """\
Classifica la seguente email come phishing (1) o legittima (0).
Rispondi ESCLUSIVAMENTE con 0 o 1, senza altro testo.

Email:
{email}

Classificazione:"""

FEW_SHOT = """\
Classifica la seguente email come phishing (1) o legittima (0).
Rispondi ESCLUSIVAMENTE con 0 o 1, senza altro testo.

Esempi:
{examples}

Email:
{email}

Classificazione:"""


def _tfidf() -> dict:
    return dict(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        min_df=2,
    )


def get_classical_models() -> dict[str, Pipeline]:
    """The ten classical TF-IDF classifiers, with the paper's hyper-parameters."""
    t = _tfidf()
    return {
        "Logistic Regression": Pipeline([("tfidf", TfidfVectorizer(**t)),
                                         ("clf", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs"))]),
        "Multinomial Naive Bayes": Pipeline([("tfidf", TfidfVectorizer(**t)),
                                             ("clf", MultinomialNB(alpha=0.1))]),
        "Complement Naive Bayes": Pipeline([("tfidf", TfidfVectorizer(**t)),
                                            ("clf", ComplementNB(alpha=0.1))]),
        "Linear SVC": Pipeline([("tfidf", TfidfVectorizer(**t)),
                                ("clf", LinearSVC(C=1.0, max_iter=2000))]),
        "SGD Classifier": Pipeline([("tfidf", TfidfVectorizer(**t)),
                                    ("clf", SGDClassifier(loss="modified_huber", max_iter=100,
                                                          random_state=42, n_jobs=-1))]),
        "Decision Tree": Pipeline([("tfidf", TfidfVectorizer(**t)),
                                   ("clf", DecisionTreeClassifier(max_depth=20, random_state=42))]),
        "Random Forest": Pipeline([("tfidf", TfidfVectorizer(**t)),
                                   ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                                  random_state=42))]),
        "Extra Trees": Pipeline([("tfidf", TfidfVectorizer(**t)),
                                 ("clf", ExtraTreesClassifier(n_estimators=200, n_jobs=-1,
                                                              random_state=42))]),
        "Gradient Boosting": Pipeline([("tfidf", TfidfVectorizer(**t)),
                                       ("clf", GradientBoostingClassifier(n_estimators=150,
                                                                          learning_rate=0.1,
                                                                          max_depth=5,
                                                                          random_state=42))]),
        "AdaBoost": Pipeline([("tfidf", TfidfVectorizer(**t)),
                              ("clf", AdaBoostClassifier(n_estimators=150, learning_rate=0.8,
                                                         random_state=42))]),
    }


def metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    m = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            m["ROC-AUC"] = float(roc_auc_score(y_true, y_score))
        except Exception:  # noqa: BLE001
            m["ROC-AUC"] = 0.0
    return m


def parse_binary_label(text: str) -> int:
    """Extract a 0/1 label from an LLM classification reply."""
    m = re.search(r"\b([01])\b", text or "")
    return int(m.group(1)) if m else 0


def few_shot_context(X_train, y_train, n_per_class: int = 3) -> str:
    pos = np.where(np.asarray(y_train) == 1)[0][:n_per_class]
    neg = np.where(np.asarray(y_train) == 0)[0][:n_per_class]
    parts = [f"Email: {X_train[i][:300]}...\nClassificazione: 1" for i in pos]
    parts += [f"Email: {X_train[i][:300]}...\nClassificazione: 0" for i in neg]
    return "\n\n".join(parts)
