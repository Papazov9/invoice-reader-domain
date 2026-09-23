"""Trainable document-type classifier (assist-only).

A light TF-IDF + logistic-regression model over the document text, trained on labeled
examples (see backend/training). It is consulted ONLY when the keyword detector is unsure
(returns `other`) and the model is confident; decisive keyword phrases always win, and the
classifier never touches amounts or parties — only the document type.

sklearn/joblib are guarded so the app runs if the model artifact or libraries are absent
(it then degrades to the keyword detector).
"""

from __future__ import annotations

from pathlib import Path

from app.core import get_settings

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    _SKLEARN = True
except Exception:  # pragma: no cover - depends on environment
    _SKLEARN = False

_model = None
_tried = False


def build_model():
    """A fresh, untrained pipeline. Word 1-2 grams handle BG keyword cues; char_wb 3-5
    grams add robustness to OCR noise and spacing."""
    from sklearn.pipeline import FeatureUnion

    features = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def train(texts: list[str], labels: list[str]):
    model = build_model()
    model.fit(texts, labels)
    return model


def save(model, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)


def _load():
    global _model, _tried
    if _model is not None or _tried:
        return _model
    _tried = True
    if not _SKLEARN:
        return None
    path = Path(get_settings().doctype_model_path)
    if path.exists():
        try:
            _model = joblib.load(path)
        except Exception:  # pragma: no cover - corrupt artifact
            _model = None
    return _model


def predict(text: str) -> tuple[str, float] | None:
    """(doc_type, probability) from the trained model, or None when unavailable."""
    model = _load()
    if model is None or not (text or "").strip():
        return None
    try:
        proba = model.predict_proba([text])[0]
        idx = int(proba.argmax())
        return str(model.classes_[idx]), float(proba[idx])
    except Exception:  # pragma: no cover
        return None


def reset() -> None:
    """Drop the cached model (call after training/replacing the artifact)."""
    global _model, _tried
    _model = None
    _tried = False
