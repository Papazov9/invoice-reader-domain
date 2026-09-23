"""Trainable field selectors (assist-only).

A small classifier per field group scores the deterministic candidates and PICKS which is
which. The picked candidate's value is verbatim from the document, so amounts stay exact —
only the *selection* is learned. Models are loaded once and gated by config; absent → the
selectors return nothing and the deterministic extractor stands.

Artifacts (gitignored) under field_models_dir:
    amounts.joblib  party.joblib  number.joblib  date.joblib  direction.joblib
"""

from __future__ import annotations

from pathlib import Path

from app.core import get_settings

from . import candidates as C

try:
    import joblib
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    _SK = True
except Exception:  # pragma: no cover - depends on environment
    _SK = False

# field group -> artifact filename
_FILES = {"amounts": "amounts.joblib", "party": "party.joblib",
          "number": "number.joblib", "date": "date.joblib", "direction": "direction.joblib"}

_bundle: dict | None = None
_tried = False


def build_selector():
    """DictVectorizer + logistic regression over candidate feature dicts."""
    return Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def save_selector(model, group: str, models_dir: str | Path | None = None) -> None:
    d = Path(models_dir or get_settings().field_models_dir)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / _FILES[group])


def reset() -> None:
    global _bundle, _tried
    _bundle, _tried = None, False


def set_models(bundle: dict) -> None:
    """Inject in-memory selectors (used by cross-validation) instead of loading from disk."""
    global _bundle, _tried
    _bundle, _tried = bundle, True


def _models() -> dict:
    global _bundle, _tried
    if _bundle is not None or _tried:
        return _bundle or {}
    _tried = True
    out: dict = {}
    if _SK and get_settings().field_models_enabled:
        d = Path(get_settings().field_models_dir)
        for group, fname in _FILES.items():
            p = d / fname
            if p.exists():
                try:
                    out[group] = joblib.load(p)
                except Exception:  # pragma: no cover
                    pass
    _bundle = out
    return out


def _proba(model, row: dict) -> dict[str, float]:
    p = model.predict_proba([row])[0]
    return {str(c): float(v) for c, v in zip(model.classes_, p)}


def _best(model, cands, cls: str, min_proba: float):
    """The candidate with the highest probability for class `cls`, if above threshold."""
    best, best_p = None, min_proba
    for c in cands:
        p = _proba(model, c.features()).get(cls, 0.0)
        if p >= best_p:
            best, best_p = c, p
    return best


# --- runtime selection (returns verbatim values; None when no model/weak) ---

_OPP_CUR = {"EUR": ("лв", "bgn", "лева"), "BGN": ("eur", "€", "евро")}


def select_amounts(text: str) -> dict[str, str]:
    model = _models().get("amounts")
    if model is None:
        return {}
    cands = C.amount_candidates(text)
    # On euro-transition invoices that print both currencies, drop candidates on lines in
    # the OTHER currency so the selector can't pick the BGN equivalent of a EUR figure.
    from app.tools.ingest.currency import detect_currency_text

    opp = _OPP_CUR.get(detect_currency_text(text) or "")
    if opp:
        lines = text.split("\n")
        kept = [c for c in cands if not any(t in lines[c.line_idx].lower() for t in opp)]
        cands = kept or cands
    mp = get_settings().field_model_min_proba
    out: dict[str, str] = {}
    for cls in ("net", "vat", "total"):
        c = _best(model, cands, cls, mp)
        if c is not None:
            out[cls] = str(c.value)
    return out


def select_parties(text: str) -> dict[str, C.PartyCandidate]:
    model = _models().get("party")
    if model is None:
        return {}
    cands = C.party_candidates(text)
    mp = get_settings().field_model_min_proba
    out: dict[str, C.PartyCandidate] = {}
    for cls in ("supplier", "recipient"):
        c = _best(model, cands, cls, mp)
        if c is not None:
            out[cls] = c
    return out


def select_number(text: str, exclude: set[str] | None = None) -> str | None:
    model = _models().get("number")
    if model is None:
        return None
    c = _best(model, C.number_candidates(text, exclude), "yes", get_settings().field_model_min_proba)
    return c.value if c else None


def select_date(text: str) -> str | None:
    model = _models().get("date")
    if model is None:
        return None
    c = _best(model, C.date_candidates(text), "yes", get_settings().field_model_min_proba)
    return c.value if c else None


def select_direction(text: str) -> str | None:
    model = _models().get("direction")
    if model is None or not (text or "").strip():
        return None
    try:
        proba = model.predict_proba([text])[0]
        idx = int(proba.argmax())
        if float(proba[idx]) >= get_settings().field_model_min_proba:
            return str(model.classes_[idx])
    except Exception:  # pragma: no cover
        return None
    return None


def available() -> bool:
    return get_settings().field_models_enabled and bool(_models())
