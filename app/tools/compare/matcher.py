"""Document comparison and duplicate detection.

Word + char TF-IDF, cosine, then linear fusion. Two entry points:
compare_documents scores a pair of documents with evidence; find_duplicates ranks
candidates against a query and flags likely duplicates.

The vectorizer is fit per call on the documents in scope. Fine for pairwise and
small-batch dedup; ledger-wide search uses the pgvector index instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core import get_settings
from app.domain import DocCandidate, DuplicateMatch, MatchEvidence
from app.tools.nlp import normalize, split_identifier


def prepare_texts(doc: DocCandidate) -> tuple[str, str]:
    """Build (word_text, char_text) representations from a DocCandidate.

    word_text: field names are identifier-split, values are normalized (semantic).
    char_text: everything normalized but identifiers kept whole, for pattern matching
    that survives OCR typos and abbreviations.
    """
    word_parts: list[str] = []
    char_parts: list[str] = []
    for unit in doc.units:
        if not unit.text:
            continue
        if unit.kind == "fieldName":
            word_parts.append(split_identifier(unit.text))
            char_parts.append(normalize(unit.text))
        else:  # value / context
            n = normalize(unit.text)
            word_parts.append(n)
            char_parts.append(n)
    return " ".join(p for p in word_parts if p), " ".join(p for p in char_parts if p)


def _cosine_row(texts: list[str], analyzer: str, ngram: tuple[int, int]) -> np.ndarray:
    """Fit TF-IDF over *texts* and return similarity of texts[0] vs texts[1:]."""
    non_empty = [t for t in texts if t.strip()]
    if len(non_empty) < 2:
        return np.zeros(len(texts) - 1)
    kwargs = {"analyzer": analyzer, "ngram_range": ngram, "min_df": 1}
    if analyzer == "char":
        kwargs["analyzer"] = "char_wb"
    try:
        matrix = TfidfVectorizer(**kwargs).fit_transform(texts)
    except ValueError:
        # empty vocabulary (e.g. all stop-like) means no signal
        return np.zeros(len(texts) - 1)
    return cosine_similarity(matrix[0:1], matrix[1:]).ravel()


def _fuse(word: float, char: float) -> float:
    w_word, w_char = get_settings().ir_weights_normalized
    return float(w_word * word + w_char * char)


def compare_documents(a: DocCandidate, b: DocCandidate) -> MatchEvidence:
    """Compare two documents and return word/char/fused similarity."""
    settings = get_settings()
    a_word, a_char = prepare_texts(a)
    b_word, b_char = prepare_texts(b)

    word_sim = float(_cosine_row([a_word, b_word], "word", settings.ir_word_ngram)[0])
    char_sim = float(_cosine_row([a_char, b_char], "char", settings.ir_char_ngram)[0])
    return MatchEvidence(
        word_similarity=word_sim,
        char_similarity=char_sim,
        fused_score=_fuse(word_sim, char_sim),
    )


def find_duplicates(
    query: DocCandidate,
    candidates: list[DocCandidate],
    *,
    top_k: int = 5,
) -> list[DuplicateMatch]:
    """Rank *candidates* by similarity to *query*; flag those above the dup threshold."""
    if not candidates:
        return []

    settings = get_settings()
    q_word, q_char = prepare_texts(query)
    prepared = [prepare_texts(c) for c in candidates]

    word_sims = _cosine_row([q_word, *[p[0] for p in prepared]], "word", settings.ir_word_ngram)
    char_sims = _cosine_row([q_char, *[p[1] for p in prepared]], "char", settings.ir_char_ngram)

    matches: list[DuplicateMatch] = []
    for cand, w, c in zip(candidates, word_sims, char_sims):
        fused = _fuse(float(w), float(c))
        matches.append(
            DuplicateMatch(
                candidate_id=cand.id,
                score=round(fused, 6),
                is_duplicate=fused >= settings.ir_duplicate_threshold,
                evidence=MatchEvidence(
                    word_similarity=float(w), char_similarity=float(c), fused_score=fused
                ),
            )
        )

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:top_k]
