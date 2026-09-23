"""Domain tool layer.

Each subpackage is a typed capability the orchestration/agent layer can call:

- nlp: identifier tokenization + text normalization (Cyrillic-aware)
- ingest: XML / OCR / invoice-field extraction into domain objects
- compare: TF-IDF + cosine + fusion comparison and duplicate detection
- validate: deterministic invoice rule engine (no LLM in the math)
"""
