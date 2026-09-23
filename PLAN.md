# Invoice Reader — Technical Plan

## Overview

Bulgarian invoice processing microservice. Accepts PDF and image uploads, extracts
structured invoice data via Tesseract OCR (with optional Ollama vision fallback), and
returns a typed `Invoice` JSON response. Self-contained: no external API keys required;
Tesseract runs locally and the vision model (qwen2.5vl) is served by the companion
Ollama container defined in `docker-compose.yml`.

Entry point: `app.py` (project root) → wraps `app.ingest_service:app`.

---

## 1. Project Structure

```
invoice-bot/
├── app.py                        # Root entry point (stack rule: app object + __main__)
├── requirements.txt              # Pinned deps for the lean ingest service
├── requirements-ingest.txt       # Original unpinned source (imported, keep for reference)
├── Dockerfile                    # Multi-stage build: system deps + pip install
├── docker-compose.yml            # Ollama (qwen2.5vl) + reader services
├── .env.example                  # Token template
├── README.md                     # Deployment guide (imported)
└── app/
    ├── __init__.py               # version = "0.1.0"
    ├── ingest_service.py         # Lean FastAPI app (the real app, imported from GitHub)
    ├── domain/
    │   ├── __init__.py           # Re-exports: Invoice, Party, LineItem, …
    │   └── models.py             # Pydantic v2 domain models (Invoice, Party, LineItem,
    │                             #   TaxLine, Company, CompanyGroup, ValidationResult)
    └── tools/
        └── ingest/
            ├── __init__.py       # Public API: extract_from_pdf_bytes,
            │                     #   extract_from_image_bytes, ocr_status
            ├── extract.py        # Top-level coordinator (OCR → parse → augment → group)
            ├── ocr.py            # Tesseract wrapper (OcrResult, mean_conf)
            ├── invoice_extractor.py  # Regex field extraction (Bulgarian locale)
            ├── document_types.py    # DocType enum + direction detection
            ├── xml_invoice.py       # ERP schema resolver (SAP/UBL/Controlisy)
            ├── csv_invoice.py       # VAT ledger parser
            ├── xml_parser.py        # XXE-safe defusedxml wrapper
            ├── vision_extract.py    # Ollama vision fallback (qwen2.5vl)
            ├── llm_assist.py        # Few-shot LLM assist for weak fields (optional)
            ├── field_models.py      # Optional sklearn selectors
            ├── candidates.py        # Candidate generation for field selection
            ├── company.py           # Company identity derivation + grouping
            ├── company_lookup.py    # web.company.guru register scraper (optional)
            ├── eik.py               # BULSTAT/EIK modulo-11 validation
            ├── currency.py          # Currency detection (BGN/EUR, Cyrillic spellings)
            ├── bg_amount_words.py   # Bulgarian words → Decimal ("петстотин" → 500)
            ├── controlisy.py        # Controlisy export format parser
            ├── ms_rowset.py         # MS SQL rowset format parser
            └── nlp/
                ├── normalize.py     # Unicode-aware text normalisation (Cyrillic-safe)
                └── tokenizer.py     # camelCase/snake_case identifier splitter
```

> All files inside `app/` already exist (imported from the upstream repo).
> Only `app.py` and `requirements.txt` are newly created by this plan.

---

## 2. Dependencies (`requirements.txt`)

Lean set — covers the stateless ingest microservice only (no Postgres, no RAG index,
no torch/sentence-transformers):

| Package | Pinned version | Purpose |
|---|---|---|
| `fastapi` | 0.115.6 | Web framework |
| `uvicorn[standard]` | 0.32.1 | ASGI server (with websockets, http-tools) |
| `python-multipart` | 0.0.12 | Form/file upload parsing |
| `pydantic` | 2.10.4 | Domain model validation |
| `pydantic-settings` | 2.7.0 | Env-driven config |
| `pytesseract` | 0.3.13 | Tesseract OCR Python binding |
| `pdf2image` | 1.17.0 | PDF → PIL image (via poppler) |
| `pillow` | 11.1.0 | Image processing |
| `opencv-python-headless` | 4.10.0.84 | Deskew / denoise preprocessing |
| `numpy` | 1.26.4 | Array ops used by OpenCV |
| `beautifulsoup4` | 4.12.3 | Company lookup HTML parsing |
| `defusedxml` | 0.7.1 | XXE-safe XML parsing |
| `requests` | 2.32.3 | Vision model HTTP calls (Ollama) |

System packages (installed in Dockerfile, not pip):
- `tesseract-ocr`, `tesseract-ocr-bul` — OCR engine + Bulgarian language data
- `poppler-utils` — `pdftotext` / `pdftoppm` used by `pdf2image`

---

## 3. Data Model

No persistence (stateless service). All types are Pydantic v2 models defined in
`app/domain/models.py`.

### Core types

```
Invoice
  id: str                        # doc_id passed by caller
  source: "xml" | "ocr" | "csv" | "manual"
  doc_type: DocType              # INVOICE | CREDIT_NOTE | FISCAL_RECEIPT | …
  direction: Direction           # SALE | PURCHASE | UNKNOWN
  number: str | None
  date: str | None               # ISO YYYY-MM-DD
  currency: str                  # BGN | EUR
  supplier: Party | None
  recipient: Party | None
  line_items: list[LineItem]
  tax_lines: list[TaxLine]
  net_amount: Decimal | None
  vat_amount: Decimal | None
  total_amount: Decimal | None
  field_confidence: dict[str, float]
  vision_used: bool
  extra: dict                    # IBAN, MRN, fiscal device #, …
  company_key: str | None
  company_name: str | None

Party
  name: str | None
  vat: str | None                # BG + 9-10 digits, or EU format
  eik: str | None                # BULSTAT 9 or 13 digits (validated)
  address: str | None
  source: "regex" | "vision" | "llm"

LineItem
  description: str | None
  quantity: Decimal | None
  unit: str | None
  unit_price: Decimal | None
  amount: Decimal | None
  code: str | None

TaxLine
  rate: Decimal                  # 0.0, 0.09, or 0.20 (Bulgarian VAT)
  base: Decimal | None
  amount: Decimal | None
```

---

## 4. API Endpoints

Defined in `app/ingest_service.py`, mounted at root.

### Authentication

Optional shared secret. Set `INGEST_SERVICE_TOKEN` env var; pass as:
```
x-ingest-token: <token>
```
If the env var is unset, the endpoints are open (development mode).

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | Open | OCR availability + version. Returns `{"status": "ok", "version": "…", "ocr": {…}}` |
| `POST` | `/documents/extract-pdf` | Token | Upload PDF → structured `Invoice` |
| `POST` | `/documents/extract-image` | Token | Upload image (JPEG/PNG/TIFF) → structured `Invoice` |

#### `POST /documents/extract-pdf`

**Form fields:**
- `file` (required) — PDF binary upload
- `perspective` (default `"auto"`) — `"sale"` or `"purchase"` hint to bias direction detection
- `vision` (default `true`) — whether to attempt Ollama vision fallback on low-confidence pages

**Response `200`:**
```json
{
  "invoice": {
    "id": "invoice-2024-01",
    "source": "ocr",
    "doc_type": "INVOICE",
    "direction": "PURCHASE",
    "number": "00001234",
    "date": "2024-03-15",
    "currency": "BGN",
    "supplier": {"name": "ДОСТАВЧИК ООД", "vat": "BG123456789", "eik": "123456789"},
    "recipient": null,
    "line_items": [...],
    "tax_lines": [{"rate": "0.20", "base": "1000.00", "amount": "200.00"}],
    "net_amount": "1000.00",
    "vat_amount": "200.00",
    "total_amount": "1200.00",
    "field_confidence": {"number": 0.95, "supplier.name": 0.88},
    "vision_used": false,
    "extra": {},
    "company_key": "BG123456789",
    "company_name": null
  }
}
```

**Error `422`:** OCR pipeline failure  
**Error `503`:** Tesseract not installed on host

#### `POST /documents/extract-image`

Same fields and response shape as `extract-pdf`, but accepts a single image file
(JPEG, PNG, TIFF, BMP) instead of a PDF.

---

## 5. Data Flow

```
Client → POST /documents/extract-pdf (multipart: file, perspective, vision)
  │
  ├─ require_ingest_token()   ← 401 if token wrong
  ├─ _ensure_ocr()            ← 503 if tesseract absent
  │
  ├─ extract_from_pdf_bytes(bytes, doc_id, source="ocr", perspective, use_vision)
  │    │
  │    ├─ pdf2image: PDF → list[PIL.Image]
  │    ├─ for each page:
  │    │    ├─ preprocess (OpenCV deskew/denoise/threshold)
  │    │    └─ pytesseract → text + per-word confidence
  │    ├─ OcrResult(text, low_conf_tokens, mean_conf)
  │    │
  │    ├─ invoice_extractor.extract(text)  ← regex (Bulgarian locale)
  │    │    ├─ doc_type detection (header keywords, weighted signals)
  │    │    ├─ direction detection (sale/purchase)
  │    │    ├─ EIK/VAT patterns → Party objects
  │    │    ├─ invoice number regex
  │    │    ├─ date (normalise to ISO, handle DD.MM.YYYY / DD/MM/YYYY)
  │    │    ├─ amounts (EU decimal comma → Decimal, handle "16 143,38")
  │    │    └─ reverse-charge detection (чл. 117/163 ЗДДС)
  │    │
  │    ├─ [if mean_conf < OCR_VISION_CONF_MIN and use_vision and model configured]
  │    │    └─ vision_extract(pages) → fill uncertain fields only
  │    │         └─ HTTP POST to Ollama (qwen2.5vl) with page images + BG prompt
  │    │
  │    ├─ company.derive_key(invoice) → company_key
  │    └─ [optional] company_lookup(eik) → canonical legal name
  │
  └─ return {"invoice": Invoice}
```

---

## 6. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | HTTP listen port |
| `INGEST_SERVICE_TOKEN` | _(unset)_ | Shared secret for `x-ingest-token`. Open if unset. |
| `OCR_LANGUAGES` | `bul+eng` | Tesseract language pack(s) |
| `OCR_DPI` | `300` | PDF rasterisation DPI |
| `OCR_PREFER_EMBEDDED_TEXT` | `true` | Use PDF text layer when available |
| `OCR_PREPROCESS` | `true` | Enable OpenCV pre-processing |
| `OCR_DESKEW` | `true` | Deskew tilted scans |
| `OCR_DENOISE` | `false` | Median-blur denoising |
| `OCR_THRESHOLD` | `otsu` | Binarisation: `otsu` / `adaptive` / `none` |
| `OCR_WORD_CONF_MIN` | `0.6` | Flag tokens below this confidence |
| `OCR_VISION_FALLBACK` | `true` | Enable vision model fallback |
| `OCR_VISION_MODEL` | `ollama/qwen2.5vl:7b` | Vision model identifier |
| `OCR_VISION_API_BASE` | `http://ollama:11434` | Ollama base URL |
| `OCR_VISION_CONF_MIN` | `0.75` | Trigger vision if page OCR conf below this |
| `COMPANY_LOOKUP_ENABLED` | `true` | Scrape web.company.guru for canonical names |
| `COMPANY_LOOKUP_TIMEOUT` | `5` | HTTP timeout for register lookups (seconds) |

---

## 7. Build & Run

```bash
# Local (requires tesseract-ocr + tesseract-ocr-bul installed)
pip install -r requirements.txt
python app.py

# Docker (with Ollama vision)
docker compose up -d --build
```

Health check: `GET /health` → `{"status": "ok", "ocr": {"available": true, …}}`
