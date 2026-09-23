# Lean image for the invoice-reading microservice (app.ingest_service) — no RAG/torch, no DB.
FROM python:3.11-slim AS builder
WORKDIR /build
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

# Runtime system deps:
#   tesseract-ocr + bul/eng : OCR engine with Bulgarian + English language data
#   poppler-utils           : pdf2image / pdftotext for PDF invoices
#   libglib2.0-0            : shared lib opencv-python-headless dlopens at import
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-bul \
        tesseract-ocr-eng \
        poppler-utils \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OCR_VISION_FALLBACK=true

EXPOSE 8000

# Healthcheck hits the open /health (no token needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.ingest_service:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
