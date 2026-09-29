# Invoice Reader — deterministic-first, with a Claude Haiku 4.5 safety net

Self-contained service that reads Bulgarian expense invoices (PDF or photo) and returns
structured JSON.

- **Digital PDFs** are read **exactly and for free** by a deterministic pipeline
  (`pdftotext` embedded text → locale-aware field extraction → arithmetic reconciliation).
  No model runs. Instant.
- **Photos / scans** go through Tesseract OCR (+ OpenCV deskew/denoise/threshold) into the
  same deterministic extractor. When that read comes out **incomplete or the amounts don't
  reconcile** (missing vendor, items that don't sum to the tax base, unreadable totals), the
  page is sent to **Claude Haiku 4.5** as a fallback — a few cents per hard photo. A clean
  read never calls Claude.

No local vision model, so **no GPU, no large-RAM requirement, and no swap** — the container
is a few hundred MB of RAM.

It's the "Наш" reader behind the toggle in the service-gumialex CRM
(*Настройки → Четец на фактури за разходи*). The CRM's `parse-expense-invoice` edge function
calls this service; if it's ever unreachable it also falls back to Claude, so invoicing never
breaks.

- **Public domain:** `https://invoice-bot.unparalleled.bg` → the `reader` container, port `8000`
- **Auth:** every extract request must carry `x-ingest-token: <INGEST_SERVICE_TOKEN>`. Only `/health` is open.

---

## Requirements
- Docker + Docker Compose
- **~1 GB RAM** is plenty (no local model). CPU only.
- No model download, no GPU, no large volume.

## Deploy
The host/bot must provide **two** secrets in the environment (never committed):

- `INGEST_SERVICE_TOKEN` — a long random string; must match the gumialex Supabase secret.
- `ANTHROPIC_API_KEY` — the Anthropic key for the Claude Haiku 4.5 fallback.

```bash
# both values present in the environment / project .env
docker compose up -d --build
```

Point the domain `invoice-bot.unparalleled.bg` at the `reader` service's port `8000`
(TLS terminated by your bot/ingress).

Check it's up:
```bash
curl https://invoice-bot.unparalleled.bg/health
# {"status":"ok","ocr":{"available":true,"pdftotext":true,"ocr":true,"vision":true,...}}
```

## Environment variables
| Var | Set by | Purpose |
|-----|--------|---------|
| `INGEST_SERVICE_TOKEN` | **you (required)** | shared secret; must match the gumialex Supabase secret |
| `ANTHROPIC_API_KEY` | **you (required)** | Anthropic key for the Claude Haiku 4.5 fallback |
| `OCR_VISION_MODEL` | preset | `anthropic/claude-haiku-4-5-20251001` |
| `OCR_VISION_FALLBACK` | preset | `true` — allow the fallback on hard reads |
| `LLM_ASSIST_ENABLED` | preset | `false` — the vision call already reads every field |
| `COMPANY_LOOKUP_ENABLED` | preset | `false` — no external register scrape |
| `LLM_TIMEOUT` | preset | `60` seconds |

## Endpoints
| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | none | liveness + OCR/vision status |
| POST | `/documents/extract-image` | `x-ingest-token` | read a photo/scan (multipart `file`, `perspective=purchase`, `vision=true`) |
| POST | `/documents/extract-pdf` | `x-ingest-token` | read a PDF invoice |

`vision=false` skips the Claude fallback entirely (fully free/offline — e.g. bulk uploads or
when you want to guarantee zero API cost). Returns `{"invoice": { ... }}` with number, dates,
both parties' EIK/VAT, net/vat/total, currency, and line items.

---

## Wire it to the CRM (service-gumialex)
In **Supabase → Project Settings → Edge Functions → Secrets** (production project
`nuhhxicnqfxbgedlowjk`):

```
INGEST_SERVICE_URL   = https://invoice-bot.unparalleled.bg
INGEST_SERVICE_TOKEN = <the same long-random-secret used above>
```

Then in the CRM: **Настройки → Четец на фактури за разходи → switch to „Наш"**.
Keep `ANTHROPIC_API_KEY` set in Supabase too, so the edge function's own Claude fallback still
works if this service is ever down.

## Security notes
- Only `/health` is public; extraction requires the `x-ingest-token`. Keep the token secret and long.
- The service holds no database credentials and writes nothing — it only reads the uploaded file and returns JSON.
- The Anthropic key lives only in the server `.env` / environment — never in this repo.
- Optional hardening: put Cloudflare Access / an IP allowlist in front of the domain.

## Cost model
- **Digital PDF:** €0 — deterministic, no API call.
- **Clean photo whose read reconciles:** €0 — deterministic, no API call.
- **Hard photo (no vendor / items don't sum / unreadable totals):** one Claude Haiku 4.5 call,
  ~a fraction of a cent. Amounts Claude returns are still cross-checked against the arithmetic
  (net+VAT=total, rows sum to net) before they're accepted.

## Operations
- **No warm-up, no model load** — reads are fast from the first request.
- **Update:** `git pull && docker compose up -d --build`.
- **Logs:** `docker compose logs -f reader`.
