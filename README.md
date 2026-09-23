# Invoice Reader — free OCR + local vision microservice

Self-contained service that reads Bulgarian expense invoices (PDF or photo) and returns
structured JSON. Deterministic OCR (Tesseract + pdftotext) fused with a **free local vision
model** (`qwen2.5vl:7b` via Ollama) + arithmetic reconciliation. **No paid API.**

It's the "Наш" reader behind the toggle in the service-gumialex CRM
(*Настройки → Четец на фактури за разходи*). The CRM's `parse-expense-invoice` edge function
calls this service; if it's ever unreachable it automatically falls back to Claude, so
invoicing never breaks.

- **Public domain:** `https://invoice-bot.unparalleled.bg` → the `reader` container, port `8000`
- **Auth:** every extract request must carry `x-ingest-token: <INGEST_SERVICE_TOKEN>`. Only `/health` is open.

---

## Requirements
- Docker + Docker Compose
- **~8 GB RAM** (the `qwen2.5vl:7b` model is ~6 GB in memory). 16 GB comfortable. CPU works (~15–45 s/invoice); an NVIDIA GPU makes it ~2–5 s (uncomment the `deploy` block in `docker-compose.yml`).
- ~8 GB disk for the model (persisted in the `ollama_models` volume).

## Deploy
The host/bot must provide one secret in the environment: `INGEST_SERVICE_TOKEN` (a long random string).

```bash
INGEST_SERVICE_TOKEN='<long-random-secret>' docker compose up -d --build
```

On first start the model is pulled automatically (a few minutes, one time) into a persistent
volume and kept warm. Point the domain `invoice-bot.unparalleled.bg` at the `reader` service's
port `8000` (TLS terminated by your bot/ingress).

Check it's up:
```bash
curl https://invoice-bot.unparalleled.bg/health
# {"status":"ok","ocr":{"available":true,"vision":true,...}}
```

## Environment variables
| Var | Set by | Purpose |
|-----|--------|---------|
| `INGEST_SERVICE_TOKEN` | **you (required)** | shared secret; must match the gumialex Supabase secret |
| `OCR_VISION_MODEL` | preset | `ollama/qwen2.5vl:7b` |
| `OCR_VISION_API_BASE` | preset | `http://ollama:11434` (internal) |
| `OCR_VISION_FALLBACK` | preset | `true` |
| `OLLAMA_KEEP_ALIVE` | preset | `-1` — keep model warm |
| `LLM_TIMEOUT` | preset | `180` seconds |

## Endpoints
| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | none | liveness + OCR/vision status |
| POST | `/documents/extract-image` | `x-ingest-token` | read a photo/scan (multipart `file`, `perspective=purchase`, `vision=true`) |
| POST | `/documents/extract-pdf` | `x-ingest-token` | read a PDF invoice |

Returns `{"invoice": { ... }}` with number, dates, both parties' EIK/VAT, net/vat/total,
currency, and line items.

---

## Wire it to the CRM (service-gumialex)
In **Supabase → Project Settings → Edge Functions → Secrets** (production project
`nuhhxicnqfxbgedlowjk`):

```
INGEST_SERVICE_URL   = https://invoice-bot.unparalleled.bg
INGEST_SERVICE_TOKEN = <the same long-random-secret used above>
```

Then in the CRM: **Настройки → Четец на фактури за разходи → switch to „Наш"**.
That's it — expense-invoice reads now use this service. Keep `ANTHROPIC_API_KEY` set in
Supabase so the automatic Claude fallback still works if this service is down.

## Security notes
- Only `/health` is public; extraction requires the `x-ingest-token`. Keep the token secret and long.
- Ollama is **not** exposed publicly (internal compose network only).
- The service holds no database credentials and writes nothing — it only reads the uploaded file and returns JSON.
- Optional hardening: put Cloudflare Access / an IP allowlist in front of the domain.

## Operations
- **First read after a restart** may take ~40–60 s (model load); `OLLAMA_KEEP_ALIVE=-1` keeps it warm afterward, so subsequent reads are ~15–45 s (CPU) or ~2–5 s (GPU).
- **Update:** `git pull && docker compose up -d --build`.
- **Logs:** `docker compose logs -f reader` / `... logs -f ollama`.
- **Model persists** in the `ollama_models` volume — not re-downloaded on restart.
