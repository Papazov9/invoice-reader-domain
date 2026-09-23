"""Root entry point for the invoice-reader microservice.

Wraps app.ingest_service so the stack runner can find `app` at the project root
and start it with `python app.py` or directly via uvicorn.

Run directly:   python app.py
Via uvicorn:    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os

import uvicorn

from app.ingest_service import app  # noqa: F401  (re-export for uvicorn discovery)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
