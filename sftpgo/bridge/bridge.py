#!/usr/bin/env python3
"""
feed-bridge — the "isolated proxy" between the Primex FTP drop and Supabase.

Flow:
  Primex --(plain FTP)--> SFTPGo writes file into /srv/sftpgo/data/primex
                          (the same volume this bridge sees at WATCH_DIR)
  bridge polls WATCH_DIR, and for each NEW, fully-uploaded CSV:
    1) uploads it to the Supabase Storage bucket `feed-quarantine`
    2) calls the `primex-feed-import` edge function with x-import-secret
    3) moves the local file to processed/ (ok) or failed/ (error)

It holds the Supabase service-role key; SFTPGo never sees it. The edge function
does all parsing/validation/atomic apply — this bridge only ferries the file.

Env (set in docker-compose / .env):
  SUPABASE_URL                 e.g. https://nuhhxicnqfxbgedlowjk.supabase.co
  SUPABASE_SERVICE_ROLE_KEY    prod service-role key (NEVER committed)
  PRIMEX_IMPORT_SECRET         == the PRIMEX_IMPORT_SECRET Supabase secret
  WATCH_DIR                    default /data/primex
  BUCKET                       default feed-quarantine
  IMPORT_FUNCTION              default primex-feed-import
  POLL_SECONDS                 default 30
  STABLE_SECONDS               default 20  (file must be unchanged this long)
  ALLOWED_EXT                  default .csv (comma-separated, e.g. .csv,.txt)
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path

import requests

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
IMPORT_SECRET = os.environ["PRIMEX_IMPORT_SECRET"]
WATCH_DIR = Path(os.environ.get("WATCH_DIR", "/data/primex"))
BUCKET = os.environ.get("BUCKET", "feed-quarantine")
IMPORT_FUNCTION = os.environ.get("IMPORT_FUNCTION", "primex-feed-import")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "30"))
STABLE_SECONDS = int(os.environ.get("STABLE_SECONDS", "20"))
ALLOWED_EXT = tuple(
    e.strip().lower() for e in os.environ.get("ALLOWED_EXT", ".csv").split(",") if e.strip()
)

PROCESSED_DIR = WATCH_DIR / "processed"
FAILED_DIR = WATCH_DIR / "failed"
RESERVED = {"processed", "failed"}


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def is_stable(p: Path) -> bool:
    """True once the file has stopped growing (upload finished)."""
    try:
        st = p.stat()
    except FileNotFoundError:
        return False
    if st.st_size == 0:
        return False
    return (time.time() - st.st_mtime) >= STABLE_SECONDS


def upload_to_bucket(p: Path) -> None:
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{p.name}"
    with p.open("rb") as fh:
        data = fh.read()
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
            "apikey": SERVICE_ROLE_KEY,
            "Content-Type": "text/csv",
            "x-upsert": "true",
        },
        data=data,
        timeout=120,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"storage upload {r.status_code}: {r.text[:300]}")


def trigger_import(filename: str) -> dict:
    url = f"{SUPABASE_URL}/functions/v1/{IMPORT_FUNCTION}"
    r = requests.post(
        url,
        headers={"x-import-secret": IMPORT_SECRET, "Content-Type": "application/json"},
        data=json.dumps({"path": filename}),
        timeout=300,
    )
    body = {}
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text[:500]}
    if r.status_code != 200 or body.get("ok") is False or body.get("error"):
        raise RuntimeError(f"import {r.status_code}: {json.dumps(body)[:500]}")
    return body


def move(p: Path, dest_dir: Path, note: dict | None = None) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = dest_dir / f"{stamp}_{p.name}"
    shutil.move(str(p), str(target))
    if note is not None:
        target.with_suffix(target.suffix + ".json").write_text(
            json.dumps(note, ensure_ascii=False, indent=2)
        )


def process(p: Path) -> None:
    log(f"processing {p.name} ({p.stat().st_size} bytes)")
    try:
        upload_to_bucket(p)
        result = trigger_import(p.name)
        log(f"imported {p.name}: {json.dumps(result, ensure_ascii=False)[:300]}")
        move(p, PROCESSED_DIR)
    except Exception as e:
        log(f"FAILED {p.name}: {e}")
        move(p, FAILED_DIR, note={"error": str(e), "file": p.name})


def scan_once() -> None:
    if not WATCH_DIR.exists():
        return
    for p in sorted(WATCH_DIR.iterdir()):
        if p.is_dir() or p.name in RESERVED or p.name.startswith("."):
            continue
        if ALLOWED_EXT and not p.name.lower().endswith(ALLOWED_EXT):
            continue
        if is_stable(p):
            process(p)


def main() -> int:
    log(
        f"feed-bridge up. watch={WATCH_DIR} bucket={BUCKET} fn={IMPORT_FUNCTION} "
        f"poll={POLL_SECONDS}s stable={STABLE_SECONDS}s ext={ALLOWED_EXT}"
    )
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            scan_once()
        except Exception as e:
            log(f"scan error: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
