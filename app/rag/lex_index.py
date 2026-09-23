"""Startup build and periodic refresh of the lex laws index.

On startup we make sure the index exists, building it once if it is missing. A daemon
thread then rebuilds it on an interval so amendments to the legislation get picked up.
Everything is best-effort and off the request path: if lex's dependencies or the network
are unavailable it logs and the laws RAG simply stays empty. A file lock keeps a single
process building at a time when the API runs with more than one worker.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_LEX_DIR = Path(__file__).resolve().parents[2] / "lex"
_STORAGE = _LEX_DIR / "storage"
_LOCK = _STORAGE / ".ingest.lock"
_STAMP = _STORAGE / ".last_ingest"
_LOCK_TTL = 6 * 3600  # a lock older than this is treated as stale (a build that died)


def _index_exists() -> bool:
    chroma = _STORAGE / "chroma"
    return (_STORAGE / "bm25.pkl").exists() and chroma.is_dir() and any(chroma.iterdir())


def _seconds_since_build() -> float:
    try:
        return time.time() - float(_STAMP.read_text().strip())
    except (OSError, ValueError):
        return float("inf")


def _stamp() -> None:
    try:
        _STAMP.write_text(str(int(time.time())))
    except OSError:
        pass


def _acquire_lock() -> bool:
    _STORAGE.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - _LOCK.stat().st_mtime > _LOCK_TTL:
                _LOCK.unlink(missing_ok=True)
                return _acquire_lock()
        except FileNotFoundError:
            return _acquire_lock()
        return False


def _build_index() -> None:
    """Scrape, embed and index the legislation. Forces CPU, because the bge models
    exhaust the Apple MPS GPU pool (no-op on Linux/CUDA)."""
    if str(_LEX_DIR) not in sys.path:
        sys.path.insert(0, str(_LEX_DIR))
    import torch

    torch.backends.mps.is_available = lambda: False
    # Cap CPU threads so embedding ~20k chunks doesn't saturate every core and starve the
    # single uvicorn worker — otherwise the app stops responding (Cloudflare 5xx) for the
    # whole build. Leave at least one core for the web loop; LEX_BUILD_THREADS overrides.
    try:
        import os

        env = os.environ.get("LEX_BUILD_THREADS")
        threads = int(env) if env else max(1, (os.cpu_count() or 2) // 2)
        torch.set_num_threads(threads)
        logger.info("lex build limited to %d CPU threads", threads)
    except Exception:  # pragma: no cover - torch/env quirks
        pass

    from rag.ingest import IngestPipeline

    n = IngestPipeline().run(reset=True)
    _stamp()
    logger.info("lex index built: %d chunks", n)

    from app.rag import laws

    laws.reset_pipeline()  # reopen the retriever against the fresh stores


def _build_guarded(reason: str) -> None:
    if not _acquire_lock():
        logger.info("lex index build skipped: another worker holds the lock")
        return
    try:
        logger.info("lex index build starting (%s)", reason)
        _build_index()
    except Exception as exc:
        logger.warning("lex index build failed (%s): %s", reason, exc)
    finally:
        _LOCK.unlink(missing_ok=True)


def _maintenance(interval_hours: float) -> None:
    if not _index_exists():
        _build_guarded("missing on startup")
    elif _seconds_since_build() == float("inf"):
        _stamp()  # pre-existing index (built manually or shipped): mark it fresh

    interval = max(1.0, interval_hours) * 3600
    while True:
        time.sleep(interval)
        if _seconds_since_build() >= interval:
            _build_guarded("scheduled refresh")


def start() -> None:
    """Kick off index maintenance in a background daemon thread."""
    from app.core import get_settings

    settings = get_settings()
    if not settings.lex_auto_index:
        return
    threading.Thread(
        target=_maintenance,
        args=(settings.lex_refresh_interval_hours,),
        daemon=True,
        name="lex-index",
    ).start()
    logger.info(
        "lex index maintenance started (refresh every %sh)",
        settings.lex_refresh_interval_hours,
    )


def is_building() -> bool:
    """A rebuild is in progress (a fresh lock is held)."""
    try:
        return _LOCK.exists() and (time.time() - _LOCK.stat().st_mtime) <= _LOCK_TTL
    except OSError:
        return False


def index_status() -> dict:
    """Snapshot for the admin UI: whether the index exists, is building, and its age."""
    secs = _seconds_since_build()
    return {
        "exists": _index_exists(),
        "building": is_building(),
        "seconds_since_build": None if secs == float("inf") else int(secs),
    }


def trigger_rebuild() -> dict:
    """Start a manual rebuild in a background daemon thread (the same lock-guarded build
    the scheduler uses). No-op if a build is already running. Returns immediately."""
    if is_building():
        return {"started": False, "building": True}
    threading.Thread(
        target=_build_guarded, args=("manual reload",), daemon=True, name="lex-reindex"
    ).start()
    return {"started": True, "building": True}
