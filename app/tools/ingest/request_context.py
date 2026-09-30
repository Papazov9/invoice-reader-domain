"""Request-scoped overrides for the reader.

A multi-tenant proxy (e.g. the gumialex `parse-expense-invoice` edge function) can hold each
tenant's OWN LLM API key — in Supabase, never on the reader host — and forward it per request
as the `x-llm-api-key` header. The extraction endpoints stash that key here for the duration
of the request; the Claude (litellm) calls read it, falling back to the `LLM_API_KEY` env only
when no per-request key was sent. This keeps the key out of the reader's environment and bills
each tenant's vision usage to that tenant's own key.
"""
from __future__ import annotations

from contextvars import ContextVar, Token

_request_api_key: ContextVar[str | None] = ContextVar("request_api_key", default=None)


def set_request_api_key(key: str | None) -> Token:
    """Set the per-request API key (blank/whitespace treated as absent). Returns a token to
    pass to reset_request_api_key in a finally block."""
    return _request_api_key.set((key or "").strip() or None)


def get_request_api_key() -> str | None:
    """The per-request API key for the current context, or None when none was forwarded."""
    return _request_api_key.get()


def reset_request_api_key(token: Token) -> None:
    _request_api_key.reset(token)
