"""Public entry point: question -> AgentAnswer (routes invoices vs laws)."""
from __future__ import annotations

import uuid
from typing import Callable

from sqlalchemy.orm import Session

from app.rag.answer import AgentAnswer, assemble
from app.rag.llm import litellm_complete
from app.rag.loop import run_tool_loop
from app.rag.prompts import build_messages
from app.rag.tools import TOOL_SCHEMAS, make_dispatch


def run(
    db: Session,
    tenant_id: uuid.UUID,
    message: str,
    history: list[dict],
    *,
    complete: Callable | None = None,
    model: str = "",
    max_steps: int = 5,
) -> AgentAnswer:
    """Route `message` through the tool-calling loop (invoice tools + query_law)
    and assemble cards. `complete` defaults to the LiteLLM adapter; tests inject a
    fake. `model` is surfaced in the response for transparency."""
    complete = complete or litellm_complete
    messages = build_messages(message, history)
    dispatch = make_dispatch(db, tenant_id)
    result = run_tool_loop(messages, TOOL_SCHEMAS, dispatch, complete, max_steps=max_steps)
    return assemble(result, model=model, query=message)
