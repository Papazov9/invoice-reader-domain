"""Combined tool registry for the orchestrator.

Schemas = the invoice tools (imported from invoice_rag) + query_law (the laws RAG).
dispatch delegates invoice tool names to invoice_rag's tenant-bound dispatch and
handles query_law via the LawsRetriever (lex). This is the only place that wires
the two RAGs together — invoice_rag stays free of the laws RAG.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.rag.laws import LawsRetriever
from invoice_rag.agent.tools import INVOICE_TOOL_SCHEMAS, make_invoice_dispatch

_QUERY_LAW_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_law",
        "description": (
            "Answer questions about Bulgarian tax/accounting LAW — what the law says or requires "
            "(ЗДДС, ЗКПО, ДОПК, rates, rules, deadlines). Retrieves the relevant legal articles. "
            "Use for legal/regulatory questions, NOT for the company's own invoice data. A question "
            "may need this AND an invoice tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "the legal question, in the user's words"}},
            "required": ["query"],
        },
    },
}

TOOL_SCHEMAS = [*INVOICE_TOOL_SCHEMAS, _QUERY_LAW_SCHEMA]


def make_dispatch(db: Session, tenant_id: uuid.UUID):
    """dispatch(name, args) over invoices + laws, bound to this tenant session."""
    invoice_dispatch = make_invoice_dispatch(db, tenant_id)

    def dispatch(name: str, args: dict) -> Any:
        if name == "query_law":
            try:
                chunks = LawsRetriever(db).retrieve(args["query"], top_k=5)
                return [c.model_dump() for c in chunks]
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}
        return invoice_dispatch(name, args)

    return dispatch
