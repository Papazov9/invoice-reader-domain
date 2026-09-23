"""Chat RAG package: the tool-calling orchestrator (`run`) that routes between the
invoice tools (invoice_rag) and the Bulgarian laws RAG (LawsRetriever), plus the
shared chunk types and the no-model echo fallback."""
from app.rag.base import Citation, RetrievedChunk, Retriever
from app.rag.laws import LawsRetriever
from app.rag.llm import EchoLLMClient, LLMClient, litellm_complete
from app.rag.run import run

__all__ = [
    "RetrievedChunk",
    "Citation",
    "Retriever",
    "LawsRetriever",
    "EchoLLMClient",
    "LLMClient",
    "litellm_complete",
    "run",
]
