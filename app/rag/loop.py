"""Generic, domain-agnostic tool-calling loop.

Drives a model (via the injected `complete` callable) until it returns prose or
hits max_steps. Knows nothing about invoices, laws, tenants, or cards — it just
runs whatever tools the model asks for.

`complete(messages, tool_schemas) -> {"content": str|None, "tool_calls": list|None}`
is the only model dependency. tool_calls follow the OpenAI/LiteLLM shape:
  {"id": ..., "type": "function", "function": {"name": ..., "arguments": <json str>}}
`dispatch(name, args) -> Any` runs a tool and returns a JSON-serializable result.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    name: str
    args: dict
    result: Any


@dataclass
class LoopResult:
    final_text: str
    calls: list[ToolCall] = field(default_factory=list)


def run_tool_loop(
    messages: list[dict],
    tool_schemas: list[dict],
    dispatch: Callable[[str, dict], Any],
    complete: Callable[[list[dict], list[dict]], dict],
    *,
    max_steps: int = 5,
) -> LoopResult:
    messages = list(messages)  # work on a copy — never mutate the caller's list
    calls: list[ToolCall] = []
    for _ in range(max_steps):
        msg = complete(messages, tool_schemas)
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return LoopResult(final_text=msg.get("content") or "", calls=calls)
        # echo the assistant turn (with its tool_calls) back into the transcript
        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc["function"]
            args = json.loads(fn.get("arguments") or "{}")
            result = dispatch(fn["name"], args)
            calls.append(ToolCall(name=fn["name"], args=args, result=result))
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, default=str),
            })
    return LoopResult(final_text="", calls=calls)  # hit the cap
