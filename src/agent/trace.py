"""Turn LangGraph stream updates into a flat, JSON-friendly step trace."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

SNIPPET = 300


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def steps_from_update(update: dict) -> Iterator[dict]:
    """Yield steps for one `stream_mode="updates"` chunk: {node_name: state_delta}."""
    for node, delta in update.items():
        for msg in (delta or {}).get("messages", []):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                if _text(msg.content).strip():
                    yield {"type": "thought", "content": _text(msg.content)}
                for call in msg.tool_calls:
                    yield {"type": "tool_call", "tool": call["name"], "args": call["args"]}
            elif isinstance(msg, ToolMessage):
                text = _text(msg.content)
                yield {
                    "type": "tool_result",
                    "tool": msg.name,
                    "content": text[:SNIPPET] + ("..." if len(text) > SNIPPET else ""),
                }
            elif isinstance(msg, AIMessage):
                yield {"type": "answer", "node": node, "content": _text(msg.content)}


def _inputs(question: str) -> tuple[dict, dict]:
    return {"messages": [HumanMessage(question)], "steps": 0}, {"recursion_limit": 60}


def run_steps(graph, question: str) -> Iterator[dict]:
    inputs, config = _inputs(question)
    for update in graph.stream(inputs, config, stream_mode="updates"):
        yield from steps_from_update(update)


async def arun_steps(graph, question: str) -> AsyncIterator[dict]:
    inputs, config = _inputs(question)
    async for update in graph.astream(inputs, config, stream_mode="updates"):
        for step in steps_from_update(update):
            yield step
