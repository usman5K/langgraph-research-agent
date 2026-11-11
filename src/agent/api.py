"""FastAPI service: POST /ask (answer + trace) and POST /ask/stream (Server-Sent Events)."""

import json
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.graph import default_graph
from agent.trace import arun_steps


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    steps: list[dict]


@lru_cache
def _cached_graph():
    return default_graph()


def get_graph():
    try:
        return _cached_graph()
    except RuntimeError as exc:  # e.g. missing OPENAI_API_KEY
        raise HTTPException(status_code=503, detail=str(exc)) from exc


GraphDep = Annotated[Any, Depends(get_graph)]

app = FastAPI(title="LangGraph Research Agent", version="0.1.0")
INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Single-page web UI (vanilla JS) that renders the SSE stream live."""
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, graph: GraphDep) -> AskResponse:
    steps = [step async for step in arun_steps(graph, req.question)]
    answer = next((s["content"] for s in reversed(steps) if s["type"] == "answer"), "")
    return AskResponse(answer=answer, steps=steps)


@app.post("/ask/stream")
async def ask_stream(req: AskRequest, graph: GraphDep) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        try:
            async for step in arun_steps(graph, req.question):
                yield f"event: {step['type']}\ndata: {json.dumps(step, default=str)}\n\n"
        except Exception as exc:  # surface failures to the client instead of a dead stream
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
