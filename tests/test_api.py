import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agent.api import app, get_graph
from agent.graph import build_graph
from agent.tools import calculator
from tests.conftest import ScriptedChatModel, tool_call


@pytest.fixture
def client():
    def fake_graph():
        model = ScriptedChatModel(
            responses=[tool_call("calculator", {"expression": "6*7"}), AIMessage("42 [1]")]
        )
        return build_graph(model, [calculator])

    app.dependency_overrides[get_graph] = fake_graph
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ask_returns_answer_and_trace(client):
    res = client.post("/ask", json={"question": "what is 6*7?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "42 [1]"
    assert [s["type"] for s in body["steps"]] == ["tool_call", "tool_result", "answer"]
    assert body["steps"][1]["content"] == "6*7 = 42"


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_stream_emits_sse_events(client):
    with client.stream("POST", "/ask/stream", json={"question": "6*7?"}) as res:
        assert res.headers["content-type"].startswith("text/event-stream")
        text = "".join(res.iter_text())
    events = [
        line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event:")
    ]
    assert events == ["tool_call", "tool_result", "answer", "done"]
    first = json.loads(text.split("data: ", 1)[1].split("\n", 1)[0])
    assert first["tool"] == "calculator"


def test_index_serves_web_ui(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    html = res.text
    assert "/ask/stream" in html and "<form" in html
    assert 'aria-live="polite"' in html and "AbortController" in html
    assert "prefers-color-scheme: dark" in html and "?demo=1" in html


def test_missing_api_key_is_a_clean_503(monkeypatch):
    monkeypatch.setattr(
        "agent.api._cached_graph", lambda: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    res = TestClient(app).post("/ask", json={"question": "hi"})
    assert res.status_code == 503
