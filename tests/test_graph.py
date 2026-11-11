from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from agent.graph import build_graph, route_after_agent
from agent.tools import calculator
from agent.trace import run_steps
from tests.conftest import ScriptedChatModel, tool_call


def run(model, max_steps=6, question="q"):
    graph = build_graph(model, [calculator], max_steps=max_steps)
    return graph.invoke({"messages": [HumanMessage(question)], "steps": 0})


def test_route_goes_to_tools_when_llm_calls_a_tool():
    state = {"messages": [tool_call("calculator", {"expression": "1+1"})], "steps": 1}
    assert route_after_agent(state) == "tools"


def test_route_ends_when_llm_answers():
    assert route_after_agent({"messages": [AIMessage("done")], "steps": 1}) == END


def test_tool_call_runs_tool_then_answers():
    model = ScriptedChatModel(
        responses=[
            tool_call("calculator", {"expression": "(42 - 30) / 30 * 100"}),
            AIMessage("40% [1]"),
        ]
    )
    result = run(model)
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and "= 40.0" in tool_msgs[0].content
    assert result["messages"][-1].content == "40% [1]"
    assert result["steps"] == 2
    # the tool result is fed back to the LLM on its second turn
    assert isinstance(model.calls[1][-1], ToolMessage)


def test_no_tool_call_goes_straight_to_end():
    model = ScriptedChatModel(responses=[AIMessage("Paris.")])
    result = run(model)
    assert result["steps"] == 1
    assert [type(m) for m in result["messages"]] == [HumanMessage, AIMessage]


def test_max_steps_guard_stops_a_looping_model():
    model = ScriptedChatModel(responses=[tool_call("calculator", {"expression": "1+1"})])
    result = run(model, max_steps=3)
    assert len(model.calls) == 3  # the LLM was invoked exactly max_steps times
    assert result["steps"] == 3
    assert "Stopped after 3 steps" in result["messages"][-1].content


def test_trace_lists_call_result_and_answer():
    model = ScriptedChatModel(
        responses=[tool_call("calculator", {"expression": "2*21"}), AIMessage("42")]
    )
    steps = list(run_steps(build_graph(model, [calculator]), "q"))
    assert [s["type"] for s in steps] == ["tool_call", "tool_result", "answer"]
    assert steps[0] == {"type": "tool_call", "tool": "calculator", "args": {"expression": "2*21"}}
