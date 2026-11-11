"""The agent loop as an explicit LangGraph state machine.

    START -> agent --(tool calls?)--> tools -> agent ... --(no tool calls)--> END

The LLM ("agent" node) decides every step: call one or more tools, or answer.
A step counter in the graph state caps how many times the LLM may run.
"""

from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

SYSTEM_PROMPT = """You are a careful research assistant.
Work step by step: decide which tool to call, read the result, and repeat until you can answer.
- Use `web_search` for public, current facts. Use `search_docs` for Kestrel Analytics internals.
- Use `calculator` for every calculation; never do arithmetic in your head.
- Do not invent facts. If the tools did not give you the answer, say so.
Finish with a concise answer followed by a "Sources:" list of the URLs or doc sections
you relied on (cite them inline as [1], [2], ...)."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    """Conditional edge: go to the tools node if the LLM asked for tools, otherwise finish."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph(model: BaseChatModel, tools: Sequence[BaseTool], max_steps: int = 6):
    """Compile the agent graph. `model` is any tool-calling chat model (real or fake)."""
    llm = model.bind_tools(list(tools))

    def agent(state: AgentState) -> dict:
        steps = state.get("steps", 0)
        if steps >= max_steps:  # step guard: stop the loop instead of burning tokens forever
            return {
                "messages": [
                    AIMessage(
                        content=f"Stopped after {max_steps} steps without a final answer. "
                        "Try a narrower question or raise MAX_STEPS."
                    )
                ]
            }
        response = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response], "steps": steps + 1}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def default_graph():
    """Build the graph with the configured OpenAI model and the three built-in tools."""
    from langchain_openai import ChatOpenAI

    from agent.config import get_settings
    from agent.tools import ALL_TOOLS

    settings = get_settings()
    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add it.")
    model = ChatOpenAI(
        model=settings.model, temperature=settings.temperature, api_key=settings.openai_api_key
    )
    return build_graph(model, ALL_TOOLS, max_steps=settings.max_steps)
