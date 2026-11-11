"""CLI: python -m agent "your question"  -> prints every step of the agent loop."""

import argparse
import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.trace import run_steps

STYLES = {"thought": "cyan", "tool_call": "yellow", "tool_result": "blue", "answer": "green"}


def render(console: Console, step: dict) -> None:
    kind = step["type"]
    if kind == "tool_call":
        body, title = json.dumps(step["args"], ensure_ascii=False), f"tool call: {step['tool']}"
    elif kind == "tool_result":
        body, title = step["content"], f"result: {step['tool']}"
    elif kind == "answer":
        body, title = Markdown(step["content"]), "final answer"
    else:
        body, title = step["content"], kind
    console.print(Panel(body, title=title, title_align="left", border_style=STYLES.get(kind)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent", description=__doc__)
    parser.add_argument("question", help="What should the agent research?")
    args = parser.parse_args(argv)
    console = Console()

    from agent.graph import default_graph

    try:
        graph = default_graph()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    console.print(Panel(args.question, title="question", title_align="left"))
    for step in run_steps(graph, args.question):
        render(console, step)
    return 0
