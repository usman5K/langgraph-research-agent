from agent.tools.calculator import calculator
from agent.tools.docs_search import search_docs
from agent.tools.web_search import web_search

ALL_TOOLS = [web_search, calculator, search_docs]

__all__ = ["ALL_TOOLS", "calculator", "search_docs", "web_search"]
