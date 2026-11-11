"""Web search: Tavily when TAVILY_API_KEY is set, otherwise DuckDuckGo via `ddgs` (no key)."""

from langchain_core.tools import tool

from agent.config import get_settings

MAX_RESULTS = 5


def _tavily(query: str, api_key: str) -> list[dict]:
    from tavily import TavilyClient

    response = TavilyClient(api_key=api_key).search(query, max_results=MAX_RESULTS)
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in response.get("results", [])
    ]


def _duckduckgo(query: str) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        hits = ddgs.text(query, max_results=MAX_RESULTS) or []
    return [
        {"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")}
        for h in hits
    ]


@tool
def web_search(query: str) -> str:
    """Search the public web for current facts, news and statistics.

    Returns numbered results with title, URL and snippet. Cite the URLs you use.
    """
    key = get_settings().tavily_api_key
    try:
        results = _tavily(query, key.get_secret_value()) if key else _duckduckgo(query)
    except Exception as exc:  # network/provider errors go back to the LLM, not up the stack
        return f"Error: web search failed ({exc.__class__.__name__}: {exc})"
    if not results:
        return "No results."
    return "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet'][:400]}"
        for i, r in enumerate(results, 1)
    )
