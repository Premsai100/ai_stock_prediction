import sys
from pathlib import Path
import json
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tavily import TavilyClient
from config.settings import TAVILY_API

client = TavilyClient(TAVILY_API)


def search_query(query: str) -> tuple[str, str]:
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
    )
    combined = "\n\n---\n\n".join(
        res["content"]
        for res in response["results"]
        if res.get("content") and len(res["content"]) > 200
    )
    return query, combined


def scrape_stock_news(queries: list[str]) -> dict:

    summary = {}

    for query in queries:
        query, content = search_query(query)
        if content:
            summary[query] = content
            print(f"{query[:50]} ({len(content)} chars)")
        else:
            print(f"{query[:50]} — no content")

    return summary