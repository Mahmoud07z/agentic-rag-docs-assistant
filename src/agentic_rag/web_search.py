"""Optional web search adapter used when local documents are insufficient."""

from __future__ import annotations

from dataclasses import dataclass

from tavily import TavilyClient


@dataclass(frozen=True, slots=True)
class WebPassage:
    title: str
    url: str
    content: str


class WebSearch:
    def __init__(self, api_key: str):
        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, *, limit: int = 5) -> list[WebPassage]:
        response = self._client.search(query, max_results=limit, search_depth="basic")
        passages = []
        for result in response.get("results", []):
            content = (result.get("content") or "").strip()
            if content:
                passages.append(WebPassage(
                    title=result.get("title") or result.get("url") or "Web result",
                    url=result.get("url") or "",
                    content=content,
                ))
        return passages
