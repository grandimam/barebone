from __future__ import annotations

import json
import re

from typing import Any
from typing import Literal

from .base import Tool
from .base import Param


class WebFetch(Tool):
    """Fetch a web page and convert to readable markdown/text."""

    url: str = Param(description="URL to fetch")
    extract: str | None = Param(default=None, description="Optional: what to look for")
    timeout: int = Param(default=30, description="Request timeout in seconds")

    async def execute(self) -> str:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required: pip install httpx")

        try:
            from markdownify import markdownify
        except ImportError:
            markdownify = None

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; LoopflowBot/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout) as client:
            response = await client.get(self.url, headers=headers)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if "application/json" in content_type:
            try:
                data = response.json()
                return json.dumps(data, indent=2)[:50000]
            except Exception:
                return response.text[:50000]

        if "text/plain" in content_type:
            return response.text[:50000]

        html = response.text

        if markdownify:
            text = markdownify(html, heading_style="ATX", strip=["script", "style"])
        else:
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

        text = text[:50000]

        if self.extract:
            return f"Extracted from {self.url} (looking for: {self.extract}):\n\n{text}"

        return f"Content from {self.url}:\n\n{text}"


class WebSearch(Tool):
    """Search the web using DuckDuckGo. Extend this class to add custom providers."""

    query: str = Param(description="Search query")
    num_results: int = Param(default=5, description="Number of results")

    async def execute(self) -> str:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError("duckduckgo-search is required: pip install duckduckgo-search")

        with DDGS() as ddgs:
            results = list(ddgs.text(self.query, max_results=self.num_results))

        return self.format_results(results)

    def format_results(self, results: list[dict]) -> str:
        """Format search results. Override this to customize output."""
        if not results:
            return f"No results found for: {self.query}"

        output = [f"Search results for: {self.query}\n"]
        for i, r in enumerate(results, 1):
            output.append(f"{i}. {r.get('title', 'No title')}")
            output.append(f"   URL: {r.get('href', 'No URL')}")
            output.append(f"   {r.get('body', 'No description')}\n")

        return "\n".join(output)


class HttpRequest(Tool):
    """Make HTTP requests to APIs. Supports GET, POST, PUT, PATCH, DELETE."""

    url: str = Param(description="URL to request")
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = Param(
        default="GET", description="HTTP method"
    )
    headers: dict[str, str] | None = Param(default=None, description="Request headers")
    body: dict[str, Any] | None = Param(default=None, description="Request body (JSON)")
    timeout: int = Param(default=30, description="Timeout in seconds")

    async def execute(self) -> str:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required: pip install httpx")

        request_headers = self.headers or {}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if isinstance(self.body, dict):
                response = await client.request(
                    self.method, self.url, headers=request_headers, json=self.body
                )
            elif isinstance(self.body, str):
                response = await client.request(
                    self.method, self.url, headers=request_headers, content=self.body
                )
            else:
                response = await client.request(
                    self.method, self.url, headers=request_headers
                )

        output = [
            f"HTTP {response.status_code} {response.reason_phrase}",
            f"URL: {response.url}",
            "",
        ]

        for header in ["content-type", "content-length", "x-ratelimit-remaining"]:
            if header in response.headers:
                output.append(f"{header}: {response.headers[header]}")

        output.append("")

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                data = response.json()
                output.append(json.dumps(data, indent=2)[:30000])
            except Exception:
                output.append(response.text[:30000])
        else:
            output.append(response.text[:30000])

        return "\n".join(output)


__all__ = [
    "WebFetch",
    "WebSearch",
    "HttpRequest",
]
