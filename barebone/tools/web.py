import httpx
import json

from barebone.types import NullableStr

from typing import Literal
from typing import Any
from tools import tool

from markdownify import markdownify
from duckduckgo_search import DDGS as _DDGS


@tool
async def web_fetch(url: str, extract: NullableStr = None, timeout: int = 30) -> str:
    """Fetch a web page and convert to readable markdown/text."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BareboneBot/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url, headers=headers)
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
    text = markdownify(html, heading_style="ATX", strip=["script", "style"])

    text = text[:50000]

    if extract:
        return f"Extracted from {url} (looking for: {extract}):\n\n{text}"

    return f"Content from {url}:\n\n{text}"


@tool
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    if _DDGS is None:
        raise ImportError("duckduckgo-search is required: pip install duckduckgo-search")

    with _DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))

    if not results:
        return f"No results found for: {query}"

    output = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        output.append(f"{i}. {r.get('title', 'No title')}")
        output.append(f"   URL: {r.get('href', 'No URL')}")
        output.append(f"   {r.get('body', 'No description')}\n")

    return "\n".join(output)


@tool
async def http_request(
    url: str,
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 30,
) -> str:
    """Make HTTP requests to APIs. Supports GET, POST, PUT, PATCH, DELETE."""
    request_headers = headers or {}

    async with httpx.AsyncClient(timeout=timeout) as client:
        if isinstance(body, dict):
            response = await client.request(method, url, headers=request_headers, json=body)
        elif isinstance(body, str):
            response = await client.request(method, url, headers=request_headers, content=body)
        else:
            response = await client.request(method, url, headers=request_headers)

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

