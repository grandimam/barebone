from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Literal
from typing import get_type_hints

from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

from barebone.types import Tool


def tool(
    fn_or_name: Callable | str | None = None,
    description: str | None = None,
) -> Any:
    def decorator(fn: Callable) -> _ToolWrapper:
        name = fn_or_name if isinstance(fn_or_name, str) else fn.__name__
        desc = description or _extract_description(fn)
        return _ToolWrapper(fn, name, desc)

    if callable(fn_or_name):
        return decorator(fn_or_name)
    return decorator


def _extract_description(fn: Callable) -> str:
    if fn.__doc__:
        first_para = fn.__doc__.strip().split("\n\n")[0]
        return " ".join(line.strip() for line in first_para.split("\n"))
    return f"Execute {fn.__name__}"


def _build_pydantic_model(fn: Callable) -> type:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn) if hasattr(fn, "__annotations__") else {}

    fields = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = hints.get(name, Any)
        if param.default is inspect.Parameter.empty:
            fields[name] = (annotation, ...)
        else:
            fields[name] = (annotation, Field(default=param.default))

    return create_model(fn.__name__, **fields)


class _ToolWrapper:
    def __init__(self, fn: Callable, name: str, description: str):
        self._fn = fn
        self._name = name
        self._description = description
        self._model = _build_pydantic_model(fn)

    def to_tool(self) -> Tool:
        schema = self._model.model_json_schema()
        properties = {
            k: {key: val for key, val in v.items() if key != "title"}
            for k, v in schema.get("properties", {}).items()
        }

        return Tool(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": schema.get("required", []),
            },
            handler=self._fn,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    @property
    def name(self) -> str:
        return self._name


class QuestionOption(BaseModel):
    label: str
    description: str


class Question(BaseModel):
    question: str
    header: str
    options: list[QuestionOption]
    multiSelect: bool = False


def _parse_response(response: str, options: list[QuestionOption], multi_select: bool) -> str:
    response = response.strip()
    if not response:
        return "(no response)"

    try:
        if multi_select:
            indices = [int(s.strip()) - 1 for s in response.split(",")]
        else:
            indices = [int(response) - 1]

        labels = [options[i].label for i in indices if 0 <= i < len(options)]
        return ", ".join(labels) if labels else response
    except ValueError:
        return response


@tool
def ask_user_question(questions: list[Question]) -> dict[str, Any]:
    """Ask the user clarifying questions with multiple-choice options.

    Use this tool when you need clarification, additional information,
    or confirmation from the user before proceeding with a task.
    """
    answers: dict[str, str] = {}

    for q in questions:
        print(f"\n{q.header}: {q.question}")

        for i, opt in enumerate(q.options, 1):
            print(f"  {i}. {opt.label} - {opt.description}")

        if q.multiSelect:
            print("  (Enter numbers separated by commas, or type your own answer)")
        else:
            print("  (Enter a number, or type your own answer)")

        response = input("> ").strip()
        answers[q.question] = _parse_response(response, q.options, q.multiSelect)

    return {
        "questions": [q.model_dump() for q in questions],
        "answers": answers,
    }


@tool
def read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file from the filesystem. Returns contents with line numbers."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {file_path}")

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    selected = lines[offset : offset + limit]

    result = []
    for i, line in enumerate(selected, start=offset + 1):
        result.append(f"{i:6d}\t{line.rstrip()}")

    return "\n".join(result)


@tool
def write(file_path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    path = Path(file_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Successfully wrote {len(content)} bytes to {file_path}"


@tool
def edit(file_path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing a unique string with new content."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_text(encoding="utf-8")

    if old_string not in content:
        raise ValueError(f"String not found in file: {old_string[:50]}...")

    count = content.count(old_string)
    if count > 1:
        raise ValueError(f"String appears {count} times. Provide more context to make it unique.")

    new_content = content.replace(old_string, new_string)
    path.write_text(new_content, encoding="utf-8")

    return f"Successfully edited {file_path}"


@tool
async def bash(command: str, cwd: str | None = None, timeout: int = 120) -> str:
    """Execute a bash command and return the output."""
    working_dir = cwd or os.getcwd()

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=working_dir,
            env={**os.environ, "TERM": "dumb"},
        )

        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        output = stdout.decode("utf-8", errors="replace")

        if process.returncode != 0:
            return f"Exit code: {process.returncode}\n{output}"

        return output or "(no output)"

    except TimeoutError:
        process.kill()
        raise TimeoutError(f"Command timed out after {timeout}s")
    except Exception as e:
        raise RuntimeError(f"Command failed: {e}")


@tool
def glob(pattern: str, path: str | None = None) -> str:
    """Find files matching a glob pattern."""
    search_path = Path(path or os.getcwd()).expanduser().resolve()

    if not search_path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    matches = sorted(search_path.glob(pattern))

    if not matches:
        return f"No files matching '{pattern}'"

    return "\n".join(str(m) for m in matches[:100])


@tool
def grep(pattern: str, path: str | None = None, file_glob: str = "**/*") -> str:
    """Search for a regex pattern in files."""
    search_path = Path(path or os.getcwd()).expanduser().resolve()

    if not search_path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    regex = re.compile(pattern)
    results = []

    for file_path in search_path.glob(file_glob):
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{file_path}:{i}: {line.strip()}")

                    if len(results) >= 100:
                        results.append("... (truncated)")
                        return "\n".join(results)
        except Exception:
            continue

    if not results:
        return f"No matches for '{pattern}'"

    return "\n".join(results)


@tool
async def web_fetch(url: str, extract: str | None = None, timeout: int = 30) -> str:
    """Fetch a web page and convert to readable markdown/text."""
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required: pip install httpx")

    try:
        from markdownify import markdownify
    except ImportError:
        markdownify = None

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

    if markdownify:
        text = markdownify(html, heading_style="ATX", strip=["script", "style"])
    else:
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

    text = text[:50000]

    if extract:
        return f"Extracted from {url} (looking for: {extract}):\n\n{text}"

    return f"Content from {url}:\n\n{text}"


def _format_search_results(query: str, results: list[dict]) -> str:
    if not results:
        return f"No results found for: {query}"

    output = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        output.append(f"{i}. {r.get('title', 'No title')}")
        output.append(f"   URL: {r.get('href', 'No URL')}")
        output.append(f"   {r.get('body', 'No description')}\n")

    return "\n".join(output)


@tool
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError("duckduckgo-search is required: pip install duckduckgo-search")

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))

    return _format_search_results(query, results)


@tool
async def http_request(
    url: str,
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 30,
) -> str:
    """Make HTTP requests to APIs. Supports GET, POST, PUT, PATCH, DELETE."""
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required: pip install httpx")

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


__all__ = [
    # Decorator
    "tool",
    # Models
    "Question",
    "QuestionOption",
    # Builtin tools
    "ask_user_question",
    "read",
    "write",
    "edit",
    "bash",
    "glob",
    "grep",
    # Web tools
    "web_fetch",
    "web_search",
    "http_request",
]
