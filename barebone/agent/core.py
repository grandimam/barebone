from __future__ import annotations

import asyncio
import os
from typing import Any
from typing import TypeVar

from pydantic import BaseModel

from barebone.agent.router import Router
from barebone.agent.types import Message
from barebone.agent.types import Response
from barebone.agent.types import ToolCall
from barebone.tools import Tool
from barebone.tools.base import resolve_tool
from barebone.tools.base import tools_to_schema
from barebone.tools.types import ToolDef

T = TypeVar("T", bound=BaseModel)

_router: Router | None = None


def _get_router(api_key: str | None = None) -> Router:
    global _router

    if api_key:
        if api_key.startswith("sk-ant-oat-"):
            return Router(anthropic_oauth=api_key)
        elif api_key.startswith("sk-ant-"):
            return Router(anthropic_api_key=api_key)
        elif api_key.startswith("sk-or-"):
            return Router(openrouter=api_key)
        return Router(anthropic_api_key=api_key)

    if _router:
        return _router

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if anthropic_key or openrouter_key:
        _router = Router(anthropic_api_key=anthropic_key, openrouter=openrouter_key)
        return _router

    from barebone.auth import TokenManager
    token_manager = TokenManager.auto()
    if token_manager.has_credentials:
        token = asyncio.run(token_manager.get_token())
        _router = Router(anthropic_oauth=token)
        return _router

    raise ValueError(
        "No credentials found. Either:\n"
        "  1. Set ANTHROPIC_API_KEY environment variable\n"
        "  2. Set OPENROUTER_API_KEY environment variable\n"
        "  3. Pass api_key parameter"
    )


async def acomplete(
    model: str,
    messages: list[Message],
    *,
    system: str | None = None,
    tools: list[type[Tool] | ToolDef] | None = None,
    api_key: str | None = None,
    max_tokens: int = 8192,
    temperature: float | None = None,
) -> Response:
    router = _get_router(api_key)

    tools_schema = None
    if tools:
        tool_defs = [resolve_tool(t) for t in tools]
        tools_schema = tools_to_schema(tool_defs)

    return await router.complete(
        model=model,
        messages=messages,
        system=system,
        tools=tools_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def complete(
    model: str,
    messages: list[Message],
    **kwargs: Any,
) -> Response:
    return asyncio.run(acomplete(model, messages, **kwargs))


async def aexecute(
    tool_call: ToolCall,
    tools: list[type[Tool] | ToolDef],
) -> str:
    handlers = {}
    for tool in tools:
        tool_def = resolve_tool(tool)
        handlers[tool_def.name] = tool_def.handler

    handler = handlers.get(tool_call.name)
    if not handler:
        return f"Error: Unknown tool '{tool_call.name}'"

    try:
        if asyncio.iscoroutinefunction(handler):
            result = await handler(**tool_call.arguments)
        else:
            result = handler(**tool_call.arguments)
        return str(result) if result is not None else "Success"
    except Exception as e:
        return f"Error: {e}"


def execute(tool_call: ToolCall, tools: list[type[Tool] | ToolDef]) -> str:
    return asyncio.run(aexecute(tool_call, tools))


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str) -> Message:
    return Message(role="assistant", content=content)


def tool_result(tool_call: ToolCall, result: str) -> Message:
    return Message(
        role="tool_result",
        content=result,
        tool_call_id=tool_call.id,
        name=tool_call.name,
    )


__all__ = [
    "complete",
    "acomplete",
    "execute",
    "aexecute",
    "user",
    "assistant",
    "tool_result",
]
