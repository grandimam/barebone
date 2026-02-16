from __future__ import annotations

import asyncio
from typing import Any
from typing import AsyncIterator
from typing import TypeVar

from pydantic import BaseModel

from barebone.agent.router import Router
from barebone.tools import Tool
from barebone.tools.base import resolve_tool
from barebone.tools.base import tools_to_schema
from barebone.common.dataclasses import ToolDef
from barebone.common.dataclasses import Message
from barebone.common.dataclasses import ToolCall
from barebone.common.dataclasses import Response
from barebone.common.dataclasses import StreamEvent

T = TypeVar("T", bound=BaseModel)


def _get_router(api_key: str) -> Router:
    if api_key.startswith("sk-ant-oat"):
        return Router(anthropic_oauth=api_key)
    if api_key.startswith("sk-or-"):
        return Router(openrouter=api_key)
    return Router(anthropic_api_key=api_key)


def _model_to_tool_schema(model: type[T]) -> dict[str, Any]:
    schema = model.model_json_schema()
    properties = {
        k: {key: val for key, val in v.items() if key != "title"}
        for k, v in schema.get("properties", {}).items()
    }
    return {
        "name": model.__name__,
        "description": model.__doc__ or f"Return a {model.__name__}",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": schema.get("required", []),
        },
    }


async def complete(
    model: str,
    messages: list[Message],
    *,
    api_key: str,
    system: str | None = None,
    tools: list[type[Tool] | ToolDef] | None = None,
    response_model: type[T] | None = None,
    max_tokens: int = 8192,
    temperature: float | None = None,
) -> Response:
    router = _get_router(api_key)

    tools_schema = None
    if tools:
        tool_defs = [resolve_tool(t) for t in tools]
        tools_schema = tools_to_schema(tool_defs)

    if response_model:
        tools_schema = [_model_to_tool_schema(response_model)]

    response = await router.complete(
        model=model,
        messages=messages,
        system=system,
        tools=tools_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if response_model and response.tool_calls:
        tc = response.tool_calls[0]
        response.parsed = response_model(**tc.arguments)

    return response


async def stream(
    model: str,
    messages: list[Message],
    *,
    api_key: str,
    system: str | None = None,
    tools: list[type[Tool] | ToolDef] | None = None,
    max_tokens: int = 8192,
    temperature: float | None = None,
) -> AsyncIterator[StreamEvent]:
    router = _get_router(api_key)

    tools_schema = None
    if tools:
        tool_defs = [resolve_tool(t) for t in tools]
        tools_schema = tools_to_schema(tool_defs)

    async for event in router.stream(
        model=model,
        messages=messages,
        system=system,
        tools=tools_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        yield event


async def execute(
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


__all__ = [
    "complete",
    "stream",
    "execute",
]
