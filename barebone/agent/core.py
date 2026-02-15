from __future__ import annotations

import asyncio
from typing import Any
from typing import AsyncIterator
from typing import TypeVar

from pydantic import BaseModel

from barebone.agent.router import Router
from barebone.agent.types import Message
from barebone.agent.types import Response
from barebone.agent.types import StreamEvent
from barebone.agent.types import ToolCall
from barebone.tools import Tool
from barebone.tools.base import resolve_tool
from barebone.tools.base import tools_to_schema
from barebone.tools.types import ToolDef

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


async def acomplete(
    model: str,
    messages: list[Message],
    *,
    api_key: str,
    system: str | None = None,
    tools: list[type[Tool] | ToolDef] | None = None,
    response_model: type[T] | None = None,
    max_tokens: int = 8192,
    temperature: float | None = None,
    timeout: float | None = None,
) -> Response:
    router = _get_router(api_key)

    tools_schema = None
    if tools:
        tool_defs = [resolve_tool(t) for t in tools]
        tools_schema = tools_to_schema(tool_defs)

    if response_model:
        tools_schema = [_model_to_tool_schema(response_model)]

    coro = router.complete(
        model=model,
        messages=messages,
        system=system,
        tools=tools_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if timeout:
        response = await asyncio.wait_for(coro, timeout=timeout)
    else:
        response = await coro

    if response_model and response.tool_calls:
        tc = response.tool_calls[0]
        response.parsed = response_model(**tc.arguments)

    return response


def complete(
    model: str,
    messages: list[Message],
    **kwargs: Any,
) -> Response:
    return asyncio.run(acomplete(model, messages, **kwargs))


async def astream(
    model: str,
    messages: list[Message],
    *,
    api_key: str,
    system: str | None = None,
    tools: list[type[Tool] | ToolDef] | None = None,
    max_tokens: int = 8192,
    temperature: float | None = None,
    timeout: float | None = None,
) -> AsyncIterator[StreamEvent]:
    router = _get_router(api_key)

    tools_schema = None
    if tools:
        tool_defs = [resolve_tool(t) for t in tools]
        tools_schema = tools_to_schema(tool_defs)

    if timeout:
        async with asyncio.timeout(timeout):
            async for event in router.stream(
                model=model,
                messages=messages,
                system=system,
                tools=tools_schema,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                yield event
    else:
        async for event in router.stream(
            model=model,
            messages=messages,
            system=system,
            tools=tools_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield event


def stream(
    model: str,
    messages: list[Message],
    **kwargs: Any,
):
    return astream(model, messages, **kwargs)


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
    "stream",
    "astream",
    "execute",
    "aexecute",
    "user",
    "assistant",
    "tool_result",
]
