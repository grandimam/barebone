from __future__ import annotations

import json
from typing import Any
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .types import Done
from .types import Error
from .types import Event
from .types import Message
from .types import Request
from .types import Response
from .types import TextContent
from .types import TextDelta
from .types import ImageContent
from .types import ToolCall
from .types import ToolCallEnd
from .types import ToolCallStart


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4",
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def _convert_content(self, content: str | list | None) -> str | list[dict[str, Any]]:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        result: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, TextContent):
                result.append({"type": "text", "text": item.text})
            elif isinstance(item, ImageContent):
                result.append({"type": "image_url", "image_url": {"url": item.source}})
        return result

    def _to_api_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.tool_results:
                for tr in msg.tool_results:
                    result.append({
                        "role": "tool",
                        "tool_call_id": tr.id,
                        "content": tr.content,
                    })
            elif msg.tool_calls:
                result.append({
                    "role": "assistant",
                    "content": msg.content if isinstance(msg.content, str) else "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                result.append({
                    "role": msg.role,
                    "content": self._convert_content(msg.content),
                })
        return result

    def _to_api_tools(self, tools: list[object] | None) -> list[dict[str, Any]] | None:
        result: list[dict[str, Any]] = []
        for t in tools:
            tool_def = t.to_tool() if hasattr(t, "to_tool") else t
            result.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": tool_def.parameters,
                },
            })
        return result

    async def stream(self, request: Request) -> AsyncIterator[Event]:
        api_messages = self._to_api_messages(request.messages)
        if request.system:
            api_messages.insert(0, {"role": "system", "content": request.system})

        params: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        if request.tools:
            params["tools"] = self._to_api_tools(request.tools)
        if request.temperature is not None:
            params["temperature"] = request.temperature

        content = ""
        tool_calls: list[ToolCall] = []
        tool_call_map: dict[int, dict[str, Any]] = {}

        try:
            stream = await self._client.chat.completions.create(**params)
            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if delta.content:
                    content += delta.content
                    yield TextDelta(id=request.id, text=delta.content)

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_map:
                            tool_call_map[idx] = {
                                "id": tc_delta.id or "",
                                "name": tc_delta.function.name if tc_delta.function else "",
                                "arguments_json": "",
                            }
                            yield ToolCallStart(
                                id=tool_call_map[idx]["id"],
                                request_id=request.id,
                                name=tool_call_map[idx]["name"],
                            )
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_call_map[idx]["arguments_json"] += tc_delta.function.arguments

            for tc_data in tool_call_map.values():
                try:
                    arguments = json.loads(tc_data["arguments_json"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tc = ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=arguments)
                tool_calls.append(tc)
                yield ToolCallEnd(
                    id=tc.id,
                    request_id=request.id,
                    name=tc.name,
                    arguments=tc.arguments,
                )

            yield Done(
                id=request.id,
                response=Response(
                    content=content or None,
                    tool_calls=tool_calls,
                    stop_reason="stop",
                ),
            )

        except Exception as e:
            yield Error(id=request.id, error=str(e))

    async def close(self) -> None:
        await self._client.close()
