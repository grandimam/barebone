import json
from collections.abc import AsyncIterator
from typing import Any
from typing import Union

import httpx
import openai

from barebone.common.dataclasses import ToolCall
from barebone.common.dataclasses import Usage
from barebone.common.dataclasses import Response
from barebone.common.dataclasses import ToolCallStart
from barebone.common.dataclasses import ToolCallDelta
from barebone.common.dataclasses import ToolCallEnd
from barebone.common.dataclasses import Done
from barebone.common.dataclasses import StreamEvent
from barebone.common.dataclasses import ModelInfo
from barebone.common.dataclasses import TextDelta
from barebone.common.dataclasses import Message
from barebone.auth.base import Provider


class OpenRouterMixin:

    async def fetch_models(self, force: bool = False) -> list[ModelInfo]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.client.api_key}"},
            )
            response.raise_for_status()
            data = response.json()

        models = []
        for model_data in data.get("data", []):
            pricing = model_data.get("pricing", {})
            models.append(ModelInfo(
                id=model_data["id"],
                name=model_data.get("name", model_data["id"]),
                description=model_data.get("description", ""),
                context_length=model_data.get("context_length", 4096),
                pricing_prompt=float(pricing.get("prompt", 0)),
                pricing_completion=float(pricing.get("completion", 0)),
            ))
        return models

    def _convert_messages(
        self, messages: list[Message], system: Union[str, None]
    ) -> list[dict[str, Any]]:
        result = []

        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "tool_result":
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": str(msg.content),
                })
            elif msg.role == "assistant" and isinstance(msg.content, list):
                tool_calls = []
                text_content = ""
                for block in msg.content:
                    if isinstance(block, ToolCall):
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.arguments),
                            },
                        })
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_content += block.get("text", "")
                msg_dict: dict[str, Any] = {"role": "assistant"}
                if text_content:
                    msg_dict["content"] = text_content
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                result.append(msg_dict)
            else:
                result.append({"role": msg.role, "content": msg.content})

        return result

    def _convert_tools(self, tools: Union[list[dict[str, Any]], None]) -> Union[list[dict[str, Any]], None]:
        if not tools:
            return None

        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for tool in tools
        ]

    def _parse_response(
        self, response: openai.types.chat.ChatCompletion, model: str
    ) -> Response:
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        return Response(
            content=message.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
            usage=Usage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
            model=model,
            provider=self.name,
        )


class OpenRouterProvider(Provider, OpenRouterMixin):
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, app_name: str = "bare"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            default_headers={
                "HTTP-Referer": f"https://github.com/{app_name}",
                "X-Title": app_name,
            },
        )

    @property
    def name(self) -> str:
        return "openrouter"

    async def complete(
        self,
        model: str,
        messages: list[Message],
        system: Union[str, None] = None,
        tools: Union[list[dict[str, Any]], None] = None,
        max_tokens: int = 8192,
        temperature: Union[float, None] = None,
    ) -> Response:
        params: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages, system),
            "max_tokens": max_tokens,
        }
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            params["tools"] = converted_tools

        if temperature is not None:
            params["temperature"] = temperature

        response = await self.client.chat.completions.create(**params)
        return self._parse_response(response, model)

    async def stream(
        self,
        model: str,
        messages: list[Message],
        system: Union[str, None] = None,
        tools: Union[list[dict[str, Any]], None] = None,
        max_tokens: int = 8192,
        temperature: Union[float, None] = None,
    ) -> AsyncIterator[StreamEvent]:
        params: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages, system),
            "max_tokens": max_tokens,
            "stream": True,
        }

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            params["tools"] = converted_tools

        if temperature is not None:
            params["temperature"] = temperature

        content = ""
        tool_calls_map: dict[int, dict[str, Any]] = {}
        usage = Usage()

        async for chunk in await self.client.chat.completions.create(**params):
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage.input_tokens = chunk.usage.prompt_tokens
                    usage.output_tokens = chunk.usage.completion_tokens
                    usage.total_tokens = chunk.usage.total_tokens
                continue

            delta = chunk.choices[0].delta

            # Text content
            if delta.content:
                content += delta.content
                yield TextDelta(text=delta.content)

            # Tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index

                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function else "",
                            "arguments_json": "",
                        }
                        if tc.id and tc.function and tc.function.name:
                            yield ToolCallStart(id=tc.id, name=tc.function.name)

                    if tc.function and tc.function.arguments:
                        tool_calls_map[idx]["arguments_json"] += tc.function.arguments
                        yield ToolCallDelta(
                            id=tool_calls_map[idx]["id"],
                            arguments_delta=tc.function.arguments,
                        )

        # Finalize tool calls
        final_tool_calls = []
        for idx in sorted(tool_calls_map.keys()):
            tc_data = tool_calls_map[idx]
            try:
                arguments = json.loads(tc_data["arguments_json"] or "{}")
            except json.JSONDecodeError:
                arguments = {}

            tc = ToolCall(
                id=tc_data["id"],
                name=tc_data["name"],
                arguments=arguments,
            )
            final_tool_calls.append(tc)
            yield ToolCallEnd(id=tc.id, name=tc.name, arguments=tc.arguments)

        yield Done(
            response=Response(
                content=content,
                tool_calls=final_tool_calls,
                stop_reason="stop",
                usage=usage,
                model=model,
                provider=self.name,
            )
        )
