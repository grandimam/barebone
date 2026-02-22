from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any

from barebone.providers import AnthropicProvider
from barebone.providers import _BaseProvider
from barebone.providers import OpenAIProvider
from barebone.types import Content
from barebone.types import ImageContent
from barebone.types import Message
from barebone.types import Response
from barebone.types import TextContent
from barebone.types import Tool
from barebone.types import ToolCall
from barebone.types import ToolResult
from barebone.types import NullableStr


def _detect_provider(api_key: str) -> str:
    if api_key.startswith("sk-ant-"):
        return "anthropic"
    elif api_key.startswith("sk-"):
        return "openai"
    else:
        return "anthropic"


def _create_provider(api_key: str, model: str) -> _BaseProvider:
    provider_type = _detect_provider(api_key)

    if provider_type == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)
    elif provider_type == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


class Agent:
    def __init__(
        self,
        model: NullableStr = None,
        api_key: NullableStr = None,
        *,
        provider: _BaseProvider | None = None,
        tools: list[Callable] | None = None,
        system: NullableStr = None,
        messages: list[Message] | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
        timeout: float | None = None,
    ):
        if not api_key and not provider:
            raise ValueError("Either api_key or provider is required")

        self._model = model
        self._api_key = api_key
        self._timeout = timeout

        if provider:
            self._provider = provider
            self._provider_type = getattr(provider, "name", "custom")
        else:
            self._provider_type = _detect_provider(api_key)
            self._provider = _create_provider(api_key, model)

        self._messages = messages if messages is not None else []
        self._tools = tools if tools is not None else []
        self._system = system
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def provider(self) -> _BaseProvider:
        return self._provider

    @property
    def provider_type(self) -> str:
        return self._provider_type

    @property
    def messages(self) -> list[Message]:
        return self._messages

    @property
    def tools(self) -> list:
        return self._tools

    @property
    def system(self) -> NullableStr:
        return self._system

    async def __aenter__(self) -> "Agent":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if hasattr(self._provider, "close"):
            await self._provider.close()

    def _get_tool_handler(self, name: str) -> Callable | None:
        for t in self._tools:
            if hasattr(t, "to_tool"):
                tool_def = t.to_tool()
                if tool_def.name == name:
                    return tool_def.handler
            elif isinstance(t, Tool) and t.name == name:
                return t.handler
        return None

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        handler = self._get_tool_handler(tool_call.name)
        if not handler:
            return ToolResult(
                id=tool_call.id,
                content=f"Unknown tool: {tool_call.name}",
                is_error=True,
            )

        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(**tool_call.arguments)
            else:
                result = handler(**tool_call.arguments)
            return ToolResult(
                id=tool_call.id,
                content=str(result) if result is not None else "Success",
            )
        except Exception as e:
            return ToolResult(id=tool_call.id, content=str(e), is_error=True)

    def _create_message(
        self, prompt: str | list[Content], images: list[str] | None = None
    ) -> Message:
        if isinstance(prompt, list):
            return Message(role="user", content=prompt)

        if images:
            content: list[Content] = [TextContent(type="text", text=prompt)]
            for img in images:
                content.append(ImageContent(type="image", source=img))
            return Message(role="user", content=content)

        return Message(role="user", content=prompt)

    async def run(
        self,
        prompt: str | list[Content],
        *,
        images: list[str] | None = None,
        max_iterations: int = 10,
        timeout: float | None = None,
    ) -> Response:
        effective_timeout = timeout or self._timeout

        async def _run() -> Response:
            self._messages.append(self._create_message(prompt, images))

            response: Response | None = None
            for _ in range(max_iterations):
                response = await self._provider.complete(
                    messages=self._messages,
                    tools=self._tools,
                    system=self._system,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )

                if not response.tool_calls:
                    if response.content:
                        self._messages.append(
                            Message(role="assistant", content=response.content)
                        )
                    return response

                self._messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                results = []
                for tc in response.tool_calls:
                    result = await self._execute_tool(tc)
                    results.append(result)

                self._messages.append(Message(role="user", tool_results=results))

            return response

        if effective_timeout:
            return await asyncio.wait_for(_run(), timeout=effective_timeout)
        return await _run()

    def run_sync(
        self,
        prompt: str | list[Content],
        *,
        images: list[str] | None = None,
        max_iterations: int = 10,
        timeout: float | None = None,
    ) -> Response:
        return asyncio.run(
            self.run(prompt, images=images, max_iterations=max_iterations, timeout=timeout)
        )

    async def stream(
        self,
        prompt: str | list[Content],
        *,
        images: list[str] | None = None,
        max_iterations: int = 10,
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        effective_timeout = timeout or self._timeout
        start_time = asyncio.get_event_loop().time() if effective_timeout else None

        self._messages.append(self._create_message(prompt, images))

        for _ in range(max_iterations):
            if effective_timeout and start_time:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= effective_timeout:
                    raise asyncio.TimeoutError("Stream timeout exceeded")

            response: Response | None = None
            content = ""
            tool_calls: list[ToolCall] = []

            async for event in self._provider.stream(
                messages=self._messages,
                tools=self._tools,
                system=self._system,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            ):
                event_type = event.get("type")
                if event_type == "text_delta":
                    content += event.get("text", "")
                    yield event
                elif event_type == "tool_call_start":
                    yield event
                elif event_type == "tool_call_end":
                    tc = ToolCall(
                        id=event["id"],
                        name=event["name"],
                        arguments=event["arguments"],
                    )
                    tool_calls.append(tc)
                    yield event
                elif event_type == "done":
                    response = event["response"]

            if not tool_calls:
                if content:
                    self._messages.append(Message(role="assistant", content=content))
                yield {"type": "done", "response": response}
                return

            self._messages.append(
                Message(
                    role="assistant",
                    content=content or None,
                    tool_calls=tool_calls,
                )
            )

            results = []
            for tc in tool_calls:
                result = await self._execute_tool(tc)
                results.append(result)
                yield {
                    "type": "tool_result",
                    "id": result.id,
                    "content": result.content,
                    "is_error": result.is_error,
                }

            self._messages.append(Message(role="user", tool_results=results))

        if response:
            yield {"type": "done", "response": response}

    def clear_messages(self) -> None:
        self._messages.clear()

    def add_tool(self, tool: Callable) -> None:
        self._tools.append(tool)


__all__ = ["Agent"]
