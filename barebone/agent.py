from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from barebone.types import Message
from barebone.types import Response
from barebone.types import Tool
from barebone.types import ToolCall
from barebone.types import ToolResult


@dataclass
class Agent:
    provider: Any
    tools: list[object] = field(default_factory=list)
    system: str | None = None
    max_tokens: int = 8192
    temperature: float | None = None
    messages: list[Message] = field(default_factory=list)

    def _get_tool_handler(self, name: str) -> Callable | None:
        for t in self.tools:
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
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**tool_call.arguments)
            else:
                result = handler(**tool_call.arguments)
            return ToolResult(
                id=tool_call.id,
                content=str(result) if result is not None else "Success",
            )
        except Exception as e:
            return ToolResult(id=tool_call.id, content=str(e), is_error=True)

    async def run(self, prompt: str, max_iterations: int = 10) -> Response:
        self.messages.append(Message(role="user", content=prompt))

        for _ in range(max_iterations):
            response = await self.provider.complete(
                messages=self.messages,
                tools=self.tools,
                system=self.system,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            if not response.tool_calls:
                if response.content:
                    self.messages.append(Message(role="assistant", content=response.content))
                return response

            self.messages.append(
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

            self.messages.append(Message(role="user", tool_results=results))

        return response


__all__ = ["Agent"]
