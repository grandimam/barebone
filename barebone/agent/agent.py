from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Literal

from barebone.agent.router import Router
from barebone.agent.hooks import Hooks
from barebone.tools.base import is_tool_class
from barebone.tools.base import tools_to_schema
from barebone.common.dataclasses import ToolDef, Message, Response, Done, StreamEvent

_BUILTIN_TOOLS: dict[str, type] = {}

Provider = Literal["anthropic", "openai-codex", "openrouter"]


def _get_builtin_tools() -> dict[str, type]:
    global _BUILTIN_TOOLS
    if not _BUILTIN_TOOLS:
        from barebone.tools.builtin import AskUserQuestion, Read, Write, Edit, Bash, Glob, Grep
        from barebone.tools.web import WebFetch, WebSearch, HttpRequest

        _BUILTIN_TOOLS = {
            "AskUserQuestion": AskUserQuestion,
            "Read": Read,
            "Write": Write,
            "Edit": Edit,
            "Bash": Bash,
            "Glob": Glob,
            "Grep": Grep,
            "WebFetch": WebFetch,
            "WebSearch": WebSearch,
            "HttpRequest": HttpRequest,
        }
    return _BUILTIN_TOOLS


def _create_router(provider: Provider, api_key: str | None = None) -> Router:
    if provider == "anthropic":
        if not api_key:
            raise ValueError("api_key is required for anthropic provider")
        if api_key.startswith("sk-ant-oat"):
            return Router(anthropic_oauth=api_key)
        return Router(anthropic_api_key=api_key)

    elif provider == "openai-codex":
        return Router(codex=True)

    elif provider == "openrouter":
        if not api_key:
            raise ValueError("api_key is required for openrouter provider")
        return Router(openrouter=api_key)

    else:
        raise ValueError(f"Unknown provider: {provider}")


class Agent:

    def __init__(
        self,
        model: str,
        *,
        provider: Provider,
        api_key: str | None = None,
        tools: list[Any] | None = None,
        system: str | None = None,
        hooks: Hooks | None = None,
        max_turns: int = 10,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ):
        self._model = model
        self._provider = provider
        self._system = system
        self._router = _create_router(provider, api_key)
        self._hooks = hooks
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._messages: list[Message] = []
        self._tool_defs: list[ToolDef] = []

        if tools:
            for t in tools:
                self.add_tool(t)

    def add_tool(self, tool: Any) -> None:
        if isinstance(tool, ToolDef):
            tool_def = tool
        elif isinstance(tool, str):
            builtin_tools = _get_builtin_tools()
            if tool not in builtin_tools:
                raise ValueError(
                    f"Unknown built-in tool '{tool}'. "
                    f"Available: {', '.join(builtin_tools.keys())}"
                )
            tool_def = builtin_tools[tool].to_tool_def()
        elif is_tool_class(tool):
            tool_def = tool.to_tool_def()
        elif hasattr(tool, "to_tool_def"):
            tool_def = tool.to_tool_def()
        else:
            raise TypeError(
                f"Cannot convert {type(tool).__name__} to tool. "
                "Use a Tool subclass, @tool decorated function, ToolDef, "
                "or built-in tool name."
            )

        if not any(t.name == tool_def.name for t in self._tool_defs):
            self._tool_defs.append(tool_def)

    @property
    def messages(self) -> list[Message]:
        return self._messages

    @property
    def tools(self) -> list[ToolDef]:
        return self._tool_defs.copy()

    def clear_messages(self) -> None:
        self._messages = []

    async def run(self, prompt: str) -> Response:
        self._messages.append(Message(role="user", content=prompt))
        tools_schema = tools_to_schema(self._tool_defs) if self._tool_defs else None

        for _ in range(self._max_turns):
            response = await self._router.complete(
                model=self._model,
                messages=self._messages,
                system=self._system,
                tools=tools_schema,
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
                    content=[
                        {"type": "text", "text": response.content} if response.content else None,
                        *[
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                            for tc in response.tool_calls
                        ],
                    ],
                )
            )

            tool_results = []
            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result,
                })

            self._messages.append(Message(role="user", content=tool_results))

        return response

    async def _execute_tool(self, tool_call: Any) -> str:
        from barebone.agent.hooks import Deny

        handler = None
        for tool_def in self._tool_defs:
            if tool_def.name == tool_call.name:
                handler = tool_def.handler
                break

        if not handler:
            return f"Error: Unknown tool '{tool_call.name}'"

        try:
            if self._hooks:
                for hook in self._hooks._before:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(tool_call)
                    else:
                        hook(tool_call)

            if asyncio.iscoroutinefunction(handler):
                result = await handler(**tool_call.arguments)
            else:
                result = handler(**tool_call.arguments)

            result_str = str(result) if result is not None else "Success"

            if self._hooks:
                for hook in self._hooks._after:
                    if asyncio.iscoroutinefunction(hook):
                        new_result = await hook(tool_call, result_str)
                    else:
                        new_result = hook(tool_call, result_str)
                    if new_result is not None:
                        result_str = new_result

            return result_str

        except Deny as e:
            return f"Denied: {e.reason}"
        except Exception as e:
            return f"Error: {e}"

    def run_sync(self, prompt: str) -> Response:
        return asyncio.run(self.run(prompt))

    async def stream(self, prompt: str) -> AsyncIterator[StreamEvent]:
        self._messages.append(Message(role="user", content=prompt))
        tools_schema = tools_to_schema(self._tool_defs) if self._tool_defs else None

        for _ in range(self._max_turns):
            response = None

            async for event in self._router.stream(
                model=self._model,
                messages=self._messages,
                system=self._system,
                tools=tools_schema,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            ):
                yield event

                if isinstance(event, Done):
                    response = event.response

            if response is None:
                return

            if not response.tool_calls:
                if response.content:
                    self._messages.append(
                        Message(role="assistant", content=response.content)
                    )
                return

            content_parts = []
            if response.content:
                content_parts.append({"type": "text", "text": response.content})
            for tc in response.tool_calls:
                content_parts.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })

            self._messages.append(Message(role="assistant", content=content_parts))

            tool_results = []
            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result,
                })

            self._messages.append(Message(role="user", content=tool_results))


__all__ = ["Agent"]
