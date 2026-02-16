from __future__ import annotations

import asyncio
from typing import Any
from typing import Callable

from ..common.dataclasses import ToolCall


class Deny(Exception):
    """Raised by before hooks to reject a tool call."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class Hooks:
    """Composable hooks for tool execution lifecycle.

    Example:
        hooks = Hooks()

        @hooks.before
        def validate(tool_call):
            if tool_call.name == "dangerous":
                raise Deny("Not allowed")

        @hooks.after
        def log(tool_call, result):
            print(f"{tool_call.name}: {result}")

        # In your loop
        for tool_call in response.tool_calls:
            result = hooks.run(tool_call, tools)
            messages.append(tool_result(tool_call, result))
    """

    Deny = Deny

    def __init__(self):
        self._before: list[Callable] = []
        self._after: list[Callable] = []

    def before(self, fn: Callable) -> Callable:
        """Register a before hook. Called before tool execution.

        Args:
            fn: Function(tool_call) -> None. Raise Deny to reject.

        Returns:
            The original function (for decorator use).
        """
        self._before.append(fn)
        return fn

    def after(self, fn: Callable) -> Callable:
        """Register an after hook. Called after tool execution.

        Args:
            fn: Function(tool_call, result) -> None or new_result.

        Returns:
            The original function (for decorator use).
        """
        self._after.append(fn)
        return fn

    def run(
        self,
        tool_call: ToolCall,
        tools: list[Any],
    ) -> str:
        """Execute tool with hooks: before -> execute -> after.

        Args:
            tool_call: The tool call to execute.
            tools: List of available tools.

        Returns:
            Tool execution result (possibly modified by after hooks).

        Raises:
            Deny: If a before hook rejects the tool call.
        """
        return asyncio.run(self.arun(tool_call, tools))

    async def arun(
        self,
        tool_call: ToolCall,
        tools: list[Any],
    ) -> str:
        """Execute tool with hooks: before -> execute -> after (async).

        Args:
            tool_call: The tool call to execute.
            tools: List of available tools.

        Returns:
            Tool execution result (possibly modified by after hooks).

        Raises:
            Deny: If a before hook rejects the tool call.
        """
        from .core import aexecute

        # Run before hooks
        for hook in self._before:
            if asyncio.iscoroutinefunction(hook):
                await hook(tool_call)
            else:
                hook(tool_call)

        # Execute tool
        result = await aexecute(tool_call, tools)

        # Run after hooks
        for hook in self._after:
            if asyncio.iscoroutinefunction(hook):
                new_result = await hook(tool_call, result)
            else:
                new_result = hook(tool_call, result)

            if new_result is not None:
                result = new_result

        return result


__all__ = [
    "Hooks",
    "Deny",
]
