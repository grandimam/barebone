from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal
from typing import Protocol
from typing import Union


@dataclass
class TextContent:
    type: Literal["text"]
    text: str


@dataclass
class ImageContent:
    type: Literal["image"]
    source: str  # URL or base64 data URI


Content = Union[TextContent, ImageContent]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    id: str
    content: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    content: str | list[Content] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class Response:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"


class Provider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> Response: ...

    async def stream(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> AsyncIterator[Response]: ...


__all__ = [
    "TextContent",
    "ImageContent",
    "Content",
    "ToolCall",
    "ToolResult",
    "Tool",
    "Message",
    "Response",
    "Provider",
]
