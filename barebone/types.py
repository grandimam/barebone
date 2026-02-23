from __future__ import annotations

import time
from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal
from typing import Self
from typing import Union

from pydantic import BaseModel


type NullableStr = str | None


@dataclass
class TextContent:
    type: Literal["text"]
    text: str


@dataclass
class ImageContent:
    type: Literal["image"]
    source: str


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
    content: NullableStr = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"

    @property
    def done(self) -> bool:
        return len(self.tool_calls) == 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class Messages:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def user(self, content: str) -> Self:
        self._messages.append(Message(role="user", content=content))
        return self

    def assistant(self, response: Response) -> Self:
        self._messages.append(
            Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            )
        )
        return self

    def tool_results(self, results: list[ToolResult]) -> Self:
        self._messages.append(Message(role="user", tool_results=results))
        return self

    def clear(self) -> Self:
        self._messages.clear()
        return self

    @property
    def list(self) -> list[Message]:
        return self._messages

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __getitem__(self, index: int) -> Message:
        return self._messages[index]


@dataclass
class OAuthCredentials:
    access_token: str
    refresh_token: str
    expires_at: float
    account_id: NullableStr = None

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at


class QuestionOption(BaseModel):
    label: str
    description: str


class Question(BaseModel):
    question: str
    header: str
    options: list[QuestionOption]
    multiSelect: bool = False


Content = Union[TextContent, ImageContent]


__all__ = [
    "Content",
    "ImageContent",
    "Message",
    "Messages",
    "Response",
    "TextContent",
    "Tool",
    "ToolCall",
    "ToolResult",
]
