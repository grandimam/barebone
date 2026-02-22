from __future__ import annotations

import time

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal
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
