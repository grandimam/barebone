from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class Message:
    role: str
    content: Any
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Response:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider: str = ""

@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallStart:
    id: str
    name: str


@dataclass
class ToolCallDelta:
    id: str
    arguments_delta: str


@dataclass
class ToolCallEnd:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Done:
    response: Response


StreamEvent = TextDelta | ToolCallStart | ToolCallDelta | ToolCallEnd | Done


__all__ = [
    "Message",
    "ToolCall",
    "Usage",
    "Response",
    "TextDelta",
    "ToolCallStart",
    "ToolCallDelta",
    "ToolCallEnd",
    "Done",
    "StreamEvent",
]
