from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Union


@dataclass
class ModelInfo:
    id: str
    name: str
    description: str
    context_length: int
    pricing_prompt: float
    pricing_completion: float


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable


@dataclass
class Message:
    role: str
    content: Any
    tool_call_id: Union[str, None] = None
    name: Union[str, None] = None


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


StreamEvent = Union[TextDelta, ToolCallStart, ToolCallDelta, ToolCallEnd, Done]
