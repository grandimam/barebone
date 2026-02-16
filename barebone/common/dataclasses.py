import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Union
from pydantic import BaseModel


class AnthropicMessage(BaseModel):
    model: str
    max_tokens: int
    system: str | None = None
    temperature: float | None = None

    def to_dict(self):
        pass


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


@dataclass
class OAuthCredentials:
    access_token: str
    refresh_token: str
    expires_at: float

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "accessToken": self.access_token,
            "refreshToken": self.refresh_token,
            "expiresAt": int(self.expires_at * 1000),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthCredentials":
        expires = data.get("expiresAt", 0)
        if expires > 1e12:
            expires = expires / 1000
        return cls(
            access_token=data["accessToken"],
            refresh_token=data["refreshToken"],
            expires_at=expires,
        )


StreamEvent = Union[TextDelta, ToolCallStart, ToolCallDelta, ToolCallEnd, Done]
