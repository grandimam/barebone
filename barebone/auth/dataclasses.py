import time

from typing import Any
from dataclasses import dataclass


@dataclass
class ModelInfo:
    id: str
    name: str
    description: str
    context_length: int
    pricing_prompt: float
    pricing_completion: float
    

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