from __future__ import annotations

from datetime import datetime
from typing import Any

from barebone.agent.types import Message
from barebone.memory.base import MemoryBackend


class InMemoryBackend(MemoryBackend):

    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages
        self._messages: list[dict[str, Any]] = []

    def log(self, role: str, content: str) -> None:
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def get_messages(self, limit: int = 50) -> list[Message]:
        recent = self._messages[-limit:]
        return [Message(role=m["role"], content=m["content"]) for m in recent]

    def clear(self) -> None:
        self._messages.clear()


__all__ = ["InMemoryBackend"]
