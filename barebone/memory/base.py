from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from barebone.agent.types import Message


@dataclass
class LogEntry:
    role: str
    content: str
    timestamp: datetime


class MemoryBackend(ABC):

    @abstractmethod
    def log(self, role: str, content: str) -> None:
        pass

    @abstractmethod
    def get_messages(self, limit: int = 50) -> list[Message]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class Memory:

    def __init__(
        self,
        path: str | None = None,
        backend: MemoryBackend | None = None,
        max_messages: int = 100,
    ):
        if backend is not None:
            self._backend = backend
        elif path is not None:
            from barebone.memory.sqlite import SQLiteBackend
            self._backend = SQLiteBackend(path, max_messages)
        else:
            from barebone.memory.inmemory import InMemoryBackend
            self._backend = InMemoryBackend(max_messages)

        self._max_messages = max_messages

    @property
    def backend(self) -> MemoryBackend:
        return self._backend

    def log(self, role: str, content: str) -> None:
        self._backend.log(role, content)

    def get_messages(self, limit: int | None = None) -> list[Message]:
        return self._backend.get_messages(limit or self._max_messages)

    def clear(self) -> None:
        self._backend.clear()

    def close(self) -> None:
        if hasattr(self._backend, "close"):
            self._backend.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *args) -> None:
        self.close()


__all__ = [
    "Memory",
    "MemoryBackend",
    "LogEntry",
]
