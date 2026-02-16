from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any
from typing import ClassVar

from barebone.common.dataclasses import Message
from barebone.common.dataclasses import Response
from barebone.common.dataclasses import StreamEvent


class Provider(ABC):
    name: ClassVar[str | None] = None

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> Response:
        ...

    @abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        ...
