from __future__ import annotations

import uuid
import asyncio
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Callable
from typing import Coroutine
from enum import Enum
from collections.abc import AsyncIterator


class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Context:
    agent_id: str
    run_id: str
    state: dict[str, Any]
    status: AgentStatus = AgentStatus.PENDING
    started_at: datetime = field(default_factory=datetime.utcnow)
    _inbox: asyncio.Queue[AgentEvent] = field(default_factory=asyncio.Queue)
    _checkpoint_callback: Callable[[], Coroutine[Any, Any, None]] | None = None
    _suspend_callback: Callable[[str, dict], Coroutine[Any, Any, Any]] | None = None

    async def checkpoint(self) -> None:
        if self._checkpoint_callback:
            await self._checkpoint_callback()

    async def suspend(self, reason: str, **kwargs: Any) -> Any:
        if self._suspend_callback:
            return await self._suspend_callback(reason, kwargs)
        event = await self._inbox.get()
        return event.data

    async def emit(self, event: AgentEvent) -> None:
        pass

    def receive_event(self, event: AgentEvent) -> None:
        self._inbox.put_nowait(event)


@dataclass
class AgentSpec:
    fn: Callable[[Context], Coroutine[Any, Any, Any]]
    name: str

    def __call__(self, ctx: Context) -> Coroutine[Any, Any, Any]:
        return self.fn(ctx)


def agent(fn: Callable[[Context], Coroutine[Any, Any, Any]]) -> AgentSpec:
    return AgentSpec(fn=fn, name=fn.__name__)


@dataclass
class AgentHandle:
    agent_id: str
    _context: Context
    _task: asyncio.Task | None = None
    _result: Any = None
    _error: Exception | None = None

    @property
    def status(self) -> AgentStatus:
        return self._context.status

    @property
    def state(self) -> dict[str, Any]:
        return self._context.state

    @property
    def result(self) -> Any:
        return self._result

    @property
    def error(self) -> Exception | None:
        return self._error

    def send(self, event: AgentEvent) -> None:
        self._context.receive_event(event)

    async def wait(self) -> Any:
        if self._task:
            return await self._task
        return self._result


class Runtime:
    def __init__(self) -> None:
        self._agents: dict[str, AgentHandle] = {}

    async def start(
        self,
        agent_spec: AgentSpec,
        state: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> AgentHandle:
        agent_id = agent_id or str(uuid.uuid4())
        run_id = str(uuid.uuid4())

        ctx = Context(
            agent_id=agent_id,
            run_id=run_id,
            state=state or {},
            status=AgentStatus.RUNNING,
        )

        handle = AgentHandle(agent_id=agent_id, _context=ctx)
        self._agents[agent_id] = handle

        async def run_agent() -> Any:
            try:
                result = await agent_spec(ctx)
                ctx.status = AgentStatus.COMPLETED
                handle._result = result
                return result
            except Exception as e:
                ctx.status = AgentStatus.FAILED
                handle._error = e
                raise

        handle._task = asyncio.create_task(run_agent())
        return handle

    def get(self, agent_id: str) -> AgentHandle | None:
        return self._agents.get(agent_id)

    async def stop(self, agent_id: str) -> None:
        handle = self._agents.get(agent_id)
        if handle and handle._task:
            handle._task.cancel()
            handle._context.status = AgentStatus.FAILED
