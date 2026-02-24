from __future__ import annotations

import json
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from dataclasses import asdict

from .types import Checkpoint


class Storage(ABC):
    @abstractmethod
    async def save(self, agent_id: str, checkpoint: Checkpoint) -> None:
        pass

    @abstractmethod
    async def load(self, agent_id: str) -> Checkpoint | None:
        pass

    @abstractmethod
    async def delete(self, agent_id: str) -> None:
        pass


class MemoryStorage(Storage):
    def __init__(self) -> None:
        self._data: dict[str, Checkpoint] = {}

    async def save(self, agent_id: str, checkpoint: Checkpoint) -> None:
        self._data[agent_id] = checkpoint

    async def load(self, agent_id: str) -> Checkpoint | None:
        return self._data.get(agent_id)

    async def delete(self, agent_id: str) -> None:
        self._data.pop(agent_id, None)


class FileStorage(Storage):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    def _file_path(self, agent_id: str) -> Path:
        return self._path / f"{agent_id}.json"

    async def save(self, agent_id: str, checkpoint: Checkpoint) -> None:
        data = asdict(checkpoint)
        self._file_path(agent_id).write_text(json.dumps(data, indent=2))

    async def load(self, agent_id: str) -> Checkpoint | None:
        path = self._file_path(agent_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return Checkpoint(**data)

    async def delete(self, agent_id: str) -> None:
        path = self._file_path(agent_id)
        if path.exists():
            path.unlink()
