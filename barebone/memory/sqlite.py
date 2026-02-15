from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from barebone.agent.types import Message
from barebone.memory.base import MemoryBackend


class SQLiteBackend(MemoryBackend):

    def __init__(self, path: str, max_messages: int = 100):
        self.path = Path(path).expanduser().resolve()
        self.max_messages = max_messages

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON messages(timestamp DESC);
        """)
        self._conn.commit()

    def log(self, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, datetime.now().isoformat())
        )
        self._conn.commit()
        self._trim_messages()

    def _trim_messages(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        if count > self.max_messages:
            self._conn.execute(
                """DELETE FROM messages WHERE id IN (
                    SELECT id FROM messages ORDER BY timestamp ASC LIMIT ?
                )""",
                (count - self.max_messages,)
            )
            self._conn.commit()

    def get_messages(self, limit: int = 50) -> list[Message]:
        rows = self._conn.execute(
            "SELECT role, content FROM messages ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [Message(role=r["role"], content=r["content"]) for r in reversed(rows)]

    def clear(self) -> None:
        self._conn.execute("DELETE FROM messages")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


__all__ = ["SQLiteBackend"]
