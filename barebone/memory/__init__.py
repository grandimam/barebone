"""Memory module - simple message logging."""

from barebone.memory.base import Memory, MemoryBackend, LogEntry
from barebone.memory.sqlite import SQLiteBackend
from barebone.memory.inmemory import InMemoryBackend

# Convenience aliases
SQLiteMemory = lambda path, **kw: Memory(path=path, **kw)
InMemoryMemory = lambda **kw: Memory(backend=InMemoryBackend(**kw))

__all__ = [
    "Memory",
    "MemoryBackend",
    "LogEntry",
    "SQLiteBackend",
    "InMemoryBackend",
    "SQLiteMemory",
    "InMemoryMemory",
]
