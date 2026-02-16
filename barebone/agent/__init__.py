from barebone.agent.agent import Agent
from barebone.agent.core import complete
from barebone.agent.core import stream
from barebone.agent.core import execute
from barebone.agent.hooks import Hooks
from barebone.agent.hooks import Deny
from barebone.common.dataclasses import ToolCall
from barebone.common.dataclasses import Message


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str) -> Message:
    return Message(role="assistant", content=content)


def tool_result(tool_call: ToolCall, result: str) -> Message:
    return Message(
        role="tool_result",
        content=result,
        tool_call_id=tool_call.id,
        name=tool_call.name,
    )

__all__ = [
    "Agent",
    "complete",
    "stream",
    "execute",
    "user",
    "assistant",
    "tool_result",
    "Hooks",
    "Deny",
]
