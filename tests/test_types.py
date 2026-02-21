from barebone.types import Message
from barebone.types import Response
from barebone.types import Tool
from barebone.types import ToolCall
from barebone.types import ToolResult


class TestToolCall:
    def test_creation(self):
        tc = ToolCall(id="tc_123", name="read", arguments={"file_path": "/test.txt"})
        assert tc.id == "tc_123"
        assert tc.name == "read"
        assert tc.arguments == {"file_path": "/test.txt"}

    def test_empty_arguments(self):
        tc = ToolCall(id="tc_456", name="list_files", arguments={})
        assert tc.arguments == {}


class TestToolResult:
    def test_success_result(self):
        tr = ToolResult(id="tc_123", content="File contents here")
        assert tr.id == "tc_123"
        assert tr.content == "File contents here"
        assert tr.is_error is False

    def test_error_result(self):
        tr = ToolResult(id="tc_123", content="File not found", is_error=True)
        assert tr.is_error is True


class TestTool:
    def test_creation(self):
        def handler(x: int) -> int:
            return x * 2

        tool = Tool(
            name="double",
            description="Doubles a number",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            handler=handler,
        )
        assert tool.name == "double"
        assert tool.description == "Doubles a number"
        assert tool.handler(5) == 10


class TestMessage:
    def test_user_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls == []
        assert msg.tool_results == []

    def test_assistant_message(self):
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"

    def test_system_message(self):
        msg = Message(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"

    def test_message_with_tool_calls(self):
        tc = ToolCall(id="tc_1", name="read", arguments={})
        msg = Message(role="assistant", content=None, tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "read"

    def test_message_with_tool_results(self):
        tr = ToolResult(id="tc_1", content="Success")
        msg = Message(role="user", tool_results=[tr])
        assert len(msg.tool_results) == 1


class TestResponse:
    def test_text_response(self):
        resp = Response(content="Hello!")
        assert resp.content == "Hello!"
        assert resp.tool_calls == []
        assert resp.stop_reason == "stop"

    def test_response_with_tool_calls(self):
        tc = ToolCall(id="tc_1", name="bash", arguments={"command": "ls"})
        resp = Response(content=None, tool_calls=[tc], stop_reason="tool_use")
        assert resp.content is None
        assert len(resp.tool_calls) == 1
        assert resp.stop_reason == "tool_use"
