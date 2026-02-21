from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from barebone.agent import Agent
from barebone.tools import tool
from barebone.types import Response
from barebone.types import Tool
from barebone.types import ToolCall


class TestAgentGetToolHandler:
    def test_find_tool_from_tool_object(self):
        def handler():
            return "result"

        t = Tool(name="my_tool", description="", parameters={}, handler=handler)
        agent = Agent(provider=MagicMock(), tools=[t])

        found = agent._get_tool_handler("my_tool")
        assert found is handler

    def test_find_tool_from_decorated_function(self):
        @tool
        def my_decorated_tool():
            """A test tool."""
            return "decorated result"

        agent = Agent(provider=MagicMock(), tools=[my_decorated_tool])
        found = agent._get_tool_handler("my_decorated_tool")
        assert found() == "decorated result"

    def test_tool_not_found(self):
        agent = Agent(provider=MagicMock(), tools=[])
        found = agent._get_tool_handler("nonexistent")
        assert found is None


class TestAgentExecuteTool:
    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        def sync_handler(x: int) -> int:
            return x * 2

        t = Tool(name="double", description="", parameters={}, handler=sync_handler)
        agent = Agent(provider=MagicMock(), tools=[t])

        tc = ToolCall(id="tc_1", name="double", arguments={"x": 5})
        result = await agent._execute_tool(tc)

        assert result.id == "tc_1"
        assert result.content == "10"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        async def async_handler(msg: str) -> str:
            return f"processed: {msg}"

        t = Tool(name="process", description="", parameters={}, handler=async_handler)
        agent = Agent(provider=MagicMock(), tools=[t])

        tc = ToolCall(id="tc_2", name="process", arguments={"msg": "hello"})
        result = await agent._execute_tool(tc)

        assert result.content == "processed: hello"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        agent = Agent(provider=MagicMock(), tools=[])
        tc = ToolCall(id="tc_3", name="unknown", arguments={})
        result = await agent._execute_tool(tc)

        assert "Unknown tool" in result.content
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_execute_tool_with_exception(self):
        def failing_handler():
            raise ValueError("Something went wrong")

        t = Tool(name="fail", description="", parameters={}, handler=failing_handler)
        agent = Agent(provider=MagicMock(), tools=[t])

        tc = ToolCall(id="tc_4", name="fail", arguments={})
        result = await agent._execute_tool(tc)

        assert "Something went wrong" in result.content
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_execute_tool_returns_none(self):
        def void_handler():
            pass

        t = Tool(name="void", description="", parameters={}, handler=void_handler)
        agent = Agent(provider=MagicMock(), tools=[t])

        tc = ToolCall(id="tc_5", name="void", arguments={})
        result = await agent._execute_tool(tc)

        assert result.content == "Success"
        assert result.is_error is False


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_simple_response_no_tools(self):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = Response(
            content="Hello!", tool_calls=[], stop_reason="stop"
        )

        agent = Agent(provider=mock_provider, tools=[])
        response = await agent.run("Hi")

        assert response.content == "Hello!"
        assert len(agent.messages) == 2  # user prompt + assistant response
        assert agent.messages[0].role == "user"
        assert agent.messages[0].content == "Hi"
        assert agent.messages[1].role == "assistant"
        assert agent.messages[1].content == "Hello!"

    @pytest.mark.asyncio
    async def test_run_with_tool_use(self):
        def get_time() -> str:
            return "12:00 PM"

        t = Tool(name="get_time", description="", parameters={}, handler=get_time)

        mock_provider = AsyncMock()
        # First call returns tool use
        # Second call returns final response
        mock_provider.complete.side_effect = [
            Response(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="get_time", arguments={})],
                stop_reason="tool_use",
            ),
            Response(content="The time is 12:00 PM", tool_calls=[], stop_reason="stop"),
        ]

        agent = Agent(provider=mock_provider, tools=[t])
        response = await agent.run("What time is it?")

        assert response.content == "The time is 12:00 PM"
        assert mock_provider.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_run_respects_max_iterations(self):
        mock_provider = AsyncMock()
        # Always return tool calls to test iteration limit
        mock_provider.complete.return_value = Response(
            content=None,
            tool_calls=[ToolCall(id="tc_1", name="dummy", arguments={})],
            stop_reason="tool_use",
        )

        def dummy_handler():
            return "done"

        t = Tool(name="dummy", description="", parameters={}, handler=dummy_handler)
        agent = Agent(provider=mock_provider, tools=[t])

        await agent.run("Loop forever", max_iterations=3)
        assert mock_provider.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_run_passes_system_prompt(self):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = Response(content="OK", tool_calls=[])

        agent = Agent(
            provider=mock_provider,
            tools=[],
            system="You are a helpful assistant.",
            max_tokens=4096,
            temperature=0.7,
        )
        await agent.run("Test")

        call_kwargs = mock_provider.complete.call_args.kwargs
        assert call_kwargs["system"] == "You are a helpful assistant."
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["temperature"] == 0.7


class TestAgentMessageHistory:
    @pytest.mark.asyncio
    async def test_messages_accumulate(self):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = Response(content="Response", tool_calls=[])

        agent = Agent(provider=mock_provider, tools=[])

        await agent.run("First")
        assert len(agent.messages) == 2

        await agent.run("Second")
        assert len(agent.messages) == 4

    @pytest.mark.asyncio
    async def test_tool_results_in_messages(self):
        def echo(text: str) -> str:
            return text

        t = Tool(name="echo", description="", parameters={}, handler=echo)

        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = [
            Response(
                content="Calling echo",
                tool_calls=[ToolCall(id="tc_1", name="echo", arguments={"text": "hello"})],
            ),
            Response(content="Done", tool_calls=[]),
        ]

        agent = Agent(provider=mock_provider, tools=[t])
        await agent.run("Echo hello")

        # Find the message with tool results
        tool_result_msg = next(
            (m for m in agent.messages if m.tool_results), None
        )
        assert tool_result_msg is not None
        assert tool_result_msg.tool_results[0].content == "hello"
