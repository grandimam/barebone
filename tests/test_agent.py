from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from barebone.agent import Agent
from barebone.agent import _detect_provider
from barebone.tools import tool
from barebone.types import Message
from barebone.types import Response
from barebone.types import Tool
from barebone.types import ToolCall
from barebone.types import ToolResult


class Test_Tutorial1_CoreTypes:
    def test_tool_call_structure(self):
        tc = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"city": "Tokyo"},
        )

        assert tc.id == "call_123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Tokyo"}

    def test_tool_result_structure(self):
        success = ToolResult(id="call_123", content="Sunny, 25C")
        assert success.is_error is False

        error = ToolResult(id="call_456", content="City not found", is_error=True)
        assert error.is_error is True

    def test_message_structure(self):
        user_msg = Message(role="user", content="Hello!")
        assert user_msg.role == "user"

        assistant_msg = Message(
            role="assistant",
            content="Let me check that for you.",
            tool_calls=[ToolCall(id="tc1", name="search", arguments={})],
        )
        assert len(assistant_msg.tool_calls) == 1

    def test_response_structure(self):
        text_response = Response(content="Hello! How can I help?")
        assert text_response.content == "Hello! How can I help?"
        assert text_response.tool_calls == []

        tool_response = Response(
            content=None,
            tool_calls=[ToolCall(id="tc1", name="calculator", arguments={"expr": "2+2"})],
        )
        assert len(tool_response.tool_calls) == 1


class Test_Tutorial2_ToolDecorator:
    def test_basic_tool_creation(self):
        @tool
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"

        assert greet("Alice") == "Hello, Alice!"

        tool_def = greet.to_tool()
        assert tool_def.name == "greet"
        assert tool_def.description == "Greet someone by name."

    def test_tool_with_custom_name(self):
        @tool("say_hello")
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        tool_def = greet.to_tool()
        assert tool_def.name == "say_hello"

    def test_tool_parameter_schema(self):
        @tool
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        tool_def = add_numbers.to_tool()
        params = tool_def.parameters

        assert params["type"] == "object"
        assert "a" in params["properties"]
        assert "b" in params["properties"]
        assert set(params["required"]) == {"a", "b"}

    def test_tool_with_optional_params(self):
        @tool
        def search(query: str, limit: int = 10) -> str:
            """Search for something."""
            return f"Searching '{query}' (limit={limit})"

        tool_def = search.to_tool()
        assert "query" in tool_def.parameters["required"]


class Test_Tutorial3_ProviderDetection:
    def test_anthropic_key_detection(self):
        assert _detect_provider("sk-ant-api03-xxxxx") == "anthropic"

    def test_openai_key_detection(self):
        assert _detect_provider("sk-proj-xxxxx") == "openai"
        assert _detect_provider("sk-xxxxx") == "openai"

    def test_unknown_defaults_to_anthropic(self):
        assert _detect_provider("some-other-key") == "anthropic"


class Test_Tutorial4_CreatingAgents:
    def test_agent_requires_authentication(self):
        with pytest.raises(ValueError, match="Either api_key or provider is required"):
            Agent()

    def test_agent_with_provider(self):
        mock_provider = MagicMock()
        mock_provider.name = "mock"

        agent = Agent(provider=mock_provider)

        assert agent.provider is mock_provider
        assert agent.provider_type == "mock"

    def test_agent_with_tools(self):
        @tool
        def helper() -> str:
            """A helper tool."""
            return "helped"

        mock_provider = MagicMock()
        mock_provider.name = "mock"

        agent = Agent(provider=mock_provider, tools=[helper])

        assert len(agent.tools) == 1

    def test_agent_with_system_prompt(self):
        mock_provider = MagicMock()
        mock_provider.name = "mock"

        agent = Agent(
            provider=mock_provider,
            system="You are a helpful coding assistant.",
        )

        assert agent.system == "You are a helpful coding assistant."


class Test_Tutorial5_RunningAgent:
    @pytest.mark.asyncio
    async def test_simple_conversation(self):
        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.return_value = Response(
            content="Hello! I'm here to help.",
            tool_calls=[],
        )

        agent = Agent(provider=mock_provider)
        response = await agent.run("Hi there!")

        assert response.content == "Hello! I'm here to help."

    @pytest.mark.asyncio
    async def test_conversation_history(self):
        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.return_value = Response(content="Response", tool_calls=[])

        agent = Agent(provider=mock_provider)

        await agent.run("First message")
        assert len(agent.messages) == 2

        await agent.run("Second message")
        assert len(agent.messages) == 4

    @pytest.mark.asyncio
    async def test_system_prompt_passed_to_provider(self):
        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.return_value = Response(content="OK", tool_calls=[])

        agent = Agent(
            provider=mock_provider,
            system="Be concise.",
            max_tokens=1000,
            temperature=0.5,
        )
        await agent.run("Hello")

        call_kwargs = mock_provider.complete.call_args.kwargs
        assert call_kwargs["system"] == "Be concise."
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["temperature"] == 0.5


class Test_Tutorial6_ToolExecution:
    @pytest.mark.asyncio
    async def test_tool_execution_flow(self):
        def get_time() -> str:
            return "12:00 PM"

        t = Tool(name="get_time", description="Get current time", parameters={}, handler=get_time)

        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.side_effect = [
            Response(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="get_time", arguments={})],
            ),
            Response(content="The time is 12:00 PM", tool_calls=[]),
        ]

        agent = Agent(provider=mock_provider, tools=[t])
        response = await agent.run("What time is it?")

        assert response.content == "The time is 12:00 PM"
        assert mock_provider.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_result_in_history(self):
        def echo(text: str) -> str:
            return text.upper()

        t = Tool(name="echo", description="Echo text", parameters={}, handler=echo)

        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.side_effect = [
            Response(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="echo", arguments={"text": "hello"})],
            ),
            Response(content="Done", tool_calls=[]),
        ]

        agent = Agent(provider=mock_provider, tools=[t])
        await agent.run("Echo hello")

        tool_msg = next((m for m in agent.messages if m.tool_results), None)
        assert tool_msg is not None
        assert tool_msg.tool_results[0].content == "HELLO"

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self):
        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.return_value = Response(
            content=None,
            tool_calls=[ToolCall(id="tc_1", name="loop", arguments={})],
        )

        def loop_handler():
            return "looping"

        t = Tool(name="loop", description="", parameters={}, handler=loop_handler)
        agent = Agent(provider=mock_provider, tools=[t])

        await agent.run("Loop", max_iterations=3)
        assert mock_provider.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_async_tool_support(self):
        async def async_fetch(url: str) -> str:
            return f"Fetched: {url}"

        t = Tool(name="fetch", description="", parameters={}, handler=async_fetch)

        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.side_effect = [
            Response(
                content=None,
                tool_calls=[
                    ToolCall(id="tc_1", name="fetch", arguments={"url": "https://example.com"})
                ],
            ),
            Response(content="Done", tool_calls=[]),
        ]

        agent = Agent(provider=mock_provider, tools=[t])
        await agent.run("Fetch example.com")

        tool_msg = next((m for m in agent.messages if m.tool_results), None)
        assert "Fetched: https://example.com" in tool_msg.tool_results[0].content


class Test_Tutorial7_Streaming:
    @pytest.mark.asyncio
    async def test_streaming_text(self):
        async def mock_stream(*args, **kwargs):
            yield {"type": "text_delta", "text": "Hello"}
            yield {"type": "text_delta", "text": " World"}
            yield {"type": "done", "response": Response(content="Hello World", tool_calls=[])}

        mock_provider = MagicMock()
        mock_provider.name = "mock"
        mock_provider.stream = mock_stream

        agent = Agent(provider=mock_provider)

        chunks = []
        async for event in agent.stream("Hi"):
            if event["type"] == "text_delta":
                chunks.append(event["text"])

        assert chunks == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_streaming_with_tools(self):
        def calculator(expr: str) -> str:
            return str(eval(expr))

        t = Tool(name="calc", description="", parameters={}, handler=calculator)
        call_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                yield {"type": "tool_call_start", "id": "tc_1", "name": "calc"}
                yield {
                    "type": "tool_call_end",
                    "id": "tc_1",
                    "name": "calc",
                    "arguments": {"expr": "2+2"},
                }
                yield {
                    "type": "done",
                    "response": Response(
                        content=None,
                        tool_calls=[ToolCall(id="tc_1", name="calc", arguments={"expr": "2+2"})],
                    ),
                }
            else:
                yield {"type": "text_delta", "text": "The answer is 4"}
                yield {"type": "done", "response": Response(content="The answer is 4", tool_calls=[])}

        mock_provider = MagicMock()
        mock_provider.name = "mock"
        mock_provider.stream = mock_stream

        agent = Agent(provider=mock_provider, tools=[t])

        events = []
        async for event in agent.stream("What is 2+2?"):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "tool_call_start" in event_types
        assert "tool_result" in event_types
        assert "text_delta" in event_types


class Test_Tutorial8_ErrorHandling:
    @pytest.mark.asyncio
    async def test_tool_error_captured(self):
        def failing_tool() -> str:
            raise ValueError("Something went wrong!")

        t = Tool(name="fail", description="", parameters={}, handler=failing_tool)

        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.side_effect = [
            Response(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="fail", arguments={})],
            ),
            Response(content="I encountered an error", tool_calls=[]),
        ]

        agent = Agent(provider=mock_provider, tools=[t])
        await agent.run("Do the thing")

        tool_msg = next((m for m in agent.messages if m.tool_results), None)
        assert tool_msg.tool_results[0].is_error is True
        assert "Something went wrong" in tool_msg.tool_results[0].content

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.side_effect = [
            Response(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="nonexistent", arguments={})],
            ),
            Response(content="Tool not found", tool_calls=[]),
        ]

        agent = Agent(provider=mock_provider, tools=[])
        await agent.run("Use unknown tool")

        tool_msg = next((m for m in agent.messages if m.tool_results), None)
        assert tool_msg.tool_results[0].is_error is True
        assert "Unknown tool" in tool_msg.tool_results[0].content


class Test_Tutorial9_SyncAPI:
    def test_run_sync(self):
        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.return_value = Response(content="Hello!", tool_calls=[])

        agent = Agent(provider=mock_provider)
        response = agent.run_sync("Hi")

        assert response.content == "Hello!"

    def test_clear_messages(self):
        mock_provider = MagicMock()
        mock_provider.name = "mock"

        agent = Agent(provider=mock_provider)
        agent._messages.append(Message(role="user", content="test"))

        assert len(agent.messages) == 1
        agent.clear_messages()
        assert len(agent.messages) == 0

    def test_add_tool(self):
        mock_provider = MagicMock()
        mock_provider.name = "mock"

        @tool
        def new_tool() -> str:
            return "result"

        agent = Agent(provider=mock_provider, tools=[])
        assert len(agent.tools) == 0

        agent.add_tool(new_tool)
        assert len(agent.tools) == 1


class Test_Tutorial10_ResourceCleanup:
    @pytest.mark.asyncio
    async def test_context_manager(self):
        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.return_value = Response(content="Hello!", tool_calls=[])
        mock_provider.close = AsyncMock()

        async with Agent(provider=mock_provider) as agent:
            response = await agent.run("Hi")
            assert response.content == "Hello!"

        mock_provider.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_method(self):
        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.close = AsyncMock()

        agent = Agent(provider=mock_provider)
        await agent.close()

        mock_provider.close.assert_called_once()


class Test_Tutorial11_Vision:
    @pytest.mark.asyncio
    async def test_image_in_prompt(self):
        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.return_value = Response(
            content="I see an image", tool_calls=[]
        )

        agent = Agent(provider=mock_provider)
        response = await agent.run("What is this?", images=["https://example.com/image.png"])

        assert response.content == "I see an image"
        assert len(agent.messages) == 2

    @pytest.mark.asyncio
    async def test_image_with_data_uri(self):
        mock_provider = AsyncMock()
        mock_provider.name = "mock"
        mock_provider.complete.return_value = Response(content="Image processed", tool_calls=[])

        agent = Agent(provider=mock_provider)
        data_uri = "data:image/png;base64,iVBORw0KGgo="
        response = await agent.run("Describe this", images=[data_uri])

        assert response.content == "Image processed"
