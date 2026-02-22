import json
import time

import pytest

from barebone.providers import _decode_jwt_payload
from barebone.providers import _extract_codex_account_id
from barebone.providers import _to_tool
from barebone.providers import AnthropicProvider
from barebone.providers import OAuthCredentials
from barebone.providers import OpenAIProvider
from barebone.tools import tool
from barebone.types import Message
from barebone.types import Tool
from barebone.types import ToolCall
from barebone.types import ToolResult


class TestOAuthCredentials:
    def test_not_expired(self):
        creds = OAuthCredentials(
            access_token="token",
            refresh_token="refresh",
            expires_at=time.time() + 3600,
        )
        assert creds.is_expired is False

    def test_expired(self):
        creds = OAuthCredentials(
            access_token="token",
            refresh_token="refresh",
            expires_at=time.time() - 100,
        )
        assert creds.is_expired is True


class TestToTool:
    def test_with_tool_object(self):
        t = Tool(name="test", description="desc", parameters={}, handler=lambda: None)
        result = _to_tool(t)
        assert result is t

    def test_with_decorated_function(self):
        @tool
        def my_tool():
            """My description."""
            pass

        result = _to_tool(my_tool)
        assert result.name == "my_tool"
        assert result.description == "My description."

    def test_with_invalid_object(self):
        with pytest.raises(TypeError, match="Expected @tool decorated function"):
            _to_tool("not a tool")


class TestDecodeJwtPayload:
    def test_valid_jwt(self):
        # Create a simple test JWT (header.payload.signature)
        import base64

        payload = {"sub": "123", "exp": 9999999999}
        encoded_payload = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )
        token = f"header.{encoded_payload}.signature"

        result = _decode_jwt_payload(token)
        assert result["sub"] == "123"

    def test_invalid_jwt_format(self):
        assert _decode_jwt_payload("invalid") is None
        assert _decode_jwt_payload("only.two") is None

    def test_invalid_base64(self):
        assert _decode_jwt_payload("a.!!!invalid!!!.b") is None


class TestExtractCodexAccountId:
    def test_extracts_account_id(self):
        import base64

        payload = {"https://api.openai.com/auth": {"chatgpt_account_id": "acc_123"}}
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        token = f"header.{encoded}.sig"

        result = _extract_codex_account_id(token)
        assert result == "acc_123"

    def test_missing_claim(self):
        import base64

        payload = {"sub": "123"}
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        token = f"header.{encoded}.sig"

        result = _extract_codex_account_id(token)
        assert result is None


class TestAnthropicProviderInit:
    def test_requires_api_key_or_credentials(self):
        with pytest.raises(ValueError, match="Either api_key or credentials required"):
            AnthropicProvider()

    def test_with_api_key(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider._api_key == "test-key"
        assert provider._credentials is None

    def test_with_credentials(self):
        creds = OAuthCredentials(
            access_token="token",
            refresh_token="refresh",
            expires_at=time.time() + 3600,
        )
        provider = AnthropicProvider(credentials=creds)
        assert provider._credentials is creds
        assert provider._api_key is None

    def test_custom_model(self):
        provider = AnthropicProvider(api_key="key", model="claude-3-opus-20240229")
        assert provider._model == "claude-3-opus-20240229"


class TestAnthropicProviderMessageConversion:
    def setup_method(self):
        self.provider = AnthropicProvider(api_key="test-key")

    def test_simple_user_message(self):
        messages = [Message(role="user", content="Hello")]
        result = self.provider._to_api_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message(self):
        messages = [Message(role="assistant", content="Hi there!")]
        result = self.provider._to_api_messages(messages)
        assert result == [{"role": "assistant", "content": "Hi there!"}]

    def test_message_with_tool_calls(self):
        tc = ToolCall(id="tc_1", name="read", arguments={"path": "/test"})
        messages = [Message(role="assistant", content="Let me read", tool_calls=[tc])]
        result = self.provider._to_api_messages(messages)

        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "tool_use"
        assert result[0]["content"][1]["id"] == "tc_1"
        assert result[0]["content"][1]["name"] == "read"

    def test_message_with_tool_results(self):
        tr = ToolResult(id="tc_1", content="file contents", is_error=False)
        messages = [Message(role="user", tool_results=[tr])]
        result = self.provider._to_api_messages(messages)

        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "tool_result"
        assert result[0]["content"][0]["tool_use_id"] == "tc_1"
        assert result[0]["content"][0]["content"] == "file contents"


class TestAnthropicProviderToolConversion:
    def setup_method(self):
        self.provider = AnthropicProvider(api_key="test-key")

    def test_converts_tools(self):
        @tool
        def my_tool(x: int) -> int:
            """A test tool."""
            return x

        result = self.provider._to_api_tools([my_tool])
        assert len(result) == 1
        assert result[0]["name"] == "my_tool"
        assert result[0]["description"] == "A test tool."
        assert "input_schema" in result[0]

    def test_none_tools(self):
        result = self.provider._to_api_tools(None)
        assert result is None

    def test_empty_tools(self):
        result = self.provider._to_api_tools([])
        assert result is None


class TestOpenAIProviderInit:
    def test_basic_init(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider._api_key == "test-key"
        assert provider._model == "gpt-4o"
        assert provider._base_url == "https://api.openai.com/v1"

    def test_custom_config(self):
        provider = OpenAIProvider(
            api_key="key",
            model="gpt-4-turbo",
            base_url="https://custom.api.com/v1",
        )
        assert provider._model == "gpt-4-turbo"
        assert provider._base_url == "https://custom.api.com/v1"


class TestOpenAIProviderMessageConversion:
    def setup_method(self):
        self.provider = OpenAIProvider(api_key="test-key")

    def test_simple_message(self):
        messages = [Message(role="user", content="Hello")]
        result = self.provider._to_api_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_message_with_tool_calls(self):
        tc = ToolCall(id="tc_1", name="func", arguments={"x": 1})
        messages = [Message(role="assistant", content="Calling", tool_calls=[tc])]
        result = self.provider._to_api_messages(messages)

        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["type"] == "function"
        assert result[0]["tool_calls"][0]["function"]["name"] == "func"

    def test_message_with_tool_results(self):
        tr = ToolResult(id="tc_1", content="result")
        messages = [Message(role="user", tool_results=[tr])]
        result = self.provider._to_api_messages(messages)

        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tc_1"
        assert result[0]["content"] == "result"


class TestOpenAIProviderToolConversion:
    def setup_method(self):
        self.provider = OpenAIProvider(api_key="test-key")

    def test_converts_tools(self):
        @tool
        def my_tool(x: int) -> int:
            """A test tool."""
            return x

        result = self.provider._to_api_tools([my_tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "my_tool"
        assert result[0]["function"]["description"] == "A test tool."
