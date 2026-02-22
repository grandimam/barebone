from __future__ import annotations

import asyncio
import base64
import json
import time
from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import anthropic
import httpx

from barebone.types import ImageContent
from barebone.types import Message
from barebone.types import Response
from barebone.types import TextContent
from barebone.types import Tool
from barebone.types import ToolCall

ANTHROPIC_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
ANTHROPIC_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_API_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_JWT_CLAIM = "https://api.openai.com/auth"


@dataclass
class OAuthCredentials:
    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str | None = None

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at


def _to_tool(t: object) -> Tool:
    if isinstance(t, Tool):
        return t
    if hasattr(t, "to_tool"):
        return t.to_tool()
    raise TypeError(f"Expected @tool decorated function, got {type(t)}")


def _decode_jwt_payload(token: str) -> dict[str, Any] | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except Exception:
        return None


def _extract_codex_account_id(token: str) -> str | None:
    payload = _decode_jwt_payload(token)
    if not payload:
        return None
    auth_claim = payload.get(CODEX_JWT_CLAIM, {})
    account_id = auth_claim.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) else None


class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> Response: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> AsyncIterator[Any]: ...


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        credentials: OAuthCredentials | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        if not api_key and not credentials:
            raise ValueError("Either api_key or credentials required")

        self._model = model
        self._api_key = api_key
        self._credentials = credentials
        self._refresh_lock = asyncio.Lock()
        self._http_client: httpx.AsyncClient | None = None
        self._client: anthropic.AsyncAnthropic | None = None

    async def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._api_key:
            if not self._client:
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            return self._client

        # OAuth flow
        await self._ensure_valid_token()
        if not self._client or self._credentials_changed:
            self._client = anthropic.AsyncAnthropic(
                auth_token=self._credentials.access_token,
                default_headers={"anthropic-beta": "oauth-2025-04-20"},
            )
            self._credentials_changed = False
        return self._client

    _credentials_changed: bool = False

    async def _ensure_valid_token(self) -> None:
        if not self._credentials:
            return
        if not self._credentials.is_expired:
            return

        async with self._refresh_lock:
            if not self._credentials.is_expired:
                return
            await self._refresh_token()

    async def _refresh_token(self) -> None:
        if not self._credentials:
            raise ValueError("No credentials to refresh")

        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=30.0)

        response = await self._http_client.post(
            ANTHROPIC_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "client_id": ANTHROPIC_CLIENT_ID,
                "refresh_token": self._credentials.refresh_token,
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise ValueError(f"Anthropic token refresh failed: {response.text}")

        data = response.json()
        self._credentials = OAuthCredentials(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=time.time() + data["expires_in"] - 300,
        )
        self._credentials_changed = True

    def _convert_content(self, content: str | list | None) -> str | list[dict[str, Any]]:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        result = []
        for item in content:
            if isinstance(item, TextContent):
                result.append({"type": "text", "text": item.text})
            elif isinstance(item, ImageContent):
                source = item.source
                if source.startswith("data:"):
                    parts = source.split(",", 1)
                    media_type = parts[0].split(":")[1].split(";")[0]
                    data = parts[1] if len(parts) > 1 else ""
                    result.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    })
                else:
                    result.append({
                        "type": "image",
                        "source": {"type": "url", "url": source},
                    })
        return result

    def _to_api_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result = []
        for msg in messages:
            if msg.tool_results:
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tr.id,
                                "content": tr.content,
                                "is_error": tr.is_error,
                            }
                            for tr in msg.tool_results
                        ],
                    }
                )
            elif msg.tool_calls:
                content = []
                if msg.content:
                    if isinstance(msg.content, str):
                        content.append({"type": "text", "text": msg.content})
                    else:
                        content.extend(self._convert_content(msg.content))
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                result.append({"role": "assistant", "content": content})
            else:
                result.append({
                    "role": msg.role,
                    "content": self._convert_content(msg.content),
                })
        return result

    def _to_api_tools(self, tools: list[object] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        resolved = [_to_tool(t) for t in tools]
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in resolved
        ]

    def _parse_response(self, response: Any) -> Response:
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )
        return Response(
            content=content or None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "stop",
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> Response:
        client = await self._get_client()
        params: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_api_messages(messages),
            "max_tokens": max_tokens,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = self._to_api_tools(tools)
        if temperature is not None:
            params["temperature"] = temperature

        response = await client.messages.create(**params)
        return self._parse_response(response)

    async def stream(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> AsyncIterator[Any]:
        client = await self._get_client()
        params: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_api_messages(messages),
            "max_tokens": max_tokens,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = self._to_api_tools(tools)
        if temperature is not None:
            params["temperature"] = temperature

        content = ""
        tool_calls: list[ToolCall] = []
        current_tool: dict[str, Any] | None = None

        async with client.messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "arguments_json": "",
                        }
                        yield {
                            "type": "tool_call_start",
                            "id": current_tool["id"],
                            "name": current_tool["name"],
                        }
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        content += event.delta.text
                        yield {"type": "text_delta", "text": event.delta.text}
                    elif event.delta.type == "input_json_delta" and current_tool:
                        current_tool["arguments_json"] += event.delta.partial_json
                elif event.type == "content_block_stop":
                    if current_tool:
                        try:
                            arguments = json.loads(current_tool["arguments_json"] or "{}")
                        except json.JSONDecodeError:
                            arguments = {}
                        tc = ToolCall(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            arguments=arguments,
                        )
                        tool_calls.append(tc)
                        yield {
                            "type": "tool_call_end",
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        current_tool = None

        yield {
            "type": "done",
            "response": Response(
                content=content or None,
                tool_calls=tool_calls,
                stop_reason="stop",
            ),
        }

    async def close(self) -> None:
        if self._client:
            await self._client.close()
        if self._http_client:
            await self._http_client.aclose()


class CodexProvider(BaseProvider):
    name = "codex"

    def __init__(
        self,
        credentials: OAuthCredentials,
        model: str = "gpt-4.1",
    ):
        self._credentials = credentials
        self._model = model
        self._refresh_lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def _ensure_valid_token(self) -> None:
        if not self._credentials.is_expired:
            return

        async with self._refresh_lock:
            if not self._credentials.is_expired:
                return
            await self._refresh_token()

    async def _refresh_token(self) -> None:
        client = await self._get_client()

        response = await client.post(
            CODEX_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._credentials.refresh_token,
                "client_id": CODEX_CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise ValueError(f"Codex token refresh failed: {response.text}")

        data = response.json()
        account_id = _extract_codex_account_id(data["access_token"])

        self._credentials = OAuthCredentials(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=time.time() + data["expires_in"],
            account_id=account_id,
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._credentials.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self._credentials.account_id:
            headers["chatgpt-account-id"] = self._credentials.account_id
        return headers

    def _to_api_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result = []
        for msg in messages:
            if msg.tool_results:
                for tr in msg.tool_results:
                    result.append(
                        {
                            "type": "function_call_output",
                            "call_id": tr.id,
                            "output": tr.content,
                        }
                    )
            elif msg.tool_calls:
                for tc in msg.tool_calls:
                    result.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    )
                if msg.content:
                    result.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": msg.content}],
                        }
                    )
            elif msg.role == "user":
                result.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": msg.content or ""}],
                    }
                )
            elif msg.role == "assistant":
                result.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": msg.content or ""}],
                    }
                )
        return result

    def _to_api_tools(self, tools: list[object] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        resolved = [_to_tool(t) for t in tools]
        return [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in resolved
        ]

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> Response:
        content = ""
        tool_calls: list[ToolCall] = []

        async for event in self.stream(messages, tools, system, max_tokens, temperature):
            if event.get("type") == "text_delta":
                content += event.get("text", "")
            elif event.get("type") == "tool_call_end":
                tool_calls.append(
                    ToolCall(
                        id=event["id"],
                        name=event["name"],
                        arguments=event["arguments"],
                    )
                )
            elif event.get("type") == "done":
                return event["response"]

        return Response(content=content or None, tool_calls=tool_calls)

    async def stream(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> AsyncIterator[Any]:
        await self._ensure_valid_token()

        client = await self._get_client()
        headers = self._build_headers()

        payload: dict[str, Any] = {
            "model": self._model,
            "store": False,
            "stream": True,
            "instructions": system or "You are a helpful assistant.",
            "input": self._to_api_messages(messages),
        }
        if tools:
            payload["tools"] = self._to_api_tools(tools)
        if temperature is not None:
            payload["temperature"] = temperature

        content = ""
        tool_calls: list[ToolCall] = []
        current_tool: dict[str, Any] | None = None

        async with client.stream("POST", CODEX_API_URL, json=payload, headers=headers) as response:
            if response.status_code == 401:
                await self._refresh_token()
                async for event in self.stream(messages, tools, system, max_tokens, temperature):
                    yield event
                return
            if response.status_code == 429:
                raise ValueError("ChatGPT usage limit reached")
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "response.output_text.delta":
                    delta_text = event.get("delta", "")
                    content += delta_text
                    yield {"type": "text_delta", "text": delta_text}

                elif event_type == "response.function_call_arguments.delta":
                    if current_tool:
                        current_tool["arguments_json"] += event.get("delta", "")

                elif event_type == "response.output_item.added":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        current_tool = {
                            "id": item.get("call_id", ""),
                            "name": item.get("name", ""),
                            "arguments_json": "",
                        }
                        yield {
                            "type": "tool_call_start",
                            "id": current_tool["id"],
                            "name": current_tool["name"],
                        }

                elif event_type == "response.output_item.done":
                    item = event.get("item", {})
                    if item.get("type") == "function_call" and current_tool:
                        try:
                            arguments = json.loads(current_tool["arguments_json"] or "{}")
                        except json.JSONDecodeError:
                            arguments = {}
                        tc = ToolCall(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            arguments=arguments,
                        )
                        tool_calls.append(tc)
                        yield {
                            "type": "tool_call_end",
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        current_tool = None

                elif event_type == "error":
                    raise ValueError(f"Codex error: {event.get('message', 'Unknown')}")

                elif event_type == "response.failed":
                    err = event.get("response", {}).get("error", {})
                    msg = err.get("message", "Request failed")
                    raise ValueError(f"Codex failed: {msg}")

        yield {
            "type": "done",
            "response": Response(
                content=content or None,
                tool_calls=tool_calls,
                stop_reason="stop",
            ),
        }

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
    ):
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    def _convert_content(self, content: str | list | None) -> str | list[dict[str, Any]]:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        result = []
        for item in content:
            if isinstance(item, TextContent):
                result.append({"type": "text", "text": item.text})
            elif isinstance(item, ImageContent):
                result.append({
                    "type": "image_url",
                    "image_url": {"url": item.source},
                })
        return result

    def _to_api_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result = []
        for msg in messages:
            if msg.tool_results:
                for tr in msg.tool_results:
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.id,
                            "content": tr.content,
                        }
                    )
            elif msg.tool_calls:
                result.append(
                    {
                        "role": "assistant",
                        "content": msg.content if isinstance(msg.content, str) else "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
            else:
                result.append({
                    "role": msg.role,
                    "content": self._convert_content(msg.content),
                })
        return result

    def _to_api_tools(self, tools: list[object] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        resolved = [_to_tool(t) for t in tools]
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in resolved
        ]

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> Response:
        client = await self._get_client()

        api_messages = self._to_api_messages(messages)
        if system:
            api_messages.insert(0, {"role": "system", "content": system})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = self._to_api_tools(tools)
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        response = await client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        message = choice["message"]

        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                )

        return Response(
            content=message.get("content"),
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason", "stop"),
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> AsyncIterator[Any]:
        client = await self._get_client()

        api_messages = self._to_api_messages(messages)
        if system:
            api_messages.insert(0, {"role": "system", "content": system})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = self._to_api_tools(tools)
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        content = ""
        tool_calls: list[ToolCall] = []
        tool_call_map: dict[int, dict[str, Any]] = {}

        async with client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta = event.get("choices", [{}])[0].get("delta", {})

                if delta.get("content"):
                    content += delta["content"]
                    yield {"type": "text_delta", "text": delta["content"]}

                if delta.get("tool_calls"):
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta["index"]
                        if idx not in tool_call_map:
                            tool_call_map[idx] = {
                                "id": tc_delta.get("id", ""),
                                "name": tc_delta.get("function", {}).get("name", ""),
                                "arguments_json": "",
                            }
                            yield {
                                "type": "tool_call_start",
                                "id": tool_call_map[idx]["id"],
                                "name": tool_call_map[idx]["name"],
                            }
                        if tc_delta.get("function", {}).get("arguments"):
                            tool_call_map[idx]["arguments_json"] += tc_delta["function"][
                                "arguments"
                            ]

        for tc_data in tool_call_map.values():
            try:
                arguments = json.loads(tc_data["arguments_json"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tc = ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=arguments)
            tool_calls.append(tc)
            yield {
                "type": "tool_call_end",
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            }

        yield {
            "type": "done",
            "response": Response(
                content=content or None,
                tool_calls=tool_calls,
                stop_reason="stop",
            ),
        }

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


__all__ = [
    "BaseProvider",
    "AnthropicProvider",
    "CodexProvider",
    "OpenAIProvider",
    "OAuthCredentials",
]
