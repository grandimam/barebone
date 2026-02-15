from __future__ import annotations

import anthropic
import base64
import httpx
import asyncio
import hashlib
import json
import secrets
import time
import webbrowser

from typing import Any
from typing import Callable

from collections.abc import AsyncIterator

from barebone.auth.dataclasses import OAuthCredentials
from barebone.auth.constants import CLIENT_ID
from barebone.auth.constants import AUTHORIZE_URL
from barebone.auth.constants import TOKEN_URL
from barebone.auth.constants import REDIRECT_URI
from barebone.auth.constants import SCOPES
from barebone.auth.constants import CLAUDE_CREDENTIALS_PATH

from barebone.agent.types import Done
from barebone.agent.types import Message
from barebone.agent.types import Response
from barebone.agent.types import StreamEvent
from barebone.agent.types import TextDelta
from barebone.agent.types import ToolCall
from barebone.agent.types import ToolCallEnd
from barebone.agent.types import ToolCallStart
from barebone.agent.types import Usage
from barebone.auth.base import Provider


class AnthropicProvider(Provider):

    CLAUDE_CODE_HEADERS = {
        "accept": "application/json",
        "anthropic-dangerous-direct-browser-access": "true",
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
        "user-agent": "claude-cli/2.1.2 (external, cli)",
        "x-app": "cli",
    }

    CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

    def __init__(self, oauth_token: str | None = None, api_key: str | None = None):
        self.is_oauth = oauth_token is not None and "sk-ant-oat" in oauth_token

        if self.is_oauth:
            self.client = anthropic.AsyncAnthropic(
                api_key=None,
                auth_token=oauth_token,
                default_headers=self.CLAUDE_CODE_HEADERS,
            )
        else:
            self.client = anthropic.AsyncAnthropic(api_key=api_key or oauth_token)

    @property
    def name(self) -> str:
        return "anthropic"

    def _build_system(self, system: str | None) -> list[dict[str, str]]:
        blocks = []

        if self.is_oauth:
            blocks.append({"type": "text", "text": self.CLAUDE_CODE_IDENTITY})

        if system:
            blocks.append({"type": "text", "text": system})

        return blocks if blocks else None

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result = []

        for msg in messages:
            if msg.role == "tool_result":
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": str(msg.content),
                        }
                    ],
                })
            elif msg.role == "assistant" and isinstance(msg.content, list):
                content = []
                for block in msg.content:
                    if isinstance(block, dict):
                        content.append(block)
                    elif isinstance(block, ToolCall):
                        content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.arguments,
                        })
                result.append({"role": "assistant", "content": content})
            else:
                result.append({"role": msg.role, "content": msg.content})

        return result

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None

        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
            }
            for tool in tools
        ]

    def _parse_response(self, response: anthropic.types.Message, model: str) -> Response:
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
            content=content,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "stop",
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            model=model,
            provider=self.name,
        )

    async def complete(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> Response:
        params: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
        }

        system_blocks = self._build_system(system)
        if system_blocks:
            params["system"] = system_blocks

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            params["tools"] = converted_tools

        if temperature is not None:
            params["temperature"] = temperature

        response = await self.client.messages.create(**params)
        return self._parse_response(response, model)

    async def stream(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        params: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
        }

        system_blocks = self._build_system(system)
        if system_blocks:
            params["system"] = system_blocks

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            params["tools"] = converted_tools

        if temperature is not None:
            params["temperature"] = temperature

        content = ""
        tool_calls: list[ToolCall] = []
        current_tool: dict[str, Any] | None = None
        usage = Usage()

        async with self.client.messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        usage.input_tokens = event.message.usage.input_tokens

                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "arguments_json": "",
                        }
                        yield ToolCallStart(
                            id=event.content_block.id,
                            name=event.content_block.name,
                        )

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        content += event.delta.text
                        yield TextDelta(text=event.delta.text)
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
                        yield ToolCallEnd(
                            id=tc.id,
                            name=tc.name,
                            arguments=tc.arguments,
                        )
                        current_tool = None

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        usage.output_tokens = event.usage.output_tokens
                        usage.total_tokens = usage.input_tokens + usage.output_tokens

        yield Done(
            response=Response(
                content=content,
                tool_calls=tool_calls,
                stop_reason="stop",
                usage=usage,
                model=model,
                provider=self.name,
            )
        )


class TokenManager:

    def __init__(
        self,
        credentials: OAuthCredentials | None = None,
        credentials_path: Any | None = None,
        on_refresh: Callable[[OAuthCredentials], None] | None = None,
    ):
        self._credentials = credentials
        self._credentials_path = credentials_path
        self._on_refresh = on_refresh
        self._refresh_lock: asyncio.Lock | None = None
        self._http_client: httpx.AsyncClient | None = None

    @classmethod
    def auto(cls) -> TokenManager:
        if creds := _read_claude_credentials():
            return cls(credentials=creds)
        
        return cls(credentials=None)

    @classmethod
    def login(
        cls,
        open_browser: bool = True,
        on_auth_url: Callable[[str], None] | None = None,
    ) -> TokenManager:
        credentials = asyncio.get_event_loop().run_until_complete(_perform_login(open_browser, on_auth_url))
        manager = cls(credentials=credentials)
        manager._save_credentials()
        return manager

    @property
    def has_credentials(self) -> bool:
        return self._credentials is not None

    @property
    def credentials(self) -> OAuthCredentials | None:
        return self._credentials

    async def get_token(self) -> str:
        if self._credentials is None:
            raise ValueError(
                "No credentials available. Call TokenManager.login() or ensure "
                "Claude Code is installed and authenticated."
            )

        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()

        async with self._refresh_lock:
            if self._credentials.is_expired:
                await self._refresh_token()

        return self._credentials.access_token

    def get_token_sync(self) -> str:
        return asyncio.get_event_loop().run_until_complete(self.get_token())

    async def _refresh_token(self) -> None:
        if self._credentials is None:
            raise ValueError("No credentials to refresh")

        if self._http_client is None:
            self._http_client = httpx.AsyncClient()

        response = await self._http_client.post(
            TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": self._credentials.refresh_token,
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Token refresh failed: {response.text}",
                request=response.request,
                response=response,
            )

        data = response.json()
        self._credentials = OAuthCredentials(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=time.time() + data["expires_in"] - 300,
        )

        self._save_credentials()
        _write_claude_credentials(self._credentials)

        if self._on_refresh:
            self._on_refresh(self._credentials)

    def _save_credentials(self) -> None:
        if not self._credentials:
            return

        self._credentials_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "anthropic": self._credentials.to_dict(),
            "version": 1,
        }

        with open(self._credentials_path, "w") as f:
            json.dump(data, f, indent=2)

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


def _read_claude_credentials() -> OAuthCredentials | None:
    if not CLAUDE_CREDENTIALS_PATH.exists():
        return None

    try:
        with open(CLAUDE_CREDENTIALS_PATH) as f:
            data = json.load(f)

        oauth = data.get("claudeAiOauth")
        if not oauth:
            return None

        access_token = oauth.get("accessToken")
        refresh_token = oauth.get("refreshToken")
        expires_at = oauth.get("expiresAt")

        if not access_token or not refresh_token or not expires_at:
            return None

        # expiresAt is in milliseconds
        return OAuthCredentials(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at / 1000,
        )

    except Exception:
        return None


def _write_claude_credentials(credentials: OAuthCredentials) -> bool:
    if not CLAUDE_CREDENTIALS_PATH.exists():
        return False

    try:
        with open(CLAUDE_CREDENTIALS_PATH) as f:
            data = json.load(f)

        if "claudeAiOauth" not in data:
            return False

        data["claudeAiOauth"]["accessToken"] = credentials.access_token
        data["claudeAiOauth"]["refreshToken"] = credentials.refresh_token
        data["claudeAiOauth"]["expiresAt"] = int(credentials.expires_at * 1000)

        with open(CLAUDE_CREDENTIALS_PATH, "w") as f:
            json.dump(data, f, indent=2)

        return True

    except Exception:
        return False


def _generate_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(32)
    challenge_bytes = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()
    return verifier, challenge


async def _perform_login(
    open_browser: bool = True,
    on_auth_url: Callable[[str], None] | None = None,
) -> OAuthCredentials:
    verifier, challenge = _generate_pkce()
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    auth_url = f"{AUTHORIZE_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

    if on_auth_url:
        on_auth_url(auth_url)
    elif open_browser:
        print(f"\nOpening browser for authentication...")
        print(f"If browser doesn't open, visit:\n{auth_url}\n")
        webbrowser.open(auth_url)
    else:
        print(f"\nVisit this URL to authenticate:\n{auth_url}\n")

    print("After authorizing, paste the code here (format: code#state):")
    auth_code = input("> ").strip()

    if "#" not in auth_code:
        raise ValueError("Invalid authorization code format. Expected: code#state")

    code, state = auth_code.split("#", 1)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": REDIRECT_URI,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise ValueError(f"Token exchange failed: {response.text}")

        data = response.json()

    return OAuthCredentials(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        expires_at=time.time() + data["expires_in"] - 300,
    )


def get_auth_url() -> tuple[str, str]:
    verifier, challenge = _generate_pkce()

    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }

    auth_url = f"{AUTHORIZE_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    return auth_url, verifier


async def exchange_code(auth_code: str, verifier: str) -> OAuthCredentials:
    parts = auth_code.split("#")
    code = parts[0]
    state = parts[1] if len(parts) > 1 else verifier

    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": REDIRECT_URI,
                "code_verifier": verifier,
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Token exchange failed: {response.text}")

        data = response.json()

        return OAuthCredentials(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=time.time() + data["expires_in"] - 300,
        )


async def refresh_token(refresh: str) -> OAuthCredentials:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh,
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Token refresh failed: {response.text}")

        data = response.json()

        return OAuthCredentials(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=time.time() + data["expires_in"] - 300,
        )


def is_token_expired(credentials: OAuthCredentials) -> bool:
    return credentials.is_expired


async def ensure_valid_token(credentials: OAuthCredentials) -> OAuthCredentials:
    if credentials.is_expired:
        return await refresh_token(credentials.refresh_token)
    return credentials
