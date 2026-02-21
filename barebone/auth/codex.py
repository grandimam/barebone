"""
Codex OAuth Provider for barebone.

Based on pi-ai's openai-codex-responses.ts implementation:
- Base URL: https://chatgpt.com/backend-api
- Endpoint: /codex/responses
- Headers: Authorization, chatgpt-account-id, OpenAI-Beta
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from barebone.auth.base import Provider
from barebone.common.dataclasses import (
    Done,
    Message,
    Response,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)

# Constants from pi-ai
CODEX_HOME = Path.home() / ".codex"
CODEX_AUTH_FILE = CODEX_HOME / "auth.json"
CODEX_KEYCHAIN_SERVICE = "Codex Auth"
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_ENDPOINT = "/codex/responses"
JWT_CLAIM_PATH = "https://api.openai.com/auth"


@dataclass
class CodexCredentials:
    """Codex OAuth credentials."""

    access_token: str
    refresh_token: str
    account_id: str | None = None
    expires_at: float | None = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "account_id": self.account_id,
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodexCredentials:
        tokens = data.get("tokens", {})
        return cls(
            access_token=tokens.get("access_token", ""),
            refresh_token=tokens.get("refresh_token", ""),
            account_id=tokens.get("account_id"),
        )


def _extract_account_id_from_jwt(token: str) -> str | None:
    """Extract chatgpt_account_id from JWT token payload."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        # Decode the payload (second part)
        payload_b64 = parts[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload_json = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_json)
        # Extract account ID from the claim path
        auth_claim = payload.get(JWT_CLAIM_PATH, {})
        return auth_claim.get("chatgpt_account_id")
    except Exception:
        return None


def _compute_keychain_account() -> str:
    """Compute keychain account name from Codex home path."""
    codex_home = str(CODEX_HOME)
    account_hash = hashlib.sha256(codex_home.encode()).hexdigest()[:16]
    return f"cli|{account_hash}"


def _read_keychain_credentials() -> CodexCredentials | None:
    """Read Codex credentials from macOS Keychain."""
    if platform.system() != "Darwin":
        return None

    account = _compute_keychain_account()

    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                CODEX_KEYCHAIN_SERVICE,
                "-a",
                account,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout.strip())
        tokens = data.get("tokens", {})

        access_token = tokens.get("access_token")
        if not access_token:
            return None

        # Extract account_id from JWT if not in stored data
        account_id = tokens.get("account_id") or _extract_account_id_from_jwt(access_token)

        # Compute expiry from last_refresh
        last_refresh_raw = data.get("last_refresh")
        if last_refresh_raw:
            try:
                from datetime import datetime
                last_refresh = datetime.fromisoformat(
                    last_refresh_raw.replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                last_refresh = time.time()
        else:
            last_refresh = time.time()

        expires_at = last_refresh + 3600

        return CodexCredentials(
            access_token=access_token,
            refresh_token=tokens.get("refresh_token", ""),
            account_id=account_id,
            expires_at=expires_at,
        )
    except Exception:
        return None


def _read_file_credentials() -> CodexCredentials | None:
    """Read Codex credentials from ~/.codex/auth.json file."""
    if not CODEX_AUTH_FILE.exists():
        return None

    try:
        with open(CODEX_AUTH_FILE) as f:
            data = json.load(f)

        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")

        if not access_token:
            return None

        # Extract account_id from JWT if not in stored data
        account_id = tokens.get("account_id") or _extract_account_id_from_jwt(access_token)

        try:
            mtime = CODEX_AUTH_FILE.stat().st_mtime
            expires_at = mtime + 3600
        except Exception:
            expires_at = time.time() + 3600

        return CodexCredentials(
            access_token=access_token,
            refresh_token=tokens.get("refresh_token", ""),
            account_id=account_id,
            expires_at=expires_at,
        )
    except Exception:
        return None


def read_codex_credentials() -> CodexCredentials | None:
    """Read Codex credentials from keychain or file."""
    creds = _read_keychain_credentials()
    if creds:
        return creds
    return _read_file_credentials()


class CodexTokenManager:
    """Manages Codex OAuth token lifecycle."""

    def __init__(
        self,
        credentials: CodexCredentials | None = None,
        on_refresh: Callable[[CodexCredentials], None] | None = None,
    ):
        self._credentials = credentials
        self._on_refresh = on_refresh
        self._refresh_lock: asyncio.Lock | None = None

    @classmethod
    def auto(cls) -> CodexTokenManager:
        """Create a TokenManager, auto-loading credentials from Codex CLI."""
        creds = read_codex_credentials()
        return cls(credentials=creds)

    @property
    def has_credentials(self) -> bool:
        return self._credentials is not None

    @property
    def credentials(self) -> CodexCredentials | None:
        return self._credentials

    async def get_token(self) -> str:
        """Get a valid access token."""
        if self._credentials is None:
            raise ValueError("No credentials. Run 'codex login' first.")

        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()

        async with self._refresh_lock:
            if self._credentials.is_expired:
                raise ValueError("Codex token expired. Run 'codex login' to re-authenticate.")

        return self._credentials.access_token

    async def close(self) -> None:
        pass


class CodexProvider(Provider):
    """
    Provider for ChatGPT Codex Responses API.

    Based on pi-ai's openai-codex-responses.ts:
    - Base URL: https://chatgpt.com/backend-api
    - Endpoint: /codex/responses
    """

    name = "openai-codex"

    def __init__(
        self,
        credentials: CodexCredentials | None = None,
        token_manager: CodexTokenManager | None = None,
        base_url: str = CODEX_BASE_URL,
    ):
        if token_manager:
            self._token_manager = token_manager
        elif credentials:
            self._token_manager = CodexTokenManager(credentials=credentials)
        else:
            self._token_manager = CodexTokenManager.auto()

        if not self._token_manager.has_credentials:
            raise ValueError("No Codex credentials. Run 'codex login' to authenticate.")

        self._base_url = base_url
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    def _build_headers(self, token: str, account_id: str | None) -> dict[str, str]:
        """Build headers for Codex API (from pi-ai)."""
        headers = {
            "Authorization": f"Bearer {token}",
            "OpenAI-Beta": "responses=experimental",
            "originator": "pi",
            "User-Agent": f"barebone ({platform.system()} {platform.release()}; {platform.machine()})",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        if account_id:
            headers["chatgpt-account-id"] = account_id
        return headers

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert barebone messages to Codex Responses API format."""
        result = []
        for msg in messages:
            if msg.role == "tool_result":
                result.append({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": str(msg.content),
                })
            elif msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            result.append({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": block.get("text", "")}],
                            })
                    elif isinstance(block, ToolCall):
                        result.append({
                            "type": "function_call",
                            "call_id": block.id,
                            "name": block.name,
                            "arguments": json.dumps(block.arguments),
                        })
            elif msg.role == "user":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                result.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}],
                })
            elif msg.role == "assistant":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                result.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })
        return result

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert barebone tools to Codex format."""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            }
            for tool in tools
        ]

    async def complete(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> Response:
        """
        Complete using ChatGPT Codex Responses API.

        Note: Codex API only supports streaming, so we collect the full response.
        """
        content = ""
        tool_calls: list[ToolCall] = []
        usage = Usage()
        stop_reason = "stop"

        async for event in self.stream(model, messages, system, tools, max_tokens, temperature):
            if isinstance(event, TextDelta):
                content += event.text
            elif isinstance(event, ToolCallEnd):
                tool_calls.append(ToolCall(id=event.id, name=event.name, arguments=event.arguments))
            elif isinstance(event, Done):
                usage = event.response.usage
                stop_reason = event.response.stop_reason

        return Response(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=model,
            provider=self.name,
        )

    async def stream(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        token = await self._token_manager.get_token()
        creds = self._token_manager.credentials
        account_id = creds.account_id if creds else None

        if not account_id:
            account_id = _extract_account_id_from_jwt(token)

        client = await self._get_client()
        headers = self._build_headers(token, account_id)

        instructions = system or "You are a helpful assistant."

        payload: dict[str, Any] = {
            "model": model,
            "store": False,
            "stream": True,
            "instructions": instructions,
            "input": self._convert_messages(messages),
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
        }

        if tools:
            payload["tools"] = self._convert_tools(tools)

        if temperature is not None:
            payload["temperature"] = temperature

        url = f"{self._base_url}{CODEX_ENDPOINT}"

        content = ""
        tool_calls: list[ToolCall] = []
        current_tool: dict[str, Any] | None = None
        usage = Usage()

        async with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code == 401:
                raise ValueError("Codex token expired. Run 'codex login'.")
            if response.status_code == 429:
                raise ValueError("ChatGPT usage limit reached.")

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
                    yield TextDelta(text=delta_text)

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
                        yield ToolCallStart(id=current_tool["id"], name=current_tool["name"])

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
                        yield ToolCallEnd(id=tc.id, name=tc.name, arguments=tc.arguments)
                        current_tool = None

                elif event_type in ("response.done", "response.completed"):
                    resp = event.get("response", {})
                    usage_data = resp.get("usage", {})
                    usage = Usage(
                        input_tokens=usage_data.get("input_tokens", 0),
                        output_tokens=usage_data.get("output_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )

                elif event_type == "error":
                    msg = event.get("message", "Codex error")
                    raise ValueError(f"Codex error: {msg}")

                elif event_type == "response.failed":
                    msg = event.get("response", {}).get("error", {}).get("message", "Request failed")
                    raise ValueError(f"Codex failed: {msg}")

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

    async def get_usage(self) -> dict[str, Any]:
        """Get Codex usage stats from ChatGPT backend API."""
        token = await self._token_manager.get_token()
        creds = self._token_manager.credentials
        account_id = creds.account_id if creds else _extract_account_id_from_jwt(token)

        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "barebone",
        }
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id

        response = await client.get(f"{CODEX_BASE_URL}/wham/usage", headers=headers)

        if response.status_code in (401, 403):
            raise ValueError("Token expired or unauthorized")

        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


def run_codex_exec(
    prompt: str,
    model: str = "gpt-5-codex",
    sandbox: str = "workspace-write",
    timeout: int = 120,
    json_output: bool = True,
) -> str:
    """Run Codex non-interactively via subprocess."""
    cmd = ["codex", "exec", "--full-auto", "-m", model, "-s", sandbox]
    if json_output:
        cmd.append("--json")
    cmd.append(prompt)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Codex exec failed: {result.stderr}")
    return result.stdout


async def run_codex_exec_async(
    prompt: str,
    model: str = "gpt-5-codex",
    sandbox: str = "workspace-write",
    timeout: int = 120,
    json_output: bool = True,
) -> str:
    """Async version of run_codex_exec."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_codex_exec(prompt, model, sandbox, timeout, json_output),
    )
