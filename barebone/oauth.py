from __future__ import annotations

import asyncio
import base64
import hashlib
import http.server
import json
import secrets
import socketserver
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlparse

import httpx

from barebone.types import NullableStr
from barebone.types import OAuthCredentials

CODEX_AUTH_FILE = Path.home() / ".codex" / "auth.json"
CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
CODEX_SCOPE = "openid profile email offline_access"
CODEX_JWT_CLAIM = "https://api.openai.com/auth"


def load_credentials(path: Path = CODEX_AUTH_FILE) -> OAuthCredentials:
    payload = json.loads(path.read_text())
    data = payload["tokens"]
    return OAuthCredentials(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        expires_at=data.get("expires_at", 0),
        account_id=data.get("account_id"),
    )


def save_credentials(credentials: OAuthCredentials, path: Path = CODEX_AUTH_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokens": {
            "access_token": credentials.access_token,
            "refresh_token": credentials.refresh_token,
            "expires_at": credentials.expires_at,
            "account_id": credentials.account_id,
        }
    }
    path.write_text(json.dumps(payload, indent=2))


@dataclass
class PKCEPair:
    verifier: str
    challenge: str


@dataclass
class AuthorizationFlow:
    verifier: str
    state: str
    url: str


@dataclass
class CallbackResult:
    code: NullableStr = None
    error: NullableStr = None


def generate_pkce() -> PKCEPair:
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return PKCEPair(verifier=verifier, challenge=challenge)


def create_state() -> str:
    return secrets.token_hex(16)


def decode_jwt_payload(token: str) -> dict[str, Any] | None:
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


def extract_account_id(access_token: str) -> NullableStr:
    payload = decode_jwt_payload(access_token)
    if not payload:
        return None
    auth_claim = payload.get(CODEX_JWT_CLAIM, {})
    account_id = auth_claim.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) else None


def create_authorization_flow(originator: str = "barebone") -> AuthorizationFlow:
    pkce = generate_pkce()
    state = create_state()

    params = {
        "response_type": "code",
        "client_id": CODEX_CLIENT_ID,
        "redirect_uri": CODEX_REDIRECT_URI,
        "scope": CODEX_SCOPE,
        "code_challenge": pkce.challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": originator,
    }

    url = f"{CODEX_AUTHORIZE_URL}?{urlencode(params)}"
    return AuthorizationFlow(verifier=pkce.verifier, state=state, url=url)


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    result: CallbackResult | None = None
    expected_state: str = ""

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        params = parse_qs(parsed.query)
        state = params.get("state", [None])[0]
        code = params.get("code", [None])[0]
        error = params.get("error", [None])[0]

        if state != self.expected_state:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"State mismatch")
            _CallbackHandler.result = CallbackResult(error="State mismatch")
            return

        if error:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"Error: {error}".encode())
            _CallbackHandler.result = CallbackResult(error=error)
            return

        if not code:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing authorization code")
            _CallbackHandler.result = CallbackResult(error="Missing code")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = b"""<!doctype html>
<html><head><title>Authentication successful</title></head>
<body><p>Authentication successful. Return to your terminal.</p></body></html>"""
        self.wfile.write(html)
        _CallbackHandler.result = CallbackResult(code=code)


class CallbackServer:
    def __init__(self, state: str, port: int = 1455):
        self._state = state
        self._port = port
        self._server: socketserver.TCPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        _CallbackHandler.result = None
        _CallbackHandler.expected_state = self._state

        try:
            self._server = socketserver.TCPServer(("127.0.0.1", self._port), _CallbackHandler)
            self._server.timeout = 1.0
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()
            return True
        except OSError:
            return False

    def _serve(self) -> None:
        if not self._server:
            return
        while _CallbackHandler.result is None:
            self._server.handle_request()

    def wait_for_code(self, timeout: float = 120.0) -> CallbackResult:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if _CallbackHandler.result is not None:
                return _CallbackHandler.result
            time.sleep(0.1)
        return CallbackResult(error="Timeout waiting for callback")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None


async def exchange_code_for_tokens(
    code: str,
    verifier: str,
    redirect_uri: str = CODEX_REDIRECT_URI,
) -> OAuthCredentials:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            CODEX_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CODEX_CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise ValueError(f"Token exchange failed: {response.text}")

        data = response.json()
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        expires_in = data.get("expires_in")

        if not access_token or not refresh_token or not isinstance(expires_in, int):
            raise ValueError(f"Invalid token response: {data}")

        account_id = extract_account_id(access_token)
        if not account_id:
            raise ValueError("Failed to extract account_id from token")

        return OAuthCredentials(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=time.time() + expires_in,
            account_id=account_id,
        )


async def login_openai_codex(
    on_auth_url: Callable[[str], None] | None = None,
    open_browser: bool = True,
    originator: str = "barebone",
) -> OAuthCredentials:
    flow = create_authorization_flow(originator=originator)
    server = CallbackServer(state=flow.state)

    if not server.start():
        raise RuntimeError("Failed to start callback server on port 1455")

    try:
        if on_auth_url:
            on_auth_url(flow.url)

        if open_browser:
            webbrowser.open(flow.url)

        result = await asyncio.to_thread(server.wait_for_code, 120.0)

        if result.error:
            raise ValueError(f"OAuth failed: {result.error}")

        if not result.code:
            raise ValueError("No authorization code received")

        return await exchange_code_for_tokens(result.code, flow.verifier)
    finally:
        server.stop()


__all__ = [
    "OAuthCredentials",
    "load_credentials",
    "save_credentials",
    "login_openai_codex",
    "exchange_code_for_tokens",
    "create_authorization_flow",
    "extract_account_id",
]
