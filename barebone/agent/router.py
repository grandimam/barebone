from typing import Any

from barebone.auth.anthropic import AnthropicProvider
from barebone.auth.openrouter import OpenRouterProvider
from barebone.auth.base import Provider
from barebone.agent.types import Message, Response


class Router:

    def __init__(
        self,
        anthropic_oauth: str | None = None,
        anthropic_api_key: str | None = None,
        openrouter: str | None = None,
    ):
        self._providers: dict[str, Provider] = {}

        if anthropic_oauth or anthropic_api_key:
            self._providers["anthropic"] = AnthropicProvider(
                oauth_token=anthropic_oauth,
                api_key=anthropic_api_key,
            )

        if openrouter:
            self._providers["openrouter"] = OpenRouterProvider(api_key=openrouter)

    def get_provider(self, model: str) -> tuple[Provider, str]:
        model_lower = model.lower()

        if model_lower.startswith("claude-") or model.startswith("anthropic/"):
            if "anthropic" not in self._providers:
                raise ValueError("No Anthropic provider configured")
            model_id = model.removeprefix("anthropic/")
            return self._providers["anthropic"], model_id

        if "openrouter" not in self._providers:
            raise ValueError("No OpenRouter provider configured")
        return self._providers["openrouter"], model

    async def complete(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ) -> Response:
        provider, model_id = self.get_provider(model)
        return await provider.complete(
            model=model_id,
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def stream(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
    ):
        provider, model_id = self.get_provider(model)
        async for event in provider.stream(
            model=model_id,
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield event

    @property
    def providers(self) -> dict[str, Provider]:
        return self._providers.copy()


__all__ = ["Router"]
