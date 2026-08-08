from __future__ import annotations

from typing import Optional

import httpx

from src.agents.providers.openai_compatible import OpenAICompatibleProvider
from src.schemas.providers import ProviderMetadata, ProviderModel, ProviderType


class AI302Provider(OpenAICompatibleProvider):
    """302.AI is an OpenAI-compatible aggregator.

    302.AI's backend is a proxy over many upstream providers and does not
    reliably expose a standard ``/v1/models`` listing to user API keys (the
    official integration ships a static catalog instead). So we first try the
    live endpoint, and if it is unavailable we fall back to a curated catalog
    of the models 302.AI exposes (sourced from models.dev).
    """

    # Curated catalog of models exposed through 302.AI's OpenAI-compatible
    # endpoint. ``supports_tools`` reflects function-calling availability.
    CATALOG_MODELS: list[ProviderModel] = []

    def __init__(self, metadata: ProviderMetadata) -> None:
        super().__init__(metadata)

    async def list_models(self, api_key: str, base_url: Optional[str] = None) -> list[ProviderModel]:
        api_key = (api_key or "").strip()
        base = (base_url or self.metadata.default_base_url).rstrip("/")
        headers = self._headers(api_key)
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(f"{base}/models", headers=headers)
                if response.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        "models listing unavailable",
                        request=response.request,
                        response=response,
                    )
                payload = response.json()
            items = payload.get("data", payload)
            if isinstance(items, list) and items:
                live: list[ProviderModel] = []
                for item in items:
                    model_id = item.get("id") or item.get("name")
                    if not model_id:
                        continue
                    live.append(
                        ProviderModel(
                            id=model_id,
                            provider=self.metadata.id,
                            label=model_id,
                            owned_by=item.get("owned_by") or "302.AI",
                            supports_tools=True,
                        )
                    )
                if live:
                    live.sort(key=lambda m: m.label.lower())
                    return live
        except Exception:
            pass

        return list(self.CATALOG_MODELS)


AI302Provider.CATALOG_MODELS = [
    ProviderModel(id="MiniMax-M1", provider=ProviderType.AI_302, label="MiniMax-M1", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="MiniMax-M2", provider=ProviderType.AI_302, label="MiniMax-M2", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="MiniMax-M2.1", provider=ProviderType.AI_302, label="MiniMax-M2.1", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="MiniMax-M2.7", provider=ProviderType.AI_302, label="MiniMax-M2.7", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="MiniMax-M2.7-highspeed", provider=ProviderType.AI_302, label="MiniMax-M2.7-highspeed", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="chatgpt-4o-latest", provider=ProviderType.AI_302, label="chatgpt-4o-latest", owned_by="302.AI", supports_tools=False),
    ProviderModel(id="claude-3-5-haiku-20241022", provider=ProviderType.AI_302, label="claude-3-5-haiku-20241022", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-3-5-haiku-latest", provider=ProviderType.AI_302, label="claude-3-5-haiku-latest", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-haiku-4-5", provider=ProviderType.AI_302, label="claude-haiku-4-5", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-haiku-4-5-20251001", provider=ProviderType.AI_302, label="claude-haiku-4-5-20251001", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-opus-4-1-20250805", provider=ProviderType.AI_302, label="claude-opus-4-1-20250805", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-opus-4-1-20250805-thinking", provider=ProviderType.AI_302, label="claude-opus-4-1-20250805-thinking", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-opus-4-20250514", provider=ProviderType.AI_302, label="claude-opus-4-20250514", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-opus-4-5", provider=ProviderType.AI_302, label="claude-opus-4-5", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-opus-4-5-20251101", provider=ProviderType.AI_302, label="claude-opus-4-5-20251101", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-opus-4-5-20251101-thinking", provider=ProviderType.AI_302, label="claude-opus-4-5-20251101-thinking", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-opus-4-6", provider=ProviderType.AI_302, label="claude-opus-4-6", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-opus-4-6-thinking", provider=ProviderType.AI_302, label="claude-opus-4-6-thinking", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-opus-4-7", provider=ProviderType.AI_302, label="claude-opus-4-7", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-sonnet-4-20250514", provider=ProviderType.AI_302, label="claude-sonnet-4-20250514", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-sonnet-4-5", provider=ProviderType.AI_302, label="claude-sonnet-4-5", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-sonnet-4-5-20250929", provider=ProviderType.AI_302, label="claude-sonnet-4-5-20250929", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-sonnet-4-5-20250929-thinking", provider=ProviderType.AI_302, label="claude-sonnet-4-5-20250929-thinking", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-sonnet-4-6", provider=ProviderType.AI_302, label="claude-sonnet-4-6", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="claude-sonnet-4-6-thinking", provider=ProviderType.AI_302, label="claude-sonnet-4-6-thinking", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="deepseek-chat", provider=ProviderType.AI_302, label="deepseek-chat", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="deepseek-reasoner", provider=ProviderType.AI_302, label="deepseek-reasoner", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="deepseek-v3.2", provider=ProviderType.AI_302, label="deepseek-v3.2", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="deepseek-v3.2-thinking", provider=ProviderType.AI_302, label="deepseek-v3.2-thinking", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="doubao-seed-1-6-thinking-250715", provider=ProviderType.AI_302, label="doubao-seed-1-6-thinking-250715", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="doubao-seed-1-6-vision-250815", provider=ProviderType.AI_302, label="doubao-seed-1-6-vision-250815", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="doubao-seed-1-8-251215", provider=ProviderType.AI_302, label="doubao-seed-1-8-251215", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="doubao-seed-code-preview-251028", provider=ProviderType.AI_302, label="doubao-seed-code-preview-251028", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gemini-2.0-flash-lite", provider=ProviderType.AI_302, label="gemini-2.0-flash-lite", owned_by="302.AI", supports_tools=False),
    ProviderModel(id="gemini-2.5-flash", provider=ProviderType.AI_302, label="gemini-2.5-flash", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gemini-2.5-flash-image", provider=ProviderType.AI_302, label="gemini-2.5-flash-image", owned_by="302.AI", supports_tools=False),
    ProviderModel(id="gemini-2.5-flash-lite-preview-09-2025", provider=ProviderType.AI_302, label="gemini-2.5-flash-lite-preview-09-2025", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gemini-2.5-flash-nothink", provider=ProviderType.AI_302, label="gemini-2.5-flash-nothink", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gemini-2.5-flash-preview-09-2025", provider=ProviderType.AI_302, label="gemini-2.5-flash-preview-09-2025", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gemini-2.5-pro", provider=ProviderType.AI_302, label="gemini-2.5-pro", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gemini-3-flash-preview", provider=ProviderType.AI_302, label="gemini-3-flash-preview", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gemini-3-pro-image-preview", provider=ProviderType.AI_302, label="gemini-3-pro-image-preview", owned_by="302.AI", supports_tools=False),
    ProviderModel(id="gemini-3-pro-preview", provider=ProviderType.AI_302, label="gemini-3-pro-preview", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gemini-3.1-flash-image-preview", provider=ProviderType.AI_302, label="gemini-3.1-flash-image-preview", owned_by="302.AI", supports_tools=False),
    ProviderModel(id="glm-4.5", provider=ProviderType.AI_302, label="glm-4.5", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-4.5-air", provider=ProviderType.AI_302, label="glm-4.5-air", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-4.5-airx", provider=ProviderType.AI_302, label="glm-4.5-airx", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-4.5-x", provider=ProviderType.AI_302, label="glm-4.5-x", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-4.5v", provider=ProviderType.AI_302, label="glm-4.5v", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-4.6", provider=ProviderType.AI_302, label="glm-4.6", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-4.6v", provider=ProviderType.AI_302, label="glm-4.6v", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-4.7", provider=ProviderType.AI_302, label="glm-4.7", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-4.7-flashx", provider=ProviderType.AI_302, label="glm-4.7-flashx", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-5", provider=ProviderType.AI_302, label="glm-5", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-5-turbo", provider=ProviderType.AI_302, label="glm-5-turbo", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-5.1", provider=ProviderType.AI_302, label="glm-5.1", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-5v-turbo", provider=ProviderType.AI_302, label="glm-5v-turbo", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="glm-for-coding", provider=ProviderType.AI_302, label="glm-for-coding", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-4.1", provider=ProviderType.AI_302, label="gpt-4.1", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-4.1-mini", provider=ProviderType.AI_302, label="gpt-4.1-mini", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-4.1-nano", provider=ProviderType.AI_302, label="gpt-4.1-nano", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-4o", provider=ProviderType.AI_302, label="gpt-4o", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5", provider=ProviderType.AI_302, label="gpt-5", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5-mini", provider=ProviderType.AI_302, label="gpt-5-mini", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5-pro", provider=ProviderType.AI_302, label="gpt-5-pro", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5-thinking", provider=ProviderType.AI_302, label="gpt-5-thinking", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5.1", provider=ProviderType.AI_302, label="gpt-5.1", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5.1-chat-latest", provider=ProviderType.AI_302, label="gpt-5.1-chat-latest", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5.2", provider=ProviderType.AI_302, label="gpt-5.2", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5.2-chat-latest", provider=ProviderType.AI_302, label="gpt-5.2-chat-latest", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5.4", provider=ProviderType.AI_302, label="gpt-5.4", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5.4-mini", provider=ProviderType.AI_302, label="gpt-5.4-mini", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5.4-mini-2026-03-17", provider=ProviderType.AI_302, label="gpt-5.4-mini-2026-03-17", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5.4-nano", provider=ProviderType.AI_302, label="gpt-5.4-nano", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5.4-nano-2026-03-17", provider=ProviderType.AI_302, label="gpt-5.4-nano-2026-03-17", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="gpt-5.4-pro", provider=ProviderType.AI_302, label="gpt-5.4-pro", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="grok-4-1-fast-non-reasoning", provider=ProviderType.AI_302, label="grok-4-1-fast-non-reasoning", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="grok-4-1-fast-reasoning", provider=ProviderType.AI_302, label="grok-4-1-fast-reasoning", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="grok-4-fast-non-reasoning", provider=ProviderType.AI_302, label="grok-4-fast-non-reasoning", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="grok-4-fast-reasoning", provider=ProviderType.AI_302, label="grok-4-fast-reasoning", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="grok-4.1", provider=ProviderType.AI_302, label="grok-4.1", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="grok-4.20-beta-0309-non-reasoning", provider=ProviderType.AI_302, label="grok-4.20-beta-0309-non-reasoning", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="grok-4.20-beta-0309-reasoning", provider=ProviderType.AI_302, label="grok-4.20-beta-0309-reasoning", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="grok-4.20-multi-agent-beta-0309", provider=ProviderType.AI_302, label="grok-4.20-multi-agent-beta-0309", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="kimi-k2-0905-preview", provider=ProviderType.AI_302, label="kimi-k2-0905-preview", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="kimi-k2-thinking", provider=ProviderType.AI_302, label="kimi-k2-thinking", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="kimi-k2-thinking-turbo", provider=ProviderType.AI_302, label="kimi-k2-thinking-turbo", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="ministral-14b-2512", provider=ProviderType.AI_302, label="ministral-14b-2512", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="mistral-large-2512", provider=ProviderType.AI_302, label="mistral-large-2512", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="qwen-flash", provider=ProviderType.AI_302, label="qwen-flash", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="qwen-max-latest", provider=ProviderType.AI_302, label="qwen-max-latest", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="qwen-plus", provider=ProviderType.AI_302, label="qwen-plus", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="qwen3-235b-a22b", provider=ProviderType.AI_302, label="qwen3-235b-a22b", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="qwen3-235b-a22b-instruct-2507", provider=ProviderType.AI_302, label="qwen3-235b-a22b-instruct-2507", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="qwen3-30b-a3b", provider=ProviderType.AI_302, label="qwen3-30b-a3b", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="qwen3-coder-480b-a35b-instruct", provider=ProviderType.AI_302, label="qwen3-coder-480b-a35b-instruct", owned_by="302.AI", supports_tools=True),
    ProviderModel(id="qwen3-max-2025-09-23", provider=ProviderType.AI_302, label="qwen3-max-2025-09-23", owned_by="302.AI", supports_tools=True),
]
AI302Provider.CATALOG_MODELS.sort(key=lambda m: m.label.lower())