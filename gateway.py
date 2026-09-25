"""Model client for the AssemblyAI LLM gateway.

The gateway accepts OpenAI-format requests but omits `id` and `object` from
responses and passes the upstream provider's `finish_reason` through unchanged.
PydanticAI validates the full response, so the transport below fills those
fields in. It is inert against an endpoint that already returns them.
"""

import json
import os

import httpx2
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

GATEWAY = os.environ.get("LLM_GATEWAY", "https://llm-gateway.assemblyai.com/v1")
MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")

_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


class _OpenAIShaped(httpx2.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        response = await super().handle_async_request(request)
        await response.aread()
        try:
            body = json.loads(response.content)
        except ValueError:
            return response
        if not isinstance(body, dict) or "choices" not in body:
            return response
        body["id"] = body.get("id") or body.get("request_id") or "chatcmpl-gateway"
        body["object"] = body.get("object") or "chat.completion"
        for choice in body.get("choices") or []:
            reason = choice.get("finish_reason")
            choice["finish_reason"] = _FINISH.get(reason, reason or "stop")
        headers = {k: v for k, v in response.headers.items() if k.lower() not in ("content-length", "content-encoding")}
        return httpx2.Response(response.status_code, headers=headers, content=json.dumps(body).encode(), request=request)


def model() -> OpenAIChatModel:
    """PydanticAI model for MODEL via GATEWAY, authenticated with ASSEMBLYAI_API_KEY."""
    return OpenAIChatModel(
        MODEL,
        provider=OpenAIProvider(
            base_url=GATEWAY,
            # Read at import; run.py loads .env first. The placeholder keeps imports key-free.
            api_key=os.environ.get("ASSEMBLYAI_API_KEY") or "unset",
            http_client=httpx2.AsyncClient(transport=_OpenAIShaped(), timeout=60),
        ),
    )
