"""What every brain shares, so each brain is only the framework-specific part.

The platform asks this process what to say by POSTing an OpenAI-shaped chat
request. A brain answers with a `Decision`; `answer()` turns that into what the
voice needs. Nothing here belongs to any one framework.
"""

import asyncio
import json
import os
import threading
from typing import Literal

import httpx2
from assemblyai_agents.byo import Turn, call_tool, say
from pydantic import BaseModel, Field

GATEWAY = os.environ.get("LLM_GATEWAY", "https://llm-gateway.assemblyai.com/v1")
MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")


def api_key() -> str:
    return os.environ["ASSEMBLYAI_API_KEY"]


# --------------------------------------------------------------- the decision


class Decision(BaseModel):
    """One turn of the call, decided by the text model.

    A single model rather than a union of two, because every framework can
    return one Pydantic model and not every framework can return a union.
    """

    action: Literal["speak", "check_weather"] = Field(
        description="`speak` to answer the caller; `check_weather` to look a place up first."
    )
    text: str = Field(
        default="",
        description="For `speak`: one or two spoken sentences. No markdown.",
    )
    location: str = Field(
        default="", description="For `check_weather`: the place, as the caller said it."
    )
    filler: str = Field(
        default="",
        description=(
            "For `check_weather`: one short sentence spoken WHILE the lookup runs, "
            'naming the place. Like "Let me check Lisbon for you."'
        ),
    )


INSTRUCTIONS = """
You answer a live phone call about the weather. You are speaking, not writing:
short sentences, no lists, no markdown, no emoji, never a number read as a
digit string.

Choose `check_weather` when the caller names a place you have not yet looked
up, and write a `filler` sentence that names it. Otherwise choose `speak`.

When a weather result is in the conversation, use it and say the temperature in
whole degrees. If `found` is false, say you could not find that place and ask
for another; if `unreachable` is also true, say the weather service is not
answering right now and offer to try again. Never invent weather.
""".strip()


def answer(decision: Decision):
    """The decision, as the voice platform needs it."""
    if decision.action == "check_weather":
        # `saying=` puts the words in the same response as the tool call, so
        # they are spoken while the lookup runs instead of after it.
        return call_tool("get_weather", saying=decision.filler, location=decision.location)
    return say(decision.text)


# --------------------------------------------------------------- the conversation


def transcript(turn: Turn) -> str:
    """The call so far, as plain notes.

    Flattened rather than passed through as chat messages: the platform's
    transcript carries tool calls and system lines that a gateway will reject,
    and the model only needs to know what was said.
    """
    lines = []
    for message in turn.request.get("messages", []):
        role, content = message.get("role"), message.get("content")
        if role == "user" and content:
            lines.append(f"Caller: {content}")
        elif role == "assistant" and content:
            lines.append(f"You: {content.strip()}")
    if turn.pending and turn.pending.name == "get_weather":
        lines.append(f"Weather lookup returned: {json.dumps(turn.pending.value)}")
    return "\n".join(lines) or "The caller has just connected."


# --------------------------------------------------------------- async, from sync


_loop: asyncio.AbstractEventLoop | None = None


def run_sync(coro):
    """Run a coroutine on one long-lived loop, from a synchronous caller.

    Two things force this. `serve()` awaits a coroutine from a tool or a
    pre-connect handler but not from the reply endpoint, so `reply` has to be
    synchronous. And building a fresh event loop per turn would leave a
    framework's HTTP client bound to the loop that made it — the second turn
    of every call died with "bound to a different event loop".

    One loop for the process fixes both, and keeps the connection to the model
    warm, which a caller waiting in silence notices.
    """
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, daemon=True).start()
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# --------------------------------------------------------------- the gateway's shape


# AssemblyAI's gateway speaks the OpenAI request shape but not quite the OpenAI
# response shape: it omits `id` and `object`, and passes the upstream provider's
# own `finish_reason` through ("end_turn", "tool_use"). Most clients read only
# what they need and never notice. PydanticAI validates the whole response and
# rejects it, so it gets a client that fills the three fields in.
_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


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

        headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in ("content-length", "content-encoding")
        }
        return httpx2.Response(
            response.status_code, headers=headers,
            content=json.dumps(body).encode(), request=request,
        )


def openai_shaped_client() -> httpx2.AsyncClient:
    """An HTTP client that makes the gateway look exactly like OpenAI."""
    return httpx2.AsyncClient(transport=_OpenAIShaped(), timeout=60)
