"""The text LLM that decides every turn, through PydanticAI.

The platform asks this process what to say by POSTing an OpenAI-shaped chat
request. `reply()` answers it. In between sits a PydanticAI agent whose output
is a *decision*, not prose:

    Speak        -- say these words, turn over
    CheckWeather -- run the tool, and say `filler` while it runs

That second one is the whole point of pairing a text model with a voice one. A
tool call is a second or two of dead air, which on a phone reads as a dropped
line. Because a response may carry both a tool call and words, the model writes
the words itself, in the caller's own context: "let me look up Lisbon for you",
not a canned "one moment".
"""

import asyncio
import json
import os
import threading
from typing import Union

import httpx2
from assemblyai_agents.byo import Turn, call_tool, say
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

GATEWAY = os.environ.get("LLM_GATEWAY", "https://llm-gateway.assemblyai.com/v1")
MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")


class Speak(BaseModel):
    """Say something and hand the turn back to the caller."""

    text: str = Field(description="One or two spoken sentences. No markdown.")


class CheckWeather(BaseModel):
    """Look the weather up, then answer from the result on the next turn."""

    location: str = Field(description="The place to look up, as the caller said it.")
    filler: str = Field(
        description=(
            "One short sentence spoken WHILE the lookup runs, naming the place. "
            'Like "Let me check Lisbon for you."'
        )
    )


INSTRUCTIONS = """
You answer a live phone call about the weather. You are speaking, not writing:
short sentences, no lists, no markdown, no emoji, never a number read as a
digit string.

Call CheckWeather when the caller names a place you have not yet looked up.
Otherwise Speak.

When a weather result is in the conversation, use it and say the temperature in
whole degrees. If `found` is false, say you could not find that place and ask
for another; if `unreachable` is also true, say the weather service is not
answering right now and offer to try again. Never invent weather.
""".strip()


# AssemblyAI's gateway speaks the OpenAI request shape but not quite the OpenAI
# response shape: it omits `id` and `object`, and passes the upstream provider's
# own `finish_reason` through ("end_turn", "tool_use"). PydanticAI validates the
# response strictly and rejects it. Filling in the three fields is the whole fix.
#
# Point MODEL and LLM_GATEWAY at any OpenAI-compatible endpoint and this becomes
# dead weight but stays harmless -- it only fills in what is missing.
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

        body.setdefault("id", body.get("request_id") or "chatcmpl-gateway")
        body.setdefault("object", "chat.completion")
        body["id"] = body["id"] or "chatcmpl-gateway"
        body["object"] = body["object"] or "chat.completion"
        for choice in body.get("choices") or []:
            reason = choice.get("finish_reason")
            choice["finish_reason"] = _FINISH.get(reason, reason or "stop")

        # Content-length and any encoding header describe the body we replaced.
        headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in ("content-length", "content-encoding")
        }
        return httpx2.Response(
            response.status_code, headers=headers,
            content=json.dumps(body).encode(), request=request,
        )


_brain: Agent | None = None


def brain() -> Agent:
    """Built on first use, not at import.

    The key is read from the environment, and `run.py` loads `.env` after the
    imports — so building this at import time would need the key to exist
    before anything had gone looking for it, and would make this module
    unimportable in a test.
    """
    global _brain
    if _brain is None:
        _brain = Agent(
            OpenAIChatModel(
                MODEL,
                provider=OpenAIProvider(
                    base_url=GATEWAY,
                    api_key=os.environ["ASSEMBLYAI_API_KEY"],
                    http_client=httpx2.AsyncClient(transport=_OpenAIShaped(), timeout=60),
                ),
            ),
            output_type=Union[Speak, CheckWeather],
            instructions=INSTRUCTIONS,
        )
    return _brain


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
        lines.append(f"Weather lookup returned: {turn.pending.value}")
    return "\n".join(lines) or "The caller has just connected."


_loop: asyncio.AbstractEventLoop | None = None


def _run(coro):
    """Run `coro` on one long-lived loop, from a synchronous caller.

    Two things force this. `serve()` awaits a coroutine from a tool or a
    pre-connect handler but not from the reply endpoint, so `reply` has to be
    synchronous. And `run_sync()` would build a fresh event loop per turn, while
    the HTTP client's locks stay bound to the loop that made them — so the
    second turn of every call died with "bound to a different event loop".

    One loop for the process fixes both, and keeps the connection to the model
    warm, which a caller waiting in silence notices.
    """
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, daemon=True).start()
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


def reply(turn: Turn):
    """One turn: ask the text model, and translate its decision for the voice."""
    decision = _run(brain().run(transcript(turn))).output

    if isinstance(decision, CheckWeather):
        # `saying=` puts the words in the same response as the tool call, so
        # they are spoken while the lookup runs instead of after it.
        return call_tool("get_weather", saying=decision.filler, location=decision.location)

    return say(decision.text)
