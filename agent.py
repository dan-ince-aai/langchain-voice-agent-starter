"""PydanticAI agent behind the reply endpoint.

The platform POSTs an OpenAI-format chat completion request for each turn.
`reply()` returns either speech (`say`) or a platform tool call (`call_tool`).

The agent's output is a `Decision`. An output validator checks each one with
`audit()`; a failure raises `ModelRetry`, which returns the reason to the model
for another attempt (`retries=1`). Exhausted retries, a model error, or
exceeding `REPLY_BUDGET_SECONDS` produce the `FALLBACK` response.
"""

import asyncio
import json
import os
import threading
from concurrent.futures import TimeoutError as ReplyTimeout
from dataclasses import dataclass
from typing import Literal

import httpx2
from assemblyai_agents.byo import Turn, call_tool, say
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

GATEWAY = os.environ.get("LLM_GATEWAY", "https://llm-gateway.assemblyai.com/v1")
MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")

# The platform allows roughly 10s for a reply.
REPLY_BUDGET_SECONDS = 8.0
RETRIES = 1
FALLBACK = "Sorry, I missed that. Which place would you like the weather for?"


class Decision(BaseModel):
    action: Literal["speak", "check_weather"]
    text: str = Field(default="", description="For `speak`: one or two spoken sentences. No markdown.")
    location: str = Field(default="", description="For `check_weather`: the place, as the caller said it.")
    filler: str = Field(
        default="",
        description='For `check_weather`: one short sentence spoken while the lookup runs, naming the place. Example: "Let me check Lisbon for you."',
    )


INSTRUCTIONS = """
You answer a live phone call about the weather. Output is spoken: short
sentences, no lists, no markdown, no emoji, no digit strings.

Choose `check_weather` when the caller names a place that has not been looked
up yet, and always provide a `filler` sentence naming the place. Otherwise
choose `speak`.

When a weather result is in the conversation, use it. Report temperature in
whole degrees Celsius. Give Fahrenheit only if asked, using to_fahrenheit.
If `found` is false, say the place could not be found and ask for another.
If `unreachable` is true, say the weather service is unavailable and offer to
retry. Never invent weather data.
""".strip()


# --------------------------------------------------------------------------- gateway client

# The AssemblyAI LLM gateway omits `id` and `object` from responses and passes
# the upstream provider's `finish_reason` through unchanged. PydanticAI
# validates the full response, so these are filled in here. Harmless against
# an endpoint that already returns them.
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


# --------------------------------------------------------------------------- agent


@dataclass
class Deps:
    known: dict  # place (lowercase) -> weather result already in the transcript


agent = Agent(
    OpenAIChatModel(
        MODEL,
        provider=OpenAIProvider(
            base_url=GATEWAY,
            # Read at import; run.py loads .env first. Placeholder keeps imports key-free.
            api_key=os.environ.get("ASSEMBLYAI_API_KEY") or "unset",
            http_client=httpx2.AsyncClient(transport=_OpenAIShaped(), timeout=60),
        ),
    ),
    output_type=Decision,
    instructions=INSTRUCTIONS,
    deps_type=Deps,
    retries=RETRIES,
)


@agent.tool_plain
def to_fahrenheit(celsius: float) -> int:
    """Convert Celsius to whole degrees Fahrenheit."""
    return round(celsius * 9 / 5 + 32)


def audit(proposal: Decision, known: dict) -> list[str]:
    """Validate a proposal. Returns a list of problems; empty means accepted.

    A missing `filler` is defaulted in place rather than rejected.
    """
    if proposal.action == "check_weather":
        place = proposal.location.strip().lower()
        if not place:
            return ["check_weather requires a location"]
        already = [k for k in known if k in place or place in k]
        if already:
            return [f"weather for {already[0]} is already in the conversation; answer from it"]
        if not proposal.filler.strip():
            proposal.filler = f"Let me check {proposal.location.strip()} for you."
        return []

    text = proposal.text.strip()
    if not text:
        return ["empty text"]
    if any(mark in text for mark in ("**", "##", "\n-", "\n*", "```")):
        return ["text contains markdown"]
    if "degree" in text.lower() and not known:
        return ["temperature given but nothing has been looked up"]
    return []


@agent.output_validator
def check(ctx: RunContext[Deps], proposal: Decision) -> Decision:
    problems = audit(proposal, ctx.deps.known)
    if problems:
        print(f"[agent] rejected: {problems[0]}", flush=True)
        raise ModelRetry("; ".join(problems))
    detail = f"check_weather {proposal.location!r}" if proposal.action == "check_weather" else f"speak {proposal.text[:50]!r}"
    print(f"[agent] {detail}", flush=True)
    return proposal


# --------------------------------------------------------------------------- reply endpoint


def transcript(turn: Turn) -> tuple[str, dict]:
    """Flatten the platform transcript to text and collect prior weather results.

    Platform tool calls are not forwarded as chat messages: the agent was not
    given those tools. Tool results are kept in the text so later turns can
    reference them (e.g. unit conversion).
    """
    lines, known = [], {}

    def remember(result) -> None:
        if isinstance(result, dict):
            for key in (result.get("place"), result.get("location")):
                if key:
                    known[str(key).lower()] = result

    for message in turn.request.get("messages", []):
        role, content = message.get("role"), message.get("content")
        if role == "user" and content:
            lines.append(f"Caller: {content}")
        elif role == "assistant" and content:
            lines.append(f"You: {content.strip()}")
        elif role == "tool" and content:
            lines.append(f"Weather lookup returned: {content}")
            try:
                remember(json.loads(content))
            except ValueError:
                pass
    pending = turn.pending
    if pending and pending.name == "get_weather" and json.dumps(pending.value) not in "\n".join(lines):
        lines.append(f"Weather lookup returned: {json.dumps(pending.value)}")
        remember(pending.value)
    return "\n".join(lines) or "The caller has just connected.", known


# One event loop for the process. `serve()` calls `reply` synchronously from
# request threads; a loop per call would leave the HTTP client bound to a
# closed loop on the next turn.
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()


def reply(turn: Turn):
    """Reply callback for `serve()`. Must be synchronous."""
    notes, known = transcript(turn)
    future = asyncio.run_coroutine_threadsafe(agent.run(notes, deps=Deps(known)), _loop)
    try:
        decision = future.result(timeout=REPLY_BUDGET_SECONDS).output
    except ReplyTimeout:
        future.cancel()
        print(f"[agent] timeout after {REPLY_BUDGET_SECONDS:.0f}s; fallback", flush=True)
        decision = Decision(action="speak", text=FALLBACK)
    except Exception as exc:  # noqa: BLE001 — retries exhausted or model error
        print(f"[agent] {type(exc).__name__}: {str(exc)[:80]}; fallback", flush=True)
        decision = Decision(action="speak", text=FALLBACK)

    if decision.action == "check_weather":
        # `saying` is spoken while the platform runs the tool.
        return call_tool("get_weather", saying=decision.filler, location=decision.location)
    return say(decision.text)
