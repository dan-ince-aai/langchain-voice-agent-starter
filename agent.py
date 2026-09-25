"""Your text agent. This one is LangChain; the voice never knows.

The platform asks what to say by POSTing an OpenAI-shaped chat request, and
`reply()` answers it. In between is an ordinary LangChain agent: a model, a
system prompt, its own tools, and a structured answer.

That is the point of putting a framework here rather than a bare model call.
The agent can think in several steps and use its own tools inside one turn —
the Fahrenheit conversion below runs in-process, in milliseconds, and the
caller hears only the answer. Only work that is genuinely slow goes out to the
platform as a voice tool, so it can be talked over while it runs.
"""

import json
import os
from typing import Literal

from assemblyai_agents.byo import Turn, call_tool, say
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

GATEWAY = os.environ.get("LLM_GATEWAY", "https://llm-gateway.assemblyai.com/v1")
MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")


# --------------------------------------------------------------- what a turn decides


class Decision(BaseModel):
    """One turn of the call: say something, or look somewhere up first."""

    action: Literal["speak", "check_weather"]
    text: str = Field(default="", description="For `speak`: one or two spoken sentences. No markdown.")
    location: str = Field(default="", description="For `check_weather`: the place, as the caller said it.")
    filler: str = Field(
        default="",
        description='For `check_weather`: one short sentence spoken WHILE the lookup runs, naming the place. Like "Let me check Lisbon for you."',
    )


INSTRUCTIONS = """
You answer a live phone call about the weather. You are speaking, not writing:
short sentences, no lists, no markdown, no emoji, never a number read as a
digit string.

Choose `check_weather` when the caller names a place you have not looked up
yet, and always write a `filler` sentence that names the place. Otherwise
choose `speak`.

When a weather result is in the conversation, use it and say the temperature in
whole degrees Celsius. Only give Fahrenheit if the caller asks for it, and then
call to_fahrenheit rather than working it out.
If `found` is false, say you could not find that place and ask for another; if
`unreachable` is also true, say the weather service is not answering right now
and offer to try again. Never invent weather.
""".strip()


# --------------------------------------------------------------- the agent's own tools


@tool
def to_fahrenheit(celsius: float) -> int:
    """Convert a Celsius temperature to whole degrees Fahrenheit."""
    return round(celsius * 9 / 5 + 32)


_agent = None


def agent():
    """Built on first use, so the module imports without a key."""
    global _agent
    if _agent is None:
        _agent = create_agent(
            ChatOpenAI(model=MODEL, base_url=GATEWAY, api_key=os.environ["ASSEMBLYAI_API_KEY"]),
            tools=[to_fahrenheit],
            system_prompt=INSTRUCTIONS,
            response_format=ToolStrategy(Decision),
        )
    return _agent


# --------------------------------------------------------------- the seam


def transcript(turn: Turn) -> str:
    """The call so far, as plain notes.

    Flattened rather than passed through as chat messages: the platform's
    transcript carries its own tool calls, which this agent was never given
    and would reject, and the model only needs to know what was said.
    """
    lines, results = [], []
    for message in turn.request.get("messages", []):
        role, content = message.get("role"), message.get("content")
        if role == "user" and content:
            lines.append(f"Caller: {content}")
        elif role == "assistant" and content:
            lines.append(f"You: {content.strip()}")
        elif role == "tool" and content:
            # Every result stays in the notes, not only the newest. Two turns
            # after a lookup the caller asks "and in Fahrenheit?", and the model
            # needs the Celsius figure, not its own rounded sentence about it.
            lines.append(f"Weather lookup returned: {content}")
            results.append(content)
    pending = turn.pending
    if pending and pending.name == "get_weather" and json.dumps(pending.value) not in results:
        lines.append(f"Weather lookup returned: {json.dumps(pending.value)}")
    return "\n".join(lines) or "The caller has just connected."


def reply(turn: Turn):
    """One turn: run the agent, and translate its decision for the voice.

    Synchronous on purpose. `serve()` awaits a coroutine from a tool or a
    pre-connect handler but not from here, so an `async def reply` is never
    awaited and the turn goes out empty.
    """
    decision: Decision = agent().invoke({"messages": [("human", transcript(turn))]})["structured_response"]

    if decision.action == "check_weather":
        # `saying=` puts the words in the same response as the tool call, so
        # they are spoken while the lookup runs instead of after it.
        return call_tool(
            "get_weather",
            saying=decision.filler or f"Let me check {decision.location} for you.",
            location=decision.location,
        )
    return say(decision.text)
