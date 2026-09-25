"""LangGraph agent behind the reply endpoint.

The platform POSTs an OpenAI-format chat completion request for each turn.
`reply()` returns either speech (`say`) or a platform tool call (`call_tool`).

Graph:

    think -> check -> END
               |-> think     (rejected; reason appended; up to MAX_ATTEMPTS)
               |-> fallback  (rejected MAX_ATTEMPTS times)

`think` runs a LangChain agent with local tools and structured output.
`check` validates the proposal with `audit()`. Model exceptions count as a
failed attempt. `reply()` enforces a wall-clock budget; on timeout it returns
the fallback response.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as ReplyTimeout
from typing import Literal, Optional, TypedDict

from assemblyai_agents.byo import Turn, call_tool, say
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

GATEWAY = os.environ.get("LLM_GATEWAY", "https://llm-gateway.assemblyai.com/v1")
MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")

# The platform allows roughly 10s for a reply.
REPLY_BUDGET_SECONDS = 8.0
MAX_ATTEMPTS = 2
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


# --------------------------------------------------------------------------- think


@tool
def to_fahrenheit(celsius: float) -> int:
    """Convert Celsius to whole degrees Fahrenheit."""
    return round(celsius * 9 / 5 + 32)


_model = None


def propose(messages: list) -> Decision:
    """Run the LangChain agent and return its structured output."""
    global _model
    if _model is None:
        _model = create_agent(
            ChatOpenAI(model=MODEL, base_url=GATEWAY, api_key=os.environ["ASSEMBLYAI_API_KEY"]),
            tools=[to_fahrenheit],
            system_prompt=INSTRUCTIONS,
            response_format=ToolStrategy(Decision),
        )
    return _model.invoke({"messages": messages})["structured_response"]


class State(TypedDict):
    notes: str                  # flattened transcript
    known: dict                 # place (lowercase) -> weather result already in the transcript
    proposal: Optional[Decision]
    problems: list[str]         # audit failures for the current proposal
    attempts: int


def fresh(notes: str, known: dict) -> State:
    return {"notes": notes, "known": known, "proposal": None, "problems": [], "attempts": 0}


def think(state: State) -> dict:
    messages = [("human", state["notes"])]
    if state["problems"]:
        messages.append(("human", "Previous answer rejected: " + "; ".join(state["problems"]) + ". Provide a corrected answer."))
    attempt = state["attempts"] + 1
    try:
        proposal = propose(messages)
    except Exception as exc:  # noqa: BLE001 — counted as a failed attempt
        print(f"[agent] attempt {attempt}: error {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        proposal = None
    else:
        detail = f"check_weather {proposal.location!r}" if proposal.action == "check_weather" else f"speak {proposal.text[:50]!r}"
        print(f"[agent] attempt {attempt}: {detail}", flush=True)
    return {"proposal": proposal, "attempts": attempt}


# --------------------------------------------------------------------------- check


def audit(proposal: Optional[Decision], known: dict) -> list[str]:
    """Validate a proposal. Returns a list of problems; empty means accepted.

    A missing `filler` is defaulted in place rather than rejected.
    """
    if proposal is None:
        return ["no usable answer"]

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


def check(state: State) -> dict:
    problems = audit(state["proposal"], state["known"])
    if problems:
        print(f"[agent] rejected: {problems[0]}", flush=True)
    return {"problems": problems}


def route(state: State) -> str:
    if not state["problems"]:
        return END
    return "think" if state["attempts"] < MAX_ATTEMPTS else "fallback"


def fallback(state: State) -> dict:
    print("[agent] fallback", flush=True)
    return {"proposal": Decision(action="speak", text=FALLBACK), "problems": []}


_graph = StateGraph(State)
_graph.add_node("think", think)
_graph.add_node("check", check)
_graph.add_node("fallback", fallback)
_graph.add_edge(START, "think")
_graph.add_edge("think", "check")
_graph.add_conditional_edges("check", route, {"think": "think", "fallback": "fallback", END: END})
_graph.add_edge("fallback", END)
graph = _graph.compile()


# --------------------------------------------------------------------------- reply endpoint


def transcript(turn: Turn) -> tuple[str, dict]:
    """Flatten the platform transcript to text and collect prior weather results.

    Platform tool calls are not forwarded as chat messages: the LangChain agent
    was not given those tools and would reject them. Tool results are kept in
    the text so later turns can reference them (e.g. unit conversion).
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


_pool = ThreadPoolExecutor(max_workers=4)


def reply(turn: Turn):
    """Reply callback for `serve()`.

    Must be synchronous: `serve()` does not await the reply callable.
    """
    notes, known = transcript(turn)
    try:
        decision = _pool.submit(graph.invoke, fresh(notes, known)).result(timeout=REPLY_BUDGET_SECONDS)["proposal"]
    except ReplyTimeout:
        print(f"[agent] timeout after {REPLY_BUDGET_SECONDS:.0f}s; fallback", flush=True)
        decision = Decision(action="speak", text=FALLBACK)

    if decision.action == "check_weather":
        # `saying` is spoken while the platform runs the tool.
        return call_tool("get_weather", saying=decision.filler, location=decision.location)
    return say(decision.text)
