"""Your agent. A LangGraph graph around the model, not a passthrough to it.

A bare model call is a gateway: whatever comes back gets spoken. An agent puts
a loop around the model, and that loop is what a framework is for. This one has
three steps:

  think     the model reasons, uses its own tools, and proposes a Decision
  check     the proposal is held to rules the model cannot be trusted with —
            no temperature it never looked up, no second lookup of a place it
            already has, nothing that is not speakable
  answer    a proposal that passes is spoken; one that fails goes back to think
            with the reason; two failures, an error, or a hung model and the
            caller gets a safe sentence instead of silence or an invention

The platform asks what to say by POSTing an OpenAI-shaped chat request, and
`reply()` answers it. Everything above is inside that one function.
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

# The platform gives a reply endpoint about ten seconds. A model that has not
# answered in eight is not going to; the caller gets a sentence, not a timeout.
REPLY_BUDGET_SECONDS = 8.0
MAX_ATTEMPTS = 2
FALLBACK = "Sorry, I missed that. Which place would you like the weather for?"


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
call to_fahrenheit rather than working it out. If `found` is false, say you
could not find that place and ask for another; if `unreachable` is also true,
say the weather service is not answering right now and offer to try again.
Never invent weather.
""".strip()


# --------------------------------------------------------------- think


@tool
def to_fahrenheit(celsius: float) -> int:
    """Convert a Celsius temperature to whole degrees Fahrenheit."""
    return round(celsius * 9 / 5 + 32)


_model = None


def propose(messages: list) -> Decision:
    """Ask the model for a Decision. It may use its own tools on the way."""
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
    notes: str                  # the call so far, as plain text
    known: dict                 # place -> weather result already in the call
    proposal: Optional[Decision]
    problems: list              # why the last proposal was rejected
    attempts: int


def fresh(notes: str, known: dict) -> State:
    return {"notes": notes, "known": known, "proposal": None, "problems": [], "attempts": 0}


def think(state: State) -> dict:
    messages = [("human", state["notes"])]
    if state["problems"]:
        messages.append(("human", "Your last answer was rejected: " + "; ".join(state["problems"]) + ". Give a corrected answer."))
    try:
        proposal = propose(messages)
    except Exception as exc:  # noqa: BLE001 — any model failure is one more attempt, not a crash
        print(f"[agent] attempt {state['attempts'] + 1}: model failed — {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        proposal = None
    else:
        what = f"check_weather {proposal.location!r}" if proposal.action == "check_weather" else f"speak {proposal.text[:50]!r}"
        print(f"[agent] attempt {state['attempts'] + 1}: {what}", flush=True)
    return {"proposal": proposal, "attempts": state["attempts"] + 1}


# --------------------------------------------------------------- check


def audit(proposal: Optional[Decision], known: dict) -> list[str]:
    """Everything wrong with a proposal. Empty means it can be spoken.

    These are the rules the model is told and cannot be trusted to keep. Each
    one was the model's actual behaviour on a call before it was a rule here.
    """
    if proposal is None:
        return ["the model gave no usable answer"]

    if proposal.action == "check_weather":
        place = proposal.location.strip().lower()
        if not place:
            return ["check_weather needs a place"]
        already = [k for k in known if k in place or place in k]
        if already:
            return [f"you already have the weather for {already[0]}; answer from it instead of looking it up again"]
        if not proposal.filler.strip():
            # Fixed rather than rejected: the words are ours to supply.
            proposal.filler = f"Let me check {proposal.location.strip()} for you."
        return []

    text = proposal.text.strip()
    if not text:
        return ["there is nothing to say"]
    if any(mark in text for mark in ("**", "##", "\n-", "\n*", "```")):
        return ["spoken text must not contain markdown"]
    if "degree" in text.lower() and not known:
        return ["you have not looked anything up, so you cannot give a temperature"]
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
    print("[agent] falling back to a safe sentence", flush=True)
    return {"proposal": Decision(action="speak", text=FALLBACK), "problems": []}


# --------------------------------------------------------------- the graph


_graph = StateGraph(State)
_graph.add_node("think", think)
_graph.add_node("check", check)
_graph.add_node("fallback", fallback)
_graph.add_edge(START, "think")
_graph.add_edge("think", "check")
_graph.add_conditional_edges("check", route, {"think": "think", "fallback": "fallback", END: END})
_graph.add_edge("fallback", END)
graph = _graph.compile()


# --------------------------------------------------------------- the seam


def transcript(turn: Turn) -> tuple[str, dict]:
    """The call so far as plain notes, and every weather result it contains.

    Flattened rather than passed through as chat messages: the platform's
    transcript carries its own tool calls, which this agent was never given
    and would reject. Every lookup result stays in — two turns later the
    caller asks "and in Fahrenheit?", and the model needs the Celsius figure,
    not its own rounded sentence about it.
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
    """One turn: run the graph within the budget, and translate the answer.

    Synchronous on purpose. `serve()` awaits a coroutine from a tool or a
    pre-connect handler but not from here, so an `async def reply` is never
    awaited and the turn goes out empty.
    """
    notes, known = transcript(turn)
    try:
        decision = _pool.submit(graph.invoke, fresh(notes, known)).result(timeout=REPLY_BUDGET_SECONDS)["proposal"]
    except ReplyTimeout:
        print(f"[agent] no answer in {REPLY_BUDGET_SECONDS:.0f}s — falling back", flush=True)
        decision = Decision(action="speak", text=FALLBACK)

    if decision.action == "check_weather":
        # `saying=` puts the words in the same response as the tool call, so
        # they are spoken while the lookup runs instead of after it.
        return call_tool("get_weather", saying=decision.filler, location=decision.location)
    return say(decision.text)
