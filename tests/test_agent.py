"""Validation and retry behaviour with the model replaced by `FunctionModel`.

No API key or network required.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
from agent import Decision, audit  # noqa: E402
from assemblyai_agents.byo import Say, Turn  # noqa: E402
from pydantic_ai.messages import ModelResponse, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

LISBON = {"lisbon": {"found": True, "place": "Lisbon", "celsius": 21.4, "description": "overcast"}}


def turn(messages=None) -> Turn:
    messages = messages or [{"role": "user", "content": "hi"}]
    return Turn.from_request({"model": "weather-line", "stream": True, "tools": [], "messages": messages})


def scripted(*decisions):
    """A model that returns the given Decisions in order, as structured output."""
    queue = list(decisions)
    calls = []

    def model(messages, info):
        calls.append(messages)
        decision = queue.pop(0)
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=decision.model_dump())])

    return FunctionModel(model), calls


# audit


def test_temperature_without_lookup_is_rejected():
    assert audit(Decision(action="speak", text="It's twenty degrees in Lisbon."), known={})


def test_temperature_with_lookup_is_accepted():
    assert audit(Decision(action="speak", text="It's twenty-one degrees in Lisbon."), known=LISBON) == []


def test_known_place_is_not_looked_up_again():
    assert audit(Decision(action="check_weather", location="Lisbon", filler="x"), known=LISBON)


def test_new_place_is_looked_up():
    assert audit(Decision(action="check_weather", location="Porto", filler="x"), known=LISBON) == []


def test_missing_filler_is_defaulted():
    proposal = Decision(action="check_weather", location="Lisbon")
    assert audit(proposal, known={}) == []
    assert "Lisbon" in proposal.filler


def test_lookup_without_location_is_rejected():
    assert audit(Decision(action="check_weather", location="  "), known={})


def test_markdown_is_rejected():
    assert audit(Decision(action="speak", text="**Lisbon**: overcast"), known=LISBON)


def test_empty_text_is_rejected():
    assert audit(Decision(action="speak", text=""), known={})


# retry loop


def test_rejected_output_is_retried():
    model, calls = scripted(
        Decision(action="speak", text="It's thirty degrees."),
        Decision(action="speak", text="Which place would you like?"),
    )
    with agent.agent.override(model=model):
        answer = agent.reply(turn())

    assert isinstance(answer, Say) and answer.text == "Which place would you like?"
    assert len(calls) == 2


def test_rejection_reason_is_returned_to_the_model():
    model, calls = scripted(
        Decision(action="speak", text="It's thirty degrees."),
        Decision(action="speak", text="Which place?"),
    )
    with agent.agent.override(model=model):
        agent.reply(turn())

    assert "nothing has been looked up" in str(calls[1])


def test_exhausted_retries_fall_back():
    model, calls = scripted(*[Decision(action="speak", text="It's thirty degrees.")] * (agent.RETRIES + 1))
    with agent.agent.override(model=model):
        answer = agent.reply(turn())

    assert answer.text == agent.FALLBACK
    assert len(calls) == agent.RETRIES + 1


def test_model_exception_falls_back():
    def broken(messages, info):
        raise RuntimeError("gateway down")

    with agent.agent.override(model=FunctionModel(broken)):
        assert agent.reply(turn()).text == agent.FALLBACK


def test_timeout_falls_back_within_budget(monkeypatch):
    async def hung(messages, info):
        await asyncio.sleep(30)

    monkeypatch.setattr(agent, "REPLY_BUDGET_SECONDS", 0.2)
    with agent.agent.override(model=FunctionModel(hung)):
        answer = agent.reply(turn())

    assert answer.text == agent.FALLBACK
