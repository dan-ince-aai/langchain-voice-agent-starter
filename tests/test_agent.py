"""The agent, through the turns a call actually has.

These talk to the real model, so they need ASSEMBLYAI_API_KEY and skip without
it. They pin the contract, not the wording: a place the caller names becomes a
tool call with a filler that names it; a result in hand becomes a spoken
answer that uses it; a Fahrenheit question is answered from the agent's own
tool without going back to the platform.

    .venv/bin/python -m pytest -q
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assemblyai_agents.byo import Call, Say, Turn  # noqa: E402

import agent  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.environ.get("ASSEMBLYAI_API_KEY"), reason="needs ASSEMBLYAI_API_KEY"
)

LISBON = {"found": True, "place": "Lisbon", "country": "Portugal",
          "celsius": 21.4, "wind_kph": 11.6, "description": "overcast"}


def turn(messages) -> Turn:
    return Turn.from_request({"model": "weather-line", "stream": True, "tools": [], "messages": messages})


ASKED = [{"role": "user", "content": "what's the weather in Lisbon?"}]
LOOKED_UP = ASKED + [
    {"role": "assistant", "content": "Let me check Lisbon for you.",
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"location": "Lisbon"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": json.dumps(LISBON)},
]


def test_a_named_place_becomes_a_lookup_with_a_filler():
    answer = agent.reply(turn(ASKED))

    assert isinstance(answer, Call), answer
    assert answer.name == "get_weather"
    assert "lisbon" in answer.arguments["location"].lower()
    assert "lisbon" in answer.saying.lower(), answer.saying


def test_a_result_in_hand_becomes_a_spoken_answer():
    answer = agent.reply(turn(LOOKED_UP))

    assert isinstance(answer, Say), answer
    assert "lisbon" in answer.text.lower()
    assert "overcast" in answer.text.lower() or "twenty" in answer.text.lower(), answer.text


def test_a_fahrenheit_question_is_answered_in_process():
    """No new lookup: the agent has the Celsius figure and its own converter."""
    answer = agent.reply(turn(LOOKED_UP + [
        {"role": "assistant", "content": "It's overcast in Lisbon, twenty one degrees."},
        {"role": "user", "content": "what's that in fahrenheit?"},
    ]))

    assert isinstance(answer, Say), answer
    assert "71" in answer.text or "seventy" in answer.text.lower(), answer.text


def test_the_converter_is_exact():
    assert agent.to_fahrenheit.invoke({"celsius": 21.4}) == 71   # 70.52, rounded
    assert agent.to_fahrenheit.invoke({"celsius": -40}) == -40
