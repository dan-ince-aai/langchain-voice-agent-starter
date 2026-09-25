"""End-to-end turns against the real model. Skipped without ASSEMBLYAI_API_KEY."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assemblyai_agents.byo import Call, Say, Turn  # noqa: E402

import agent  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("ASSEMBLYAI_API_KEY"), reason="needs ASSEMBLYAI_API_KEY")

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


def test_named_place_produces_lookup_with_filler():
    answer = agent.reply(turn(ASKED))

    assert isinstance(answer, Call), answer
    assert answer.name == "get_weather"
    assert "lisbon" in answer.arguments["location"].lower()
    assert "lisbon" in answer.saying.lower(), answer.saying


def test_lookup_result_produces_spoken_answer():
    answer = agent.reply(turn(LOOKED_UP))

    assert isinstance(answer, Say), answer
    assert "lisbon" in answer.text.lower()
    assert "overcast" in answer.text.lower() or "twenty" in answer.text.lower(), answer.text


def test_fahrenheit_is_answered_with_agent_tool():
    answer = agent.reply(turn(LOOKED_UP + [
        {"role": "assistant", "content": "It's overcast in Lisbon, twenty one degrees."},
        {"role": "user", "content": "what's that in fahrenheit?"},
    ]))

    assert isinstance(answer, Say), answer
    assert "71" in answer.text or "seventy" in answer.text.lower(), answer.text


def test_to_fahrenheit():
    assert agent.to_fahrenheit(21.4) == 71
    assert agent.to_fahrenheit(-40) == -40
