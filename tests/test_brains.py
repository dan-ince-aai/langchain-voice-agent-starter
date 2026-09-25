"""Every brain, through the same two turns.

These talk to the real model, so they need ASSEMBLYAI_API_KEY and skip without
it. What they pin is the contract each brain has to meet, not the wording:
a place the caller names becomes a tool call with a filler that names it, and
a result in hand becomes a spoken answer that uses it.

    .venv/bin/python -m pytest -q                 # all brains
    .venv/bin/python -m pytest -q -k langchain    # one
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assemblyai_agents.byo import Call, Say, Turn  # noqa: E402

import brains  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.environ.get("ASSEMBLYAI_API_KEY"), reason="needs ASSEMBLYAI_API_KEY"
)

LISBON = {"found": True, "place": "Lisbon", "country": "Portugal",
          "celsius": 21.4, "wind_kph": 11.6, "description": "overcast"}


def turn(messages) -> Turn:
    return Turn.from_request({"model": "weather-brain", "stream": True, "tools": [], "messages": messages})


ASKED = [{"role": "user", "content": "what's the weather in Lisbon?"}]
ANSWERED = ASKED + [
    {"role": "assistant", "content": "Let me check Lisbon for you.",
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"location": "Lisbon"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": json.dumps(LISBON)},
]


@pytest.mark.parametrize("name", brains.AVAILABLE)
def test_a_named_place_becomes_a_lookup_with_a_filler(name):
    answer = brains.load(name)(turn(ASKED))

    assert isinstance(answer, Call), answer
    assert answer.name == "get_weather"
    assert "lisbon" in answer.arguments["location"].lower()
    assert answer.saying and "lisbon" in answer.saying.lower(), answer.saying


@pytest.mark.parametrize("name", brains.AVAILABLE)
def test_a_result_in_hand_becomes_a_spoken_answer(name):
    answer = brains.load(name)(turn(ANSWERED))

    assert isinstance(answer, Say), answer
    spoken = answer.text.lower()
    assert "lisbon" in spoken
    assert "overcast" in spoken or "twenty" in spoken or "21" in spoken, answer.text
