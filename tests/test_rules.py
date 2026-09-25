"""The loop around the model, with the model taken out.

No key, no network. `propose` is replaced with a stand-in, so what these pin
is the agent's behaviour when the model is wrong, slow, or broken — which is
the part that makes it an agent rather than a gateway.

    .venv/bin/python -m pytest -q tests/test_rules.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
from agent import Decision, audit, fresh  # noqa: E402

LISBON = {"lisbon": {"found": True, "place": "Lisbon", "celsius": 21.4, "description": "overcast"}}


# --------------------------------------------------------------- the rules


def test_a_temperature_with_nothing_looked_up_is_rejected():
    assert audit(Decision(action="speak", text="It's twenty degrees in Lisbon."), known={})


def test_a_temperature_from_a_real_lookup_passes():
    assert audit(Decision(action="speak", text="It's twenty-one degrees in Lisbon."), known=LISBON) == []


def test_a_known_place_is_not_looked_up_again():
    assert audit(Decision(action="check_weather", location="Lisbon", filler="One moment."), known=LISBON)


def test_a_new_place_is_looked_up():
    assert audit(Decision(action="check_weather", location="Porto", filler="Checking Porto."), known=LISBON) == []


def test_a_missing_filler_is_supplied_not_rejected():
    proposal = Decision(action="check_weather", location="Lisbon")

    assert audit(proposal, known={}) == []
    assert "Lisbon" in proposal.filler


def test_a_lookup_with_no_place_is_rejected():
    assert audit(Decision(action="check_weather", location="  "), known={})


def test_markdown_is_rejected():
    assert audit(Decision(action="speak", text="**Lisbon**: overcast"), known=LISBON)


def test_an_empty_answer_is_rejected():
    assert audit(Decision(action="speak", text=""), known={})


# --------------------------------------------------------------- the loop


def test_a_corrected_second_proposal_is_spoken(monkeypatch):
    proposals = iter([
        Decision(action="speak", text="It's thirty degrees."),   # invented: nothing looked up
        Decision(action="speak", text="Which place would you like?"),
    ])
    monkeypatch.setattr(agent, "propose", lambda messages: next(proposals))

    out = agent.graph.invoke(fresh("Caller: hi", known={}))

    assert out["proposal"].text == "Which place would you like?"
    assert out["attempts"] == 2


def test_the_rejection_reason_reaches_the_second_attempt(monkeypatch):
    seen = []

    def stand_in(messages):
        seen.append(messages)
        return Decision(action="speak", text="It's thirty degrees.")

    monkeypatch.setattr(agent, "propose", stand_in)
    agent.graph.invoke(fresh("Caller: hi", known={}))

    assert len(seen) == 2
    assert "rejected" in seen[1][-1][1]


def test_two_bad_proposals_become_the_fallback(monkeypatch):
    monkeypatch.setattr(agent, "propose", lambda messages: Decision(action="speak", text="It's thirty degrees."))

    out = agent.graph.invoke(fresh("Caller: hi", known={}))

    assert out["proposal"].text == agent.FALLBACK
    assert out["attempts"] == agent.MAX_ATTEMPTS


def test_a_model_failure_is_a_sentence_not_silence(monkeypatch):
    def broken(messages):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(agent, "propose", broken)

    out = agent.graph.invoke(fresh("Caller: hi", known={}))

    assert out["proposal"].text == agent.FALLBACK


def test_a_hung_model_is_a_sentence_within_the_budget(monkeypatch):
    import time

    def hung(messages):
        time.sleep(30)

    monkeypatch.setattr(agent, "propose", hung)
    monkeypatch.setattr(agent, "REPLY_BUDGET_SECONDS", 0.2)
    from assemblyai_agents.byo import Say, Turn

    turn = Turn.from_request({"model": "weather-line", "stream": True, "tools": [],
                              "messages": [{"role": "user", "content": "hi"}]})
    started = time.perf_counter()
    answer = agent.reply(turn)

    assert isinstance(answer, Say) and answer.text == agent.FALLBACK
    assert time.perf_counter() - started < 2
