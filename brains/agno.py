"""The brain, in Agno.

`output_schema=Decision` makes the agent answer with the model itself.
`OpenAILike` is Agno's client for any OpenAI-compatible endpoint, which the
gateway is. Synchronous: `Agent.run` needs no event loop.
"""

from agno.agent import Agent
from agno.models.openai.like import OpenAILike

from bridge import GATEWAY, INSTRUCTIONS, MODEL, Decision, Turn, answer, api_key, transcript

_agent: Agent | None = None


def agent() -> Agent:
    """Built on first use, so the module imports without a key."""
    global _agent
    if _agent is None:
        _agent = Agent(
            model=OpenAILike(id=MODEL, base_url=GATEWAY, api_key=api_key()),
            instructions=INSTRUCTIONS,
            output_schema=Decision,
            markdown=False,
        )
    return _agent


def reply(turn: Turn):
    decision = agent().run(transcript(turn)).content
    return answer(decision)
