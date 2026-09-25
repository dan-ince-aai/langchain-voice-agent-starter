"""The brain, in PydanticAI.

Structured output does the deciding: the agent's `output_type` is `Decision`,
so the model chooses by filling in a model rather than by writing prose we
would then have to parse.
"""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from bridge import (
    GATEWAY, INSTRUCTIONS, MODEL, Decision, Turn,
    answer, api_key, openai_shaped_client, run_sync, transcript,
)

_agent: Agent | None = None


def agent() -> Agent:
    """Built on first use, so the module imports without a key."""
    global _agent
    if _agent is None:
        _agent = Agent(
            OpenAIChatModel(
                MODEL,
                provider=OpenAIProvider(
                    base_url=GATEWAY,
                    api_key=api_key(),
                    # PydanticAI validates the whole response; see bridge.py.
                    http_client=openai_shaped_client(),
                ),
            ),
            output_type=Decision,
            instructions=INSTRUCTIONS,
        )
    return _agent


def reply(turn: Turn):
    decision = run_sync(agent().run(transcript(turn))).output
    return answer(decision)
