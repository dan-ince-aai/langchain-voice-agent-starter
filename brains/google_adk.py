"""The brain, in Google ADK.

ADK reaches non-Gemini models through LiteLLM, and `openai/<model>` with an
`api_base` is how LiteLLM talks to any OpenAI-compatible endpoint. The agent's
`output_schema` makes its final response the `Decision` as JSON text.

ADK is async and keeps a session per conversation; the transcript already
carries the history, so each turn gets a fresh session and the runner is
driven through the shared event loop.
"""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from bridge import GATEWAY, INSTRUCTIONS, MODEL, Decision, Turn, answer, api_key, run_sync, transcript

APP = "weather-line"
_runner: InMemoryRunner | None = None


def runner() -> InMemoryRunner:
    """Built on first use, so the module imports without a key."""
    global _runner
    if _runner is None:
        _runner = InMemoryRunner(
            agent=LlmAgent(
                name="brain",
                model=LiteLlm(model=f"openai/{MODEL}", api_base=GATEWAY, api_key=api_key()),
                instruction=INSTRUCTIONS,
                output_schema=Decision,
            ),
            app_name=APP,
        )
    return _runner


async def decide(notes: str) -> Decision:
    run = runner()
    session = await run.session_service.create_session(app_name=APP, user_id="caller")
    message = types.Content(role="user", parts=[types.Part(text=notes)])
    final = ""
    async for event in run.run_async(user_id="caller", session_id=session.id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            final = event.content.parts[0].text or ""
    return Decision.model_validate_json(final)


def reply(turn: Turn):
    return answer(run_sync(decide(transcript(turn))))
