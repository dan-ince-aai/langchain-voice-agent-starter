"""The brain, in LangChain.

`with_structured_output` binds `Decision` as the one tool the model may call,
so the decision comes back as a model. Synchronous end to end: `ChatOpenAI`
has a sync client, so no event-loop bridge is needed here.
"""

from langchain_openai import ChatOpenAI

from bridge import GATEWAY, INSTRUCTIONS, MODEL, Decision, Turn, answer, api_key, transcript

_llm = None


def llm():
    """Built on first use, so the module imports without a key."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=MODEL, base_url=GATEWAY, api_key=api_key()).with_structured_output(
            Decision, method="function_calling"
        )
    return _llm


def reply(turn: Turn):
    decision = llm().invoke([("system", INSTRUCTIONS), ("human", transcript(turn))])
    return answer(decision)
