# Give your text agent a voice

You already have a text agent — a prompt, some tools, a framework you like.
This puts a phone call in front of it, and shows the same call answered by four
different frameworks so you can see how little of it is framework-specific.

| Brain | File | Lines |
| --- | --- | --- |
| [PydanticAI](https://ai.pydantic.dev) | `brains/pydantic_ai.py` | ~40 |
| [LangChain](https://python.langchain.com) | `brains/langchain.py` | ~25 |
| [Agno](https://docs.agno.com) | `brains/agno.py` | ~30 |
| [Google ADK](https://google.github.io/adk-docs) | `brains/google_adk.py` | ~50 |

```bash
BRAIN=agno .venv/bin/python run.py
```

AssemblyAI handles the ear and the mouth: speech to text, turn-taking,
barge-in, text to speech, the phone line. It does not decide what to say. Your
agent does, over one HTTP endpoint you already know how to write.

```
caller ──phone/browser──▶  AssemblyAI  ──POST /v1/chat/completions──▶  brains/<yours>.py
                          speech, turns                                 your text agent
                          TTS, telephony  ◀── say this / call that ──
```

The example is a weather line. Ask it for a city, it looks the weather up, and
it talks to you while the lookup runs:

```
  [  3.0s] agent:  Weather line, which place would you like?
  [  3.3s] caller: what is the weather in Reykjavik
  [ 10.4s] agent:  Let me check Reykjavik for you.
  [ 16.4s] agent:  In Reykjavik, Iceland, it's seven degrees and overcast with light winds.
```

That third line is the point. See [Talking over a tool](#talking-over-a-tool).

## Run it

Python 3.11+, an [AssemblyAI API key](https://www.assemblyai.com/dashboard),
and [ngrok](https://ngrok.com/download) — the platform calls your laptop, so it
needs a public address.

```bash
./setup.sh                        # venv + dependencies
# put your key in .env
.venv/bin/python run.py           # tunnel, deploy, serve — PydanticAI brain
BRAIN=langchain .venv/bin/python run.py   # or agno, google_adk
```

Then open the [playground](https://www.assemblyai.com/playground/voice-agent)
and pick the agent it printed, or call it from a second terminal:

```bash
.venv/bin/python call.py "what is the weather in Lisbon"
```

`call.py` speaks in the macOS `say` voice, streams it in as a caller would, and
writes the agent's side to `call.wav`. No phone, no browser.

One API key covers everything. The text model is reached through AssemblyAI's
LLM gateway with the same key, whichever brain is running.

## The seam

Every brain is one function:

```python
def reply(turn: Turn) -> Say | Call:
    ...
```

`turn` carries the conversation so far and the result of any tool that just
ran. You return `say("...")` to speak, or `call_tool("name", ...)` to run one of
the agent's tools and answer on the next turn.

The four brains all do it the same way: hand the model the transcript, get a
`Decision` back as structured output, translate it. What differs is only how
each framework asks for structured output — `output_type`,
`with_structured_output`, `output_schema`, `output_schema` — and whether it is
sync or async. `bridge.py` holds the parts that are the same: the `Decision`
model, the instructions, the transcript flattening, and an event-loop bridge
for the async ones.

To plug in your own agent, add `brains/mine.py` with a `reply`, and add its name
to `AVAILABLE` in `brains/__init__.py`. Then `BRAIN=mine`.

## Talking over a tool

A tool that takes a few seconds is a few seconds of silence, which a caller
reads as a dropped line. One response can carry both a tool call and words, so
the agent says something while the tool runs:

```python
return call_tool("get_weather", saying="Let me check Lisbon for you.", location="Lisbon")
```

The filler is written by the model, in context, naming the place — not a canned
"one moment". That is the reason to pair a text model with a voice one rather
than let the voice model do everything.

For work measured in tens of seconds, have the tool answer in stages and ask it
again; each answer is its own turn, so each gets its own spoken line.

## The files

| | |
| --- | --- |
| `brains/` | One file per framework. The only thing you replace. |
| `bridge.py` | What every brain shares. |
| `agent.py` | The voice configuration, and the weather tool. |
| `run.py` | Tunnel, deploy, serve. |
| `call.py` | Call it from the terminal. |
| `phone.py` | Put it on a real number: `list`, `buy GB`, `attach +44…`. |
| `tests/` | Every brain through the same two turns. `pytest -q`. |

## Configuration

All optional, in `.env`:

| | |
| --- | --- |
| `ASSEMBLYAI_API_KEY` | Required. |
| `BRAIN` | `pydantic_ai` (default), `langchain`, `agno`, `google_adk`. |
| `MODEL` | Default `claude-haiku-4-5-20251001`. |
| `LLM_GATEWAY` | Default AssemblyAI's gateway. Any OpenAI-compatible base URL works. |
| `VOICE` | Default `alba`. |
| `PUBLIC_BASE_URL` | Set it and ngrok is skipped. |

## Things that bit, so they don't bite you

- **`reply` must be synchronous.** `serve()` awaits a coroutine from a tool or
  a pre-connect handler but not from the reply endpoint; an `async def reply`
  is silently never awaited and the turn goes out empty. The async brains go
  through `bridge.run_sync`, which is also why there is exactly one event loop:
  a fresh loop per turn leaves the framework's HTTP client bound to the old one.
- **The gateway is not quite OpenAI-shaped.** It omits `id` and `object` and
  passes the upstream `finish_reason` through. Three of the four frameworks
  read only what they need and never notice; PydanticAI validates the whole
  response, so it gets a client from `bridge.py` that fills the fields in.
- **A tool that raises is silence.** The platform gets a 500 and the caller
  hears nothing, so the weather tool returns its failures instead.
- **`event.type` on the realtime models is an Enum**, not a string. `call.py`
  compares `.value`.
