# Voice on top of a text agent

You already have a text agent — a prompt, some tools, a framework you like.
This puts a phone call in front of it.

AssemblyAI handles the ear and the mouth: speech to text, turn-taking,
barge-in, text to speech, the phone line. It does not decide what to say. That
stays with your agent, reached over one HTTP endpoint you already know how to
write.

```
caller ──phone/browser──▶  AssemblyAI  ──POST /v1/chat/completions──▶  your text agent
                          speech, turns                                 (this repo: PydanticAI)
                          TTS, telephony  ◀── say this / call that ──
```

The example is a weather line. Ask it for a city, it looks the weather up, and
it talks to you while the lookup runs.

```
  [  3.0s] agent:  Weather line, which place would you like?
  [  3.3s] caller: what is the weather in Reykjavik
  [ 10.4s] agent:  Let me check Reykjavik for you.
  [ 16.4s] agent:  In Reykjavik, Iceland, it's seven degrees and overcast with light winds.
```

That third line is the point — see [Talking over a tool](#talking-over-a-tool).

## Run it

You need Python 3.11+, an [AssemblyAI API key](https://www.assemblyai.com/dashboard),
and [ngrok](https://ngrok.com/download) (the platform calls your laptop, so it
needs a public address).

```bash
./setup.sh                 # venv + dependencies
# put your key in .env
.venv/bin/python run.py    # tunnel, deploy, serve
```

Then either open the [playground](https://www.assemblyai.com/playground/voice-agent)
and pick the agent it printed, or call it from a second terminal:

```bash
.venv/bin/python call.py "what is the weather in Lisbon"
```

`call.py` speaks with the macOS `say` voice, streams it in as a caller, and
writes the agent's side to `call.wav`. No phone, no browser.

One API key covers everything: the text model is reached through AssemblyAI's
own LLM gateway with the same key.

## The four files

| File | What it is |
| --- | --- |
| `brain.py` | **Your agent.** The only file you replace. |
| `agent.py` | The voice configuration, and the weather tool. |
| `run.py` | Tunnel, deploy, serve. |
| `call.py` | Call it from the terminal. |
| `phone.py` | Put it on a real phone number. |

## Bringing your own agent

`brain.py` is the adapter, and it is small on purpose. One function is the
whole contract:

```python
def reply(turn: Turn) -> Say | Call:
    ...
```

`turn` carries the conversation so far and the result of any tool that just
ran. You return one of two things: `say("...")` to speak, or
`call_tool("name", ...)` to run one of the agent's tools and answer on the next
turn.

Here that decision comes from a PydanticAI agent whose output is a union of two
models — `Speak` or `CheckWeather` — so the text model chooses by returning a
type. Nothing about the seam is PydanticAI-specific: LangGraph, an OpenAI call,
a rules engine or a `if "weather" in text` are all the same shape. Replace
`brain.py` and keep the rest.

Two things to keep from this file whatever you swap in:

- **`reply` is synchronous.** `serve()` awaits a coroutine from a tool or a
  pre-connect handler, but not from here, so an `async def reply` is silently
  never awaited. `_run()` bridges to one long-lived event loop, which also
  keeps the connection to your model warm.
- **The transcript is flattened to plain notes** before the model sees it. The
  platform's message list carries tool calls and system lines that a gateway
  will reject.

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

## A phone number

```bash
.venv/bin/python phone.py list        # what the account has
.venv/bin/python phone.py buy GB      # buy one, attach it (asks first — it charges)
.venv/bin/python phone.py attach +44… # or attach one you already own
```

## Configuration

Everything optional lives in `.env`:

| | |
| --- | --- |
| `ASSEMBLYAI_API_KEY` | Required. |
| `MODEL` | Default `claude-haiku-4-5-20251001`. |
| `LLM_GATEWAY` | Default AssemblyAI's gateway. Any OpenAI-compatible base URL works. |
| `VOICE` | Default `alba`. |
| `PUBLIC_BASE_URL` | Set it and ngrok is skipped. |

## A wart worth knowing

AssemblyAI's LLM gateway speaks the OpenAI request shape but not quite the
OpenAI response shape — it omits `id` and `object`, and passes the upstream
provider's own `finish_reason` through. PydanticAI validates strictly and
rejects that, so `brain.py` carries a small transport that fills the three
fields in. Point `LLM_GATEWAY` and `MODEL` at any OpenAI-compatible endpoint
and it becomes dead weight, harmlessly.
