# Give your text agent a voice

You have a text agent — a model, a prompt, some tools, in a framework you like.
This puts a phone call in front of it.

AssemblyAI handles the ear and the mouth: speech to text, turn-taking,
barge-in, text to speech, the phone line. It does not decide what to say. Your
agent does, over one HTTP endpoint you already know how to write.

```
caller ──phone/browser──▶  AssemblyAI  ──POST /v1/chat/completions──▶  agent.py
                          speech, turns                                 your LangChain agent
                          TTS, telephony  ◀── say this / call that ──
```

The example is a weather line, and the agent is LangChain. Ask it for a city,
it looks the weather up, and it talks to you while the lookup runs:

```
  [  2.8s] agent:  Weather line, which place would you like?
  [ 13.1s] caller: what is the weather in Lisbon
  [ 21.0s] agent:  Let me check Lisbon for you.
  [ 26.4s] agent:  In Lisbon it's overcast and twenty-one degrees Celsius with a light wind.
  [ 36.4s] caller: and what is that in fahrenheit
  [ 45.4s] agent:  That's seventy degrees Fahrenheit.
```

The third line is spoken while the lookup runs. The last one never leaves the
agent: it is a LangChain tool, called in-process, and the platform is not
involved.

## Run it

Python 3.11+, an [AssemblyAI API key](https://www.assemblyai.com/dashboard),
and [ngrok](https://ngrok.com/download) — the platform calls your laptop, so it
needs a public address.

```bash
./setup.sh                 # venv + dependencies
# put your key in .env
.venv/bin/python run.py    # tunnel, deploy, serve
```

Then open the [playground](https://www.assemblyai.com/playground/voice-agent)
and pick the agent it printed, or call it from a second terminal:

```bash
.venv/bin/python call.py "what is the weather in Lisbon"
```

`call.py` speaks in the macOS `say` voice, streams it in as a caller would, and
writes the agent's side to `call.wav`. No phone, no browser.

One API key covers everything: the text model is reached through AssemblyAI's
LLM gateway with the same key.

## Two files matter

**`agent.py` is your agent.** It is an ordinary LangChain agent — `create_agent`
with a model, a system prompt, its own tools and a structured answer — plus one
function that is the whole contract with the voice:

```python
def reply(turn: Turn) -> Say | Call:
    ...
```

`turn` carries the conversation so far and the result of any platform tool that
just ran. You return `say("...")` to speak, or `call_tool("name", ...)` to run
one of the platform's tools and answer on the next turn. Swap LangChain for
anything else and that function is still the seam.

**`voice.py` is the platform's side.** The voice, the greeting, and the tools
the platform runs itself.

## Which tools go where

This is the design decision the example exists to show.

**Fast tools live in your agent.** `to_fahrenheit` is a LangChain tool. The
agent calls it mid-turn, in-process, in a millisecond, and the caller hears only
the answer. That is what a framework buys you over a bare model call: the
agent can take several steps and use several tools before it decides what to
say, and none of it touches the voice.

**Slow tools live on the platform.** The weather lookup is two network calls —
a couple of seconds of silence on a phone line, which a caller reads as a
dropped call. Declared in `voice.py`, it runs on the platform, and that lets the
agent hand back words in the same response as the call:

```python
return call_tool("get_weather", saying="Let me check Lisbon for you.", location="Lisbon")
```

The filler is written by the model, in context, naming the place — not a canned
"one moment". For work measured in tens of seconds, have the tool answer in
stages and ask it again; each answer is its own turn, so each gets its own line.

## The other files

| | |
| --- | --- |
| `run.py` | Tunnel, deploy, serve. |
| `call.py` | Call it from the terminal. |
| `phone.py` | Put it on a real number: `list`, `buy GB`, `attach +44…`. |
| `tests/` | The turns a call actually has. `pytest -q`. |

## Configuration

All optional, in `.env`:

| | |
| --- | --- |
| `ASSEMBLYAI_API_KEY` | Required. |
| `MODEL` | Default `claude-haiku-4-5-20251001`. |
| `LLM_GATEWAY` | Default AssemblyAI's gateway. Any OpenAI-compatible base URL works. |
| `VOICE` | Default `alba`. |
| `PUBLIC_BASE_URL` | Set it and ngrok is skipped. |

## Things that bit, so they don't bite you

- **`reply` must be synchronous.** `serve()` awaits a coroutine from a tool or
  a pre-connect handler but not from the reply endpoint; an `async def reply`
  is silently never awaited and the turn goes out empty.
- **Flatten the transcript.** The platform's message list carries its own tool
  calls, which your agent was never given and will reject. `agent.py` turns it
  into plain notes.
- **A tool that raises is silence.** The platform gets a 500 and the caller
  hears nothing, so the weather tool returns its failures instead.
- **`event.type` on the realtime models is an Enum**, not a string. `call.py`
  compares `.value`.
