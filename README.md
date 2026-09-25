# langchain-voice-agent-starter

A LangChain/LangGraph agent answering a phone call through the AssemblyAI
Voice Agents API. The platform handles speech-to-text, turn detection,
barge-in, text-to-speech and telephony; the agent decides what to say.

```
  agent:  Weather line, which place would you like?
  caller: what is the weather in Lisbon
  agent:  Let me check Lisbon for you.            # spoken while the platform tool runs
  agent:  In Lisbon it's overcast and twenty-one degrees Celsius.
  caller: and what is that in fahrenheit
  agent:  That's seventy degrees Fahrenheit.      # local LangChain tool, no platform round trip
```

## Requirements

- Python 3.11+
- An AssemblyAI API key (also used for the LLM gateway)
- ngrok, or a public HTTPS address in `PUBLIC_BASE_URL`
- For `call.py`: PortAudio (`brew install portaudio`)

## Run

```bash
./setup.sh
# set ASSEMBLYAI_API_KEY in .env
.venv/bin/python run.py      # tunnel, deploy, serve
.venv/bin/python call.py     # microphone client, in a second terminal
```

`run.py` prints the agent id; it can also be used from the
[playground](https://www.assemblyai.com/playground/voice-agent). Use
headphones with `call.py`.

## Layout

| File | Purpose |
| --- | --- |
| `agent.py` | LangGraph agent and the reply endpoint callback. |
| `voice.py` | Platform configuration: voice, greeting, platform tools. |
| `run.py` | Tunnel, create/update the agent, serve. |
| `call.py` | Microphone client. |
| `phone.py` | Phone numbers: `list`, `buy <country>`, `attach <number>`. |
| `tests/` | `test_rules.py` runs offline; `test_agent.py` needs a key. |

## How it works

The platform calls `POST /v1/chat/completions` on this process for every turn.
`agent.reply(turn)` returns either `say(text)` or `call_tool(name, ...)`. This is
the whole integration surface; the agent behind it can be anything.

`agent.py` implements it as a LangGraph graph:

```
think -> check -> END
           |-> think     rejected; reason appended; up to MAX_ATTEMPTS
           |-> fallback  after MAX_ATTEMPTS, on model error, or on timeout
```

- `think` runs a LangChain agent (`create_agent`) with structured output
  (`Decision`) and local tools (`to_fahrenheit`).
- `check` validates the proposal with `audit()`: no temperature without a
  lookup in the transcript, no repeat lookup of a known place, no empty or
  markdown text. A missing filler sentence is defaulted.
- `reply` enforces `REPLY_BUDGET_SECONDS` and returns `FALLBACK` on timeout.

### Tool placement

`get_weather` is a platform tool (`voice.py`), not a LangChain tool, because a
platform tool call can carry speech in the same response:

```python
return call_tool("get_weather", saying="Let me check Lisbon for you.", location="Lisbon")
```

The platform speaks `saying` while it runs the tool, then calls the reply
endpoint again with the result. Tools that complete quickly (`to_fahrenheit`)
run inside the LangChain agent and never involve the platform.

### Transcript handling

The platform sends the full conversation each turn, including its own tool
calls. `transcript()` flattens it to text and retains tool results, so later
turns can reference earlier lookups without re-running them.

## Configuration

| Variable | Default |
| --- | --- |
| `ASSEMBLYAI_API_KEY` | required |
| `MODEL` | `claude-haiku-4-5-20251001` |
| `LLM_GATEWAY` | `https://llm-gateway.assemblyai.com/v1` (any OpenAI-compatible endpoint) |
| `PUBLIC_BASE_URL` | unset (ngrok) |
| `TOOL_SECRET` | `local-dev-secret` |

## Tests

```bash
.venv/bin/python -m pytest -q tests/test_rules.py   # offline
.venv/bin/python -m pytest -q                       # includes model-backed turns
```

## Notes

- The reply callback must be synchronous; `serve()` does not await it.
- Platform tools should return errors as values. An exception produces a 500
  and no speech.
- `event.type` on realtime event models is an `Enum`; compare `.value`.
