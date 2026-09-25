# Give your text agent a voice

You have a text agent. This makes it answer the phone.

AssemblyAI does the listening, the talking, and the phone line. Your agent
decides what to say. It's a LangChain agent here, but any framework works the
same way.

```
  agent:  Weather line, which place would you like?
  caller: what is the weather in Lisbon
  agent:  Let me check Lisbon for you.              ← said while the lookup runs
  agent:  In Lisbon it's overcast and twenty-one degrees Celsius.
  caller: and what is that in fahrenheit
  agent:  That's seventy degrees Fahrenheit.        ← worked out inside the agent
```

## Run it

```bash
./setup.sh                                   # needs Python 3.11+ and ngrok
# put your AssemblyAI API key in .env
.venv/bin/python run.py                      # you're live
.venv/bin/python call.py                     # talk to it — mic in, speaker out
```

One key does everything. Add a real phone number with `python phone.py buy GB`.

## How it works

Two files.

**`agent.py` is your agent.** A normal LangChain agent with one extra function:

```python
def reply(turn) -> Say | Call
```

The platform sends what the caller said; you send back words to speak, or a
tool to run. That's the whole integration.

**`voice.py` is the phone.** The voice, the greeting, and the weather lookup.

## The one idea worth knowing

Put **fast** tools in your agent and **slow** tools on the platform.

The Fahrenheit conversion is a LangChain tool. The agent uses it mid-thought
and the caller just hears the answer. The weather lookup is slow, so it's a
platform tool — and a platform tool can be *talked over*:

```python
return call_tool("get_weather", saying="Let me check Lisbon for you.", location="Lisbon")
```

Silence on a phone sounds like a dropped call. This is how you avoid it.

## Files

| | |
| --- | --- |
| `agent.py` | Your agent. The only file you'd rewrite. |
| `voice.py` | Voice, greeting, platform tools. |
| `run.py` | Tunnel, deploy, serve. |
| `call.py` | Talk to it from your laptop. |
| `phone.py` | Buy or attach a number. |
