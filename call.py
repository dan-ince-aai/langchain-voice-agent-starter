"""Talk to the agent. Microphone in, speaker out.

    python call.py

Say something when it greets you. Ctrl-C hangs up. Needs the agent running:
start `run.py` in another terminal first.

Everything real happens in the SDK's `AgentConnection`: it mints a session
token, opens the WebSocket, binds the deployed agent, pumps the microphone in
and the agent's voice out, and stops the voice when you interrupt it. This file
only prints the transcript.
"""

import asyncio
import sys

from assemblyai_agents import DeviceAudioNotInstalledError
from assemblyai_agents.connection import AgentConnection

from run import ID_FILE, load_env


def text_of(event) -> str:
    return getattr(event, "text", None) or getattr(event, "message", None) or str(event)


async def main() -> int:
    load_env()
    if not ID_FILE.exists():
        sys.exit("No .agent_id yet — run `python run.py` first.")
    agent_id = ID_FILE.read_text().strip()

    async with AgentConnection(agent_id=agent_id) as call:
        call.on_ready(lambda *_: print("connected — go ahead and talk\n"))
        call.on_user_transcript(lambda event: print(f"  you:    {text_of(event)}"))
        call.on_agent_transcript(lambda event: print(f"  agent:  {text_of(event)}"))
        call.on_error(lambda event: print(f"  error:  {text_of(event)}"))
        await call.run()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nhung up")
    except DeviceAudioNotInstalledError as exc:
        sys.exit(f"{exc}\n  macOS: brew install portaudio && pip install pyaudio")
