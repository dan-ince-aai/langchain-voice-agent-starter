"""Microphone client for the deployed agent.

    python call.py

Requires `run.py` to be running, and the `[audio]` extra (PortAudio + pyaudio).
Ctrl-C hangs up. Use headphones: the microphone will otherwise pick up the
agent's own audio.
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
        sys.exit("No .agent_id; run `python run.py` first.")
    agent_id = ID_FILE.read_text().strip()

    async with AgentConnection(agent_id=agent_id) as call:
        call.on_ready(lambda *_: print("connected\n"))
        call.on_user_transcript(lambda event: print(f"  you:    {text_of(event)}"))
        call.on_agent_transcript(lambda event: print(f"  agent:  {text_of(event)}"))
        call.on_error(lambda event: print(f"  error:  {text_of(event)}"))
        await call.run()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nended")
    except DeviceAudioNotInstalledError as exc:
        sys.exit(f"{exc}\n  macOS: brew install portaudio && pip install pyaudio")
