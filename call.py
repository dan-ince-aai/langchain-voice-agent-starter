"""Call the agent from here, with no phone and no browser.

    python call.py "what is the weather in Lisbon" "and in fahrenheit?"

Each argument is one thing the caller says, spoken in the macOS `say` voice and
streamed in the way a caller would, one after the agent has answered the last.
The transcript prints as it arrives, so you can watch the filler land while the
tool is still running, and the agent's side is written to `call.wav`.

Needs the agent running: start `run.py` in another terminal first.
"""

import asyncio
import base64
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from assemblyai_agents import AsyncClient

from run import ID_FILE, load_env

RATE = 24000
FRAME = int(RATE * 0.02) * 2  # 20ms of 16-bit mono


def speak(sentence: str) -> bytes:
    """The caller's line, as the PCM the platform expects."""
    with tempfile.TemporaryDirectory() as tmp:
        aiff, wav = Path(tmp) / "c.aiff", Path(tmp) / "c.wav"
        subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), sentence], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1", str(aiff), str(wav)],
            check=True,
        )
        with wave.open(str(wav), "rb") as fh:
            return fh.readframes(fh.getnframes())


async def say_to(session, pcm: bytes) -> None:
    """Stream a turn in at the speed it would be spoken, then go quiet."""
    for i in range(0, len(pcm), FRAME):
        await session.send_audio(pcm[i:i + FRAME])
        await asyncio.sleep(0.02)
    # Silence, so the platform hears the caller stop.
    for _ in range(40):
        await session.send_audio(b"\x00" * FRAME)
        await asyncio.sleep(0.02)


async def main() -> int:
    load_env()
    turns = sys.argv[1:] or ["what is the weather in Lisbon"]
    if not ID_FILE.exists():
        sys.exit("No .agent_id yet — run `python run.py` first.")
    agent_id = ID_FILE.read_text().strip()

    track, started = bytearray(), time.perf_counter()
    answered = False  # has the agent finished a reply since the caller last spoke?

    def at() -> float:
        return time.perf_counter() - started

    session = await AsyncClient().sessions.connect()
    await session.update(agent_id=agent_id)
    print(f"calling {agent_id}\n")

    # The caller speaks into quiet, not on a count of replies. The client cannot
    # see the platform running a tool: after "let me check Lisbon for you" the
    # line goes silent for as long as the lookup and the next reply take, and
    # speaking into that gap talks over the answer. A person waits through it,
    # so this waits ten seconds of nothing before taking a turn, and eight with
    # nothing left to say before hanging up.
    events = session.__aiter__()
    while at() < 150:
        try:
            event = await asyncio.wait_for(events.__anext__(), timeout=10 if turns else 8)
        except asyncio.TimeoutError:
            if turns and answered:
                sentence = turns.pop(0)
                answered = False
                print(f"  [{at():5.1f}s] caller: {sentence}")
                await say_to(session, speak(sentence))
                continue
            break
        except StopAsyncIteration:
            break

        # `type` is an Enum on these models, not a str, so read its value.
        kind = getattr(event, "type", None)
        kind = getattr(kind, "value", kind)

        if kind == "reply.audio":
            track.extend(base64.b64decode(event.data))
        elif kind == "transcript.agent":
            print(f"  [{at():5.1f}s] agent:  {event.text}")
        elif kind == "session.error":
            print(f"  [{at():5.1f}s] error:  {event.message}")
        elif kind == "reply.done":
            answered = True

    await session.end()
    await session.close()

    out = Path(__file__).parent / "call.wav"
    with wave.open(str(out), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(RATE)
        fh.writeframes(bytes(track))
    print(f"\nwrote {out.name} — {len(track) / 2 / RATE:.1f}s of the agent speaking")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
