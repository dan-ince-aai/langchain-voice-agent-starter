"""Call the agent from here, with no phone and no browser.

    python call.py "what is the weather in Lisbon"

Speaks the sentence in the macOS `say` voice, streams it to the agent the way a
caller would, and writes what the agent said back to `call.wav`. The transcript
prints as it arrives, so you can watch the filler land while the tool is still
running.

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
    sentence = " ".join(sys.argv[1:]) or "what is the weather in Lisbon"
    if not ID_FILE.exists():
        sys.exit("No .agent_id yet — run `python run.py` first.")

    caller = speak(sentence)
    track, started, spoken = bytearray(), time.perf_counter(), False

    def at() -> float:
        return time.perf_counter() - started

    session = await AsyncClient().sessions.connect()
    await session.update(agent_id=ID_FILE.read_text().strip())
    print(f"calling {ID_FILE.read_text().strip()}\n")

    # Stop on quiet rather than on a count of replies: a transcript can arrive
    # after the `reply.done` it belongs to, so counting clips the last line.
    events = session.__aiter__()
    while at() < 90:
        try:
            event = await asyncio.wait_for(events.__anext__(), timeout=8)
        except (asyncio.TimeoutError, StopAsyncIteration):
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
        elif kind == "reply.done" and not spoken:
            spoken = True
            print(f"  [{at():5.1f}s] caller: {sentence}")
            await say_to(session, caller)

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
