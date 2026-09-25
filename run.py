"""Start the agent locally.

    python run.py

Opens an ngrok tunnel (unless PUBLIC_BASE_URL is set), creates or updates the
agent on the platform with that address, and serves the tool and reply
endpoints on PORT.
"""

import contextlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(os.environ.get("PORT", "8000"))
ID_FILE = Path(__file__).parent / ".agent_id"


def load_env() -> None:
    env = Path(__file__).parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


@contextlib.contextmanager
def tunnel(port: int):
    """Public HTTPS address for `port` for the duration of the block.

    Set PUBLIC_BASE_URL to skip ngrok.
    """
    if os.environ.get("PUBLIC_BASE_URL"):
        yield os.environ["PUBLIC_BASE_URL"].rstrip("/")
        return

    process = subprocess.Popen(
        ["ngrok", "http", str(port), "--log", "stdout"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        address = _ngrok_address()
        if not address:
            raise SystemExit("ngrok did not come up. Is it installed and authenticated?")
        print(f"tunnel  {address} -> :{port}")
        # Keep the yield outside the retry loop: an exception from the `with`
        # body is raised at the yield and must not be caught by the retry.
        yield address
    finally:
        process.terminate()


def _ngrok_address() -> str:
    """HTTPS address of the local ngrok tunnel, polling until available."""
    for _ in range(40):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as raw:
                for entry in json.load(raw).get("tunnels", []):
                    if entry.get("public_url", "").startswith("https://"):
                        return entry["public_url"]
        except Exception:
            continue
    return ""


def deploy(base_url: str) -> str:
    """Create the agent, or update the one recorded in .agent_id."""
    from assemblyai_agents import Client, NotFoundError

    from voice import build

    api = Client()
    declared = build(base_url)
    stored = ID_FILE.read_text().strip() if ID_FILE.exists() else ""
    if stored:
        try:
            api.agents.update(stored, declared)
            print(f"agent   {stored} (updated)")
            return stored
        except NotFoundError:
            print(f"agent   {stored} not found; creating")
    created = api.agents.create(declared)
    ID_FILE.write_text(created.id)
    print(f"agent   {created.id} (created)")
    return created.id


def main() -> int:
    load_env()
    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        sys.exit("ASSEMBLYAI_API_KEY is not set. Copy .env.example to .env and set it.")
    os.environ.setdefault("TOOL_SECRET", "local-dev-secret")

    from assemblyai_agents.serving import serve

    import agent
    import voice

    with tunnel(PORT) as base_url:
        agent_id = deploy(base_url)
        secret = os.environ["TOOL_SECRET"]
        print(f"model   {agent.MODEL} via {agent.GATEWAY}")
        print(f"\nplayground: https://www.assemblyai.com/playground/voice-agent  (agent {agent_id})")
        print("local mic:  python call.py\n")
        serve(
            voice.build(base_url),
            reply=agent.reply,
            port=PORT,
            tool_secret=secret,
            llm_key=secret,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
