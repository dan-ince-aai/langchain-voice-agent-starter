"""Start everything: a public address, the agent, and this process serving it.

    python run.py                  # PydanticAI brain
    BRAIN=langchain python run.py  # or agno, google_adk

The platform reaches your laptop over HTTPS -- a phone call has no client on
the other end for it to ask -- so this opens an ngrok tunnel, points a stored
agent at it, and serves the tool and the reply endpoint on the same port.
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
    """A public HTTPS address for `port`, for as long as the block runs.

    `PUBLIC_BASE_URL` skips this entirely, for a staging host or a real deploy.
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
        # Outside the retry loop on purpose: an exception raised by the body of
        # the `with` is thrown back in at the yield, and a broad `except` around
        # it would swallow the caller's error and start another tunnel.
        yield address
    finally:
        process.terminate()


def _ngrok_address() -> str:
    """The https address of the running tunnel, once it has one."""
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
    """Create the agent, or update the one this project already made."""
    from assemblyai_agents import Client, NotFoundError

    from agent import build

    api = Client()
    declared = build(base_url)
    stored = ID_FILE.read_text().strip() if ID_FILE.exists() else ""
    if stored:
        try:
            api.agents.update(stored, declared)
            print(f"agent   {stored} (updated)")
            return stored
        except NotFoundError:
            print(f"agent   {stored} is gone, creating another")
    created = api.agents.create(declared)
    ID_FILE.write_text(created.id)
    print(f"agent   {created.id} (created)")
    return created.id


def main() -> int:
    load_env()
    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        sys.exit(
            "No ASSEMBLYAI_API_KEY.\n"
            "  cp .env.example .env, then put your key in it."
        )
    os.environ.setdefault("TOOL_SECRET", "local-dev-secret")

    from assemblyai_agents.serving import serve

    import agent
    import bridge
    import brains

    brain = os.environ.get("BRAIN", "pydantic_ai")
    reply = brains.load(brain)

    with tunnel(PORT) as base_url:
        agent_id = deploy(base_url)
        secret = os.environ["TOOL_SECRET"]
        print(f"brain   {brain}  (others: {', '.join(b for b in brains.AVAILABLE if b != brain)})")
        print(f"model   {bridge.MODEL} via {bridge.GATEWAY}")
        print(f"\nTalk to it:  https://www.assemblyai.com/playground/voice-agent  (agent {agent_id})")
        print("Or attach a phone number to that agent and call it.\n")
        serve(
            agent.build(base_url),
            reply=reply,
            port=PORT,
            tool_secret=secret,
            llm_key=secret,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
