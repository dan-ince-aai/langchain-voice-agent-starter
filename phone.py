"""Put the agent on a real phone number.

    python phone.py list          # numbers on this account
    python phone.py buy GB        # buy one and point it at this agent
    python phone.py attach +44…   # point a number you already own at it

Buying a number costs money on your AssemblyAI account, so `buy` asks first.
Everything else in this project is free to run.
"""

import sys

from assemblyai_agents import Client
from assemblyai_agents.models.rest import (
    NumberType,
    PhoneNumberAssignAgentRequest,
    PurchaseAvailablePhoneNumberRequest,
)

from run import ID_FILE, load_env


def agent_id() -> str:
    if not ID_FILE.exists():
        sys.exit("No .agent_id yet — run `python run.py` first.")
    return ID_FILE.read_text().strip()


def main() -> int:
    load_env()
    command = sys.argv[1] if len(sys.argv) > 1 else "list"
    api = Client()

    if command == "list":
        for number in api.phone_numbers.list():
            mine = " <- this agent" if number.agent_id == agent_id() else ""
            print(f"  {number.phone_number}  agent={number.agent_id or '-'}{mine}")
        return 0

    if command == "buy":
        country = (sys.argv[2] if len(sys.argv) > 2 else "GB").upper()
        print(f"Buying a {country} number and attaching it to {agent_id()}.")
        if input("This charges your account. Type yes to continue: ").strip() != "yes":
            return 1
        bought = api.phone_numbers.purchase_available(
            PurchaseAvailablePhoneNumberRequest(
                country_code=country,
                number_type=NumberType.local,
                agent_id=agent_id(),
                label="weather line",
            )
        )
        print(f"  bought {bought.phone_number} — call it.")
        return 0

    if command == "attach":
        number = sys.argv[2]
        api.phone_numbers.assign_agent(
            number, PhoneNumberAssignAgentRequest(agent_id=agent_id())
        )
        print(f"  {number} now rings this agent.")
        return 0

    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
