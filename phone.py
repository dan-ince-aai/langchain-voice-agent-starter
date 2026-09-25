"""Phone number management for the deployed agent.

    python phone.py list          # numbers on this account
    python phone.py buy GB        # purchase a number and assign it to this agent
    python phone.py attach +44…   # assign an existing number to this agent

`buy` charges the account and asks for confirmation.
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
        sys.exit("No .agent_id; run `python run.py` first.")
    return ID_FILE.read_text().strip()


def main() -> int:
    load_env()
    command = sys.argv[1] if len(sys.argv) > 1 else "list"
    api = Client()

    if command == "list":
        for number in api.phone_numbers.list():
            mine = "  (this agent)" if number.agent_id == agent_id() else ""
            print(f"  {number.phone_number}  agent={number.agent_id or '-'}{mine}")
        return 0

    if command == "buy":
        country = (sys.argv[2] if len(sys.argv) > 2 else "GB").upper()
        print(f"Purchase a {country} number and assign it to {agent_id()}.")
        if input("This charges the account. Type yes to continue: ").strip() != "yes":
            return 1
        bought = api.phone_numbers.purchase_available(
            PurchaseAvailablePhoneNumberRequest(
                country_code=country,
                number_type=NumberType.local,
                agent_id=agent_id(),
                label="weather line",
            )
        )
        print(f"  {bought.phone_number}")
        return 0

    if command == "attach":
        number = sys.argv[2]
        api.phone_numbers.assign_agent(number, PhoneNumberAssignAgentRequest(agent_id=agent_id()))
        print(f"  {number} -> {agent_id()}")
        return 0

    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
