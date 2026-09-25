"""The voice in front of the agent: the platform's configuration, and the one
tool the platform runs rather than the agent.

The weather lookup is here and not in `agent.py` for one reason: it is slow.
Two network calls is a couple of seconds of silence on a phone line. A tool
declared here runs on the platform, which means the agent can hand back a
sentence to speak in the same response as the call — see `reply()` — so the
caller hears "let me check Lisbon for you" while the lookup is in flight.
Anything fast belongs in the agent instead.
"""

import os

import httpx
from assemblyai_agents import VoiceAgent, tool
from assemblyai_agents.models.rest import LlmConfigRequest

SECRET = os.environ.get("TOOL_SECRET", "local-dev-secret")
VOICE = os.environ.get("VOICE", "alba")


@tool(timeout_seconds=15)
async def get_weather(location: str) -> dict:
    """Look up the current weather somewhere.

    Args:
        location: The town or city the caller named, like "Lisbon".
    """
    # A tool that raises answers the platform with a 500, which the caller hears
    # as silence. Returning the failure instead lets the agent say something.
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            found = await http.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
            )
            places = (found.json() or {}).get("results") or []
            if not places:
                return {"found": False, "location": location}

            place = places[0]
            weather = await http.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,wind_speed_10m,weather_code",
                },
            )
            now = (weather.json() or {}).get("current") or {}
    except httpx.HTTPError:
        return {"found": False, "location": location, "unreachable": True}

    return {
        "found": True,
        "place": place["name"],
        "country": place.get("country", ""),
        "celsius": now.get("temperature_2m"),
        "wind_kph": now.get("wind_speed_10m"),
        "description": _CODES.get(now.get("weather_code"), "hard to describe"),
    }


# Open-Meteo returns a WMO code. The model reads the word, not the number.
_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog", 51: "drizzling", 53: "drizzling",
    55: "drizzling heavily", 61: "raining lightly", 63: "raining",
    65: "raining heavily", 71: "snowing lightly", 73: "snowing",
    75: "snowing heavily", 80: "showery", 81: "showery", 82: "pouring",
    95: "thundery", 96: "thundery with hail", 99: "thundery with hail",
}

TOOLS = [get_weather]


def build(base_url: str) -> VoiceAgent:
    """The agent as the platform will store it."""
    auth = {"name": "Authorization", "value": f"Bearer {SECRET}"}
    return VoiceAgent(
        name="Weather Line",
        voice=VOICE,
        # Replaced on every turn by agent.py, so this only has to be true.
        system_prompt="You are a friendly weather line.",
        greeting="Weather line, which place would you like?",
        tools=[t.hosted_at(f"{base_url}/tools/{t.name}", headers=[auth]) for t in TOOLS],
        llm=LlmConfigRequest(base_url=f"{base_url}/v1", model="weather-line", api_key=SECRET),
    )
