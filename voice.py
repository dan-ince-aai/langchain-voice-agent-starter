"""Platform configuration: voice, greeting, and platform-hosted tools.

`get_weather` is declared here as a platform tool rather than as a LangChain
tool so that `reply()` can return speech in the same response as the tool call
(`call_tool(..., saying=...)`). Tools that complete quickly belong in
`agent.py` instead.
"""

import os

import httpx
from assemblyai_agents import VoiceAgent, tool
from assemblyai_agents.models.rest import LlmConfigRequest

SECRET = os.environ.get("TOOL_SECRET", "local-dev-secret")
VOICE = os.environ.get("VOICE", "alba")


@tool(timeout_seconds=15)
async def get_weather(location: str) -> dict:
    """Current weather for a place.

    Args:
        location: Town or city name as spoken by the caller.
    """
    # Return failures rather than raising: an exception becomes a 500 and the
    # caller hears nothing.
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
        "description": _CODES.get(now.get("weather_code"), "unknown"),
    }


# WMO weather codes, as words for speech.
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
    """Agent definition as stored on the platform."""
    auth = {"name": "Authorization", "value": f"Bearer {SECRET}"}
    return VoiceAgent(
        name="Weather Line",
        voice=VOICE,
        # Not used for generation: every turn is answered by agent.reply.
        system_prompt="You are a weather line.",
        greeting="Weather line, which place would you like?",
        tools=[t.hosted_at(f"{base_url}/tools/{t.name}", headers=[auth]) for t in TOOLS],
        llm=LlmConfigRequest(base_url=f"{base_url}/v1", model="weather-line", api_key=SECRET),
    )
