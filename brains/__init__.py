"""One brain per framework. Every one exports `reply(turn) -> Say | Call`."""

import importlib

AVAILABLE = ("pydantic_ai", "langchain", "agno", "google_adk")


def load(name: str):
    """The `reply` function of the named brain."""
    if name not in AVAILABLE:
        raise SystemExit(f"BRAIN={name!r} is not one of: {', '.join(AVAILABLE)}")
    return importlib.import_module(f"brains.{name}").reply
