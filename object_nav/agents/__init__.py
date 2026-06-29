"""Agent implementations for ObjectNav experiments."""

from importlib import import_module
from typing import Any

__all__ = ["RandomAgent"]


def __getattr__(name: str) -> Any:
    if name == "RandomAgent":
        random_agent = import_module("object_nav.agents.random")
        return random_agent.RandomAgent

    raise AttributeError(f"module 'object_nav.agents' has no attribute {name!r}")
