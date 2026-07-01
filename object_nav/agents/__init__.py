"""Agent implementations for ObjectNav experiments."""

from importlib import import_module
from typing import Any

__all__ = [
    "InteractiveKeyboardAgent",
    "KeyboardControls",
    "RandomActionAgent",
    "RandomAgent",
]


def __getattr__(name: str) -> Any:
    if name in {"RandomActionAgent", "RandomAgent"}:
        random_agent = import_module("object_nav.agents.random")
        return getattr(random_agent, name)

    if name in {"InteractiveKeyboardAgent", "KeyboardControls"}:
        interactive_agent = import_module("object_nav.agents.interactive")
        return getattr(interactive_agent, name)

    raise AttributeError(f"module 'object_nav.agents' has no attribute {name!r}")
