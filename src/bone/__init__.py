"""Bone — grawitacja N-ciał z kinematyką szczególnej teorii względności.

Otwarta przestrzeń bez ścian, wszystkie pary w oddziaływaniu, a jakość wyniku
jest mierzona, nie zakładana.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["Config", "Engine", "State", "__version__"]


def __getattr__(name: str):
    # leniwy import: `import bone` nie ma powodu ciągnąć numpy ani torcha
    if name == "Config":
        from bone.config import Config

        return Config
    if name == "Engine":
        from bone.engine import Engine

        return Engine
    if name == "State":
        from bone.state import State

        return State
    raise AttributeError(f"module 'bone' has no attribute {name!r}")
