"""Studio: serwer HTTP i frontend 3D."""

from __future__ import annotations

__all__ = ["serve"]


def serve(*args, **kwargs):
    from bone.studio.server import serve as _serve

    return _serve(*args, **kwargs)
