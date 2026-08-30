"""Pakowanie stanu do podglądu na żywo.

Format binarny, nie JSON. Sto tysięcy cząstek jako listy liczb w JSON-ie to
kilkanaście megabajtów tekstu na klatkę i sekundy parsowania po stronie
przeglądarki; ten sam ładunek jako Float32Array to 1,6 MB i zero parsowania —
bufor wchodzi prosto do atrybutu geometrii three.js.

Układ bufora (little endian):

    magic   uint32  0x424F4E45 ("BONE")
    n       uint32  liczba wysłanych cząstek
    half    float32 połowa rozciągłości sceny (do ustawienia kamery)
    time    float32 czas symulacji
    xyz     float32[n*3]
    shade   float32[n]  β = |v|/c, gotowe do mapy kolorów
"""

from __future__ import annotations

import struct

import numpy as np

from bone.config import Config
from bone.state import State

MAGIC = 0x424F4E45
HEADER = struct.Struct("<IIff")


def pack_view(state: State, cfg: Config) -> bytes:
    stride = max(1, int(cfg.run.point_stride))
    positions = np.ascontiguousarray(state.positions[::stride], dtype=np.float32)
    shade = np.ascontiguousarray(
        state.speed_over_c(cfg.physics.c)[::stride], dtype=np.float32
    )
    n = int(positions.shape[0])
    half = float(np.abs(positions).max() * 1.05 + 1.0) if n else 1.0
    return b"".join(
        (
            HEADER.pack(MAGIC, n, half, float(state.time)),
            positions.tobytes(),
            shade.tobytes(),
        )
    )


def empty_view() -> bytes:
    return HEADER.pack(MAGIC, 0, 1.0, 0.0)
