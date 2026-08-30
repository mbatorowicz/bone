"""Wejście Vercela (asgi.py w katalogu głównym).

Pakiet leży w ``src/``, więc dokładamy go do ``sys.path`` zanim zaimportujemy
``bone`` — Vercel szuka pliku modułu entrypointu, a nie zainstalowanego
pakietu setuptools.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bone.studio.asgi import app

__all__ = ["app"]
