"""Wykrywanie i helpery GPU (PyTorch CUDA) — fallback CPU."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np

_TORCH = None
_TORCH_ERR: str | None = None


def _load_torch():
    global _TORCH, _TORCH_ERR
    if _TORCH is not None or _TORCH_ERR is not None:
        return _TORCH
    try:
        import torch

        _TORCH = torch
        return torch
    except Exception as exc:  # noqa: BLE001
        _TORCH_ERR = str(exc)
        return None


@lru_cache(maxsize=1)
def gpu_enabled() -> bool:
    """True gdy CUDA dostępna i nie wyłączona przez BONE_GPU=0."""
    if os.environ.get("BONE_GPU", "1").strip() in ("0", "false", "off", "no"):
        return False
    torch = _load_torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def device():
    torch = _load_torch()
    if torch is None:
        return None
    return torch.device("cuda" if gpu_enabled() else "cpu")


def device_name() -> str:
    torch = _load_torch()
    if torch is None:
        return f"cpu (brak torch: {_TORCH_ERR or '?'})"
    if gpu_enabled():
        try:
            return f"cuda:{torch.cuda.get_device_name(0)}"
        except Exception:  # noqa: BLE001
            return "cuda"
    return "cpu"


def as_tensor(x: np.ndarray, *, dtype=None, device_override=None):
    torch = _load_torch()
    assert torch is not None
    dev = device_override or device()
    dt = dtype or torch.float32
    # pola structured array (traits) mają niestandardowe strides
    arr = np.ascontiguousarray(np.asarray(x))
    return torch.as_tensor(arr, dtype=dt, device=dev)


def to_numpy(t) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def gpu_info() -> dict[str, Any]:
    torch = _load_torch()
    info: dict[str, Any] = {
        "enabled": gpu_enabled(),
        "device": device_name(),
        "torch": getattr(torch, "__version__", None) if torch else None,
    }
    if torch is not None and gpu_enabled():
        try:
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / (1024**3), 2)
            info["capability"] = f"{props.major}.{props.minor}"
        except Exception:  # noqa: BLE001
            pass
    return info
