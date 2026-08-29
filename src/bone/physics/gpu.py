"""Opcjonalne CUDA (PyTorch)."""

from __future__ import annotations

import os

_TORCH = None
_OK: bool | None = None


def gpu_enabled() -> bool:
    global _TORCH, _OK
    if _OK is not None:
        return _OK
    if os.environ.get("BONE_GPU", "1") in {"0", "false", "False"}:
        _OK = False
        return False
    try:
        import torch

        _TORCH = torch
        _OK = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        _OK = False
    return _OK


def device_label() -> str:
    if not gpu_enabled():
        return "cpu"
    assert _TORCH is not None
    name = _TORCH.cuda.get_device_name(0)
    return f"cuda:{name}"
