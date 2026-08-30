"""Wybór backendu sił.

`auto` bierze pod uwagę liczbę cząstek i dostępność CUDA. Reguła jest prosta:
dopóki dokładne O(N²) mieści się w budżecie czasu, liczymy dokładnie; powyżej
przechodzimy na siatkę, która obejmuje wszystkie pary kosztem rozdzielczości.
"""

from __future__ import annotations

import os

from bone.backends.base import Backend, Field
from bone.backends.exact import ExactNumpy, ExactTorch
from bone.backends.mesh import MeshBackend

__all__ = ["Backend", "Field", "ExactNumpy", "ExactTorch", "MeshBackend", "make_backend", "cuda_available", "cuda_name"]

_CUDA: bool | None = None
_CUDA_NAME = "brak"


def cuda_available() -> bool:
    global _CUDA, _CUDA_NAME
    if _CUDA is not None:
        return _CUDA
    if os.environ.get("BONE_NO_GPU", "") not in {"", "0", "false"}:
        _CUDA = False
        return False
    try:
        import torch

        _CUDA = bool(torch.cuda.is_available())
        if _CUDA:
            _CUDA_NAME = torch.cuda.get_device_name(0)
    except Exception:
        _CUDA = False
    return _CUDA


def cuda_name() -> str:
    cuda_available()
    return _CUDA_NAME


def make_backend(cfg, n_particles: int) -> Backend:
    """Zbuduj backend zgodnie z konfiguracją solvera."""
    solver = cfg.solver
    use_cuda = solver.device == "cuda" or (solver.device == "auto" and cuda_available())
    if solver.device == "cuda" and not cuda_available():
        raise RuntimeError("wymuszono device='cuda', ale CUDA jest niedostępna")

    kind = solver.backend
    if kind == "auto":
        kind = "exact" if n_particles <= solver.exact_max_particles else "mesh"

    if kind == "exact":
        if use_cuda:
            return ExactTorch(device="cuda", dtype="float32")
        return ExactNumpy()
    if kind == "mesh":
        return MeshBackend(
            grid=solver.grid,
            margin=solver.box_margin,
            device="cuda" if use_cuda else "cpu",
            dtype="float32" if use_cuda else "float64",
        )
    raise ValueError(f"nieznany backend: {solver.backend!r}")


def reference_backend() -> Backend:
    """Backend wzorcowy do mierzenia błędu — zawsze dokładny, zawsze na CPU w float64."""
    return ExactNumpy()
