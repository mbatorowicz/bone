"""Dokładne O(N²) — każda para, bez odcięcia i bez list sąsiadów.

Sztuczka polega na tym, żeby nigdy nie materializować tablicy wektorów par
o kształcie (tile, N, 3). Rozwijamy kwadrat odległości

    r²_ij = |xᵢ|² − 2 xᵢ·xⱼ + |xⱼ|²

więc macierz odległości powstaje jednym mnożeniem macierzy, a suma sił

    Σⱼ wⱼ (xᵢ − xⱼ) = (Σⱼ wⱼ) xᵢ − W·X

to drugie mnożenie macierzy. Wariant naiwny (broadcast do (tile,N,3)) mierzył
0,3 Gpar/s na karcie testowej, ten przez GEMM ok. 0,6 Gpar/s przy trzykrotnie
mniejszym zużyciu pamięci.

Gdzie jest wąskie gardło: NIE w GEMM. Pomiar pokazał, że włączenie TF32 nie
zmienia czasu ani o procent (49,1 vs 49,2 ms przy N = 4000), za to psuje siłę
o 0,25% RMS i 3% w najgorszym przypadku. Koszt siedzi w przejściach
elementowych po macierzy (tile, N) — rsqrt, potęgowanie, mnożenie — czyli
w przepustowości pamięci. Dlatego TF32 zostaje wyłączony: jest darmowy tylko
kosztem dokładności, której tu nie chcemy oddawać. Dalsze przyspieszenie
wymagałoby zrośnięcia tych przejść w jedno jądro CUDA, a nie zmiany precyzji.

Człon własny j=i znosi się w sile analitycznie ((Σw)xᵢ zawiera wᵢᵢxᵢ, a W·X
zawiera dokładnie to samo), ale i tak zerujemy przekątną — jawnie, bo poleganie
na kasowaniu się dwóch dużych liczb w float32 to proszenie się o kłopoty.

Uwaga numeryczna: rozwinięcie kwadratu traci cyfry, gdy r ≪ |x|. Ratuje nas
softening — do r² dodajemy ε², które jest zwykle o rzędy wielkości większe niż
błąd obcięcia. Dlatego backend odmawia pracy przy absurdalnie małym ε.
"""

from __future__ import annotations

import numpy as np

from bone.backends.base import Backend, Field

#: budżet pamięci na jeden kafel macierzy (bajty)
_TILE_BUDGET = 256 * 1024 * 1024


def _tile_rows(n: int, itemsize: int, budget: int = _TILE_BUDGET) -> int:
    rows = budget // max(n * itemsize, 1)
    return int(max(1, min(n, rows)))


def exact_forces_for(
    positions: np.ndarray,
    masses: np.ndarray,
    G: float,
    softening: float,
    rows: np.ndarray,
) -> np.ndarray:
    """Dokładna siła działająca na wybrane cząstki ze strony WSZYSTKICH pozostałych.

    Służy do mierzenia błędu backendów przybliżonych: koszt to O(|rows|·N)
    zamiast O(N²), więc kontrolę można włączyć nawet przy milionie cząstek.
    """
    x = np.ascontiguousarray(positions, dtype=np.float64)
    m = np.ascontiguousarray(masses, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    eps2 = float(softening) ** 2
    x2 = np.einsum("ij,ij->i", x, x)
    out = np.empty((rows.size, 3), dtype=np.float64)

    chunk = max(1, _tile_rows(x.shape[0], x.dtype.itemsize))
    for start in range(0, rows.size, chunk):
        sel = rows[start : start + chunk]
        block = x[sel]
        r2 = x2[sel, None] + x2[None, :] - 2.0 * (block @ x.T)
        np.maximum(r2, 0.0, out=r2)
        r2 += eps2
        w = r2 ** -1.5
        w[np.arange(sel.size), sel] = 0.0
        w *= m[None, :]
        out[start : start + chunk] = -G * m[sel, None] * (
            w.sum(axis=1)[:, None] * block - w @ x
        )
    return out


class ExactNumpy(Backend):
    """Dokładne siły na CPU. Sensowne do kilku tysięcy cząstek."""

    name = "exact-cpu"

    def compute(self, positions, masses, G, softening) -> Field:
        x = np.ascontiguousarray(positions, dtype=np.float64)
        m = np.ascontiguousarray(masses, dtype=np.float64)
        n = x.shape[0]
        force = np.zeros_like(x)
        phi = np.zeros(n, dtype=np.float64)
        if n < 2:
            return Field(force=force, potential=phi)

        eps2 = float(softening) ** 2
        x2 = np.einsum("ij,ij->i", x, x)
        rows = _tile_rows(n, x.dtype.itemsize)

        for start in range(0, n, rows):
            stop = min(start + rows, n)
            block = x[start:stop]
            r2 = x2[start:stop, None] + x2[None, :] - 2.0 * (block @ x.T)
            np.maximum(r2, 0.0, out=r2)
            r2 += eps2

            inv_r = 1.0 / np.sqrt(r2)
            diag = np.arange(start, stop)
            inv_r[np.arange(stop - start), diag] = 0.0

            phi[start:stop] = -G * (inv_r @ m)

            w = inv_r
            w **= 3
            w *= m[None, :]
            force[start:stop] = -G * m[start:stop, None] * (
                w.sum(axis=1)[:, None] * block - w @ x
            )

        return Field(force=force, potential=phi)


class ExactTorch(Backend):
    """Dokładne siły na GPU. Ta sama formuła, ta sama konwencja znaków."""

    name = "exact-cuda"

    def __init__(self, device: str = "cuda", dtype: str = "float32") -> None:
        import torch

        self._torch = torch
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype)
        if self._device.type == "cuda":
            # TF32 psuje r² przy małym ε (10-bitowa mantysa w rozwinięciu kwadratu),
            # a nic tu nie przyspiesza — patrz pomiar w nagłówku modułu
            torch.backends.cuda.matmul.allow_tf32 = False

    def describe(self) -> str:
        if self._device.type != "cuda":
            return f"{self.name} ({self._device})"
        name = self._torch.cuda.get_device_name(self._device)
        return f"{self.name} ({name}, {str(self._dtype).split('.')[-1]})"

    def compute(self, positions, masses, G, softening) -> Field:
        torch = self._torch
        n = int(positions.shape[0])
        if n < 2:
            return Field(force=np.zeros_like(positions), potential=np.zeros(n))

        x = torch.as_tensor(positions, dtype=self._dtype, device=self._device)
        m = torch.as_tensor(masses, dtype=self._dtype, device=self._device)
        eps2 = float(softening) ** 2

        x2 = (x * x).sum(1)
        xt = x.mT.contiguous()
        force = torch.empty_like(x)
        phi = torch.empty(n, dtype=self._dtype, device=self._device)
        rows = _tile_rows(n, x.element_size())

        for start in range(0, n, rows):
            stop = min(start + rows, n)
            block = x[start:stop]
            r2 = x2[start:stop, None] + x2[None, :]
            r2.addmm_(block, xt, alpha=-2.0)
            r2.clamp_(min=0.0).add_(eps2)

            inv_r = r2.rsqrt_()
            idx = torch.arange(start, stop, device=self._device)
            inv_r[torch.arange(stop - start, device=self._device), idx] = 0.0

            phi[start:stop] = inv_r @ m
            w = inv_r.pow_(3).mul_(m)
            force[start:stop] = torch.addmm(
                w.sum(1, keepdim=True) * block, w, x, alpha=-1.0
            )

        phi.mul_(-G)
        force.mul_(-G * m[:, None])
        return Field(
            force=force.to(torch.float64).cpu().numpy(),
            potential=phi.to(torch.float64).cpu().numpy(),
        )

    def close(self) -> None:
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()
