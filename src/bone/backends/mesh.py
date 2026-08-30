"""Particle-Mesh z izolowanymi brzegami — grawitacja wszystkich par w O(N + M log M).

Koszt zależy od siatki, nie od liczby par, więc milion cząstek liczy się tyle
samo co sto tysięcy. Wszystkie pary wchodzą do wyniku; ograniczeniem jest
rozdzielczość przestrzenna (oczko siatki), a nie to, ilu sąsiadów zdążyliśmy
policzyć.

IZOLOWANE BRZEGI. Zwykły PM na FFT jest periodyczny: cząstka wylatująca jedną
ścianą wraca drugą, a grawitacja przyciąga nieskończoną siatkę kopii pudła. To
byłyby ściany tylnymi drzwiami, więc stosujemy metodę Hockneya: siatkę mas
rozszerzamy zerami do podwójnego boku, a jądro próbkujemy we współrzędnych
zawiniętych (indeks > P/2 oznacza ujemną odległość). Splot cykliczny na takiej
siatce jest równy splotowi liniowemu na oryginalnym obszarze, czyli układ jest
naprawdę otwarty — bez kopii i bez odbić.

Potencjał liczymy splotem z jądrem Plummera, a przyspieszenie różnicą skończoną
czwartego rzędu na siatce. To dwie transformaty na krok zamiast czterech; przy
podwojonym boku każda transformata jest droga, więc ta różnica decyduje o tym,
czy symulacja jest interaktywna.

Energia: potencjał zawiera człon własny cząstki (jej własna masa rozsmarowana
na 8 węzłów). Jest on stały w czasie dopóki nie zmieni się oczko siatki, więc
DRYF energii pozostaje wiarygodny, ale wartość bezwzględna U ma przesunięcie.
Do pomiarów energii służy backend `exact`.
"""

from __future__ import annotations

import numpy as np

from bone.backends.base import Backend, Field
from bone.grid import Box, cic_weights, fit_box

#: ile pustych komórek zostawić między chmurą a ścianą pudła; szablon różnicowy
#: czwartego rzędu sięga dwóch komórek, a przy ścianie zawija się przez FFT
_EDGE_CELLS = 3


def _self_kernel(h: float, G: float, softening: float) -> np.ndarray:
    """Macierz 8×8 oddziaływań między węzłami jednej komórki CIC.

    Cząstka rozsmarowana na 8 węzłów przyciąga samą siebie. Ten człon trzeba
    odjąć od potencjału, inaczej energia ma nie tylko przesunięcie, ale i szum:
    wagi zmieniają się, gdy cząstka wędruje wewnątrz komórki. Mając tę macierz,
    człon własny wynosi dokładnie mᵢ·(wᵢᵀ K wᵢ).
    """
    corners = np.array(
        [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
        dtype=np.float64,
    ) * h
    diff = corners[:, None, :] - corners[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", diff, diff)
    return -G / np.sqrt(r2 + softening * softening)


def _kernel_grid(padded: int, h: float, G: float, softening: float, dtype) -> np.ndarray:
    """Jądro Plummera −G/√(r²+ε²) we współrzędnych zawiniętych.

    Indeks powyżej połowy boku odpowiada ujemnej odległości — to właśnie ta
    konwencja sprawia, że splot cykliczny na rozszerzonej siatce jest splotem
    liniowym na siatce oryginalnej.

    Budowane płatami, bo przy siatce 256³ rozszerzonej do 512³ pojedyncza
    tablica pośrednia w float64 to ponad gigabajt.
    """
    index = np.arange(padded)
    coord = (np.where(index <= padded // 2, index, index - padded) * h).astype(dtype)
    squared = coord * coord
    eps2 = np.asarray(softening * softening, dtype=dtype)
    plane = squared[:, None] + squared[None, :] + eps2

    out = np.empty((padded, padded, padded), dtype=dtype)
    for i in range(padded):
        np.sqrt(plane + squared[i], out=out[i])
        np.divide(np.asarray(-G, dtype=dtype), out[i], out=out[i])
    return out


def _cic_deconvolution(padded: int, h: float, clamp: float) -> np.ndarray:
    """Odwrotność okna CIC w przestrzeni Fouriera (dla siatki rfft).

    Rozłożenie masy na 8 węzłów, a potem odczyt tymi samymi wagami, mnoży pole
    przez kwadrat okna W(k) = Π sinc²(kᵢh/2). To rozmycie jest głównym źródłem
    błędu PM — większym niż sama rozdzielczość siatki, co widać po tym, że błąd
    nie maleje przy zwiększaniu ε. Podzielenie widma przez W² kasuje je i nic
    nie kosztuje, bo mnoży się przez zapamiętane jądro.

    Korekta rośnie nieograniczenie przy częstotliwości Nyquista, gdzie i tak nie
    ma wiarygodnej informacji, więc jest przycięta do ``clamp``.
    """
    fx = np.fft.fftfreq(padded, d=h)
    fz = np.fft.rfftfreq(padded, d=h)
    wx = np.sinc(fx * h) ** 2
    wz = np.sinc(fz * h) ** 2
    window = wx[:, None, None] * wx[None, :, None] * wz[None, None, :]
    correction = 1.0 / np.maximum(window * window, 1e-12)
    return np.minimum(correction, clamp)


class MeshBackend(Backend):
    """PM na siatce. `device='cuda'` liczy transformaty na GPU."""

    name = "mesh"

    def __init__(
        self,
        grid: int = 64,
        margin: float = 0.15,
        device: str = "cpu",
        dtype: str = "float32",
        deconvolve: bool = True,
        deconv_clamp: float = 4.0,
    ) -> None:
        if grid < 16:
            raise ValueError("siatka poniżej 16 komórek nie ma sensu")
        self.grid = int(grid)
        self.margin = float(margin)
        self.device_name = device
        self.deconvolve = bool(deconvolve)
        self.deconv_clamp = float(deconv_clamp)
        self._torch = None
        self._dtype = np.float32 if dtype == "float32" else np.float64
        if device == "cuda":
            import torch

            self._torch = torch
            self._device = torch.device("cuda")
            self._tdtype = torch.float32 if dtype == "float32" else torch.float64
        self._box: Box | None = None
        self._kernel_ft = None
        self._kernel_key: tuple | None = None
        self._last_softening: float | None = None
        self._requested: float = 0.0
        self.refits = 0

    @property
    def approximate(self) -> bool:
        return True

    def effective_softening(self, requested: float) -> float:
        """Nie mniej niż oczko siatki — poniżej tej skali pole i tak jest gładkie."""
        if self._box is None:
            return float(requested)
        return float(max(requested, self._box.h))

    def describe(self) -> str:
        where = "cuda" if self._torch is not None else "cpu"
        if self._box is None:
            return f"{self.name} {self.grid}³→{2 * self.grid}³ ({where})"
        detail = f"{where}, oczko {self._box.h:.3g}"
        if self._last_softening is not None and self._last_softening > self._requested + 1e-12:
            detail += f", ε podniesione do {self._last_softening:.3g}"
        return f"{self.name} {self.grid}³→{2 * self.grid}³ ({detail})"

    # ------------------------------------------------------------------ box

    def _ensure_box(self, positions: np.ndarray) -> Box:
        needed = fit_box(positions, self.grid, self.margin, edge=_EDGE_CELLS)
        box = self._box
        too_coarse = box is not None and box.h > 1.8 * needed.h
        if box is None or too_coarse or not box.contains(positions):
            self._box = needed
            self.refits += 1
        return self._box

    def _ensure_kernel(self, box: Box, G: float, softening: float):
        key = (2 * box.ng, round(box.h, 12), round(G, 12), round(softening, 12))
        if self._kernel_key == key and self._kernel_ft is not None:
            return self._kernel_ft
        padded = 2 * box.ng
        kernel = _kernel_grid(padded, box.h, G, softening, self._dtype)
        if self._torch is not None:
            tensor = self._torch.as_tensor(kernel, dtype=self._tdtype, device=self._device)
            del kernel
            spectrum = self._torch.fft.rfftn(tensor)
            del tensor
            if self.deconvolve:
                correction = _cic_deconvolution(padded, box.h, self.deconv_clamp)
                spectrum *= self._torch.as_tensor(
                    correction, dtype=spectrum.real.dtype, device=self._device
                )
            self._kernel_ft = spectrum
        else:
            spectrum = np.fft.rfftn(kernel, axes=(0, 1, 2))
            if self.deconvolve:
                spectrum *= _cic_deconvolution(padded, box.h, self.deconv_clamp)
            self._kernel_ft = spectrum
        self._kernel_key = key
        return self._kernel_ft

    # -------------------------------------------------------------- compute

    def compute(self, positions, masses, G, softening) -> Field:
        n = int(positions.shape[0])
        if n < 2:
            return Field(force=np.zeros_like(positions), potential=np.zeros(n))

        box = self._ensure_box(positions)
        # siatka nie rozdzieli skali mniejszej niż oczko — liczymy tym, co jest
        # osiągalne, i mówimy o tym wprost przez `effective_softening`
        self._requested = float(softening)
        eps = max(float(softening), box.h)
        self._last_softening = eps

        kernel_ft = self._ensure_kernel(box, G, eps)
        idx, wgt = cic_weights(positions, box)
        if self._torch is not None:
            return self._compute_torch(positions, masses, idx, wgt, box, kernel_ft, eps, G)
        return self._compute_numpy(positions, masses, idx, wgt, box, kernel_ft, eps, G)

    def _compute_numpy(self, positions, masses, idx, wgt, box, kernel_ft, softening, G) -> Field:
        ng, p = box.ng, 2 * box.ng
        rho = np.zeros(ng * ng * ng, dtype=self._dtype)
        np.add.at(rho, idx.ravel(), (wgt * masses[:, None]).ravel().astype(self._dtype))

        padded = np.zeros((p, p, p), dtype=self._dtype)
        padded[:ng, :ng, :ng] = rho.reshape(ng, ng, ng)
        axes = (0, 1, 2)
        phi_full = np.fft.irfftn(
            np.fft.rfftn(padded, axes=axes) * kernel_ft, s=(p, p, p), axes=axes
        )
        phi = np.ascontiguousarray(phi_full[:ng, :ng, :ng])

        accel = _gradient_4th_numpy(phi, box.h)
        phi_flat = phi.reshape(-1)
        potential = np.einsum("nk,nk->n", phi_flat[idx], wgt)
        self_k = _self_kernel(box.h, G, softening)
        potential -= masses * np.einsum("nk,kl,nl->n", wgt, self_k, wgt)

        force = np.empty_like(positions)
        for axis in range(3):
            flat = accel[axis].reshape(-1)
            force[:, axis] = np.einsum("nk,nk->n", flat[idx], wgt)
        force *= masses[:, None]
        return Field(force=force, potential=potential)

    def _compute_torch(self, positions, masses, idx, wgt, box, kernel_ft, softening, G) -> Field:
        torch = self._torch
        ng, p = box.ng, 2 * box.ng
        dev, dt = self._device, self._tdtype

        t_idx = torch.as_tensor(idx, device=dev)
        t_wgt = torch.as_tensor(wgt, dtype=dt, device=dev)
        t_m = torch.as_tensor(masses, dtype=dt, device=dev)

        rho = torch.zeros(ng * ng * ng, dtype=dt, device=dev)
        rho.index_add_(0, t_idx.reshape(-1), (t_wgt * t_m[:, None]).reshape(-1))

        padded = torch.zeros((p, p, p), dtype=dt, device=dev)
        padded[:ng, :ng, :ng] = rho.view(ng, ng, ng)
        phi = torch.fft.irfftn(torch.fft.rfftn(padded) * kernel_ft, s=(p, p, p))[
            :ng, :ng, :ng
        ].contiguous()
        del padded

        accel = _gradient_4th_torch(phi, box.h, torch)
        phi_flat = phi.reshape(-1)
        potential = (phi_flat[t_idx] * t_wgt).sum(1)
        self_k = torch.as_tensor(_self_kernel(box.h, G, softening), dtype=dt, device=dev)
        potential -= t_m * torch.einsum("nk,kl,nl->n", t_wgt, self_k, t_wgt)

        force = torch.empty((positions.shape[0], 3), dtype=dt, device=dev)
        for axis in range(3):
            force[:, axis] = (accel[axis].reshape(-1)[t_idx] * t_wgt).sum(1)
        force *= t_m[:, None]
        return Field(
            force=force.to(torch.float64).cpu().numpy(),
            potential=potential.to(torch.float64).cpu().numpy(),
        )

    def close(self) -> None:
        self._kernel_ft = None
        self._kernel_key = None
        if self._torch is not None:
            self._torch.cuda.empty_cache()


def _gradient_4th_numpy(phi: np.ndarray, h: float) -> list[np.ndarray]:
    """a = −∇φ, różnica centralna czwartego rzędu.

    ``np.roll`` zawija na brzegach, ale margines ``_EDGE_CELLS`` gwarantuje, że
    żadna cząstka nie czyta zawiniętych komórek.
    """
    out = []
    scale = 1.0 / (12.0 * h)
    for axis in range(3):
        d = (
            np.roll(phi, 2, axis=axis)
            - 8.0 * np.roll(phi, 1, axis=axis)
            + 8.0 * np.roll(phi, -1, axis=axis)
            - np.roll(phi, -2, axis=axis)
        )
        out.append(-d * scale)
    return out


def _gradient_4th_torch(phi, h: float, torch) -> list:
    out = []
    scale = 1.0 / (12.0 * h)
    for axis in range(3):
        d = (
            torch.roll(phi, 2, dims=axis)
            - 8.0 * torch.roll(phi, 1, dims=axis)
            + 8.0 * torch.roll(phi, -1, dims=axis)
            - torch.roll(phi, -2, dims=axis)
        )
        out.append(-d * scale)
    return out
