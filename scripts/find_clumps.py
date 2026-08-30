#!/usr/bin/env python
"""Wykrywanie zagęszczeń masy w gotowym stanie symulacji.

Po co osobne narzędzie, skoro grudki widać na obrazku: bo obrazek nie odróżnia
fizyki od artefaktu rozdzielczości. Solver siatkowy nie rozdziela grawitacji
poniżej rozmiaru oczka, więc każda struktura o skali oczka jest podejrzana z
definicji. Ten skrypt podaje liczbę i masy zgęstek ORAZ ich rozmiar wyrażony
w oczkach siatki, dzięki czemu można powiedzieć, czy wynik jest wiarygodny.

Metoda: gęstość lokalna z odległości do k-tego sąsiada, odsianie tła progiem
gęstości, a potem friends-of-friends na tym, co zostało. Oba progi są jawnymi
argumentami, a `--sweep` pokazuje, jak wynik od nich zależy — bo liczba grudek,
która skacze przy drobnej zmianie progu, nie jest liczbą grudek.

    python scripts/find_clumps.py runs/frag --sweep
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from bone import relativity as sr
from bone.config import Config
from bone.io import checkpoint, trajectory


def local_density(positions: np.ndarray, masses: np.ndarray, k: int = 32) -> np.ndarray:
    """ρ z promienia kuli obejmującej k najbliższych sąsiadów.

    Estymator kNN, a nie histogram na siatce, bo siatka narzuciłaby własną skalę
    — dokładnie tę, o którą tu podejrzewamy artefakt.
    """
    tree = cKDTree(positions)
    distance, index = tree.query(positions, k=k + 1)
    radius = distance[:, -1]
    enclosed = masses[index[:, 1:]].sum(axis=1)
    volume = 4.0 / 3.0 * np.pi * np.maximum(radius, 1e-12) ** 3
    return enclosed / volume


def boundness(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    G: float,
    softening: float,
) -> tuple[float, float]:
    """Zwraca (2K/|U|, E/|U|) dla grupy cząstek w jej własnym środku masy.

    To jest różnica między „nową małą galaktyką" a falą gęstości. Zagęszczenie
    może być wyraźne na histogramie i zniknąć bez śladu, jeśli jego cząstki nie
    są ze sobą związane — przelatują tylko przez wspólne miejsce. Dopiero
    E = K + U < 0 znaczy, że powstał osobny obiekt.

    Energia własna liczona dokładnie, O(n²) po cząstkach zgęstki. Przy kilku
    tysiącach cząstek to milisekundy, a przybliżenie byłoby tu bez sensu — całe
    pytanie dotyczy znaku sumy dwóch dużych liczb.
    """
    total = float(masses.sum())
    v_com = (masses[:, None] * velocities).sum(axis=0) / total
    relative = velocities - v_com
    kinetic = 0.5 * float(np.dot(masses, np.einsum("ij,ij->i", relative, relative)))

    diff = positions[:, None, :] - positions[None, :, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff) + softening * softening)
    pair = masses[:, None] * masses[None, :] / r
    np.fill_diagonal(pair, 0.0)
    potential = -0.5 * G * float(pair.sum())

    scale = abs(potential) + 1e-300
    return 2.0 * kinetic / scale, (kinetic + potential) / scale


def find_clumps(
    positions: np.ndarray,
    masses: np.ndarray,
    density: np.ndarray,
    contrast: float,
    link: float,
    min_mass_frac: float,
    velocities: np.ndarray | None = None,
    G: float = 0.0,
    softening: float = 0.0,
) -> list[dict]:
    """Grupy friends-of-friends wśród cząstek gęstszych niż ``contrast``×mediana."""
    threshold = contrast * float(np.median(density))
    selected = np.flatnonzero(density > threshold)
    if selected.size < 2:
        return []

    subset = positions[selected]
    pairs = cKDTree(subset).query_pairs(r=link, output_type="ndarray")
    if pairs.size == 0:
        return []

    size = subset.shape[0]
    graph = coo_matrix(
        (np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(size, size)
    )
    count, label = connected_components(graph, directed=False)

    total = float(masses.sum())
    clumps = []
    for group in range(count):
        member = selected[label == group]
        mass = float(masses[member].sum())
        if mass < min_mass_frac * total:
            continue
        pos = positions[member]
        center = np.average(pos, axis=0, weights=masses[member])
        spread = np.sqrt(np.average(((pos - center) ** 2).sum(axis=1), weights=masses[member]))
        entry = {
            "n": int(member.size),
            "mass": mass,
            "mass_frac": mass / total,
            "center": center.tolist(),
            "radius_rms": float(spread),
            "peak_density": float(density[member].max()),
        }
        if velocities is not None:
            virial, energy = boundness(pos, velocities[member], masses[member], G, softening)
            entry["virial"] = virial
            entry["energy_ratio"] = energy
            entry["bound"] = energy < 0.0
        clumps.append(entry)
    clumps.sort(key=lambda c: -c["mass"])
    return clumps


def _use_utf8() -> None:
    """Konsola Windows domyślnie nie zna „×" ani „ρ" (cp1250)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def _spans(positions: np.ndarray) -> tuple[float, float]:
    """Rozciągłość pełna i rozciągłość masy głównej (percentyle 0,5–99,5).

    Różnica między nimi jest ważna, a nie kosmetyczna: pudło siatki w
    ``mesh.py`` dopasowuje się do rozciągłości PEŁNEJ, więc kilku uciekinierów
    potrafi rozdmuchać pudło i pogorszyć oczko wszystkim pozostałym cząstkom.
    Sama liczba max−min nie mówi, czy układ się rozszerzył, czy tylko kilka
    cząstek wyleciało.
    """
    full = float((positions.max(axis=0) - positions.min(axis=0)).max())
    lo = np.percentile(positions, 0.5, axis=0)
    hi = np.percentile(positions, 99.5, axis=0)
    return full, float((hi - lo).max())


def _load(
    run: str, frame: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Config, float, int]:
    """Pozycje, masy i (dla checkpointu) prędkości.

    Klatka trajektorii nie zawiera ani mas, ani wektorów prędkości — zapisywane
    są tylko pozycje i szybkość do kolorowania. Dlatego test związania działa
    wyłącznie na stanie końcowym. Klatki są mimo to przydatne, bo fragmentacja
    często kończy się dużo wcześniej niż bieg.
    """
    state, cfg = checkpoint.load(run)
    if frame is None:
        velocities = sr.velocity(state.masses, state.momenta, cfg.physics.c)
        return state.positions, state.masses, velocities, cfg, state.time, state.step

    loaded = trajectory.load_frame(run, frame)
    if loaded is None:
        total = trajectory.read_meta(run).get("n_frames", 0)
        raise SystemExit(f"klatka {frame} nie istnieje (jest ich {total})")
    positions, _shade, when = loaded
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.full(positions.shape[0], cfg.spawn.total_mass / positions.shape[0])
    return positions, masses, None, cfg, when, -1


def _cell_size(cfg, span: float) -> float:
    """Oczko siatki, jakie backend PM ustawia dla danej rozciągłości."""
    usable = max(cfg.solver.grid - 6, 1)  # _EDGE_CELLS = 3 z każdej strony
    return span * (1.0 + max(cfg.solver.box_margin, 0.0)) / usable


def main() -> int:
    ap = argparse.ArgumentParser(description="Zlicz zagęszczenia masy w stanie symulacji")
    ap.add_argument("run", help="katalog biegu z checkpoint.npz")
    ap.add_argument("--contrast", type=float, default=10.0, help="próg gęstości ×mediana")
    ap.add_argument("--link", type=float, default=0.0, help="długość wiązania FoF (0 = auto)")
    ap.add_argument("--min-mass", type=float, default=0.005, dest="min_mass")
    ap.add_argument("--neighbors", type=int, default=32)
    ap.add_argument("--frame", type=int, help="numer klatki trajektorii (domyślnie stan końcowy)")
    ap.add_argument("--sweep", action="store_true", help="pokaż zależność od progów")
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()
    _use_utf8()

    positions, masses, velocities, cfg, when, step = _load(args.run, args.frame)
    density = local_density(positions, masses, args.neighbors)

    span_full, span_bulk = _spans(positions)
    # backend dokładny nie ma siatki — skalą odniesienia jest wtedy softening
    gridless = cfg.solver.backend == "exact"
    eps = cfg.physics.softening
    cell_used = eps if gridless else _cell_size(cfg, span_full)
    cell_bulk = eps if gridless else _cell_size(cfg, span_bulk)
    # domyślne wiązanie: 3 skale rozdzielczości. Mniej byłoby bezsensowne (pole
    # jest tam gładkie), a dużo więcej zlepiłoby osobne zgęstki w jedną
    link = args.link if args.link > 0 else 3.0 * cell_used

    def detect(contrast: float, link_length: float) -> list[dict]:
        return find_clumps(
            positions, masses, density, contrast, link_length, args.min_mass,
            velocities, cfg.physics.G, eps,
        )

    clumps = detect(args.contrast, link)

    if args.as_json:
        print(json.dumps(
            {"cell_used": cell_used, "cell_bulk": cell_bulk, "link": link, "clumps": clumps},
            indent=2,
        ))
        return 0

    where = "stan końcowy" if args.frame is None else f"klatka {args.frame}"
    print(f"bieg:        {args.run}   ({where})")
    print(f"cząstek:     {positions.shape[0]}   t = {when:.4f}"
          + (f"   krok = {step}" if step >= 0 else ""))
    print(f"rozciągłość: pełna {span_full:.2f}   masy głównej {span_bulk:.2f}")
    if gridless:
        print(f"rozdzielczość: backend dokładny, bez siatki — skalą jest ε = {eps:.4f}")
    else:
        print(f"oczko:       użyte {cell_used:.4f}   bez uciekinierów byłoby {cell_bulk:.4f}")
    print(f"wiązanie FoF: {link:.4f}")
    print(f"próg:        gęstość > {args.contrast:g} × mediana, masa > {args.min_mass:.1%} układu")
    if not gridless and span_full > 2.5 * span_bulk:
        print(
            f"\nUWAGA: pudło siatki jest {span_full / span_bulk:.1f}× większe niż masa główna.\n"
            "Pudło w mesh.py dopasowuje się do rozciągłości PEŁNEJ, więc garść szybkich\n"
            "uciekinierów zabrała rozdzielczość całej reszcie — to nie jest wynik fizyczny,\n"
            "tylko utrata rozdzielczości w trakcie biegu."
        )
    print()

    cell = cell_used

    unit = "ε" if gridless else "oczek"
    if not clumps:
        print("Nie znaleziono zagęszczeń powyżej progu.")
    else:
        has_energy = "energy_ratio" in clumps[0]
        head = f"{'#':>3}  {'cząstek':>8}  {'% masy':>7}  {'r_rms':>7}  {f'r/{unit}':>8}"
        if has_energy:
            head += f"  {'2K/|U|':>7}  {'E/|U|':>7}  związana"
        print(head)
        print("-" * len(head))
        for i, c in enumerate(clumps, 1):
            line = (
                f"{i:>3}  {c['n']:>8}  {c['mass_frac']:>6.2%}  {c['radius_rms']:>7.3f}  "
                f"{c['radius_rms'] / cell:>8.1f}"
            )
            if has_energy:
                line += (
                    f"  {c['virial']:>7.2f}  {c['energy_ratio']:>+7.2f}"
                    f"  {'TAK' if c['bound'] else 'nie':>8}"
                )
            print(line)
        print("-" * len(head))
        detected = sum(c["mass_frac"] for c in clumps)
        print(f"     razem {len(clumps)} zgęstek, {detected:.1%} masy układu")
        if has_energy:
            real = [c for c in clumps if c["bound"]]
            print(
                f"     z tego GRAWITACYJNIE ZWIĄZANYCH: {len(real)}, "
                f"{sum(c['mass_frac'] for c in real):.1%} masy układu"
            )
            if not real:
                print(
                    "\n  Żadna zgęstka nie jest związana — to fale gęstości, nie nowe obiekty.\n"
                    "  Cząstki przelatują tylko przez wspólne miejsce i się rozejdą."
                )

        smallest = min(c["radius_rms"] / cell for c in clumps)
        if smallest < 2.0:
            print(
                f"\nUWAGA: najmniejsza zgęstka ma {smallest:.1f} {unit} — to skala, na której\n"
                "nie warto ufać rozdzielczości. Powtórz drobniej i porównaj liczbę."
            )

    if args.sweep:
        print("\nZależność od progów (liczba zgęstek):")
        contrasts = [3.0, 5.0, 10.0, 20.0, 50.0]
        links = [1.5 * cell, 3.0 * cell, 6.0 * cell]
        header = "  kontrast |" + "".join(f"  wiązanie {ln / cell:.1f}×{unit}" for ln in links)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for contrast in contrasts:
            cells = []
            for ln in links:
                found = detect(contrast, ln)
                bound_count = sum(1 for c in found if c.get("bound", False))
                cells.append(f"{len(found):>10} ({bound_count})")
            print(f"  {contrast:>8.0f} |" + "".join(cells))
        print(
            "\n  W nawiasie liczba zgęstek grawitacyjnie związanych. Wynik jest wiarygodny\n"
            "  wtedy, gdy w środku tabeli jest plateau — liczba niezależna od progu."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
