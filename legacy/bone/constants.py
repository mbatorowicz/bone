"""Stałe fizyczne i parametry startowe symulacji."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Konfiguracja jednego przebiegu."""

    grid_n: int = 24
    spacing: float = 1.0
    # liczba punktów (dla geometrii ≠ cube); cube używa grid_n^3
    n_particles: int = 16000
    # 0 cube … 7 gaussian, 8 donut, 9 sphere_band; steps<=0 → aż user przerwie
    geometry: int = 0
    steps: int = 5000
    dt: float = 0.018
    # przyspieszenie: jeden krok z dt×N (pomija pośrednie — mniej CPU, grubsza numeryka)
    sim_speed: int = 1
    seed: int = 42

    # Fizyka
    G: float = 0.09
    c: float = 8.0
    soft_eps: float = 0.42
    r_cut: float = 3.0
    # sąsiedzi: kNN cap + lista Verlet (unikaj P≈N² w klastrach)
    max_neighbors: int = 24
    max_pairs: int = 1_500_000
    neighbor_rebuild_every: int = 8  # przebudowa KDTree co N kroków
    neighbor_skin: float = 0.35  # ułamek spacing → skin Verlet
    core_repulsion: float = 0.7
    damping: float = 0.0003
    drag_quad: float = 0.00015
    predisposition_force: float = 0.0008
    wall_stiffness: float = 0.35
    wall_margin: float = 3.0
    # 0 = zimny start (bezruch); >0 opcjonalny kick parametryczny (spin×r)
    orbital_seed_speed: float = 0.0
    circularize_rate: float = 2.2
    vrel_cap_factor: float = 1.35
    # lepkość pionowa: gasi v∥L (dysk emergentny); 0 = bez spłaszczania
    disk_flatten_rate: float = 0.0
    # rzadziej CPU: orbity / cechy+ekonomia
    orbit_every: int = 2
    trait_every: int = 4

    # Emergencja nierówności → geometria (bez sztucznej studni)
    inequality_drive: float = 0.0  # tempo zaostrzania exploit/matthew w czasie
    inequality_delay: float = 0.0  # czas przed zaostrzeniem
    inequality_timescale: float = 40.0
    greed_bias: float = 1.0  # skala przejmowania (wyzysk)
    generosity_bias: float = 1.0  # skala rozdawnictwa (dary)
    energy_exchange: float = 0.0  # wymiana zdrowia↔wealth, skalowana γ
    collapse_stop_ratio: float = 0.05  # auto-stop: r_half/r0
    wealth_concentration_stop: float = 0.45  # auto-stop: top-5% wealth/masy

    # Noworodki: średnie i względne σ
    newborn_mean: float = 0.35
    newborn_sigma_frac: float = 0.03

    # Ekonomia (przepływ dóbr) — geometria wyłania się stąd
    wealth_mean: float = 1.0
    trade_rate: float = 0.04
    exploit_rate: float = 0.18
    gift_rate: float = 0.02
    matthew_rate: float = 0.22  # kapitał płynie do zdolniejszych
    labor_rate: float = 0.018
    consume_rate: float = 0.012
    capital_return: float = 0.04  # renta: dw/dt ∝ w · ability · (w/mean)^α
    # masa grawitacyjna = m₀(wytrwałość) + energia(zdrowie)×m₀ + bogactwo
    wealth_mass_coupling: float = 0.35  # bogactwo → masa
    energy_mass_coupling: float = 0.4  # zdrowie/energia → masa (E/c²)
    gini_threshold: float = 0.05  # hitting time nierówności

    # Progi zdarzeń
    knowledge_threshold: float = 0.25
    cluster_min_size: int = 12
    cluster_max_frac: float = 0.35  # klaster ≠ cały wszechświat
    cluster_link_radius: float = 1.2
    cluster_affinity_min: float = 0.65
    hate_conflict_threshold: float = 0.55
    death_health: float = 0.02

    # Snapshoty / I/O / widok 3D
    snapshot_every: int = 100
    traj_every: int = 24  # co ile kroków zapisać klatkę sześcianu
    live_every: int = 12  # co ile kroków odświeżyć LIVE JSON (Studio)
    out_dir: str = "out"
    make_gif: bool = True
    open_view: bool = False


# Nazwy cech w structured array / raportach
TRAIT_NAMES = (
    "predisposition_x",
    "predisposition_y",
    "predisposition_z",
    "endurance",  # m0 — baza masy
    "health",  # energia → wkład do m_graw + śmierć
    "knowledge",  # |p|
    "wisdom",  # tau
    "learning_speed",  # 1/m_eff
    "honesty",
    "loyalty",
    "love",
    "anger",
    "hatred",
    "ability",  # sigma
    "wealth",  # dobra / pieniądze
)
