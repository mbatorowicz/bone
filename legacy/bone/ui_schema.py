"""Schemat parametrów Studio: strefy panelu + scope (spawn / runtime)."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Literal

from bone.constants import SimConfig
from bone.geometry import GEOMETRY_LABELS, GEOMETRY_NAMES

Scope = Literal["spawn", "runtime", "viz"]

# Kolejność stref w panelu
PANEL_ZONES: list[tuple[str, str, str]] = [
    ("control", "Sterowanie", "Start / stop / kontynuacja"),
    ("presets", "Presety", "Gotowe zestawy parametrów"),
    ("start", "Narodziny", "Rozkład startowy cech i pozycji"),
    ("dynamics", "Ruch i grawitacja", "m = wytrwałość + energia + bogactwo"),
    ("emergence", "Cechy → przepływ", "Miłość/nienawiść/chciwość kształtują układ"),
    ("economy", "Dobra i praca", "Handel, praca, konsumpcja, renta"),
    ("thresholds", "Progi zdarzeń", "Kiedy liczymy T_* w metrykach"),
    ("view", "Widok", "LIVE / REPLAY — tylko wizualizacja"),
]

# (key, label, min, max, step, zone_id, scope)
PARAM_DEFS: list[tuple[str, str, float, float, float, str, Scope]] = [
    # Narodziny (spawn)
    ("geometry", "Kształt początkowy", 0, 9, 1, "start", "spawn"),
    ("grid_n", "Siatka N (sześcian: N³)", 6, 32, 1, "start", "spawn"),
    ("n_particles", "Liczba jednostek", 500, 30000, 500, "start", "spawn"),
    ("spacing", "Odległość startowa", 0.5, 2.0, 0.05, "start", "spawn"),
    ("seed", "Ziarno losowości", 0, 9999, 1, "start", "spawn"),
    ("orbital_seed_speed", "Pęd kątowy L₀ (0=bezruch)", 0.0, 1.0, 0.01, "start", "spawn"),
    ("newborn_mean", "Średnia cech noworodka", 0.05, 0.9, 0.01, "start", "spawn"),
    ("newborn_sigma_frac", "Rozrzut cech σ/μ", 0.005, 0.15, 0.005, "start", "spawn"),
    # Ruch i grawitacja
    ("steps", "Limit kroków (0=bez limitu)", 0, 100000, 50, "dynamics", "runtime"),
    ("sim_speed", "Szybkość czasu (×dt)", 1, 100, 1, "dynamics", "runtime"),
    ("dt", "Krok czasu dt", 0.005, 0.06, 0.001, "dynamics", "runtime"),
    ("G", "Stała G (siła grawitacji)", 0.005, 0.4, 0.005, "dynamics", "runtime"),
    ("energy_mass_coupling", "Zdrowie/energia → masa", 0.0, 1.5, 0.05, "dynamics", "runtime"),
    ("wealth_mass_coupling", "Bogactwo → masa", 0.0, 1.5, 0.05, "dynamics", "runtime"),
    ("c", "Limit prędkości c", 2.0, 20.0, 0.5, "dynamics", "runtime"),
    ("soft_eps", "Miękkie jądro ε", 0.05, 1.2, 0.01, "dynamics", "runtime"),
    ("r_cut", "Zasięg sił grawitacji", 1.0, 8.0, 0.1, "dynamics", "runtime"),
    ("max_neighbors", "Max sąsiadów na cząstkę (K)", 8, 128, 4, "dynamics", "runtime"),
    ("neighbor_rebuild_every", "Przebudowa sąsiadów co N", 1, 20, 1, "dynamics", "runtime"),
    ("orbit_every", "Orbity co N kroków", 1, 16, 1, "dynamics", "runtime"),
    ("trait_every", "Cechy/ekonomia co N kroków", 1, 16, 1, "dynamics", "runtime"),
    ("core_repulsion", "Odpychanie bliskiego jądra", 0.0, 3.0, 0.05, "dynamics", "runtime"),
    ("damping", "Tłumienie ruchu (liniowe)", 0.0, 0.02, 0.0001, "dynamics", "runtime"),
    ("drag_quad", "Opór kwadratowy", 0.0, 0.02, 0.0001, "dynamics", "runtime"),
    ("predisposition_force", "Predyspozycje → dryf siłowy", 0.0, 0.02, 0.0001, "dynamics", "runtime"),
    ("wall_stiffness", "Sztywność granic", 0.0, 3.0, 0.05, "dynamics", "runtime"),
    ("wall_margin", "Margines granic", 0.0, 8.0, 0.1, "dynamics", "runtime"),
    ("circularize_rate", "Cyrkularyzacja orbit", 0.0, 6.0, 0.1, "dynamics", "runtime"),
    ("disk_flatten_rate", "Lepkość pionowa (dysk)", 0.0, 2.0, 0.05, "dynamics", "runtime"),
    ("vrel_cap_factor", "Limit v względnej / v_okr", 0.5, 3.0, 0.05, "dynamics", "runtime"),
    ("traj_every", "Zapis klatki co N kroków", 1, 80, 1, "dynamics", "runtime"),
    ("live_every", "Odśwież LIVE co N", 1, 40, 1, "dynamics", "runtime"),
    ("snapshot_every", "Metryki co N", 5, 200, 5, "dynamics", "runtime"),
    # Cechy → przepływ
    ("inequality_drive", "Tempo zaostrzania nierówności", 0.0, 2.5, 0.05, "emergence", "runtime"),
    ("inequality_delay", "Opóźnienie zaostrzenia", 0.0, 400.0, 1.0, "emergence", "runtime"),
    ("inequality_timescale", "Skala czasu nierówności", 5.0, 400.0, 1.0, "emergence", "runtime"),
    ("greed_bias", "Złość/nienawiść → wyzysk", 0.0, 3.0, 0.05, "emergence", "runtime"),
    ("generosity_bias", "Miłość/uczciwość → dary", 0.0, 3.0, 0.05, "emergence", "runtime"),
    ("energy_exchange", "Zdrowie ↔ bogactwo (×γ)", 0.0, 1.0, 0.01, "emergence", "runtime"),
    ("exploit_rate", "Siła wyzysku", 0.0, 0.5, 0.01, "emergence", "runtime"),
    ("gift_rate", "Siła darów", 0.0, 0.3, 0.01, "emergence", "runtime"),
    ("matthew_rate", "Zdolności → efekt Mateusza", 0.0, 0.6, 0.01, "emergence", "runtime"),
    ("collapse_stop_ratio", "Stop przy kolapsie r½/r₀", 0.01, 0.5, 0.01, "emergence", "runtime"),
    ("wealth_concentration_stop", "Stop przy top-5% bogactwa", 0.1, 0.9, 0.01, "emergence", "runtime"),
    # Dobra i praca
    ("wealth_mean", "Średnie bogactwo startowe", 0.1, 5.0, 0.05, "economy", "runtime"),
    ("trade_rate", "Handel (zaufanie/uczciwość)", 0.0, 0.5, 0.01, "economy", "runtime"),
    ("labor_rate", "Praca (zdolności×zdrowie)", 0.0, 0.1, 0.001, "economy", "runtime"),
    ("consume_rate", "Konsumpcja (koszt życia ×γ)", 0.0, 0.1, 0.001, "economy", "runtime"),
    ("capital_return", "Renta kapitału (zdolności)", 0.0, 0.2, 0.005, "economy", "runtime"),
    # Progi
    ("knowledge_threshold", "Próg wiedzy (T_knowledge)", 0.05, 1.0, 0.01, "thresholds", "runtime"),
    ("cluster_min_size", "Min. rozmiar społeczności", 2, 80, 1, "thresholds", "runtime"),
    ("cluster_max_frac", "Max ułamek społeczności", 0.05, 0.8, 0.01, "thresholds", "runtime"),
    ("cluster_link_radius", "Promień więzi lokalnej", 0.4, 4.0, 0.05, "thresholds", "runtime"),
    ("cluster_affinity_min", "Min. powinowactwo (s_ij)", 0.0, 1.5, 0.05, "thresholds", "runtime"),
    ("hate_conflict_threshold", "Próg nienawiści (konflikt)", 0.1, 1.0, 0.01, "thresholds", "runtime"),
    ("death_health", "Próg zdrowia (śmierć)", 0.001, 0.2, 0.001, "thresholds", "runtime"),
    ("gini_threshold", "Próg Giniego (nierówność)", 0.01, 0.8, 0.01, "thresholds", "runtime"),
]

VIZ_SLIDERS: list[tuple[str, str, float, float, float]] = [
    ("live_fps", "Szybkość klatek LIVE (Hz)", 1, 30, 1),
    ("replay_speed", "Prędkość odtwarzania ×", 0.25, 8.0, 0.25),
    ("point_size", "Rozmiar punktów", 0.04, 0.5, 0.01),
    ("opacity", "Krycie punktów", 0.2, 1.0, 0.05),
    ("brightness", "Jasność tła", 0.0, 0.3, 0.01),
]

# Kompat: stary format (key, label, min, max, step, group)
PARAM_SLIDERS: list[tuple[str, str, float, float, float, str]] = [
    (k, lab, mn, mx, st, zone) for k, lab, mn, mx, st, zone, _scope in PARAM_DEFS
]

SPAWN_KEYS: frozenset[str] = frozenset(k for k, *_, scope in PARAM_DEFS if scope == "spawn")
RUNTIME_KEYS: frozenset[str] = frozenset(k for k, *_, scope in PARAM_DEFS if scope == "runtime")

_INT_KEYS = frozenset(
    {
        "grid_n",
        "n_particles",
        "geometry",
        "steps",
        "sim_speed",
        "seed",
        "traj_every",
        "live_every",
        "snapshot_every",
        "max_neighbors",
        "max_pairs",
        "neighbor_rebuild_every",
        "orbit_every",
        "trait_every",
        "cluster_min_size",
    }
)


def panel_boot_schema() -> dict[str, Any]:
    """Dane do zbudowania panelu w Studio (strefy + suwaki + scope)."""
    zones: dict[str, list[dict[str, Any]]] = {zid: [] for zid, _, _ in PANEL_ZONES}
    for key, label, mn, mx, step, zone, scope in PARAM_DEFS:
        if zone not in zones:
            zones[zone] = []
        zones[zone].append(
            {
                "key": key,
                "label": label,
                "min": mn,
                "max": mx,
                "step": step,
                "scope": scope,
            }
        )
    return {
        "zoneOrder": [
            {"id": zid, "title": title, "hint": hint} for zid, title, hint in PANEL_ZONES
        ],
        "zones": zones,
        "spawnKeys": sorted(SPAWN_KEYS),
        "runtimeKeys": sorted(RUNTIME_KEYS),
        "geometryOptions": geometry_options(),
        "viz": [
            {"key": k, "label": l, "min": a, "max": b, "step": c} for k, l, a, b, c in VIZ_SLIDERS
        ],
    }


def default_param_values() -> dict[str, float | int | bool | str]:
    cfg = SimConfig()
    out: dict[str, float | int | bool | str] = {}
    names = {f.name for f in fields(SimConfig)}
    for key, *_rest in PARAM_DEFS:
        if key in names:
            out[key] = getattr(cfg, key)
    out["make_gif"] = False
    out["open_view"] = False
    out["out_dir"] = "out"
    return out


def config_from_params(params: dict) -> SimConfig:
    """Zbuduj SimConfig z dict suwaków (z rzutowaniem typów)."""
    cfg0 = SimConfig()
    kwargs: dict[str, Any] = {}
    for f in fields(SimConfig):
        if f.name not in params:
            continue
        if f.name == "out_dir":
            kwargs[f.name] = str(params[f.name])
            continue
        if f.name in ("make_gif", "open_view"):
            kwargs[f.name] = bool(params[f.name])
            continue
        val = params[f.name]
        if f.name in _INT_KEYS:
            kwargs[f.name] = int(round(float(val)))
        else:
            kwargs[f.name] = float(val)
    kwargs.setdefault("make_gif", False)
    kwargs.setdefault("open_view", False)
    kwargs.setdefault("out_dir", "out")
    kwargs["grid_n"] = int(max(4, min(32, kwargs.get("grid_n", cfg0.grid_n))))
    kwargs["n_particles"] = int(
        max(100, min(30000, kwargs.get("n_particles", cfg0.n_particles)))
    )
    n_geom = len(GEOMETRY_NAMES) - 1
    kwargs["geometry"] = int(max(0, min(n_geom, kwargs.get("geometry", cfg0.geometry))))
    kwargs["steps"] = int(max(0, min(200000, kwargs.get("steps", cfg0.steps))))
    if "sim_speed" in kwargs:
        kwargs["sim_speed"] = int(max(1, min(100, kwargs["sim_speed"])))
    return SimConfig(**kwargs)


def merge_runtime_params(base: SimConfig, params: dict) -> SimConfig:
    """Zastosuj tylko parametry runtime (hot-apply w trakcie biegu)."""
    from dataclasses import asdict, replace

    data = asdict(base)
    for key in RUNTIME_KEYS:
        if key not in params:
            continue
        if key in _INT_KEYS:
            data[key] = int(round(float(params[key])))
        else:
            data[key] = float(params[key])
    data["grid_n"] = int(max(4, min(32, data["grid_n"])))
    data["n_particles"] = int(max(100, min(30000, data["n_particles"])))
    data["steps"] = int(max(0, min(200000, data["steps"])))
    if "sim_speed" in data:
        data["sim_speed"] = int(max(1, min(100, data["sim_speed"])))
    # zachowaj pola spoza RUNTIME (spawn, io)
    return replace(base, **{k: data[k] for k in RUNTIME_KEYS if k in data})


def stable_orbit_preset(geometry: int = 8) -> dict[str, float | int | bool | str]:
    """Zrównoważony gift≈exploit + małe L0 + dyssypacja pionowa → emergentny dysk."""
    base = default_param_values()
    base.update(
        {
            "geometry": int(geometry),
            "n_particles": 8000,
            "grid_n": 20,
            "steps": 0,
            "dt": 0.014,
            "G": 0.065,
            "soft_eps": 0.50,
            "r_cut": 3.2,
            "core_repulsion": 0.90,
            "damping": 0.0002,
            "drag_quad": 0.00008,
            "predisposition_force": 0.0012,
            "circularize_rate": 3.8,
            "disk_flatten_rate": 0.85,
            "orbital_seed_speed": 0.12,
            "vrel_cap_factor": 1.10,
            "wall_stiffness": 0.40,
            "inequality_drive": 0.0,
            "greed_bias": 0.8,
            "generosity_bias": 1.0,
            "energy_exchange": 0.15,
            "exploit_rate": 0.08,
            "gift_rate": 0.10,
            "matthew_rate": 0.05,
            "wealth_mass_coupling": 0.35,
            "energy_mass_coupling": 0.45,
            "traj_every": 24,
            "live_every": 12,
            "snapshot_every": 100,
            "max_neighbors": 24,
            "neighbor_rebuild_every": 8,
            "orbit_every": 2,
            "trait_every": 4,
            "make_gif": False,
        }
    )
    return base


def orbit_then_singularity_preset(geometry: int = 8) -> dict[str, float | int | bool | str]:
    """Kolaps społeczny: wysoki exploit+matthew+wealth→masa, niski gift — bez sztucznej studni."""
    base = stable_orbit_preset(geometry)
    base.update(
        {
            "steps": 40000,
            "inequality_drive": 1.6,
            "inequality_delay": 8.0,
            "inequality_timescale": 24.0,
            "greed_bias": 2.2,
            "generosity_bias": 0.25,
            "energy_exchange": 0.45,
            "exploit_rate": 0.28,
            "gift_rate": 0.02,
            "matthew_rate": 0.42,
            "wealth_mass_coupling": 1.1,
            "energy_mass_coupling": 0.55,
            "capital_return": 0.08,
            "collapse_stop_ratio": 0.08,
            "wealth_concentration_stop": 0.55,
            "circularize_rate": 1.2,
            "disk_flatten_rate": 0.25,
            "orbital_seed_speed": 0.06,
        }
    )
    return base


def geometry_options() -> list[dict[str, str | int]]:
    return [
        {"value": i, "label": f"{i}: {GEOMETRY_LABELS[name]}"}
        for i, name in enumerate(GEOMETRY_NAMES)
    ]
