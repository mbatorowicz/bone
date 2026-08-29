"""Config fizyki relatywistycznej — bez ekonomii / ścian."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

# liczba brył — nazwy w bone.domain.geometry.GEOMETRY_NAMES
_N_GEOMS = 10


@dataclass(frozen=True)
class SpawnConfig:
    n_particles: int = 4000
    geometry: int = 8  # donut domyślnie
    spacing: float = 1.0
    seed: int = 42
    mass_mean: float = 1.0
    mass_sigma: float = 0.15
    # ułamek c: pęd kątowy startowy (dysk/torus)
    orbital_seed: float = 0.25
    # izotropowy kick (ułamek c); 0 = tylko orbital / spoczynek
    thermal_seed: float = 0.0


@dataclass(frozen=True)
class PhysicsConfig:
    dt: float = 0.02
    G: float = 0.12
    c: float = 10.0
    soft_eps: float = 0.35
    r_cut: float = 5.0
    core_repulsion: float = 0.35
    # tłumienie ≈0 — inaczej układ się „rozpełza” energetycznie w dół, ale głównie
    # rozpraszanie pochodziło ze ścian + s_ij<0; tu czysta grawitacja
    damping: float = 0.0
    max_neighbors: int = 32
    max_pairs: int = 1_500_000
    neighbor_rebuild_every: int = 6
    neighbor_skin: float = 0.4
    sim_speed: int = 1
    # cap |F| (stabilność numeryczna)
    force_cap: float = 40.0


@dataclass(frozen=True)
class IoConfig:
    steps: int = 0
    out_dir: str = "out"
    live_every: int = 8
    traj_every: int = 16
    snapshot_every: int = 50
    point_stride: int = 3


@dataclass(frozen=True)
class AppConfig:
    spawn: SpawnConfig = SpawnConfig()
    physics: PhysicsConfig = PhysicsConfig()
    io: IoConfig = IoConfig()

    def to_flat(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for layer in ("spawn", "physics", "io"):
            d.update(asdict(getattr(self, layer)))
        return d

    @staticmethod
    def from_flat(data: dict[str, Any]) -> AppConfig:
        def pick(cls: type, src: dict) -> Any:
            names = {f.name for f in fields(cls)}
            kw = {k: src[k] for k in names if k in src}
            return cls(**kw)

        return AppConfig(
            spawn=pick(SpawnConfig, data),
            physics=pick(PhysicsConfig, data),
            io=pick(IoConfig, data),
        )


_UI: list[tuple[str, str, float, float, float, str, str]] = [
    ("n_particles", "Liczba cząstek", 200, 20000, 200, "start", "spawn"),
    ("geometry", "Bryła startowa", 0, _N_GEOMS - 1, 1, "start", "spawn"),
    ("spacing", "Skala rozmiaru", 0.4, 2.5, 0.05, "start", "spawn"),
    ("seed", "Ziarno", 0, 9999, 1, "start", "spawn"),
    ("mass_mean", "Masa spoczynkowa ⟨m⟩", 0.2, 3.0, 0.05, "start", "spawn"),
    ("orbital_seed", "Orbit start (ułamek c)", 0.0, 0.8, 0.01, "start", "spawn"),
    ("thermal_seed", "Temperatura start (ułamek c)", 0.0, 0.4, 0.01, "start", "spawn"),
    ("G", "Stała G", 0.01, 0.5, 0.005, "dynamics", "runtime"),
    ("c", "Prędkość światła c", 2.0, 30.0, 0.5, "dynamics", "runtime"),
    ("r_cut", "Zasięg grawitacji", 1.5, 12.0, 0.1, "dynamics", "runtime"),
    ("soft_eps", "Miękkie jądro ε", 0.05, 1.5, 0.05, "dynamics", "runtime"),
    ("core_repulsion", "Odpychanie jądra", 0.0, 2.0, 0.05, "dynamics", "runtime"),
    ("dt", "Krok dt", 0.005, 0.06, 0.001, "dynamics", "runtime"),
    ("sim_speed", "Szybkość czasu ×", 1, 40, 1, "dynamics", "runtime"),
    ("max_neighbors", "Max sąsiadów K", 8, 64, 4, "advanced", "runtime"),
    ("damping", "Tłumienie", 0.0, 0.01, 0.0001, "advanced", "runtime"),
    ("live_every", "Odśwież LIVE co N", 1, 40, 1, "view", "runtime"),
    ("traj_every", "Zapis klatki co N", 1, 80, 1, "view", "runtime"),
    ("snapshot_every", "Metryki co N", 10, 200, 10, "view", "runtime"),
]

_SPAWN_KEYS = frozenset(k for k, *_, sc in _UI if sc == "spawn")
_RUNTIME_KEYS = frozenset(k for k, *_, sc in _UI if sc == "runtime")
_INT_KEYS = frozenset(
    {
        "n_particles",
        "geometry",
        "seed",
        "sim_speed",
        "max_neighbors",
        "steps",
        "live_every",
        "traj_every",
        "snapshot_every",
        "point_stride",
        "neighbor_rebuild_every",
        "max_pairs",
    }
)


def schema_for_ui() -> dict[str, Any]:
    from bone.domain.geometry import GEOMETRY_NAMES

    zones = [
        ("start", "Start"),
        ("dynamics", "Dynamika SR"),
        ("advanced", "Zaawansowane"),
        ("view", "Widok"),
    ]
    by: dict[str, list] = {z: [] for z, _ in zones}
    for key, label, mn, mx, step, zone, scope in _UI:
        item = {
            "key": key,
            "label": label,
            "min": mn,
            "max": mx,
            "step": step,
            "scope": scope,
        }
        if key == "geometry":
            item["options"] = [
                {"value": i, "label": name} for i, name in enumerate(GEOMETRY_NAMES)
            ]
        by.setdefault(zone, []).append(item)
    return {
        "zones": [{"id": z, "label": lab, "sliders": by.get(z, [])} for z, lab in zones],
        "presets": [
            {"id": "galaxy", "label": "Galaktyka"},
            {"id": "cluster", "label": "Gromada"},
            {"id": "burst", "label": "Burst SR"},
        ],
        "geometry_names": list(GEOMETRY_NAMES),
        "defaults": AppConfig().to_flat(),
        "thesis": "Czysta grawitacja relatywistyczna (p = γ m v) — bez ścian, bez warstwy społecznej.",
    }


def merge_runtime(base: AppConfig, params: dict[str, Any]) -> AppConfig:
    flat = base.to_flat()
    for k, v in params.items():
        if k in _SPAWN_KEYS:
            continue
        if k not in flat and k not in _RUNTIME_KEYS:
            continue
        flat[k] = int(round(float(v))) if k in _INT_KEYS else float(v)
    return AppConfig.from_flat(flat)


def _flat(updates: dict[str, Any]) -> AppConfig:
    d = AppConfig().to_flat()
    d.update(updates)
    return AppConfig.from_flat(d)


def preset_galaxy() -> AppConfig:
    """Dysk/torus + orbit → trzyma się bez ścian."""
    return _flat(
        {
            "geometry": 8,
            "n_particles": 5000,
            "orbital_seed": 0.35,
            "thermal_seed": 0.02,
            "G": 0.14,
            "c": 12.0,
            "r_cut": 6.0,
            "soft_eps": 0.4,
            "core_repulsion": 0.25,
            "damping": 0.0,
        }
    )


def preset_cluster() -> AppConfig:
    """Zimna kula — kolaps grawitacyjny."""
    return _flat(
        {
            "geometry": 2,
            "n_particles": 4000,
            "orbital_seed": 0.0,
            "thermal_seed": 0.01,
            "G": 0.18,
            "c": 10.0,
            "r_cut": 5.0,
            "soft_eps": 0.3,
            "core_repulsion": 0.5,
        }
    )


def preset_burst() -> AppConfig:
    """Wysokie v/c — efekty γ widoczne."""
    return _flat(
        {
            "geometry": 7,
            "n_particles": 3000,
            "orbital_seed": 0.15,
            "thermal_seed": 0.25,
            "G": 0.1,
            "c": 4.0,
            "r_cut": 5.0,
            "dt": 0.012,
        }
    )


# aliasy kompatybilności ze starym API Studio
preset_balance = preset_galaxy
preset_exploit = preset_cluster
preset_tribes = preset_burst
