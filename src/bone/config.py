"""Konfiguracja: zamrożone dataclassy, presety i schemat dla UI.

Jedno źródło prawdy. Schemat UI jest generowany z tej samej listy pól, które
opisują config, więc suwak nie może istnieć bez pola ani odwrotnie.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

GEOMETRIES = (
    "ball",
    "sphere_shell",
    "disk",
    "torus",
    "gaussian",
    "filament",
    "two_clumps",
    "plummer",
)

BACKENDS = ("auto", "exact", "mesh")
DEVICES = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class SpawnConfig:
    n_particles: int = 4000
    geometry: str = "plummer"
    radius: float = 8.0
    seed: int = 42
    #: masa CAŁEGO układu, nie jednej cząstki. Dzięki temu suwak liczby cząstek
    #: zmienia rozdzielczość, a nie fizykę: ten sam obiekt próbkowany gęściej.
    #: Gdyby parametrem była masa pojedynczej cząstki, przejście z 4 na 100 tys.
    #: cząstek zwiększyłoby masę układu 25-krotnie i rozwaliłoby warunek startowy.
    total_mass: float = 4000.0
    mass_spread: float = 0.15
    #: ułamek prędkości okrężnej nadawany na starcie (0 = zimny start)
    rotation: float = 0.6
    #: izotropowy rozrzut prędkości jako ułamek c
    temperature: float = 0.0


@dataclass(frozen=True)
class PhysicsConfig:
    #: dobrane tak, aby przy domyślnym starcie (4000 cząstek o masie 1, promień 8)
    #: prędkość okrężna na brzegu wynosiła ≈0,3 c — patrz `gravity_for_beta`
    G: float = 0.16
    c: float = 30.0
    #: softening Plummera; ma wymiar DŁUGOŚCI (poprzednia wersja mieszała ε z ε²)
    softening: float = 0.25
    #: górny limit kroku; faktyczny krok wybiera kryterium adaptacyjne
    dt_max: float = 0.02
    #: dokładność kroku adaptacyjnego: dt ≤ η·√(ε/a_max)
    accuracy: float = 0.03
    adaptive_dt: bool = True


@dataclass(frozen=True)
class SolverConfig:
    backend: Literal["auto", "exact", "mesh"] = "auto"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    #: powyżej tylu cząstek `auto` przełącza się z `exact` na `mesh`
    exact_max_particles: int = 4000
    #: bok siatki PM; koszt rośnie jak (2·grid)³·log, nie z liczbą par
    grid: int = 64
    #: margines pudła siatki względem rozciągłości chmury
    box_margin: float = 0.15
    #: co ile kroków mierzyć błąd siły backendu względem dokładnego O(N²); 0 = nigdy
    error_check_every: int = 0
    error_check_sample: int = 512


@dataclass(frozen=True)
class RunConfig:
    steps: int = 0  # 0 = bez limitu
    out_dir: str = "runs/latest"
    diagnostics_every: int = 20
    live_every: int = 4
    trajectory_every: int = 20
    point_stride: int = 1
    time_scale: float = 1.0  # ile czasu symulacji na jeden krok pętli (×dt)


@dataclass(frozen=True)
class Config:
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    run: RunConfig = field(default_factory=RunConfig)

    _SECTIONS = ("spawn", "physics", "solver", "run")

    def to_flat(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self._SECTIONS:
            out.update(asdict(getattr(self, name)))
        return out

    @classmethod
    def from_flat(cls, data: dict[str, Any]) -> Config:
        kinds = {
            "spawn": SpawnConfig,
            "physics": PhysicsConfig,
            "solver": SolverConfig,
            "run": RunConfig,
        }
        built = {}
        for name, kind in kinds.items():
            kw = {}
            for f in fields(kind):
                if f.name not in data:
                    continue
                kw[f.name] = _coerce(f.type, data[f.name])
            built[name] = kind(**kw)
        return cls(**built)

    def replace_flat(self, updates: dict[str, Any]) -> Config:
        flat = self.to_flat()
        flat.update(updates)
        return Config.from_flat(flat)


def _coerce(kind: Any, value: Any) -> Any:
    """Rzutuj wartość z JSON-a na typ pola. Adnotacje są napisami (PEP 563)."""
    text = kind if isinstance(kind, str) else getattr(kind, "__name__", str(kind))
    if "bool" in text:
        if isinstance(value, str):
            return value.lower() not in {"", "0", "false", "no"}
        return bool(value)
    if "int" in text:
        return int(round(float(value)))
    if "float" in text:
        return float(value)
    return value


# ---------------------------------------------------------------- schemat UI

#: (klucz, etykieta, min, max, krok, sekcja panelu, czy działa w trakcie biegu)
_CONTROLS: tuple[tuple[str, str, float, float, float, str, bool], ...] = (
    ("n_particles", "Liczba cząstek", 100, 200_000, 100, "start", False),
    ("radius", "Promień startowy", 1.0, 40.0, 0.5, "start", False),
    ("seed", "Ziarno losowe", 0, 9999, 1, "start", False),
    ("total_mass", "Masa układu", 100.0, 100_000.0, 100.0, "start", False),
    ("mass_spread", "Rozrzut mas", 0.0, 1.0, 0.01, "start", False),
    ("rotation", "Rotacja (ułamek v_okrężnej)", 0.0, 1.4, 0.02, "start", False),
    ("temperature", "Temperatura (ułamek c)", 0.0, 0.5, 0.005, "start", False),
    ("G", "Stała grawitacji G", 0.05, 5.0, 0.05, "dynamics", True),
    ("c", "Prędkość światła c", 2.0, 200.0, 1.0, "dynamics", True),
    ("softening", "Softening ε", 0.02, 3.0, 0.01, "dynamics", True),
    ("dt_max", "Maksymalny krok dt", 0.001, 0.2, 0.001, "dynamics", True),
    ("accuracy", "Dokładność kroku η", 0.005, 0.2, 0.005, "dynamics", True),
    ("time_scale", "Tempo symulacji ×", 1, 40, 1, "dynamics", True),
    ("grid", "Siatka PM (bok)", 32, 256, 32, "solver", False),
    ("box_margin", "Margines pudła", 0.0, 1.0, 0.05, "solver", True),
    ("exact_max_particles", "Próg exact→mesh", 500, 20_000, 500, "solver", False),
    ("error_check_every", "Pomiar błędu co N kroków", 0, 500, 10, "solver", True),
    ("live_every", "Odświeżanie widoku co N", 1, 40, 1, "view", True),
    ("trajectory_every", "Zapis klatki co N", 1, 200, 1, "view", True),
    ("diagnostics_every", "Diagnostyka co N", 1, 200, 1, "view", True),
    ("point_stride", "Co która cząstka na ekranie", 1, 20, 1, "view", True),
)

_SECTION_LABELS = (
    ("start", "Warunki początkowe"),
    ("dynamics", "Dynamika"),
    ("solver", "Solver"),
    ("view", "Widok i zapis"),
)

RUNTIME_KEYS = frozenset(k for k, *_, live in _CONTROLS if live)
STARTUP_KEYS = frozenset(k for k, *_, live in _CONTROLS if not live)

_CHOICES: dict[str, tuple[str, ...]] = {
    "geometry": GEOMETRIES,
    "backend": BACKENDS,
    "device": DEVICES,
}


def ui_schema() -> dict[str, Any]:
    defaults = Config().to_flat()
    groups: dict[str, list[dict[str, Any]]] = {s: [] for s, _ in _SECTION_LABELS}
    for key, label, lo, hi, step, section, live in _CONTROLS:
        groups[section].append(
            {
                "key": key,
                "label": label,
                "min": lo,
                "max": hi,
                "step": step,
                "live": live,
                "value": defaults[key],
            }
        )
    return {
        "groups": [
            {"id": s, "label": lab, "controls": groups[s]} for s, lab in _SECTION_LABELS
        ],
        "choices": {
            "geometry": {"label": "Kształt startowy", "options": list(GEOMETRIES), "live": False},
            "backend": {"label": "Backend sił", "options": list(BACKENDS), "live": False},
            "device": {"label": "Urządzenie", "options": list(DEVICES), "live": False},
        },
        "presets": [{"id": k, "label": v[0]} for k, v in PRESETS.items()],
        "defaults": defaults,
        "runtime_keys": sorted(RUNTIME_KEYS),
    }


def apply_runtime(base: Config, params: dict[str, Any]) -> Config:
    """Zastosuj tylko te klucze, które wolno zmieniać w trakcie biegu.

    Klucze startowe i nieznane są po cichu pomijane — dzięki temu frontend może
    wysłać cały stan panelu, a serwer nie przestawi liczby cząstek w locie.
    """
    updates = {k: v for k, v in params.items() if k in RUNTIME_KEYS}
    return base.replace_flat(updates) if updates else base


def startup_config(base: Config, params: dict[str, Any]) -> Config:
    """Zbuduj config startowy: wszystkie znane klucze, łącznie z wyborami tekstowymi."""
    flat = base.to_flat()
    known = set(flat)
    for k, v in params.items():
        if k not in known:
            continue
        if k in _CHOICES:
            if v in _CHOICES[k]:
                flat[k] = v
            continue
        flat[k] = v
    return Config.from_flat(flat)


# ------------------------------------------------------------------ presety


def _preset(**updates: Any) -> Config:
    return Config().replace_flat(updates)


def gravity_for_beta(total_mass: float, radius: float, c: float, beta: float) -> float:
    """G takie, że prędkość okrężna na brzegu chmury wynosi ``beta·c``.

    Bez tego dobór G jest zgadywaniem: przy G = 1 i masie układu 20 000 prędkość
    okrężna wynosi √(GM/R) ≈ 45, więc dla c = 30 warunek orbity kołowej jest
    NIESPEŁNIALNY. Prędkości trafiają wtedy na limit 0,95 c, układ startuje
    z energią dodatnią i po prostu się rozlatuje — co wygląda jak błąd fizyki,
    a jest błędem doboru parametrów. Ta funkcja czyni intencję („chcę brzeg
    przy 0,25 c") jawną.
    """
    return float((beta * c) ** 2 * radius / max(total_mass, 1e-30))


# Softening w presetach siatkowych jest dobrany do oczka, które wyjdzie z
# rozciągłości danego kształtu. Gdyby był mniejszy, solver i tak podniósłby go
# do rozmiaru komórki — lepiej poprosić o to, co jest osiągalne.


def preset_galaxy() -> Config:
    """Wirujący dysk — rotacja równoważy grawitację, struktura się utrzymuje."""
    mass, radius, c = 4_000.0, 10.0, 30.0
    return _preset(
        geometry="disk", n_particles=20_000, radius=radius, total_mass=mass,
        rotation=1.0, temperature=0.004, c=c,
        G=gravity_for_beta(mass, radius, c, 0.25),
        softening=0.3, dt_max=0.01, grid=96,
    )


def preset_collapse() -> Config:
    """Zimna kula — kolaps grawitacyjny, wzrost γ, potem wirializacja."""
    mass, radius, c = 4_000.0, 10.0, 25.0
    return _preset(
        geometry="ball", n_particles=20_000, radius=radius, total_mass=mass,
        rotation=0.0, temperature=0.0, c=c,
        G=gravity_for_beta(mass, radius, c, 0.3),
        softening=0.3, dt_max=0.01, grid=96,
    )


def preset_relativistic() -> Config:
    """Wysokie β — efekty relatywistyczne widoczne gołym okiem w HUD."""
    mass, radius, c = 4_000.0, 6.0, 10.0
    return _preset(
        geometry="plummer", n_particles=8_000, radius=radius, total_mass=mass,
        rotation=0.5, temperature=0.1, c=c,
        G=gravity_for_beta(mass, radius, c, 0.7),
        softening=0.5, dt_max=0.004, accuracy=0.02, grid=64,
    )


def preset_merger() -> Config:
    """Dwie gromady na kursie kolizyjnym."""
    mass, radius, c = 4_000.0, 12.0, 30.0
    return _preset(
        geometry="two_clumps", n_particles=24_000, radius=radius, total_mass=mass,
        rotation=0.35, temperature=0.01, c=c,
        G=gravity_for_beta(mass, radius, c, 0.25),
        softening=0.4, dt_max=0.01, grid=96,
    )


def preset_precision() -> Config:
    """Mało cząstek, backend dokładny — do sprawdzania zachowania energii."""
    mass, radius, c = 4_000.0, 8.0, 30.0
    return _preset(
        geometry="plummer", n_particles=2_000, radius=radius, total_mass=mass,
        rotation=0.5, temperature=0.0, c=c,
        G=gravity_for_beta(mass, radius, c, 0.2),
        backend="exact", softening=0.25, dt_max=0.005, accuracy=0.015,
        diagnostics_every=25,
    )


PRESETS: dict[str, tuple[str, Any]] = {
    "galaxy": ("Galaktyka", preset_galaxy),
    "collapse": ("Kolaps", preset_collapse),
    "relativistic": ("Relatywistyczny", preset_relativistic),
    "merger": ("Zderzenie", preset_merger),
    "precision": ("Precyzja", preset_precision),
}


def preset(name: str) -> Config:
    if name not in PRESETS:
        raise KeyError(f"nieznany preset: {name!r}; dostępne: {sorted(PRESETS)}")
    return PRESETS[name][1]()
