"""Konfiguracja: zamrożone dataclassy, presety i schemat dla UI.

Jedno źródło prawdy. Schemat UI jest generowany z tej samej listy pól, które
opisują config, więc suwak nie może istnieć bez pola ani odwrotnie.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

#: Kolejność jest kolejnością w interfejsie: najpierw bryły zwarte, potem kształty
#: o wyróżnionej osi, na końcu rozkłady nieostre i układy złożone.
GEOMETRIES = (
    "ball",
    "cube",
    "cylinder",
    "disk",
    "torus",
    "sphere_shell",
    "filament",
    "gaussian",
    "two_clumps",
    "plummer",
)

#: Etykiety dla interfejsu wraz z tym, co dla danego kształtu znaczy „grubość".
#: Bez tej informacji suwak grubości jest dla większości kształtów zagadką.
GEOMETRY_LABELS: dict[str, str] = {
    "ball": "Kula (jednorodna)",
    "cube": "Kostka",
    "cylinder": "Walec",
    "disk": "Dysk — grubość = σ w pionie",
    "torus": "Torus — grubość = przekrój",
    "sphere_shell": "Powłoka sferyczna",
    "filament": "Włókno — grubość = σ przekroju",
    "gaussian": "Chmura gaussowska",
    "two_clumps": "Dwie gromady",
    "plummer": "Sfera Plummera (równowaga)",
}

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
    #: grubość poprzeczna jako ułamek promienia; używają jej kształty o wyróżnionym
    #: przekroju: `filament`, `torus`, `disk`. To ona, a nie promień, wyznacza
    #: długość fali fragmentacji (λ ≈ 3,6·σ), więc bez tego pokrętła te kształty
    #: były samopodobne i zawsze mieściły tyle samo długości Jeansa, niezależnie
    #: od promienia.
    thickness: float = 0.08
    #: spłaszczenie osi z, mnożnik stosowany do KAŻDEGO kształtu po wylosowaniu.
    #: 1 = proporcje własne kształtu, poniżej 1 placek, powyżej 1 cygaro. Rozdziela
    #: rozmiar od proporcji, więc zmiana kształtu nie zmienia przy okazji skali
    #: i porównania między biegami pozostają uczciwe.
    flatten: float = 1.0
    #: ułamek prędkości okrężnej nadawany na starcie (0 = zimny start)
    rotation: float = 0.6
    #: izotropowy rozrzut prędkości jako ułamek c
    temperature: float = 0.0
    #: docelowy stosunek wirialny 2K/|U|; > 0 przejmuje kontrolę nad `temperature`.
    #: 1 = równowaga, poniżej = układ się zapada, powyżej = rozlatuje.
    #:
    #: Istnieje, bo `temperature` NIE JEST porównywalna między kształtami. Ta sama
    #: dyspersja daje 2K/|U| = 1,34 dla kostki i 0,49 dla włókna — zmierzone,
    #: różnica 2,7-krotna. Bieg z ustaloną temperaturą miesza więc wpływ geometrii
    #: z wpływem tego, jak daleko od równowagi dany kształt wystartował, i nie
    #: pozwala rozstrzygnąć, co spowodowało wynik. Przy zadanym wiriale każdy
    #: kształt startuje z równowagi liczonej dla NIEGO i porównanie jest uczciwe.
    virial: float = 0.0


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

    # --- dyssypacja (patrz bone.cooling); 0 = model w pełni zachowawczy ---
    #: tempo tłumienia dyspersji prędkości przy ŚREDNIEJ gęstości układu, w 1/czas.
    #: Odwrotność jest czasem chłodzenia: 2,0 znaczy „e-krotny spadek dyspersji
    #: na pół jednostki czasu". Domyślne zero, bo dyssypacja zmienia klasę modelu
    #: i nie powinna się włączać niepostrzeżenie.
    cooling_rate: float = 0.0
    #: wykładnik zależności tempa od gęstości. 1 odpowiada emisyjności ∝ n² na
    #: jednostkę objętości, czyli tempu na cząstkę ∝ n — to dzięki temu gęste
    #: obszary chłodzą się szybciej i zapadają wybiórczo, zamiast żeby cały układ
    #: opadał równomiernie. 0 wyłącza zależność.
    cooling_density_power: float = 1.0
    #: bezwzględna podłoga dyspersji jako ułamek c — odpowiednik temperatury,
    #: poniżej której ośrodek przestaje promieniować. Ustala największą skalę
    #: fragmentu; 0 wyłącza.
    cooling_floor: float = 0.0
    #: podłoga NUMERYCZNA: ile cząstek musi liczyć masa Jeansa. To ona wykonuje
    #: tu właściwą pracę, bo podłoga nałożona na samą dyspersję nie umie obronić
    #: rozdzielczości — λ = σ√(π/Gρ) zależy też od gęstości, więc przy zapadaniu
    #: λ schodzi pod oczko siatki przy niezmienionym σ (zmierzone: błąd siły
    #: 10,5%, dryf energii ±7%).
    #:
    #: Dlaczego kryterium na MASĘ, a nie na λ ≥ 4ε (Truelove). Warunek na λ też
    #: został zmierzony i nie pomógł, bo gęstość jest odczytywana ze zgrubnej
    #: siatki chłodzenia i dla zwartych zgęstek jest zaniżona kilkudziesięciokrotnie
    #: — a próg na λ zależy od niej jak √ρ, więc dziedziczy ten błąd niemal w całości.
    #: Próg na masę Jeansa zależy od gęstości jak ρ^(1/6), czyli ten sam błąd
    #: gęstości przenosi się na niego dwukrotnie zamiast ośmiokrotnie. Do tego
    #: w kodzie na 10⁵ cząstek to rozdzielczość masowa wiąże pierwsza, nie siłowa.
    cooling_min_particles: float = 1000.0
    #: bok siatki, na której mierzony jest LOKALNY przepływ masowy. Ustala skalę
    #: słowa „lokalny": za gęsta daje kilka cząstek na komórkę i dyspersję z szumu,
    #: za zgrubna wlicza do przepływu ruch osobnych zgęstek względem siebie.
    #: 0 = dobierz z liczby cząstek (`bone.cooling.auto_grid`). Domyślnie automat,
    #: bo właściwa wartość skaluje się jak N^(1/3) — od 10 przy czterech tysiącach
    #: cząstek do 31 przy stu dwudziestu — i żadna stała nie jest tu poprawna.
    cooling_grid: int = 0


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


@dataclass(frozen=True)
class Slider:
    """Kontrolka liczbowa. ``live`` = wolno ruszać w trakcie biegu."""

    key: str
    label: str
    lo: float
    hi: float
    step: float
    section: str
    live: bool
    hint: str = ""
    kind: str = "range"


@dataclass(frozen=True)
class Choice:
    """Kontrolka wyboru z listy. Wartości muszą być kluczami z ``options``."""

    key: str
    label: str
    options: tuple[str, ...]
    section: str
    live: bool
    hint: str = ""
    labels: dict[str, str] | None = None
    kind: str = "choice"


#: Grupy panelu w kolejności wyświetlania. Podział idzie po PYTANIU, na które
#: odpowiada dana grupa, a nie po tym, w której dataclassie leży pole: „co
#: budujemy", „z czego", „jak to się rusza", „jakie prawa", „co to liczy".
#: Poprzedni podział szedł po dataclassach i mieszał kształt obiektu z rozrzutem
#: mas i temperaturą w jednej kupie zatytułowanej „warunki początkowe".
_SECTION_LABELS = (
    ("shape", "Kształt"),
    ("matter", "Materia"),
    ("motion", "Ruch początkowy"),
    ("laws", "Prawa fizyki"),
    ("cooling", "Dyssypacja"),
    ("integration", "Całkowanie"),
    ("solver", "Solver"),
    ("view", "Widok i zapis"),
)

#: Ile pierwszych grup jest rozwiniętych na starcie — te, które określają
#: badany obiekt. Reszta to nastawy, których zwykle nie rusza się co bieg.
OPEN_SECTIONS = 3

_CONTROLS: tuple[Slider | Choice, ...] = (
    # ---- Kształt: co budujemy
    Choice(
        "geometry", "Kształt", GEOMETRIES, "shape", False,
        hint="Rozmiar ustawia promień, proporcje — spłaszczenie.",
        labels=GEOMETRY_LABELS,
    ),
    Slider(
        "radius", "Promień / połowa boku", 1.0, 40.0, 0.5, "shape", False,
        hint="Skala kształtu. Dla kostki to połowa boku.",
    ),
    Slider(
        "thickness", "Grubość przekroju (× promień)", 0.005, 1.0, 0.005, "shape", False,
        hint="Działa dla dysku, torusa i włókna — reszta kształtów jej nie używa.",
    ),
    Slider(
        "flatten", "Spłaszczenie osi z", 0.05, 4.0, 0.05, "shape", False,
        hint="1 = proporcje własne kształtu, mniej = placek, więcej = cygaro.",
    ),
    # ---- Materia: z czego
    Slider(
        "n_particles", "Liczba cząstek", 100, 200_000, 100, "matter", False,
        hint="Rozdzielczość, nie fizyka: masa układu jest niezależna.",
    ),
    Slider("total_mass", "Masa układu", 100.0, 100_000.0, 100.0, "matter", False),
    Slider(
        "mass_spread", "Rozrzut mas cząstek", 0.0, 1.0, 0.01, "matter", False,
        hint="Suma jest przeskalowana do masy układu, więc to tylko niejednorodność.",
    ),
    Slider("seed", "Ziarno losowe", 0, 9999, 1, "matter", False),
    # ---- Ruch początkowy: jak to się rusza na starcie
    Slider(
        "rotation", "Rotacja (× v okrężna)", 0.0, 1.4, 0.02, "motion", False,
        hint="1 = orbita kołowa z masy zamkniętej w promieniu, dla dowolnego G.",
    ),
    Slider(
        "temperature", "Dyspersja prędkości (× c)", 0.0, 0.5, 0.005, "motion", False,
        hint="Rozrzut izotropowy. Pomijana, gdy zadany jest wirial.",
    ),
    Slider(
        "virial", "Wirial 2K/|U| (0 = ręcznie)", 0.0, 2.0, 0.05, "motion", False,
        hint="1 = równowaga. Jedyny sposób uczciwego porównania różnych kształtów.",
    ),
    # ---- Prawa fizyki
    Slider(
        "G", "Stała grawitacji G", 0.05, 5.0, 0.05, "laws", True,
        hint="Patrz gravity_for_beta, jeśli chcesz zadaną prędkość na brzegu.",
    ),
    Slider("c", "Prędkość światła c", 2.0, 200.0, 1.0, "laws", True),
    Slider(
        "softening", "Softening ε", 0.02, 3.0, 0.01, "laws", True,
        hint="Najmniejsza sensowna skala. Solver siatkowy podnosi ją do oczka.",
    ),
    # ---- Dyssypacja
    Slider(
        "cooling_rate", "Chłodzenie λ (0 = brak)", 0.0, 20.0, 0.1, "cooling", True,
        hint="Odwrotność to czas chłodzenia. Warto porównać z czasem dynamicznym.",
    ),
    Slider(
        "cooling_density_power", "Zależność od gęstości", 0.0, 2.0, 0.1, "cooling", True,
        hint="1 = tempo ∝ gęstość, czyli gęste obszary zapadają się wybiórczo.",
    ),
    Slider(
        "cooling_floor", "Podłoga dyspersji (× c)", 0.0, 0.05, 0.001, "cooling", True,
        hint="Podłoga fizyczna: ustala największy fragment. 0 wyłącza.",
    ),
    Slider(
        "cooling_min_particles", "Podłoga: cząstek na masę Jeansa",
        0.0, 5000.0, 100.0, "cooling", True,
        hint="Podłoga numeryczna — pilnuje, żeby fragmenty były rozdzielone.",
    ),
    Slider(
        "cooling_grid", "Siatka chłodzenia (0 = automat)", 0, 128, 4, "cooling", False,
        hint="Ustala, jak duży obszar liczy się jako lokalny przepływ masowy.",
    ),
    # ---- Całkowanie
    Slider("dt_max", "Maksymalny krok dt", 0.001, 0.2, 0.001, "integration", True),
    Slider(
        "accuracy", "Dokładność kroku η", 0.005, 0.2, 0.005, "integration", True,
        hint="Krok adaptacyjny: dt ≤ η·√(ε/a_max). Mniej = dokładniej i wolniej.",
    ),
    Slider("time_scale", "Tempo symulacji ×", 1, 40, 1, "integration", True),
    # ---- Solver
    Choice("backend", "Backend sił", BACKENDS, "solver", False,
           hint="exact liczy wszystkie pary, mesh rozwiązuje Poissona na siatce."),
    Choice("device", "Urządzenie", DEVICES, "solver", False),
    Slider("grid", "Siatka PM (bok)", 32, 256, 32, "solver", False),
    Slider("box_margin", "Margines pudła", 0.0, 1.0, 0.05, "solver", True),
    Slider("exact_max_particles", "Próg exact→mesh", 500, 20_000, 500, "solver", False),
    Slider(
        "error_check_every", "Pomiar błędu co N kroków", 0, 500, 10, "solver", True,
        hint="Porównanie z dokładnym O(N²) na próbce. 0 = nie mierz.",
    ),
    # ---- Widok i zapis
    Slider("live_every", "Odświeżanie widoku co N", 1, 40, 1, "view", True),
    Slider("trajectory_every", "Zapis klatki co N", 1, 200, 1, "view", True),
    Slider("diagnostics_every", "Diagnostyka co N", 1, 200, 1, "view", True),
    Slider("point_stride", "Co która cząstka na ekranie", 1, 20, 1, "view", True),
)

RUNTIME_KEYS = frozenset(c.key for c in _CONTROLS if c.live)
STARTUP_KEYS = frozenset(c.key for c in _CONTROLS if not c.live)

#: Dozwolone wartości pól wyboru — wprost z tych samych kontrolek, więc nie ma
#: jak się rozjechać z listą pokazywaną w interfejsie.
_CHOICES: dict[str, tuple[str, ...]] = {
    c.key: c.options for c in _CONTROLS if isinstance(c, Choice)
}


def _control_schema(control: Slider | Choice, value: Any) -> dict[str, Any]:
    common = {
        "key": control.key,
        "label": control.label,
        "kind": control.kind,
        "live": control.live,
        "hint": control.hint,
        "value": value,
    }
    if isinstance(control, Choice):
        labels = control.labels or {}
        common["options"] = [
            {"value": o, "label": labels.get(o, o)} for o in control.options
        ]
    else:
        common.update({"min": control.lo, "max": control.hi, "step": control.step})
    return common


def ui_schema() -> dict[str, Any]:
    defaults = Config().to_flat()
    groups: dict[str, list[dict[str, Any]]] = {s: [] for s, _ in _SECTION_LABELS}
    for control in _CONTROLS:
        groups[control.section].append(_control_schema(control, defaults[control.key]))
    return {
        "groups": [
            {"id": s, "label": lab, "controls": groups[s]} for s, lab in _SECTION_LABELS
        ],
        "open_sections": OPEN_SECTIONS,
        # Pełna konfiguracja jedzie ze schematem, żeby chip mógł wystartować
        # bieg bez drugiego żądania. Osobny /api/preset zostaje dla testów
        # i dla klienta, który schemat już ma, a chce odświeżyć jeden preset.
        "presets": [
            {"id": k, "label": v[0], "config": v[1]().to_flat()} for k, v in PRESETS.items()
        ],
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
        # grubość podana wprost: dysk czyta ją teraz z konfiguracji, a domyślne
        # 0,08 dałoby dysk dwa razy grubszy niż ten, na którym preset był dobrany
        geometry="disk", n_particles=20_000, radius=radius, total_mass=mass,
        thickness=0.04,
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


def preset_fragmentation() -> Config:
    """Wirujący cienki torus rozpadający się na osobne zgęstki.

    Fragmentacja wymaga, żeby najbardziej niestabilna długość fali była wyraźnie
    mniejsza od obiektu ORAZ żeby rosła szybciej niż jakikolwiek mod globalny.
    Drugi warunek eliminuje po kolei niemal wszystkie kształty:

    * zwirializowana kula — λ_Jeansa jest u niej rzędu promienia, brak okna;
    * samograwitujący dysk — λ_kryt ≈ 4πR, czyli więcej niż on sam, więc wychodzi
      jeden bar, a nie rój grudek;
    * włókno — ma swobodne końce, a te zapadają się do środka w tempie ~1/grubość,
      czyli DOKŁADNIE tak samo jak rośnie fragmentacja. Pocienianie włókna
      przyspiesza oba mody równo i nie rozdziela ich. Zmierzone dwa razy, dla
      grubości 0,64 i 0,28: za każdym razem włókno zbiera się w jeden obiekt.

    Cienki torus usuwa oba globalne mody naraz. Nie ma końców, więc nie ma efektu
    brzegowego; ma rotację, więc nie zapada się promieniowo. Zostaje wyłącznie
    fragmentacja podłużna.

    Liczby wynikają z dwóch warunków. Przekrój musi być stabilny, czyli masa
    liniowa poniżej krytycznej M_lin < 2σ_v²/G: przy obwodzie 2π·8 = 50 i masie
    4000 mamy M_lin = 79,5, co wymaga σ_v > 1,69 — stąd σ_v = 1,7, na granicy.
    Długość fali λ = σ_v√(π/Gρ) ≈ 1,5 mieści się wtedy 33 razy na obwodzie
    i wypada na 15 oczkach siatki, więc jest rozdzielona z zapasem.
    """
    mass, radius, c = 4_000.0, 8.0, 30.0
    sigma_1d = 1.7
    return _preset(
        geometry="torus", n_particles=120_000, radius=radius, total_mass=mass,
        # przekrój pierścienia: promień mniejszy 0,085·8 = 0,68, czyli σ_rms ≈ 0,48
        thickness=0.085,
        # rotacja lekko ponad 1, bo `circular_speed` liczy prędkość okrężną z masy
        # zamkniętej w promieniu, a cienki pierścień przyciąga swoje elementy
        # mocniej niż punktowa masa w środku — potrzebuje o kilka procent więcej
        rotation=1.05,
        # `temperature` jest zadawane jako ułamek c, a spawn rozdziela je na trzy
        # składowe, więc σ na składową to temperature·c/√3 — stąd ten przelicznik
        temperature=sigma_1d * (3.0**0.5) / c, c=c,
        G=gravity_for_beta(mass, radius, c, 0.2),
        softening=0.03, dt_max=0.004, accuracy=0.025,
        # pudło siatki dopasowuje się do rozciągłości pierścienia (17,4), a istotna
        # jest grubość przekroju (0,48) — stąd siatka musi być gruba, mimo że sam
        # obiekt jest cienki: oczko 0,098 daje 15 komórek na długość fali
        grid=192, box_margin=0.05,
        error_check_every=100,
        # zgęstki rosną przez ~2,5 jednostki czasu, a potem zaczynają się łączyć;
        # gęsty zapis klatek pozwala analizować właściwą chwilę, nie tylko koniec
        trajectory_every=10,
    )


def preset_dissipation() -> Config:
    """Zwirializowana kula, która się chłodzi — i dlatego fragmentuje.

    To jest kontrprzykład do presetu `fragmentation`. Tam trzeba było uciekać od
    kuli do pierścienia, bo w modelu w pełni zachowawczym zwirializowana chmura
    jest zamknięta: jej rozkład prędkości podtrzymuje ją przeciw zapadaniu, a
    energii nie ma gdzie podziać. Tutaj energia ma gdzie się podziać, więc ta
    sama kula robi to, czego przedtem zrobić nie mogła.

    Mechanizm jest ilościowy i wart wypisania, bo to on decyduje o doborze λ.
    Długość Jeansa λ = σ√(π/Gρ) jest w zwirializowanej kuli rzędu jej promienia,
    czyli mieści się w niej JEDNA masa Jeansa — stąd zapadanie monolityczne albo
    żadne. Chłodzenie obniża σ, więc λ maleje, a liczba mas Jeansa rośnie jak
    (R/λ)³. Spadek σ trzykrotny daje kilka fragmentów, pięciokrotny kilkadziesiąt.

    Tempo λ = 2 jest dobrane względem czasu dynamicznego kuli (≈ 1,1): chłodzenie
    musi być szybsze od zapadania, inaczej chmura opadnie całością zanim zdąży się
    podzielić, ale nie o rzędy wielkości szybsze, bo wtedy σ trafia w podłogę
    wcześniej niż niestabilność urośnie.

    Podłoga jest tu warunkiem sensowności wyniku, nie ozdobą. Stały próg na samą
    dyspersję został zmierzony i nie wystarcza: λ = σ√(π/Gρ) zależy też od
    gęstości, więc gdy kula zapada się i ρ rośnie, λ schodzi pod oczko siatki przy
    niezmienionym σ — wyszedł błąd siły 10,5% i dryf energii ±7%. Pracę wykonuje
    próg na liczbę cząstek w masie Jeansa: przy tysiącu cząstek na fragment
    dyspersja zatrzymuje się na tyle wysoko, żeby fragmenty pozostały rozdzielone.

    Wynika z tego ograniczenie, które trzeba wypowiedzieć wprost: masa fragmentu
    jest tu ustalona przez rozdzielczość, a nie przez fizykę. Prawdziwa masa
    fragmentu wynikałaby z funkcji chłodzenia i równania stanu gazu, których ten
    model nie ma. Wynik mówi więc „chłodzenie prowadzi do fragmentacji", a nie
    „fragmenty mają taką a taką masę".

    Kula JEDNORODNA, nie profil Plummera, i to nie jest kwestia gustu. Plummer
    rozciąga cząstki do promienia 6a, mając połowę masy w 1,3a, więc pudło siatki
    dopasowuje się do rozmiaru dwadzieścia razy większego od interesującego
    obszaru. Zmierzone: oczko wychodziło 0,288 i podnosiło ε trzykrotnie ponad
    zadane, a błąd siły dobijał 3,7%. Kula jednorodna ma zwarty brzeg, więc oczko
    schodzi do 0,144 przy tej samej siatce, no i dodatkowo daje jednorodną długość
    Jeansa — czyli fragmenty o podobnej masie zamiast jednego dominującego jądra.
    """
    mass, radius, c = 4_000.0, 8.0, 30.0
    return _preset(
        geometry="ball", n_particles=120_000, radius=radius, total_mass=mass,
        # warunek wirialny 2K = |U| dla kuli jednorodnej daje σ² = GM/(5R);
        # przy G = 0,072 wychodzi σ ≈ 2,68, czyli temperature = σ√3/c ≈ 0,155.
        # Rotacja przejmuje część podparcia, więc dyspersja jest odrobinę niższa.
        rotation=0.2, temperature=0.15, c=c,
        G=gravity_for_beta(mass, radius, c, 0.2),
        # ε ledwo nad oczkiem siatki (0,144), żeby nie było po cichu podnoszone
        softening=0.15, dt_max=0.01, accuracy=0.03,
        grid=128, box_margin=0.1,
        # λ = 1 przy czasie dynamicznym 2,7: chłodzenie szybsze od zapadania, ale
        # nie o rzędy wielkości. Przy λ = 2 zmierzony dryf energii dobijał 6%,
        # bo zgęstki zapadały się poniżej oczka szybciej, niż krok zdążył zmaleć.
        # Podłoga wyłącznie numeryczna: 2000 cząstek na masę Jeansa daje λ ≈ 3,3,
        # czyli 22 długości softeningu, i ~60 fragmentów w kuli. Bezwzględna
        # podłoga dyspersji jest tu zbędna, bo nie ma fizycznej temperatury,
        # którą miałaby reprezentować — patrz uwaga o masie fragmentu wyżej.
        cooling_rate=1.0, cooling_density_power=1.0,
        cooling_floor=0.0, cooling_min_particles=2000.0,
        # automat wybierze tu 31³, czyli ~9 cząstek na zajętą komórkę — tyle
        # trzeba, żeby dyspersja nie była zdominowana przez szum próbkowania
        cooling_grid=0,
        error_check_every=200, trajectory_every=10,
    )


def _shape_probe(geometry: str, **overrides: Any) -> Config:
    """Wspólna podstawa presetów kształtu — jedyną różnicą jest geometria.

    Sens tych presetów jest porównawczy, więc masa, promień, G, c i rozdzielczość
    MUSZĄ być identyczne. Gdyby każdy kształt miał własne liczby, różnice w wyniku
    dałoby się przypisać czemukolwiek, a nie kształtowi.

    Warunek startowy jest zadany WIRIAŁEM, nie temperaturą, i to jest tu rzecz
    najważniejsza. Ustalona dyspersja nie jest porównywalna między kształtami:
    ta sama liczba daje 2K/|U| = 1,34 dla kostki i 0,49 dla włókna, czyli różnicę
    2,7-krotną. Bieg z ustaloną temperaturą mierzyłby więc mieszankę wpływu
    geometrii i wpływu tego, jak daleko od równowagi kształt wystartował. Przy
    2K/|U| = 1 każdy kształt startuje z własnej równowagi i jedyną zmienną
    pozostaje geometria.
    """
    mass, radius, c = 4_000.0, 8.0, 30.0
    G = gravity_for_beta(mass, radius, c, 0.2)
    return _preset(
        geometry=geometry, n_particles=40_000, radius=radius, total_mass=mass,
        rotation=0.0, virial=1.0, c=c, G=G,
        softening=0.15, dt_max=0.01, accuracy=0.03,
        grid=128, box_margin=0.1,
        error_check_every=200, trajectory_every=20,
        **overrides,
    )


def preset_shape_cube() -> Config:
    """Kostka zwirializowana — brzeg płaski, więc narożniki opadają pierwsze."""
    return _shape_probe("cube")


def preset_shape_cylinder() -> Config:
    """Walec zwirializowany — jedna wyróżniona os, dwa swobodne końce."""
    return _shape_probe("cylinder")


def preset_shape_torus() -> Config:
    """Gruby torus bez rotacji — dziura w środku zamyka się albo nie."""
    return _shape_probe("torus", thickness=0.35)


def preset_shape_slab() -> Config:
    """Kula spłaszczona do placka — ten sam materiał, inne proporcje.

    Preset istnieje po to, żeby pokazać, co robi samo spłaszczenie: masa, promień
    i dyspersja są takie jak w pozostałych presetach kształtu, zmienia się tylko
    proporcja pionowa. Placek zapada się najpierw w cieńszym kierunku, bo tam
    droga swobodnego spadku jest krótsza — i to jest widoczne od pierwszych kroków.
    """
    return _shape_probe("ball", flatten=0.15)


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
    "fragmentation": ("Fragmentacja", preset_fragmentation),
    "dissipation": ("Dyssypacja", preset_dissipation),
    # presety porównawcze kształtów: identyczne liczby, różna geometria
    "shape_cube": ("Kostka", preset_shape_cube),
    "shape_cylinder": ("Walec", preset_shape_cylinder),
    "shape_torus": ("Torus", preset_shape_torus),
    "shape_slab": ("Placek", preset_shape_slab),
    "precision": ("Precyzja", preset_precision),
}


def preset(name: str) -> Config:
    if name not in PRESETS:
        raise KeyError(f"nieznany preset: {name!r}; dostępne: {sorted(PRESETS)}")
    return PRESETS[name][1]()
