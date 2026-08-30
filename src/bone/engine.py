"""Silnik: stan + backend + integrator + diagnostyka.

Zmiany konfiguracji w trakcie biegu są czytane w KAŻDEJ iteracji. To wygląda na
drobiazg, ale poprzednia wersja czytała częstotliwość odświeżania raz przed
pętlą, przez co suwaki opisane w UI jako „działa na żywo" nie robiły nic.
Jeśli parametr jest oznaczony jako runtime, musi działać jak runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bone import integrator
from bone.backends import Backend, cuda_available, make_backend
from bone.config import Config
from bone.cooling import Cooling
from bone.diagnostics import Diagnostics, measure_force_error
from bone.spawn import make_state
from bone.state import State


@dataclass
class Callbacks:
    on_view: Callable[[State, Config], None] | None = None
    on_frame: Callable[[State, Config], None] | None = None
    on_diagnostics: Callable[[dict[str, float]], None] | None = None
    should_stop: Callable[[], bool] | None = None
    poll_config: Callable[[], Config | None] | None = None


class Engine:
    def __init__(
        self,
        cfg: Config | None = None,
        state: State | None = None,
        backend: Backend | None = None,
    ) -> None:
        self.cfg = cfg or Config()
        self.state = state if state is not None else make_state(self.cfg)
        self.backend = backend or make_backend(self.cfg, self.state.n)
        self.diagnostics = Diagnostics()
        self.cooling = self._make_cooling(self.cfg)
        self._rng = np.random.default_rng(self.cfg.spawn.seed + 1)
        self._last_dt = float(self.cfg.physics.dt_max)
        self._pending_error: dict[str, float] = {}
        self._last_force_error: float | None = None
        #: skumulowana energia odprowadzona przez dyssypację; wielkością zachowaną
        #: w modelu z chłodzeniem jest E_tot + ta suma, i to jej dryf mierzymy
        self.energy_removed = 0.0
        integrator.ensure_field(self.backend, self.state, self.cfg.physics)

    # ------------------------------------------------------------------ bieg

    @staticmethod
    def _make_cooling(cfg: Config) -> Cooling:
        """Chłodzenie liczy się tam, gdzie liczą się siły.

        Rozstrzygnięcie 'auto' musi przebiegać dokładnie tak samo jak w fabryce
        backendów, inaczej scatter/gather chłodzenia wylądowałby na procesorze
        obok solvera na GPU — a wtedy operator pomocniczy kosztuje więcej niż
        rozwiązanie równania Poissona, co było zmierzone.
        """
        device = cfg.solver.device
        use_cuda = device == "cuda" or (device == "auto" and cuda_available())
        return Cooling(
            grid=cfg.physics.cooling_grid, device="cuda" if use_cuda else "cpu"
        )

    def step(self) -> float:
        self._last_dt = integrator.step(self.backend, self.state, self.cfg.physics)
        # Rozdzielenie operatorów: całkowanie zachowawcze, potem dyssypacja.
        # Chłodzenie zmienia tylko pędy, więc siły policzone w kroku pozostają
        # aktualne i nie ma potrzeby ich unieważniać.
        self.energy_removed += self.cooling.apply(self.state, self.cfg.physics, self._last_dt)
        return self._last_dt

    def advance(self, n_steps: int, should_stop: Callable[[], bool] | None = None) -> float:
        """Wykonaj n kroków. Zwraca sumaryczny upływ czasu symulacji.

        `should_stop` jest sprawdzane po każdym kroku, nie po całej paczce. Bez
        tego żądanie zatrzymania czekało na `time_scale` kroków — przy suwaku
        ustawionym na 40 i dużym układzie to kilka sekund ciszy po kliknięciu
        Stop, czyli przycisk wyglądający na zepsuty.
        """
        elapsed = 0.0
        for _ in range(max(1, n_steps)):
            elapsed += self.step()
            if should_stop is not None and should_stop():
                break
        return elapsed

    def collect_diagnostics(self) -> dict[str, float]:
        potential = self.state.potential
        if potential is None:
            potential = np.zeros(self.state.n)
        extra = dict(self._pending_error)
        self._pending_error.clear()
        extra["approximate"] = 1.0 if self.backend.approximate else 0.0
        extra["softening_eff"] = self.effective_softening
        if self.cfg.physics.cooling_rate > 0.0:
            extra["cooling_ppc"] = self.cooling.particles_per_cell
        return self.diagnostics.observe(
            self.state,
            potential,
            self.cfg.physics.c,
            self._last_dt,
            extra,
            energy_removed=self.energy_removed,
        )

    @property
    def effective_softening(self) -> float:
        return self.backend.effective_softening(self.cfg.physics.softening)

    def check_backend_error(self) -> dict[str, float]:
        """Zmierz błąd przybliżenia względem dokładnego O(N²).

        Wzorzec liczymy z softeningiem, którym backend NAPRAWDĘ pracuje —
        inaczej mierzylibyśmy nie błąd metody, tylko różnicę dwóch modeli.
        """
        if self.state.forces is None:
            return {}
        result = measure_force_error(
            self.state,
            self.state.forces,
            self.cfg.physics.G,
            self.effective_softening,
            self.cfg.solver.error_check_sample,
            self._rng,
        )
        self._pending_error = result
        self._last_force_error = result.get("force_err_rms")
        return result

    def run(self, callbacks: Callbacks | None = None) -> Diagnostics:
        cb = callbacks or Callbacks()
        cfg = self.cfg
        unlimited = cfg.run.steps <= 0
        performed = 0

        if cb.on_view is not None:
            cb.on_view(self.state, cfg)

        while unlimited or performed < cfg.run.steps:
            if cb.should_stop is not None and cb.should_stop():
                break
            if cb.poll_config is not None:
                updated = cb.poll_config()
                if updated is not None:
                    self.apply_config(updated)

            # runtime: odczytywane co iterację, bo mogły się zmienić
            run = self.cfg.run
            self.advance(max(1, int(run.time_scale)), cb.should_stop)
            performed += 1

            check_every = int(self.cfg.solver.error_check_every)
            if check_every > 0 and performed % check_every == 0:
                self.check_backend_error()

            if cb.on_diagnostics is not None and performed % max(1, run.diagnostics_every) == 0:
                cb.on_diagnostics(self.collect_diagnostics())
            if cb.on_view is not None and performed % max(1, run.live_every) == 0:
                cb.on_view(self.state, self.cfg)
            if cb.on_frame is not None and performed % max(1, run.trajectory_every) == 0:
                cb.on_frame(self.state, self.cfg)

            if not self.state.is_finite():
                raise FloatingPointError(
                    f"stan przestał być skończony na kroku {self.state.step}"
                )

        return self.diagnostics

    # -------------------------------------------------------------- config

    def apply_config(self, cfg: Config) -> None:
        """Podmień konfigurację runtime; przebuduj backend, jeśli zmieniły się
        jego parametry. Siły są unieważniane, bo zależą od G i ε."""
        old = self.cfg
        self.cfg = cfg
        solver_changed = old.solver != cfg.solver
        physics_changed = (
            old.physics.G != cfg.physics.G or old.physics.softening != cfg.physics.softening
        )
        if (
            old.physics.cooling_grid != cfg.physics.cooling_grid
            or old.solver.device != cfg.solver.device
        ):
            self.cooling = self._make_cooling(cfg)
        if solver_changed:
            self.backend.close()
            self.backend = make_backend(cfg, self.state.n)
        if solver_changed or physics_changed:
            self.state.forces = None
            self.state.potential = None
            integrator.ensure_field(self.backend, self.state, cfg.physics)
            self.diagnostics.reset_reference()

    def describe(self) -> str:
        if self.cfg.physics.cooling_rate > 0.0:
            return f"{self.backend.describe()} · {self.cooling.describe()}"
        return self.backend.describe()

    def accuracy_hint(self) -> str:
        """Co zrobić z wysokim błędem siły — sama liczba nie mówi nic o działaniu.

        Błąd rośnie w trakcie biegu, kiedy układ wytworzy strukturę mniejszą od
        oczka siatki: PM rozdziela grawitację tylko do rozmiaru komórki.
        """
        err = self._last_force_error
        if err is None or not self.backend.approximate:
            return ""
        if err < 0.02:
            return ""
        finer = min(2 * self.cfg.solver.grid, 384)
        remedy = (
            f"zagęść siatkę do {finer}³" if finer > self.cfg.solver.grid else "użyj backendu „exact”"
        )
        return (
            f"błąd siły {err:.1%} — układ ma strukturę drobniejszą od oczka siatki; {remedy}"
        )

    def close(self) -> None:
        self.backend.close()
