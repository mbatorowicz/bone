"""Silnik SR — bez society / ścian / auto-stop społecznego."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from bone.config.schema import AppConfig
from bone.domain.universe import Universe, spawn
from bone.metrics.observe import MetricsTracker
from bone.physics.forces import compute_forces
from bone.physics.integrate import leapfrog_step
from bone.physics.neighbors import get_or_build_neighbors

LiveCb = Callable[[Universe], None]
ProgressCb = Callable[[int, int, Universe, dict], None]
StopFn = Callable[[], bool]
PollCfg = Callable[[], AppConfig | None]


class Engine:
    def __init__(self, cfg: AppConfig | None = None, universe: Universe | None = None):
        self.cfg = cfg or AppConfig()
        self.universe = universe or spawn(self.cfg)
        self.universe.config = self.cfg
        self.tracker = MetricsTracker()
        self._forces = compute_forces(
            self.universe, neighbors=get_or_build_neighbors(self.universe)
        )

    def step_once(self) -> dict | None:
        u = self.universe
        cfg = u.config
        phys = cfg.physics
        speed = int(max(1, min(40, phys.sim_speed)))
        base_dt = float(phys.dt)
        if speed > 1:
            u.config = replace(cfg, physics=replace(phys, dt=base_dt * speed))
        try:
            self._forces, _ = leapfrog_step(u, forces=self._forces)
        finally:
            p = u.config.physics
            u.config = replace(u.config, physics=replace(p, dt=base_dt))
            self.cfg = u.config

        if u.step % max(1, cfg.io.snapshot_every) == 0:
            return self.tracker.observe(u)
        return None

    def run(
        self,
        *,
        on_live: LiveCb | None = None,
        on_progress: ProgressCb | None = None,
        should_stop: StopFn | None = None,
        poll_config: PollCfg | None = None,
        on_frame: Callable[[Universe], None] | None = None,
    ) -> MetricsTracker:
        unlimited = self.cfg.io.steps <= 0
        total = self.cfg.io.steps
        live_every = max(1, self.cfg.io.live_every)
        traj_every = max(1, self.cfg.io.traj_every)
        s = 0
        if on_live is not None:
            on_live(self.universe)
        while True:
            s += 1
            if not unlimited and s > total:
                break
            if should_stop is not None and should_stop():
                break
            if poll_config is not None:
                updated = poll_config()
                if updated is not None:
                    self.cfg = updated
                    self.universe.config = updated
            row = self.step_once()
            u = self.universe
            if on_live is not None and s % live_every == 0:
                on_live(u)
            if on_frame is not None and s % traj_every == 0:
                on_frame(u)
            if row is not None and on_progress is not None:
                on_progress(s, total if not unlimited else s, u, row)
        return self.tracker
