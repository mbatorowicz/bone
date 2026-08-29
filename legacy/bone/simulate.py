"""Pętla główna symulacji — z checkpointami i kontynuacją ewolucji."""

from __future__ import annotations

import json
import webbrowser
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from bone.checkpoint import load_checkpoint, load_trajectory_lists, save_checkpoint
from bone.constants import SimConfig
from bone.evolve_traits import evolve_traits
from bone.geometry import resolve_geometry
from bone.gravity import compute_forces, potential_energy
from bone.integrate import leapfrog_step
from bone.metrics import MetricsTracker
from bone.singularity import concentration_reached
from bone.traits import Universe, spawn_newborns
from bone.trajectory import TrajectoryWriter
from bone.viewer import (
    local_density_colors,
    save_cube_frames_png,
    save_cube_gif,
    save_cube_html,
    save_live_frame,
)
from bone.viz import save_scatter_projection, save_timeseries, save_trait_histograms


ProgressCb = Callable[[int, int, Universe, dict], None]
LiveCb = Callable[[Universe], None]
StopCb = Callable[[], bool]
ConfigUpdateCb = Callable[[], SimConfig | None]


def _cube_half(cfg: SimConfig) -> float:
    return 0.5 * (cfg.grid_n - 1) * cfg.spacing + cfg.wall_margin


def _record_frame(
    universe: Universe,
    writer: TrajectoryWriter,
    *,
    ram_frames: list[np.ndarray],
    ram_times: list[float],
    ram_colors: list[np.ndarray],
) -> None:
    """Klatka do chunków + krótki bufor RAM (HTML/GIF na końcu)."""
    radius = universe.config.spacing * 1.6
    neigh = getattr(universe, "neighbors", None)
    if neigh is not None:
        deg = neigh.pair_degree(universe.n, radius, universe.positions)
        col = local_density_colors(universe.positions, radius=radius, degree=deg)
    else:
        col = local_density_colors(universe.positions, radius=radius)
    pos = universe.positions.astype(np.float32, copy=True)
    writer.add(pos, universe.t, col, half=_cube_half(universe.config))
    # ogranicz RAM — ostatnie ~400 klatek do GIF/HTML
    ram_frames.append(pos)
    ram_times.append(universe.t)
    ram_colors.append(col)
    max_ram = 400
    if len(ram_frames) > max_ram:
        del ram_frames[: len(ram_frames) - max_ram]
        del ram_times[: len(ram_times) - max_ram]
        del ram_colors[: len(ram_colors) - max_ram]


def _flush_live(
    universe: Universe,
    writer: TrajectoryWriter,
    frames: list[np.ndarray],
    times: list[float],
    colors: list[np.ndarray],
    out: Path,
    *,
    write_html: bool = False,
    compressed_ckpt: bool = False,
) -> None:
    """Checkpoint + chunk trajektorii; HTML rzadko."""
    writer.flush_chunk()
    save_checkpoint(universe, out / "checkpoint.npz", compressed=compressed_ckpt)
    np.savez(
        out / "final_state.npz",
        positions=universe.positions,
        velocities=universe.velocities,
        knowledge=universe.traits["knowledge"],
        love=universe.traits["love"],
        hatred=universe.traits["hatred"],
        loyalty=universe.traits["loyalty"],
        health=universe.traits["health"],
        wisdom=universe.traits["wisdom"],
        endurance=universe.traits["endurance"],
        wealth=universe.traits["wealth"],
        alive=universe.traits["alive"],
        t=np.array([universe.t]),
        step=np.array([universe.step]),
    )
    if write_html and frames:
        half = _cube_half(universe.config)
        save_cube_html(frames, times, colors, out, half=half)


def _finalize_outputs(
    universe: Universe,
    cfg: SimConfig,
    out: Path,
    frames: list[np.ndarray],
    times: list[float],
    colors: list[np.ndarray],
    tracker: MetricsTracker,
    continuing: bool,
    interrupted: bool,
    writer: TrajectoryWriter,
) -> dict:
    u_pot = potential_energy(universe)
    events = tracker.events.as_dict()
    traj_path = writer.finalize()

    summary = {
        "config": {
            "grid_n": cfg.grid_n,
            "geometry": resolve_geometry(cfg.geometry),
            "steps": cfg.steps,
            "dt": cfg.dt,
            "seed": cfg.seed,
            "G": cfg.G,
            "c": cfg.c,
            "r_cut": cfg.r_cut,
            "continued": continuing,
            "interrupted": interrupted,
            "absolute_step": universe.step,
            "t": universe.t,
        },
        "final": tracker.history[-1] if tracker.history else {},
        "potential_energy": u_pot,
        "events": events,
        "n_particles": universe.n,
        "n_frames": writer.total_frames,
    }

    (out / "events.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if tracker.history:
        save_timeseries(tracker.history, out)
    save_scatter_projection(universe, out, tag="final")
    save_trait_histograms(universe, out)

    half = _cube_half(cfg)
    save_checkpoint(universe, out / "checkpoint.npz", compressed=True)
    if frames:
        save_cube_frames_png(frames, times, colors, out, half=half)
        html_path = save_cube_html(frames, times, colors, out, half=half)
    else:
        html_path = out / "cube_view.html"
    gif_path = None
    if cfg.make_gif and len(frames) >= 2:
        print("Generuje animacje GIF szescianu...", flush=True)
        gif_path = save_cube_gif(frames, times, colors, out, half=half, fps=12, stride=1)

    print("\n=== Hitting times ===", flush=True)
    for key, val in events.items():
        print(f"  {key}: {val if val is not None else 'nie osiągnięto'}", flush=True)
    if interrupted:
        print("\nPrzerwano przez uzytkownika — checkpoint zapisany.", flush=True)
    if traj_path is not None:
        print(f"\nTrajektoria: {traj_path.resolve()}", flush=True)
    print(f"Checkpoint:  {(out / 'checkpoint.npz').resolve()}", flush=True)
    print(f"WIDOK 3D:    {html_path.resolve()}", flush=True)
    if gif_path is not None:
        print(f"Animacja:    {gif_path.resolve()}", flush=True)
    print(f"Wyniki w:    {out.resolve()}", flush=True)
    print("Kontynuuj ewolucje: python -m bone --continue --steps 5000 --no-gif", flush=True)

    if cfg.open_view and html_path.exists():
        webbrowser.open(html_path.resolve().as_uri())

    return summary


def run_simulation(
    cfg: SimConfig | None = None,
    progress_every: int = 100,
    *,
    resume: bool = False,
    resume_path: str | Path | None = None,
    on_progress: ProgressCb | None = None,
    on_live: LiveCb | None = None,
    should_stop: StopCb | None = None,
    poll_config: ConfigUpdateCb | None = None,
) -> dict:
    """
    Uruchom lub kontynuuj symulację.
    resume=True → wczytaj checkpoint i dołóż kolejne cfg.steps kroków.
    steps <= 0 → bieg nieskończony aż Ctrl+C / Stop w Studio (Ty decydujesz).
    poll_config → co krok: nowy SimConfig (hot-apply runtime z Studio).
    """
    cfg = cfg or SimConfig()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    times: list[float] = []
    colors: list[np.ndarray] = []
    tracker = MetricsTracker(cfg)

    ckpt = Path(resume_path) if resume_path else out / "checkpoint.npz"
    legacy = out / "final_state.npz"
    continuing = False
    interrupted = False

    if resume:
        src = ckpt if ckpt.exists() else legacy
        if not src.exists():
            raise FileNotFoundError(
                f"Brak checkpointu do kontynuacji ({ckpt} / {legacy}). "
                "Najpierw uruchom symulację od zera."
            )
        universe = load_checkpoint(src, cfg)
        continuing = True
        writer = TrajectoryWriter(out)
        if (out / "trajectory.npz").exists():
            try:
                frames, times, colors = load_trajectory_lists(out / "trajectory.npz")
                # trzymaj tylko ogon w RAM
                if len(frames) > 400:
                    frames, times, colors = frames[-400:], times[-400:], colors[-400:]
            except Exception:  # noqa: BLE001
                frames, times, colors = [], [], []
        print(
            f"Kontynuacja: t={universe.t:.3f} step={universe.step} N={universe.n} "
            f"+{cfg.steps if cfg.steps > 0 else '∞'} krokow (G={cfg.G})",
            flush=True,
        )
    else:
        # nowy bieg — wyczyść stare chunki
        frames_dir = out / "frames"
        if frames_dir.exists():
            for old in frames_dir.glob("chunk_*.npz"):
                old.unlink(missing_ok=True)
        writer = TrajectoryWriter(out)
        rng = np.random.default_rng(cfg.seed)
        universe = spawn_newborns(cfg, rng)
        limit_txt = str(cfg.steps) if cfg.steps > 0 else "∞ (az przerwiesz)"
        from bone.gpu import device_name, gpu_enabled

        print(
            f"Start: N={universe.n} geom={resolve_geometry(cfg.geometry)} "
            f"grid={cfg.grid_n} steps={limit_txt} dt={cfg.dt} G={cfg.G} "
            f"ineq={cfg.inequality_drive} greed={cfg.greed_bias} "
            f"gift={cfg.generosity_bias} seed={cfg.seed} "
            f"accel={'CUDA' if gpu_enabled() else 'CPU'} ({device_name()})",
            flush=True,
        )
        _record_frame(
            universe, writer, ram_frames=frames, ram_times=times, ram_colors=colors
        )
        save_scatter_projection(universe, out, tag="start")
        _flush_live(
            universe, writer, frames, times, colors, out, write_html=True
        )
        save_live_frame(
            universe.positions, universe.velocities, universe.t, universe.step, out
        )

    print(
        "Podglad LIVE + zapis klatek (traj_every). "
        "Stop w Studio albo Ctrl+C — Ty decydujesz.",
        flush=True,
    )

    tracker.observe(universe)
    from bone.neighbors import get_or_build_neighbors

    forces = compute_forces(universe, neighbors=get_or_build_neighbors(universe))
    if on_live is not None:
        on_live(universe)
    unlimited = cfg.steps <= 0
    total = cfg.steps if not unlimited else 0
    total_label = "∞" if unlimited else str(total)
    prog_every = progress_every if progress_every > 0 else 100
    if unlimited and progress_every <= 0:
        prog_every = 100
    traj_flush_every = max(1, int(cfg.traj_every) * 4)  # rzadziej checkpoint na dysk
    html_every = max(traj_flush_every * 4, 80)
    live_every = max(1, int(getattr(cfg, "live_every", 12)))
    trait_every = max(1, int(getattr(cfg, "trait_every", 4)))
    # przy Studio (on_live) nie pisz live.npz w pętli — tylko traj/chunki
    write_disk_live = on_live is None
    disk_live_every = max(1, int(getattr(cfg, "live_every", 12)))

    s = 0
    try:
        while True:
            s += 1
            if not unlimited and s > total:
                break
            if should_stop is not None and should_stop():
                interrupted = True
                print(f"Stop uzytkownika przy kroku {s} (abs={universe.step}).", flush=True)
                break

            if poll_config is not None:
                updated = poll_config()
                if updated is not None:
                    universe.config = updated
                    cfg = updated
                    tracker.cfg = updated
                    live_every = max(1, int(getattr(cfg, "live_every", 12)))
                    trait_every = max(1, int(getattr(cfg, "trait_every", 4)))

            speed = int(max(1, min(100, getattr(cfg, "sim_speed", 1))))
            base_dt = float(cfg.dt)
            if speed > 1:
                universe.config = replace(cfg, dt=base_dt * speed)
            try:
                forces, neigh = leapfrog_step(universe, forces=forces)
                if s % trait_every == 0:
                    evolve_traits(universe, neighbors=neigh)
            finally:
                universe.config = replace(universe.config, dt=base_dt)
                cfg = universe.config

            if on_live is not None and (
                s % live_every == 0 or (not unlimited and s == total)
            ):
                on_live(universe)
            if write_disk_live and (
                s % disk_live_every == 0 or (not unlimited and s == total)
            ):
                save_live_frame(
                    universe.positions,
                    universe.velocities,
                    universe.t,
                    universe.step,
                    out,
                )

            if s % cfg.traj_every == 0 or (not unlimited and s == total):
                _record_frame(
                    universe,
                    writer,
                    ram_frames=frames,
                    ram_times=times,
                    ram_colors=colors,
                )

            if s % traj_flush_every == 0 or (not unlimited and s == total):
                _flush_live(
                    universe,
                    writer,
                    frames,
                    times,
                    colors,
                    out,
                    write_html=(s % html_every == 0),
                    compressed_ckpt=False,
                )

            if s % cfg.snapshot_every == 0 or (not unlimited and s == total):
                row = tracker.observe(universe)
                if prog_every and s % prog_every == 0:
                    print(
                        f"step={s}/{total_label} (abs={universe.step}) t={universe.t:.3f} "
                        f"×{speed} "
                        f"alive={int(row['n_alive'])} "
                        f"|v|={row['mean_speed']:.4f} "
                        f"Gini={row.get('gini', 0):.3f} "
                        f"|L|={row.get('L_mag', 0):.3f} "
                        f"flat={row.get('flattening', 1):.3f} "
                        f"collapse={row.get('collapse_ratio', 1.0):.3f} "
                        f"clusters={int(row['n_clusters'])}",
                        flush=True,
                    )
                if on_progress is not None:
                    on_progress(s, total if not unlimited else s, universe, row)
                if concentration_reached(universe, row.get("collapse_ratio", 1.0)):
                    print(
                        f"Emergencja: collapse={row.get('collapse_ratio', 1):.4f} "
                        f"topW={row.get('top_wealth_frac', 0):.3f} "
                        f"topM={row.get('top_mass_frac', 0):.3f} — stop.",
                        flush=True,
                    )
                    break
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nCtrl+C przy kroku {s} — zapisuje stan...", flush=True)

    _flush_live(
        universe,
        writer,
        frames,
        times,
        colors,
        out,
        write_html=True,
        compressed_ckpt=True,
    )

    return _finalize_outputs(
        universe,
        cfg,
        out,
        frames,
        times,
        colors,
        tracker,
        continuing,
        interrupted,
        writer,
    )


def warm_start_check(cfg: SimConfig | None = None) -> Universe:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    return spawn_newborns(cfg, rng)
