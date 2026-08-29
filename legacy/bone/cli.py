"""CLI: bone-sim / python -m bone.cli"""

from __future__ import annotations

import argparse
from dataclasses import fields

from bone.constants import SimConfig
from bone.simulate import run_simulation
from bone.ui_schema import orbit_then_singularity_preset, stable_orbit_preset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bone — grawitacyjna symulacja cech (start w bezruchu)."
    )
    p.add_argument("--steps", type=int, default=None, help="Liczba krokow (0 = az Stop)")
    p.add_argument("--dt", type=float, default=None, help="Krok calkowania")
    p.add_argument("--seed", type=int, default=None, help="Ziarno RNG")
    p.add_argument("--grid", type=int, default=None, help="Rozmiar siatki N (cube: N^3)")
    p.add_argument(
        "--particles",
        type=int,
        default=None,
        dest="n_particles",
        help="Liczba punktow (geometrie inne niz cube)",
    )
    p.add_argument(
        "--geometry",
        type=int,
        default=None,
        help="0 cube … 8 donut, 9 sphere_band",
    )
    p.add_argument("--until-stop", action="store_true", help="Bez limitu krokow (Ctrl+C)")
    p.add_argument(
        "--stable-orbits",
        action="store_true",
        help="Preset: zrównoważony gift≈exploit (układ rozproszony)",
    )
    p.add_argument(
        "--orbit-to-singularity",
        action="store_true",
        help="Preset: kolaps społeczny (wysoki exploit/matthew, bez sztucznej studni)",
    )
    p.add_argument(
        "--inequality",
        type=float,
        default=None,
        dest="inequality_drive",
        help="Tempo zaostrzania nierówności",
    )
    p.add_argument("--inequality-delay", type=float, default=None, dest="inequality_delay")
    p.add_argument("--greed", type=float, default=None, dest="greed_bias")
    p.add_argument("--generosity", type=float, default=None, dest="generosity_bias")
    p.add_argument("--G", type=float, default=None)
    p.add_argument("--c", type=float, default=None)
    p.add_argument("--r-cut", type=float, default=None, dest="r_cut")
    p.add_argument("--orbit-speed", type=float, default=None, dest="orbital_seed_speed")
    p.add_argument("--circularize", type=float, default=None, dest="circularize_rate")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--snapshot-every", type=int, default=None, dest="snapshot_every")
    p.add_argument("--traj-every", type=int, default=None, dest="traj_every")
    p.add_argument("--progress-every", type=int, default=100, dest="progress_every")
    p.add_argument("--view", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--studio", action="store_true")
    p.add_argument("--continue", dest="resume", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.studio:
        from bone.studio import main as studio_main

        studio_main()
        return

    geometry = args.geometry
    if geometry is None:
        geometry = 8 if (args.orbit_to_singularity or args.stable_orbits) else 0

    if args.orbit_to_singularity:
        params = orbit_then_singularity_preset(geometry)
    elif args.stable_orbits:
        params = stable_orbit_preset(geometry)
    else:
        # pełne defaulty SimConfig
        from bone.ui_schema import default_param_values

        params = default_param_values()
        params["geometry"] = geometry

    params["geometry"] = geometry

    # tylko jawne flagi nadpisują (None = zostaw preset / default)
    explicit = {
        "n_particles": args.n_particles,
        "grid_n": args.grid,
        "seed": args.seed,
        "steps": args.steps,
        "dt": args.dt,
        "G": args.G,
        "c": args.c,
        "r_cut": args.r_cut,
        "orbital_seed_speed": args.orbital_seed_speed,
        "circularize_rate": args.circularize_rate,
        "inequality_drive": args.inequality_drive,
        "inequality_delay": args.inequality_delay,
        "greed_bias": args.greed_bias,
        "generosity_bias": args.generosity_bias,
        "snapshot_every": args.snapshot_every,
        "traj_every": args.traj_every,
        "out_dir": args.out,
    }
    for key, val in explicit.items():
        if val is not None:
            params[key] = val

    if args.until_stop:
        params["steps"] = 0
    if args.view:
        params["open_view"] = True
    if args.no_gif:
        params["make_gif"] = False
    elif args.orbit_to_singularity or args.stable_orbits:
        params.setdefault("make_gif", False)

    cfg_fields = {f.name for f in fields(SimConfig)}
    kwargs = {k: v for k, v in params.items() if k in cfg_fields}
    cfg = SimConfig(**kwargs)
    run_simulation(cfg, progress_every=args.progress_every, resume=args.resume)


if __name__ == "__main__":
    main()
