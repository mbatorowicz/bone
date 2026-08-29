"""CLI headless — silnik SR."""

from __future__ import annotations

import argparse
from pathlib import Path

from bone.config.schema import AppConfig, preset_burst, preset_cluster, preset_galaxy
from bone.engine import Engine
from bone.io.checkpoint import save_checkpoint
from bone.io.trajectory import TrajectoryWriter
from bone.physics.gpu import device_label, gpu_enabled


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Bone SR — grawitacja relatywistyczna")
    p.add_argument("--preset", choices=["galaxy", "cluster", "burst"], default="galaxy")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--particles", type=int, default=None)
    p.add_argument("--geometry", type=int, default=None)
    p.add_argument("--out", default="out")
    p.add_argument("--progress-every", type=int, default=50)
    args = p.parse_args(argv)

    presets = {"galaxy": preset_galaxy, "cluster": preset_cluster, "burst": preset_burst}
    cfg = presets[args.preset]()
    flat = cfg.to_flat()
    flat["steps"] = args.steps
    flat["out_dir"] = args.out
    if args.particles:
        flat["n_particles"] = args.particles
    if args.geometry is not None:
        flat["geometry"] = args.geometry
    cfg = AppConfig.from_flat(flat)

    print(f"Bone SR preset={args.preset} N={cfg.spawn.n_particles} steps={cfg.io.steps}", flush=True)
    print(f"accel={device_label()} enabled={gpu_enabled()}", flush=True)

    eng = Engine(cfg)
    out = Path(cfg.io.out_dir)
    writer = TrajectoryWriter(out, stride=cfg.io.point_stride)

    def on_progress(step, total, u, row):
        if step % args.progress_every == 0:
            print(
                f"step={step}/{total} t={u.t:.3f} γ={row['mean_gamma']:.3f} "
                f"r½={row['collapse_ratio']:.3f} |L|={row['L_mag']:.2f} "
                f"v/c={row['v_over_c']:.3f}",
                flush=True,
            )

    eng.run(on_progress=on_progress, on_frame=writer.add)
    writer.close()
    save_checkpoint(eng.universe, out)
    print(f"OK → {out}", flush=True)


if __name__ == "__main__":
    main()
