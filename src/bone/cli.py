"""Wiersz poleceń: bieg headless, studio i benchmark backendów."""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np

from bone.backends import cuda_available, cuda_name, make_backend
from bone.config import PRESETS, Config, preset
from bone.engine import Callbacks, Engine
from bone.io import checkpoint
from bone.io.trajectory import TrajectoryWriter
from bone.spawn import make_state


def _overrides(args: argparse.Namespace) -> dict:
    mapping = {
        "particles": "n_particles",
        "geometry": "geometry",
        "steps": "steps",
        "out": "out_dir",
        "backend": "backend",
        "device": "device",
        "grid": "grid",
        "softening": "softening",
        "gravity": "G",
        "light_speed": "c",
        "check_every": "error_check_every",
        # warunek początkowy: bez tych przełączników każde odejście od presetu
        # wymagało pisania własnego skryptu, mimo że pola w configu istnieją
        "radius": "radius",
        "mass": "total_mass",
        "rotation": "rotation",
        "temperature": "temperature",
        "thickness": "thickness",
        "flatten": "flatten",
        "virial": "virial",
        "accuracy": "accuracy",
        "dt_max": "dt_max",
        "frame_every": "trajectory_every",
        "cooling": "cooling_rate",
        "cooling_power": "cooling_density_power",
        "cooling_floor": "cooling_floor",
        "cooling_grid": "cooling_grid",
    }
    return {
        key: getattr(args, name)
        for name, key in mapping.items()
        if getattr(args, name, None) is not None
    }


def cmd_run(args: argparse.Namespace) -> int:
    cfg = preset(args.preset) if args.preset else Config()
    cfg = cfg.replace_flat(_overrides(args))

    engine = Engine(cfg)
    out = Path(cfg.run.out_dir)
    writer = TrajectoryWriter(out, stride=max(1, cfg.run.point_stride))

    print(f"Bone · {engine.state.n} cząstek · {engine.describe()}", flush=True)
    print(f"G={cfg.physics.G} c={cfg.physics.c} ε={cfg.physics.softening}", flush=True)

    started = time.perf_counter()

    def on_diagnostics(row: dict[str, float]) -> None:
        line = (
            f"krok {int(row['step']):>7}  t={row['t']:9.4f}  dt={row['dt']:.2e}  "
            f"⟨γ⟩={row['gamma_mean']:.5f}  β_max={row['beta_max']:.4f}  "
            f"E={row['E_tot']:+.6e}  dryf={row['E_drift']:+.2e}  wirial={row['virial']:.3f}"
        )
        if "force_err_rms" in row:
            line += f"  błąd_siły={row['force_err_rms']:.2%}"
        if row.get("E_cooled", 0.0) > 0.0:
            line += f"  wypromieniowane={row['E_cooled'] / abs(row['E_tot']):.1%}"
        print(line, flush=True)
        hint = engine.accuracy_hint()
        if hint and hint != on_diagnostics.last_hint:
            on_diagnostics.last_hint = hint
            print(f"  ↳ {hint}", flush=True)

    on_diagnostics.last_hint = ""

    try:
        engine.run(
            Callbacks(
                on_diagnostics=on_diagnostics,
                on_frame=None if args.no_frames else (lambda st, c: writer.add(st, c)),
            )
        )
    except KeyboardInterrupt:
        print("\nPrzerwano.", flush=True)
    finally:
        writer.close()
        checkpoint.save(engine.state, engine.cfg, out)
        engine.close()

    wall = time.perf_counter() - started
    steps = engine.state.step
    rate = steps / wall if wall > 0 else 0.0
    print(f"\n{steps} kroków w {wall:.2f} s ({rate:.1f} kroków/s) → {out}", flush=True)
    return 0


def cmd_studio(args: argparse.Namespace) -> int:
    from bone.studio.server import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    print(f"CUDA: {cuda_name() if cuda_available() else 'niedostępna'}\n", flush=True)
    header = f"{'N':>9}  {'backend':<26}  {'ms/krok':>9}  {'kroków/s':>9}  {'błąd RMS':>9}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    from bone.diagnostics import measure_force_error

    rng = np.random.default_rng(0)
    for n in args.sizes:
        for kind in args.backends:
            cfg = Config().replace_flat(
                {"n_particles": n, "backend": kind, "device": args.device, "grid": args.grid}
            )
            try:
                state = make_state(cfg)
                backend = make_backend(cfg, n)
                phys = cfg.physics
                field = backend.compute(state.positions, state.masses, phys.G, phys.softening)
                start = time.perf_counter()
                for _ in range(args.repeats):
                    field = backend.compute(state.positions, state.masses, phys.G, phys.softening)
                ms = (time.perf_counter() - start) / args.repeats * 1e3

                err = "—"
                if backend.approximate and n <= args.error_max:
                    state.forces = field.force
                    # wzorzec musi mieć ten sam softening, którym liczył backend
                    eps = backend.effective_softening(phys.softening)
                    measured = measure_force_error(state, field.force, phys.G, eps, 256, rng)
                    err = f"{measured['force_err_rms']:.2%}"
                print(
                    f"{n:>9}  {backend.describe():<26}  {ms:>9.2f}  {1000 / ms:>9.1f}  {err:>9}",
                    flush=True,
                )
                backend.close()
            except Exception as exc:
                print(f"{n:>9}  {kind:<26}  {type(exc).__name__}: {exc}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bone", description="Grawitacja N-ciał z kinematyką szczególnej teorii względności"
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="bieg headless")
    run.add_argument("--preset", choices=sorted(PRESETS))
    run.add_argument("--particles", type=int)
    run.add_argument("--geometry")
    run.add_argument("--steps", type=int, default=500)
    run.add_argument("--out", default="runs/latest")
    run.add_argument("--backend", choices=["auto", "exact", "mesh"])
    run.add_argument("--device", choices=["auto", "cpu", "cuda"])
    run.add_argument("--grid", type=int)
    run.add_argument("--softening", type=float)
    run.add_argument("--gravity", type=float, help="stała G")
    run.add_argument("--light-speed", type=float, dest="light_speed", help="prędkość światła c")
    run.add_argument("--check-every", type=int, dest="check_every", help="pomiar błędu co N kroków")
    run.add_argument("--radius", type=float, help="promień startowy")
    run.add_argument("--mass", type=float, help="masa całego układu")
    run.add_argument("--rotation", type=float, help="ułamek prędkości okrężnej")
    run.add_argument("--temperature", type=float, help="dyspersja prędkości jako ułamek c")
    run.add_argument("--thickness", type=float, help="grubość przekroju (dysk, torus, włókno)")
    run.add_argument("--flatten", type=float, help="spłaszczenie osi z (1 = bez zmiany)")
    run.add_argument("--virial", type=float,
                     help="docelowe 2K/|U| na starcie; 0 = użyj --temperature")
    run.add_argument("--accuracy", type=float, help="dokładność kroku η")
    run.add_argument("--dt-max", type=float, dest="dt_max", help="górny limit kroku")
    run.add_argument("--frame-every", type=int, dest="frame_every", help="zapis klatki co N kroków")
    run.add_argument("--cooling", type=float, help="tempo chłodzenia λ (0 = brak dyssypacji)")
    run.add_argument("--cooling-power", type=float, dest="cooling_power",
                     help="wykładnik zależności chłodzenia od gęstości")
    run.add_argument("--cooling-floor", type=float, dest="cooling_floor",
                     help="podłoga dyspersji jako ułamek c")
    run.add_argument("--cooling-grid", type=int, dest="cooling_grid",
                     help="bok siatki mierzącej lokalny przepływ")
    run.add_argument("--no-frames", action="store_true", help="nie zapisuj trajektorii")
    run.set_defaults(func=cmd_run)

    studio = sub.add_parser("studio", help="serwer z podglądem 3D")
    studio.add_argument("--host", default="127.0.0.1")
    studio.add_argument("--port", type=int, default=8765)
    studio.add_argument("--no-browser", action="store_true")
    studio.set_defaults(func=cmd_studio)

    bench = sub.add_parser("bench", help="porównanie backendów")
    bench.add_argument("--sizes", type=int, nargs="+", default=[2000, 10000, 50000])
    bench.add_argument("--backends", nargs="+", default=["exact", "mesh"])
    bench.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    bench.add_argument("--grid", type=int, default=64)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--error-max", type=int, default=200_000, dest="error_max")
    bench.set_defaults(func=cmd_bench)

    return parser


def _use_utf8() -> None:
    """Konsola Windows domyślnie używa strony kodowej, która nie zna „γ" ani „³".

    Bez tego opis backendu wywala UnicodeEncodeError, a polskie znaki wychodzą
    jako krzaki. `errors="replace"` gwarantuje, że nawet egzotyczny terminal
    czegoś nie wysypie.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _use_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        args = parser.parse_args(["studio", *(argv or [])])
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
