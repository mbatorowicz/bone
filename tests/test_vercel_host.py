"""Limity hosta serverless i wspólny dispatcher HTTP."""

from __future__ import annotations

import os

from bone.config import Config, preset
from bone.studio.server import apply_host_limits, dispatch_get, on_serverless


def test_on_serverless_reads_vercel_env(monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    assert on_serverless() is False
    monkeypatch.setenv("VERCEL", "1")
    assert on_serverless() is True


def test_host_limits_clamp_particles_and_outdir(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    cfg = preset("galaxy")
    limited = apply_host_limits(cfg)
    assert limited.run.out_dir == "/tmp/bone-runs"
    assert limited.solver.device == "cpu"
    assert limited.spawn.n_particles <= 8_000


def test_host_limits_noop_locally(monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    cfg = Config()
    assert apply_host_limits(cfg) is cfg or apply_host_limits(cfg) == cfg


def test_serverless_tick_advances(monkeypatch, tmp_path):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.chdir(tmp_path)
    from bone.studio.server import SESSION, dispatch_get, dispatch_post, stop_and_wait

    stop_and_wait()
    started = dispatch_post(
        "/api/start",
        {
            "params": {
                "n_particles": 200,
                "backend": "exact",
                "device": "cpu",
                "steps": 6,
                "live_every": 1,
                "diagnostics_every": 1,
                "time_scale": 1,
            }
        },
    )
    assert started.status == 200
    body = __import__("json").loads(started.body)
    assert body["config"]["out_dir"] == "/tmp/bone-runs"
    assert body["config"]["n_particles"] == 200

    sizes = []
    for _ in range(8):
        view = dispatch_get("/api/view")
        assert view.status == 200
        sizes.append(len(view.body))
    assert max(sizes) > 16  # więcej niż pusty nagłówek

    status = __import__("json").loads(dispatch_get("/api/status").body)
    # po 6 krokach bieg powinien się domknąć
    assert status["running"] is False or status["diagnostics"]
    stop_and_wait()
