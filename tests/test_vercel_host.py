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


def test_dispatch_schema_and_static():
    schema = dispatch_get("/api/schema")
    assert schema.status == 200
    assert b"groups" in schema.body or b"fields" in schema.body or b"presets" in schema.body
    index = dispatch_get("/")
    assert index.status == 200
    assert b"html" in index.body.lower()
