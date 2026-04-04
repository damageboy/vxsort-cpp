"""Focused tests for bitonic_compiler CLI plumbing."""

from __future__ import annotations

import sys
import types


def test_main_passes_runtime_ui_into_wave_config(monkeypatch):
    import bitonic_compiler

    captured: dict[str, object] = {}

    class FakeWaveConfig:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class FakeWaveEngine:
        def __init__(self, config) -> None:
            captured["engine_config"] = config

        def run(self) -> dict:
            return {"wave_count": 0, "interrupted": False}

        def export_to_solution_nodes(self) -> list:
            return []

    monkeypatch.setitem(
        sys.modules,
        "wave_engine",
        types.SimpleNamespace(WaveConfig=FakeWaveConfig, WaveEngine=FakeWaveEngine),
    )
    monkeypatch.setattr(
        bitonic_compiler,
        "export_solutions_to_json",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bitonic_compiler.py",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--runtime-ui",
            "textual",
        ],
    )

    bitonic_compiler.main()

    assert captured["runtime_ui"] == "textual"
