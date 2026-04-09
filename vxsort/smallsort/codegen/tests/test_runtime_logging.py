from __future__ import annotations

import json
import logging

import pytest

from runtime_logging import RuntimeLoggingConfig, configure_runtime_logging, log_event


def test_parse_invalid_level_raises_value_error(tmp_path):
    config = RuntimeLoggingConfig(
        enabled=True,
        level="BANANAS",
        file_path=str(tmp_path / "runtime.log"),
    )

    with pytest.raises(ValueError):
        configure_runtime_logging(config)


def test_json_logging_writes_structured_event(tmp_path):
    log_path = tmp_path / "runtime.jsonl"
    config = RuntimeLoggingConfig(
        enabled=True,
        level="DEBUG",
        file_path=str(log_path),
        format="json",
    )
    logger, shutdown = configure_runtime_logging(config)

    try:
        log_event(
            logger,
            logging.INFO,
            "mca_complete",
            solution_index=7,
            cycles=3.5,
        )
    finally:
        shutdown()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    payload = json.loads(lines[-1])
    assert payload["event"] == "mca_complete"
    assert payload["solution_index"] == 7
    assert payload["cycles"] == pytest.approx(3.5)


def test_disabled_logging_writes_nothing(tmp_path):
    log_path = tmp_path / "runtime.log"
    config = RuntimeLoggingConfig(
        enabled=False,
        level="DEBUG",
        file_path=str(log_path),
    )
    logger, shutdown = configure_runtime_logging(config)

    try:
        log_event(logger, logging.INFO, "heartbeat", wave=1)
    finally:
        shutdown()

    assert not log_path.exists()
