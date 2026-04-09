"""Runtime logging utilities for wave synthesis/scoring observability."""

from __future__ import annotations

import atexit
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from queue import SimpleQueue
from collections.abc import Callable
from typing import Any

_VALID_FORMATS = {"text", "json"}


@dataclass(slots=True)
class RuntimeLoggingConfig:
    """Configuration for runtime logging sinks and formatting."""

    enabled: bool = False
    level: str = "INFO"
    file_path: str | None = None
    format: str = "text"
    max_bytes: int = 0
    backup_count: int = 3
    console: bool = False
    logger_name: str = "vxsort.runtime"


def parse_log_level(level: str | int) -> int:
    """Return numeric logging level from string/int input."""
    if isinstance(level, int):
        return level

    normalized = level.strip().upper()
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    if normalized not in mapping:
        msg = f"Unknown log level: {level!r}"
        raise ValueError(msg)
    return mapping[normalized]


class EventTextFormatter(logging.Formatter):
    """Text formatter that prints event + key/value fields."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        ts = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        event = getattr(record, "event_name", record.getMessage())
        fields = getattr(record, "event_fields", {})
        if fields:
            rendered_fields = " ".join(
                f"{key}={value!r}" for key, value in fields.items()
            )
            return f"{ts} [{record.levelname}] {event} | {rendered_fields}"
        return f"{ts} [{record.levelname}] {event}"


class EventJsonFormatter(logging.Formatter):
    """JSONL formatter for high-volume machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event_name", record.getMessage()),
        }
        payload.update(getattr(record, "event_fields", {}))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def _make_formatter(fmt: str) -> logging.Formatter:
    if fmt not in _VALID_FORMATS:
        msg = f"Unsupported log format: {fmt!r}. Expected one of {_VALID_FORMATS}"
        raise ValueError(msg)
    if fmt == "json":
        return EventJsonFormatter()
    return EventTextFormatter()


def configure_runtime_logging(
    config: RuntimeLoggingConfig,
) -> tuple[logging.Logger, Callable[[], None]]:
    """Configure queue-backed runtime logging and return (logger, shutdown)."""
    logger = logging.getLogger(config.logger_name)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    if not config.enabled:
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

        def _noop_shutdown() -> None:
            return

        return logger, _noop_shutdown

    level = parse_log_level(config.level)
    formatter = _make_formatter(config.format)

    target_handlers: list[logging.Handler] = []

    if config.file_path:
        if config.max_bytes and config.max_bytes > 0:
            file_handler: logging.Handler = RotatingFileHandler(
                config.file_path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(config.file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        target_handlers.append(file_handler)

    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        target_handlers.append(console_handler)

    if not target_handlers:
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

        def _noop_shutdown() -> None:
            return

        return logger, _noop_shutdown

    queue_obj: SimpleQueue[Any] = SimpleQueue()
    queue_handler = QueueHandler(queue_obj)
    queue_handler.setLevel(level)
    logger.addHandler(queue_handler)
    logger.setLevel(level)
    logger.propagate = False

    listener = QueueListener(queue_obj, *target_handlers, respect_handler_level=True)
    listener.start()

    def _shutdown() -> None:
        try:
            listener.stop()
        except Exception:
            pass
        for handler in target_handlers:
            try:
                handler.flush()
            except Exception:
                pass
            try:
                handler.close()
            except Exception:
                pass
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

    atexit.register(_shutdown)
    return logger, _shutdown


def log_event(
    logger: logging.Logger,
    level: int | str,
    event: str,
    **fields: Any,
) -> None:
    """Emit one structured runtime event."""
    level_no = parse_log_level(level) if isinstance(level, str) else level
    if not logger.isEnabledFor(level_no):
        return

    logger.log(
        level_no,
        event,
        extra={"event_name": event, "event_fields": fields},
    )
