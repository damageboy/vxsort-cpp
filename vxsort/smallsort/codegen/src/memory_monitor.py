"""Memory usage monitoring for Rich progress display.

Collects RSS (Resident Set Size) of the main process and its worker
children, formatting the result as a single status line suitable for
rendering below progress bars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import psutil


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Point-in-time RSS snapshot of the main process and its workers."""

    main_rss_bytes: int
    worker_rss_bytes: int
    worker_count: int

    @property
    def total_rss_bytes(self) -> int:
        return self.main_rss_bytes + self.worker_rss_bytes

    @property
    def worker_avg_rss_bytes(self) -> int:
        if self.worker_count == 0:
            return 0
        return self.worker_rss_bytes // self.worker_count

    def format(self) -> str:
        main = _fmt_bytes(self.main_rss_bytes)
        total = _fmt_bytes(self.total_rss_bytes)

        if self.worker_count == 0:
            return f"Memory: Main {main} | Workers: none | Total {total}"

        workers = _fmt_bytes(self.worker_rss_bytes)
        avg = _fmt_bytes(self.worker_avg_rss_bytes)
        return (
            f"Memory: Main {main} | "
            f"Workers {workers} ({self.worker_count} x ~{avg}) | "
            f"Total {total}"
        )


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable MB or GB."""
    mb = n / (1024 * 1024)
    if mb < 1024:
        return f"{mb:.0f} MB"
    gb = mb / 1024
    return f"{gb:.1f} GB"


def collect_memory_snapshot() -> MemorySnapshot | None:
    """Collect RSS for the current process and its direct children.

    Returns None if the process tree cannot be read (e.g. process exited).
    Individual children that have exited between enumeration and measurement
    are silently skipped.
    """
    try:
        proc = psutil.Process(os.getpid())
        main_rss = proc.memory_info().rss

        worker_rss = 0
        worker_count = 0
        for child in proc.children(recursive=False):
            try:
                worker_rss += child.memory_info().rss
                worker_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return MemorySnapshot(
            main_rss_bytes=main_rss,
            worker_rss_bytes=worker_rss,
            worker_count=worker_count,
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
