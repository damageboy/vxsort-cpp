"""Runtime-only instruction attempt statistics."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .bitonic_types import InstructionSpec
except ImportError:
    from bitonic_types import InstructionSpec  # type: ignore[no-redef]


_IMM_MARKERS = ("imm",)
_CONTROL_VECTOR_MARKERS = (
    "permutex2var",
    "permutexvar",
    "permutevar",
    "blendv",
)


@dataclass(frozen=True)
class InstructionAttemptStats:
    """Counters for a normalized instruction key."""

    attempts_total: int = 0
    attempts_no_valid: int = 0
    attempts_valid: int = 0
    attempts_unique: int = 0


def normalize_instruction_stats_key(spec: InstructionSpec) -> str:
    """Return a stable key for an instruction family.

    The key is the intrinsic name, optionally suffixed with ``_imm`` for
    immediate-controlled instructions or ``_v`` for actual control-vector
    instruction families.
    """
    args = spec.args or {}
    arg_names = {str(name).lower() for name in args}

    if any(any(marker in name for marker in _IMM_MARKERS) for name in arg_names):
        return f"{spec.intrinsic_name}_imm"

    intrinsic_name = spec.intrinsic_name.lower()
    if any(marker in intrinsic_name for marker in _CONTROL_VECTOR_MARKERS):
        return f"{spec.intrinsic_name}_v"

    return spec.intrinsic_name


class InstructionStatsCollector:
    """Collect instruction attempt outcomes keyed by normalized instruction."""

    def __init__(self) -> None:
        self._by_stage: dict[int, dict[str, InstructionAttemptStats]] = {}

    def record_attempt(
        self, stage_idx: int, instruction_keys: list[str], outcome: str
    ) -> None:
        """Record one attempt outcome for each participating instruction key."""
        if outcome not in {"no_valid", "valid", "unique"}:
            raise ValueError(f"Unknown instruction outcome: {outcome}")

        stage_stats = self._by_stage.setdefault(stage_idx, {})
        for key in instruction_keys:
            stats = stage_stats.get(key)
            if stats is None:
                stats = InstructionAttemptStats()
            updates = {
                "attempts_total": stats.attempts_total + 1,
                "attempts_no_valid": stats.attempts_no_valid,
                "attempts_valid": stats.attempts_valid,
                "attempts_unique": stats.attempts_unique,
            }
            if outcome == "no_valid":
                updates["attempts_no_valid"] += 1
            elif outcome == "valid":
                updates["attempts_valid"] += 1
            else:
                updates["attempts_unique"] += 1
            stage_stats[key] = InstructionAttemptStats(**updates)

    def snapshot_by_stage(self) -> dict[int, dict[str, InstructionAttemptStats]]:
        """Return an immutable snapshot of counters keyed by stage index."""
        return {
            stage_idx: dict(stage_stats)
            for stage_idx, stage_stats in self._by_stage.items()
        }

    def snapshot_aggregated(self) -> dict[str, InstructionAttemptStats]:
        """Return counters aggregated across all stages."""
        aggregated: dict[str, InstructionAttemptStats] = {}
        for stage_stats in self._by_stage.values():
            for key, stats in stage_stats.items():
                prev = aggregated.get(key, InstructionAttemptStats())
                aggregated[key] = InstructionAttemptStats(
                    attempts_total=prev.attempts_total + stats.attempts_total,
                    attempts_no_valid=prev.attempts_no_valid + stats.attempts_no_valid,
                    attempts_valid=prev.attempts_valid + stats.attempts_valid,
                    attempts_unique=prev.attempts_unique + stats.attempts_unique,
                )
        return aggregated

    def snapshot(self) -> dict[str, InstructionAttemptStats]:
        """Backward-compatible aggregated snapshot of current counters."""
        return self.snapshot_aggregated()
