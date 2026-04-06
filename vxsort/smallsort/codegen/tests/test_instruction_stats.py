"""Tests for instruction-level attempt statistics."""

from bitonic_types import InstructionSpec
from instruction_stats import (
    InstructionAttemptStats,
    InstructionStatsCollector,
    normalize_instruction_stats_key,
)


def test_normalize_instruction_stats_key_uses_imm_suffix():
    spec = InstructionSpec(
        "_mm256_blend_epi32",
        {"a": "top", "b": "bottom", "imm8": 3},
    )

    assert normalize_instruction_stats_key(spec) == "_mm256_blend_epi32/imm"


def test_normalize_instruction_stats_key_uses_v_suffix():
    spec = InstructionSpec(
        "_mm256_permutevar8x32_epi32",
        {"a": "top", "idx": "control"},
    )

    assert normalize_instruction_stats_key(spec) == "_mm256_permutevar8x32_epi32/v"


def test_normalize_instruction_stats_key_leaves_plain_intrinsic_plain():
    spec = InstructionSpec(
        "_mm512_unpacklo_epi32",
        {},
    )

    assert normalize_instruction_stats_key(spec) == "_mm512_unpacklo_epi32"


def test_normalize_instruction_stats_key_does_not_treat_src_mask_as_v():
    spec = InstructionSpec(
        "_mm512_mask_unpacklo_epi32",
        {"src": "fallback", "mask": "control"},
    )

    assert normalize_instruction_stats_key(spec) == "_mm512_mask_unpacklo_epi32"


def test_normalize_instruction_stats_key_marks_imm16_variants_as_imm():
    spec = InstructionSpec(
        "_mm256_shuffle_epi16",
        {"a": "top", "imm16": 0xAA},
    )

    assert normalize_instruction_stats_key(spec) == "_mm256_shuffle_epi16/imm"


def test_collector_tracks_no_valid_valid_and_unique_counts():
    collector = InstructionStatsCollector()

    collector.record_attempt(0, ["foo/imm"], outcome="no_valid")
    collector.record_attempt(0, ["foo/imm"], outcome="valid")
    collector.record_attempt(0, ["foo/imm"], outcome="unique")

    stats = collector.snapshot()["foo/imm"]

    assert stats == InstructionAttemptStats(
        attempts_total=3,
        attempts_no_valid=1,
        attempts_valid=1,
        attempts_unique=1,
    )


def test_collector_counts_repeated_instruction_occurrences_twice():
    collector = InstructionStatsCollector()

    collector.record_attempt(0, ["foo/imm", "foo/imm"], outcome="valid")

    stats = collector.snapshot()["foo/imm"]

    assert stats == InstructionAttemptStats(
        attempts_total=2,
        attempts_no_valid=0,
        attempts_valid=2,
        attempts_unique=0,
    )


def test_select_stage_rows_uses_selected_stage_only():
    from textual_progress_ui import select_stage_rows

    collector = InstructionStatsCollector()
    collector.record_attempt(0, ["a"], outcome="unique")
    collector.record_attempt(1, ["b"], outcome="valid")

    rows = select_stage_rows(collector.snapshot_by_stage(), selected_stage=1, top_k=10)

    assert [row.key for row in rows] == ["b"]


def test_collector_keeps_per_stage_counters_and_aggregates_for_display():
    collector = InstructionStatsCollector()

    collector.record_attempt(1, ["foo/imm"], outcome="valid")
    collector.record_attempt(2, ["foo/imm"], outcome="unique")
    collector.record_attempt(2, ["bar/v"], outcome="no_valid")

    by_stage = collector.snapshot_by_stage()
    assert by_stage[1]["foo/imm"] == InstructionAttemptStats(
        attempts_total=1,
        attempts_no_valid=0,
        attempts_valid=1,
        attempts_unique=0,
    )
    assert by_stage[2]["foo/imm"] == InstructionAttemptStats(
        attempts_total=1,
        attempts_no_valid=0,
        attempts_valid=0,
        attempts_unique=1,
    )
    assert by_stage[2]["bar/v"] == InstructionAttemptStats(
        attempts_total=1,
        attempts_no_valid=1,
        attempts_valid=0,
        attempts_unique=0,
    )

    aggregated = collector.snapshot_aggregated()
    assert aggregated["foo/imm"] == InstructionAttemptStats(
        attempts_total=2,
        attempts_no_valid=0,
        attempts_valid=1,
        attempts_unique=1,
    )
    assert aggregated["bar/v"] == InstructionAttemptStats(
        attempts_total=1,
        attempts_no_valid=1,
        attempts_valid=0,
        attempts_unique=0,
    )


def test_collector_can_prepopulate_stage_with_zero_counters():
    collector = InstructionStatsCollector()

    collector.prepopulate_stage(3, ["alpha/imm", "beta/v"])

    by_stage = collector.snapshot_by_stage()
    assert by_stage[3]["alpha/imm"] == InstructionAttemptStats()
    assert by_stage[3]["beta/v"] == InstructionAttemptStats()
