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

    assert normalize_instruction_stats_key(spec) == "_mm256_blend_epi32_imm"


def test_normalize_instruction_stats_key_uses_v_suffix():
    spec = InstructionSpec(
        "_mm256_permutevar8x32_epi32",
        {"a": "top", "idx": "control"},
    )

    assert normalize_instruction_stats_key(spec) == "_mm256_permutevar8x32_epi32_v"


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

    assert normalize_instruction_stats_key(spec) == "_mm256_shuffle_epi16_imm"


def test_collector_tracks_no_valid_valid_and_unique_counts():
    collector = InstructionStatsCollector()

    collector.record_attempt(["foo_imm"], outcome="no_valid")
    collector.record_attempt(["foo_imm"], outcome="valid")
    collector.record_attempt(["foo_imm"], outcome="unique")

    stats = collector.snapshot()["foo_imm"]

    assert stats == InstructionAttemptStats(
        attempts_total=3,
        attempts_no_valid=1,
        attempts_valid=1,
        attempts_unique=1,
    )


def test_collector_counts_repeated_instruction_occurrences_twice():
    collector = InstructionStatsCollector()

    collector.record_attempt(["foo_imm", "foo_imm"], outcome="valid")

    stats = collector.snapshot()["foo_imm"]

    assert stats == InstructionAttemptStats(
        attempts_total=2,
        attempts_no_valid=0,
        attempts_valid=2,
        attempts_unique=0,
    )
