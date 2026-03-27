"""Tests for wave checkpoint save/load (per-stage zstd files + master.json)."""

import json
import os
import tempfile

import pytest
import zstandard as zstd
from bitonic_types import InstructionSpec, PermutationGadget, VectorState
from transition_table import TransitionTable
from wave_checkpoint import WaveCheckpoint, WaveMasterConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gadget(
    top_name: str = "_mm256_permute_ps",
    top_args: dict | None = None,
    bot_name: str = "_mm256_blend_ps",
    bot_args: dict | None = None,
) -> PermutationGadget:
    top_args = top_args or {"a": "top", "imm8": 0xB1}
    bot_args = bot_args or {"a": "top", "b": "bottom", "imm8": 0xAA}
    return PermutationGadget(
        top_instructions=[InstructionSpec(top_name, top_args)],
        bottom_instructions=[InstructionSpec(bot_name, bot_args)],
        validated=True,
    )


def _make_multi_instruction_gadget() -> PermutationGadget:
    """Gadget with multiple instructions per side."""
    return PermutationGadget(
        top_instructions=[
            InstructionSpec("_mm256_permute_ps", {"a": "top", "imm8": 0xB1}),
            InstructionSpec(
                "_mm256_blend_ps", {"a": "prev", "b": "bottom", "imm8": 0x55}
            ),
        ],
        bottom_instructions=[
            InstructionSpec("_mm256_shuffle_epi32", {"a": "bottom", "imm8": 0x1B}),
        ],
        validated=True,
    )


def _vs(top: list[int], bottom: list[int]) -> VectorState:
    return VectorState(top=top, bottom=bottom)


def _populated_tt(num_stages: int = 2) -> TransitionTable:
    """Build a TransitionTable with known data for round-trip testing."""
    tt = TransitionTable(num_stages)

    # Stage 0: one transition with one gadget
    inp0 = _vs([1, 3, 2, 4], [5, 7, 6, 8])
    out0 = _vs([1, 2, 3, 4], [5, 6, 7, 8])
    g0 = _make_gadget()
    tt.add_transition(0, inp0, out0, g0)
    tt.record_attempt(0, 5)
    tt.record_attempted_pair(0, inp0.as_tuple(), 0)
    tt.record_attempted_pair(0, inp0.as_tuple(), 1)
    tt.stages[0].forwarded_outputs.add(out0.as_tuple())
    tt.stages[0].consecutive_zero_budgets = 2

    # Stage 1: two transitions, one with two gadgets
    inp1 = out0  # chain from stage 0
    out1a = _vs([1, 2, 3, 4], [8, 7, 6, 5])
    out1b = _vs([4, 3, 2, 1], [5, 6, 7, 8])
    g1a = _make_gadget(
        top_name="_mm256_shuffle_epi32", top_args={"a": "top", "imm8": 0x1B}
    )
    g1b = _make_multi_instruction_gadget()
    g1c = _make_gadget(
        bot_name="_mm256_shuffle_epi32", bot_args={"a": "bottom", "imm8": 0x4E}
    )
    tt.add_transition(1, inp1, out1a, g1a)
    tt.add_transition(1, inp1, out1a, g1b)  # second gadget same transition
    tt.add_transition(1, inp1, out1b, g1c)
    tt.record_attempt(1, 10)
    tt.record_attempted_pair(1, inp1.as_tuple(), 3)

    return tt


# ---------------------------------------------------------------------------
# WaveMasterConfig round-trip
# ---------------------------------------------------------------------------


class TestWaveMasterConfig:
    """Master config serialization/deserialization."""

    def test_roundtrip_basic(self):
        config = WaveMasterConfig(
            num_vecs=2,
            vm="AVX2",
            prim_type="i64",
            gadget_depth=1,
            natural_order=False,
            retroactive_input=False,
            num_stages=3,
            wave_count=0,
            best_scores={},
            stage_stats={},
            exhausted_stages=[],
            target_cpus=["znver3"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_master(config)
            loaded = ckpt.load_master()
            assert loaded == config

    def test_roundtrip_with_scores(self):
        config = WaveMasterConfig(
            num_vecs=2,
            vm="AVX2",
            prim_type="i32",
            gadget_depth=2,
            natural_order=True,
            retroactive_input=True,
            num_stages=4,
            wave_count=7,
            best_scores={
                "znver3": {"throughput": 3.5, "cycles": 42.0, "path_index": 2},
                "skylake": {"throughput": 2.8, "cycles": 35.0, "path_index": 0},
            },
            stage_stats={
                0: {"attempts": 100, "distinct_outputs": 5, "success_rate": 0.05},
                1: {"attempts": 200, "distinct_outputs": 3, "success_rate": 0.015},
            },
            exhausted_stages=[0, 2],
            target_cpus=["znver3", "skylake"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_master(config)
            loaded = ckpt.load_master()
            assert loaded == config

    def test_stage_stats_int_keys_roundtrip(self):
        """JSON serializes dict keys as strings; verify int keys survive."""
        config = WaveMasterConfig(
            num_vecs=2,
            vm="AVX2",
            prim_type="i64",
            gadget_depth=1,
            natural_order=False,
            retroactive_input=False,
            num_stages=2,
            wave_count=1,
            best_scores={},
            stage_stats={
                0: {"attempts": 10, "distinct_outputs": 2, "success_rate": 0.2}
            },
            exhausted_stages=[],
            target_cpus=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_master(config)
            loaded = ckpt.load_master()
            assert 0 in loaded.stage_stats
            assert isinstance(list(loaded.stage_stats.keys())[0], int)


# ---------------------------------------------------------------------------
# Stage save/load round-trip
# ---------------------------------------------------------------------------


class TestStageSaveLoad:
    """Per-stage save/load through TransitionTable."""

    def test_single_stage_roundtrip(self):
        tt = _populated_tt(num_stages=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_stage(0, tt)

            # Load into a fresh table
            tt2 = TransitionTable(2)
            ckpt.load_stage(0, tt2)

            sd_orig = tt.stages[0]
            sd_loaded = tt2.stages[0]

            # Transitions match
            assert set(sd_orig.transitions.keys()) == set(sd_loaded.transitions.keys())
            for key in sd_orig.transitions:
                assert len(sd_orig.transitions[key]) == len(sd_loaded.transitions[key])
                for g_orig, g_loaded in zip(
                    sd_orig.transitions[key], sd_loaded.transitions[key]
                ):
                    assert len(g_orig.top_instructions) == len(
                        g_loaded.top_instructions
                    )
                    assert len(g_orig.bottom_instructions) == len(
                        g_loaded.bottom_instructions
                    )
                    for i_orig, i_loaded in zip(
                        g_orig.top_instructions, g_loaded.top_instructions
                    ):
                        assert i_orig.intrinsic_name == i_loaded.intrinsic_name
                        assert i_orig.args == i_loaded.args
                    for i_orig, i_loaded in zip(
                        g_orig.bottom_instructions, g_loaded.bottom_instructions
                    ):
                        assert i_orig.intrinsic_name == i_loaded.intrinsic_name
                        assert i_orig.args == i_loaded.args

            # Bookkeeping
            assert sd_loaded.attempts == sd_orig.attempts
            assert sd_loaded.attempted_pairs == sd_orig.attempted_pairs
            assert sd_loaded.forwarded_outputs == sd_orig.forwarded_outputs
            assert (
                sd_loaded.consecutive_zero_budgets == sd_orig.consecutive_zero_budgets
            )

    def test_multi_stage_roundtrip(self):
        tt = _populated_tt(num_stages=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_stage(0, tt)
            ckpt.save_stage(1, tt)

            tt2 = TransitionTable(2)
            ckpt.load_stage(0, tt2)
            ckpt.load_stage(1, tt2)

            for stage_idx in range(2):
                sd_orig = tt.stages[stage_idx]
                sd_loaded = tt2.stages[stage_idx]
                assert set(sd_orig.transitions.keys()) == set(
                    sd_loaded.transitions.keys()
                )
                assert sd_loaded.attempts == sd_orig.attempts
                assert sd_loaded.attempted_pairs == sd_orig.attempted_pairs

    def test_multi_instruction_gadget_roundtrip(self):
        """Gadgets with multiple instructions per side survive serialization."""
        tt = _populated_tt(num_stages=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_stage(1, tt)

            tt2 = TransitionTable(2)
            ckpt.load_stage(1, tt2)

            # Stage 1 has a transition with a multi-instruction gadget
            sd = tt2.stages[1]
            total_gadgets = sum(len(gs) for gs in sd.transitions.values())
            assert total_gadgets == 3  # g1a, g1b, g1c

    def test_empty_stage_roundtrip(self):
        """An empty stage round-trips without error."""
        tt = TransitionTable(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_stage(0, tt)

            tt2 = TransitionTable(1)
            ckpt.load_stage(0, tt2)

            assert tt2.stages[0].transitions == {}
            assert tt2.stages[0].attempts == 0
            assert tt2.stages[0].attempted_pairs == set()

    def test_stage_file_is_zstd_compressed(self):
        """Stage files on disk are zstd-compressed."""
        tt = _populated_tt(num_stages=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_stage(0, tt)

            stage_path = os.path.join(tmpdir, "stage_00.json.zst")
            assert os.path.exists(stage_path)

            # Verify it's valid zstd
            with open(stage_path, "rb") as f:
                raw = f.read()
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(raw)
            data = json.loads(decompressed)
            assert "transitions" in data


# ---------------------------------------------------------------------------
# Scored paths save/load
# ---------------------------------------------------------------------------


class TestScoredPaths:
    """Scored paths round-trip."""

    def test_roundtrip(self):
        # Scored paths are plain JSON-serializable dicts (no tuples).
        scored = [
            {
                "path_index": 0,
                "path": [[0, [[1, 2], [3, 4]], [[1, 3], [2, 4]]]],
                "scores": {"znver3": {"throughput": 3.5, "cycles": 42.0}},
            },
            {
                "path_index": 1,
                "path": [[0, [[1, 2], [3, 4]], [[2, 1], [4, 3]]]],
                "scores": {"znver3": {"throughput": 4.0, "cycles": 50.0}},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_scored_paths(scored)
            loaded = ckpt.load_scored_paths()
            assert loaded == scored

    def test_empty_scored_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_scored_paths([])
            loaded = ckpt.load_scored_paths()
            assert loaded == []

    def test_scored_paths_file_is_zstd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_scored_paths([{"x": 1}])
            path = os.path.join(tmpdir, "scored_paths.json.zst")
            assert os.path.exists(path)
            with open(path, "rb") as f:
                raw = f.read()
            dctx = zstd.ZstdDecompressor()
            data = json.loads(dctx.decompress(raw))
            assert data == [{"x": 1}]

    def test_load_missing_returns_empty(self):
        """load_scored_paths returns [] when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            loaded = ckpt.load_scored_paths()
            assert loaded == []


# ---------------------------------------------------------------------------
# Atomic write safety
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """Atomic write behavior."""

    def test_atomic_write_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_master(
                WaveMasterConfig(
                    num_vecs=2,
                    vm="AVX2",
                    prim_type="i64",
                    gadget_depth=1,
                    natural_order=False,
                    retroactive_input=False,
                    num_stages=1,
                    wave_count=0,
                    best_scores={},
                    stage_stats={},
                    exhausted_stages=[],
                    target_cpus=[],
                )
            )
            master_path = os.path.join(tmpdir, "master.json")
            assert os.path.exists(master_path)

    def test_no_temp_files_left_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_master(
                WaveMasterConfig(
                    num_vecs=2,
                    vm="AVX2",
                    prim_type="i64",
                    gadget_depth=1,
                    natural_order=False,
                    retroactive_input=False,
                    num_stages=1,
                    wave_count=0,
                    best_scores={},
                    stage_stats={},
                    exhausted_stages=[],
                    target_cpus=[],
                )
            )
            files = os.listdir(tmpdir)
            tmp_files = [f for f in files if f.endswith(".tmp")]
            assert tmp_files == []

    def test_overwrite_existing_master(self):
        """save_master overwrites previous master.json atomically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            config1 = WaveMasterConfig(
                num_vecs=2,
                vm="AVX2",
                prim_type="i64",
                gadget_depth=1,
                natural_order=False,
                retroactive_input=False,
                num_stages=1,
                wave_count=0,
                best_scores={},
                stage_stats={},
                exhausted_stages=[],
                target_cpus=[],
            )
            config2 = WaveMasterConfig(
                num_vecs=2,
                vm="AVX2",
                prim_type="i64",
                gadget_depth=1,
                natural_order=False,
                retroactive_input=False,
                num_stages=1,
                wave_count=5,
                best_scores={},
                stage_stats={},
                exhausted_stages=[],
                target_cpus=[],
            )
            ckpt.save_master(config1)
            ckpt.save_master(config2)
            loaded = ckpt.load_master()
            assert loaded.wave_count == 5


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------


class TestCheckpointDirectory:
    """WaveCheckpoint directory management."""

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "nested", "checkpoints")
            WaveCheckpoint(subdir)
            assert os.path.isdir(subdir)

    def test_exists_false_when_no_master(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            assert not ckpt.exists()

    def test_exists_true_after_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_master(
                WaveMasterConfig(
                    num_vecs=2,
                    vm="AVX2",
                    prim_type="i64",
                    gadget_depth=1,
                    natural_order=False,
                    retroactive_input=False,
                    num_stages=1,
                    wave_count=0,
                    best_scores={},
                    stage_stats={},
                    exhausted_stages=[],
                    target_cpus=[],
                )
            )
            assert ckpt.exists()

    def test_load_master_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            with pytest.raises(FileNotFoundError):
                ckpt.load_master()

    def test_load_stage_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            tt = TransitionTable(1)
            with pytest.raises(FileNotFoundError):
                ckpt.load_stage(0, tt)

    def test_load_scored_paths_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            assert ckpt.load_scored_paths() == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case handling."""

    def test_gadget_with_no_instructions(self):
        """Identity gadget (empty instruction lists) round-trips."""
        tt = TransitionTable(1)
        inp = _vs([1, 2], [3, 4])
        out = _vs([1, 2], [3, 4])
        gadget = PermutationGadget(
            top_instructions=[], bottom_instructions=[], validated=True
        )
        tt.add_transition(0, inp, out, gadget)
        tt.record_attempt(0, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_stage(0, tt)
            tt2 = TransitionTable(1)
            ckpt.load_stage(0, tt2)

            sd = tt2.stages[0]
            assert len(sd.transitions) == 1
            key = list(sd.transitions.keys())[0]
            assert len(sd.transitions[key]) == 1
            g = sd.transitions[key][0]
            assert g.top_instructions == []
            assert g.bottom_instructions == []

    def test_large_imm_values(self):
        """Arguments with large integer values survive JSON serialization."""
        tt = TransitionTable(1)
        inp = _vs([1, 2, 3, 4], [5, 6, 7, 8])
        out = _vs([8, 7, 6, 5], [4, 3, 2, 1])
        gadget = PermutationGadget(
            top_instructions=[
                InstructionSpec(
                    "_mm256_permutexvar_epi32", {"idx": 0xFFFFFFFF, "a": "top"}
                ),
            ],
            bottom_instructions=[],
            validated=True,
        )
        tt.add_transition(0, inp, out, gadget)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_stage(0, tt)
            tt2 = TransitionTable(1)
            ckpt.load_stage(0, tt2)

            key = list(tt2.stages[0].transitions.keys())[0]
            loaded_gadget = tt2.stages[0].transitions[key][0]
            assert loaded_gadget.top_instructions[0].args["idx"] == 0xFFFFFFFF

    def test_attempted_pairs_with_multiple_inputs(self):
        """Attempted pairs with multiple different input tuples round-trip."""
        tt = TransitionTable(1)
        inp_a = _vs([1, 2], [3, 4])
        inp_b = _vs([4, 3], [2, 1])
        tt.record_attempted_pair(0, inp_a.as_tuple(), 0)
        tt.record_attempted_pair(0, inp_a.as_tuple(), 5)
        tt.record_attempted_pair(0, inp_b.as_tuple(), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = WaveCheckpoint(tmpdir)
            ckpt.save_stage(0, tt)
            tt2 = TransitionTable(1)
            ckpt.load_stage(0, tt2)
            assert tt2.stages[0].attempted_pairs == tt.stages[0].attempted_pairs
