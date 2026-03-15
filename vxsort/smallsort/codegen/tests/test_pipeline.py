"""Tests for pipelined stage processing in the bitonic super-optimizer."""

from __future__ import annotations

import pytest

from bitonic_super_optimizer import (
    BitonicSuperVectorizer,
    StageTracker,
    VectorState,
    PermutationGadget,
    InstructionSpec,
    _BATCH_THRESHOLDS,
)
from utils import primitive_type, vector_machine


# ---------------------------------------------------------------------------
# StageTracker unit tests
# ---------------------------------------------------------------------------


class TestStageTracker:
    """Unit tests for StageTracker bookkeeping."""

    def test_initial_state(self):
        t = StageTracker(stage_idx=0, stage_pairs=[(1, 2)])
        assert t.completed_jobs == 0
        assert t.total_jobs_submitted == 0
        assert not t.finalized
        assert not t.all_inputs_received
        assert not t.is_complete
        assert t.completion_fraction == 0.0

    def test_completion_fraction(self):
        t = StageTracker(stage_idx=0, stage_pairs=[])
        t.total_jobs_submitted = 100
        t.completed_jobs = 50
        assert t.completion_fraction == 0.5

    def test_is_complete_requires_all_inputs(self):
        """is_complete is False until all_inputs_received even if all jobs done."""
        t = StageTracker(stage_idx=0, stage_pairs=[])
        t.total_jobs_submitted = 10
        t.total_jobs_expected = 10
        t.completed_jobs = 10
        # Still not complete — all_inputs_received is False
        assert not t.is_complete

        t.all_inputs_received = True
        assert t.is_complete

    def test_should_process_batch_thresholds(self):
        t = StageTracker(stage_idx=0, stage_pairs=[])
        t.total_jobs_submitted = 100

        # Below 50% — no batch
        t.completed_jobs = 49
        assert not t.should_process_batch()

        # At 50% — first batch
        t.completed_jobs = 50
        assert t.should_process_batch()

        # Consume the threshold
        t.next_threshold_index = 1

        # Below 75% — no batch
        t.completed_jobs = 74
        assert not t.should_process_batch()

        # At 75% — second batch
        t.completed_jobs = 75
        assert t.should_process_batch()

    def test_thresholds_exhausted(self):
        t = StageTracker(stage_idx=0, stage_pairs=[])
        t.total_jobs_submitted = 100
        t.completed_jobs = 100
        t.next_threshold_index = len(_BATCH_THRESHOLDS)
        assert not t.should_process_batch()


# ---------------------------------------------------------------------------
# _process_batch unit test
# ---------------------------------------------------------------------------


class TestProcessBatch:
    """Test the incremental batch processor."""

    def test_accumulates_unique_outputs(self):
        t = StageTracker(stage_idx=0, stage_pairs=[])

        state_a = VectorState(top=[1, 2], bottom=[3, 4])
        state_b = VectorState(top=[2, 1], bottom=[4, 3])

        gadget_a = PermutationGadget(
            top_instructions=[InstructionSpec("test_inst", {"a": "top", "imm8": 0})],
            bottom_instructions=[],
            validated=True,
        )
        gadget_b = PermutationGadget(
            top_instructions=[InstructionSpec("test_inst", {"a": "top", "imm8": 1})],
            bottom_instructions=[],
            validated=True,
        )

        input_state = VectorState(top=[1, 3], bottom=[2, 4])
        # Simulate two results: one producing state_a, one producing state_b
        t.unprocessed_results = [
            (
                [(gadget_a, state_a)],
                input_state,
                {"parent_path": ()},
                0.0,
                0.0,
            ),
            (
                [(gadget_b, state_b)],
                input_state,
                {"parent_path": ()},
                0.0,
                0.0,
            ),
        ]

        new_outputs = BitonicSuperVectorizer._process_batch(t)

        assert len(new_outputs) == 2
        assert len(t.unique_outputs) == 2
        assert len(t.forwarded_output_tuples) == 2
        assert len(t.unprocessed_results) == 0

    def test_deduplicates_forwarded_outputs(self):
        """Second batch should not re-forward already-forwarded outputs."""
        t = StageTracker(stage_idx=0, stage_pairs=[])

        state = VectorState(top=[1, 2], bottom=[3, 4])
        gadget = PermutationGadget(
            top_instructions=[], bottom_instructions=[], validated=True
        )
        input_state = VectorState(top=[1, 3], bottom=[2, 4])

        # First batch
        t.unprocessed_results = [
            ([(gadget, state)], input_state, {"parent_path": ()}, 0.0, 0.0),
        ]
        new1 = BitonicSuperVectorizer._process_batch(t)
        assert len(new1) == 1

        # Second batch with same output
        t.unprocessed_results = [
            ([(gadget, state)], input_state, {"parent_path": ()}, 0.0, 0.0),
        ]
        new2 = BitonicSuperVectorizer._process_batch(t)
        assert len(new2) == 0  # Already forwarded

    def test_empty_results_no_crash(self):
        t = StageTracker(stage_idx=0, stage_pairs=[])
        t.unprocessed_results = [
            ([], VectorState(top=[1], bottom=[2]), {"parent_path": ()}, 0.0, 0.0),
        ]
        new = BitonicSuperVectorizer._process_batch(t)
        assert len(new) == 0


# ---------------------------------------------------------------------------
# _finalize_stage unit test
# ---------------------------------------------------------------------------


class TestFinalizeStage:
    """Test deterministic finalization."""

    def test_sorts_transitions_and_gadgets(self):
        t = StageTracker(stage_idx=0, stage_pairs=[])

        input_tuple = ((1, 3), (2, 4))
        output_a = ((1, 2), (3, 4))
        output_b = ((1, 4), (2, 3))

        # Insert in reverse order to test sorting
        t.gadgets_by_transition[((), input_tuple, output_b)] = [
            PermutationGadget(
                top_instructions=[InstructionSpec("z_inst", {"a": "top", "imm8": 1})],
                bottom_instructions=[],
                validated=True,
            ),
            PermutationGadget(
                top_instructions=[InstructionSpec("a_inst", {"a": "top", "imm8": 0})],
                bottom_instructions=[],
                validated=True,
            ),
        ]
        t.gadgets_by_transition[((), input_tuple, output_a)] = [
            PermutationGadget(
                top_instructions=[], bottom_instructions=[], validated=True
            ),
        ]

        t.unique_outputs[output_a] = VectorState(
            top=list(output_a[0]), bottom=list(output_a[1])
        )
        t.unique_outputs[output_b] = VectorState(
            top=list(output_b[0]), bottom=list(output_b[1])
        )

        BitonicSuperVectorizer._finalize_stage(t)

        assert t.finalized
        nodes = t.nodes_by_parent[()]
        assert len(nodes) == 2
        # output_a < output_b lexicographically
        assert nodes[0].output_state.as_tuple() == output_a
        assert nodes[1].output_state.as_tuple() == output_b
        # Gadgets within output_b node should be sorted
        assert nodes[1].gadgets[0].top_instructions[0].intrinsic_name == "a_inst"
        assert nodes[1].gadgets[1].top_instructions[0].intrinsic_name == "z_inst"


# ---------------------------------------------------------------------------
# Integration: pipelined execution
# ---------------------------------------------------------------------------


class TestStageTrackerDepthFields:
    """Tests for depth-stratified search fields on StageTracker."""

    def test_default_depth_fields(self):
        """New StageTracker has depth fields with correct defaults."""
        t = StageTracker(stage_idx=0, stage_pairs=[(1, 2)])
        assert not t.shallow_complete
        assert not t.depth_escalated
        assert t.all_input_states == []

    def test_all_input_states_accumulation(self):
        """all_input_states records forwarded inputs."""
        t = StageTracker(stage_idx=1, stage_pairs=[(1, 2)])
        state_a = VectorState(top=[1, 2], bottom=[3, 4])
        state_b = VectorState(top=[2, 1], bottom=[4, 3])
        t.all_input_states.extend([(state_a, ()), (state_b, ())])
        assert len(t.all_input_states) == 2


class TestPipelinedExecution:
    """Verify pipelined execution works correctly."""

    @pytest.mark.slow
    def test_avx2_i64_single_stage(self):
        """Single-stage run should work correctly."""
        sv = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX2)
        roots, complete = sv.build_solution_tree(
            depth_limit=1,
            gadget_depth=1,
            max_unique_outputs=1,
        )
        assert len(roots) > 0
        assert complete

    @pytest.mark.slow
    def test_depth2_threshold_skips_deep(self):
        """With gadget_depth=2 and high coverage, depth-2 should be skipped."""
        sv = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX2)
        roots, complete = sv.build_solution_tree(
            depth_limit=1,
            gadget_depth=2,
            max_unique_outputs=1,
            depth2_threshold=0.1,
        )
        assert len(roots) > 0
        assert complete
