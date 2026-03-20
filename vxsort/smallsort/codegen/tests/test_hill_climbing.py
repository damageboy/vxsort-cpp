"""Tests for hill_climb_path: first-improvement hill climbing over gadget assignments."""

import pytest
from bitonic_types import VectorState, PermutationGadget, InstructionSpec
from transition_table import TransitionTable
from wave_engine import hill_climb_path


def _make_state(top, bottom):
    return VectorState(top=list(top), bottom=list(bottom))


def _make_gadget(name="instr", n=1):
    top = [InstructionSpec(f"{name}_{i}", {"arg": i}) for i in range(n)]
    return PermutationGadget(
        top_instructions=top, bottom_instructions=[], validated=True
    )


class TestHillClimbing:
    def test_hill_climb_selects_better_gadget(self):
        """Hill climbing should swap in a gadget that scores better."""
        tt = TransitionTable(num_stages=2)
        s0_in = _make_state([0, 1, 2, 3], [4, 5, 6, 7])
        s0_out = _make_state([1, 0, 3, 2], [5, 4, 7, 6])
        s1_out = _make_state([0, 1, 2, 3], [4, 5, 6, 7])

        gadget_slow = _make_gadget("slow", 3)
        gadget_fast = _make_gadget("fast", 1)

        tt.add_transition(0, s0_in, s0_out, gadget_slow)
        tt.add_transition(0, s0_in, s0_out, gadget_fast)
        tt.add_transition(1, s0_out, s1_out, _make_gadget("only", 1))

        path = [
            (0, s0_in.as_tuple(), s0_out.as_tuple()),
            (1, s0_out.as_tuple(), s1_out.as_tuple()),
        ]

        # Mock scorer: returns lower score for "fast" gadget
        def mock_scorer(_path_spec, gadget_assignments):
            if any(
                g.top_instructions
                and g.top_instructions[0].intrinsic_name.startswith("fast")
                for g in gadget_assignments
            ):
                return 5.0
            return 10.0

        assignments, score = hill_climb_path(path, tt, mock_scorer, max_passes=3)
        assert score == pytest.approx(5.0)
        assert assignments[0].top_instructions[0].intrinsic_name.startswith("fast")

    def test_hill_climb_no_improvement_stays_at_initial(self):
        """When all alternatives are worse, keep initial assignment."""
        tt = TransitionTable(num_stages=1)
        s0_in = _make_state([0, 1], [2, 3])
        s0_out = _make_state([1, 0], [3, 2])

        gadget_a = _make_gadget("a", 1)
        gadget_b = _make_gadget("b", 2)

        tt.add_transition(0, s0_in, s0_out, gadget_a)
        tt.add_transition(0, s0_in, s0_out, gadget_b)

        path = [(0, s0_in.as_tuple(), s0_out.as_tuple())]

        # Scorer that always returns the same score
        def constant_scorer(_path_spec, _gadget_assignments):
            return 7.0

        assignments, score = hill_climb_path(path, tt, constant_scorer)
        assert score == pytest.approx(7.0)
        # Should pick the one with fewest instructions (gadget_a has 1)
        assert assignments[0].instruction_count() == 1

    def test_hill_climb_max_passes_cap(self):
        """Hill climbing respects max_passes limit."""
        tt = TransitionTable(num_stages=1)
        s0_in = _make_state([0, 1], [2, 3])
        s0_out = _make_state([1, 0], [3, 2])

        gadget_a = _make_gadget("a", 1)
        gadget_b = _make_gadget("b", 2)

        tt.add_transition(0, s0_in, s0_out, gadget_a)
        tt.add_transition(0, s0_in, s0_out, gadget_b)

        path = [(0, s0_in.as_tuple(), s0_out.as_tuple())]

        call_count = 0

        def counting_scorer(_path_spec, _gadget_assignments):
            nonlocal call_count
            call_count += 1
            # Always improving to force max passes
            return 100.0 - call_count

        hill_climb_path(path, tt, counting_scorer, max_passes=2)
        # Should stop after max_passes even if still improving
        # With 1 stage, 2 gadgets, max_passes=2: initial + up to 2 passes
        assert call_count <= 10  # reasonable upper bound
