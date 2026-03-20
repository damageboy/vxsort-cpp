"""Tests for TransitionTable data structure."""

import pytest
from bitonic_types import InstructionSpec, PermutationGadget, VectorState
from transition_table import StageData, TransitionTable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gadget(
    top_name: str = "vperm",
    bot_name: str = "vperm",
    top_args: dict | None = None,
    bot_args: dict | None = None,
) -> PermutationGadget:
    """Create a simple PermutationGadget for testing."""
    top_args = top_args or {"ctrl": 0x1B}
    bot_args = bot_args or {"ctrl": 0x1B}
    return PermutationGadget(
        top_instructions=[InstructionSpec(top_name, top_args)],
        bottom_instructions=[InstructionSpec(bot_name, bot_args)],
        validated=True,
    )


def _vs(top: list[int], bottom: list[int]) -> VectorState:
    """Shorthand VectorState constructor."""
    return VectorState(top=top, bottom=bottom)


# ---------------------------------------------------------------------------
# StageData basics
# ---------------------------------------------------------------------------


class TestStageData:
    """StageData construction and basic operations."""

    def test_initial_state(self):
        sd = StageData()
        assert sd.transitions == {}
        assert sd.unique_outputs == set()
        assert sd.unique_inputs == set()
        assert sd.attempted_pairs == set()
        assert sd.forwarded_outputs == set()
        assert sd.consecutive_zero_budgets == 0
        assert sd.attempts == 0


# ---------------------------------------------------------------------------
# TransitionTable basics: add/lookup/dedup
# ---------------------------------------------------------------------------


class TestTransitionTableBasics:
    """Core add_transition / lookup / deduplication."""

    def test_create_with_num_stages(self):
        tt = TransitionTable(num_stages=3)
        assert len(tt.stages) == 3
        for sd in tt.stages:
            assert isinstance(sd, StageData)

    def test_add_transition_returns_true_for_new(self):
        tt = TransitionTable(num_stages=1)
        inp = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        out = _vs([0, 1, 2, 3], [4, 5, 6, 7])
        gadget = _make_gadget()

        added = tt.add_transition(
            stage=0, input_state=inp, output_state=out, gadget=gadget
        )
        assert added is True

    def test_add_transition_returns_false_for_duplicate_gadget(self):
        tt = TransitionTable(num_stages=1)
        inp = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        out = _vs([0, 1, 2, 3], [4, 5, 6, 7])
        gadget = _make_gadget()

        tt.add_transition(0, inp, out, gadget)
        added = tt.add_transition(0, inp, out, gadget)
        assert added is False

    def test_add_transition_allows_different_gadgets_for_same_pair(self):
        tt = TransitionTable(num_stages=1)
        inp = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        out = _vs([0, 1, 2, 3], [4, 5, 6, 7])
        g1 = _make_gadget(top_args={"ctrl": 0x1B})
        g2 = _make_gadget(top_args={"ctrl": 0xFF})

        assert tt.add_transition(0, inp, out, g1) is True
        assert tt.add_transition(0, inp, out, g2) is True

        key = (inp.as_tuple(), out.as_tuple())
        assert len(tt.stages[0].transitions[key]) == 2

    def test_add_transition_tracks_unique_inputs_and_outputs(self):
        tt = TransitionTable(num_stages=1)
        inp = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        out = _vs([0, 1, 2, 3], [4, 5, 6, 7])
        gadget = _make_gadget()

        tt.add_transition(0, inp, out, gadget)

        assert inp.as_tuple() in tt.stages[0].unique_inputs
        assert out.as_tuple() in tt.stages[0].unique_outputs

    def test_lookup_transitions(self):
        tt = TransitionTable(num_stages=1)
        inp = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        out = _vs([0, 1, 2, 3], [4, 5, 6, 7])
        gadget = _make_gadget()

        tt.add_transition(0, inp, out, gadget)

        result = tt.get_transitions(0, inp.as_tuple())
        assert len(result) == 1
        assert result[out.as_tuple()] == [gadget]

    def test_lookup_transitions_empty(self):
        tt = TransitionTable(num_stages=1)
        inp_key = _vs([9, 8, 7, 6], [5, 4, 3, 2]).as_tuple()
        result = tt.get_transitions(0, inp_key)
        assert result == {}

    def test_multiple_outputs_from_same_input(self):
        tt = TransitionTable(num_stages=1)
        inp = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        out1 = _vs([0, 1, 2, 3], [4, 5, 6, 7])
        out2 = _vs([1, 0, 3, 2], [5, 4, 7, 6])
        g1 = _make_gadget(top_args={"ctrl": 0x1B})
        g2 = _make_gadget(top_args={"ctrl": 0xFF})

        tt.add_transition(0, inp, out1, g1)
        tt.add_transition(0, inp, out2, g2)

        result = tt.get_transitions(0, inp.as_tuple())
        assert len(result) == 2
        assert out1.as_tuple() in result
        assert out2.as_tuple() in result


# ---------------------------------------------------------------------------
# stage_stats
# ---------------------------------------------------------------------------


class TestStageStats:
    """stage_stats() returns correct summaries."""

    def test_empty_stage(self):
        tt = TransitionTable(num_stages=1)
        stats = tt.stage_stats(0)
        assert stats["attempts"] == 0
        assert stats["distinct_outputs"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["transition_count"] == 0
        assert stats["total_gadgets"] == 0

    def test_populated_stage(self):
        tt = TransitionTable(num_stages=1)
        inp = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        out1 = _vs([0, 1, 2, 3], [4, 5, 6, 7])
        out2 = _vs([1, 0, 3, 2], [5, 4, 7, 6])
        g1 = _make_gadget(top_args={"ctrl": 0x1B})
        g2 = _make_gadget(top_args={"ctrl": 0xFF})
        g3 = _make_gadget(top_args={"ctrl": 0xAA})

        tt.add_transition(0, inp, out1, g1)
        tt.add_transition(0, inp, out1, g2)  # second gadget for same pair
        tt.add_transition(0, inp, out2, g3)

        # Mark some attempts
        tt.stages[0].attempts = 5

        stats = tt.stage_stats(0)
        assert stats["attempts"] == 5
        assert stats["distinct_outputs"] == 2
        assert stats["success_rate"] == pytest.approx(2 / 5)
        assert stats["transition_count"] == 2  # 2 (input, output) pairs
        assert stats["total_gadgets"] == 3  # 3 gadgets total


# ---------------------------------------------------------------------------
# weakest_stage
# ---------------------------------------------------------------------------


class TestWeakestStage:
    """weakest_stage() identifies the stage most in need of work."""

    def test_all_empty_returns_first(self):
        tt = TransitionTable(num_stages=3)
        assert tt.weakest_stage() == 0

    def test_returns_lowest_success_rate(self):
        tt = TransitionTable(num_stages=3)

        # Stage 0: 2 outputs from 4 attempts => 50%
        tt.stages[0].attempts = 4
        tt.stages[0].unique_outputs = {((0,), (1,)), ((2,), (3,))}

        # Stage 1: 1 output from 4 attempts => 25%
        tt.stages[1].attempts = 4
        tt.stages[1].unique_outputs = {((0,), (1,))}

        # Stage 2: 3 outputs from 4 attempts => 75%
        tt.stages[2].attempts = 4
        tt.stages[2].unique_outputs = {((0,), (1,)), ((2,), (3,)), ((4,), (5,))}

        assert tt.weakest_stage() == 1

    def test_tie_breaks_by_earliest_stage(self):
        tt = TransitionTable(num_stages=3)

        # All stages: 1 output from 4 attempts => 25%
        for i in range(3):
            tt.stages[i].attempts = 4
            tt.stages[i].unique_outputs = {((0,), (1,))}

        assert tt.weakest_stage() == 0

    def test_zero_attempts_treated_as_zero_rate(self):
        tt = TransitionTable(num_stages=2)

        # Stage 0: 2 outputs from 4 attempts => 50%
        tt.stages[0].attempts = 4
        tt.stages[0].unique_outputs = {((0,), (1,)), ((2,), (3,))}

        # Stage 1: 0 attempts => rate 0.0 (weakest)
        tt.stages[1].attempts = 0

        assert tt.weakest_stage() == 1


# ---------------------------------------------------------------------------
# Path enumeration
# ---------------------------------------------------------------------------


class TestEnumerateCompletePaths:
    """enumerate_complete_paths() DFS traversal."""

    def _build_linear_2stage(self):
        """Build a simple 2-stage table with one path.

        Stage 0: inp_a -> out_a
        Stage 1: out_a -> out_b
        """
        tt = TransitionTable(num_stages=2)
        inp_a = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        out_a = _vs([0, 2, 1, 3], [4, 6, 5, 7])
        out_b = _vs([0, 1, 2, 3], [4, 5, 6, 7])
        g1 = _make_gadget(top_args={"ctrl": 0x1B})
        g2 = _make_gadget(top_args={"ctrl": 0xFF})

        tt.add_transition(0, inp_a, out_a, g1)
        tt.add_transition(1, out_a, out_b, g2)

        return tt, inp_a, out_a, out_b, g1, g2

    def test_simple_2stage_path(self):
        tt, inp_a, out_a, out_b, g1, g2 = self._build_linear_2stage()

        paths = tt.enumerate_complete_paths(inp_a.as_tuple())
        assert len(paths) == 1

        path = paths[0]
        assert len(path) == 2  # one entry per stage

        # Each step: (input_tuple, output_tuple, gadget)
        step0_inp, step0_out, step0_gadget = path[0]
        assert step0_inp == inp_a.as_tuple()
        assert step0_out == out_a.as_tuple()
        assert step0_gadget is g1

        step1_inp, step1_out, step1_gadget = path[1]
        assert step1_inp == out_a.as_tuple()
        assert step1_out == out_b.as_tuple()
        assert step1_gadget is g2

    def test_branching_paths(self):
        """Two distinct outputs at stage 0 that both continue to stage 1."""
        tt = TransitionTable(num_stages=2)
        inp = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        mid1 = _vs([0, 2, 1, 3], [4, 6, 5, 7])
        mid2 = _vs([2, 0, 3, 1], [6, 4, 7, 5])
        final = _vs([0, 1, 2, 3], [4, 5, 6, 7])

        g1 = _make_gadget(top_args={"ctrl": 0x1B})
        g2 = _make_gadget(top_args={"ctrl": 0xFF})
        g3 = _make_gadget(top_args={"ctrl": 0xAA})
        g4 = _make_gadget(top_args={"ctrl": 0xBB})

        tt.add_transition(0, inp, mid1, g1)
        tt.add_transition(0, inp, mid2, g2)
        tt.add_transition(1, mid1, final, g3)
        tt.add_transition(1, mid2, final, g4)

        paths = tt.enumerate_complete_paths(inp.as_tuple())
        assert len(paths) == 2

    def test_max_paths_budget(self):
        """max_paths limits the number of returned paths."""
        tt = TransitionTable(num_stages=2)
        inp = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        mid1 = _vs([0, 2, 1, 3], [4, 6, 5, 7])
        mid2 = _vs([2, 0, 3, 1], [6, 4, 7, 5])
        final = _vs([0, 1, 2, 3], [4, 5, 6, 7])

        g1 = _make_gadget(top_args={"ctrl": 0x1B})
        g2 = _make_gadget(top_args={"ctrl": 0xFF})
        g3 = _make_gadget(top_args={"ctrl": 0xAA})
        g4 = _make_gadget(top_args={"ctrl": 0xBB})

        tt.add_transition(0, inp, mid1, g1)
        tt.add_transition(0, inp, mid2, g2)
        tt.add_transition(1, mid1, final, g3)
        tt.add_transition(1, mid2, final, g4)

        paths = tt.enumerate_complete_paths(inp.as_tuple(), max_paths=1)
        assert len(paths) == 1

    def test_dead_end_not_returned(self):
        """A path that can't reach the last stage is not a complete path."""
        tt = TransitionTable(num_stages=2)
        inp = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        mid = _vs([0, 2, 1, 3], [4, 6, 5, 7])

        g1 = _make_gadget(top_args={"ctrl": 0x1B})

        # Only stage 0 has a transition; stage 1 is empty
        tt.add_transition(0, inp, mid, g1)

        paths = tt.enumerate_complete_paths(inp.as_tuple())
        assert len(paths) == 0

    def test_exclude_paths(self):
        """exclude_paths filters already-known paths."""
        tt, inp_a, out_a, out_b, g1, g2 = self._build_linear_2stage()

        # The single path
        paths = tt.enumerate_complete_paths(inp_a.as_tuple())
        assert len(paths) == 1

        # Convert path to a hashable form for exclusion
        path_key = tuple((step[0], step[1]) for step in paths[0])

        # Now exclude it
        paths2 = tt.enumerate_complete_paths(inp_a.as_tuple(), exclude_paths={path_key})
        assert len(paths2) == 0

    def test_multiple_gadgets_per_transition_produce_separate_paths(self):
        """If a transition has 2 gadgets, enumerate produces 2 paths."""
        tt = TransitionTable(num_stages=1)
        inp = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        out = _vs([0, 1, 2, 3], [4, 5, 6, 7])

        g1 = _make_gadget(top_args={"ctrl": 0x1B})
        g2 = _make_gadget(top_args={"ctrl": 0xFF})

        tt.add_transition(0, inp, out, g1)
        tt.add_transition(0, inp, out, g2)

        paths = tt.enumerate_complete_paths(inp.as_tuple())
        assert len(paths) == 2
        gadgets = {p[0][2].top_instructions[0].args["ctrl"] for p in paths}
        assert gadgets == {0x1B, 0xFF}

    def test_3_stage_path(self):
        """A 3-stage linear path works correctly."""
        tt = TransitionTable(num_stages=3)
        s0 = _vs([3, 2, 1, 0], [7, 6, 5, 4])
        s1 = _vs([2, 3, 0, 1], [6, 7, 4, 5])
        s2 = _vs([0, 1, 2, 3], [6, 7, 4, 5])
        s3 = _vs([0, 1, 2, 3], [4, 5, 6, 7])

        g1 = _make_gadget(top_args={"ctrl": 1})
        g2 = _make_gadget(top_args={"ctrl": 2})
        g3 = _make_gadget(top_args={"ctrl": 3})

        tt.add_transition(0, s0, s1, g1)
        tt.add_transition(1, s1, s2, g2)
        tt.add_transition(2, s2, s3, g3)

        paths = tt.enumerate_complete_paths(s0.as_tuple())
        assert len(paths) == 1
        assert len(paths[0]) == 3
        assert paths[0][0][2] is g1
        assert paths[0][1][2] is g2
        assert paths[0][2][2] is g3
