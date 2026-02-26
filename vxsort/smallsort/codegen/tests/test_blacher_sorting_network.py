"""Concrete verification of the Blacher AVX2/i32 sorting network.

Tests that the sorting network from BM_blacher.avx2.cpp, encoded as
fixture_2xAVX2_i32_blacher.json, produces correctly sorted output for
concrete inputs: sorted, reversed, and random permutations.

Uses Z3 BitVecVal (concrete values) evaluated through the AVX instruction
pipeline — no SMT solver invocation needed, just simplify(). This keeps
test time well under a second.
"""

import os
import random

import pytest
from z3 import BitVecVal, Concat, Extract, simplify

import z3_avx
from bitonic_verifier import (
    BitonicPathVerifier,
    VerifyStep,
    load_solutions_from_json,
)
from utils import primitive_type, vector_machine

FIXTURE_DIR = os.path.dirname(__file__)


def _extract_linear_steps(bundle):
    """Walk a linear-chain fixture (one gadget, one child per node) into VerifySteps."""
    steps = []
    node = bundle.roots[0]
    while True:
        steps.append(
            VerifyStep(
                gadget=node.gadgets[0],
                input_state=node.input_state,
                output_state=node.output_state,
            )
        )
        if not node.children:
            break
        node = node.children[0]
    return steps


def _apply_sorting_network(verifier, steps, values, natural_order):
    """Apply the sorting network to concrete 32-bit values, return output list.

    Uses Z3 simplify() to evaluate concrete BitVecVal expressions through
    the AVX instruction pipeline. No solver invocation needed.
    """
    lane_width = verifier.lane_width
    total_bits = verifier.total_bits

    elems = [BitVecVal(v, lane_width) for v in values]

    # Build initial registers (Concat expects MSB-first, so reverse lane order)
    initial = steps[0].input_state
    top_lanes = [elems[label - 1] for label in initial.top]
    bottom_lanes = [elems[label - 1] for label in initial.bottom]
    top_vec = simplify(Concat(top_lanes[::-1]))
    bottom_vec = simplify(Concat(bottom_lanes[::-1]))

    for step_idx, step in enumerate(steps):
        gadget = step.gadget
        is_last = step_idx == len(steps) - 1

        new_top = verifier._apply_concrete_instructions(
            top_vec, bottom_vec, gadget.top_instructions, is_top=True
        )
        new_bottom = verifier._apply_concrete_instructions(
            top_vec, bottom_vec, gadget.bottom_instructions, is_top=False
        )

        if natural_order and is_last:
            top_vec = simplify(new_top)
            bottom_vec = simplify(new_bottom)
        else:
            top_vec = simplify(
                z3_avx.generic_min(new_top, new_bottom, total_bits, lane_width)
            )
            bottom_vec = simplify(
                z3_avx.generic_max(new_top, new_bottom, total_bits, lane_width)
            )

    # Extract output values in label order 1..N
    final = steps[-1].output_state
    label_to_pos = {}
    for lane, label in enumerate(final.top):
        label_to_pos[label] = ("top", lane)
    for lane, label in enumerate(final.bottom):
        label_to_pos[label] = ("bottom", lane)

    result = []
    for i in range(1, len(values) + 1):
        vec_name, lane = label_to_pos[i]
        vec = top_vec if vec_name == "top" else bottom_vec
        lo = lane * lane_width
        hi = lo + lane_width - 1
        result.append(simplify(Extract(hi, lo, vec)).as_long())

    return result


class TestBlacherSortingNetwork:
    """Concrete verification of the Blacher AVX2/i32 16-element sorting network."""

    @pytest.fixture
    def blacher(self):
        path = os.path.join(FIXTURE_DIR, "fixture_2xAVX2_i32_blacher.json")
        bundle = load_solutions_from_json(path)
        steps = _extract_linear_steps(bundle)
        verifier = BitonicPathVerifier(vector_machine.AVX2, primitive_type.i32)
        return verifier, steps, bundle.natural_order

    def test_fixture_has_11_stages(self, blacher):
        """10 permutation+COEX stages + 1 natural-order reorder stage."""
        _, steps, natural_order = blacher
        assert len(steps) == 11
        assert natural_order is True
        # Stages numbered 0..10
        for i, step in enumerate(steps):
            assert step.input_state is not None
            assert step.output_state is not None

    def test_initial_and_final_states(self, blacher):
        """Input starts at identity, output ends at natural order."""
        _, steps, _ = blacher
        assert steps[0].input_state.top == [1, 2, 3, 4, 5, 6, 7, 8]
        assert steps[0].input_state.bottom == [9, 10, 11, 12, 13, 14, 15, 16]
        assert steps[-1].output_state.top == [1, 2, 3, 4, 5, 6, 7, 8]
        assert steps[-1].output_state.bottom == [9, 10, 11, 12, 13, 14, 15, 16]

    def test_sorted_input(self, blacher):
        """Already-sorted input stays sorted."""
        verifier, steps, natural_order = blacher
        values = list(range(1, 17))
        result = _apply_sorting_network(verifier, steps, values, natural_order)
        assert result == sorted(values)

    def test_reversed_input(self, blacher):
        """Reversed input gets sorted."""
        verifier, steps, natural_order = blacher
        values = list(range(16, 0, -1))
        result = _apply_sorting_network(verifier, steps, values, natural_order)
        assert result == sorted(values)

    def test_random_permutations(self, blacher):
        """20 random permutations all get sorted correctly."""
        verifier, steps, natural_order = blacher
        rng = random.Random(42)
        for trial in range(20):
            values = list(range(1, 17))
            rng.shuffle(values)
            result = _apply_sorting_network(verifier, steps, values, natural_order)
            assert result == sorted(values), f"Trial {trial} failed for input: {values}"

    def test_signed_negative_values(self, blacher):
        """Signed 32-bit values including negatives get sorted."""
        verifier, steps, natural_order = blacher
        signed_vals = list(range(-8, 8))
        # Convert to unsigned 32-bit representation for BitVecVal
        unsigned_vals = [v % (1 << 32) for v in signed_vals]
        rng = random.Random(99)
        rng.shuffle(unsigned_vals)
        result = _apply_sorting_network(verifier, steps, unsigned_vals, natural_order)
        # Convert back to signed for comparison
        signed_result = [v if v < (1 << 31) else v - (1 << 32) for v in result]
        assert signed_result == sorted(signed_result)

    def test_duplicate_free_wide_range(self, blacher):
        """Large spread of values across the 32-bit signed range."""
        verifier, steps, natural_order = blacher
        values = [
            0x00000001,
            0x00000100,
            0x00010000,
            0x01000000,
            0x10000000,
            0x20000000,
            0x30000000,
            0x40000000,
            0x00000002,
            0x00000200,
            0x00020000,
            0x02000000,
            0x50000000,
            0x60000000,
            0x70000000,
            0x7FFFFFFF,
        ]
        rng = random.Random(7)
        rng.shuffle(values)
        result = _apply_sorting_network(verifier, steps, values, natural_order)
        assert result == sorted(values)
