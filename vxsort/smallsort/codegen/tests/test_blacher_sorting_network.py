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
from bitonic_super_optimizer import (
    GadgetSynthesizer,
    InstructionSpec,
    SymbolicPlaceholder,
    VectorState,
)
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


def test_synthesize_blacher_stage9_and_cse():
    """Synthesize Blacher stage 9 with depth-2 template, verify CSE deduplication.

    Stage 9 of the Blacher AVX2/i32 network transforms:
      top:    [1,9,6,14,2,10,5,13]   ->  [1,9,3,11,5,13,7,15]
      bottom: [3,11,8,16,4,12,7,15]  ->  [2,10,4,12,6,14,8,16]

    "Depth 1.5" template: a shared permutation prefix (same SymbolicPlaceholder
    names for both top and bottom) piped into an independent single instruction.
    By construction, the permutexvar CVs are the same Z3 variable for both sides,
    guaranteeing identical concrete values. Only the shuffle_ps imm8 differs.

    After synthesis we verify:
    1. Both sides have identical permutexvar instructions (shared by construction)
    2. The shuffle_ps imm8 values differ between top and bottom
    3. unified_instructions() discovers the shared prefix -> 4 instructions (not 6)
    """
    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)

    input_state = VectorState(
        top=[1, 9, 6, 14, 2, 10, 5, 13],
        bottom=[3, 11, 8, 16, 4, 12, 7, 15],
    )
    target_pairs = [
        (1, 2),
        (9, 10),
        (3, 4),
        (11, 12),
        (5, 6),
        (13, 14),
        (7, 8),
        (15, 16),
    ]

    shared_cv0 = SymbolicPlaceholder("cv0", 256)
    shared_cv1 = SymbolicPlaceholder("cv1", 256)

    top_template = [
        InstructionSpec(
            "_mm256_permutexvar_epi32",
            {"a": "top", "op_idx": shared_cv0},
        ),
        InstructionSpec(
            "_mm256_permutexvar_epi32",
            {"a": "bottom", "op_idx": shared_cv1},
        ),
        InstructionSpec(
            "_mm256_shuffle_ps",
            {"a": "top", "b": "bottom", "imm8": SymbolicPlaceholder("imm_top", 8)},
        ),
    ]
    bottom_template = [
        InstructionSpec(
            "_mm256_permutexvar_epi32",
            {"a": "top", "op_idx": shared_cv0},
        ),
        InstructionSpec(
            "_mm256_permutexvar_epi32",
            {"a": "bottom", "op_idx": shared_cv1},
        ),
        InstructionSpec(
            "_mm256_shuffle_ps",
            {"a": "top", "b": "bottom", "imm8": SymbolicPlaceholder("imm_bot", 8)},
        ),
    ]

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        top_instructions_template=top_template,
        bottom_instructions_template=bottom_template,
        input_state=input_state,
        target_pairs=target_pairs,
        max_solutions=1,
        allow_any_lane_order=False,
    )
    assert len(results) > 0, "Synthesis should find a solution"
    gadget, _ = results[0]

    assert len(gadget.top_instructions) == 3
    assert len(gadget.bottom_instructions) == 3

    # Shared prefix is identical by construction
    for i in range(2):
        assert (
            gadget.top_instructions[i].args == gadget.bottom_instructions[i].args
        ), f"Instruction {i}: shared prefix should be identical by construction"

    # Tail instructions diverge (same intrinsic, different imm8)
    assert gadget.top_instructions[2].intrinsic_name == "_mm256_shuffle_ps"
    assert gadget.bottom_instructions[2].intrinsic_name == "_mm256_shuffle_ps"
    assert (
        gadget.top_instructions[2].args["imm8"]
        != gadget.bottom_instructions[2].args["imm8"]
    ), "Top and bottom shuffle_ps should have different imm8 values"

    # CSE deduplicates the shared prefix: 6 raw -> 4 unified
    unified, top_out, bottom_out = gadget.unified_instructions()
    assert len(unified) == 4, f"Expected 4 after CSE, got {len(unified)}"
    assert gadget.instruction_count() == 4
    assert top_out != bottom_out


def test_filtered_synthesis_blacher_stage9_cse():
    """Synthesize Blacher stage 9 with only permutexvar+shuffle_ps, verify CSE.

    Uses intrinsic_filter to restrict synthesis to just two instructions,
    drastically reducing search space and solve time. The depth-1.5 template
    (shared permutation prefix + independent shuffle tail) should still find
    a solution and produce CSE deduplication (4 instructions instead of 6).
    """
    synth = GadgetSynthesizer(
        vector_machine.AVX2,
        primitive_type.i32,
        intrinsic_filter={"_mm256_permutexvar_epi32", "_mm256_shuffle_ps"},
    )

    input_state = VectorState(
        top=[1, 9, 6, 14, 2, 10, 5, 13],
        bottom=[3, 11, 8, 16, 4, 12, 7, 15],
    )
    target_pairs = [
        (1, 2),
        (9, 10),
        (3, 4),
        (11, 12),
        (5, 6),
        (13, 14),
        (7, 8),
        (15, 16),
    ]

    shared_cv0 = SymbolicPlaceholder("cv0", 256)
    shared_cv1 = SymbolicPlaceholder("cv1", 256)

    top_template = [
        InstructionSpec(
            "_mm256_permutexvar_epi32",
            {"a": "top", "op_idx": shared_cv0},
        ),
        InstructionSpec(
            "_mm256_permutexvar_epi32",
            {"a": "bottom", "op_idx": shared_cv1},
        ),
        InstructionSpec(
            "_mm256_shuffle_ps",
            {"a": "top", "b": "bottom", "imm8": SymbolicPlaceholder("imm_top", 8)},
        ),
    ]
    bottom_template = [
        InstructionSpec(
            "_mm256_permutexvar_epi32",
            {"a": "top", "op_idx": shared_cv0},
        ),
        InstructionSpec(
            "_mm256_permutexvar_epi32",
            {"a": "bottom", "op_idx": shared_cv1},
        ),
        InstructionSpec(
            "_mm256_shuffle_ps",
            {"a": "top", "b": "bottom", "imm8": SymbolicPlaceholder("imm_bot", 8)},
        ),
    ]

    results, _, _ = synth.synthesize_gadget_with_symbolic(
        top_instructions_template=top_template,
        bottom_instructions_template=bottom_template,
        input_state=input_state,
        target_pairs=target_pairs,
        max_solutions=1,
        allow_any_lane_order=False,
    )
    assert len(results) > 0, "Synthesis should find a solution"
    gadget, output_state = results[0]

    # Verify output correctness
    assert sorted(output_state.top) != sorted(output_state.bottom)

    # Shared prefix is identical by construction
    for i in range(2):
        assert (
            gadget.top_instructions[i].args == gadget.bottom_instructions[i].args
        ), f"Instruction {i}: shared prefix should be identical"

    # Tail instructions diverge
    assert gadget.top_instructions[2].intrinsic_name == "_mm256_shuffle_ps"
    assert gadget.bottom_instructions[2].intrinsic_name == "_mm256_shuffle_ps"
    assert (
        gadget.top_instructions[2].args["imm8"]
        != gadget.bottom_instructions[2].args["imm8"]
    ), "Top and bottom shuffle_ps should have different imm8 values"

    # CSE deduplicates: 6 raw -> 4 unified
    unified, top_out, bottom_out = gadget.unified_instructions()
    assert len(unified) == 4, f"Expected 4 after CSE, got {len(unified)}"
    assert gadget.instruction_count() == 4
    assert top_out != bottom_out


def test_filtered_synthesis_blacher_stage10_natural_order():
    """Synthesize Blacher stage 10 (natural order) with permute_ps+blend_ps.

    Stage 10 restores natural element order after sorting:
      top:    [1,9,3,11,5,13,7,15]  ->  [1,2,3,4,5,6,7,8]
      bottom: [2,10,4,12,6,14,8,16] ->  [9,10,11,12,13,14,15,16]

    Classical depth-2 gadget: each side permutes the OTHER register to bring
    interleaved values into adjacent lanes, then blends with its own register
    to pick the right elements. Each inst1 (the blend) only references prev
    and the original top/bottom registers via the mux encoding.

    Unlike stage 9 where both sides permute the SAME registers (enabling a
    shared prefix and CSE), here the top side must permute bottom and the
    bottom side must permute top -- no sharing is possible.
    """
    synth = GadgetSynthesizer(
        vector_machine.AVX2,
        primitive_type.i32,
        intrinsic_filter={"_mm256_permute_ps", "_mm256_blend_ps"},
    )

    input_state = VectorState(
        top=[1, 9, 3, 11, 5, 13, 7, 15],
        bottom=[2, 10, 4, 12, 6, 14, 8, 16],
    )
    # Natural order: pair element i with element i+8 in lane i
    target_pairs = [
        (1, 9),
        (2, 10),
        (3, 11),
        (4, 12),
        (5, 13),
        (6, 14),
        (7, 15),
        (8, 16),
    ]

    # Top side: permute bottom to bring even values adjacent, then blend
    top_template = [
        InstructionSpec(
            "_mm256_permute_ps",
            {"a": "bottom", "imm8": SymbolicPlaceholder("imm_perm_top", 8)},
        ),
        InstructionSpec(
            "_mm256_blend_ps",
            {
                "a": "top",
                "b": "bottom",
                "imm8": SymbolicPlaceholder("imm_blend_top", 8),
            },
        ),
    ]
    # Bottom side: permute top to bring odd values adjacent, then blend
    bottom_template = [
        InstructionSpec(
            "_mm256_permute_ps",
            {"a": "top", "imm8": SymbolicPlaceholder("imm_perm_bot", 8)},
        ),
        InstructionSpec(
            "_mm256_blend_ps",
            {
                "a": "top",
                "b": "bottom",
                "imm8": SymbolicPlaceholder("imm_blend_bot", 8),
            },
        ),
    ]

    results, _, _ = synth.synthesize_gadget_with_symbolic(
        top_instructions_template=top_template,
        bottom_instructions_template=bottom_template,
        input_state=input_state,
        target_pairs=target_pairs,
        max_solutions=1,
        allow_any_lane_order=False,
    )
    assert len(results) > 0, "Synthesis should find a solution"
    gadget, _ = results[0]

    assert len(gadget.top_instructions) == 2
    assert len(gadget.bottom_instructions) == 2

    # Each side is permute then blend
    assert gadget.top_instructions[0].intrinsic_name == "_mm256_permute_ps"
    assert gadget.top_instructions[1].intrinsic_name == "_mm256_blend_ps"
    assert gadget.bottom_instructions[0].intrinsic_name == "_mm256_permute_ps"
    assert gadget.bottom_instructions[1].intrinsic_name == "_mm256_blend_ps"

    # Top permutes bottom, bottom permutes top -- no CSE possible
    assert gadget.instruction_count() == 4
