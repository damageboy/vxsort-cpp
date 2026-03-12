"""Concrete verification of the Blacher AVX2/i32 sorting network.

Tests that the sorting network from BM_blacher.avx2.cpp, encoded as
fixture_2xAVX2_i32_blacher.json, produces correctly sorted output for
concrete inputs: sorted, reversed, and random permutations.

Uses Z3 BitVecVal (concrete values) evaluated through the AVX instruction
pipeline — no SMT solver invocation needed, just simplify(). This keeps
test time well under a second.
"""

import glob
import json
import os
import random

import pytest
from z3 import BitVecVal, Concat, Extract, simplify

import z3_avx
from bitonic_super_optimizer import (
    GadgetSynthesizer,
    VectorState,
)
from bitonic_types import GadgetGraph, InputRef, IntrinsicNode, Mux, Symbolic
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

    "Depth 1.5" template: a shared permutation prefix (same Symbolic
    nodes for both top and bottom) piped into an independent single instruction.
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

    shared_cv0 = Symbolic("cv0", 256)
    shared_cv1 = Symbolic("cv1", 256)

    # Shared prefix nodes (same Python objects for both sides → CSE)
    perm_top_input = IntrinsicNode(
        "_mm256_permutexvar_epi32",
        {"a": InputRef("top"), "op_idx": shared_cv0},
    )
    perm_bot_input = IntrinsicNode(
        "_mm256_permutexvar_epi32",
        {"a": InputRef("bottom"), "op_idx": shared_cv1},
    )

    # Tail nodes: each side shuffles the shared prefix results with a different imm8
    top_node = IntrinsicNode(
        "_mm256_shuffle_ps",
        {"a": perm_top_input, "b": perm_bot_input, "imm8": Symbolic("imm_top", 8)},
    )
    bottom_node = IntrinsicNode(
        "_mm256_shuffle_ps",
        {"a": perm_top_input, "b": perm_bot_input, "imm8": Symbolic("imm_bot", 8)},
    )

    graph = GadgetGraph(top=top_node, bottom=bottom_node)

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        graph,
        input_state,
        target_pairs,
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

    shared_cv0 = Symbolic("cv0", 256)
    shared_cv1 = Symbolic("cv1", 256)

    # Shared prefix nodes (same Python objects for both sides → CSE)
    perm_top_input = IntrinsicNode(
        "_mm256_permutexvar_epi32",
        {"a": InputRef("top"), "op_idx": shared_cv0},
    )
    perm_bot_input = IntrinsicNode(
        "_mm256_permutexvar_epi32",
        {"a": InputRef("bottom"), "op_idx": shared_cv1},
    )

    # Tail nodes: each side shuffles the shared prefix results with a different imm8
    top_node = IntrinsicNode(
        "_mm256_shuffle_ps",
        {"a": perm_top_input, "b": perm_bot_input, "imm8": Symbolic("imm_top", 8)},
    )
    bottom_node = IntrinsicNode(
        "_mm256_shuffle_ps",
        {"a": perm_top_input, "b": perm_bot_input, "imm8": Symbolic("imm_bot", 8)},
    )

    graph = GadgetGraph(top=top_node, bottom=bottom_node)

    results, _, _ = synth.synthesize_gadget_with_symbolic(
        graph,
        input_state,
        target_pairs,
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

    top_input = InputRef("top")
    bottom_input = InputRef("bottom")

    # Top side: permute bottom to bring even values adjacent, then blend
    perm_top = IntrinsicNode(
        "_mm256_permute_ps",
        {"a": bottom_input, "imm8": Symbolic("imm_perm_top", 8)},
    )
    blend_top = IntrinsicNode(
        "_mm256_blend_ps",
        {
            "a": Mux(
                select=Symbolic("sel_a_blend_top", 2),
                sources=(top_input, bottom_input, perm_top),
                pruning="at_least_one_last",
            ),
            "b": Mux(
                select=Symbolic("sel_b_blend_top", 2),
                sources=(top_input, bottom_input, perm_top),
                pruning="at_least_one_last",
            ),
            "imm8": Symbolic("imm_blend_top", 8),
        },
    )

    # Bottom side: permute top to bring odd values adjacent, then blend
    perm_bot = IntrinsicNode(
        "_mm256_permute_ps",
        {"a": top_input, "imm8": Symbolic("imm_perm_bot", 8)},
    )
    blend_bot = IntrinsicNode(
        "_mm256_blend_ps",
        {
            "a": Mux(
                select=Symbolic("sel_a_blend_bot", 2),
                sources=(top_input, bottom_input, perm_bot),
                pruning="at_least_one_last",
            ),
            "b": Mux(
                select=Symbolic("sel_b_blend_bot", 2),
                sources=(top_input, bottom_input, perm_bot),
                pruning="at_least_one_last",
            ),
            "imm8": Symbolic("imm_blend_bot", 8),
        },
    )

    graph = GadgetGraph(top=blend_top, bottom=blend_bot)

    results, _, _ = synth.synthesize_gadget_with_symbolic(
        graph,
        input_state,
        target_pairs,
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


def test_build_shared_prefix_graphs_structure():
    """_build_shared_prefix_graphs produces graphs with shared prefix nodes."""
    synth = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)
    graphs = synth._build_shared_prefix_graphs()

    assert len(graphs) > 0, "Should produce at least one shared-prefix graph"

    for g in graphs:
        assert g.top is not None
        assert g.bottom is not None
        assert isinstance(g.top, IntrinsicNode)
        assert isinstance(g.bottom, IntrinsicNode)

        # Tail nodes should reference shared prefix nodes
        top_intrinsic_ops = [
            v for v in g.top.operands.values() if isinstance(v, IntrinsicNode)
        ]
        bot_intrinsic_ops = [
            v for v in g.bottom.operands.values() if isinstance(v, IntrinsicNode)
        ]

        assert (
            len(top_intrinsic_ops) == 2
        ), f"Tail should have 2 prefix refs, got {len(top_intrinsic_ops)}"
        assert (
            len(bot_intrinsic_ops) == 2
        ), f"Tail should have 2 prefix refs, got {len(bot_intrinsic_ops)}"

        # Shared by identity: same Python objects (possibly in different order
        # for cross-ordering graphs)
        top_ids = {id(x) for x in top_intrinsic_ops}
        bot_ids = {id(x) for x in bot_intrinsic_ops}
        assert top_ids == bot_ids, (
            "Prefix nodes must be the same Python objects (shared by identity), "
            "though possibly in different order (cross-ordering)"
        )


def _has_shared_prefix(graph: GadgetGraph) -> bool:
    """Check whether a GadgetGraph has shared-prefix structure.

    A shared-prefix graph has top and bottom IntrinsicNode tails whose
    IntrinsicNode operands are the same Python objects (shared by identity).
    """
    if not isinstance(graph.top, IntrinsicNode) or not isinstance(
        graph.bottom, IntrinsicNode
    ):
        return False
    top_intrinsic_ops = [
        v for v in graph.top.operands.values() if isinstance(v, IntrinsicNode)
    ]
    bot_intrinsic_ops = [
        v for v in graph.bottom.operands.values() if isinstance(v, IntrinsicNode)
    ]
    if len(top_intrinsic_ops) != 2 or len(bot_intrinsic_ops) != 2:
        return False
    return {id(x) for x in top_intrinsic_ops} == {id(x) for x in bot_intrinsic_ops}


def test_precompute_includes_shared_prefix_graphs():
    """precompute_all_candidates includes shared-prefix graphs at depth >= 2."""
    synth = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)

    shared_prefix_graphs = synth._build_shared_prefix_graphs()
    assert len(shared_prefix_graphs) > 0, "Precondition: shared-prefix graphs exist"

    # depth=1 should NOT include any shared-prefix graphs
    candidates_d1 = synth.precompute_all_candidates(gadget_depth=1)
    shared_in_d1 = [g for g in candidates_d1 if _has_shared_prefix(g)]
    assert (
        len(shared_in_d1) == 0
    ), f"depth=1 should not include shared-prefix graphs, found {len(shared_in_d1)}"

    # depth=2 should include shared-prefix graphs
    candidates_d2 = synth.precompute_all_candidates(gadget_depth=2)
    shared_in_d2 = [g for g in candidates_d2 if _has_shared_prefix(g)]
    assert len(shared_in_d2) >= len(shared_prefix_graphs), (
        f"depth=2 should include at least {len(shared_prefix_graphs)} "
        f"shared-prefix graphs, found {len(shared_in_d2)}"
    )


def test_shared_prefix_tail_symbolics_differ():
    """Top and bottom tails have independently-named Symbolic operands."""
    synth = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)
    graphs = synth._build_shared_prefix_graphs()

    for g in graphs:
        top_symbolics = {
            (k, v.name) for k, v in g.top.operands.items() if isinstance(v, Symbolic)
        }
        bot_symbolics = {
            (k, v.name) for k, v in g.bottom.operands.items() if isinstance(v, Symbolic)
        }
        top_keys = {k for k, _ in top_symbolics}
        bot_keys = {k for k, _ in bot_symbolics}
        assert (
            top_keys == bot_keys
        ), "Top and bottom tails should have same operand keys"

        top_names = {n for _, n in top_symbolics}
        bot_names = {n for _, n in bot_symbolics}
        assert top_names.isdisjoint(
            bot_names
        ), f"Top and bottom Symbolic names must differ: {top_names} vs {bot_names}"


def test_instruction_count_filter_rejects_above_4():
    """Gadgets with instruction_count > 4 would be filtered by the worker."""
    from bitonic_types import InstructionSpec, PermutationGadget

    # Construct a gadget with 3 instructions per side, no CSE (all different)
    gadget_no_cse = PermutationGadget(
        top_instructions=[
            InstructionSpec("_mm256_permute_ps", {"a": "top", "imm8": 0x44}),
            InstructionSpec("_mm256_permute_ps", {"a": "prev", "imm8": 0x55}),
            InstructionSpec(
                "_mm256_shuffle_ps", {"a": "prev", "b": "top", "imm8": 0x88}
            ),
        ],
        bottom_instructions=[
            InstructionSpec("_mm256_permute_ps", {"a": "bottom", "imm8": 0x66}),
            InstructionSpec("_mm256_permute_ps", {"a": "prev", "imm8": 0x77}),
            InstructionSpec(
                "_mm256_shuffle_ps", {"a": "prev", "b": "bottom", "imm8": 0x99}
            ),
        ],
        validated=True,
    )
    assert gadget_no_cse.instruction_count() == 6  # no dedup possible

    # Construct a gadget with 3 per side but shared prefix (CSE -> 4)
    gadget_cse = PermutationGadget(
        top_instructions=[
            InstructionSpec("_mm256_permutexvar_epi32", {"a": "top", "op_idx": 123}),
            InstructionSpec("_mm256_permutexvar_epi32", {"a": "bottom", "op_idx": 456}),
            InstructionSpec(
                "_mm256_shuffle_ps", {"a": "result_0", "b": "prev", "imm8": 0xAA}
            ),
        ],
        bottom_instructions=[
            InstructionSpec("_mm256_permutexvar_epi32", {"a": "top", "op_idx": 123}),
            InstructionSpec("_mm256_permutexvar_epi32", {"a": "bottom", "op_idx": 456}),
            InstructionSpec(
                "_mm256_shuffle_ps", {"a": "result_0", "b": "prev", "imm8": 0xBB}
            ),
        ],
        validated=True,
    )
    assert gadget_cse.instruction_count() == 4  # prefix deduped

    # Simulate the filter
    results = [(gadget_no_cse, None), (gadget_cse, None)]
    filtered = [(g, os) for g, os in results if g.instruction_count() <= 4]
    assert len(filtered) == 1
    assert filtered[0][0] is gadget_cse


def test_generated_shared_prefix_solves_blacher_stage9():
    """Generated shared-prefix graphs can solve Blacher stage 9 with CSE."""
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

    shared_graphs = synth._build_shared_prefix_graphs()
    assert len(shared_graphs) > 0

    # Try all generated shared-prefix graphs until one solves the stage
    found = False
    for graph in shared_graphs:
        results, _, _ = synth.synthesize_gadget_with_symbolic(
            graph,
            input_state,
            target_pairs,
            max_solutions=1,
            allow_any_lane_order=False,
        )
        if results:
            gadget, _ = results[0]
            # Must CSE-collapse to 4 instructions
            assert gadget.instruction_count() == 4
            assert len(gadget.top_instructions) == 3
            assert len(gadget.bottom_instructions) == 3
            # Shared prefix: first 2 instructions identical across sides
            for i in range(2):
                assert (
                    gadget.top_instructions[i].args
                    == gadget.bottom_instructions[i].args
                ), f"Prefix instruction {i} should be identical across sides"
            found = True
            break

    assert found, "No generated shared-prefix graph solved Blacher stage 9"


def test_shape_f_symmetric_dual_has_one_ordering():
    """Symmetric duals (isomorphic_order=True) produce only one operand ordering in Shape F."""
    synth = GadgetSynthesizer(
        vector_machine.AVX2,
        primitive_type.i32,
        intrinsic_filter={"_mm256_permute_ps", "_mm256_blend_ps"},
    )
    graphs = synth._build_shared_prefix_graphs()

    # blend_ps is symmetric + has Symbolics → one ordering
    blend_graphs = [g for g in graphs if g.top.name == "_mm256_blend_ps"]
    assert len(blend_graphs) > 0, "Should produce blend_ps graphs"

    # Count distinct orderings by looking at which prefix node is first reg operand
    orderings_seen = set()
    for g in blend_graphs:
        top_refs = [
            id(v) for v in g.top.operands.values() if isinstance(v, IntrinsicNode)
        ]
        orderings_seen.add(tuple(top_refs))

    assert (
        len(orderings_seen) == 1
    ), f"blend_ps (symmetric) should produce 1 ordering, got {len(orderings_seen)}"


def test_shape_f_asymmetric_parametric_dual_has_two_orderings():
    """Asymmetric duals with Symbolics produce two same-ordering variants in Shape F."""
    synth = GadgetSynthesizer(
        vector_machine.AVX2,
        primitive_type.i32,
        intrinsic_filter={"_mm256_permute_ps", "_mm256_shuffle_ps"},
    )
    graphs = synth._build_shared_prefix_graphs()

    shuffle_graphs = [g for g in graphs if g.top.name == "_mm256_shuffle_ps"]
    assert len(shuffle_graphs) > 0, "Should produce shuffle_ps graphs"

    # Count distinct orderings
    orderings_seen = set()
    for g in shuffle_graphs:
        top_refs = [
            id(v) for v in g.top.operands.values() if isinstance(v, IntrinsicNode)
        ]
        orderings_seen.add(tuple(top_refs))

    assert len(orderings_seen) == 2, (
        f"shuffle_ps (asymmetric+parametric) should produce 2 orderings, "
        f"got {len(orderings_seen)}"
    )


def test_shape_f_asymmetric_parameterless_dual_has_cross_ordering():
    """Asymmetric duals without Symbolics produce cross-ordering in Shape F."""
    synth = GadgetSynthesizer(
        vector_machine.AVX2,
        primitive_type.i32,
        intrinsic_filter={"_mm256_permute_ps", "_mm256_unpacklo_epi32"},
    )
    graphs = synth._build_shared_prefix_graphs()

    unpack_graphs = [g for g in graphs if g.top.name == "_mm256_unpacklo_epi32"]
    assert len(unpack_graphs) > 0, "Should produce unpacklo graphs (cross-ordering)"

    for g in unpack_graphs:
        top_reg_ids = [
            id(v) for v in g.top.operands.values() if isinstance(v, IntrinsicNode)
        ]
        bot_reg_ids = [
            id(v) for v in g.bottom.operands.values() if isinstance(v, IntrinsicNode)
        ]
        # Cross-ordering: top and bottom must use DIFFERENT operand orders
        assert top_reg_ids != bot_reg_ids, (
            "unpacklo (asymmetric, no Symbolics) must use cross-ordering "
            "(top and bottom tails have swapped operands)"
        )


# ---------------------------------------------------------------------------
# JSON fixture schema validation
# ---------------------------------------------------------------------------

FIXTURE_DIR = os.path.join(os.path.dirname(__file__))


def _validate_instruction(inst: dict, path: str) -> list[str]:
    """Validate a single instruction dict. Returns list of error strings."""
    errors = []
    if "name" not in inst:
        errors.append(f"{path}: missing 'name'")
    elif not isinstance(inst["name"], str):
        errors.append(f"{path}: 'name' must be str")
    if "args" not in inst:
        errors.append(f"{path}: missing 'args'")
    elif not isinstance(inst["args"], dict):
        errors.append(f"{path}: 'args' must be dict")
    return errors


def _validate_gadget(gadget: dict, path: str) -> list[str]:
    """Validate a gadget dict. Returns list of error strings."""
    errors = []
    for key in ("top_instructions", "bottom_instructions", "unified_instructions"):
        if key not in gadget:
            errors.append(f"{path}: missing '{key}'")
        elif not isinstance(gadget[key], list):
            errors.append(f"{path}: '{key}' must be list")
        else:
            for i, inst in enumerate(gadget[key]):
                errors.extend(_validate_instruction(inst, f"{path}.{key}[{i}]"))

    for key in ("top_output_index", "bottom_output_index"):
        if key not in gadget:
            errors.append(f"{path}: missing '{key}'")
        elif not isinstance(gadget[key], int):
            errors.append(f"{path}: '{key}' must be int")

    # unified_instructions must be <= sum of top + bottom
    if (
        "unified_instructions" in gadget
        and "top_instructions" in gadget
        and "bottom_instructions" in gadget
        and isinstance(gadget["unified_instructions"], list)
    ):
        n_unified = len(gadget["unified_instructions"])
        n_raw = len(gadget["top_instructions"]) + len(gadget["bottom_instructions"])
        if n_unified > n_raw:
            errors.append(f"{path}: unified ({n_unified}) > top+bottom ({n_raw})")

    return errors


def _validate_fixture(data: dict) -> list[str]:
    """Validate a full solution JSON fixture. Returns list of error strings."""
    errors = []

    # Top-level keys
    for key in ("roots", "nodes", "vector_machine", "primitive_type", "num_vecs"):
        if key not in data:
            errors.append(f"missing top-level key '{key}'")

    if not isinstance(data.get("roots"), list):
        errors.append("'roots' must be a list")
    if not isinstance(data.get("nodes"), dict):
        errors.append("'nodes' must be a dict")
        return errors  # can't continue without nodes

    # Validate each node
    for node_id, node in data["nodes"].items():
        npath = f"nodes.{node_id}"
        for key in ("stage", "input_state", "output_state", "gadgets", "children"):
            if key not in node:
                errors.append(f"{npath}: missing '{key}'")

        if not isinstance(node.get("gadgets"), list):
            errors.append(f"{npath}: 'gadgets' must be list")
        else:
            for i, gadget in enumerate(node["gadgets"]):
                errors.extend(_validate_gadget(gadget, f"{npath}.gadgets[{i}]"))

        # Children must reference existing nodes
        for child_id in node.get("children", []):
            if child_id not in data["nodes"]:
                errors.append(f"{npath}: child '{child_id}' not in nodes")

    # Roots must reference existing nodes
    for root_id in data.get("roots", []):
        if root_id not in data["nodes"]:
            errors.append(f"root '{root_id}' not in nodes")

    return errors


@pytest.mark.parametrize(
    "fixture_path",
    sorted(glob.glob(os.path.join(FIXTURE_DIR, "fixture_*.json"))),
    ids=lambda p: os.path.basename(p),
)
def test_fixture_schema_validation(fixture_path):
    """All JSON fixtures must conform to the solution schema."""
    with open(fixture_path) as f:
        data = json.load(f)
    errors = _validate_fixture(data)
    assert errors == [], (
        f"Schema errors in {os.path.basename(fixture_path)}:\n" + "\n".join(errors)
    )


@pytest.mark.parametrize(
    "fixture_path",
    sorted(glob.glob(os.path.join(FIXTURE_DIR, "fixture_*.json"))),
    ids=lambda p: os.path.basename(p),
)
def test_fixture_unified_instructions_consistency(fixture_path):
    """unified_instructions in fixtures must match recomputed values."""
    from bitonic_super_optimizer import InstructionSpec, PermutationGadget

    with open(fixture_path) as f:
        data = json.load(f)

    for node_id, node in data["nodes"].items():
        for gi, gd in enumerate(node["gadgets"]):
            top_insts = [
                InstructionSpec(i["name"], dict(i["args"]))
                for i in gd["top_instructions"]
            ]
            bottom_insts = [
                InstructionSpec(i["name"], dict(i["args"]))
                for i in gd["bottom_instructions"]
            ]
            gadget = PermutationGadget(
                top_instructions=top_insts,
                bottom_instructions=bottom_insts,
                validated=True,
            )
            unified, top_out, bottom_out = gadget.unified_instructions()

            # Check counts match
            assert len(gd["unified_instructions"]) == len(unified), (
                f"{node_id}.gadgets[{gi}]: unified count mismatch: "
                f"fixture={len(gd['unified_instructions'])}, "
                f"recomputed={len(unified)}"
            )
            assert (
                gd["top_output_index"] == top_out
            ), f"{node_id}.gadgets[{gi}]: top_output_index mismatch"
            assert (
                gd["bottom_output_index"] == bottom_out
            ), f"{node_id}.gadgets[{gi}]: bottom_output_index mismatch"
