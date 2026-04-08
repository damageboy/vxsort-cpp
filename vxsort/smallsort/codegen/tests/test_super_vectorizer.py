#!/usr/bin/env python3
"""Tests for the BitonicSuperVectorizer."""

import sys

import pytest
from bitonic_sorter import BitonicSorter
from asm_exporter import (
    RegisterAllocator,
    _emit_compare_swap_lines,
)
from bitonic_super_optimizer import (
    BitonicSuperVectorizer,
    GadgetSynthesizer,
    InstructionSpec,
    PermutationGadget,
    VectorState,
    graph_max_depth,
)
from bitonic_types import (
    GadgetGraph,
    InputRef,
    IntrinsicNode,
    Mux,
    Symbolic,
)
from gadget_synthesizer import _node_depth
from functional import seq
from util.enums import primitive_type, vector_machine
from z3 import And, If, Int, Not, Solver, unsat


def test_bitonic_sorter():
    """Test that BitonicSorter generates correct comparison stages."""

    def sorted_list(x):
        """Check if a list of z3 variables is sorted."""
        return And([x[i] <= x[i + 1] for i in range(len(x) - 1)])

    def compare_and_swap_z3(x, y):
        """z3 symbolic implementation of compare-and-swap (min, max)."""
        return If(x < y, x, y), If(x < y, y, x), x < y

    def verify_network(pairs_to_compare, N: int):
        """Verify that a given sorting network (list of pairs) correctly sorts N elements."""
        s = Solver()

        # Initial array of symbolic variables
        a = [Int(f"x_{i}") for i in range(N)]

        # Apply all comparison stages (pairs are 1-based, so subtract 1 for 0-based array access)
        for i, j in pairs_to_compare:
            x1, y1, _ = compare_and_swap_z3(a[i - 1], a[j - 1])
            a[i - 1] = x1
            a[j - 1] = y1

        # We want to prove that for all inputs, the result is sorted.
        # So we look for a counter-example: an input where the result is NOT sorted.
        s.add(Not(sorted_list(a)))

        # If unsat, then no counter-example exists, so the network is correct.
        return s.check() == unsat

    N = 16
    sorter = BitonicSorter(N)

    flat_pairs = list(
        seq(sorted(sorter.stages)).flat_map(lambda sid: sorter.stages[sid])
    )

    assert verify_network(
        flat_pairs, N
    ), f"Bitonic sorting network for N={N} failed verification!"


def test_vector_state():
    """Test VectorState creation and manipulation."""
    print("Testing VectorState...")

    state = VectorState(
        top=[0, 1, 2, 3, 4, 5, 6, 7], bottom=[8, 9, 10, 11, 12, 13, 14, 15]
    )

    print(f"  Initial state: {state}")

    state_copy = state.copy()
    assert state_copy.top == state.top
    assert state_copy.bottom == state.bottom
    assert state_copy is not state

    print("✓ VectorState test passed\n")


def test_gadget_synthesizer_init():
    """Test GadgetSynthesizer initialization."""
    print("Testing GadgetSynthesizer initialization...")

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)

    print(f"  Elements per vector: {synthesizer.elements_per_vector}")
    print(f"  Available intrinsics: {len(synthesizer.available_intrinsics)}")

    for name in sorted(synthesizer.available_intrinsics.keys()):
        print(f"    - {name}")

    assert (
        synthesizer.elements_per_vector == 8
    ), "AVX2 i32 should have 8 elements per vector"
    assert len(synthesizer.available_intrinsics) > 0, "Should have available intrinsics"

    print("✓ GadgetSynthesizer initialization test passed\n")


def test_bitonicsupervectorizer_init():
    """Test BitonicSuperVectorizer initialization."""
    print("Testing BitonicSuperVectorizer initialization...")

    super_opt = BitonicSuperVectorizer(2, primitive_type.i32, vector_machine.AVX2)

    print(f"  Total elements: {super_opt.total_elements}")
    print(f"  Elements per vector: {super_opt.elements_per_vector}")
    print(f"  Number of stages: {len(super_opt.bitonic_sorter.stages)}")

    assert super_opt.total_elements == 16, "Should have 16 total elements"
    assert super_opt.elements_per_vector == 8, "Should have 8 elements per vector"

    initial_state = super_opt._create_initial_state()
    print(f"  Initial state: {initial_state}")

    assert len(initial_state.top) == 8, "Top should have 8 elements"
    assert len(initial_state.bottom) == 8, "Bottom should have 8 elements"

    print("✓ BitonicSuperVectorizer initialization test passed\n")


def test_instruction_enumeration():
    """Test instruction enumeration."""
    print("Testing instruction enumeration...")

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)

    # Test single input instructions
    single_insts = synthesizer._enumerate_single_input_intrinsics(InputRef("test_reg"))
    print(f"  Single input instructions: {len(single_insts)}")
    for inst in single_insts[:5]:  # Show first 5
        print(f"    - {inst}")

    # Test dual input instructions
    dual_insts = synthesizer._enumerate_dual_input_intrinsics(
        InputRef("reg1"), InputRef("reg2")
    )
    print(f"  Dual input instructions: {len(dual_insts)}")
    for inst in dual_insts[:5]:  # Show first 5
        print(f"    - {inst}")

    assert len(single_insts) > 0, "Should have single input instructions"
    assert len(dual_insts) > 0, "Should have dual input instructions"

    print("✓ Instruction enumeration test passed\n")


def test_generate_candidate_graphs_order_and_counts():
    """Candidate graph generation produces expected counts and structure."""
    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i64)

    top_depth = 2
    bottom_depth = 1
    candidates = synthesizer._generate_candidate_graphs_at_depth(
        top_depth, bottom_depth
    )

    # Top depth-2 graphs: (single,single) + (dual,single) + (single,dual)
    # For dual→single (Shape D), asymmetric duals generate both orderings.
    n_single_top = len(synthesizer.single_intrinsics_top)
    n_dual = len(synthesizer.dual_intrinsics)
    n_asym = sum(1 for d in synthesizer.dual_intrinsics if not d.isomorphic_order)
    expected_top_count = (
        n_single_top * n_single_top  # single → single
        + (n_dual + n_asym) * n_single_top  # dual → single (with swapped variants)
        + n_single_top * n_dual  # single → dual
    )

    # Bottom depth-1 graphs: single + dual + asymmetric swapped variants
    n_single_bottom = len(synthesizer.single_intrinsics_bottom)
    expected_bottom_count = n_single_bottom + n_dual + n_asym

    expected_total = expected_top_count * expected_bottom_count
    assert (
        len(candidates) == expected_total
    ), f"Expected {expected_total} candidates, got {len(candidates)}"

    # All candidates should be GadgetGraph instances
    for g in candidates:
        assert isinstance(g, GadgetGraph)
        assert isinstance(g.top, IntrinsicNode)  # depth-2 top is never None
        assert isinstance(g.bottom, IntrinsicNode)  # depth-1 bottom is never None

    # Strict (0, 0) identity graph is excluded from synthesis candidates.
    identity_candidates = synthesizer._generate_candidate_graphs_at_depth(0, 0)
    assert identity_candidates == []


def test_output_state_computation():
    """Test computing output state after applying a gadget."""
    print("Testing output state computation...")

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)

    # Test case: identity gadget (no instructions) should preserve state
    input_state = VectorState(
        top=[0, 1, 2, 3, 4, 5, 6, 7], bottom=[8, 9, 10, 11, 12, 13, 14, 15]
    )
    identity_gadget = PermutationGadget(
        top_instructions=[], bottom_instructions=[], validated=True
    )

    output_state = synthesizer.compute_output_state(input_state, identity_gadget)
    print("  Identity gadget:")
    print(f"    Input:  {input_state}")
    print(f"    Output: {output_state}")

    assert output_state.top == input_state.top, "Identity should preserve top"
    assert output_state.bottom == input_state.bottom, "Identity should preserve bottom"

    print("✓ Output state computation test passed\n")


@pytest.mark.parametrize("vm", [vector_machine.AVX2, vector_machine.AVX512])
@pytest.mark.parametrize("dt", [primitive_type.i32, primitive_type.i64])
def test_first_stage_initial_state_matches_pairs(vm, dt):
    """Test that the initial state is constructed to match first stage pairs (no Z3)."""
    super_opt = BitonicSuperVectorizer(2, dt, vm)

    initial_state = super_opt._create_initial_state()
    first_stage_pairs = super_opt.bitonic_sorter.stages[0]

    for i, (a, b) in enumerate(first_stage_pairs):
        assert initial_state.top[i] == a, f"Top element at index {i} should be {
            a
        }, got {initial_state.top[i]}"
        assert initial_state.bottom[i] == b, f"Bottom element at index {i} should be {
            b
        }, got {initial_state.bottom[i]}"


@pytest.mark.parametrize(
    "vm, dt",
    [
        (vector_machine.AVX2, primitive_type.i32),
        (vector_machine.AVX2, primitive_type.i64),
        pytest.param(
            vector_machine.AVX512,
            primitive_type.i64,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            vector_machine.AVX512,
            primitive_type.i32,
            marks=pytest.mark.slow,
        ),
    ],
)
def test_first_stage_excludes_zero_instruction_gadgets(vm, dt):
    """Synthesis should not produce strict 0-instruction gadgets."""
    super_opt = BitonicSuperVectorizer(2, dt, vm)

    solutions, _ = super_opt.build_solution_tree(
        depth_limit=1, gadget_depth=1, natural_order=False, max_unique_outputs=1
    )

    assert len(solutions) > 0, "Should find at least one valid solution"
    assert all(
        g.instruction_count() > 0 for s in solutions for g in s.gadgets
    ), "Synthesis should exclude strict 0-instruction gadgets"


def test_natural_order_with_explicit_permute_nodes():
    """Natural-order stage can be solved with explicit permutation instructions."""
    print("Testing natural order stage with explicit permute nodes...")

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i64)
    n = synthesizer.elements_per_vector  # 4 for AVX2 i64

    input_state = VectorState(
        top=list(range(1, n + 1)),
        bottom=list(range(n + 1, 2 * n + 1)),
    )
    target_pairs = [(i + 1, n + i + 1) for i in range(n)]

    top_node = IntrinsicNode(
        "_mm256_permutexvar_epi64",
        {"a": InputRef("top"), "op_idx": Symbolic("ctrl_top_nat", 256)},
    )
    bottom_node = IntrinsicNode(
        "_mm256_permutexvar_epi64",
        {"a": InputRef("bottom"), "op_idx": Symbolic("ctrl_bot_nat", 256)},
    )
    graph = GadgetGraph(top=top_node, bottom=bottom_node)

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        graph,
        input_state=input_state,
        target_pairs=target_pairs,
        allow_any_lane_order=False,
    )

    assert len(results) >= 1, f"Expected >=1 result, got {len(results)}"
    gadget, output_state = results[0]
    assert gadget.instruction_count() > 0
    assert output_state.top == list(range(1, n + 1))
    assert output_state.bottom == list(range(n + 1, 2 * n + 1))
    print("  Output state:", output_state)
    print("✓ Natural order explicit-permute test passed\n")


def test_natural_order_strict_constraints():
    """Strict constraints pin pair[i] to lane i with exact top/bottom assignment."""
    print("Testing natural order strict constraints...")

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i64)
    n = synthesizer.elements_per_vector  # 4

    # Scrambled input: top=[3,1,4,2], bottom=[7,5,8,6]
    input_state = VectorState(top=[3, 1, 4, 2], bottom=[7, 5, 8, 6])
    target_pairs = [(i + 1, n + i + 1) for i in range(n)]

    # Build graph with IntrinsicNode templates and Symbolic immediates
    top_node = IntrinsicNode(
        "_mm256_permutexvar_epi64",
        {"a": InputRef("top"), "op_idx": Symbolic("ctrl_top", 256)},
    )
    bottom_node = IntrinsicNode(
        "_mm256_permutexvar_epi64",
        {"a": InputRef("bottom"), "op_idx": Symbolic("ctrl_bottom", 256)},
    )
    graph = GadgetGraph(top=top_node, bottom=bottom_node)

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        graph,
        input_state=input_state,
        target_pairs=target_pairs,
        max_solutions=1,
        allow_any_lane_order=False,
    )

    assert len(results) >= 1, "Should find at least one solution"
    gadget, output_state = results[0]
    assert output_state.top == [
        1,
        2,
        3,
        4,
    ], f"Expected [1,2,3,4], got {output_state.top}"
    assert output_state.bottom == [
        5,
        6,
        7,
        8,
    ], f"Expected [5,6,7,8], got {output_state.bottom}"
    print(f"  Found gadget: {gadget}")
    print(f"  Output state: {output_state}")
    print("✓ Natural order strict constraints test passed\n")


def test_natural_order_integration():
    """Verify natural_order=True injects a stage and sets metadata correctly."""
    print("Testing natural order integration (AVX2 i64)...")

    super_opt = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX2)
    n = super_opt.elements_per_vector  # 4
    num_bitonic_stages = len(super_opt.bitonic_sorter.stages)

    # Call build_solution_tree with natural_order=True but depth_limit=1
    # to verify the stage was injected without running the full tree
    _, _ = super_opt.build_solution_tree(
        natural_order=True, depth_limit=1, gadget_depth=1
    )

    # Verify the natural order stage was injected
    assert (
        super_opt._natural_order_stage is not None
    ), "Natural order stage should be set"
    assert super_opt._natural_order_stage == num_bitonic_stages, (
        f"Natural order stage should be {num_bitonic_stages}, "
        f"got {super_opt._natural_order_stage}"
    )

    # Verify the injected stage has the correct pairs
    nat_stage = super_opt._natural_order_stage
    expected_pairs = [(i + 1, n + i + 1) for i in range(n)]
    assert super_opt.bitonic_sorter.stages[nat_stage] == expected_pairs, (
        f"Expected pairs {expected_pairs}, "
        f"got {super_opt.bitonic_sorter.stages[nat_stage]}"
    )

    # Verify the total number of stages increased by 1
    assert len(super_opt.bitonic_sorter.stages) == num_bitonic_stages + 1, (
        f"Expected {num_bitonic_stages + 1} stages, "
        f"got {len(super_opt.bitonic_sorter.stages)}"
    )

    print(f"  Injected natural order stage {nat_stage} with pairs {expected_pairs}")

    # Now test the natural order stage directly with a known input
    # Simulate a post-sort scrambled state and synthesize just the natural order gadget
    scrambled_input = VectorState(top=[3, 1, 4, 2], bottom=[7, 5, 8, 6])
    synthesizer = super_opt.synthesizer

    top_node = IntrinsicNode(
        "_mm256_permutexvar_epi64",
        {"a": InputRef("top"), "op_idx": Symbolic("ctrl_top", 256)},
    )
    bottom_node = IntrinsicNode(
        "_mm256_permutexvar_epi64",
        {"a": InputRef("bottom"), "op_idx": Symbolic("ctrl_bottom", 256)},
    )
    graph = GadgetGraph(top=top_node, bottom=bottom_node)

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        graph,
        input_state=scrambled_input,
        target_pairs=expected_pairs,
        max_solutions=1,
        allow_any_lane_order=False,
    )

    assert len(results) >= 1, "Should find at least one solution for natural order"
    _, output_state = results[0]
    assert output_state.top == [
        1,
        2,
        3,
        4,
    ], f"Expected [1,2,3,4], got {output_state.top}"
    assert output_state.bottom == [
        5,
        6,
        7,
        8,
    ], f"Expected [5,6,7,8], got {output_state.bottom}"

    print(f"  Natural order gadget output: {output_state}")
    print("✓ Natural order integration test passed\n")


def test_mux_encoding_chains_instructions():
    """Test that depth-2 single→single gadgets hardwire the chain correctly.

    Creates a 2-instruction graph [permute_ps → permute_ps] on the top side
    with an identity bottom. The second instruction's register operand is
    hardwired to the first instruction's output (no mux — selecting top/bottom
    would degenerate to depth-1).

    We verify that all results have inst2 with args["a"] == "prev",
    proving the hardwired chain from inst1.
    """
    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)
    n = synthesizer.elements_per_vector  # 8 for AVX2 i32

    # Get target pairs from the first bitonic sort stage (16-element network)
    sorter = BitonicSorter(2 * n)
    target_pairs = sorter.stages[0]

    # Initial input state for stage 0: top gets first elements of each pair,
    # bottom gets second elements. This matches how BitonicSuperVectorizer
    # constructs the initial state from the first stage pairs.
    input_state = VectorState(
        top=[p[0] for p in target_pairs],
        bottom=[p[1] for p in target_pairs],
    )

    # Build a depth-2 graph: permute_ps → permute_ps for top side (hardwired).
    # The first instruction operates on "top" directly.
    # The second instruction's "a" operand is hardwired to inst1's output.
    inst1_node = IntrinsicNode(
        "_mm256_permute_ps",
        {
            "a": InputRef("top"),
            "imm8": Symbolic("imm8_permute_ps_inst1", 8),
        },
    )
    inst2_node = IntrinsicNode(
        "_mm256_permute_ps",
        {
            "a": inst1_node,
            "imm8": Symbolic("imm8_permute_ps_inst2", 8),
        },
    )
    graph = GadgetGraph(top=inst2_node, bottom=None)

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        graph,
        input_state=input_state,
        target_pairs=target_pairs,
        max_solutions=1,
        max_unique_outputs=3,
    )

    assert len(results) > 0, "Should find at least one valid gadget"

    # All returned gadgets must be validated
    for gadget, output_state in results:
        assert gadget.validated, f"Gadget should be validated: {gadget}"

    # All results must have inst2 with args["a"] == "prev" (hardwired chain)
    for gadget, output_state in results:
        assert len(gadget.top_instructions) == 2
        assert (
            gadget.top_instructions[1].args["a"] == "prev"
        ), f"Expected hardwired 'prev', got '{gadget.top_instructions[1].args['a']}'"


def test_single_dual_template_combination():
    """Test shape E: (single, dual) 2-instruction graph with constrained mux.

    Creates a depth-2 gadget with a single-input instruction (permute_ps)
    followed by a dual-input instruction (shuffle_ps). The mux encoding
    lets Z3 decide whether inst2's "a" and "b" operands read from top,
    bottom, or inst1's output, but at least one must use inst1's output
    to avoid degenerating to a depth-1 gadget.
    """
    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)
    n = synthesizer.elements_per_vector

    sorter = BitonicSorter(2 * n)
    target_pairs = sorter.stages[0]

    input_state = VectorState(
        top=[p[0] for p in target_pairs],
        bottom=[p[1] for p in target_pairs],
    )

    top_ref = InputRef("top")
    bottom_ref = InputRef("bottom")

    # First instruction: single-input permute on top
    inst1_node = IntrinsicNode(
        "_mm256_permute_ps",
        {
            "a": top_ref,
            "imm8": Symbolic("imm8_permute_ps_sd_test", 8),
        },
    )

    # Second instruction: dual-input shuffle with muxed operands
    # Mux over {top, bottom, inst1_node} with at_least_one_last pruning
    mux_a = Mux(
        select=Symbolic("sel_a_shuffle_ps_sd_test", 2),
        sources=(top_ref, bottom_ref, inst1_node),
        pruning="at_least_one_last",
    )
    mux_b = Mux(
        select=Symbolic("sel_b_shuffle_ps_sd_test", 2),
        sources=(top_ref, bottom_ref, inst1_node),
        pruning="at_least_one_last",
    )
    inst2_node = IntrinsicNode(
        "_mm256_shuffle_ps",
        {
            "a": mux_a,
            "b": mux_b,
            "imm8": Symbolic("imm8_shuffle_ps_sd_test", 8),
        },
    )

    # Bottom: single-input permute on bottom
    bottom_node = IntrinsicNode(
        "_mm256_permute_ps",
        {
            "a": bottom_ref,
            "imm8": Symbolic("imm8_permute_ps_sd_bottom", 8),
        },
    )

    graph = GadgetGraph(top=inst2_node, bottom=bottom_node)

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        graph,
        input_state=input_state,
        target_pairs=target_pairs,
        max_solutions=1,
        max_unique_outputs=3,
    )

    assert (
        len(results) >= 1
    ), "Should find at least one gadget for (single, dual) template"

    # Check that at least one result has inst2 reading from "prev"
    found_prev = False
    for gadget, output_state in results:
        assert gadget.validated
        inst2_concrete = gadget.top_instructions[1]
        a_arg = inst2_concrete.args.get("a")
        b_arg = inst2_concrete.args.get("b")
        if a_arg == "prev" or b_arg == "prev":
            found_prev = True

    assert found_prev, (
        "At least one result should have inst2 reading from 'prev'. "
        f"Got: {[(g.top_instructions[1].args.get('a'), g.top_instructions[1].args.get('b')) for g, _ in results]}"
    )


def test_asm_exporter_register_mapping():
    """Test that the canonical emitter resolves register names correctly.

    Verifies absolute (top_reg, bottom_reg, is_top) register mapping
    via _resolve_source, which is used by both the OSACA emitter and
    the asm export path.
    """
    from asm_exporter import _resolve_source

    # "bottom" on bottom side → ymm1
    assert _resolve_source("bottom", "ymm0", "ymm1", is_top=False) == "ymm1"
    # "top" on bottom side → ymm0
    assert _resolve_source("top", "ymm0", "ymm1", is_top=False) == "ymm0"
    # "prev" on top side → ymm0 (dest = top for top chain)
    assert _resolve_source("prev", "ymm0", "ymm1", is_top=True) == "ymm0"
    # "prev" on bottom side → ymm1 (dest = bottom for bottom chain)
    assert _resolve_source("prev", "ymm0", "ymm1", is_top=False) == "ymm1"


def test_bitonicsupervectorizer_is_single_use():
    """A BitonicSuperVectorizer instance should reject a second synthesis run."""
    super_opt = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX2)
    roots, _ = super_opt.build_solution_tree(
        depth_limit=1, gadget_depth=1, natural_order=False, max_unique_outputs=1
    )
    assert len(roots) > 0

    with pytest.raises(RuntimeError, match="single-use"):
        super_opt.build_solution_tree(
            depth_limit=1, gadget_depth=1, natural_order=False, max_unique_outputs=1
        )


def test_avx512_i32_synthesis_depth1():
    """Test that AVX512 i32 synthesis infrastructure works.

    Full depth-2 synthesis is too slow for CI (~4 min for 16-element vectors).
    """
    super_opt = BitonicSuperVectorizer(2, primitive_type.i32, vector_machine.AVX512)
    assert super_opt.elements_per_vector == 16, "AVX512 i32 should have 16 elements"
    assert super_opt.total_elements == 32, "Should have 32 total elements"
    # Verify intrinsics are registered
    assert len(super_opt.synthesizer.available_intrinsics) > 0
    # Verify single and dual templates enumerate without error
    single = super_opt.synthesizer._enumerate_single_input_intrinsics(InputRef("test"))
    dual = super_opt.synthesizer._enumerate_dual_input_intrinsics(
        InputRef("top"), InputRef("bottom")
    )
    assert len(single) == 6, f"Expected 6 single-input templates, got {len(single)}"
    assert len(dual) == 12, f"Expected 12 dual-input templates, got {len(dual)}"
    print(f"AVX512 i32: {len(single)} single + {len(dual)} dual templates registered")


def test_register_allocator_pair_alternation():
    """Verify src/dst properties, swap behavior, and temp start at num_vecs*2."""
    ra = RegisterAllocator(vector_machine.AVX2, primitive_type.i32, 2)

    # Initial state: pair 0 is source
    assert ra.src_top == "ymm0"
    assert ra.src_bottom == "ymm1"
    assert ra.dst_top == "ymm2"
    assert ra.dst_bottom == "ymm3"
    assert ra._pair_index == 0

    # Temp registers start at num_vecs * 2 = 4
    tmp = ra.allocate_temp()
    assert tmp == "ymm4"
    ra.reset_temps()

    # After swap, pair 1 is source
    ra.swap_pairs()
    assert ra._pair_index == 1
    assert ra.src_top == "ymm2"
    assert ra.src_bottom == "ymm3"
    assert ra.dst_top == "ymm0"
    assert ra.dst_bottom == "ymm1"

    # Swap back
    ra.swap_pairs()
    assert ra._pair_index == 0
    assert ra.src_top == "ymm0"

    # reset_temps preserves pair state
    ra.swap_pairs()
    ra.reset_temps()
    assert ra._pair_index == 1  # preserved


def test_compare_swap_lines_native_no_copy():
    """AVX2 i32: exactly 2 instructions, no vmovdqa, writes to dst pair."""
    ra = RegisterAllocator(vector_machine.AVX2, primitive_type.i32, 2)
    lines = _emit_compare_swap_lines(ra)

    assert len(lines) == 2, f"Expected 2 instructions, got {len(lines)}"
    # No vmovdqa copy
    for line in lines:
        assert "vmovdqa" not in line, f"Unexpected vmovdqa in: {line}"
    # First line is min writing to dst_top (ymm2)
    assert "vpminsd" in lines[0]
    assert "ymm2" in lines[0]  # dst_top
    # Second line is max writing to dst_bottom (ymm3)
    assert "vpmaxsd" in lines[1]
    assert "ymm3" in lines[1]  # dst_bottom
    # Both read from source pair
    assert "ymm0" in lines[0] and "ymm1" in lines[0]
    assert "ymm0" in lines[1] and "ymm1" in lines[1]


def test_compare_swap_lines_emulated_no_copy():
    """AVX2 i64: exactly 3 instructions (vpcmpgtq + 2x vblendvpd), no vmovdqa."""
    ra = RegisterAllocator(vector_machine.AVX2, primitive_type.i64, 2)
    lines = _emit_compare_swap_lines(ra)

    assert len(lines) == 3, f"Expected 3 instructions, got {len(lines)}"
    # No vmovdqa copy
    for line in lines:
        assert "vmovdqa" not in line, f"Unexpected vmovdqa in: {line}"
    assert "vpcmpgtq" in lines[0]
    assert "vblendvpd" in lines[1]
    assert "vblendvpd" in lines[2]
    # Writes to dst pair (ymm2, ymm3)
    assert "ymm2" in lines[1]  # dst_top in blend
    assert "ymm3" in lines[2]  # dst_bottom in blend


def test_pair_swap_across_stages():
    """Simulate 3 stages: verify pairs alternate correctly."""
    ra = RegisterAllocator(vector_machine.AVX2, primitive_type.i32, 2)

    # Stage 0: source = pair 0
    assert ra.src_top == "ymm0"
    lines0 = _emit_compare_swap_lines(ra)
    assert "ymm2" in lines0[0]  # writes to dst pair
    ra.swap_pairs()
    ra.reset_temps()

    # Stage 1: source = pair 1
    assert ra.src_top == "ymm2"
    assert ra.src_bottom == "ymm3"
    lines1 = _emit_compare_swap_lines(ra)
    assert "ymm0" in lines1[0]  # writes back to pair 0
    ra.swap_pairs()
    ra.reset_temps()

    # Stage 2: source = pair 0 again
    assert ra.src_top == "ymm0"
    assert ra._pair_index == 0


def test_expanded_mux_discovers_result_0_reference():
    """Test that the expanded mux allows inst3 to reference result_0 (not just prev).

    Creates a 3-instruction graph [permutexvar, permutexvar, shuffle_pd] on the
    top side. For the 3rd instruction, the mux offers {top, bottom, inst1, inst2}.
    If Z3 discovers a solution where inst3 references inst1 (result_0),
    the extracted gadget will contain "result_0" in its args.
    """
    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i64)
    n = synthesizer.elements_per_vector  # 4

    sorter = BitonicSorter(2 * n)
    target_pairs = sorter.stages[0]

    input_state = VectorState(
        top=[p[0] for p in target_pairs],
        bottom=[p[1] for p in target_pairs],
    )

    top_ref = InputRef("top")
    bottom_ref = InputRef("bottom")

    # 3-instruction graph: [permutexvar, permutexvar, shuffle_pd]
    # inst1 and inst2 operate on "top" directly.
    # inst3 (shuffle_pd) gets muxed operands that can choose from
    # {top, bottom, inst1, inst2}.
    inst1_node = IntrinsicNode(
        "_mm256_permutexvar_epi64",
        {"a": top_ref, "op_idx": Symbolic("ctrl_inst1", 256)},
    )
    inst2_node = IntrinsicNode(
        "_mm256_permutexvar_epi64",
        {"a": top_ref, "op_idx": Symbolic("ctrl_inst2", 256)},
    )

    # Muxes for inst3's register operands: {top, bottom, inst1, inst2}
    mux_a = Mux(
        select=Symbolic("sel_a_shuffle_pd_inst3", 2),
        sources=(top_ref, bottom_ref, inst1_node, inst2_node),
        pruning="at_least_one_last",
    )
    mux_b = Mux(
        select=Symbolic("sel_b_shuffle_pd_inst3", 2),
        sources=(top_ref, bottom_ref, inst1_node, inst2_node),
        pruning="at_least_one_last",
    )
    inst3_node = IntrinsicNode(
        "_mm256_shuffle_pd",
        {
            "a": mux_a,
            "b": mux_b,
            "imm8": Symbolic("imm8_inst3", 8),
        },
    )

    graph = GadgetGraph(top=inst3_node, bottom=None)

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        graph,
        input_state=input_state,
        target_pairs=target_pairs,
        max_solutions=1,
        max_unique_outputs=3,
    )

    assert len(results) > 0, "Should find at least one valid gadget"

    # Verify all returned gadgets are validated and have resolved source names
    for gadget, output_state in results:
        assert gadget.validated
        if len(gadget.top_instructions) == 3:
            inst3 = gadget.top_instructions[2]
            a_arg = inst3.args.get("a")
            b_arg = inst3.args.get("b")
            # For inst_idx=2: "prev" aliases result_1, "result_0" is earlier
            for arg_val in [a_arg, b_arg]:
                if isinstance(arg_val, str):
                    assert arg_val in ("top", "bottom", "prev", "result_0"), (
                        f"Unexpected source: {arg_val}. "
                        "Expected one of: top, bottom, prev, result_0"
                    )


def test_unified_instruction_deduplication():
    """Test that unified_instructions deduplicates shared instructions across sides."""
    # Create a gadget where top and bottom share the first two instructions
    inst_a = InstructionSpec("_mm256_permutexvar_epi64", {"a": "top", "op_idx": 42})
    inst_b = InstructionSpec("_mm256_permute_pd", {"a": "prev", "imm8": 0x05})
    inst_c = InstructionSpec(
        "_mm256_shuffle_pd", {"a": "result_0", "b": "prev", "imm8": 0x88}
    )
    inst_d = InstructionSpec(
        "_mm256_shuffle_pd", {"a": "result_0", "b": "prev", "imm8": 0xDD}
    )

    gadget = PermutationGadget(
        top_instructions=[inst_a, inst_b, inst_c],
        bottom_instructions=[inst_a, inst_b, inst_d],
        validated=True,
    )

    unified, top_idx, bottom_idx = gadget.unified_instructions()

    # inst_a and inst_b are shared, inst_c and inst_d are unique → 4 instructions
    assert (
        len(unified) == 4
    ), f"Expected 4 deduplicated instructions, got {len(unified)}"
    assert top_idx != bottom_idx, "Top and bottom outputs should have different indices"
    # instruction_count should return the deduplicated count
    assert (
        gadget.instruction_count() == 4
    ), f"Expected instruction_count() == 4, got {gadget.instruction_count()}"


def test_unified_no_sharing():
    """Test unified_instructions with no shared instructions."""
    inst_a = InstructionSpec("_mm256_permutexvar_epi64", {"a": "top", "op_idx": 42})
    inst_b = InstructionSpec("_mm256_permutexvar_epi64", {"a": "bottom", "op_idx": 99})

    gadget = PermutationGadget(
        top_instructions=[inst_a],
        bottom_instructions=[inst_b],
        validated=True,
    )

    unified, top_idx, bottom_idx = gadget.unified_instructions()
    assert len(unified) == 2, f"Expected 2 instructions, got {len(unified)}"
    assert top_idx == 0
    assert bottom_idx == 1


def test_cost_model_counts_deduplicated():
    """Verify that calculate_gadget_cost returns cost of deduplicated instructions."""
    from cost_model import CostModel

    cost_model = CostModel("generic")

    # Shared instruction
    inst_shared = InstructionSpec(
        "_mm256_permutexvar_epi64", {"a": "top", "op_idx": 42}
    )
    inst_top_only = InstructionSpec(
        "_mm256_shuffle_pd", {"a": "prev", "b": "bottom", "imm8": 0x88}
    )
    inst_bot_only = InstructionSpec(
        "_mm256_shuffle_pd", {"a": "prev", "b": "bottom", "imm8": 0xDD}
    )

    gadget = PermutationGadget(
        top_instructions=[inst_shared, inst_top_only],
        bottom_instructions=[inst_shared, inst_bot_only],
        validated=True,
    )

    cost = cost_model.calculate_gadget_cost(gadget)
    # Should be 3 instructions (shared + top_only + bot_only), not 4
    unified, _, _ = gadget.unified_instructions()
    assert (
        len(unified) == 3
    ), f"Expected 3 deduplicated instructions, got {len(unified)}"

    expected = sum(
        cost_model.get_instruction_cost(inst.intrinsic_name).latency for inst in unified
    )
    assert cost == expected, f"Cost {cost} != expected {expected}"


def test_intrinsic_filter_restricts_enumeration():
    """intrinsic_filter limits which instructions are enumerated."""
    only_two = {"_mm256_permutexvar_epi32", "_mm256_shuffle_ps"}
    synth = GadgetSynthesizer(
        vector_machine.AVX2,
        primitive_type.i32,
        intrinsic_filter=only_two,
    )

    all_names = set()
    for inst in synth.single_intrinsics_top:
        all_names.add(inst.name)
    for inst in synth.single_intrinsics_bottom:
        all_names.add(inst.name)
    for inst in synth.dual_intrinsics:
        all_names.add(inst.name)

    assert all_names == only_two, f"Expected only {only_two}, got {all_names}"


def test_intrinsic_filter_rejects_unknown():
    """intrinsic_filter raises ValueError for intrinsics not in the VM/type."""
    with pytest.raises(ValueError, match="not available"):
        GadgetSynthesizer(
            vector_machine.AVX2,
            primitive_type.i32,
            intrinsic_filter={"_mm512_permutexvar_epi64"},
        )


# ---------------------------------------------------------------------------
# Depth classification tests
# ---------------------------------------------------------------------------


def test_node_depth_none():
    """None node has depth 0."""
    assert _node_depth(None) == 0


def test_node_depth_leaf():
    """A single IntrinsicNode with no IntrinsicNode children has depth 1."""
    node = IntrinsicNode(
        "_mm256_permute_ps",
        {"a": InputRef("top"), "imm8": Symbolic("ctrl", 8)},
    )
    assert _node_depth(node) == 1


def test_node_depth_chain():
    """Two chained IntrinsicNodes have depth 2."""
    inner = IntrinsicNode(
        "_mm256_permute_ps",
        {"a": InputRef("top"), "imm8": Symbolic("ctrl1", 8)},
    )
    outer = IntrinsicNode(
        "_mm256_permute_ps",
        {"a": inner, "imm8": Symbolic("ctrl2", 8)},
    )
    assert _node_depth(outer) == 2


def test_node_depth_with_mux():
    """Mux containing an IntrinsicNode increases depth."""
    inner = IntrinsicNode(
        "_mm256_permute_ps",
        {"a": InputRef("top"), "imm8": Symbolic("ctrl1", 8)},
    )
    mux = Mux(
        select=Symbolic("sel", 2),
        sources=(InputRef("top"), InputRef("bottom"), inner),
    )
    outer = IntrinsicNode(
        "_mm256_shuffle_ps",
        {"a": mux, "b": InputRef("bottom"), "imm8": Symbolic("ctrl2", 8)},
    )
    assert _node_depth(outer) == 2


def test_graph_max_depth_identity():
    """Identity graph (both None) has depth 0."""
    g = GadgetGraph(top=None, bottom=None)
    assert graph_max_depth(g) == 0


def test_graph_max_depth_asymmetric():
    """Depth is the max of top and bottom."""
    inner = IntrinsicNode(
        "_mm256_permute_ps",
        {"a": InputRef("top"), "imm8": Symbolic("ctrl1", 8)},
    )
    outer = IntrinsicNode(
        "_mm256_permute_ps",
        {"a": inner, "imm8": Symbolic("ctrl2", 8)},
    )
    leaf = IntrinsicNode(
        "_mm256_permute_ps",
        {"a": InputRef("bottom"), "imm8": Symbolic("ctrl3", 8)},
    )
    g = GadgetGraph(top=outer, bottom=leaf)
    assert graph_max_depth(g) == 2


def test_stratified_candidates_counts():
    """Stratified split sums to total candidates, shallow has expected count."""
    synth = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i64)
    shallow, deep = synth.precompute_candidates_stratified(2)
    all_cands = synth.precompute_all_candidates(2)
    assert len(shallow) + len(deep) == len(all_cands)
    # All shallow candidates should have graph_max_depth <= 1
    for g in shallow:
        assert graph_max_depth(g) <= 1
    # All deep candidates should have graph_max_depth >= 2
    for g in deep:
        assert graph_max_depth(g) >= 2


def test_stratified_depth1_no_deep():
    """With gadget_depth=1, deep list should be empty."""
    synth = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i64)
    shallow, deep = synth.precompute_candidates_stratified(1)
    assert len(deep) == 0
    assert len(shallow) > 0


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running BitonicSuperVectorizer Tests")
    print("=" * 60 + "\n")

    try:
        test_bitonic_sorter()
        test_vector_state()
        test_gadget_synthesizer_init()
        test_bitonicsupervectorizer_init()
        test_instruction_enumeration()
        test_output_state_computation()
        test_first_stage_initial_state_matches_pairs()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
