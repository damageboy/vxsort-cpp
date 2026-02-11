#!/usr/bin/env python3
"""Tests for the BitonicSuperVectorizer."""

import pytest
import sys

from utils import vector_machine, primitive_type
from bitonic_sorter import BitonicSorter
from bitonic_super_optimizer import (
    GadgetSynthesizer,
    BitonicSuperVectorizer,
    PermutationGadget,
    VectorState,
)
from functional import seq
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

    assert verify_network(flat_pairs, N), (
        f"Bitonic sorting network for N={N} failed verification!"
    )


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

    assert synthesizer.elements_per_vector == 8, (
        "AVX2 i32 should have 8 elements per vector"
    )
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
    single_insts = synthesizer._enumerate_single_input_instructions("test_reg")
    print(f"  Single input instructions: {len(single_insts)}")
    for inst in single_insts[:5]:  # Show first 5
        print(f"    - {inst}")

    # Test dual input instructions
    dual_insts = synthesizer._enumerate_dual_input_instructions("reg1", "reg2")
    print(f"  Dual input instructions: {len(dual_insts)}")
    for inst in dual_insts[:5]:  # Show first 5
        print(f"    - {inst}")

    assert len(single_insts) > 0, "Should have single input instructions"
    assert len(dual_insts) > 0, "Should have dual input instructions"

    print("✓ Instruction enumeration test passed\n")


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


@pytest.mark.parametrize("vm", [vector_machine.AVX2])
@pytest.mark.parametrize("dt", [primitive_type.i32, primitive_type.i64])
# @pytest.mark.parametrize("vm", [vector_machine.AVX2, vector_machine.AVX512])
# @pytest.mark.parametrize("dt", [primitive_type.i16, primitive_type.i32, primitive_type.i64])
def test_first_stage_requires_no_permutation(vm, dt):
    """Test that the initial state is constructed to make first stage a null operation."""
    print(f"Testing first stage requires no permutation for {vm.name}, {dt.name}...")

    super_opt = BitonicSuperVectorizer(2, dt, vm)

    # Get initial state and first stage pairs
    initial_state = super_opt._create_initial_state()
    first_stage_pairs = super_opt.bitonic_sorter.stages[0]

    print(f"  First stage pairs: {first_stage_pairs}")
    print(f"Initial state: {initial_state}")

    # Verify that initial state matches first stage pairs
    for i, (a, b) in enumerate(first_stage_pairs):
        assert initial_state.top[i] == a, (
            f"Top element at index {i} should be {a}, got {initial_state.top[i]}"
        )
        assert initial_state.bottom[i] == b, (
            f"Bottom element at index {i} should be {b}, got {initial_state.bottom[i]}"
        )

    # Build solution tree for just the first stage
    solutions, _ = super_opt.build_solution_tree(depth_limit=1)

    print(f"  Found {len(solutions)} valid solution tree root(s)")

    # Verify that the first gadget is a null (0-instruction) gadget
    assert len(solutions) > 0, "Should find at least one valid solution"
    # Find solutions that have at least one gadget with 0 instructions
    null_solutions = [s for s in solutions if any(g.instruction_count() == 0 for g in s.gadgets)]
    assert len(null_solutions) > 0, (
        "Should find at least one solution with null (0-instruction) gadget for first stage"
    )

    # Get minimum instruction count from first solution's gadgets
    min_instructions = min(g.instruction_count() for g in solutions[0].gadgets)
    print(
        f"  ✓ First gadget requires {min_instructions} instructions (as expected)"
    )
    print("✓ First stage null permutation test passed\n")


def test_natural_order_identity():
    """When input state IS already natural order, the stage should find a 0-instruction gadget."""
    print("Testing natural order identity (already in natural order)...")

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i64)
    n = synthesizer.elements_per_vector  # 4 for AVX2 i64

    # Input is already in natural order
    input_state = VectorState(
        top=list(range(1, n + 1)),
        bottom=list(range(n + 1, 2 * n + 1)),
    )
    # Target pairs: (1, 5), (2, 6), (3, 7), (4, 8)
    target_pairs = [(i + 1, n + i + 1) for i in range(n)]

    # Empty gadget (0 instructions) should satisfy strict constraints
    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        top_instructions_template=[],
        bottom_instructions_template=[],
        input_state=input_state,
        target_pairs=target_pairs,
        allow_any_lane_order=False,
    )

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    gadget, output_state = results[0]
    assert gadget.instruction_count() == 0, "Should be a 0-instruction gadget"
    assert output_state.top == list(range(1, n + 1))
    assert output_state.bottom == list(range(n + 1, 2 * n + 1))
    print("  Output state:", output_state)
    print("✓ Natural order identity test passed\n")


def test_natural_order_strict_constraints():
    """Strict constraints pin pair[i] to lane i with exact top/bottom assignment."""
    print("Testing natural order strict constraints...")

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i64)
    n = synthesizer.elements_per_vector  # 4

    # Scrambled input: top=[3,1,4,2], bottom=[7,5,8,6]
    input_state = VectorState(top=[3, 1, 4, 2], bottom=[7, 5, 8, 6])
    target_pairs = [(i + 1, n + i + 1) for i in range(n)]

    # Build InstructionSpec templates with symbolic placeholders
    from bitonic_super_optimizer import InstructionSpec, SymbolicPlaceholder

    top_template = [
        InstructionSpec(
            "_mm256_permutexvar_epi64",
            {"a": "top", "op_idx": SymbolicPlaceholder("ctrl_top", 256)},
        )
    ]
    bottom_template = [
        InstructionSpec(
            "_mm256_permutexvar_epi64",
            {"a": "bottom", "op_idx": SymbolicPlaceholder("ctrl_bottom", 256)},
        )
    ]

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        top_instructions_template=top_template,
        bottom_instructions_template=bottom_template,
        input_state=input_state,
        target_pairs=target_pairs,
        max_solutions=1,
        allow_any_lane_order=False,
    )

    assert len(results) >= 1, "Should find at least one solution"
    gadget, output_state = results[0]
    assert output_state.top == [1, 2, 3, 4], f"Expected [1,2,3,4], got {output_state.top}"
    assert output_state.bottom == [5, 6, 7, 8], f"Expected [5,6,7,8], got {output_state.bottom}"
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
        natural_order=True, depth_limit=1, gadget_depth=2
    )

    # Verify the natural order stage was injected
    assert super_opt._natural_order_stage is not None, (
        "Natural order stage should be set"
    )
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

    from bitonic_super_optimizer import InstructionSpec, SymbolicPlaceholder

    top_template = [
        InstructionSpec(
            "_mm256_permutexvar_epi64",
            {"a": "top", "op_idx": SymbolicPlaceholder("ctrl_top", 256)},
        )
    ]
    bottom_template = [
        InstructionSpec(
            "_mm256_permutexvar_epi64",
            {"a": "bottom", "op_idx": SymbolicPlaceholder("ctrl_bottom", 256)},
        )
    ]

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        top_instructions_template=top_template,
        bottom_instructions_template=bottom_template,
        input_state=scrambled_input,
        target_pairs=expected_pairs,
        max_solutions=1,
        allow_any_lane_order=False,
    )

    assert len(results) >= 1, "Should find at least one solution for natural order"
    _, output_state = results[0]
    assert output_state.top == [1, 2, 3, 4], f"Expected [1,2,3,4], got {output_state.top}"
    assert output_state.bottom == [5, 6, 7, 8], f"Expected [5,6,7,8], got {output_state.bottom}"

    print(f"  Natural order gadget output: {output_state}")
    print("✓ Natural order integration test passed\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running BitonicSuperVectorizer Tests")
    print("=" * 60 + "\n")

    try:
        test_bitonic_sorter()
        test_vector_state()
        test_gadget_synthesizer_init()
        test_pair_id_mapping()
        test_bitonicsupervectorizer_init()
        test_instruction_enumeration()
        test_output_state_computation()
        test_first_stage_requires_no_permutation()

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
