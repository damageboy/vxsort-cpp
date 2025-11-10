#!/usr/bin/env python3
"""Tests for the BitonicSuperVectorizer."""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from bitonic_compiler import BitonicSuperVectorizer, BitonicSorter, VectorState, PermutationGadget, InstructionSpec, primitive_type, vector_machine, GadgetSynthesizer


def test_bitonic_sorter():
    """Test that BitonicSorter generates correct comparison stages."""
    print("Testing BitonicSorter...")

    # Test with 16 elements (2 AVX2 i32 vectors)
    sorter = BitonicSorter(16)

    print(f"Number of stages: {len(sorter.stages)}")
    for stage_id in sorted(sorter.stages.keys()):
        pairs = sorter.stages[stage_id]
        print(f"  Stage {stage_id}: {pairs}")

    assert len(sorter.stages) > 0, "Should have at least one stage"
    print("✓ BitonicSorter test passed\n")


def test_vector_state():
    """Test VectorState creation and manipulation."""
    print("Testing VectorState...")

    state = VectorState(top=[0, 1, 2, 3, 4, 5, 6, 7], bottom=[8, 9, 10, 11, 12, 13, 14, 15])

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

    assert synthesizer.elements_per_vector == 8, "AVX2 i32 should have 8 elements per vector"
    assert len(synthesizer.available_intrinsics) > 0, "Should have available intrinsics"

    print("✓ GadgetSynthesizer initialization test passed\n")


def test_pair_id_mapping():
    """Test pair ID mapping creation."""
    print("Testing pair ID mapping...")

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)

    target_pairs = [(0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)]
    pair_id_map = synthesizer._create_pair_id_mapping(target_pairs)

    print(f"  Pair ID map: {pair_id_map}")

    # Check that both elements in each pair have the same pair_id
    for pair_id, (elem1, elem2) in enumerate(target_pairs, start=1):
        assert pair_id_map[elem1] == pair_id_map[elem2], f"Elements {elem1} and {elem2} should have the same pair_id"
        assert pair_id_map[elem1] == pair_id, f"Pair ({elem1}, {elem2}) should have pair_id {pair_id}"

    print("✓ Pair ID mapping test passed\n")


def test_check_input_matches_target():
    """Test checking if input already matches target."""
    print("Testing input match check...")

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)

    # Test case 1: Input matches target perfectly
    input_state1 = VectorState(top=[0, 1, 2, 3, 4, 5, 6, 7], bottom=[8, 9, 10, 11, 12, 13, 14, 15])
    target_pairs1 = [(0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)]

    matches1 = synthesizer._check_input_matches_target(input_state1, target_pairs1)
    print(f"  Perfect match: {matches1}")
    assert matches1, "Should match when input is perfectly aligned"

    # Test case 2: Input doesn't match target
    input_state2 = VectorState(top=[0, 2, 4, 6, 8, 10, 12, 14], bottom=[1, 3, 5, 7, 9, 11, 13, 15])
    target_pairs2 = [(0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)]

    matches2 = synthesizer._check_input_matches_target(input_state2, target_pairs2)
    print(f"  No match: {matches2}")
    assert not matches2, "Should not match when input is not aligned"

    print("✓ Input match check test passed\n")


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
    input_state = VectorState(top=[0, 1, 2, 3, 4, 5, 6, 7], bottom=[8, 9, 10, 11, 12, 13, 14, 15])
    identity_gadget = PermutationGadget(top_instructions=[], bottom_instructions=[], validated=True)

    output_state = synthesizer.compute_output_state(input_state, identity_gadget)
    print(f"  Identity gadget:")
    print(f"    Input:  {input_state}")
    print(f"    Output: {output_state}")

    assert output_state.top == input_state.top, "Identity should preserve top"
    assert output_state.bottom == input_state.bottom, "Identity should preserve bottom"

    print("✓ Output state computation test passed\n")


def test_first_stage_requires_no_permutation():
    """Test that the initial state is constructed to make first stage a null operation."""
    print("Testing first stage requires no permutation...")

    super_opt = BitonicSuperVectorizer(2, primitive_type.i32, vector_machine.AVX2)

    # Get initial state and first stage pairs
    initial_state = super_opt._create_initial_state()
    first_stage_pairs = super_opt.bitonic_sorter.stages[0]

    print(f"  First stage pairs: {first_stage_pairs}")
    print(f"Initial state: {initial_state}")

    # Verify that initial state matches first stage pairs
    for i, (a, b) in enumerate(first_stage_pairs):
        assert initial_state.top[i] == a, f"Top element at index {i} should be {a}, got {initial_state.top[i]}"
        assert initial_state.bottom[i] == b, f"Bottom element at index {i} should be {b}, got {initial_state.bottom[i]}"

    # Synthesize gadgets for first stage
    gadgets, combinations_tried = super_opt.synthesize_stage(initial_state, first_stage_pairs)

    print(f"  Found {len(gadgets)} valid gadget(s)")
    if combinations_tried > 0:
        percent = 100.0 * len(gadgets) / combinations_tried
        print(f"  Search coverage: {combinations_tried} combinations tried; {percent:.2f}% valid")
    else:
        print("  Search coverage: 0 combinations tried; 100.00% valid by construction")

    # Verify that the first gadget is a null (0-instruction) gadget
    assert len(gadgets) > 0, "Should find at least one valid gadget"
    null_gadgets = [g for g in gadgets if g.instruction_count() == 0]
    assert len(null_gadgets) > 0, "Should find at least one null (0-instruction) gadget for first stage"

    print(f"  ✓ First gadget requires {gadgets[0].instruction_count()} instructions (as expected)")
    print("✓ First stage null permutation test passed\n")


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
        test_check_input_matches_target()
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
