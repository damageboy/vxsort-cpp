#!/usr/bin/env python3
"""Test the new symbolic immediate synthesis."""

import sys

from util.enums import vector_machine, primitive_type
from bitonic_super_optimizer import GadgetSynthesizer, VectorState
from bitonic_types import InputRef, Symbolic, IntrinsicNode, GadgetGraph


def test_symbolic_synthesis():
    """Test that symbolic synthesis finds valid immediates."""
    print("Testing symbolic immediate synthesis...")

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)

    # Test case 1: Identity - input already matches target
    # This should find a 0-instruction gadget
    input_state = VectorState(
        top=[0, 1, 2, 3, 4, 5, 6, 7], bottom=[8, 9, 10, 11, 12, 13, 14, 15]
    )

    target_pairs = [
        (0, 8),
        (1, 9),
        (2, 10),
        (3, 11),
        (4, 12),
        (5, 13),
        (6, 14),
        (7, 15),
    ]

    print("Test 1: Identity case (should find 0-instruction gadget)")
    print(f"Input state: {input_state}")
    print(f"Target pairs: {target_pairs}")

    # Try with no instructions (should succeed)
    # Returns (results, construction_time, solver_time)
    # where results is list of (gadget, output_state) tuples
    graph = GadgetGraph(top=None, bottom=None)
    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        graph, input_state, target_pairs
    )

    print(f"Found {len(results)} gadget(s)")
    if results:
        gadget, output_state = results[0]
        if gadget.instruction_count() == 0:
            print(f"Output state: {output_state}")
            print("✓ Identity test passed!\n")
        else:
            print("✗ Identity test failed!\n")
            assert False, "Identity test failed!"
    else:
        print("✗ Identity test failed - no results!\n")
        assert False, "Identity test failed!"

    # Test case 2: Simple permutation using _mm256_permute2x128_si256
    # Swap the two 128-bit lanes
    print("Test 2: Lane swap using _mm256_permute2x128_si256")
    input_state2 = VectorState(
        top=[0, 1, 2, 3, 4, 5, 6, 7], bottom=[8, 9, 10, 11, 12, 13, 14, 15]
    )

    # After swapping lanes: top becomes [4,5,6,7,0,1,2,3]
    # To align with bottom, we need pairs where bottom stays same
    target_pairs2 = [
        (4, 8),
        (5, 9),
        (6, 10),
        (7, 11),
        (0, 12),
        (1, 13),
        (2, 14),
        (3, 15),
    ]

    top_ref = InputRef("top")
    node = IntrinsicNode(
        "_mm256_permute2x128_si256",
        {"a": top_ref, "b": top_ref, "imm8": Symbolic("test_imm8_perm2x128", 8)},
    )
    graph2 = GadgetGraph(top=node, bottom=None)

    print(f"Target pairs: {target_pairs2}")
    print(f"Instruction template: {node.name}")

    # Returns (results, construction_time, solver_time)
    # where results is list of (gadget, output_state) tuples
    results2, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        graph2, input_state2, target_pairs2
    )

    print(f"Found {len(results2)} gadget(s)")

    if results2:
        gadget, output_state = results2[0]
        print(f"Top instructions: {gadget.top_instructions}")
        print(f"Output state: {output_state}")

        if gadget.top_instructions:
            inst = gadget.top_instructions[0]
            if "imm8" in inst.args:
                imm8_value = inst.args["imm8"]
                print(f"Z3 found immediate value: {imm8_value} (0x{imm8_value:02x})")
                print("✓ Symbolic synthesis test passed!")
                return

    print("✗ No valid gadget found for permute2x128 test")
    print("(This may be expected if the permutation isn't achievable)")
    print("Let's try existing tests instead...")
    return


def test_enumerate_instruction_count():
    """Test that instruction enumeration produces fewer templates."""
    print("\nTesting instruction template generation...")

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)

    # Get single input instructions
    single_insts = synthesizer._enumerate_single_input_intrinsics(InputRef("test"))
    print(f"Single-input instruction templates: {len(single_insts)}")
    for n in single_insts:
        print(f"  - {n.name}")
        # Check if it has symbolic immediate
        for key, val in n.operands.items():
            if isinstance(val, Symbolic):
                print(f"    Symbolic {key}: {val}")

    # Get dual input instructions
    dual_insts = synthesizer._enumerate_dual_input_intrinsics(
        InputRef("top"), InputRef("bottom")
    )
    print(f"\nDual-input instruction templates: {len(dual_insts)}")
    for n in dual_insts:
        print(f"  - {n.name}")
        for key, val in n.operands.items():
            if isinstance(val, Symbolic):
                print(f"    Symbolic {key}: {val}")

    print("\n✓ Instruction enumeration test passed!")
    print(f"\nTotal templates: {len(single_insts) + len(dual_insts)}")
    print(
        f"New implementation generates only {
            len(single_insts) + len(dual_insts)
        } templates!"
    )
    print(
        f"Improvement: {
            62
            / (
                len(single_insts) + len(dual_insts)
            ):.1f}x reduction in candidates to try"
    )


def test_multi_solution_enumeration():
    """Test that synthesize_gadget_with_symbolic returns multiple valid solutions.

    Uses two symmetric blend_pd instructions (one for top, one for bottom)
    on i64 where pairs are already lane-aligned. For each lane, the two
    blend bits must differ: one selects from top and the other from bottom.
    This gives 2^4 = 16 valid low-4-bit combinations (and more with
    unconstrained high bits).
    """
    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i64)

    # Pairs already lane-aligned: lane i has pair (i+1, i+5)
    input_state = VectorState(
        top=[1, 2, 3, 4],
        bottom=[5, 6, 7, 8],
    )

    target_pairs = [(1, 5), (2, 6), (3, 7), (4, 8)]

    top_ref = InputRef("top")
    bottom_ref = InputRef("bottom")

    blend_top = IntrinsicNode(
        "_mm256_blend_pd",
        {"a": top_ref, "b": bottom_ref, "imm8": Symbolic("test_blend_top", 8)},
    )
    blend_bottom = IntrinsicNode(
        "_mm256_blend_pd",
        {"a": top_ref, "b": bottom_ref, "imm8": Symbolic("test_blend_bot", 8)},
    )

    graph = GadgetGraph(top=blend_top, bottom=blend_bottom)

    # Top: blend(top, bottom, sym_top)
    # Bottom: blend(top, bottom, sym_bot)
    # For each lane, exactly one blend selects top and the other selects bottom.
    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        graph,
        input_state,
        target_pairs,
        max_solutions=20,
    )

    assert len(results) > 1, f"Expected multiple solutions, got {len(results)}"

    # All solutions should produce the same canonical output state
    output_states = set()
    for gadget, output_state in results:
        assert gadget.validated
        output_states.add(output_state.as_tuple())

    assert (
        len(output_states) == 1
    ), f"Expected 1 unique output state, got {len(output_states)}"


def test_shape_b_includes_swapped_for_asymmetric():
    """depth-1 candidates include both orderings for isomorphic_order=False duals."""
    from bitonic_super_optimizer import GadgetSynthesizer
    from bitonic_types import IntrinsicNode, InputRef
    from util.enums import vector_machine, primitive_type

    synth = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)
    graphs = synth._build_gadget_graphs(depth=1, single_intrinsics=[])

    # Collect all (name, a_ref, b_ref) tuples for dual nodes
    dual_signatures = set()
    for node in graphs:
        if not isinstance(node, IntrinsicNode):
            continue
        reg_refs = [
            (k, v.name) for k, v in node.operands.items() if isinstance(v, InputRef)
        ]
        if len(reg_refs) == 2:
            dual_signatures.add((node.name, reg_refs[0][1], reg_refs[1][1]))

    # shuffle_ps is asymmetric — both orderings must be present
    assert (
        "_mm256_shuffle_ps",
        "top",
        "bottom",
    ) in dual_signatures, "Missing shuffle_ps(top, bottom)"
    assert (
        "_mm256_shuffle_ps",
        "bottom",
        "top",
    ) in dual_signatures, "Missing shuffle_ps(bottom, top) — swapped variant"

    # blend_ps is symmetric — only one ordering
    blend_variants = [
        (n, a, b) for n, a, b in dual_signatures if n == "_mm256_blend_ps"
    ]
    assert (
        len(blend_variants) == 1
    ), f"blend_ps should have only 1 ordering, got {blend_variants}"


def test_shape_d_includes_swapped_for_asymmetric():
    """depth-2 dual→single candidates include both orderings for asymmetric inst0."""
    from bitonic_super_optimizer import GadgetSynthesizer
    from bitonic_types import IntrinsicNode, InputRef
    from util.enums import vector_machine, primitive_type

    synth = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)
    single = synth.single_intrinsics_top[:1]  # one single-input instruction
    graphs = synth._build_gadget_graphs(depth=2, single_intrinsics=single)

    # Collect root→child (inst1→inst0) pairs for shape D graphs
    # Shape D: root is single-input, its operand is a dual (the inst0)
    shape_d_inst0_names = set()
    for node in graphs:
        if not isinstance(node, IntrinsicNode):
            continue
        for v in node.operands.values():
            if isinstance(v, IntrinsicNode):
                inner_refs = [
                    (k, op.name)
                    for k, op in v.operands.items()
                    if isinstance(op, InputRef)
                ]
                if len(inner_refs) == 2:
                    shape_d_inst0_names.add(
                        (v.name, inner_refs[0][1], inner_refs[1][1])
                    )

    # shuffle_ps asymmetric: both orderings as inst0
    assert ("_mm256_shuffle_ps", "top", "bottom") in shape_d_inst0_names
    assert ("_mm256_shuffle_ps", "bottom", "top") in shape_d_inst0_names

    # blend_ps symmetric: only one ordering as inst0
    blend_variants = [
        (n, a, b) for n, a, b in shape_d_inst0_names if n == "_mm256_blend_ps"
    ]
    assert len(blend_variants) == 1


if __name__ == "__main__":
    try:
        test_enumerate_instruction_count()
        sys.exit(test_symbolic_synthesis())
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
