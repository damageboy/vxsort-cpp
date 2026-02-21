#!/usr/bin/env python3
"""Tests for the BitonicSuperVectorizer."""

import sys

import pytest
from bitonic_sorter import BitonicSorter
from asm_exporter import (
    RegisterAllocator,
    _emit_compare_swap_lines,
    _format_instruction,
)
from bitonic_super_optimizer import (
    BitonicSuperVectorizer,
    GadgetSynthesizer,
    InstructionSpec,
    PermutationGadget,
    SymbolicPlaceholder,
    VectorState,
)
from functional import seq
from utils import primitive_type, vector_machine
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


def test_generate_candidate_gadgets_order_and_counts():
    """Candidate gadget generation keeps legacy ordering/count semantics."""
    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i64)

    top_depth = 2
    bottom_depth = 1
    candidates = synthesizer._generate_candidate_gadgets(top_depth, bottom_depth)

    top_sequences = []
    for inst1 in synthesizer.single_insts_top:
        for inst2 in synthesizer.single_insts_top:
            top_sequences.append([inst1, inst2])
    for inst1 in synthesizer.dual_insts_top_bottom:
        for inst2 in synthesizer.single_insts_top:
            top_sequences.append([inst1, inst2])
    for inst1 in synthesizer.single_insts_top:
        for inst2 in synthesizer.dual_insts_top_bottom:
            top_sequences.append([inst1, inst2])

    bottom_sequences = [[inst] for inst in synthesizer.single_insts_bottom]
    bottom_sequences.extend([[inst] for inst in synthesizer.dual_insts_top_bottom])

    expected = []
    for top_seq in top_sequences:
        for bottom_seq in bottom_sequences:
            expected.append((top_seq, bottom_seq))

    assert candidates == expected
    assert synthesizer._generate_candidate_gadgets(0, 0) == [([], [])]
    assert synthesizer._generate_candidate_gadgets(-1, 1) == [
        ([], bottom_seq) for bottom_seq in bottom_sequences
    ]


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
def test_first_stage_null_gadget(vm, dt):
    """Z3 synthesis: first stage should yield a 0-instruction gadget."""
    super_opt = BitonicSuperVectorizer(2, dt, vm)

    solutions, _ = super_opt.build_solution_tree(
        depth_limit=1, gadget_depth=1, natural_order=False, max_unique_outputs=1
    )

    assert len(solutions) > 0, "Should find at least one valid solution"
    null_solutions = [
        s for s in solutions if any(g.instruction_count() == 0 for g in s.gadgets)
    ]
    assert (
        len(null_solutions) > 0
    ), "Should find at least one solution with null (0-instruction) gadget for first stage"


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
    """Test that the multiplexer-based operand selection in 2-instruction gadgets works.

    Creates a 2-instruction template [permute_ps, permute_ps] on the top side
    with an empty bottom sequence. For the second instruction, the synthesizer
    creates a Z3 select variable that chooses between {top, bottom, prev}. After
    synthesis, the select variable is resolved to a source name string stored in
    InstructionSpec.args.

    We verify that at least one result has inst2 with args["a"] == "prev",
    proving the mux actually selected the chained output from inst1.
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

    # Build a 2-instruction template: [permute_ps, permute_ps] for top side.
    # The first instruction operates on "top" directly.
    # The second instruction's "a" arg is "top" -- but since it is inst2
    # (inst_idx > 0), _apply_instructions will create a mux select variable
    # choosing between {top_reg, bottom_reg, prev_output}.
    top_template = [
        InstructionSpec(
            "_mm256_permute_ps",
            {
                "a": "top",
                "imm8": SymbolicPlaceholder("imm8_permute_ps_inst1", 8),
            },
        ),
        InstructionSpec(
            "_mm256_permute_ps",
            {
                "a": "top",
                "imm8": SymbolicPlaceholder("imm8_permute_ps_inst2", 8),
            },
        ),
    ]
    bottom_template = []

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        top_instructions_template=top_template,
        bottom_instructions_template=bottom_template,
        input_state=input_state,
        target_pairs=target_pairs,
        max_solutions=1,
        max_unique_outputs=3,
    )

    assert len(results) > 0, "Should find at least one valid gadget"

    # All returned gadgets must be validated
    for gadget, output_state in results:
        assert gadget.validated, f"Gadget should be validated: {gadget}"

    # At least one result should have inst2 with args["a"] == "prev",
    # proving the mux selected the chained output from inst1
    prev_results = [
        (gadget, output_state)
        for gadget, output_state in results
        if len(gadget.top_instructions) == 2
        and gadget.top_instructions[1].args.get("a") == "prev"
    ]
    assert len(prev_results) > 0, (
        'Expected at least one gadget with inst2 args[\'a\'] == \'prev\', '
        f'but got: {[(g.top_instructions[1].args.get("a") if len(g.top_instructions) == 2 else "N/A") for g, _ in results]}'
    )

    print(
        f"  Found {len(results)} total results, {len(prev_results)} with prev chaining"
    )
    for gadget, output_state in prev_results:
        inst2 = gadget.top_instructions[1]
        a_val = inst2.args["a"]
        imm8_val = inst2.args["imm8"]
        print(f"    inst2: {inst2.intrinsic_name}(a={a_val}, imm8={imm8_val})")
    print("  All gadgets validated: True")


def test_single_dual_template_combination():
    """Test that (single, dual) 2-instruction templates work with mux encoding.

    Creates a depth-2 gadget with a single-input instruction (permute_ps)
    followed by a dual-input instruction (shuffle_ps). The mux encoding
    lets Z3 decide whether inst2's "a" and "b" operands read from top,
    bottom, or prev (the output of inst1).
    """
    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)
    n = synthesizer.elements_per_vector

    sorter = BitonicSorter(2 * n)
    target_pairs = sorter.stages[0]

    input_state = VectorState(
        top=[p[0] for p in target_pairs],
        bottom=[p[1] for p in target_pairs],
    )

    inst1 = InstructionSpec(
        "_mm256_permute_ps",
        {
            "a": "top",
            "imm8": SymbolicPlaceholder("imm8_permute_ps_sd_test", 8),
        },
    )
    inst2 = InstructionSpec(
        "_mm256_shuffle_ps",
        {
            "a": "top",
            "b": "bottom",
            "imm8": SymbolicPlaceholder("imm8_shuffle_ps_sd_test", 8),
        },
    )

    top_template = [inst1, inst2]
    bottom_template = [
        InstructionSpec(
            "_mm256_permute_ps",
            {
                "a": "bottom",
                "imm8": SymbolicPlaceholder("imm8_permute_ps_sd_bottom", 8),
            },
        )
    ]

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        top_instructions_template=top_template,
        bottom_instructions_template=bottom_template,
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
    """Test that _format_instruction uses correct absolute register mapping.

    Previously, bottom-side instructions had their registers swapped because
    the exporter used relative (dest_reg, other_reg) mapping. Now it uses
    absolute (top_reg, bottom_reg, is_top) mapping.
    """
    reg_alloc = RegisterAllocator(vector_machine.AVX2, primitive_type.i32, 2)

    # Bottom single-input: args["a"] = "bottom" should use ymm1 (bottom reg)
    inst_bottom_single = InstructionSpec(
        "_mm256_permute_ps", {"a": "bottom", "imm8": 0xB1}
    )
    asm = _format_instruction(
        inst_bottom_single, reg_alloc, "ymm0", "ymm1", is_top=False
    )
    # dest=ymm1 (bottom), src=ymm1 (bottom)
    assert "ymm1, ymm1" in asm, f"Bottom single should read from ymm1, got: {asm}"

    reg_alloc.reset_temps()

    # Bottom dual-input: args["a"]="top", args["b"]="bottom"
    inst_bottom_dual = InstructionSpec(
        "_mm256_shuffle_ps", {"a": "top", "b": "bottom", "imm8": 0x4E}
    )
    asm = _format_instruction(inst_bottom_dual, reg_alloc, "ymm0", "ymm1", is_top=False)
    # dest=ymm1, src1=ymm0 (top), src2=ymm1 (bottom)
    assert (
        "ymm1, ymm0, ymm1" in asm
    ), f"Bottom dual should have src1=ymm0, src2=ymm1, got: {asm}"

    reg_alloc.reset_temps()

    # "prev" source on top side → should resolve to ymm0 (top = dest for top chain)
    inst_prev_top = InstructionSpec("_mm256_permute_ps", {"a": "prev", "imm8": 0xB1})
    asm = _format_instruction(inst_prev_top, reg_alloc, "ymm0", "ymm1", is_top=True)
    assert "ymm0, ymm0" in asm, f"prev on top side should use ymm0, got: {asm}"

    reg_alloc.reset_temps()

    # "prev" source on bottom side → should resolve to ymm1 (bottom = dest for bottom chain)
    inst_prev_bottom = InstructionSpec("_mm256_permute_ps", {"a": "prev", "imm8": 0xB1})
    asm = _format_instruction(inst_prev_bottom, reg_alloc, "ymm0", "ymm1", is_top=False)
    assert "ymm1, ymm1" in asm, f"prev on bottom side should use ymm1, got: {asm}"


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
    """Test that AVX512 i32 synthesis infrastructure works (null gadget for first stage).

    Full depth-2 synthesis is too slow for CI (~4 min for 16-element vectors).
    The parametrized test_first_stage_requires_no_permutation covers i32+AVX512.
    """
    super_opt = BitonicSuperVectorizer(2, primitive_type.i32, vector_machine.AVX512)
    assert super_opt.elements_per_vector == 16, "AVX512 i32 should have 16 elements"
    assert super_opt.total_elements == 32, "Should have 32 total elements"
    # Verify intrinsics are registered
    assert len(super_opt.synthesizer.available_intrinsics) > 0
    # Verify single and dual templates enumerate without error
    single = super_opt.synthesizer._enumerate_single_input_instructions("test")
    dual = super_opt.synthesizer._enumerate_dual_input_instructions("top", "bottom")
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
