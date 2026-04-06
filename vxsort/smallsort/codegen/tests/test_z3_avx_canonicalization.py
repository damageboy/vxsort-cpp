"""Test that z3_avx intrinsics add canonicalization constraints for don't-care bits."""

from z3 import BitVec, Solver, sat, Extract, main_ctx

import z3_avx


def test_permutexvar_256_epi32_canonicalization():
    """Solver-aware permutexvar should constrain don't-care bits to zero."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.ymm_reg("a", ctx=ctx)
    op_idx = BitVec("op_idx", 256)
    _result = z3_avx._generic_permutexvar(a, op_idx, 256, 32, solver=solver)
    solver.add(Extract(31, 3, op_idx) != 0)
    result = solver.check()
    assert result != sat, "Don't-care bits should be constrained to zero"


def test_permutexvar_256_epi64_canonicalization():
    """Solver-aware permutexvar 64-bit should constrain don't-care bits."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.ymm_reg("a", ctx=ctx)
    op_idx = BitVec("op_idx", 256)
    _result = z3_avx._generic_permutexvar(a, op_idx, 256, 64, solver=solver)
    solver.add(Extract(63, 2, op_idx) != 0)
    assert solver.check() != sat


def test_permutex2var_512_epi32_canonicalization():
    """Solver-aware permutex2var should constrain don't-care bits."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.zmm_reg("a", ctx=ctx)
    op_idx = BitVec("op_idx", 512)
    b = z3_avx.zmm_reg("b", ctx=ctx)
    _result = z3_avx._generic_permutex2var(a, op_idx, b, 32, solver=solver)
    solver.add(Extract(31, 5, op_idx) != 0)
    assert solver.check() != sat


def test_permutevar_256_ps_canonicalization():
    """Solver-aware permutevar_ps should constrain don't-care bits."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.ymm_reg("a", ctx=ctx)
    b = BitVec("ctrl", 256)
    _result = z3_avx._generic_permutevar(a, b, 256, 32, solver=solver)
    solver.add(Extract(31, 2, b) != 0)
    assert solver.check() != sat


def test_permutevar_256_pd_canonicalization():
    """Solver-aware permutevar_pd should constrain don't-care bits."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.ymm_reg("a", ctx=ctx)
    b = BitVec("ctrl", 256)
    _result = z3_avx._generic_permutevar(a, b, 256, 64, solver=solver)
    solver.add(Extract(63, 2, b) != 0)
    assert solver.check() != sat


def test_permutevar_256_pd_bit0_canonicalization():
    """Solver-aware permutevar_pd should also constrain bit 0 to zero."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.ymm_reg("a", ctx=ctx)
    b = BitVec("ctrl", 256)
    _result = z3_avx._generic_permutevar(a, b, 256, 64, solver=solver)
    # For pd, only bit[1] is used; bit[0] must also be zeroed
    solver.add(Extract(0, 0, b) != 0)
    assert solver.check() != sat


def test_blendv_256_pd_canonicalization():
    """Solver-aware blendv should constrain non-sign bits to zero."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.ymm_reg("a", ctx=ctx)
    b = z3_avx.ymm_reg("b", ctx=ctx)
    mask = BitVec("mask", 256)
    _result = z3_avx._generic_blendv(a, b, mask, 256, 64, solver=solver)
    solver.add(Extract(62, 0, mask) != 0)
    assert solver.check() != sat


def test_blend_256_pd_canonicalization():
    """Solver-aware blend_pd should constrain unused imm8 bits."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.ymm_reg("a", ctx=ctx)
    b = z3_avx.ymm_reg("b", ctx=ctx)
    imm = BitVec("imm8", 8)
    _result = z3_avx._generic_blend(a, b, imm, 256, 64, solver=solver)
    solver.add(Extract(7, 4, imm) != 0)
    assert solver.check() != sat


def test_shuffle_pd_256_canonicalization():
    """shuffle_pd 256-bit uses 4 of 8 imm8 bits."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.ymm_reg("a", ctx=ctx)
    b = z3_avx.ymm_reg("b", ctx=ctx)
    imm = BitVec("imm8", 8)
    _result = z3_avx._shuffle_pd_generic(a, b, imm, 2, solver=solver)
    # 2 lanes x 2 bits = 4 used bits; bits [7:4] must be zero
    solver.add(Extract(7, 4, imm) != 0)
    assert solver.check() != sat


def test_permute_pd_256_canonicalization():
    """permute_pd uses only bits [1:0] of imm8."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.ymm_reg("a", ctx=ctx)
    imm = BitVec("imm8", 8)
    _result = z3_avx._permute_pd_generic(a, imm, 2, solver=solver)
    solver.add(Extract(7, 2, imm) != 0)
    assert solver.check() != sat


def test_permute2x128_canonicalization():
    """permute2x128 has unused bit 2 in each control nibble."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.ymm_reg("a", ctx=ctx)
    b = z3_avx.ymm_reg("b", ctx=ctx)
    imm = BitVec("imm8", 8)
    _result = z3_avx._mm256_permute2x128_si256(a, b, imm, solver=solver)
    # Bit 2 of low nibble should be zero
    solver.add(Extract(2, 2, imm) != 0)
    assert solver.check() != sat


def test_alignr_256_epi64_canonicalization():
    """alignr 256-bit/64-bit uses 2 of 8 imm8 bits."""
    solver = Solver()
    ctx = main_ctx()
    a = z3_avx.ymm_reg("a", ctx=ctx)
    b = z3_avx.ymm_reg("b", ctx=ctx)
    imm = BitVec("imm8", 8)
    _result = z3_avx._generic_alignr(a, b, imm, 256, 64, solver=solver)
    # 4 elements -> 2 shift bits, bits [7:2] must be zero
    solver.add(Extract(7, 2, imm) != 0)
    assert solver.check() != sat


def test_synthesizer_passes_solver_to_intrinsics():
    """Integration: synthesizer should produce canonical (minimal) control vectors."""
    from gadget_synthesizer import GadgetSynthesizer, VectorState
    from bitonic_types import InputRef, Symbolic, IntrinsicNode, GadgetGraph
    from util.enums import vector_machine, primitive_type

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i64)

    # Reverse elements: [3,2,1,0] in top, bottom stays
    input_state = VectorState(top=[0, 1, 2, 3], bottom=[4, 5, 6, 7])
    # Target: top reversed
    target_pairs = [(3, 4), (2, 5), (1, 6), (0, 7)]

    ref = InputRef("top")
    graph = GadgetGraph(
        top=IntrinsicNode(
            "_mm256_permutexvar_epi64",
            {"a": ref, "op_idx": Symbolic("ctrl_test", 256)},
        ),
        bottom=None,
    )

    results, _, _ = synthesizer.synthesize_gadget_with_symbolic(
        graph, input_state, target_pairs, max_solutions=3
    )

    assert len(results) > 0, "Should find at least one solution"

    # All solutions should have the same control vector since don't-care bits are zeroed
    ctrl_values = set()
    for gadget, _output_state in results:
        for instr in gadget.top_instructions:
            for key, val in instr.args.items():
                if key == "op_idx" and isinstance(val, int):
                    ctrl_values.add(val)

    # With canonicalization, permutexvar_epi64 uses 2 bits per 64-bit lane.
    # The reverse permutation [3,2,1,0] has exactly one encoding with zeroed
    # don't-care bits.
    assert (
        len(ctrl_values) == 1
    ), f"Expected all solutions to have same canonical control vector, got {ctrl_values}"
