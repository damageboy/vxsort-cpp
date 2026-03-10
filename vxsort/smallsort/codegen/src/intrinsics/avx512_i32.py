"""AVX512 i32 intrinsic tables."""

from __future__ import annotations

try:
    from .. import z3_avx
    from ..bitonic_types import InputRef, Symbolic, IntrinsicNode
except ImportError:
    import z3_avx  # type: ignore
    from bitonic_types import InputRef, Symbolic, IntrinsicNode  # type: ignore


def z3_functions() -> dict[str, object]:
    """Z3 function map for AVX512 i32."""
    intrinsics = {}

    # Unmasked single-input
    intrinsics["_mm512_permutexvar_epi32"] = z3_avx._mm512_permutexvar_epi32
    intrinsics["_mm512_permute_ps"] = z3_avx._mm512_permute_ps
    intrinsics["_mm512_permutevar_ps"] = z3_avx._mm512_permutevar_ps
    # Masked single-input
    intrinsics["_mm512_mask_permutexvar_epi32"] = z3_avx._mm512_mask_permutexvar_epi32
    intrinsics["_mm512_mask_permute_ps"] = z3_avx._mm512_mask_permute_ps
    intrinsics["_mm512_mask_permutevar_ps"] = z3_avx._mm512_mask_permutevar_ps
    # Unmasked dual-input
    intrinsics["_mm512_permutex2var_epi32"] = z3_avx._mm512_permutex2var_epi32
    intrinsics["_mm512_shuffle_ps"] = z3_avx._mm512_shuffle_ps
    intrinsics["_mm512_unpacklo_epi32"] = z3_avx._mm512_unpacklo_epi32
    intrinsics["_mm512_unpackhi_epi32"] = z3_avx._mm512_unpackhi_epi32
    intrinsics["_mm512_shuffle_i32x4"] = z3_avx._mm512_shuffle_i32x4
    intrinsics["_mm512_alignr_epi32"] = z3_avx._mm512_alignr_epi32
    # Masked dual-input
    intrinsics["_mm512_mask_permutex2var_epi32"] = z3_avx._mm512_mask_permutex2var_epi32
    intrinsics["_mm512_mask_shuffle_ps"] = z3_avx._mm512_mask_shuffle_ps
    intrinsics["_mm512_mask_unpacklo_epi32"] = z3_avx._mm512_mask_unpacklo_epi32
    intrinsics["_mm512_mask_unpackhi_epi32"] = z3_avx._mm512_mask_unpackhi_epi32
    intrinsics["_mm512_mask_shuffle_i32x4"] = z3_avx._mm512_mask_shuffle_i32x4
    intrinsics["_mm512_mask_alignr_epi32"] = z3_avx._mm512_mask_alignr_epi32

    return intrinsics


def single_input_nodes(input_ref: InputRef) -> list[IntrinsicNode]:
    """Single-input IntrinsicNode templates for AVX512 i32."""
    tag = input_ref.name

    return [
        # Unmasked
        IntrinsicNode(
            "_mm512_permutexvar_epi32",
            {
                "a": input_ref,
                "op_idx": Symbolic(f"ctrl_permutexvar_{tag}", 512),
            },
        ),
        IntrinsicNode(
            "_mm512_permute_ps",
            {
                "a": input_ref,
                "imm8": Symbolic(f"imm8_permute_ps_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm512_permutevar_ps",
            {
                "a": input_ref,
                "b": Symbolic(f"ctrl_permutevar_ps_{tag}", 512),
            },
        ),
        # Masked (16-bit k-mask)
        IntrinsicNode(
            "_mm512_mask_permutexvar_epi32",
            {
                "src": input_ref,
                "k": Symbolic(f"k_mask_permutexvar_{tag}", 16),
                "op_idx": Symbolic(f"ctrl_m_permutexvar_{tag}", 512),
                "a": input_ref,
            },
        ),
        IntrinsicNode(
            "_mm512_mask_permute_ps",
            {
                "src": input_ref,
                "k": Symbolic(f"k_mask_permute_ps_{tag}", 16),
                "a": input_ref,
                "imm8": Symbolic(f"imm8_m_permute_ps_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm512_mask_permutevar_ps",
            {
                "src": input_ref,
                "k": Symbolic(f"k_mask_permutevar_ps_{tag}", 16),
                "a": input_ref,
                "b": Symbolic(f"ctrl_m_permutevar_ps_{tag}", 512),
            },
        ),
    ]


def dual_input_nodes(ref1: InputRef, ref2: InputRef) -> list[IntrinsicNode]:
    """Dual-input IntrinsicNode templates for AVX512 i32."""
    tag = f"{ref1.name}_{ref2.name}"

    return [
        # Unmasked
        IntrinsicNode(
            "_mm512_permutex2var_epi32",
            {
                "a": ref1,
                "op_idx": Symbolic(f"ctrl_permutex2var_{tag}", 512),
                "b": ref2,
            },
        ),
        IntrinsicNode(
            "_mm512_shuffle_ps",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_shuffle_ps_{tag}", 8),
            },
            isomorphic_order=False,
        ),
        IntrinsicNode(
            "_mm512_unpacklo_epi32", {"a": ref1, "b": ref2}, isomorphic_order=False
        ),
        IntrinsicNode(
            "_mm512_unpackhi_epi32", {"a": ref1, "b": ref2}, isomorphic_order=False
        ),
        IntrinsicNode(
            "_mm512_shuffle_i32x4",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_shuf_i32x4_{tag}", 8),
            },
            isomorphic_order=False,
        ),
        IntrinsicNode(
            "_mm512_alignr_epi32",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_alignr_{tag}", 8),
            },
            isomorphic_order=False,
        ),
        # Masked (16-bit k-mask)
        IntrinsicNode(
            "_mm512_mask_permutex2var_epi32",
            {
                "a": ref1,
                "k": Symbolic(f"k_mask_permutex2var_{tag}", 16),
                "op_idx": Symbolic(f"ctrl_m_permutex2var_{tag}", 512),
                "b": ref2,
            },
        ),
        IntrinsicNode(
            "_mm512_mask_shuffle_ps",
            {
                "src": ref1,
                "k": Symbolic(f"k_mask_shuffle_ps_{tag}", 16),
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_m_shuffle_ps_{tag}", 8),
            },
            isomorphic_order=False,
        ),
        IntrinsicNode(
            "_mm512_mask_unpacklo_epi32",
            {
                "src": ref1,
                "k": Symbolic(f"k_mask_unpacklo_{tag}", 16),
                "a": ref1,
                "b": ref2,
            },
            isomorphic_order=False,
        ),
        IntrinsicNode(
            "_mm512_mask_unpackhi_epi32",
            {
                "src": ref1,
                "k": Symbolic(f"k_mask_unpackhi_{tag}", 16),
                "a": ref1,
                "b": ref2,
            },
            isomorphic_order=False,
        ),
        IntrinsicNode(
            "_mm512_mask_shuffle_i32x4",
            {
                "src": ref1,
                "k": Symbolic(f"k_mask_shuf_i32x4_{tag}", 16),
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_m_shuf_i32x4_{tag}", 8),
            },
            isomorphic_order=False,
        ),
        IntrinsicNode(
            "_mm512_mask_alignr_epi32",
            {
                "src": ref1,
                "k": Symbolic(f"k_mask_alignr_{tag}", 16),
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_m_alignr_{tag}", 8),
            },
            isomorphic_order=False,
        ),
    ]
