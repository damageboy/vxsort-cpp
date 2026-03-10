"""AVX512 i64 intrinsic tables."""

from __future__ import annotations

try:
    from .. import z3_avx
    from ..bitonic_types import InputRef, Symbolic, IntrinsicNode
except ImportError:
    import z3_avx  # type: ignore
    from bitonic_types import InputRef, Symbolic, IntrinsicNode  # type: ignore


def z3_functions() -> dict[str, object]:
    """Z3 function map for AVX512 i64."""
    intrinsics = {}

    # Unmasked single-input
    intrinsics["_mm512_permutexvar_epi64"] = z3_avx._mm512_permutexvar_epi64
    intrinsics["_mm512_permute_pd"] = z3_avx._mm512_permute_pd
    intrinsics["_mm512_permutevar_pd"] = z3_avx._mm512_permutevar_pd
    # Masked single-input
    intrinsics["_mm512_mask_permutexvar_epi64"] = z3_avx._mm512_mask_permutexvar_epi64
    intrinsics["_mm512_mask_permute_pd"] = z3_avx._mm512_mask_permute_pd
    intrinsics["_mm512_mask_permutevar_pd"] = z3_avx._mm512_mask_permutevar_pd
    # Unmasked dual-input
    intrinsics["_mm512_permutex2var_epi64"] = z3_avx._mm512_permutex2var_epi64
    intrinsics["_mm512_shuffle_pd"] = z3_avx._mm512_shuffle_pd
    intrinsics["_mm512_unpacklo_epi64"] = z3_avx._mm512_unpacklo_epi64
    intrinsics["_mm512_unpackhi_epi64"] = z3_avx._mm512_unpackhi_epi64
    intrinsics["_mm512_shuffle_i32x4"] = z3_avx._mm512_shuffle_i32x4
    intrinsics["_mm512_alignr_epi64"] = z3_avx._mm512_alignr_epi64
    # Masked dual-input
    intrinsics["_mm512_mask_permutex2var_epi64"] = z3_avx._mm512_mask_permutex2var_epi64
    intrinsics["_mm512_mask_shuffle_pd"] = z3_avx._mm512_mask_shuffle_pd
    intrinsics["_mm512_mask_unpacklo_epi64"] = z3_avx._mm512_mask_unpacklo_epi64
    intrinsics["_mm512_mask_unpackhi_epi64"] = z3_avx._mm512_mask_unpackhi_epi64
    intrinsics["_mm512_mask_shuffle_i32x4"] = z3_avx._mm512_mask_shuffle_i32x4
    intrinsics["_mm512_mask_alignr_epi64"] = z3_avx._mm512_mask_alignr_epi64

    return intrinsics


def single_input_nodes(input_ref: InputRef) -> list[IntrinsicNode]:
    """Single-input IntrinsicNode templates for AVX512 i64."""
    tag = input_ref.name

    return [
        # Unmasked
        IntrinsicNode(
            "_mm512_permutexvar_epi64",
            {
                "a": input_ref,
                "op_idx": Symbolic(f"ctrl_permutexvar_{tag}", 512),
            },
        ),
        IntrinsicNode(
            "_mm512_permute_pd",
            {
                "a": input_ref,
                "imm8": Symbolic(f"imm8_permute_pd_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm512_permutevar_pd",
            {
                "a": input_ref,
                "b": Symbolic(f"ctrl_permutevar_pd_{tag}", 512),
            },
        ),
        # Masked
        IntrinsicNode(
            "_mm512_mask_permutexvar_epi64",
            {
                "src": input_ref,
                "k": Symbolic(f"k_mask_permutexvar_{tag}", 8),
                "op_idx": Symbolic(f"ctrl_m_permutexvar_{tag}", 512),
                "a": input_ref,
            },
        ),
        IntrinsicNode(
            "_mm512_mask_permute_pd",
            {
                "src": input_ref,
                "k": Symbolic(f"k_mask_permute_pd_{tag}", 8),
                "a": input_ref,
                "imm8": Symbolic(f"imm8_m_permute_pd_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm512_mask_permutevar_pd",
            {
                "src": input_ref,
                "k": Symbolic(f"k_mask_permutevar_pd_{tag}", 8),
                "a": input_ref,
                "b": Symbolic(f"ctrl_m_permutevar_pd_{tag}", 512),
            },
        ),
    ]


def dual_input_nodes(ref1: InputRef, ref2: InputRef) -> list[IntrinsicNode]:
    """Dual-input IntrinsicNode templates for AVX512 i64."""
    tag = f"{ref1.name}_{ref2.name}"

    return [
        # Unmasked
        IntrinsicNode(
            "_mm512_permutex2var_epi64",
            {
                "a": ref1,
                "op_idx": Symbolic(f"ctrl_permutex2var_{tag}", 512),
                "b": ref2,
            },
        ),
        IntrinsicNode(
            "_mm512_shuffle_pd",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_shuffle_pd_{tag}", 8),
            },
            isomorphic_order=False,
        ),
        IntrinsicNode(
            "_mm512_unpacklo_epi64", {"a": ref1, "b": ref2}, isomorphic_order=False
        ),
        IntrinsicNode(
            "_mm512_unpackhi_epi64", {"a": ref1, "b": ref2}, isomorphic_order=False
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
            "_mm512_alignr_epi64",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_alignr_{tag}", 8),
            },
            isomorphic_order=False,
        ),
        # Masked
        IntrinsicNode(
            "_mm512_mask_permutex2var_epi64",
            {
                "a": ref1,
                "k": Symbolic(f"k_mask_permutex2var_{tag}", 8),
                "op_idx": Symbolic(f"ctrl_m_permutex2var_{tag}", 512),
                "b": ref2,
            },
        ),
        IntrinsicNode(
            "_mm512_mask_shuffle_pd",
            {
                "src": ref1,
                "k": Symbolic(f"k_mask_shuffle_pd_{tag}", 8),
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_m_shuffle_pd_{tag}", 8),
            },
            isomorphic_order=False,
        ),
        IntrinsicNode(
            "_mm512_mask_unpacklo_epi64",
            {
                "src": ref1,
                "k": Symbolic(f"k_mask_unpacklo_{tag}", 8),
                "a": ref1,
                "b": ref2,
            },
            isomorphic_order=False,
        ),
        IntrinsicNode(
            "_mm512_mask_unpackhi_epi64",
            {
                "src": ref1,
                "k": Symbolic(f"k_mask_unpackhi_{tag}", 8),
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
            "_mm512_mask_alignr_epi64",
            {
                "src": ref1,
                "k": Symbolic(f"k_mask_alignr_{tag}", 8),
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_m_alignr_{tag}", 8),
            },
            isomorphic_order=False,
        ),
    ]
