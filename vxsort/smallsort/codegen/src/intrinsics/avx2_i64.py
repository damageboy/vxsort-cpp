"""AVX2 i64 intrinsic tables."""

from __future__ import annotations

try:
    from .. import z3_avx
    from ..bitonic_types import InputRef, Symbolic, IntrinsicNode
except ImportError:
    import z3_avx  # type: ignore
    from bitonic_types import InputRef, Symbolic, IntrinsicNode  # type: ignore


def z3_functions() -> dict[str, object]:
    """Z3 function map for AVX2 i64."""
    intrinsics = {}

    # Single input permutes (with immediates)
    intrinsics["_mm256_permute4x64_epi64"] = z3_avx._mm256_permute4x64_epi64
    intrinsics["_mm256_permute_pd"] = z3_avx._mm256_permute_pd

    # Single input permutes (with control vectors)
    intrinsics["_mm256_permutexvar_epi64"] = z3_avx._mm256_permutexvar_epi64
    intrinsics["_mm256_permutevar_pd"] = z3_avx._mm256_permutevar_pd

    # Two input permutes/shuffles
    intrinsics["_mm256_shuffle_pd"] = z3_avx._mm256_shuffle_pd
    intrinsics["_mm256_unpacklo_epi64"] = z3_avx._mm256_unpacklo_epi64
    intrinsics["_mm256_unpackhi_epi64"] = z3_avx._mm256_unpackhi_epi64
    intrinsics["_mm256_permute2x128_si256"] = z3_avx._mm256_permute2x128_si256

    # Blends
    intrinsics["_mm256_blend_pd"] = z3_avx._mm256_blend_pd
    intrinsics["_mm256_blendv_pd"] = z3_avx._mm256_blendv_pd

    # Align operations
    intrinsics["_mm256_alignr_epi64"] = z3_avx._mm256_alignr_epi64

    return intrinsics


def single_input_nodes(input_ref: InputRef) -> list[IntrinsicNode]:
    """Single-input IntrinsicNode templates for AVX2 i64."""
    tag = input_ref.name

    return [
        IntrinsicNode(
            "_mm256_permute_pd",
            {
                "a": input_ref,
                "imm8": Symbolic(f"imm8_permute_pd_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_permute4x64_epi64",
            {
                "a": input_ref,
                "imm8": Symbolic(f"imm8_permute4x64_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_permutexvar_epi64",
            {
                "a": input_ref,
                "op_idx": Symbolic(f"ctrl_permutexvar_{tag}", 256),
            },
        ),
        IntrinsicNode(
            "_mm256_permutevar_pd",
            {
                "a": input_ref,
                "b": Symbolic(f"ctrl_permutevar_pd_{tag}", 256),
            },
        ),
    ]


def dual_input_nodes(ref1: InputRef, ref2: InputRef) -> list[IntrinsicNode]:
    """Dual-input IntrinsicNode templates for AVX2 i64."""
    tag = f"{ref1.name}_{ref2.name}"

    return [
        IntrinsicNode(
            "_mm256_shuffle_pd",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_shuffle_{tag}", 8),
            },
        ),
        IntrinsicNode("_mm256_unpacklo_epi64", {"a": ref1, "b": ref2}),
        IntrinsicNode("_mm256_unpackhi_epi64", {"a": ref1, "b": ref2}),
        IntrinsicNode(
            "_mm256_permute2x128_si256",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_perm2x128_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_blend_pd",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_blend_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_alignr_epi64",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_alignr_{tag}", 8),
            },
        ),
    ]
