from __future__ import annotations

import functools
from dataclasses import dataclass


@dataclass(frozen=True)
class IntrinsicInfo:
    intrinsic_name: str
    asm_mnemonic: str
    operand_form: str
    imm_type: str | None = None
    control_vector_width: int | None = None


def xml_string_key(info: IntrinsicInfo) -> str:
    """Return the XML 'string' key, e.g. 'VPERMILPS (YMM, YMM, I8)'."""
    return f"{info.asm_mnemonic} ({info.operand_form})"


@functools.lru_cache(maxsize=1)
def _build_registry() -> dict[str, IntrinsicInfo]:
    def _i(
        name: str,
        mnemonic: str,
        operand_form: str,
        *,
        imm_type: str | None = None,
        cv_width: int | None = None,
    ) -> IntrinsicInfo:
        return IntrinsicInfo(
            intrinsic_name=name,
            asm_mnemonic=mnemonic,
            operand_form=operand_form,
            imm_type=imm_type,
            control_vector_width=cv_width,
        )

    entries: list[IntrinsicInfo] = [
        # ── AVX2 (YMM) ──────────────────────────────────────────────────
        _i("_mm256_permute4x64_epi64", "VPERMQ", "YMM, YMM, I8", imm_type="shuffle4"),
        _i("_mm256_permute_ps", "VPERMILPS", "YMM, YMM, I8", imm_type="shuffle4"),
        _i("_mm256_permute_pd", "VPERMILPD", "YMM, YMM, I8", imm_type="shuffle2"),
        _i("_mm256_permutexvar_epi32", "VPERMD", "YMM, YMM, YMM", cv_width=32),
        _i("_mm256_permutexvar_epi64", "VPERMQ", "YMM, YMM, YMM", cv_width=64),
        _i("_mm256_permutevar_ps", "VPERMILPS", "YMM, YMM, YMM", cv_width=32),
        _i("_mm256_permutevar_pd", "VPERMILPD", "YMM, YMM, YMM", cv_width=64),
        _i("_mm256_shuffle_ps", "VSHUFPS", "YMM, YMM, YMM, I8", imm_type="shuffle4"),
        _i("_mm256_shuffle_pd", "VSHUFPD", "YMM, YMM, YMM, I8", imm_type="shuffle2"),
        _i("_mm256_shuffle_epi32", "VPSHUFD", "YMM, YMM, I8"),
        _i("_mm256_unpacklo_epi32", "VPUNPCKLDQ", "YMM, YMM, YMM"),
        _i("_mm256_unpackhi_epi32", "VPUNPCKHDQ", "YMM, YMM, YMM"),
        _i("_mm256_unpacklo_epi64", "VPUNPCKLQDQ", "YMM, YMM, YMM"),
        _i("_mm256_unpackhi_epi64", "VPUNPCKHQDQ", "YMM, YMM, YMM"),
        _i("_mm256_unpacklo_ps", "VUNPCKLPS", "YMM, YMM, YMM"),
        _i("_mm256_unpackhi_ps", "VUNPCKHPS", "YMM, YMM, YMM"),
        _i("_mm256_permute2x128_si256", "VPERM2I128", "YMM, YMM, YMM, I8"),
        _i("_mm256_permute2f128_ps", "VPERM2F128", "YMM, YMM, YMM, I8"),
        _i("_mm256_blend_ps", "VBLENDPS", "YMM, YMM, YMM, I8", imm_type="binary"),
        _i("_mm256_blend_pd", "VBLENDPD", "YMM, YMM, YMM, I8", imm_type="binary"),
        _i("_mm256_blendv_ps", "VBLENDVPS", "YMM, YMM, YMM, YMM"),
        _i("_mm256_blendv_pd", "VBLENDVPD", "YMM, YMM, YMM, YMM"),
        _i("_mm256_blend_epi32", "VPBLENDD", "YMM, YMM, YMM, I8", imm_type="binary"),
        _i("_mm256_alignr_epi32", "VALIGND", "YMM, YMM, YMM, I8"),
        _i("_mm256_alignr_epi64", "VALIGNQ", "YMM, YMM, YMM, I8"),
        _i("_mm256_min_ps", "VMINPS", "YMM, YMM, YMM"),
        _i("_mm256_max_ps", "VMAXPS", "YMM, YMM, YMM"),
        # ── AVX512 (ZMM) ────────────────────────────────────────────────
        _i("_mm512_permutexvar_epi32", "VPERMD", "ZMM, ZMM, ZMM", cv_width=32),
        _i("_mm512_permutexvar_epi64", "VPERMQ", "ZMM, ZMM, ZMM", cv_width=64),
        _i("_mm512_permutex2var_epi32", "VPERMI2D", "ZMM, ZMM, ZMM", cv_width=32),
        _i("_mm512_permutex2var_epi64", "VPERMI2Q", "ZMM, ZMM, ZMM", cv_width=64),
        _i("_mm512_permute_ps", "VPERMILPS", "ZMM, ZMM, I8", imm_type="shuffle4"),
        _i("_mm512_permute_pd", "VPERMILPD", "ZMM, ZMM, I8", imm_type="shuffle2"),
        _i("_mm512_permutevar_ps", "VPERMILPS", "ZMM, ZMM, ZMM", cv_width=32),
        _i("_mm512_permutevar_pd", "VPERMILPD", "ZMM, ZMM, ZMM", cv_width=64),
        _i("_mm512_shuffle_ps", "VSHUFPS", "ZMM, ZMM, ZMM, I8", imm_type="shuffle4"),
        _i("_mm512_shuffle_pd", "VSHUFPD", "ZMM, ZMM, ZMM, I8", imm_type="shuffle2"),
        _i("_mm512_shuffle_epi32", "VPSHUFD", "ZMM, ZMM, I8"),
        _i(
            "_mm512_shuffle_i32x4",
            "VSHUFI32X4",
            "ZMM, ZMM, ZMM, I8",
            imm_type="shuffle4",
        ),
        _i("_mm512_unpacklo_epi32", "VPUNPCKLDQ", "ZMM, ZMM, ZMM"),
        _i("_mm512_unpackhi_epi32", "VPUNPCKHDQ", "ZMM, ZMM, ZMM"),
        _i("_mm512_unpacklo_epi64", "VPUNPCKLQDQ", "ZMM, ZMM, ZMM"),
        _i("_mm512_unpackhi_epi64", "VPUNPCKHQDQ", "ZMM, ZMM, ZMM"),
        _i("_mm512_alignr_epi32", "VALIGND", "ZMM, ZMM, ZMM, I8"),
        _i("_mm512_alignr_epi64", "VALIGNQ", "ZMM, ZMM, ZMM, I8"),
        _i("_mm512_min_ps", "VMINPS", "ZMM, ZMM, ZMM"),
        _i("_mm512_max_ps", "VMAXPS", "ZMM, ZMM, ZMM"),
        # ── AVX512 masked (ZMM with K mask) ─────────────────────────────
        _i("_mm512_mask_permutexvar_epi32", "VPERMD", "ZMM, K, ZMM, ZMM", cv_width=32),
        _i("_mm512_mask_permutexvar_epi64", "VPERMQ", "ZMM, K, ZMM, ZMM", cv_width=64),
        _i(
            "_mm512_mask_permutex2var_epi32",
            "VPERMI2D",
            "ZMM, K, ZMM, ZMM",
            cv_width=32,
        ),
        _i(
            "_mm512_mask_permutex2var_epi64",
            "VPERMI2Q",
            "ZMM, K, ZMM, ZMM",
            cv_width=64,
        ),
        _i(
            "_mm512_mask_permute_ps",
            "VPERMILPS",
            "ZMM, K, ZMM, I8",
            imm_type="shuffle4",
        ),
        _i(
            "_mm512_mask_permute_pd",
            "VPERMILPD",
            "ZMM, K, ZMM, I8",
            imm_type="shuffle2",
        ),
        _i("_mm512_mask_permutevar_ps", "VPERMILPS", "ZMM, K, ZMM, ZMM", cv_width=32),
        _i("_mm512_mask_permutevar_pd", "VPERMILPD", "ZMM, K, ZMM, ZMM", cv_width=64),
        _i(
            "_mm512_mask_shuffle_ps",
            "VSHUFPS",
            "ZMM, K, ZMM, ZMM, I8",
            imm_type="shuffle4",
        ),
        _i(
            "_mm512_mask_shuffle_pd",
            "VSHUFPD",
            "ZMM, K, ZMM, ZMM, I8",
            imm_type="shuffle2",
        ),
        _i(
            "_mm512_mask_shuffle_i32x4",
            "VSHUFI32X4",
            "ZMM, K, ZMM, ZMM, I8",
            imm_type="shuffle4",
        ),
        _i("_mm512_mask_unpacklo_epi32", "VPUNPCKLDQ", "ZMM, K, ZMM, ZMM"),
        _i("_mm512_mask_unpackhi_epi32", "VPUNPCKHDQ", "ZMM, K, ZMM, ZMM"),
        _i("_mm512_mask_unpacklo_epi64", "VPUNPCKLQDQ", "ZMM, K, ZMM, ZMM"),
        _i("_mm512_mask_unpackhi_epi64", "VPUNPCKHQDQ", "ZMM, K, ZMM, ZMM"),
        _i("_mm512_mask_blend_ps", "VBLENDMPS", "ZMM, K, ZMM, ZMM"),
        _i("_mm512_mask_blend_epi32", "VPBLENDMD", "ZMM, K, ZMM, ZMM"),
        _i("_mm512_mask_alignr_epi32", "VALIGND", "ZMM, K, ZMM, ZMM, I8"),
        _i("_mm512_mask_alignr_epi64", "VALIGNQ", "ZMM, K, ZMM, ZMM, I8"),
    ]

    return {e.intrinsic_name: e for e in entries}


def get_intrinsic_registry() -> dict[str, IntrinsicInfo]:
    """Return the cached singleton registry mapping intrinsic name to IntrinsicInfo."""
    return _build_registry()
