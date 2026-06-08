# unpack Report

Python source: `tests/test_z3_avx.py`
Planned Rust source: `z3_avx/src/unpack.rs`
Planned Rust tests:
- `z3_avx/tests/unpack_epi32.rs`
- `z3_avx/tests/unpack_epi64.rs`

Intrinsics:
- `_mm256_unpacklo_epi32`, `_mm256_unpackhi_epi32`
- `_mm512_unpacklo_epi32`, `_mm512_unpackhi_epi32`
- `_mm512_mask_unpacklo_epi32`, `_mm512_mask_unpackhi_epi32`
- `_mm256_unpacklo_epi64`, `_mm256_unpackhi_epi64`
- `_mm512_unpacklo_epi64`, `_mm512_unpackhi_epi64`
- `_mm512_mask_unpacklo_epi64`, `_mm512_mask_unpackhi_epi64`

Status: Ported.

Canonicalization:
- No imm/control canonicalization.
- Masked epi32 variants must constrain k-mask bits above bit 15.
- Masked epi64 variants must constrain k-mask bits above bit 7.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:2605::TestUnpackEpi32::test_mm256_unpacklo_epi32_basic` | `z3_avx/tests/unpack_epi32.rs::test_mm256_unpacklo_epi32_basic` | Ported |
| `tests/test_z3_avx.py:2649::TestUnpackEpi32::test_mm256_unpackhi_epi32_basic` | `z3_avx/tests/unpack_epi32.rs::test_mm256_unpackhi_epi32_basic` | Ported |
| `tests/test_z3_avx.py:2691::TestUnpackEpi32::test_mm512_unpacklo_epi32_basic` | `z3_avx/tests/unpack_epi32.rs::test_mm512_unpacklo_epi32_basic` | Ported |
| `tests/test_z3_avx.py:2768::TestUnpackEpi32::test_mm512_unpackhi_epi32_basic` | `z3_avx/tests/unpack_epi32.rs::test_mm512_unpackhi_epi32_basic` | Ported |
| `tests/test_z3_avx.py:2845::TestUnpackEpi32::test_mm256_unpacklo_epi32_identity_check` | `z3_avx/tests/unpack_epi32.rs::test_mm256_unpacklo_epi32_identity_check` | Ported |
| `tests/test_z3_avx.py:2874::TestUnpackEpi32::test_mm256_unpackhi_epi32_identity_check` | `z3_avx/tests/unpack_epi32.rs::test_mm256_unpackhi_epi32_identity_check` | Ported |
| `tests/test_z3_avx.py:2903::TestUnpackEpi32::test_mm512_mask_unpacklo_epi32_mask_all_zeros` | `z3_avx/tests/unpack_epi32.rs::test_mm512_mask_unpacklo_epi32_mask_all_zeros` | Ported |
| `tests/test_z3_avx.py:2920::TestUnpackEpi32::test_mm512_mask_unpacklo_epi32_mask_all_ones` | `z3_avx/tests/unpack_epi32.rs::test_mm512_mask_unpacklo_epi32_mask_all_ones` | Ported |
| `tests/test_z3_avx.py:2938::TestUnpackEpi32::test_mm512_mask_unpackhi_epi32_alternating_mask` | `z3_avx/tests/unpack_epi32.rs::test_mm512_mask_unpackhi_epi32_alternating_mask` | Ported |
| `tests/test_z3_avx.py:2968::TestUnpackEpi32::test_mm512_mask_unpacklo_epi32_single_bit_mask` | `z3_avx/tests/unpack_epi32.rs::test_mm512_mask_unpacklo_epi32_single_bit_mask` | Ported |
| `tests/test_z3_avx.py:2996::TestUnpackEpi32::test_mm256_unpacklo_epi32_reconstruct_pattern` | `z3_avx/tests/unpack_epi32.rs::test_mm256_unpacklo_epi32_reconstruct_pattern` | Ported |
| `tests/test_z3_avx.py:3031::TestUnpackEpi32::test_mm512_mask_unpackhi_epi32_find_mask` | `z3_avx/tests/unpack_epi32.rs::test_mm512_mask_unpackhi_epi32_find_mask` | Ported |
| `tests/test_z3_avx.py:3062::TestUnpackEpi32::test_mm256_unpack_combo_lo_hi` | `z3_avx/tests/unpack_epi32.rs::test_mm256_unpack_combo_lo_hi` | Ported |
| `tests/test_z3_avx.py:3091::TestUnpackEpi32::test_mm512_unpack_lane_independence` | `z3_avx/tests/unpack_epi32.rs::test_mm512_unpack_lane_independence` | Ported |
| `tests/test_z3_avx.py:5763::TestUnpackEpi64::test_mm256_unpacklo_epi64_interleaves_low_elements` | `z3_avx/tests/unpack_epi64.rs::test_mm256_unpacklo_epi64_interleaves_low_elements` | Ported |
| `tests/test_z3_avx.py:5777::TestUnpackEpi64::test_mm256_unpackhi_epi64_interleaves_high_elements` | `z3_avx/tests/unpack_epi64.rs::test_mm256_unpackhi_epi64_interleaves_high_elements` | Ported |
| `tests/test_z3_avx.py:5791::TestUnpackEpi64::test_mm512_unpacklo_epi64_interleaves_low_elements` | `z3_avx/tests/unpack_epi64.rs::test_mm512_unpacklo_epi64_interleaves_low_elements` | Ported |
| `tests/test_z3_avx.py:5807::TestUnpackEpi64::test_mm512_unpackhi_epi64_interleaves_high_elements` | `z3_avx/tests/unpack_epi64.rs::test_mm512_unpackhi_epi64_interleaves_high_elements` | Ported |
| `tests/test_z3_avx.py:5823::TestUnpackEpi64::test_mm512_mask_unpacklo_epi64_applies_writemask` | `z3_avx/tests/unpack_epi64.rs::test_mm512_mask_unpacklo_epi64_applies_writemask` | Ported |
| `tests/test_z3_avx.py:5847::TestUnpackEpi64::test_mm512_mask_unpackhi_epi64_applies_writemask` | `z3_avx/tests/unpack_epi64.rs::test_mm512_mask_unpackhi_epi64_applies_writemask` | Ported |
| Rust-only k-mask regression | `z3_avx/tests/unpack_epi32.rs::test_mm512_mask_unpacklo_epi32_canonicalizes_dead_mask_bits` | Ported |
| Rust-only k-mask regression | `z3_avx/tests/unpack_epi64.rs::test_mm512_mask_unpacklo_epi64_canonicalizes_dead_mask_bits` | Ported |
