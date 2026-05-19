# permutexvar Report

Python source: `tests/test_z3_avx.py`
Rust source: `rust/z3_avx/src/permutexvar.rs`
Rust tests:
- `rust/z3_avx/tests/permutexvar_epi32.rs`
- `rust/z3_avx/tests/permutexvar_epi64.rs`

Intrinsics:
- `_mm256_permutexvar_epi32`
- `_mm512_permutexvar_epi32`
- `_mm512_mask_permutexvar_epi32`
- `_mm256_permutexvar_epi64`
- `_mm512_permutexvar_epi64`
- `_mm512_mask_permutexvar_epi64`

Status: Ported.

Canonicalization:
- epi32 YMM controls keep `[2:0]`; epi32 ZMM controls keep `[3:0]`.
- epi64 YMM controls keep `[1:0]`; epi64 ZMM controls keep `[2:0]`.
- Masked epi32 constrains k-mask bits above bit 15.
- Masked epi64 constrains k-mask bits above bit 7.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:286::TestPermutexvarEpi32::test_mm256_permutexvar_epi32_null_permute_works` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm256_permutexvar_epi32_null_permute_works` | Implemented |
| `tests/test_z3_avx.py:302::TestPermutexvarEpi32::test_mm256_permutexvar_epi32_null_permute_found` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm256_permutexvar_epi32_null_permute_found` | Implemented |
| `tests/test_z3_avx.py:319::TestPermutexvarEpi32::test_mm256_permutexvar_epi32_reverse_permute_found` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm256_permutexvar_epi32_reverse_permute_found` | Implemented |
| `tests/test_z3_avx.py:338::TestPermutexvarEpi32::test_mm512_permutexvar_epi32_null_permute_works` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_permutexvar_epi32_null_permute_works` | Implemented |
| `tests/test_z3_avx.py:359::TestPermutexvarEpi32::test_mm512_permutexvar_epi32_null_permute_found` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_permutexvar_epi32_null_permute_found` | Implemented |
| `tests/test_z3_avx.py:377::TestPermutexvarEpi32::test_mm512_permutexvar_epi32_reverse_permute_found` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_permutexvar_epi32_reverse_permute_found` | Implemented |
| `tests/test_z3_avx.py:511::TestMaskPermutexvarEpi32::test_mm512_mask_permutexvar_epi32_mask_all_zeros` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_mask_permutexvar_epi32_mask_all_zeros` | Implemented |
| `tests/test_z3_avx.py:534::TestMaskPermutexvarEpi32::test_mm512_mask_permutexvar_epi32_mask_all_ones` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_mask_permutexvar_epi32_mask_all_ones` | Implemented |
| `tests/test_z3_avx.py:560::TestMaskPermutexvarEpi32::test_mm512_mask_permutexvar_epi32_alternating_mask` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_mask_permutexvar_epi32_alternating_mask` | Implemented |
| `tests/test_z3_avx.py:594::TestMaskPermutexvarEpi32::test_mm512_mask_permutexvar_epi32_single_bit_mask` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_mask_permutexvar_epi32_single_bit_mask` | Implemented |
| `tests/test_z3_avx.py:628::TestMaskPermutexvarEpi32::test_mm512_mask_permutexvar_epi32_partial_mask` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_mask_permutexvar_epi32_partial_mask` | Implemented |
| `tests/test_z3_avx.py:663::TestMaskPermutexvarEpi32::test_mm512_mask_permutexvar_epi32_find_mask_for_identity` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_mask_permutexvar_epi32_find_mask_for_identity` | Implemented |
| `tests/test_z3_avx.py:689::TestMaskPermutexvarEpi32::test_mm512_mask_permutexvar_epi32_find_mask_for_full_permute` | `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_mask_permutexvar_epi32_find_mask_for_full_permute` | Implemented |
| `tests/test_z3_avx.py:402::TestPermutexvarEpi64::test_mm256_permutexvar_epi64_null_permute_works` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm256_permutexvar_epi64_null_permute_works` | Implemented |
| `tests/test_z3_avx.py:417::TestPermutexvarEpi64::test_mm256_permutexvar_epi64_null_permute_found` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm256_permutexvar_epi64_null_permute_found` | Implemented |
| `tests/test_z3_avx.py:434::TestPermutexvarEpi64::test_mm256_permutexvar_epi64_reverse_permute_found` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm256_permutexvar_epi64_reverse_permute_found` | Implemented |
| `tests/test_z3_avx.py:453::TestPermutexvarEpi64::test_mm512_permutexvar_epi64_null_permute_works` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_permutexvar_epi64_null_permute_works` | Implemented |
| `tests/test_z3_avx.py:471::TestPermutexvarEpi64::test_mm512_permutexvar_epi64_null_permute_found` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_permutexvar_epi64_null_permute_found` | Implemented |
| `tests/test_z3_avx.py:488::TestPermutexvarEpi64::test_mm512_permutexvar_epi64_reverse_permute_found` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_permutexvar_epi64_reverse_permute_found` | Implemented |
| `tests/test_z3_avx.py:719::TestMaskPermutexvarEpi64::test_mm512_mask_permutexvar_epi64_mask_all_zeros` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_mask_permutexvar_epi64_mask_all_zeros` | Implemented |
| `tests/test_z3_avx.py:742::TestMaskPermutexvarEpi64::test_mm512_mask_permutexvar_epi64_mask_all_ones` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_mask_permutexvar_epi64_mask_all_ones` | Implemented |
| `tests/test_z3_avx.py:768::TestMaskPermutexvarEpi64::test_mm512_mask_permutexvar_epi64_alternating_mask` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_mask_permutexvar_epi64_alternating_mask` | Implemented |
| `tests/test_z3_avx.py:802::TestMaskPermutexvarEpi64::test_mm512_mask_permutexvar_epi64_single_bit_mask` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_mask_permutexvar_epi64_single_bit_mask` | Implemented |
| `tests/test_z3_avx.py:836::TestMaskPermutexvarEpi64::test_mm512_mask_permutexvar_epi64_partial_mask` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_mask_permutexvar_epi64_partial_mask` | Implemented |
| `tests/test_z3_avx.py:871::TestMaskPermutexvarEpi64::test_mm512_mask_permutexvar_epi64_find_mask_for_identity` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_mask_permutexvar_epi64_find_mask_for_identity` | Implemented |
| `tests/test_z3_avx.py:897::TestMaskPermutexvarEpi64::test_mm512_mask_permutexvar_epi64_find_mask_for_full_permute` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_mask_permutexvar_epi64_find_mask_for_full_permute` | Implemented |
| `tests/test_z3_avx.py:923::TestMaskPermutexvarEpi64::test_mm512_mask_permutexvar_epi64_find_indices_and_mask` | `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_mask_permutexvar_epi64_find_indices_and_mask` | Implemented |

Additional Rust-only canonicalization tests:
- `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm256_permutexvar_epi32_canonicalizes_dead_index_bits`
- `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_permutexvar_epi32_canonicalizes_dead_index_bits`
- `rust/z3_avx/tests/permutexvar_epi32.rs::test_mm512_mask_permutexvar_epi32_canonicalizes_dead_mask_bits`
- `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm256_permutexvar_epi64_canonicalizes_dead_index_bits`
- `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_permutexvar_epi64_canonicalizes_dead_index_bits`
- `rust/z3_avx/tests/permutexvar_epi64.rs::test_mm512_mask_permutexvar_epi64_canonicalizes_dead_mask_bits`
