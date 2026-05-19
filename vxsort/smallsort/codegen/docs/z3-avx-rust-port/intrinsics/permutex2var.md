# permutex2var Report

Python source: `tests/test_z3_avx.py`
Planned Rust source: `rust/z3_avx/src/permutex2var.rs`
Planned Rust tests:
- `rust/z3_avx/tests/permutex2var_epi32.rs`
- `rust/z3_avx/tests/permutex2var_epi64.rs`

Intrinsics:
- `_mm512_permutex2var_epi32`
- `_mm512_mask_permutex2var_epi32`
- `_mm512_permutex2var_epi64`
- `_mm512_mask_permutex2var_epi64`

Status: Ported.

Canonicalization:
- epi32 controls keep offset bits `[3:0]` plus source bit `[4]`; constrain `[31:5]`.
- epi64 controls keep offset bits `[2:0]` plus source bit `[3]`; constrain `[63:4]`.
- Masked epi32 constrains k-mask bits above bit 15.
- Masked epi64 constrains k-mask bits above bit 7.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:966::TestPermutex2varEpi32::test_mm512_permutex2var_epi32_null_permute_works` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_permutex2var_epi32_null_permute_works` | Done |
| `tests/test_z3_avx.py:986::TestPermutex2varEpi32::test_mm512_permutex2var_epi32_null_permute_found` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_permutex2var_epi32_null_permute_found` | Done |
| `tests/test_z3_avx.py:1003::TestPermutex2varEpi32::test_mm512_permutex2var_epi32_select_from_b` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_permutex2var_epi32_select_from_b` | Done |
| `tests/test_z3_avx.py:1019::TestPermutex2varEpi32::test_mm512_permutex2var_epi32_reverse_permute_from_a` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_permutex2var_epi32_reverse_permute_from_a` | Done |
| `tests/test_z3_avx.py:1040::TestPermutex2varEpi32::test_mm512_permutex2var_epi32_mixed_sources` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_permutex2var_epi32_mixed_sources` | Done |
| `tests/test_z3_avx.py:2021::TestMaskPermutex2varEpi32::test_mm512_mask_permutex2var_epi32_mask_all_zeros` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_mask_all_zeros` | Done |
| `tests/test_z3_avx.py:2041::TestMaskPermutex2varEpi32::test_mm512_mask_permutex2var_epi32_mask_all_ones` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_mask_all_ones` | Done |
| `tests/test_z3_avx.py:2065::TestMaskPermutex2varEpi32::test_mm512_mask_permutex2var_epi32_alternating_mask` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_alternating_mask` | Done |
| `tests/test_z3_avx.py:2087::TestMaskPermutex2varEpi32::test_mm512_mask_permutex2var_epi32_reverse_with_partial_mask` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_reverse_with_partial_mask` | Done |
| `tests/test_z3_avx.py:2113::TestMaskPermutex2varEpi32::test_mm512_mask_permutex2var_epi32_mixed_sources_with_mask` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_mixed_sources_with_mask` | Done |
| `tests/test_z3_avx.py:2138::TestMaskPermutex2varEpi32::test_mm512_mask_permutex2var_epi32_single_bit_mask` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_single_bit_mask` | Done |
| `tests/test_z3_avx.py:2162::TestMaskPermutex2varEpi32::test_mm512_mask_permutex2var_epi32_find_identity_mask` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_find_identity_mask` | Done |
| `tests/test_z3_avx.py:2185::TestMaskPermutex2varEpi32::test_mm512_mask_permutex2var_epi32_find_full_permute_mask` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_find_full_permute_mask` | Done |
| `tests/test_z3_avx.py:2208::TestMaskPermutex2varEpi32::test_mm512_mask_permutex2var_epi32_find_partial_mask` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_find_partial_mask` | Done |
| `tests/test_z3_avx.py:2240::TestMaskPermutex2varEpi32::test_mm512_mask_permutex2var_epi32_find_indices_with_mask` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_find_indices_with_mask` | Done |
| `tests/test_z3_avx.py:2271::TestMaskPermutex2varEpi32::test_mm512_mask_permutex2var_epi32_find_reverse_partial` | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_find_reverse_partial` | Done |
| `tests/test_z3_avx.py:1080::TestPermutex2varEpi64::test_mm512_permutex2var_epi64_null_permute_works` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_permutex2var_epi64_null_permute_works` | Done |
| `tests/test_z3_avx.py:1098::TestPermutex2varEpi64::test_mm512_permutex2var_epi64_null_permute_found` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_permutex2var_epi64_null_permute_found` | Done |
| `tests/test_z3_avx.py:1115::TestPermutex2varEpi64::test_mm512_permutex2var_epi64_select_from_b` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_permutex2var_epi64_select_from_b` | Done |
| `tests/test_z3_avx.py:1130::TestPermutex2varEpi64::test_mm512_permutex2var_epi64_reverse_permute_from_a` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_permutex2var_epi64_reverse_permute_from_a` | Done |
| `tests/test_z3_avx.py:1149::TestPermutex2varEpi64::test_mm512_permutex2var_epi64_mixed_sources` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_permutex2var_epi64_mixed_sources` | Done |
| `tests/test_z3_avx.py:2303::TestMaskPermutex2varEpi64::test_mm512_mask_permutex2var_epi64_mask_all_zeros` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_mask_all_zeros` | Done |
| `tests/test_z3_avx.py:2324::TestMaskPermutex2varEpi64::test_mm512_mask_permutex2var_epi64_mask_all_ones` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_mask_all_ones` | Done |
| `tests/test_z3_avx.py:2349::TestMaskPermutex2varEpi64::test_mm512_mask_permutex2var_epi64_alternating_mask` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_alternating_mask` | Done |
| `tests/test_z3_avx.py:2378::TestMaskPermutex2varEpi64::test_mm512_mask_permutex2var_epi64_single_bit_mask` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_single_bit_mask` | Done |
| `tests/test_z3_avx.py:2403::TestMaskPermutex2varEpi64::test_mm512_mask_permutex2var_epi64_partial_mask` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_partial_mask` | Done |
| `tests/test_z3_avx.py:2432::TestMaskPermutex2varEpi64::test_mm512_mask_permutex2var_epi64_mixed_sources_with_mask` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_mixed_sources_with_mask` | Done |
| `tests/test_z3_avx.py:2458::TestMaskPermutex2varEpi64::test_mm512_mask_permutex2var_epi64_find_identity_mask` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_find_identity_mask` | Done |
| `tests/test_z3_avx.py:2477::TestMaskPermutex2varEpi64::test_mm512_mask_permutex2var_epi64_find_full_permute_mask` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_find_full_permute_mask` | Done |
| `tests/test_z3_avx.py:2501::TestMaskPermutex2varEpi64::test_mm512_mask_permutex2var_epi64_find_partial_mask` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_find_partial_mask` | Done |
| `tests/test_z3_avx.py:2534::TestMaskPermutex2varEpi64::test_mm512_mask_permutex2var_epi64_find_indices_with_mask` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_find_indices_with_mask` | Done |
| `tests/test_z3_avx.py:2565::TestMaskPermutex2varEpi64::test_mm512_mask_permutex2var_epi64_cross_source_reverse` | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_cross_source_reverse` | Done |
| Rust-only canonicalization regression | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_permutex2var_epi32_canonicalizes_dead_index_bits` | Done |
| Rust-only canonicalization regression | `rust/z3_avx/tests/permutex2var_epi32.rs::test_mm512_mask_permutex2var_epi32_canonicalizes_dead_index_and_wide_mask_bits` | Done |
| Rust-only canonicalization regression | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_permutex2var_epi64_canonicalizes_dead_index_bits` | Done |
| Rust-only canonicalization regression | `rust/z3_avx/tests/permutex2var_epi64.rs::test_mm512_mask_permutex2var_epi64_canonicalizes_dead_index_and_wide_mask_bits` | Done |
