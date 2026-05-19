# permute_pd Report

Python source: `tests/test_z3_avx.py`
Planned Rust source: `rust/z3_avx/src/permute_pd.rs`
Planned Rust tests: `rust/z3_avx/tests/permute_pd.rs`

Intrinsics:
- `_mm256_permute_pd`
- `_mm512_permute_pd`
- `_mm512_mask_permute_pd`

Status: Ported.

Canonicalization:
- `imm8` live bits are `[1:0]`; constrain `[7:2]` for symbolic immediates.
- `_mm512_mask_permute_pd` must constrain k-mask bits above bit 7.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:225::TestPermutePd::test_mm256_permute_epi64_null_permute_works` | `rust/z3_avx/tests/permute_pd.rs::test_mm256_permute_epi64_null_permute_works` | Ported |
| `tests/test_z3_avx.py:237::TestPermutePd::test_mm256_permute_epi64_null_permute_found` | `rust/z3_avx/tests/permute_pd.rs::test_mm256_permute_epi64_null_permute_found` | Ported |
| `tests/test_z3_avx.py:253::TestPermutePd::test_mm512_permute_epi64_null_permute_works` | `rust/z3_avx/tests/permute_pd.rs::test_mm512_permute_epi64_null_permute_works` | Ported |
| `tests/test_z3_avx.py:266::TestPermutePd::test_mm512_permute_epi64_null_permute_found` | `rust/z3_avx/tests/permute_pd.rs::test_mm512_permute_epi64_null_permute_found` | Ported |
| `tests/test_z3_avx.py:3243::TestMaskPermutePd::test_mm512_mask_permute_pd_mask_all_zeros` | `rust/z3_avx/tests/permute_pd.rs::test_mm512_mask_permute_pd_mask_all_zeros` | Ported |
| `tests/test_z3_avx.py:3262::TestMaskPermutePd::test_mm512_mask_permute_pd_mask_all_ones` | `rust/z3_avx/tests/permute_pd.rs::test_mm512_mask_permute_pd_mask_all_ones` | Ported |
| `tests/test_z3_avx.py:3282::TestMaskPermutePd::test_mm512_mask_permute_pd_single_bit_mask` | `rust/z3_avx/tests/permute_pd.rs::test_mm512_mask_permute_pd_single_bit_mask` | Ported |

Additional Rust tests:

| Requirement | Rust test | Status |
|---|---|---|
| `imm8` canonicalizes dead bits `[7:2]` | `rust/z3_avx/tests/permute_pd.rs::test_permute_pd_256_canonicalization` | Ported |
| `_mm512_mask_permute_pd` canonicalizes k-mask bits above bit 7 | `rust/z3_avx/tests/permute_pd.rs::test_mm512_mask_permute_pd_canonicalizes_dead_mask_bits` | Ported |
