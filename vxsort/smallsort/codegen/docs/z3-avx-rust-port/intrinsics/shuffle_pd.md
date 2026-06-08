# shuffle_pd Report

Python source: `tests/test_z3_avx.py`
Planned Rust source: `z3_avx/src/shuffle_pd.rs`
Planned Rust tests: `z3_avx/tests/shuffle_pd.rs`

Intrinsics:
- `_mm256_shuffle_pd`
- `_mm512_shuffle_pd`
- `_mm512_mask_shuffle_pd`

Status: Ported.

Canonicalization:
- `_mm256_shuffle_pd` uses imm bits `[3:0]`; constrain `[7:4]`.
- `_mm512_shuffle_pd` uses all 8 bits.
- `_mm512_mask_shuffle_pd` must constrain k-mask bits above bit 7.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:1556::TestShufflePd::test_mm256_shuffle_pd_null_permute_works` | `z3_avx/tests/shuffle_pd.rs::test_mm256_shuffle_pd_null_permute_works` | Ported |
| `tests/test_z3_avx.py:1570::TestShufflePd::test_mm256_shuffle_pd_null_permute_found` | `z3_avx/tests/shuffle_pd.rs::test_mm256_shuffle_pd_null_permute_found` | Ported |
| `tests/test_z3_avx.py:1587::TestShufflePd::test_mm256_shuffle_pd_null_permute_2vec_works` | `z3_avx/tests/shuffle_pd.rs::test_mm256_shuffle_pd_null_permute_2vec_works` | Ported |
| `tests/test_z3_avx.py:1603::TestShufflePd::test_mm256_shuffle_pd_null_permute_2vec_found` | `z3_avx/tests/shuffle_pd.rs::test_mm256_shuffle_pd_null_permute_2vec_found` | Ported |
| `tests/test_z3_avx.py:1623::TestShufflePd::test_mm512_shuffle_pd_null_permute_works` | `z3_avx/tests/shuffle_pd.rs::test_mm512_shuffle_pd_null_permute_works` | Ported |
| `tests/test_z3_avx.py:1638::TestShufflePd::test_mm512_shuffle_pd_null_permute_found` | `z3_avx/tests/shuffle_pd.rs::test_mm512_shuffle_pd_null_permute_found` | Ported |
| `tests/test_z3_avx.py:1655::TestShufflePd::test_mm512_shuffle_pd_null_permute_2vec_works` | `z3_avx/tests/shuffle_pd.rs::test_mm512_shuffle_pd_null_permute_2vec_works` | Ported |
| `tests/test_z3_avx.py:1685::TestShufflePd::test_mm512_shuffle_pd_null_permute_2vec_found` | `z3_avx/tests/shuffle_pd.rs::test_mm512_shuffle_pd_null_permute_2vec_found` | Ported |
| `tests/test_z3_avx.py:3459::TestMaskShufflePd::test_mm512_mask_shuffle_pd_mask_all_zeros` | `z3_avx/tests/shuffle_pd.rs::test_mm512_mask_shuffle_pd_mask_all_zeros` | Ported |
| `tests/test_z3_avx.py:3478::TestMaskShufflePd::test_mm512_mask_shuffle_pd_mask_all_ones` | `z3_avx/tests/shuffle_pd.rs::test_mm512_mask_shuffle_pd_mask_all_ones` | Ported |
| `tests/test_z3_avx.py:3500::TestMaskShufflePd::test_mm512_mask_shuffle_pd_alternating_mask` | `z3_avx/tests/shuffle_pd.rs::test_mm512_mask_shuffle_pd_alternating_mask` | Ported |
