# shuffle_i32x4 Report

Python source: `tests/test_z3_avx.py`
Planned Rust source: `rust/z3_avx/src/shuffle_i32x4.rs`
Planned Rust tests: `rust/z3_avx/tests/shuffle_i32x4.rs`

Intrinsics:
- `_mm512_shuffle_i32x4`
- `_mm512_mask_shuffle_i32x4`

Status: Ported.

Canonicalization:
- `imm8` uses all 8 bits.
- `_mm512_mask_shuffle_i32x4` must constrain k-mask bits above bit 15.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:1908::TestShuffleI32x4::test_mm512_shuffle_i32x4_null_permute_works` | `rust/z3_avx/tests/shuffle_i32x4.rs::test_mm512_shuffle_i32x4_null_permute_works` | Ported |
| `tests/test_z3_avx.py:1923::TestShuffleI32x4::test_mm512_shuffle_i32x4_null_permute_found` | `rust/z3_avx/tests/shuffle_i32x4.rs::test_mm512_shuffle_i32x4_null_permute_found` | Ported |
| `tests/test_z3_avx.py:1940::TestShuffleI32x4::test_mm512_shuffle_i32x4_null_permute_2vec_works` | `rust/z3_avx/tests/shuffle_i32x4.rs::test_mm512_shuffle_i32x4_null_permute_2vec_works` | Ported |
| `tests/test_z3_avx.py:1964::TestShuffleI32x4::test_mm512_shuffle_i32x4_null_permute_2vec_found` | `rust/z3_avx/tests/shuffle_i32x4.rs::test_mm512_shuffle_i32x4_null_permute_2vec_found` | Ported |
| `tests/test_z3_avx.py:1992::TestShuffleI32x4::test_mm512_shuffle_i32x4_cross_lanes` | `rust/z3_avx/tests/shuffle_i32x4.rs::test_mm512_shuffle_i32x4_cross_lanes` | Ported |
| `tests/test_z3_avx.py:3384::TestMaskShuffleI32x4::test_mask_all_ones_equals_unmasked` | `rust/z3_avx/tests/shuffle_i32x4.rs::test_mask_all_ones_equals_unmasked` | Ported |
| `tests/test_z3_avx.py:3405::TestMaskShuffleI32x4::test_mask_all_zeros_preserves_src` | `rust/z3_avx/tests/shuffle_i32x4.rs::test_mask_all_zeros_preserves_src` | Ported |
| `tests/test_z3_avx.py:3424::TestMaskShuffleI32x4::test_partial_mask` | `rust/z3_avx/tests/shuffle_i32x4.rs::test_partial_mask` | Ported |
