# permute2x128 Report

Python source: `tests/test_z3_avx.py`
Planned Rust source: `rust/z3_avx/src/permute2x128.rs`
Planned Rust tests: `rust/z3_avx/tests/permute2x128.rs`

Intrinsic:
- `_mm256_permute2x128_si256`

Status: Ported.

Canonicalization:
- In each imm nibble, source bits `[1:0]` and zero bit `[3]` are live.
- Dead bits `2` and `6` must be constrained to zero for symbolic immediates.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:1721::TestPermute2x128Si256::test_mm256_permute2x128_si256_null_permute_works` | `rust/z3_avx/tests/permute2x128.rs::test_mm256_permute2x128_si256_null_permute_works` | Ported |
| `tests/test_z3_avx.py:1736::TestPermute2x128Si256::test_mm256_permute2x128_si256_null_permute_found` | `rust/z3_avx/tests/permute2x128.rs::test_mm256_permute2x128_si256_null_permute_found` | Ported |
| `tests/test_z3_avx.py:1764::TestPermute2x128Si256::test_mm256_permute2x128_si256_null_permute_2vec_works` | `rust/z3_avx/tests/permute2x128.rs::test_mm256_permute2x128_si256_null_permute_2vec_works` | Ported |
| `tests/test_z3_avx.py:1788::TestPermute2x128Si256::test_mm256_permute2x128_si256_null_permute_2vec_found` | `rust/z3_avx/tests/permute2x128.rs::test_mm256_permute2x128_si256_null_permute_2vec_found` | Ported |
| `tests/test_z3_avx.py:1816::TestPermute2x128Si256::test_mm256_permute2x128_si256_swap_lanes` | `rust/z3_avx/tests/permute2x128.rs::test_mm256_permute2x128_si256_swap_lanes` | Ported |
| `tests/test_z3_avx.py:1841::TestPermute2x128Si256::test_mm256_permute2x128_si256_cross_vector` | `rust/z3_avx/tests/permute2x128.rs::test_mm256_permute2x128_si256_cross_vector` | Ported |
| `tests/test_z3_avx.py:1864::TestPermute2x128Si256::test_mm256_permute2x128_si256_zero_lanes` | `rust/z3_avx/tests/permute2x128.rs::test_mm256_permute2x128_si256_zero_lanes` | Ported |
| `tests/test_z3_avx.py:1885::TestPermute2x128Si256::test_mm256_permute2x128_si256_zero_both_lanes` | `rust/z3_avx/tests/permute2x128.rs::test_mm256_permute2x128_si256_zero_both_lanes` | Ported |
