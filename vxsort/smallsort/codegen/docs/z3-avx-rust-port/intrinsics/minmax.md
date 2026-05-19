# minmax Report

Python source: `tests/test_z3_avx.py`
Planned Rust source: `rust/z3_avx/src/minmax.rs`
Planned Rust tests:
- `rust/z3_avx/tests/minmax_epi32.rs`
- `rust/z3_avx/tests/minmax_epi64.rs`

Intrinsics:
- `_mm256_min_epi32`, `_mm256_max_epi32`
- `_mm512_min_epi32`, `_mm512_max_epi32`
- `_mm256_min_epi64`, `_mm256_max_epi64`
- `_mm512_min_epi64`, `_mm512_max_epi64`

Status: Ported.

Canonicalization: none.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:5940::TestMinMaxEpi32::test_mm256_min_epi32_concrete` | `rust/z3_avx/tests/minmax_epi32.rs::test_mm256_min_epi32_concrete` | Ported |
| `tests/test_z3_avx.py:5951::TestMinMaxEpi32::test_mm256_max_epi32_concrete` | `rust/z3_avx/tests/minmax_epi32.rs::test_mm256_max_epi32_concrete` | Ported |
| `tests/test_z3_avx.py:5962::TestMinMaxEpi32::test_mm256_min_epi32_signed` | `rust/z3_avx/tests/minmax_epi32.rs::test_mm256_min_epi32_signed` | Ported |
| `tests/test_z3_avx.py:5974::TestMinMaxEpi32::test_mm256_minmax_epi32_preserves_elements` | `rust/z3_avx/tests/minmax_epi32.rs::test_mm256_minmax_epi32_preserves_elements` | Ported |
| `tests/test_z3_avx.py:6000::TestMinMaxEpi32::test_mm512_min_epi32_concrete` | `rust/z3_avx/tests/minmax_epi32.rs::test_mm512_min_epi32_concrete` | Ported |
| `tests/test_z3_avx.py:6018::TestMinMaxEpi32::test_mm512_max_epi32_concrete` | `rust/z3_avx/tests/minmax_epi32.rs::test_mm512_max_epi32_concrete` | Ported |
| `tests/test_z3_avx.py:6040::TestMinMaxEpi64::test_mm256_min_epi64_concrete` | `rust/z3_avx/tests/minmax_epi64.rs::test_mm256_min_epi64_concrete` | Ported |
| `tests/test_z3_avx.py:6051::TestMinMaxEpi64::test_mm256_max_epi64_concrete` | `rust/z3_avx/tests/minmax_epi64.rs::test_mm256_max_epi64_concrete` | Ported |
| `tests/test_z3_avx.py:6062::TestMinMaxEpi64::test_mm256_min_epi64_signed` | `rust/z3_avx/tests/minmax_epi64.rs::test_mm256_min_epi64_signed` | Ported |
| `tests/test_z3_avx.py:6074::TestMinMaxEpi64::test_mm256_minmax_epi64_preserves_elements` | `rust/z3_avx/tests/minmax_epi64.rs::test_mm256_minmax_epi64_preserves_elements` | Ported |
| `tests/test_z3_avx.py:6101::TestMinMaxEpi64::test_mm512_min_epi64_concrete` | `rust/z3_avx/tests/minmax_epi64.rs::test_mm512_min_epi64_concrete` | Ported |
| `tests/test_z3_avx.py:6112::TestMinMaxEpi64::test_mm512_max_epi64_concrete` | `rust/z3_avx/tests/minmax_epi64.rs::test_mm512_max_epi64_concrete` | Ported |
