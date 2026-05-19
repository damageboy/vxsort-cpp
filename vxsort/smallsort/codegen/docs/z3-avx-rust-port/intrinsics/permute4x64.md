# permute4x64 Report

Python source: `tests/test_z3_avx.py`
Planned Rust source: `rust/z3_avx/src/permute4x64.rs`
Planned Rust tests: `rust/z3_avx/tests/permute4x64_epi64.rs`

Intrinsic:
- `_mm256_permute4x64_epi64`

Status: Ported.

Canonicalization:
- All 8 imm bits are live.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:4881::TestPermute4x64Epi64::test_mm256_permute4x64_epi64_identity` | `rust/z3_avx/tests/permute4x64_epi64.rs::test_mm256_permute4x64_epi64_identity` | Ported |
| `tests/test_z3_avx.py:4900::TestPermute4x64Epi64::test_mm256_permute4x64_epi64_reverse` | `rust/z3_avx/tests/permute4x64_epi64.rs::test_mm256_permute4x64_epi64_reverse` | Ported |
| `tests/test_z3_avx.py:4922::TestPermute4x64Epi64::test_mm256_permute4x64_epi64_broadcast_first` | `rust/z3_avx/tests/permute4x64_epi64.rs::test_mm256_permute4x64_epi64_broadcast_first` | Ported |
| `tests/test_z3_avx.py:4949::TestPermute4x64Epi64::test_mm256_permute4x64_epi64_broadcast_last` | `rust/z3_avx/tests/permute4x64_epi64.rs::test_mm256_permute4x64_epi64_broadcast_last` | Ported |
| `tests/test_z3_avx.py:4976::TestPermute4x64Epi64::test_mm256_permute4x64_epi64_swap_pairs` | `rust/z3_avx/tests/permute4x64_epi64.rs::test_mm256_permute4x64_epi64_swap_pairs` | Ported |
| `tests/test_z3_avx.py:5004::TestPermute4x64Epi64::test_mm256_permute4x64_epi64_swap_halves` | `rust/z3_avx/tests/permute4x64_epi64.rs::test_mm256_permute4x64_epi64_swap_halves` | Ported |
| `tests/test_z3_avx.py:5032::TestPermute4x64Epi64::test_mm256_permute4x64_epi64_custom_pattern` | `rust/z3_avx/tests/permute4x64_epi64.rs::test_mm256_permute4x64_epi64_custom_pattern` | Ported |
| `tests/test_z3_avx.py:5060::TestPermute4x64Epi64::test_mm256_permute4x64_epi64_symbolic_imm` | `rust/z3_avx/tests/permute4x64_epi64.rs::test_mm256_permute4x64_epi64_symbolic_imm` | Ported |
