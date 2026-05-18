# Canonicalization Test Report

Python source: `tests/test_z3_avx_canonicalization.py`
Planned Rust tests: colocate with each family test file, unless a shared `z3_avx/tests/canonicalization.rs` becomes clearer during implementation.

Status: Ported for family canonicalization tests; deferred only for the Rust synthesizer integration test because no Rust synthesizer exists yet.

Global requirement:
- Any dead symbolic control bits must be constrained through the solver at construction time.
- Any dead high k-mask bits must be constrained for AVX512 masked intrinsics.
- The constraint should be applied inside the generic handler when that handler already knows the live element count or control layout.

| Python test | Planned Rust test | Preferred Rust path | Status |
|---|---|---|---|
| `tests/test_z3_avx_canonicalization.py:8::test_permutexvar_256_epi32_canonicalization` | `test_mm256_permutexvar_epi32_canonicalizes_dead_index_bits` | `z3_avx/tests/permutexvar_epi32.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:20::test_permutexvar_256_epi64_canonicalization` | `test_mm256_permutexvar_epi64_canonicalizes_dead_index_bits` | `z3_avx/tests/permutexvar_epi64.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:31::test_permutex2var_512_epi32_canonicalization` | `test_mm512_permutex2var_epi32_canonicalizes_dead_index_bits` | `z3_avx/tests/permutex2var_epi32.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:43::test_permutevar_256_ps_canonicalization` | `test_mm256_permutevar_ps_canonicalizes_control_high_bits` | `z3_avx/tests/permutevar_ps.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:54::test_permutevar_256_pd_canonicalization` | `test_mm256_permutevar_pd_canonicalizes_control_dead_bits` | `z3_avx/tests/permutevar_pd.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:65::test_permutevar_256_pd_bit0_canonicalization` | `test_mm256_permutevar_pd_canonicalizes_control_dead_bits` | `z3_avx/tests/permutevar_pd.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:77::test_blendv_256_pd_canonicalization` | `test_blendv_256_pd_canonicalization` | `z3_avx/tests/blendv_pd.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:89::test_blend_256_pd_canonicalization` | `test_blend_256_pd_canonicalization` | `z3_avx/tests/blend_pd.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:101::test_shuffle_pd_256_canonicalization` | `test_shuffle_pd_256_canonicalization` | `z3_avx/tests/shuffle_pd.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:114::test_permute_pd_256_canonicalization` | `test_permute_pd_256_canonicalization` | `z3_avx/tests/permute_pd.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:125::test_permute2x128_canonicalization` | `test_permute2x128_canonicalization` | `z3_avx/tests/permute2x128.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:138::test_alignr_256_epi64_canonicalization` | `test_mm256_alignr_epi64_canonicalizes_imm_high_bits` | `z3_avx/tests/alignr_epi64.rs` | Ported |
| `tests/test_z3_avx_canonicalization.py:151::test_synthesizer_passes_solver_to_intrinsics` | `test_synthesizer_passes_solver_to_intrinsics` | Rust synthesizer integration test once synthesizer is ported | Deferred until Rust synthesizer exists |

Additional Rust-only k-mask tests to add:

| Requirement | Planned Rust test | Preferred Rust path | Status |
|---|---|---|---|
| `_mm512_mask_permute_ps` constrains high mask bits | `mm512_mask_permute_ps_canonicalizes_dead_mask_bits` | `z3_avx/tests/permute_ps.rs` | Ported |
| `_mm512_mask_permute_pd` constrains high mask bits | `test_mm512_mask_permute_pd_canonicalizes_dead_mask_bits` | `z3_avx/tests/permute_pd.rs` | Ported |
| `_mm512_mask_shuffle_ps` constrains high mask bits | `test_mm512_mask_shuffle_ps_canonicalizes_dead_mask_bits` | `z3_avx/tests/shuffle_ps.rs` | Ported |
| `_mm512_mask_shuffle_pd` constrains high mask bits | `test_mm512_mask_shuffle_pd_canonicalizes_dead_mask_bits` | `z3_avx/tests/shuffle_pd.rs` | Ported |
| `_mm512_mask_shuffle_i32x4` constrains high mask bits | `test_mm512_mask_shuffle_i32x4_canonicalizes_dead_mask_bits` | `z3_avx/tests/shuffle_i32x4.rs` | Ported |
| epi32 masked permutexvar constrains high mask bits | `test_mm512_mask_permutexvar_epi32_canonicalizes_dead_mask_bits` | `z3_avx/tests/permutexvar_epi32.rs` | Ported |
| epi64 masked permutexvar constrains high mask bits | `test_mm512_mask_permutexvar_epi64_canonicalizes_dead_mask_bits` | `z3_avx/tests/permutexvar_epi64.rs` | Ported |
| epi32 masked permutex2var constrains high mask bits | `test_mm512_mask_permutex2var_epi32_canonicalizes_dead_index_and_wide_mask_bits` | `z3_avx/tests/permutex2var_epi32.rs` | Ported |
| epi64 masked permutex2var constrains high mask bits | `test_mm512_mask_permutex2var_epi64_canonicalizes_dead_index_and_wide_mask_bits` | `z3_avx/tests/permutex2var_epi64.rs` | Ported |
| remaining pd masked permutevar constrains high mask bits | `test_mm512_mask_permutevar_pd_canonicalizes_dead_mask_bits` | `z3_avx/tests/permutevar_pd.rs` | Ported |
| remaining ps masked permutevar constrains high mask bits | `test_mm512_mask_permutevar_ps_canonicalizes_dead_mask_bits` | `z3_avx/tests/permutevar_ps.rs` | Ported |
| masked unpack and alignr constrain high mask bits | family-local `*_canonicalizes_dead_mask_bits` tests | `z3_avx/tests/unpack_*.rs`, `z3_avx/tests/alignr_*.rs` | Ported |

Shared Rust helper tests added in Task 0:

| Requirement | Rust test | Preferred Rust path | Status |
|---|---|---|---|
| scalar low-bit immediate helper constrains only high dead bits | `canonical_scalar_high_bits_constraints_keep_only_low_live_bits` | `z3_avx/tests/control.rs` | Ported |
| per-lane low-bit helper constrains only high dead bits in each lane | `canonical_lane_high_bits_constraints_keep_only_low_live_bits_per_lane` | `z3_avx/tests/control.rs` | Ported |
| arbitrary live range helper constrains only bits outside requested ranges | `canonical_range_constraints_keep_only_requested_live_ranges` | `z3_avx/tests/control.rs` | Ported |
| k-mask helper constrains only high non-live mask bits | `canonical_kmask_constraints_keep_only_live_low_mask_bits` | `z3_avx/tests/control.rs` | Ported |
