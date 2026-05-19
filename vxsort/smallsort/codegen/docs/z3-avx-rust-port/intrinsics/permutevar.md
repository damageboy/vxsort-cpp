# permutevar Report

Python source: `tests/test_z3_avx.py`
Rust source: `rust/z3_avx/src/permutevar.rs`
Rust tests:
- `rust/z3_avx/tests/permutevar_ps.rs`
- `rust/z3_avx/tests/permutevar_pd.rs`

Intrinsics:
- `_mm256_permutevar_ps`
- `_mm512_permutevar_ps`
- `_mm512_mask_permutevar_ps`
- `_mm256_permutevar_pd`
- `_mm512_permutevar_pd`
- `_mm512_mask_permutevar_pd`

Status: Ported.

Canonicalization:
- ps controls keep `[1:0]` in each 32-bit element.
- pd controls keep bit `[1]` in each 64-bit element; bit `[0]` and `[63:2]` are dead.
- Masked ps constrains k-mask bits above bit 15.
- Masked pd constrains k-mask bits above bit 7.

| Python test | Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:3800::TestPermutevarPs::test_mm256_permutevar_ps_identity_permute` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm256_permutevar_ps_identity_permute` | Implemented |
| `tests/test_z3_avx.py:3818::TestPermutevarPs::test_mm256_permutevar_ps_reverse_within_lanes` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm256_permutevar_ps_reverse_within_lanes` | Implemented |
| `tests/test_z3_avx.py:3852::TestPermutevarPs::test_mm256_permutevar_ps_broadcast_within_lanes` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm256_permutevar_ps_broadcast_within_lanes` | Implemented |
| `tests/test_z3_avx.py:3884::TestPermutevarPs::test_mm256_permutevar_ps_mixed_permute` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm256_permutevar_ps_mixed_permute` | Implemented |
| `tests/test_z3_avx.py:3916::TestPermutevarPs::test_mm512_permutevar_ps_identity_permute` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm512_permutevar_ps_identity_permute` | Implemented |
| `tests/test_z3_avx.py:3933::TestPermutevarPs::test_mm512_permutevar_ps_reverse_within_lanes` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm512_permutevar_ps_reverse_within_lanes` | Implemented |
| `tests/test_z3_avx.py:3975::TestPermutevarPs::test_mm512_permutevar_ps_broadcast_within_lanes` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm512_permutevar_ps_broadcast_within_lanes` | Implemented |
| `tests/test_z3_avx.py:4015::TestPermutevarPs::test_mm512_permutevar_ps_alternating_pattern` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm512_permutevar_ps_alternating_pattern` | Implemented |
| `tests/test_z3_avx.py:3534::TestMaskPermutevarPs::test_mm512_mask_permutevar_ps_mask_all_zeros` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm512_mask_permutevar_ps_mask_all_zeros` | Implemented |
| `tests/test_z3_avx.py:3553::TestMaskPermutevarPs::test_mm512_mask_permutevar_ps_identity_permute` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm512_mask_permutevar_ps_identity_permute` | Implemented |
| `tests/test_z3_avx.py:3573::TestMaskPermutevarPs::test_mm512_mask_permutevar_ps_reverse_within_lanes` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm512_mask_permutevar_ps_reverse_within_lanes` | Implemented |
| `tests/test_z3_avx.py:3617::TestMaskPermutevarPs::test_mm512_mask_permutevar_ps_broadcast_within_lanes` | `rust/z3_avx/tests/permutevar_ps.rs::test_mm512_mask_permutevar_ps_broadcast_within_lanes` | Implemented |
| `tests/test_z3_avx.py:4064::TestPermutevarPd::test_mm256_permutevar_pd_identity_permute` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm256_permutevar_pd_identity_permute` | Implemented |
| `tests/test_z3_avx.py:4085::TestPermutevarPd::test_mm256_permutevar_pd_swap_within_lanes` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm256_permutevar_pd_swap_within_lanes` | Implemented |
| `tests/test_z3_avx.py:4117::TestPermutevarPd::test_mm256_permutevar_pd_broadcast_first_within_lanes` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm256_permutevar_pd_broadcast_first_within_lanes` | Implemented |
| `tests/test_z3_avx.py:4149::TestPermutevarPd::test_mm256_permutevar_pd_broadcast_second_within_lanes` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm256_permutevar_pd_broadcast_second_within_lanes` | Implemented |
| `tests/test_z3_avx.py:4181::TestPermutevarPd::test_mm512_permutevar_pd_identity_permute` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm512_permutevar_pd_identity_permute` | Implemented |
| `tests/test_z3_avx.py:4206::TestPermutevarPd::test_mm512_permutevar_pd_swap_within_lanes` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm512_permutevar_pd_swap_within_lanes` | Implemented |
| `tests/test_z3_avx.py:4246::TestPermutevarPd::test_mm512_permutevar_pd_broadcast_first_within_lanes` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm512_permutevar_pd_broadcast_first_within_lanes` | Implemented |
| `tests/test_z3_avx.py:4286::TestPermutevarPd::test_mm512_permutevar_pd_broadcast_second_within_lanes` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm512_permutevar_pd_broadcast_second_within_lanes` | Implemented |
| `tests/test_z3_avx.py:3663::TestMaskPermutevarPd::test_mm512_mask_permutevar_pd_mask_all_zeros` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm512_mask_permutevar_pd_mask_all_zeros` | Implemented |
| `tests/test_z3_avx.py:3682::TestMaskPermutevarPd::test_mm512_mask_permutevar_pd_identity_permute` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm512_mask_permutevar_pd_identity_permute` | Implemented |
| `tests/test_z3_avx.py:3711::TestMaskPermutevarPd::test_mm512_mask_permutevar_pd_swap_within_lanes` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm512_mask_permutevar_pd_swap_within_lanes` | Implemented |
| `tests/test_z3_avx.py:3754::TestMaskPermutevarPd::test_mm512_mask_permutevar_pd_broadcast_within_lanes` | `rust/z3_avx/tests/permutevar_pd.rs::test_mm512_mask_permutevar_pd_broadcast_within_lanes` | Implemented |
