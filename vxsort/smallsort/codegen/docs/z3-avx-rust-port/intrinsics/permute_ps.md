# permute_ps Report

Python source: `tests/test_z3_avx.py`
Rust source: `z3_avx/src/permute.rs`
Rust tests: `z3_avx/tests/permute_ps.rs`

Intrinsics:
- `_mm256_permute_ps`
- `_mm512_permute_ps`
- `_mm512_mask_permute_ps`

Status: Ported in first wave.

Canonicalization:
- Symbolic `imm8` keeps all 8 bits for `permute_ps`.
- `_mm512_mask_permute_ps` constrains non-live k-mask bits above bit 15 inside the generic handler via the shared `canonical` helper.

| Python test | Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:164::TestPermutePs::test_mm256_permute_epi32_null_permute_works` | `z3_avx/tests/permute_ps.rs::mm256_permute_ps_literal_identity` | Ported |
| `tests/test_z3_avx.py:176::TestPermutePs::test_mm256_permute_epi32_null_permute_found` | `z3_avx/tests/permute_ps.rs::mm256_permute_ps_symbolic_identity_finds_identity_imm8` | Ported |
| `tests/test_z3_avx.py:192::TestPermutePs::test_mm512_permute_epi32_null_permute` | `z3_avx/tests/permute_ps.rs::mm512_permute_ps_literal_identity` | Ported |
| `tests/test_z3_avx.py:205::TestPermutePs::test_mm512_permute_epi32_null_permute_found` | `z3_avx/tests/permute_ps.rs::mm512_permute_ps_symbolic_identity_finds_identity_imm8` | Ported |
| `tests/test_z3_avx.py:3175::TestMaskPermutePs::test_mm512_mask_permute_ps_mask_all_zeros` | `z3_avx/tests/permute_ps.rs::mm512_mask_permute_ps_all_zero_mask_preserves_src` | Ported |
| `tests/test_z3_avx.py:3192::TestMaskPermutePs::test_mm512_mask_permute_ps_mask_all_ones` | `z3_avx/tests/permute_ps.rs::mm512_mask_permute_ps_all_one_mask_matches_unmasked` | Ported |
| `tests/test_z3_avx.py:3210::TestMaskPermutePs::test_mm512_mask_permute_ps_alternating_mask` | `z3_avx/tests/permute_ps.rs::mm512_mask_permute_ps_alternating_mask_selects_by_element` | Ported |
| Rust-only k-mask regression | `z3_avx/tests/permute_ps.rs::mm512_mask_permute_ps_canonicalizes_dead_mask_bits` | Ported |
