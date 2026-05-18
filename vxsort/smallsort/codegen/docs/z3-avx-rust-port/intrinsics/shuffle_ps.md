# shuffle_ps Report

Python source: `tests/test_z3_avx.py`
Planned Rust source: `z3_avx/src/shuffle_ps.rs`
Planned Rust tests: `z3_avx/tests/shuffle_ps.rs`

Intrinsics:
- `_mm256_shuffle_ps`
- `_mm512_shuffle_ps`
- `_mm512_mask_shuffle_ps`

Status: Ported.

Canonicalization:
- `imm8` uses all 8 bits.
- `_mm512_mask_shuffle_ps` must constrain k-mask bits above bit 15.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:1184::TestShufflePs::test_mm256_shuffle_ps_null_permute_works` | `z3_avx/tests/shuffle_ps.rs::test_mm256_shuffle_ps_null_permute_works` | Ported |
| `tests/test_z3_avx.py:1197::TestShufflePs::test_mm256_shuffle_ps_null_permute_found` | `z3_avx/tests/shuffle_ps.rs::test_mm256_shuffle_ps_null_permute_found` | Ported |
| `tests/test_z3_avx.py:1214::TestShufflePs::test_mm256_shuffle_ps_null_permute_2vec_works` | `z3_avx/tests/shuffle_ps.rs::test_mm256_shuffle_ps_null_permute_2vec_works` | Ported |
| `tests/test_z3_avx.py:1242::TestShufflePs::test_mm256_shuffle_ps_null_permute_2vec_found` | `z3_avx/tests/shuffle_ps.rs::test_mm256_shuffle_ps_null_permute_2vec_found` | Ported |
| `tests/test_z3_avx.py:1274::TestShufflePs::test_mm512_shuffle_ps_null_permute_works` | `z3_avx/tests/shuffle_ps.rs::test_mm512_shuffle_ps_null_permute_works` | Ported |
| `tests/test_z3_avx.py:1311::TestShufflePs::test_mm512_shuffle_ps_null_permute_found` | `z3_avx/tests/shuffle_ps.rs::test_mm512_shuffle_ps_null_permute_found` | Ported |
| `tests/test_z3_avx.py:1350::TestShufflePs::test_mm512_shuffle_ps_null_permute_2vec_works` | `z3_avx/tests/shuffle_ps.rs::test_mm512_shuffle_ps_null_permute_2vec_works` | Ported |
| `tests/test_z3_avx.py:1386::TestShufflePs::test_mm512_shuffle_ps_null_permute_2vec_found` | `z3_avx/tests/shuffle_ps.rs::test_mm512_shuffle_ps_null_permute_2vec_found` | Ported |
| `tests/test_z3_avx.py:1426::TestShufflePs::test_mm256_shuffle_ps_bitonic_stage_masks` | `z3_avx/tests/shuffle_ps.rs::test_mm256_shuffle_ps_bitonic_stage_masks` | Ported |
| `tests/test_z3_avx.py:1500::TestShufflePs::test_mm256_shuffle_ps_bitonic_stage_masks_literal` | `z3_avx/tests/shuffle_ps.rs::test_mm256_shuffle_ps_bitonic_stage_masks_literal` | Ported |
| `tests/test_z3_avx.py:3315::TestMaskShufflePs::test_mm512_mask_shuffle_ps_mask_all_zeros` | `z3_avx/tests/shuffle_ps.rs::test_mm512_mask_shuffle_ps_mask_all_zeros` | Ported |
| `tests/test_z3_avx.py:3332::TestMaskShufflePs::test_mm512_mask_shuffle_ps_mask_all_ones` | `z3_avx/tests/shuffle_ps.rs::test_mm512_mask_shuffle_ps_mask_all_ones` | Ported |
| `tests/test_z3_avx.py:3352::TestMaskShufflePs::test_mm512_mask_shuffle_ps_partial_mask` | `z3_avx/tests/shuffle_ps.rs::test_mm512_mask_shuffle_ps_partial_mask` | Ported |
