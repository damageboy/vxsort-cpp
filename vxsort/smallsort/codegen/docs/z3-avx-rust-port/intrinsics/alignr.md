# alignr Report

Python source: `tests/test_z3_avx.py`
Planned Rust source: `z3_avx/src/alignr.rs`
Planned Rust tests:
- `z3_avx/tests/alignr_epi8.rs`
- `z3_avx/tests/alignr_epi32.rs`
- `z3_avx/tests/alignr_epi64.rs`

Intrinsics:
- `_mm256_alignr_epi8`
- `_mm256_alignr_epi32`
- `_mm512_alignr_epi32`
- `_mm512_mask_alignr_epi32`
- `_mm256_alignr_epi64`
- `_mm512_alignr_epi64`
- `_mm512_mask_alignr_epi64`

Status: Ported.

Canonicalization:
- epi8 has no current Python canonicalization requirement.
- epi32 YMM keeps shift bits `[2:0]`; epi32 ZMM keeps `[3:0]`.
- epi64 YMM keeps shift bits `[1:0]`; epi64 ZMM keeps `[2:0]`.
- Masked epi32 constrains k-mask bits above bit 15.
- Masked epi64 constrains k-mask bits above bit 7.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:5095::TestAlignrEpi8::test_mm256_alignr_epi8_shift_zero` | `z3_avx/tests/alignr_epi8.rs::test_mm256_alignr_epi8_shift_zero` | Ported |
| `tests/test_z3_avx.py:5107::TestAlignrEpi8::test_mm256_alignr_epi8_shift_one` | `z3_avx/tests/alignr_epi8.rs::test_mm256_alignr_epi8_shift_one` | Ported |
| `tests/test_z3_avx.py:5127::TestAlignrEpi8::test_mm256_alignr_epi8_shift_thirty_one` | `z3_avx/tests/alignr_epi8.rs::test_mm256_alignr_epi8_shift_thirty_one` | Ported |
| `tests/test_z3_avx.py:5147::TestAlignrEpi8::test_mm256_alignr_epi8_find_shift` | `z3_avx/tests/alignr_epi8.rs::test_mm256_alignr_epi8_find_shift` | Ported |
| `tests/test_z3_avx.py:5173::TestAlignrEpi32::test_mm256_alignr_epi32_shift_zero` | `z3_avx/tests/alignr_epi32.rs::test_mm256_alignr_epi32_shift_zero` | Ported |
| `tests/test_z3_avx.py:5186::TestAlignrEpi32::test_mm256_alignr_epi32_shift_one` | `z3_avx/tests/alignr_epi32.rs::test_mm256_alignr_epi32_shift_one` | Ported |
| `tests/test_z3_avx.py:5204::TestAlignrEpi32::test_mm256_alignr_epi32_shift_seven` | `z3_avx/tests/alignr_epi32.rs::test_mm256_alignr_epi32_shift_seven` | Ported |
| `tests/test_z3_avx.py:5222::TestAlignrEpi32::test_mm256_alignr_epi32_shift_four` | `z3_avx/tests/alignr_epi32.rs::test_mm256_alignr_epi32_shift_four` | Ported |
| `tests/test_z3_avx.py:5243::TestAlignrEpi32::test_mm512_alignr_epi32_shift_zero` | `z3_avx/tests/alignr_epi32.rs::test_mm512_alignr_epi32_shift_zero` | Ported |
| `tests/test_z3_avx.py:5256::TestAlignrEpi32::test_mm512_alignr_epi32_shift_fifteen` | `z3_avx/tests/alignr_epi32.rs::test_mm512_alignr_epi32_shift_fifteen` | Ported |
| `tests/test_z3_avx.py:5272::TestAlignrEpi32::test_mm512_alignr_epi32_shift_four` | `z3_avx/tests/alignr_epi32.rs::test_mm512_alignr_epi32_shift_four` | Ported |
| `tests/test_z3_avx.py:5287::TestAlignrEpi32::test_mm256_alignr_epi32_find_shift` | `z3_avx/tests/alignr_epi32.rs::test_mm256_alignr_epi32_find_shift` | Ported |
| `tests/test_z3_avx.py:5464::TestMaskAlignrEpi32::test_mm512_mask_alignr_epi32_mask_all_zeros` | `z3_avx/tests/alignr_epi32.rs::test_mm512_mask_alignr_epi32_mask_all_zeros` | Ported |
| `tests/test_z3_avx.py:5479::TestMaskAlignrEpi32::test_mm512_mask_alignr_epi32_mask_all_ones` | `z3_avx/tests/alignr_epi32.rs::test_mm512_mask_alignr_epi32_mask_all_ones` | Ported |
| `tests/test_z3_avx.py:5499::TestMaskAlignrEpi32::test_mm512_mask_alignr_epi32_alternating_mask` | `z3_avx/tests/alignr_epi32.rs::test_mm512_mask_alignr_epi32_alternating_mask` | Ported |
| `tests/test_z3_avx.py:5521::TestMaskAlignrEpi32::test_mm512_mask_alignr_epi32_partial_mask` | `z3_avx/tests/alignr_epi32.rs::test_mm512_mask_alignr_epi32_partial_mask` | Ported |
| `tests/test_z3_avx.py:5543::TestMaskAlignrEpi32::test_mm512_mask_alignr_epi32_find_mask` | `z3_avx/tests/alignr_epi32.rs::test_mm512_mask_alignr_epi32_find_mask` | Ported |
| `tests/test_z3_avx.py:5312::TestAlignrEpi64::test_mm256_alignr_epi64_shift_zero` | `z3_avx/tests/alignr_epi64.rs::test_mm256_alignr_epi64_shift_zero` | Ported |
| `tests/test_z3_avx.py:5325::TestAlignrEpi64::test_mm256_alignr_epi64_shift_one` | `z3_avx/tests/alignr_epi64.rs::test_mm256_alignr_epi64_shift_one` | Ported |
| `tests/test_z3_avx.py:5343::TestAlignrEpi64::test_mm256_alignr_epi64_shift_two` | `z3_avx/tests/alignr_epi64.rs::test_mm256_alignr_epi64_shift_two` | Ported |
| `tests/test_z3_avx.py:5361::TestAlignrEpi64::test_mm256_alignr_epi64_shift_three` | `z3_avx/tests/alignr_epi64.rs::test_mm256_alignr_epi64_shift_three` | Ported |
| `tests/test_z3_avx.py:5377::TestAlignrEpi64::test_mm512_alignr_epi64_shift_zero` | `z3_avx/tests/alignr_epi64.rs::test_mm512_alignr_epi64_shift_zero` | Ported |
| `tests/test_z3_avx.py:5390::TestAlignrEpi64::test_mm512_alignr_epi64_shift_seven` | `z3_avx/tests/alignr_epi64.rs::test_mm512_alignr_epi64_shift_seven` | Ported |
| `tests/test_z3_avx.py:5406::TestAlignrEpi64::test_mm512_alignr_epi64_shift_four` | `z3_avx/tests/alignr_epi64.rs::test_mm512_alignr_epi64_shift_four` | Ported |
| `tests/test_z3_avx.py:5424::TestAlignrEpi64::test_mm512_alignr_epi64_shift_three` | `z3_avx/tests/alignr_epi64.rs::test_mm512_alignr_epi64_shift_three` | Ported |
| `tests/test_z3_avx.py:5442::TestAlignrEpi64::test_mm256_alignr_epi64_find_shift` | `z3_avx/tests/alignr_epi64.rs::test_mm256_alignr_epi64_find_shift` | Ported |
| `tests/test_z3_avx.py:5571::TestMaskAlignrEpi64::test_mm512_mask_alignr_epi64_mask_all_zeros` | `z3_avx/tests/alignr_epi64.rs::test_mm512_mask_alignr_epi64_mask_all_zeros` | Ported |
| `tests/test_z3_avx.py:5586::TestMaskAlignrEpi64::test_mm512_mask_alignr_epi64_mask_all_ones` | `z3_avx/tests/alignr_epi64.rs::test_mm512_mask_alignr_epi64_mask_all_ones` | Ported |
| `tests/test_z3_avx.py:5606::TestMaskAlignrEpi64::test_mm512_mask_alignr_epi64_alternating_mask` | `z3_avx/tests/alignr_epi64.rs::test_mm512_mask_alignr_epi64_alternating_mask` | Ported |
| `tests/test_z3_avx.py:5628::TestMaskAlignrEpi64::test_mm512_mask_alignr_epi64_partial_mask` | `z3_avx/tests/alignr_epi64.rs::test_mm512_mask_alignr_epi64_partial_mask` | Ported |
| `tests/test_z3_avx.py:5650::TestMaskAlignrEpi64::test_mm512_mask_alignr_epi64_single_bit_mask` | `z3_avx/tests/alignr_epi64.rs::test_mm512_mask_alignr_epi64_single_bit_mask` | Ported |
| `tests/test_z3_avx.py:5672::TestMaskAlignrEpi64::test_mm512_mask_alignr_epi64_find_shift_and_mask` | `z3_avx/tests/alignr_epi64.rs::test_mm512_mask_alignr_epi64_find_shift_and_mask` | Ported |
| Rust-only imm canonicalization regression | `z3_avx/tests/alignr_epi32.rs::test_mm256_alignr_epi32_canonicalizes_imm_high_bits` | Ported |
| Rust-only imm canonicalization regression | `z3_avx/tests/alignr_epi32.rs::test_mm512_alignr_epi32_canonicalizes_imm_high_bits` | Ported |
| Rust-only imm canonicalization regression | `z3_avx/tests/alignr_epi64.rs::test_mm512_alignr_epi64_canonicalizes_imm_high_bits` | Ported |
| Rust-only k-mask regression | `z3_avx/tests/alignr_epi32.rs::test_mm512_mask_alignr_epi32_canonicalizes_dead_mask_bits` | Ported |
| Rust-only k-mask regression | `z3_avx/tests/alignr_epi64.rs::test_mm512_mask_alignr_epi64_canonicalizes_dead_mask_bits` | Ported |
