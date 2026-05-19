# Control Helpers Report

Python source: `tests/test_z3_avx.py`
Rust source: `rust/z3_avx/src/control.rs`
Rust tests: `rust/z3_avx/tests/control.rs`

Helpers:
- `_MM_SHUFFLE`
- `decode_shuffle_mask`
- `mm_shuffle_str`
- `_MM_SHUFFLE2`
- `decode_shuffle2_mask`
- `mm_shuffle2_str`

Status: Ported.

Canonicalization: none.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:5702::TestShuffleMaskDecode::test_decode_shuffle_mask_round_trip_all_values` | `rust/z3_avx/tests/control.rs::test_decode_shuffle_mask_round_trip_all_values` | Ported |
| `tests/test_z3_avx.py:5716::TestShuffleMaskDecode::test_mm_shuffle_str_identity` | `rust/z3_avx/tests/control.rs::test_mm_shuffle_str_identity` | Ported |
| `tests/test_z3_avx.py:5721::TestShuffleMaskDecode::test_mm_shuffle_str_0x88` | `rust/z3_avx/tests/control.rs::test_mm_shuffle_str_0x88` | Ported |
| `tests/test_z3_avx.py:5726::TestShuffleMaskDecode::test_mm_shuffle_str_0xdd` | `rust/z3_avx/tests/control.rs::test_mm_shuffle_str_0xdd` | Ported |
| `tests/test_z3_avx.py:5731::TestShuffleMaskDecode::test_mm_shuffle_str_all_zeros` | `rust/z3_avx/tests/control.rs::test_mm_shuffle_str_all_zeros` | Ported |
| `tests/test_z3_avx.py:5736::TestShuffleMaskDecode::test_mm_shuffle_str_all_ones` | `rust/z3_avx/tests/control.rs::test_mm_shuffle_str_all_ones` | Ported |
| `tests/test_z3_avx.py:5741::TestShuffleMaskDecode::test_mm_shuffle_str_reverse` | `rust/z3_avx/tests/control.rs::test_mm_shuffle_str_reverse` | Ported |
| `tests/test_z3_avx.py:5746::TestShuffleMaskDecode::test_mm_shuffle_str_format` | `rust/z3_avx/tests/control.rs::test_mm_shuffle_str_format` | Ported |
| `tests/test_z3_avx.py:5875::TestShuffle2MaskDecode::test_decode_shuffle2_mask_all_zeros` | `rust/z3_avx/tests/control.rs::test_decode_shuffle2_mask_all_zeros` | Ported |
| `tests/test_z3_avx.py:5880::TestShuffle2MaskDecode::test_decode_shuffle2_mask_identity` | `rust/z3_avx/tests/control.rs::test_decode_shuffle2_mask_identity` | Ported |
| `tests/test_z3_avx.py:5885::TestShuffle2MaskDecode::test_decode_shuffle2_mask_swap` | `rust/z3_avx/tests/control.rs::test_decode_shuffle2_mask_swap` | Ported |
| `tests/test_z3_avx.py:5890::TestShuffle2MaskDecode::test_decode_shuffle2_mask_all_ones` | `rust/z3_avx/tests/control.rs::test_decode_shuffle2_mask_all_ones` | Ported |
| `tests/test_z3_avx.py:5895::TestShuffle2MaskDecode::test_decode_shuffle2_mask_mixed` | `rust/z3_avx/tests/control.rs::test_decode_shuffle2_mask_mixed` | Ported |
| `tests/test_z3_avx.py:5903::TestShuffle2MaskDecode::test_mm_shuffle2_str_all_zeros` | `rust/z3_avx/tests/control.rs::test_mm_shuffle2_str_all_zeros` | Ported |
| `tests/test_z3_avx.py:5908::TestShuffle2MaskDecode::test_mm_shuffle2_str_identity` | `rust/z3_avx/tests/control.rs::test_mm_shuffle2_str_identity` | Ported |
| `tests/test_z3_avx.py:5913::TestShuffle2MaskDecode::test_mm_shuffle2_str_swap` | `rust/z3_avx/tests/control.rs::test_mm_shuffle2_str_swap` | Ported |
| `tests/test_z3_avx.py:5918::TestShuffle2MaskDecode::test_mm_shuffle2_str_all_ones` | `rust/z3_avx/tests/control.rs::test_mm_shuffle2_str_all_ones` | Ported |
| `tests/test_z3_avx.py:5923::TestShuffle2MaskDecode::test_mm_shuffle2_str_format` | `rust/z3_avx/tests/control.rs::test_mm_shuffle2_str_format` | Ported |
