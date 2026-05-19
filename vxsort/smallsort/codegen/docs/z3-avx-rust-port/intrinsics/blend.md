# blend Report

Python source: `tests/test_z3_avx.py`
Planned Rust source: `rust/z3_avx/src/blend.rs`
Planned Rust tests:
- `rust/z3_avx/tests/blend_pd.rs`
- `rust/z3_avx/tests/blend_ps.rs`
- `rust/z3_avx/tests/blendv_pd.rs`
- `rust/z3_avx/tests/blendv_ps.rs`

Intrinsics:
- `_mm256_blend_pd`
- `_mm256_blend_ps`
- `_mm256_blendv_pd`
- `_mm256_blendv_ps`

Status: Ported.

Canonicalization:
- `blend_pd` keeps imm bits `[3:0]`; constrain `[7:4]`.
- `blend_ps` uses all 8 imm bits.
- `blendv_pd` and `blendv_ps` use only each element sign bit; constrain all other bits in the mask vector.

| Python test | Planned Rust test | Status |
|---|---|---|
| `tests/test_z3_avx.py:4330::TestBlendPd::test_mm256_blend_pd_all_from_a` | `rust/z3_avx/tests/blend_pd.rs::test_mm256_blend_pd_all_from_a` | Ported |
| `tests/test_z3_avx.py:4347::TestBlendPd::test_mm256_blend_pd_all_from_b` | `rust/z3_avx/tests/blend_pd.rs::test_mm256_blend_pd_all_from_b` | Ported |
| `tests/test_z3_avx.py:4364::TestBlendPd::test_mm256_blend_pd_alternating` | `rust/z3_avx/tests/blend_pd.rs::test_mm256_blend_pd_alternating` | Ported |
| `tests/test_z3_avx.py:4390::TestBlendPd::test_mm256_blend_pd_first_two_from_b` | `rust/z3_avx/tests/blend_pd.rs::test_mm256_blend_pd_first_two_from_b` | Ported |
| `tests/test_z3_avx.py:4415::TestBlendPd::test_mm256_blend_pd_symbolic_mask` | `rust/z3_avx/tests/blend_pd.rs::test_mm256_blend_pd_symbolic_mask` | Ported |
| `tests/test_z3_avx.py:4445::TestBlendPd::test_mm256_blend_pd_n3_stage_bottom_imm0` | `rust/z3_avx/tests/blend_pd.rs::test_mm256_blend_pd_n3_stage_bottom_imm0` | Ported |
| `tests/test_z3_avx.py:4480::TestBlendPs::test_mm256_blend_ps_all_from_a` | `rust/z3_avx/tests/blend_ps.rs::test_mm256_blend_ps_all_from_a` | Ported |
| `tests/test_z3_avx.py:4497::TestBlendPs::test_mm256_blend_ps_all_from_b` | `rust/z3_avx/tests/blend_ps.rs::test_mm256_blend_ps_all_from_b` | Ported |
| `tests/test_z3_avx.py:4514::TestBlendPs::test_mm256_blend_ps_alternating` | `rust/z3_avx/tests/blend_ps.rs::test_mm256_blend_ps_alternating` | Ported |
| `tests/test_z3_avx.py:4544::TestBlendPs::test_mm256_blend_ps_first_four_from_b` | `rust/z3_avx/tests/blend_ps.rs::test_mm256_blend_ps_first_four_from_b` | Ported |
| `tests/test_z3_avx.py:4573::TestBlendPs::test_mm256_blend_ps_symbolic_mask` | `rust/z3_avx/tests/blend_ps.rs::test_mm256_blend_ps_symbolic_mask` | Ported |
| `tests/test_z3_avx.py:4611::TestBlendvPd::test_mm256_blendv_pd_all_from_a` | `rust/z3_avx/tests/blendv_pd.rs::test_mm256_blendv_pd_all_from_a` | Ported |
| `tests/test_z3_avx.py:4632::TestBlendvPd::test_mm256_blendv_pd_all_from_b` | `rust/z3_avx/tests/blendv_pd.rs::test_mm256_blendv_pd_all_from_b` | Ported |
| `tests/test_z3_avx.py:4653::TestBlendvPd::test_mm256_blendv_pd_alternating` | `rust/z3_avx/tests/blendv_pd.rs::test_mm256_blendv_pd_alternating` | Ported |
| `tests/test_z3_avx.py:4684::TestBlendvPd::test_mm256_blendv_pd_symbolic_mask` | `rust/z3_avx/tests/blendv_pd.rs::test_mm256_blendv_pd_symbolic_mask` | Ported |
| `tests/test_z3_avx.py:4725::TestBlendvPs::test_mm256_blendv_ps_all_from_a` | `rust/z3_avx/tests/blendv_ps.rs::test_mm256_blendv_ps_all_from_a` | Ported |
| `tests/test_z3_avx.py:4746::TestBlendvPs::test_mm256_blendv_ps_all_from_b` | `rust/z3_avx/tests/blendv_ps.rs::test_mm256_blendv_ps_all_from_b` | Ported |
| `tests/test_z3_avx.py:4767::TestBlendvPs::test_mm256_blendv_ps_alternating` | `rust/z3_avx/tests/blendv_ps.rs::test_mm256_blendv_ps_alternating` | Ported |
| `tests/test_z3_avx.py:4802::TestBlendvPs::test_mm256_blendv_ps_first_four_from_b` | `rust/z3_avx/tests/blendv_ps.rs::test_mm256_blendv_ps_first_four_from_b` | Ported |
| `tests/test_z3_avx.py:4837::TestBlendvPs::test_mm256_blendv_ps_symbolic_mask` | `rust/z3_avx/tests/blendv_ps.rs::test_mm256_blendv_ps_symbolic_mask` | Ported |
