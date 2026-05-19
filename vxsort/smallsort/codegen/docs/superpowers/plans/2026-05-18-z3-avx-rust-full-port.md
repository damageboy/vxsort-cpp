# z3_avx Rust Full Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Rust `z3_avx` crate so it covers every intrinsic and every test currently in `tests/test_z3_avx.py` and `tests/test_z3_avx_canonicalization.py`.

**Architecture:** Keep shared Z3 bitvector utilities small and reusable, then port instruction families around their Python generic handlers. Canonicalization belongs where the generic handler knows the live control or mask width.

**Tech Stack:** Rust workspace, `z3` crate, Cargo integration tests, existing Python `uv` checks for regression safety.

---

## Files and reports

The detailed implementation plan is maintained in `docs/z3-avx-rust-port/port-plan.md`.

The per-family report files are maintained under `docs/z3-avx-rust-port/intrinsics/` and must be updated by the worker that ports each family:

- `control.md`
- `permute_ps.md`
- `permute_pd.md`
- `permutexvar.md`
- `permutex2var.md`
- `shuffle_ps.md`
- `shuffle_pd.md`
- `permute2x128.md`
- `shuffle_i32x4.md`
- `unpack.md`
- `permutevar.md`
- `blend.md`
- `permute4x64.md`
- `alignr.md`
- `minmax.md`
- `canonicalization.md`

## Tasks

### Task 0: Shared canonicalization and parity tracking

**Files:**
- Create: `rust/z3_avx/src/canonical.rs`
- Create: `rust/z3_avx/src/mask.rs`
- Modify: `rust/z3_avx/src/lib.rs`
- Modify: `rust/z3_avx/tests/control.rs`
- Modify: `rust/z3_avx/tests/permute_ps.rs`
- Modify: `docs/z3-avx-rust-port/intrinsics/control.md`
- Modify: `docs/z3-avx-rust-port/intrinsics/permute_ps.md`

- [ ] Add tests that prove scalar, per-lane, arbitrary-range, and k-mask canonicalization helpers constrain only dead bits.
- [ ] Implement the shared helpers.
- [ ] Port all remaining `_MM_SHUFFLE` and `_MM_SHUFFLE2` tests.
- [ ] Add widened k-mask canonicalization regression for `_mm512_mask_permute_ps`.
- [ ] Run `cargo test -p z3_avx --test control`.
- [ ] Run `cargo test -p z3_avx --test permute_ps`.
- [ ] Commit with `feat: add z3_avx canonicalization helpers`.

### Task 1: Static immediate permutes and shuffles

**Files:**
- Create: `rust/z3_avx/src/permute_pd.rs`
- Create: `rust/z3_avx/src/shuffle_ps.rs`
- Create: `rust/z3_avx/src/shuffle_pd.rs`
- Create: `rust/z3_avx/src/permute4x64.rs`
- Create/modify tests and reports listed in `docs/z3-avx-rust-port/port-plan.md`.

- [ ] Port tests for `permute_pd`, including masked and canonicalization cases.
- [ ] Port tests for `shuffle_ps`, including masked cases.
- [ ] Port tests for `shuffle_pd`, including masked and canonicalization cases.
- [ ] Port tests for `permute4x64_epi64`.
- [ ] Implement the family-local generic helpers and wrappers.
- [ ] Run the family-specific Cargo tests and then `cargo test -p z3_avx`.
- [ ] Commit with `feat: port static z3_avx permutes`.

### Task 2: Lane-level immediate operations

**Files:**
- Create: `rust/z3_avx/src/permute2x128.rs`
- Create: `rust/z3_avx/src/shuffle_i32x4.rs`
- Create/modify tests and reports listed in `docs/z3-avx-rust-port/port-plan.md`.

- [ ] Port all `permute2x128` tests, including zero-lane and canonicalization cases.
- [ ] Port all `shuffle_i32x4` tests, including masked cases.
- [ ] Implement the generic lane selector helpers.
- [ ] Add widened k-mask canonicalization regression for `_mm512_mask_shuffle_i32x4`.
- [ ] Run the family-specific Cargo tests and then `cargo test -p z3_avx`.
- [ ] Commit with `feat: port z3_avx lane permutes`.

### Task 3: Cross-lane variable permutes

**Files:**
- Create: `rust/z3_avx/src/permutexvar.rs`
- Create: `rust/z3_avx/src/permutex2var.rs`
- Create/modify tests and reports listed in `docs/z3-avx-rust-port/port-plan.md`.

- [ ] Port all `permutexvar_epi32` and `permutexvar_epi64` tests.
- [ ] Port all masked `permutexvar` tests.
- [ ] Port all `permutex2var_epi32` and `permutex2var_epi64` tests.
- [ ] Port all masked `permutex2var` tests.
- [ ] Add dead index-bit canonicalization tests for every width.
- [ ] Add widened k-mask canonicalization regressions for epi32 and epi64 masked variants.
- [ ] Run the family-specific Cargo tests and then `cargo test -p z3_avx`.
- [ ] Commit with `feat: port z3_avx variable cross-lane permutes`.

### Task 4: Within-lane variable permutes

**Files:**
- Create: `rust/z3_avx/src/permutevar.rs`
- Create/modify tests and reports listed in `docs/z3-avx-rust-port/port-plan.md`.

- [ ] Port all `permutevar_ps` and `permutevar_pd` tests.
- [ ] Port all masked `permutevar` tests.
- [ ] Add canonicalization tests for ps control bits.
- [ ] Add canonicalization tests for pd bit 0 and high bits.
- [ ] Add widened k-mask canonicalization regressions for ps and pd masked variants.
- [ ] Run the family-specific Cargo tests and then `cargo test -p z3_avx`.
- [ ] Commit with `feat: port z3_avx within-lane variable permutes`.

### Task 5: Unpack and blend

**Files:**
- Create: `rust/z3_avx/src/unpack.rs`
- Create: `rust/z3_avx/src/blend.rs`
- Create/modify tests and reports listed in `docs/z3-avx-rust-port/port-plan.md`.

- [ ] Port all unpack epi32 tests and masked tests.
- [ ] Port all unpack epi64 tests and masked tests.
- [ ] Port all blend and blendv tests.
- [ ] Add blend and blendv canonicalization tests.
- [ ] Add widened k-mask canonicalization regressions for masked unpack variants.
- [ ] Run the family-specific Cargo tests and then `cargo test -p z3_avx`.
- [ ] Commit with `feat: port z3_avx unpack and blend`.

### Task 6: Align and signed min/max

**Files:**
- Create: `rust/z3_avx/src/alignr.rs`
- Create: `rust/z3_avx/src/minmax.rs`
- Create/modify tests and reports listed in `docs/z3-avx-rust-port/port-plan.md`.

- [x] Port all alignr epi8, epi32, and epi64 tests.
- [x] Port all masked alignr tests.
- [x] Add alignr canonicalization tests.
- [x] Add widened k-mask canonicalization regressions for masked alignr variants.
- [x] Port all min/max epi32 and epi64 tests.
- [x] Run the family-specific Cargo tests and then `cargo test -p z3_avx`.
- [x] Commit with `feat: port z3_avx alignr and minmax`.

### Task 7: Final parity audit

**Files:**
- Modify all reports under `docs/z3-avx-rust-port/intrinsics/`.

- [x] Re-run `rg -n "def test_" tests/test_z3_avx.py tests/test_z3_avx_canonicalization.py`.
- [x] Confirm every Python test appears exactly once in the reports.
- [x] Confirm every report row has status `Ported` or a documented deferral for non-`z3_avx` integration scope.
- [x] Run `cargo fmt --all`.
- [x] Run `cargo test -p z3_avx`.
- [x] Run `uv run pytest`.
- [x] Run `uv run ruff check .`.
- [x] Run `uv run vulture` and inspect findings.
- [ ] Commit with `docs: audit z3_avx rust port parity`.
