# z3_avx Rust First Wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Rust workspace with a `z3_avx` crate and port the shared register utilities plus `_mm256_permute_ps` and `_mm512_permute_ps`.

**Architecture:** The crate mirrors the Python utility structure in focused Rust modules. It uses the Rust `z3` crate directly for this first milestone and models literal-or-symbolic immediates with an `Imm8` enum. Tests migrate only the first-wave behavior.

**Tech Stack:** Rust, Cargo workspace, `z3` crate, native Z3 solver.

---

### Task 1: Scaffold Workspace and First Failing Tests

**Files:**
- Create: `Cargo.toml`
- Create: `z3_avx/Cargo.toml`
- Create: `z3_avx/src/lib.rs`
- Create: `z3_avx/src/registers.rs`
- Create: `z3_avx/src/control.rs`
- Create: `z3_avx/src/lanes.rs`
- Create: `z3_avx/src/permute.rs`
- Create: `z3_avx/tests/registers.rs`

- [ ] **Step 1: Create Cargo workspace skeleton**

Create a root `Cargo.toml` workspace containing the `z3_avx` member, and create the crate files with empty module declarations.

- [ ] **Step 2: Write failing register width tests**

Add tests that expect `ymm_reg` to produce a 256-bit bit-vector and `zmm_reg` to produce a 512-bit bit-vector.

- [ ] **Step 3: Run tests to verify failure**

Run: `cargo test -p z3_avx registers_have_expected_widths -- --nocapture`

Expected: FAIL because `ymm_reg` and `zmm_reg` are not implemented.

- [ ] **Step 4: Implement minimal register constructors**

Implement `ymm_reg` and `zmm_reg` with `BV::new_const`.

- [ ] **Step 5: Run tests to verify pass**

Run: `cargo test -p z3_avx registers_have_expected_widths -- --nocapture`

Expected: PASS.

### Task 2: Port Register Construction Utilities

**Files:**
- Modify: `z3_avx/src/registers.rs`
- Modify: `z3_avx/src/lanes.rs`
- Modify: `z3_avx/tests/registers.rs`

- [ ] **Step 1: Write failing concrete lane packing test**

Add a test for `ymm_reg_with_32b_values` that constrains lane values and proves the packed register equals an expected constant.

- [ ] **Step 2: Run test to verify failure**

Run: `cargo test -p z3_avx ymm_reg_with_32b_values_packs_lanes -- --nocapture`

Expected: FAIL because the helper is not implemented.

- [ ] **Step 3: Implement concrete lane packing helpers**

Implement generic `reg_with_values`, `ymm_reg_with_32b_values`, `zmm_reg_with_32b_values`, `ymm_reg_with_64b_values`, and `zmm_reg_with_64b_values`.

- [ ] **Step 4: Run test to verify pass**

Run: `cargo test -p z3_avx ymm_reg_with_32b_values_packs_lanes -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Write failing construct-from-elements test**

Add a test that builds a YMM register from selected elements and proves the result matches the expected reordered constant.

- [ ] **Step 6: Implement extraction and construct-from-elements helpers**

Implement lane extraction, reverse-order concatenation, `construct_reg_from_elements`, and YMM/ZMM wrappers.

- [ ] **Step 7: Run register tests**

Run: `cargo test -p z3_avx --test registers -- --nocapture`

Expected: PASS.

### Task 3: Port Control Utilities

**Files:**
- Modify: `z3_avx/src/control.rs`
- Create: `z3_avx/tests/control.rs`

- [ ] **Step 1: Write failing `_MM_SHUFFLE` round-trip tests**

Add tests for `mm_shuffle`, `decode_shuffle_mask`, `mm_shuffle_str`, `mm_shuffle2`, `decode_shuffle2_mask`, and `mm_shuffle2_str`.

- [ ] **Step 2: Run tests to verify failure**

Run: `cargo test -p z3_avx --test control -- --nocapture`

Expected: FAIL because helpers are not implemented.

- [ ] **Step 3: Implement control helpers and `Imm8`**

Implement the immediate helpers and an `Imm8` enum with conversion to an 8-bit `BV`.

- [ ] **Step 4: Run control tests**

Run: `cargo test -p z3_avx --test control -- --nocapture`

Expected: PASS.

### Task 4: Port `_mm*_permute_ps`

**Files:**
- Modify: `z3_avx/src/lanes.rs`
- Modify: `z3_avx/src/permute.rs`
- Create: `z3_avx/tests/permute_ps.rs`

- [ ] **Step 1: Write failing literal identity tests**

Add tests proving `_mm256_permute_ps` and `_mm512_permute_ps` with `_MM_SHUFFLE(3,2,1,0)` are identity operations.

- [ ] **Step 2: Run tests to verify failure**

Run: `cargo test -p z3_avx --test permute_ps literal_identity -- --nocapture`

Expected: FAIL because permute helpers are not implemented.

- [ ] **Step 3: Implement selector and permute helpers**

Implement `create_if_tree`, 128-bit lane extraction, 32-bit element selection within a 128-bit lane, `permute_ps_generic`, `mm256_permute_ps`, and `mm512_permute_ps`.

- [ ] **Step 4: Run literal identity tests**

Run: `cargo test -p z3_avx --test permute_ps literal_identity -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Write failing symbolic imm8 discovery tests**

Add tests where an 8-bit symbolic immediate is solved under `input == output`, expecting `_MM_SHUFFLE(3,2,1,0)`.

- [ ] **Step 6: Run test to verify failure or incomplete behavior**

Run: `cargo test -p z3_avx --test permute_ps symbolic_identity -- --nocapture`

Expected: FAIL until symbolic immediate handling is correct.

- [ ] **Step 7: Complete symbolic immediate path**

Ensure `Imm8::Expr` is extracted with Z3 bit operations and used by the same selector path as literals.

- [ ] **Step 8: Run permute tests**

Run: `cargo test -p z3_avx --test permute_ps -- --nocapture`

Expected: PASS.

### Task 5: Final Verification

**Files:**
- All first-wave Rust files.

- [ ] **Step 1: Format Rust code**

Run: `cargo fmt --all`

Expected: no errors.

- [ ] **Step 2: Run Rust tests**

Run: `cargo test -p z3_avx`

Expected: PASS.

- [ ] **Step 3: Run required Python checks**

Run: `uv run pytest`
Run: `uv run ruff check .`
Run: `uv run vulture`

Expected: pytest and ruff pass; inspect vulture output for new dead code relevance.

- [ ] **Step 4: Review git diff**

Run: `git status --short`
Run: `git diff --stat`

Expected: only planned Rust workspace/spec/plan files changed.
