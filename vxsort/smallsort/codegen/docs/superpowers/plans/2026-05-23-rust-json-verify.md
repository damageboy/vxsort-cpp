# Rust JSON Verify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Rust `verify` command that consumes solution JSON and proves encoded bitonic sorter paths correct with Z3.

**Architecture:** JSON is the boundary between commands: `solve` writes JSON, while `verify` reads JSON and checks exactly the paths/gadgets represented there. The verifier mirrors Python's symbolic simulation and chunked bitonic-recursion proof, but uses Rust `z3_avx` and Rust imported solution structures.

**Status:** Executed on 2026-05-23 with subagents for CLI scaffolding, verifier API reconnaissance, and review. Direct `solve` ASM emission was removed so `solve` writes JSON only and `json-to-asm` consumes that JSON for assembly. Final implementation passed `cargo fmt --all --check`, `cargo clippy --all-targets --all-features --release -- -D warnings`, and `cargo test --release -q`.

**Tech Stack:** Rust 2024, `clap`, `z3`, local `z3_avx`, local `gadget_synth`, Rust release-mode tests.

---

## File Structure

- Modify `gadget_synth/src/intrinsics.rs`: expose concrete intrinsic dispatch to verifier.
- Modify `gadget_synth/src/lib.rs`: re-export the dispatch helper.
- Create `bitonic_codegen/src/verifier.rs`: symbolic verifier core, chunking, concrete gadget application, and JSON verification API.
- Modify `bitonic_codegen/src/lib.rs`: add `VerifyJsonArgs`, `CliCommand::Verify`, module export, and public verify API.
- Modify `bitonic_codegen/src/main.rs`: route `verify` subcommand and print success/failure.
- Create `bitonic_codegen/tests/verifier.rs`: focused verifier unit/integration tests.
- Modify `bitonic_codegen/tests/cli.rs`: CLI parse/binary behavior tests for `verify`.
- Modify `bitonic_codegen/Cargo.toml`: add direct `z3` and `z3_avx` dependencies if needed by verifier.

## Task 1: CLI Contract and JSON-Only Boundary

**Files:**
- Modify: `bitonic_codegen/src/lib.rs`
- Modify: `bitonic_codegen/src/main.rs`
- Modify: `bitonic_codegen/tests/cli.rs`

- [x] **Step 1: Write failing CLI tests**

Add tests that `bitonic-codegen verify --input fixture.json --top-k 1 --workers 1` parses, and the binary rejects unsupported/missing input cleanly.

- [x] **Step 2: Run tests to verify RED**

Run: `cargo test --release -q -p bitonic_codegen --test cli verify`

Expected: FAIL because `CliCommand::Verify` does not exist.

- [x] **Step 3: Add minimal CLI structures**

Add `VerifyJsonArgs { input: PathBuf, top_k: Option<usize>, workers: usize }` and `CliCommand::Verify(VerifyJsonArgs)`.

- [x] **Step 4: Add main routing stub**

Route `CliCommand::Verify` to a not-yet-implemented `verify_solution_json` function so CLI parsing passes.

- [x] **Step 5: Re-run focused CLI tests**

Run: `cargo test --release -q -p bitonic_codegen --test cli verify`

Expected: CLI parse tests pass; execution test may still fail until Task 3.

## Task 2: Verifier Core

**Files:**
- Modify: `gadget_synth/src/intrinsics.rs`
- Modify: `gadget_synth/src/lib.rs`
- Create: `bitonic_codegen/src/verifier.rs`
- Modify: `bitonic_codegen/Cargo.toml`
- Create: `bitonic_codegen/tests/verifier.rs`

- [x] **Step 1: Write failing verifier tests**

Add tests for:
- `compute_recursion_boundaries(8) == [2]` and `num_verify_chunks(8) == 3`.
- A hand-authored one-stage AVX2 i64 JSON path verifies.
- The same path with an identity/bad output state fails with a counterexample.

- [x] **Step 2: Run tests to verify RED**

Run: `cargo test --release -q -p bitonic_codegen --test verifier`

Expected: FAIL because the verifier module/API does not exist.

- [x] **Step 3: Expose intrinsic dispatch**

Make `gadget_synth::intrinsics` or a narrow `dispatch_concrete_intrinsic` wrapper public enough for `bitonic_codegen::verifier` to apply concrete `InstructionSpec`s.

- [x] **Step 4: Implement verifier data types**

Add `VerifyStep`, `VerificationResult`, and `VerifyError`. Build verifier steps from imported JSON paths by looking up each path edge in the `TransitionTable`. If a path edge has multiple gadgets, return an explicit ambiguity error for this milestone.

- [x] **Step 5: Implement symbolic simulation**

Create symbolic lane inputs, pack registers from state labels, apply top/bottom gadget instruction chains, apply signed min/max except for final natural-order stage, and extract final elements by label order.

- [x] **Step 6: Implement monolithic and chunked proofs**

Port Python's recursion boundaries, chunk levels, precondition sorted groups, per-group postcondition proof, and counterexample extraction.

- [x] **Step 7: Re-run focused verifier tests**

Run: `cargo test --release -q -p bitonic_codegen --test verifier`

Expected: PASS.

## Task 3: Integration and Verification

**Files:**
- Modify: `bitonic_codegen/src/lib.rs`
- Modify: `bitonic_codegen/src/main.rs`
- Modify: `bitonic_codegen/tests/cli.rs`
- Modify: `bitonic_codegen/tests/verifier.rs`

- [x] **Step 1: Wire `verify_solution_json`**

Read JSON, select the first `top_k.unwrap_or(all)` paths in deterministic imported order, verify them, and return a summary with path count, chunk count, and failures.

- [x] **Step 2: Print user-facing output**

On success print `All N paths verified correct.` On failure print each failed path index and counterexample, then exit non-zero through existing `exit_on_error`.

- [x] **Step 3: Add binary success/failure tests**

Use temporary JSON fixtures to test success and failure paths.

- [x] **Step 4: Run focused Rust tests**

Run: `cargo test --release -q -p bitonic_codegen --test verifier --test cli`

Expected: PASS and under one minute.

- [x] **Step 5: Run required Rust checks**

Run:
- `cargo fmt --all --check`
- `cargo clippy --all-targets --all-features --release -- -D warnings`
- `cargo test --release -q -p bitonic_codegen --test verifier --test cli`

Expected: PASS. If the full workspace is needed due shared crate changes, run the full command set from `AGENTS.md`.
