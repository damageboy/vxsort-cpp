# Rust Wave Worker Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run wave-engine synthesis jobs on Rust worker threads while keeping `TransitionTable` and path scoring single-owner on the engine thread.

**Architecture:** The engine builds deterministic `SynthesisJob` order as before. Worker threads receive immutable job payloads, synthesize candidate outputs, and send results back. The engine applies completed results in original job order so transition insertion, path discovery, and summaries remain deterministic. Worker count is configurable by CLI and defaults to available hardware parallelism.

**Tech Stack:** Rust std threads/channels/atomics, existing `GadgetSynthesizer`, existing wave-engine tests.

---

### Task 1: Worker Config

**Files:**
- Modify: `bitonic_codegen/src/lib.rs`
- Test: `bitonic_codegen/tests/cli.rs`

- [x] **Step 1:** Add a failing CLI/config test for `--workers`.
- [x] **Step 2:** Add `workers` CLI parsing and `RunConfig::worker_count`.
- [x] **Step 3:** Resolve `0` workers to `std::thread::available_parallelism()`.
- [x] **Step 4:** Pass worker count into `WaveConfig`.

### Task 2: Deterministic Worker Execution

**Files:**
- Modify: `bitonic_codegen/src/wave_engine.rs`
- Test: `bitonic_codegen/tests/wave_engine.rs`

- [x] **Step 1:** Add a failing equivalence test comparing one-worker and two-worker stage execution.
- [x] **Step 2:** Split pure job synthesis from transition-table mutation.
- [x] **Step 3:** Add a bounded worker pool for stage jobs.
- [x] **Step 4:** Apply worker results in original job order.
- [x] **Step 5:** Keep `TransitionTable`, path discovery, and top-k retention owned by the engine thread.

### Task 3: Rust Verification

**Files:**
- Rust workspace only.

- [x] **Step 1:** Run `cargo fmt --all --check`.
- [x] **Step 2:** Run focused worker/CLI tests.
- [x] **Step 3:** Run `cargo test -p bitonic_codegen -q`.
- [x] **Step 4:** Run `cargo test --release -q`.
- [x] **Step 5:** Do not run Python tests/tooling for this Rust-only change.
