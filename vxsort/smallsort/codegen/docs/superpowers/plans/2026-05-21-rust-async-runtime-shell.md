# Rust Async Runtime Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the ratatui UI onto the main thread while the wave engine runs on a background thread and sends runtime events over a channel.

**Architecture:** Keep `WaveEngine` and `TransitionTable` owned by one engine thread for now. The TUI owns the terminal on the main thread, receives `RuntimeEvent` values over `std::sync::mpsc`, redraws on a fixed tick, and uses an `Arc<AtomicBool>` cancel flag observed by the engine session. Non-TTY textual mode keeps using line output.

**Tech Stack:** Rust std threads/channels/atomics, existing `RuntimeSession`, ratatui/crossterm.

---

### Task 1: Channel Runtime Session

**Files:**
- Modify: `bitonic_codegen/src/runtime.rs`
- Test: `bitonic_codegen/tests/runtime.rs`

- [x] **Step 1: Write failing tests** proving a runtime session can forward events into a channel and reflect a shared cancel flag through `should_stop()`.
- [x] **Step 2: Run focused test** with `cargo test -p bitonic_codegen --test runtime channel -q`; expect unresolved `ChannelRuntimeSession`.
- [x] **Step 3: Implement `ChannelRuntimeSession`** using `std::sync::mpsc::Sender<RuntimeEvent>` and `Arc<AtomicBool>`.
- [x] **Step 4: Re-run focused test** and confirm it passes.

### Task 2: Main-Thread TUI Event Loop

**Files:**
- Modify: `bitonic_codegen/src/runtime_tui.rs`
- Test: `bitonic_codegen/tests/runtime_tui.rs`

- [x] **Step 1: Write failing test** for non-terminal state behavior: applying channel events and ticking without blocking, and setting cancellation on quit.
- [x] **Step 2: Run focused test** with `cargo test -p bitonic_codegen --test runtime_tui async -q`; expect missing async app API.
- [x] **Step 3: Split TUI into `TuiApp`/state operations plus terminal wrapper** so event draining and key handling can be tested separately.
- [x] **Step 4: Implement `run_tui_event_loop(receiver, cancel_flag)`** that owns terminal drawing on the main thread, polls channel events, polls keys, redraws on ticks, and exits on `RunFinished`, disconnect, `q`, or `Esc`.
- [x] **Step 5: Re-run focused TUI tests** and confirm they pass.

### Task 3: Spawn Engine For Textual TTY

**Files:**
- Modify: `bitonic_codegen/src/lib.rs`
- Modify: `bitonic_codegen/src/main.rs`
- Test: `bitonic_codegen/tests/cli.rs`

- [x] **Step 1: Write failing unit test** proving `RunConfig` can be cloned for background engine ownership.
- [x] **Step 2: Add `Clone` derives** to config/summary types needed for thread transfer.
- [x] **Step 3: Implement textual TTY path**: create channel/cancel flag, spawn an engine thread running `build_run_summary_with_session`, run TUI loop on main thread, join thread, and return `RunSummary`.
- [x] **Step 4: Keep captured textual mode line-oriented** so existing CLI tests still pass.
- [x] **Step 5: Run focused CLI/runtime tests** and a short TTY smoke command.

### Task 4: Rust Verification

**Files:**
- Rust workspace only.

- [x] **Step 1:** Run `cargo fmt --all --check`.
- [x] **Step 2:** Run `cargo test -p bitonic_codegen -q`.
- [x] **Step 3:** Run `cargo test --release -q`.
- [x] **Step 4:** Do not run Python tests/tooling for this Rust-only change.
