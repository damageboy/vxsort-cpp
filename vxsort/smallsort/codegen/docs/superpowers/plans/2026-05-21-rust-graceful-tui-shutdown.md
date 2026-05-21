# Rust Graceful TUI Shutdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `q` / `Esc` request cooperative engine cancellation while keeping the TUI open until the engine emits `RunFinished` or the event channel closes.

**Architecture:** The TUI main thread owns terminal rendering and a `TuiApp`; the engine thread observes the shared `Arc<AtomicBool>` through `ChannelRuntimeSession::should_stop()`. Quit controls set cancellation state but do not mark the TUI finished. Rendering shows a clear “cancelling - waiting for engine shutdown” status until `RunFinished` or disconnect ends the UI loop.

**Tech Stack:** Rust std atomics/channels, existing `RuntimeEvent`, ratatui test backend, Rust-only verification.

---

### Task 1: Cooperative Cancel State

**Files:**
- Modify: `rust/vxsort_codegen/src/runtime_tui.rs`
- Test: `rust/vxsort_codegen/tests/runtime_tui.rs`

- [x] **Step 1: Write failing tests**

Add tests proving:
- `TuiControl::Quit` sets the shared cancel flag.
- `TuiControl::Quit` does not make `TuiApp::is_finished()` true.
- `RunFinished` after cancellation marks the app finished.
- The `TuiState` exposes cancellation state for rendering.

- [x] **Step 2: Run focused test**

Run:

```bash
cargo test -p vxsort_codegen --test runtime_tui cancel -q
```

Expected: fail because quit currently marks the app finished and there is no exposed cancellation state.

- [x] **Step 3: Implement minimal state change**

Add `cancel_requested` to `TuiState`, expose `cancel_requested()`, and add a method to mark cancellation. Change `TuiApp::handle_control(TuiControl::Quit)` to set the cancel flag and mark state cancellation, but not set `finished = true`.

- [x] **Step 4: Run focused test**

Run:

```bash
cargo test -p vxsort_codegen --test runtime_tui cancel -q
```

Expected: pass.

### Task 2: Cancellation Rendering

**Files:**
- Modify: `rust/vxsort_codegen/src/runtime_tui.rs`
- Test: `rust/vxsort_codegen/tests/runtime_tui.rs`

- [x] **Step 1: Write failing render test**

Use `ratatui::backend::TestBackend` to render a cancelled state and assert the buffer includes:

```text
cancelling - waiting for engine shutdown
```

- [x] **Step 2: Run focused render test**

Run:

```bash
cargo test -p vxsort_codegen --test runtime_tui cancelling_render -q
```

Expected: fail because the text is not rendered yet.

- [x] **Step 3: Render cancellation status**

Show cancellation status in the summary and/or footer. Keep the UI compact and avoid adding new controls. The footer should still mention `q quit` only when not cancelling.

- [x] **Step 4: Run focused render test**

Run:

```bash
cargo test -p vxsort_codegen --test runtime_tui cancelling_render -q
```

Expected: pass.

### Task 3: Rust Verification

**Files:**
- Rust workspace only.

- [x] **Step 1:** Run `cargo fmt --all --check`.
- [x] **Step 2:** Run `cargo test -p vxsort_codegen -q`.
- [x] **Step 3:** Run `cargo test --release -q`.
- [x] **Step 4:** Do not run Python tests/tooling for this Rust-only change.
