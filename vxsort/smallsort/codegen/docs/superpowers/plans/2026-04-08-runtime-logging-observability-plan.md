# Runtime Logging + Observability for Wave Search/LLVM-MCA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit, high-volume, file-backed logging with levels so we can verify (not assume) whether search, path discovery, and LLVM-MCA scoring are progressing concurrently.

**Architecture:** Introduce a centralized runtime logging layer (stdlib `logging`) with queue-based handlers and optional JSONL output. Thread logging config from CLI into `WaveEngine`, instrument critical search/discovery/scoring events, and emit per-LLVM-MCA completion logs. Keep UI updates separate from logging so observability survives UI stalls.

**Tech Stack:** Python stdlib `logging` (+ `logging.handlers`), existing Textual UI/runtime session, pytest/ruff.

---

## File structure and responsibilities

- **Create:** `src/runtime_logging.py`
  - Central logging configuration + logger factory
  - Level parsing, format selection (text/jsonl), file handler setup, rotation
  - Queue-based non-blocking handlers and teardown helper
- **Modify:** `src/bitonic_compiler.py`
  - CLI flags for logging configuration
  - Build and pass logging config into `WaveConfig` / runtime
- **Modify:** `src/wave_engine.py`
  - Inject logger usage for lifecycle/search/discovery/scoring events
  - Add periodic heartbeat event for “stuck” diagnosis
- **Modify:** `src/util/llvm_mca_runner.py`
  - Per-subprocess start/finish/error logging with duration and result metadata
- **Create:** `tests/test_runtime_logging.py`
  - Unit tests for config parsing, level behavior, file emission
- **Modify:** `tests/test_wave_engine.py`
  - Verify key events are emitted (discovery queued, batch scored, heartbeat)
- **Modify:** `tests/test_llvm_mca.py`
  - Verify logging on completion/errors (caplog or mocked logger)
- **Modify:** `README.md` (or `docs/` if preferred)
  - Add “Runtime logging” usage and troubleshooting section

---

### Task 1: Add logging backbone module

**Files:**
- Create: `src/runtime_logging.py`
- Test: `tests/test_runtime_logging.py`

- [ ] **Step 1: Write failing tests for logging config + file output**
  - Missing/invalid level handling
  - JSONL format emits machine-readable records
  - File handler writes events and rotates when configured

- [ ] **Step 2: Run targeted tests to confirm fail**
  - Run: `uv run pytest tests/test_runtime_logging.py -q`

- [ ] **Step 3: Implement `RuntimeLoggingConfig` + `configure_runtime_logging()`**
  - Fields: `enabled`, `level`, `file_path`, `format`, `max_bytes`, `backup_count`, `console`
  - Return `(logger, shutdown_fn)` so caller can flush/stop listener cleanly

- [ ] **Step 4: Implement structured event helper**
  - e.g. `log_event(logger, level, event, **fields)`
  - Include common fields (`ts`, `event`, `wave`, `stage`, `pending_discovery`, `pending_mca`, etc.)

- [ ] **Step 5: Re-run tests for pass**
  - Run: `uv run pytest tests/test_runtime_logging.py -q`

- [ ] **Step 6: Commit**
  - `git add src/runtime_logging.py tests/test_runtime_logging.py`
  - `git commit -m "Add runtime logging infrastructure with leveled file output"`

---

### Task 2: Expose logging controls in CLI and config plumbing

**Files:**
- Modify: `src/bitonic_compiler.py`
- Modify: `src/wave_engine.py`
- Test: `tests/test_wave_engine.py`

- [ ] **Step 1: Add CLI arguments (failing parser tests if present; otherwise add unit tests)**
  - `--log-file PATH`
  - `--log-level {DEBUG,INFO,WARNING,ERROR}`
  - `--log-format {text,json}`
  - `--log-max-mb N` (optional)
  - `--log-backups N` (optional)

- [ ] **Step 2: Add `WaveConfig` logging fields**
  - `log_file`, `log_level`, `log_format`, `log_max_mb`, `log_backups`

- [ ] **Step 3: Build logger in main entry and pass into runtime**
  - Ensure disabled-by-default behavior if no `--log-file`

- [ ] **Step 4: Verify parser/config tests**
  - Run: `uv run pytest tests/test_wave_engine.py -q`

- [ ] **Step 5: Commit**
  - `git add src/bitonic_compiler.py src/wave_engine.py tests/test_wave_engine.py`
  - `git commit -m "Plumb runtime logging options through compiler and wave config"`

---

### Task 3: Instrument WaveEngine high-value events

**Files:**
- Modify: `src/wave_engine.py`
- Test: `tests/test_wave_engine.py`

- [ ] **Step 1: Write failing tests for event emission**
  - Events for:
    - wave start/end
    - stage submit/drain counts
    - discovery anchor queued/processed
    - scoring batch submitted/completed
    - periodic heartbeat every X seconds (configurable)

- [ ] **Step 2: Add logger calls at INFO/DEBUG boundaries**
  - INFO: wave summary, scoring completion, major state transitions
  - DEBUG: per-batch details and counters

- [ ] **Step 3: Add heartbeat event**
  - Include counts: `in_flight`, `pending_jobs`, `pending_discovery`, `pending_mca`, `scored_total`
  - Emit even when no UI update happens

- [ ] **Step 4: Ensure logging cannot block main loop**
  - Queue handler only in hot paths
  - Avoid heavy string formatting unless level enabled

- [ ] **Step 5: Run tests**
  - `uv run pytest tests/test_wave_engine.py -q`

- [ ] **Step 6: Commit**
  - `git add src/wave_engine.py tests/test_wave_engine.py`
  - `git commit -m "Instrument wave search/discovery/scoring with leveled events"`

---

### Task 4: Instrument llvm-mca subprocess lifecycle

**Files:**
- Modify: `src/util/llvm_mca_runner.py`
- Modify: `tests/test_llvm_mca.py`

- [ ] **Step 1: Write failing tests around log events**
  - start event includes `solution_index`, `mcpu`
  - completion event includes elapsed ms + throughput/cycles
  - error event includes return code + stderr tail

- [ ] **Step 2: Implement logging in `run_llvm_mca()`**
  - Emit per-run events at DEBUG (or INFO if requested)
  - Keep payload bounded (truncate huge stderr)

- [ ] **Step 3: Run tests**
  - `uv run pytest tests/test_llvm_mca.py -q`

- [ ] **Step 4: Commit**
  - `git add src/util/llvm_mca_runner.py tests/test_llvm_mca.py`
  - `git commit -m "Add per-llvm-mca completion/error logging"`

---

### Task 5: Documentation and operator workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add runtime logging section**
  - Explain levels and format
  - Explain where logs are written and how to tail them

- [ ] **Step 2: Add “stuck run” triage commands**
  - Example:
    - `tail -f <logfile>`
    - `rg 'event=(heartbeat|mca_complete|discovery_chunk)' <logfile>`

- [ ] **Step 3: Commit**
  - `git add README.md`
  - `git commit -m "Document runtime logging and stuck-run triage workflow"`

---

### Task 6: Full verification gate

**Files:**
- Modify: all touched files from tasks 1-5

- [ ] **Step 1: Run full test suite**
  - `uv run pytest`

- [ ] **Step 2: Run lint**
  - `uv run ruff check .`

- [ ] **Step 3: Run dead-code scan + inspect output**
  - `uv run vulture`

- [ ] **Step 4: Run one integration command with logging enabled**
  - Example:
    - `uv run python src/bitonic_compiler.py --vector-machine AVX2 --datatype i64 --depth-limit 3 --gadget-depth 1 --top-k 5 --target-cpu ZEN4 --log-file /tmp/vxsort-wave.log --log-level DEBUG --log-format json`

- [ ] **Step 5: Validate expected runtime signals in log**
  - Search, discovery, scoring events interleaving over time
  - Non-zero `mca_complete` records while search still emits heartbeat/submit/drain

- [ ] **Step 6: Final commit**
  - `git add <all changed files>`
  - `git commit -m "Add end-to-end runtime observability for search and llvm-mca scoring"`

---

## Notes / guardrails

- Keep logging optional and cheap when disabled.
- Prefer structured fields over prose strings to avoid ambiguity.
- Treat log output as source-of-truth for “is it stuck?” conversations.
- If queue backlog grows unbounded, add backpressure metrics (but avoid blocking hot loop).
