# Rust Super-Optimizer Port Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue the Python-to-Rust port until the Rust super-optimizer can produce, score, export, and verify the same smallsort codegen artifacts as the current Python pipeline.

**Architecture:** Treat Python as the reference implementation and use parity harnesses as the migration ratchet. Keep the existing Rust crates split by responsibility: `z3_avx` for symbolic AVX semantics, `gadget_synth` for template generation and local Z3 gadget synthesis, and `bitonic_codegen` for orchestration, search, scoring, persistence, export, verification, and runtime UX.

**Tech Stack:** Python `uv` test/lint tooling; Rust workspace crates `z3_avx`, `gadget_synth`, `bitonic_codegen`; Z3; JSON/JSONL parity fixtures; zstd checkpoints; optional LLVM-MCA.

---

## Current Port Status Snapshot

### Fully or mostly ported

- `src/z3_avx.py` → `z3_avx/src/*`
  - AVX2/AVX512 i32/i64 intrinsic semantics are substantially ported.
  - Canonicalization and k-mask handling have strong Rust coverage.
- `src/bitonic_sorter.py` → `bitonic_codegen/src/bitonic_sorter.rs`
  - Recursive stage generation and pair ordering are covered by tests.
- Python intrinsic candidate menu → `gadget_synth/src/intrinsics.rs`
  - Current AVX2/AVX512 i32/i64 menu is represented.
- Python template graph model → `gadget_synth/src/types.rs` and `synthesizer.rs`
  - Depth-0/1/2 graph generation, muxes, and shared-prefix templates are partially ported with dump tooling.

### Partially ported

- `src/gadget_synthesizer.py` → `gadget_synth/src/synthesizer.rs`
  - Rust can synthesize local gadget graphs, but lacks some Python options: `exclude_outputs`, SMT2 dump hooks, full `max_solutions` mode, worker metadata/timing, and Python-style unified instruction dedup APIs.
- `src/wave_engine.py` → `bitonic_codegen/src/wave_engine.rs`
  - Rust has a synchronous wave-search shell with worker threads and top-k retention, but not Python's long-lived process pool, checkpointing, target-CPU candidate ranking, async LLVM-MCA scoring, or full budget/targeting parity.
- `src/transition_table.py` → `bitonic_codegen/src/transition_table.rs`
  - Core concepts are present, but dedup currently relies on derived gadget equality instead of explicit Python `sort_key()` / unified-instruction semantics.
- `src/textual_progress_ui.py` / runtime events → `bitonic_codegen/src/runtime*.rs`
  - Rust has a useful TUI/runtime event layer, but lacks structured file logging, memory telemetry, instruction latency/attempt tables, and MCA/checkpoint events.

### Not yet ported

- `src/json_exporter.py`
- Canonical ASM emission through `src/perf_estimator.py`; Python `--export-only` still routes through legacy `src/asm_exporter.py` and must be fixed before delegation is treated as canonical
- `src/bitonic_verifier.py`
- `src/cost_model.py`, `src/path_selector.py`, LLVM-MCA integration from `src/perf_estimator.py`
- `src/wave_checkpoint.py` and legacy `src/checkpoint.py`
- Python CLI modes: `--verify-only`, `--estimate-only`, `--export-only`, checkpoint/resume, `--list-cpus`, log file flags, NASM/LLVM-MCA command plumbing
- Non-i32/i64 datatype CLI behavior. Python exposes broader datatypes (`i16/u16/i32/u32/i64/u64/f32/f64`) even though the current super-optimizer intrinsic menu is effectively i32/i64; Rust must explicitly reject, alias, or roadmap each unsupported datatype instead of silently omitting parity.

## Important Invariants

- Python remains the behavioral oracle until Rust has an equivalent feature and parity tests.
- Keep default validation under roughly one minute; use AVX2 i64 for fast integration tests.
- Do not duplicate assembly-emission behavior casually. Project guidance says there must be exactly one canonical ASM emitter in `perf_estimator.py`. Until Python `--export-only` is routed through `perf_estimator.generate_solution_asm`, do not claim an ASM delegation path is canonical.
- Do not count a Rust CLI flag as ported until it changes engine behavior or fails with a clear unsupported-mode error.
- For each implemented slice, add a small parity fixture before adding broader support.

## File Map

### Existing Rust crates

- `z3_avx/src/*`
  - Symbolic AVX intrinsic semantics. Treat as mostly complete; touch only for verifier/export semantic gaps or new intrinsic coverage.
- `gadget_synth/src/types.rs`
  - Shared graph/gadget data model; needs unified instruction and stable sort-key helpers.
- `gadget_synth/src/synthesizer.rs`
  - Candidate generation and local Z3 synthesis; needs option parity and parity harness expansion.
- `gadget_synth/src/bin/dump_gadget_synth.rs`
  - Template parity dumper; extend only if synthesized-gadget dumps are reintroduced.
- `bitonic_codegen/src/lib.rs`
  - CLI/config parsing and summary construction; immediate target for honest config plumbing.
- `bitonic_codegen/src/wave_engine.rs`
  - Search orchestration; target for config parity, candidate ranking, wave semantics, checkpoint hooks, and scoring hooks.
- `bitonic_codegen/src/transition_table.rs`
  - Transition storage/path enumeration; target for stable ordering, dedup, serialization.
- `bitonic_codegen/src/scoring.rs`
  - Currently dummy scoring; target for `CostModel`/`PathSelector` equivalent.
- `bitonic_codegen/src/runtime.rs`, `runtime_tui.rs`
  - Runtime event and UI surface.

### Rust files likely to create

- `bitonic_codegen/src/cost_model.rs`
- `bitonic_codegen/src/path_selector.rs`
- `bitonic_codegen/src/checkpoint.rs`
- `bitonic_codegen/src/json_exporter.rs`
- `bitonic_codegen/src/verifier.rs`
- `bitonic_codegen/src/llvm_mca.rs` only if LLVM-MCA is ported instead of delegated
- `bitonic_codegen/tests/parity.rs`
- `bitonic_codegen/tests/checkpoint.rs`
- `bitonic_codegen/tests/json_exporter.rs`
- `bitonic_codegen/tests/verifier.rs`

### Existing Python reference files

- `src/gadget_synthesizer.py`
- `src/bitonic_types.py`
- `src/wave_engine.py`
- `src/transition_table.py`
- `src/path_selector.py`
- `src/cost_model.py`
- `src/json_exporter.py`
- `src/perf_estimator.py`
- `src/bitonic_verifier.py`
- `src/bitonic_compiler.py`
- `src/wave_checkpoint.py`

---

### Task 1: Freeze the Rust/Python parity contract

**Files:**
- Modify: `README.rust.md`
- Modify: `docs/z3-avx-rust-port/benchmarks.md` or create `docs/rust-superoptimizer-parity.md`
- Test/Reference: `tools/dump_gadget_synth.py`, `tools/compare_gadget_synth_dumps.py`

- [ ] **Step 1: Document in-scope Rust parity modes**

Define the initial target as:

```text
--vector-machine AVX2 --datatype i64 --num-vecs 2 --gadget-depth 1|2
--natural-order optional
--retroactive-input optional
--max-waves small values for tests
```

Explicitly mark broader AVX512, non-i32/i64 datatypes, ASM, LLVM-MCA, and end-to-end verifier as later roadmap items.

- [ ] **Step 2: Document intentionally unsupported modes**

List each Python CLI mode and datatype that Rust does not yet implement and state whether the Rust CLI should reject it, alias it to an existing implementation, or ignore it temporarily. Prefer explicit rejection over silent omission for `i16/u16/u32/u64/f32/f64` until intrinsic/template coverage exists.

- [ ] **Step 3: Run the existing template parity smoke**

Run:

```bash
PY=/tmp/vxsort-py-avx2-i64-depth2.jsonl
RS=/tmp/vxsort-rs-avx2-i64-depth2.jsonl
uv run python tools/dump_gadget_synth.py templates \
  --vector-machine AVX2 --datatype i64 --gadget-depth 2 \
  --exclude-shared-prefix --output "$PY"
cargo run -q -p gadget_synth --bin dump_gadget_synth -- templates \
  --arch avx2 --dtype i64 --gadget-depth 2 \
  --exclude-shared-prefix --output "$RS"
uv run python tools/compare_gadget_synth_dumps.py "$PY" "$RS"
```

Expected: `Dumps match`.

- [ ] **Step 4: Commit documentation only**

```bash
git add README.rust.md docs/rust-superoptimizer-parity.md docs/z3-avx-rust-port/benchmarks.md
git commit -m "docs: define rust superoptimizer parity target"
```

### Task 2: Make Rust CLI/config honest

**Files:**
- Modify: `bitonic_codegen/src/lib.rs`
- Modify: `bitonic_codegen/src/wave_engine.rs`
- Test: `bitonic_codegen/tests/cli.rs`
- Test: `bitonic_codegen/tests/wave_engine.rs`

- [ ] **Step 1: Add tests for parsed-but-unused options**

Add tests proving `RunConfig.depth_limit`, `RunConfig.max_gadget_solutions`, `RunConfig.target_cpus`, and `RunConfig.estimate` are either propagated to `WaveConfig` or rejected with a clear error. Add CLI tests for unsupported Python datatypes once they are represented in the Rust parser or explicitly document why Rust accepts only `i32`/`i64` for now.

- [ ] **Step 2: Add missing `WaveConfig` fields**

Extend `WaveConfig` with:

```rust
pub depth_limit: Option<usize>,
pub max_unique_outputs: usize,
pub target_cpus: Vec<String>,
pub estimate: bool,
```

- [ ] **Step 3: Pass config values from `build_run_summary_with_session`**

Ensure `RunConfig.max_gadget_solutions` reaches worker synthesis options, and `target_cpus` reaches candidate ordering/scoring code once implemented.

- [ ] **Step 4: Resolve `--gadget-depth 3` mismatch**

Either reject it in Rust CLI with the same message as `gadget_synth` or implement depth 3 in both languages. Prefer rejection now, because Python graph generation also effectively supports only depth <= 2.

- [ ] **Step 5: Add `--max-workers` alias**

Add a Python-compatible alias for `--workers`:

```rust
#[arg(long = "workers", long_alias = "max-workers", default_value_t = 0)]
```

- [ ] **Step 6: Verify Rust-only changes**

```bash
cargo fmt --all --check
cargo test --release -q -p bitonic_codegen
```

### Task 3: Port synthesis option parity and gadget dedup keys

**Files:**
- Modify: `gadget_synth/src/types.rs`
- Modify: `gadget_synth/src/synthesizer.rs`
- Modify: `bitonic_codegen/src/wave_engine.rs`
- Modify: `bitonic_codegen/src/scoring.rs`
- Test: `gadget_synth/tests/synthesis.rs`
- Test: `bitonic_codegen/tests/wave_engine.rs`
- Test: `bitonic_codegen/tests/scoring.rs`

- [ ] **Step 1: Add `PermutationGadget::sort_key()`**

Match Python's exact ordering semantics from `InstructionSpec.sort_key()` and `PermutationGadget.sort_key()`: sort only each instruction's argument key/value pairs, then preserve the original top-instruction sequence and bottom-instruction sequence in the gadget key. Do not sort or reorder instruction chains, because dependency order is semantic.

- [ ] **Step 2: Add unified instruction helpers**

Add an API equivalent to Python `PermutationGadget.unified_instructions()` / `instruction_count()` so shared-prefix gadgets are counted once.

- [ ] **Step 3: Change `TransitionTable::add_transition` to dedup by sort key**

Keep insertion deterministic and avoid accepting semantically duplicate gadgets if derived equality diverges from Python ordering.

- [ ] **Step 4: Pass `max_unique_outputs` from `WaveConfig` into `SynthesisOptions`**

Replace hardcoded worker option:

```rust
SynthesisOptions {
    max_unique_outputs: config.max_unique_outputs,
    allow_any_lane_order,
}
```

- [ ] **Step 5: Add a focused max-output test**

Use a template/input pair that has multiple valid output states. Assert `max_unique_outputs = 1` and `3` produce different counts or at least different truncation behavior.

- [ ] **Step 6: Verify gadget synth and wave tests**

```bash
cargo fmt --all --check
cargo test --release -q -p gadget_synth
cargo test --release -q -p bitonic_codegen
```

### Task 4: Add a fast cross-language wave-summary parity harness

**Files:**
- Create: `tools/dump_wave_summary.py` or extend `src/bitonic_compiler.py` with a test-only summary mode
- Modify: `bitonic_codegen/src/main.rs` or add a JSON summary flag in `bitonic_codegen/src/lib.rs`
- Create: `tools/compare_wave_summaries.py`
- Test: `tests/test_rust_wave_parity.py` if acceptable, otherwise a script-only gate documented in `README.rust.md`
- Test: `bitonic_codegen/tests/cli.rs`

- [ ] **Step 1: Define a compact summary schema**

Include only stable fields:

```json
{
  "arch": "avx2",
  "dtype": "i64",
  "num_vecs": 2,
  "gadget_depth": 1,
  "waves": [
    {"wave": 1, "target_stage": 0, "attempts": 32, "new_outputs": 4}
  ],
  "stages": [
    {"stage": 0, "attempts": 32, "unique_outputs": 4, "transition_count": 4}
  ]
}
```

- [ ] **Step 2: Add Python summary output for small deterministic runs**

Use `WaveEngine` directly. Avoid full JSON export/verification.

- [ ] **Step 3: Add Rust summary JSON output**

Do not change human-readable default output unless a new flag is supplied, e.g. `--summary-json`.

- [ ] **Step 4: Add comparer tolerances only where behavior intentionally differs**

Candidate ordering, transition counts, and wave stats should be exact for the frozen target. If exactness fails, document and decide before broadening tolerances.

- [ ] **Step 5: Verify with AVX2 i64**

```bash
uv run python tools/dump_wave_summary.py \
  --vector-machine AVX2 --datatype i64 --num-vecs 2 \
  --gadget-depth 1 --max-waves 1 --wave-attempts 50 --wave-outputs 10 \
  --output /tmp/py-wave.json
cargo run -q -p bitonic_codegen -- \
  --vector-machine avx2 --datatype i64 --num-vecs 2 \
  --gadget-depth 1 --max-waves 1 --wave-attempts 50 --wave-outputs 10 \
  --summary-json /tmp/rs-wave.json
uv run python tools/compare_wave_summaries.py /tmp/py-wave.json /tmp/rs-wave.json
```

### Task 5: Align transition table and wave-search semantics

**Files:**
- Modify: `bitonic_codegen/src/transition_table.rs`
- Modify: `bitonic_codegen/src/wave_engine.rs`
- Test: `bitonic_codegen/tests/transition_table.rs`
- Test: `bitonic_codegen/tests/wave_engine.rs`
- Reference: `src/transition_table.py`, `src/wave_engine.py`

- [ ] **Step 1: Make transition iteration deterministic**

Prefer `BTreeMap`/sorted snapshots or explicit sorting at API boundaries where Python finalization sorts transitions.

- [ ] **Step 2: Port target-CPU candidate ranking skeleton**

Add the API even if real costs are initially generic. Python ranks by summed reciprocal throughput and interleaves multiple CPU rankings.

- [ ] **Step 3: Align weakest-stage selection**

Compare Python `_pick_target_stage()` with Rust `pick_target_stage()` and update tests for zero-budget and unproductive-wave bubbling.

- [ ] **Step 4: Add `depth_limit` behavior or reject it**

If implemented, ensure Rust does not explore stages past the inclusive depth limit. If not implemented, reject the flag clearly.

- [ ] **Step 5: Verify parity harness after each semantic change**

Run the summary comparer from Task 4 after every meaningful wave-engine change.

### Task 6: Port cost model and deterministic top-k path selection

**Files:**
- Create: `bitonic_codegen/src/cost_model.rs`
- Create: `bitonic_codegen/src/path_selector.rs`
- Modify: `bitonic_codegen/src/scoring.rs`
- Modify: `bitonic_codegen/src/lib.rs`
- Modify: `bitonic_codegen/src/wave_engine.rs`
- Test: `bitonic_codegen/tests/scoring.rs`
- Test: `bitonic_codegen/tests/path_selector.rs`
- Reference: `src/cost_model.py`, `src/path_selector.py`, `src/intrinsic_registry.py`

- [ ] **Step 1: Port generic instruction costs and CPU aliases**

Start with Python generic defaults and architecture-name normalization. Defer full uops overlay until generic path selection works.

- [ ] **Step 2: Port control-vector penalty classification**

Match Python `_is_control_vector_instruction()` for `op_idx`, blendv mask, permutevar controls, and k-mask/control-vector operands.

- [ ] **Step 3: Replace `DummyScorer` with real gadget score**

Use deduped unified instruction count and cost sums.

- [ ] **Step 4: Port path assignment and A* top-k selection**

Start with small hand-built transition tables before connecting to live wave output.

- [ ] **Step 5: Add golden top-k tests**

Use fixtures with multiple gadget alternatives where generic cost ranking is unambiguous.

- [ ] **Step 6: Verify**

```bash
cargo fmt --all --check
cargo test --release -q -p bitonic_codegen --test scoring
cargo test --release -q -p bitonic_codegen --test path_selector
```

### Task 7: Add wave checkpoint/resume

**Files:**
- Create: `bitonic_codegen/src/checkpoint.rs`
- Modify: `bitonic_codegen/src/lib.rs`
- Modify: `bitonic_codegen/src/wave_engine.rs`
- Modify: `bitonic_codegen/Cargo.toml`
- Test: `bitonic_codegen/tests/checkpoint.rs`
- Reference: `src/wave_checkpoint.py`

- [ ] **Step 1: Decide compatibility target**

Prefer Python-compatible directory layout:

```text
master.json
stage_00.json.zst
stage_01.json.zst
scored_paths.json.zst
```

If exact compatibility is deferred, document schema differences.

- [ ] **Step 2: Serialize `WaveConfig` and stage data**

Include transitions, attempted pairs, forwarded outputs, zero-budget counters, unproductive counters, attempts, wave count, exhausted/stalled stages, and scored paths.

- [ ] **Step 3: Add atomic write helpers**

Match Python behavior: write temp file, fsync as practical, then rename.

- [ ] **Step 4: Add `--checkpoint-dir` and `--resume` CLI flags**

Checkpoint after wave/stage completion; resume should avoid reattempting already attempted `(input, candidate)` pairs.

- [ ] **Step 5: Verify checkpoint roundtrip**

Test: run one wave, save, load, compare transition table and counters exactly, continue for another wave.

### Task 8: Add JSON export/import compatibility

**Files:**
- Create: `bitonic_codegen/src/json_exporter.rs`
- Modify: `bitonic_codegen/src/lib.rs`
- Modify: `bitonic_codegen/src/main.rs`
- Test: `bitonic_codegen/tests/json_exporter.rs`
- Reference: `src/json_exporter.py`, `src/bitonic_verifier.py`

- [ ] **Step 1: Port Python solution JSON schema exactly**

Include metadata fields expected by Python:

```json
{
  "natural_order": false,
  "vector_machine": "AVX2",
  "primitive_type": "i64",
  "num_vecs": 2,
  "roots": ["n0"],
  "nodes": {
    "n0": {
      "stage": 0,
      "input_state": {"top": [1, 2, 3, 4], "bottom": [5, 6, 7, 8]},
      "output_state": {"top": [1, 2, 3, 4], "bottom": [5, 6, 7, 8]},
      "gadgets": [
        {
          "top_instructions": [],
          "bottom_instructions": [],
          "unified_instructions": [],
          "top_output_index": -1,
          "bottom_output_index": -1
        }
      ],
      "gadget_count": 1,
      "children": []
    }
  }
}
```

Each gadget record must include Python-compatible `top_instructions`, `bottom_instructions`, `unified_instructions`, `top_output_index`, and `bottom_output_index` fields.

- [ ] **Step 2: Add Rust conversion from transition table/scored paths to export records**

If a legacy `SolutionNode` equivalent is needed, implement it as a small internal compatibility layer.

- [ ] **Step 3: Add deterministic node IDs and child ordering**

JSON should be diffable across runs.

- [ ] **Step 4: Add `--output-path` for normal solve output**

Keep default filename behavior compatible with Python only after explicit review.

- [ ] **Step 5: Verify Python can load Rust JSON**

```bash
cargo run -q -p bitonic_codegen -- \
  --vector-machine avx2 --datatype i64 --num-vecs 2 \
  --gadget-depth 1 --max-waves 1 --top-k 1 \
  --output-path /tmp/rust-solutions.json
uv run python - <<'PY'
from bitonic_verifier import load_solutions_from_json
bundle = load_solutions_from_json('/tmp/rust-solutions.json')
print(len(bundle.roots))
PY
```

### Task 9: Delegate or port ASM export and LLVM-MCA estimation

**Files:**
- Prefer initially: no new Rust ASM emitter
- Modify: `README.rust.md`
- Optionally create later: `bitonic_codegen/src/llvm_mca.rs`
- Reference: `src/perf_estimator.py`, `src/asm_exporter.py`, `src/util/llvm_mca_runner.py`

- [ ] **Step 1: Choose ASM strategy**

Recommended interim strategy: Rust exports Python-compatible JSON, then Python handles ASM and LLVM-MCA only after `bitonic_compiler.py --export-only` is confirmed to route through `perf_estimator.generate_solution_asm` rather than legacy `asm_exporter` formatting.

- [ ] **Step 2: Add documentation for the delegated workflow**

Example:

```bash
# First fix/verify Python export-only uses perf_estimator.generate_solution_asm.
cargo run -q -p bitonic_codegen -- ... --output-path /tmp/rust-solutions.json
uv run python src/bitonic_compiler.py --export-only /tmp/rust-solutions.json --export-format asm
uv run python src/bitonic_compiler.py --estimate-only /tmp/rust-solutions.json --target-cpu ZEN4
```

- [ ] **Step 3: Add a Python prerequisite if delegation is used**

Before documenting delegation as supported, modify or verify Python `--export-only asm` so it uses the same canonical `perf_estimator.generate_solution_asm` path as estimation.

- [ ] **Step 4: Only port the emitter after explicit approval**

If porting, replace the canonical ASM logic wholesale rather than adding a second divergent formatter.

### Task 10: Port end-to-end verifier

**Files:**
- Create: `bitonic_codegen/src/verifier.rs`
- Modify: `bitonic_codegen/src/lib.rs`
- Modify: `bitonic_codegen/src/main.rs`
- Test: `bitonic_codegen/tests/verifier.rs`
- Reference: `src/bitonic_verifier.py`

- [ ] **Step 1: Start with JSON import and one selected path**

Prove a complete path sorts symbolic inputs for AVX2 i64. Do not start with multiprocessing/chunking.

- [ ] **Step 2: Port stage semantics**

Apply gadget instructions stage-by-stage. Perform compare-swap after each normal stage; skip compare-swap for final natural-order stage.

- [ ] **Step 3: Add a negative fixture**

Create a deliberately broken path and assert verifier returns a counterexample/failure.

- [ ] **Step 4: Add `--verify` and `--verify-only`**

If JSON import/export exists, `--verify-only` should accept both Python and Rust-generated JSON.

- [ ] **Step 5: Add chunking/parallelism only after the serial verifier is trusted**

Keep AVX2 i64 top-1 verification fast in default tests.

### Task 11: Structured logging and runtime telemetry parity

**Files:**
- Modify: `bitonic_codegen/src/runtime.rs`
- Modify: `bitonic_codegen/src/runtime_tui.rs`
- Modify: `bitonic_codegen/src/wave_engine.rs`
- Modify: `bitonic_codegen/src/lib.rs`
- Test: `bitonic_codegen/tests/runtime.rs`
- Test: `bitonic_codegen/tests/runtime_tui.rs`
- Reference: `src/runtime_logging.py`, `src/textual_progress_ui.py`

- [ ] **Step 1: Add log-file CLI flags**

Mirror Python names: `--log-file`, `--log-level`, `--log-format`, `--log-max-mb`, `--log-backups`.

- [ ] **Step 2: Add structured runtime event serialization**

Include wave start/end, stage progress, job attempts, discoveries, checkpoint save/load, scoring, and finish events.

- [ ] **Step 3: Add instruction stats and latency columns after real cost model exists**

Do not invent duplicate data before Task 6 lands.

- [ ] **Step 4: Preserve current TUI cancellation tests**

Ensure non-TTY fallback remains deterministic.

### Task 12: Final integration matrix

**Files:**
- Modify: `README.rust.md`
- Create or modify: `scripts/rust_superoptimizer_parity.sh`
- Test: CI or local script only initially

- [ ] **Step 1: Add the fast matrix**

Run:

```bash
cargo fmt --all --check
cargo test --release -q
uv run pytest tests/test_gadget_synth_dump.py tests/test_wave_engine.py tests/test_transition_table.py -q
```

- [ ] **Step 2: Add cross-language smoke gates**

Include:

- template dump parity for AVX2 i64 depth 1/2;
- one-wave summary parity for AVX2 i64;
- Rust JSON loads in Python verifier;
- ASM export delegated through Python from Rust JSON;
- optional LLVM-MCA smoke only when `llvm-mca` is available.

- [ ] **Step 3: Keep slow gates opt-in**

Do not run full AVX512/depth-2 synthesis by default.

---

## Recommended Immediate Next Slice

Start with Tasks 1–4:

1. Freeze and document the parity contract.
2. Make Rust CLI/config honest, especially parsed-but-unused flags.
3. Wire `max_gadget_solutions` into Rust synthesis and add unified instruction/dedup helpers.
4. Add a fast wave-summary parity harness.

This slice does not require committing to JSON/ASM/checkpoint schema decisions yet, but it prevents silent divergence and gives every later porting task a quick regression gate.
