# Rust uiCA Estimate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Rust `estimate` command that reads solution JSON, runs in-process uiCA simulation for each selected path, writes per-path HTML execution traces, and prints a ranked throughput table with clickable trace links. LLVM-MCA is not part of this Rust command.

**Architecture:** The Rust command should reuse the existing JSON boundary and `InstructionStream` lowering rather than parsing emitted assembly. `UiPackScorer` gets a small reporting API over its existing decoded-IR simulation path, while a new estimator module owns batch selection, output-directory management, trace writing, ranking, and report data.

**Tech Stack:** Rust 2024, `clap`, local `uica-core`, local `uica-data`, local `uica-model`, existing `InstructionStream`, release-mode Cargo tests.

---

## Current Code Findings

- Python entrypoint: `pyproject.toml` maps `estimate = "bitonic_compiler:estimate_main"`.
- Python command path: `src/bitonic_compiler.py::estimate_main()` injects `--estimate-only`, then `estimate_only_from_json()` loads JSON/checkpoints and calls `perf_estimator.estimate_solutions()`.
- Python estimator behavior to preserve only at the user-experience level: `src/perf_estimator.py::estimate_solutions()` reads selected paths, emits per-path artifacts, runs a simulator, and `print_estimation_table()` prints a final ranked table. Its LLVM-MCA-specific assembly and analysis path is legacy context, not a Rust implementation target.
- Rust already has the intended simulation engine: `rust/vxsort_codegen/src/uica_scoring.rs::UiPackScorer::full_score_block()` lowers modeled instructions to synthetic `DecodedInstruction`s and calls `uica_core::simulate()`.
- Rust uiCA trace support already exists: `uica_core::simulate()` with `SimulationOptions { include_reports: true, .. }` returns `ReportBundle`, and `uica_core::report::render_trace_html(&reports.trace)` renders the Python-compatible `Execution Trace` HTML.
- Rust CLI currently has `score-json`, but it is a debug command for one path/prefix and prints instruction keys. It does not batch, rank, write traces, or produce the Python estimate-style final table.

## File Structure

- Modify `rust/vxsort_codegen/src/lib.rs`
  - Add `EstimateJsonArgs`.
  - Add `CliCommand::Estimate`.
  - Export new estimator types/API.
- Modify `rust/vxsort_codegen/src/main.rs`
  - Route `estimate`.
  - Print progress and the final ranked table.
  - Keep `score-json` available as a verbose debug command.
- Modify `rust/vxsort_codegen/src/uica_scoring.rs`
  - Add a public simulation/report method that returns throughput, cycle counts, instruction counts, and optional reports.
  - Make `full_score_block()` call that method to avoid duplicate uiCA setup.
- Create `rust/vxsort_codegen/src/uica_estimator.rs`
  - Read/import solution JSON.
  - Lower assigned paths or graph paths to `InstructionStream`.
  - Select paths by deterministic JSON order before simulation.
  - Run uiCA simulation per selected block.
  - Write HTML trace files.
  - Sort finite results by throughput, then instruction count, then original path index.
  - Return a structured report for CLI printing and tests.
- Create `rust/vxsort_codegen/tests/uica_estimator.rs`
  - Focused library tests for selection, trace writing, ranking, and per-path error handling.
- Modify `rust/vxsort_codegen/tests/uica_scoring.rs`
  - Add a report-producing simulation test.
- Modify `rust/vxsort_codegen/tests/cli.rs`
  - Add parse and binary smoke tests for `estimate`.

## Command Contract

Target command:

```bash
cargo run -p vxsort_codegen -- estimate \
  --input bitonic_solutions_AVX2_i64.json \
  --target-cpu SKL \
  --uica-data-dir uica-data \
  --top-k 5
```

CLI options:

- `--input PATH` required JSON solution file.
- `--target-cpu CPU` required single uiCA architecture for the first implementation.
- `--uica-data-dir DIR` default `uica-data`.
- `--top-k N` optional cap on paths to simulate, applied to deterministic imported JSON order.
- `--output-dir DIR` optional trace output directory; default is a fresh temp directory named like `vxsort_uica_estimate_<pid>_<millis>`.
- `--no-traces` optional escape hatch for throughput-only batch scoring.

Output:

- Start line: input path, imported path count, selected path count, target CPU, output directory.
- Per-path progress line before each full simulation.
- Final table sorted by uiCA throughput:
  - `Rank`
  - `Path`
  - `TP`
  - `Cycles`
  - `Iters`
  - `Instr`
  - `Trace`
  - `Status`
- Trace column should use OSC 8 hyperlinks with `file://<absolute_path>` and also remain readable when the terminal ignores hyperlinks.

Out of scope for the first pass:

- Any LLVM-MCA compatibility or invocation in Rust.
- Checkpoint directory input.
- Cost-model preselection equivalent to Python `PathSelector`.
- Multi-target comma-separated estimate runs.
- Automatically launching the browser.

## Task 1: CLI Shape and Public Estimator API

**Files:**
- Modify: `rust/vxsort_codegen/src/lib.rs`
- Modify: `rust/vxsort_codegen/src/main.rs`
- Create: `rust/vxsort_codegen/src/uica_estimator.rs`
- Modify: `rust/vxsort_codegen/tests/cli.rs`

- [ ] **Step 1: Write failing CLI parse tests**

Add tests that parse:

```rust
CliArgs::try_parse_from([
    "vxsort-codegen",
    "estimate",
    "--input",
    "solutions.json",
    "--target-cpu",
    "SKL",
    "--top-k",
    "5",
    "--output-dir",
    "/tmp/vxsort-uica-estimate",
])
```

Expected: FAIL because `CliCommand::Estimate` and `EstimateJsonArgs` do not exist.

- [ ] **Step 2: Add CLI structs**

Add:

```rust
#[derive(Debug, Args)]
pub struct EstimateJsonArgs {
    #[arg(long = "input")]
    pub input: PathBuf,

    #[arg(long = "target-cpu")]
    pub target_cpu: String,

    #[arg(long = "uica-data-dir", default_value = "uica-data")]
    pub uica_data_dir: PathBuf,

    #[arg(long = "top-k")]
    pub top_k: Option<usize>,

    #[arg(long = "output-dir")]
    pub output_dir: Option<PathBuf>,

    #[arg(long = "no-traces")]
    pub no_traces: bool,
}
```

Add `Estimate(EstimateJsonArgs)` to `CliCommand`.

- [ ] **Step 3: Add a compile-only estimator module skeleton**

Create `uica_estimator.rs` with:

```rust
#[derive(Clone, Debug)]
pub struct EstimateOptions {
    pub target_cpu: String,
    pub uica_data_dir: PathBuf,
    pub top_k: Option<usize>,
    pub output_dir: Option<PathBuf>,
    pub write_traces: bool,
}

#[derive(Clone, Debug)]
pub struct EstimateReport {
    pub input_path: PathBuf,
    pub target_cpu: String,
    pub output_dir: PathBuf,
    pub imported_path_count: usize,
    pub results: Vec<EstimateResult>,
}

#[derive(Clone, Debug)]
pub struct EstimateResult {
    pub path_index: usize,
    pub rank: Option<usize>,
    pub throughput: Option<f64>,
    pub cycles_simulated: u32,
    pub iterations_simulated: u32,
    pub instruction_count: usize,
    pub trace_path: Option<PathBuf>,
    pub error: Option<String>,
}
```

Expose `pub mod uica_estimator;` and `pub use uica_estimator::{...};` from `lib.rs`.

- [ ] **Step 4: Route the CLI to a stub**

In `main.rs`, route `CliCommand::Estimate(args)` to `estimate_solution_json(&args.input, options)` and print a placeholder until later tasks fill the report.

- [ ] **Step 5: Run focused CLI tests**

Run:

```bash
cargo test --release -q -p vxsort_codegen --test cli estimate
```

Expected: CLI parse tests pass; binary behavior tests may still be pending.

## Task 2: Reusable uiCA Simulation With Reports

**Files:**
- Modify: `rust/vxsort_codegen/src/uica_scoring.rs`
- Modify: `rust/vxsort_codegen/tests/uica_scoring.rs`

- [ ] **Step 1: Write a failing report test**

Add a test using existing `scorer_with_records()` and `vpermq_instruction()` helpers:

```rust
let simulation = scorer
    .simulate_block(&block(vec![vpermq_instruction()]), true)
    .expect("uiCA report simulation should succeed");
assert!(simulation.throughput_cycles_per_iteration.is_finite());
assert!(simulation.reports.is_some());
let html = uica_core::report::render_trace_html(&simulation.reports.unwrap().trace)
    .expect("trace html should render");
assert!(html.contains("Execution Trace"));
```

Expected: FAIL because `simulate_block()` does not exist.

- [ ] **Step 2: Add a public simulation result type**

Add to `uica_scoring.rs`:

```rust
pub struct UiBlockSimulation {
    pub instruction_count: usize,
    pub throughput_cycles_per_iteration: f64,
    pub cycles_simulated: u32,
    pub iterations_simulated: u32,
    pub reports: Option<uica_model::ReportBundle>,
}
```

- [ ] **Step 3: Implement `UiPackScorer::simulate_block()`**

Refactor the decoded-instruction build loop from `full_score_block()` into:

```rust
pub fn simulate_block(
    &self,
    block: &InstructionBlock,
    include_reports: bool,
) -> Result<UiBlockSimulation, String>
```

Use:

```rust
SimulationOptions {
    include_reports,
    include_trace: false,
    ..SimulationOptions::default()
}
```

Keep `include_trace: false`; HTML traces come from reports, not the plain event trace writer.

- [ ] **Step 4: Reimplement `full_score_block()` via `simulate_block()`**

`full_score_block()` should call `simulate_block(block, false)` and return `PathCost::new(instruction_count as u32, throughput, throughput)`.

- [ ] **Step 5: Run focused scoring tests**

Run:

```bash
cargo test --release -q -p vxsort_codegen --test uica_scoring simulate_block
cargo test --release -q -p vxsort_codegen --test uica_scoring full_score
```

Expected: PASS.

## Task 3: JSON Import, Path Selection, and Block Lowering

**Files:**
- Modify: `rust/vxsort_codegen/src/uica_estimator.rs`
- Create: `rust/vxsort_codegen/tests/uica_estimator.rs`

- [ ] **Step 1: Write failing selection/lowering tests**

Use existing JSON exporter helpers from `json_importer.rs`/`json_exporter.rs` tests to build assigned-path JSON with two paths. Assert:

- `estimate_solution_json()` reports `imported_path_count == 2`.
- `top_k: Some(1)` simulates only path index `0`.
- assigned JSON uses `lower_assigned_paths()`.
- legacy roots/nodes JSON uses `lower_solution_paths()`.

Expected: FAIL because estimator logic is not implemented.

- [ ] **Step 2: Implement `select_blocks_from_imported()`**

For imported JSON:

```rust
let stream = if imported.assigned_paths.is_empty() {
    lower_solution_paths(&imported.metadata, &imported.transition_table, &imported.paths, LoweringOptions::default())
} else {
    lower_assigned_paths(&imported.metadata, &imported.assigned_paths, LoweringOptions::default())
};
```

Return `(path_index, InstructionBlock)` pairs in original order, capped by `top_k`.

- [ ] **Step 3: Validate empty input behavior**

If no paths are present, return an `EstimateReport` with zero results and no error. The CLI should print `No paths to estimate.`

- [ ] **Step 4: Run estimator selection tests**

Run:

```bash
cargo test --release -q -p vxsort_codegen --test uica_estimator selection
```

Expected: PASS.

## Task 4: Batch Simulation, Trace Writing, and Ranking

**Files:**
- Modify: `rust/vxsort_codegen/src/uica_estimator.rs`
- Modify: `rust/vxsort_codegen/tests/uica_estimator.rs`

- [ ] **Step 1: Write failing trace/ranking tests**

Use a fixture uiCA datapack with records for the instructions generated by the test paths. Assert:

- an output directory is created when omitted;
- `solution_000_trace.html` exists when traces are enabled;
- the trace file contains `Execution Trace`;
- finite results are ranked by increasing throughput;
- an unsupported instruction becomes an error row instead of aborting the batch.

Expected: FAIL because simulation and trace writing are not implemented.

- [ ] **Step 2: Implement output directory creation**

Use `std::env::temp_dir()` plus process id and `SystemTime::now().duration_since(UNIX_EPOCH).as_millis()` when `options.output_dir` is `None`.

Create the directory with `fs::create_dir_all()`.

- [ ] **Step 3: Implement per-block simulation**

Instantiate one `UiPackScorer::from_data_dir(&options.uica_data_dir, &options.target_cpu)` for the batch.

For each selected block:

```rust
match scorer.simulate_block(&block, options.write_traces) {
    Ok(simulation) => { ... }
    Err(error) => { ... }
}
```

Do not stop the batch for per-path simulation errors. Store `error` in the row and sort errored rows after finite throughput rows.

- [ ] **Step 4: Render and write HTML traces**

When `write_traces` is true:

```rust
let reports = simulation.reports.as_ref().ok_or_else(...)?;
let html = uica_core::report::render_trace_html(&reports.trace)?;
fs::write(&trace_path, html)?;
```

Use deterministic filenames:

```text
solution_000_trace.html
solution_001_trace.html
```

where the number is the original zero-based path index.

- [ ] **Step 5: Rank results**

Sort table order by:

1. finite throughput before errors;
2. `throughput.total_cmp()`;
3. `instruction_count`;
4. original `path_index`.

Set `rank` only on successful finite rows.

- [ ] **Step 6: Run focused estimator tests**

Run:

```bash
cargo test --release -q -p vxsort_codegen --test uica_estimator
```

Expected: PASS.

## Task 5: Final CLI Table and Hyperlinks

**Files:**
- Modify: `rust/vxsort_codegen/src/main.rs`
- Modify: `rust/vxsort_codegen/tests/cli.rs`

- [ ] **Step 1: Write failing binary smoke test**

Create a temp JSON and temp uiCA data directory, then use the existing binary-test style:

```rust
let output = Command::new(env!("CARGO_BIN_EXE_vxsort_codegen"))
    .args([
        "estimate",
        "--input",
        input_path.to_str().expect("json path should be utf-8"),
        "--target-cpu",
        "SKL",
        "--uica-data-dir",
        data_dir.to_str().expect("fixture path should be utf-8"),
        "--output-dir",
        output_dir.to_str().expect("output path should be utf-8"),
    ])
    .output()
    .expect("binary should run");
```

Assert stdout contains:

- `uiCA estimate`
- `target cpu: SKL`
- `output dir:`
- `Rank`
- `TP`
- `Trace`
- `solution_000_trace.html`

Assert the trace HTML file exists.

- [ ] **Step 2: Implement table formatting**

Keep it dependency-free. Add local helpers in `main.rs`:

```rust
fn print_estimate_report(report: &EstimateReport) { ... }
fn terminal_file_link(path: &Path) -> String { ... }
```

Use fixed-width columns and an OSC 8 hyperlink:

```rust
format!(
    "\x1b]8;;file://{}\x1b\\{}\x1b]8;;\x1b\\",
    absolute.display(),
    label
)
```

If `canonicalize()` fails, fall back to the plain path label.

- [ ] **Step 3: Print warnings after the table**

For errored rows, print:

```text
Warnings:
  path 3: unsupported instruction for uiCA scoring: ...
```

- [ ] **Step 4: Run focused CLI tests**

Run:

```bash
cargo test --release -q -p vxsort_codegen --test cli estimate
```

Expected: PASS.

## Task 6: Required Verification

**Files:**
- All Rust files touched in this plan.

- [ ] **Step 1: Format check**

Run:

```bash
cargo fmt --all --check
```

Expected: PASS.

- [ ] **Step 2: Clippy**

Run:

```bash
cargo clippy --all-targets --all-features --release -- -D warnings
```

Expected: PASS.

- [ ] **Step 3: Focused Rust tests**

Run:

```bash
cargo test --release -q -p vxsort_codegen --test uica_scoring --test uica_estimator --test cli
```

Expected: PASS and under one minute.

- [ ] **Step 4: Optional smoke with real data**

If local `uica-data` contains `SKL`, run:

```bash
cargo run -p vxsort_codegen -- estimate \
  --input <small-solution-json> \
  --target-cpu SKL \
  --uica-data-dir uica-data \
  --top-k 1
```

Expected: one ranked row, finite `TP`, and a readable `solution_000_trace.html`.
