# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vxsort is a fast, vectorized hybrid quicksort+bitonic sorting algorithm in C++. It supports AVX2 and AVX512 vector ISAs with multiple primitive types (i16, i32, i64, u16, u32, u64, f32, f64).

The `vxsort/smallsort/codegen/` directory contains a Python-based super-optimizer that generates optimized bitonic sorter implementations using Z3 SMT solver for correctness verification.

## Build Commands

### C++ (Main Library)

```bash
mkdir build && cd build
export CC=clang CXX=clang++
cmake .. -G Ninja
ninja
ctest -J $(nproc)
```

### Python Codegen (in vxsort/smallsort/codegen/)

**Always use `uv` for Python package management. Never use pip, poetry, or conda.**

```bash
uv sync                                    # Install dependencies
uv run pytest                              # Run Python tests
uv run ruff check .                        # Lint
uv run python src/bitonic_compiler.py --depth-limit=3 --gadget-depth 1 # Full synthesis (limited depth for speed)
```

### Rust Codegen crates (in vxsort/smallsort/codegen/)

**Run Rust tests in release mode by default.** Z3-heavy debug builds are slow enough to waste iteration time.

```bash
cargo fmt --all --check                    # Format check
cargo clippy --all-targets --all-features --release -- -D warnings # Lint Rust code
cargo test --release -q                    # Run Rust tests in optimized/release mode
```

### Quick integration testing during development

When you need to run the compiler to test a feature (not the full test suite), use **AVX2 i64** —
it's the fastest configuration because 64-bit elements mean only 4 lanes per YMM register,
so Z3 solves much faster than 8-lane i32 or 16-lane i16:

```bash
uv run python src/bitonic_compiler.py --vector-machine AVX2 --datatype i64 \
  --depth-limit 3 --gadget-depth 1 --top-k 5
```

### Post work-item checklist

When you finish working on any given feature, scope verification to the files and languages touched.
Do not run the full Python test/lint/dead-code toolchain for Rust-only changes, and do not run the
Rust test suite for Python-only changes. Cross-language checks are only needed when a change affects
shared behavior, generated artifacts, workspace configuration, or the bridge between the Python and
Rust implementations.

- For Python changes: run Python tests with `uv run pytest` when relevant, plus `uv run ruff check .`
  and `uv run vulture` when Python code is touched. Fix failures or inspect reported dead code.
- For Rust changes: after every Rust work item, run `cargo fmt --all --check`,
  `cargo clippy --all-targets --all-features --release -- -D warnings`, and the relevant Rust tests
  with `cargo test --release -q` (or a focused release-mode subset during iteration). Use release mode
  by default for Rust checks. Treat Rust compiler warnings as failures: fix warnings surfaced by
  `cargo test`, `cargo build`, or `cargo clippy` before considering the work item complete.
- In general, and specifically if tests are added, it is important to ensure the relevant test suite
  doesn't run for more than 1 minute of wall clock
- For documentation-only changes, no test suite is required unless the documentation change includes
  executable examples or generated files.

## Codegen Architecture

The super-optimizer generates AVX permutation sequences for bitonic sort stages:

1. **BitonicSorter** (`bitonic_sorter.py`) - Generates comparison stages for the sorting network
2. **BitonicSuperVectorizer** (`bitonic_super_optimizer.py`) - Main entry point; orchestrates synthesis, builds solution tree, computes costs
3. **GadgetSynthesizer** - Enumerates AVX instruction combinations, validates with Z3
4. **z3_avx.py** - Z3 bindings for AVX instruction semantics (symbolic register operations)
5. **CostModel** (`cost_model.py`) - CPU microarchitecture instruction costs (supports uops.info data)
6. **BitonicCompiler** (`bitonic_compiler.py`) - Generates assembly and JSON output

### Key Data Structures

- **VectorState**: Tracks element positions in top/bottom vectors
- **InstructionSpec**: Single AVX instruction with operands
- **PermutationGadget**: Sequence of 0-3 instructions validated by Z3
- **SolutionNode**: Tree node linking gadgets across stages

### ASM Export Architecture

**There must be exactly ONE assembly emitter.** The canonical ASM emission path lives in
`perf_estimator.py` (`_emit_gadget_asm` + `generate_solution_asm`). All ASM output — whether
for LLVM-MCA estimation or `--output-format=asm` — must go through this single code path.
Never duplicate gadget emission logic. The old `asm_exporter.py` `_format_instruction` /
`_print_solution_step_as_assembly` path is legacy and must be replaced.

### Performance Estimation

Performance estimation uses **LLVM-MCA** (not OSACA). The pipeline:
1. `generate_solution_asm()` emits Intel/NASM syntax assembly
2. `sanitize_asm_for_llvm_mca()` converts NASM constructs for LLVM-MCA's Intel parser
3. `llvm_mca_runner.py` runs `llvm-mca --json` and parses `BlockRThroughput` + `TotalCycles/Iterations`

Use `--estimate` or `--estimate-only` with `--target-cpu` to run estimation.
Use `--llvm-mca-path` if auto-detection doesn't find your LLVM installation.

### Generated Output

- `bitonic_solutions_*.json` - Solution database
- `bitonic_solutions_*.asm` - Annotated x86-64 assembly
- `vxsort/smallsort/avx2/*.generated.h` - C++ headers used by the main library
