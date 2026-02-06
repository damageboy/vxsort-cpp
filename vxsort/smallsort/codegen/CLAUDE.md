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
uv run pytest                              # Run tests
uv run ruff check .                        # Lint
uv run python src/bitonic_compiler.py --depth-limit=3 --gadget-depth 1 # Full synthesis (limited depth for speed)
```

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

### Generated Output

- `bitonic_solutions_*.json` - Solution database
- `bitonic_solutions_*.asm` - Annotated x86-64 assembly
- `vxsort/smallsort/avx2/*.generated.h` - C++ headers used by the main library

## Testing

C++ tests use Google Test, organized by ISA and data type (signed/unsigned/float). Python tests use pytest with the main test file being `test_super_vectorizer.py`.
