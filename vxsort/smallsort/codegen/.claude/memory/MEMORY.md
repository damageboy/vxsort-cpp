# vxsort Codegen Memory

## Project Structure
- **z3_avx.py**: Z3 symbolic semantics for AVX instructions
- **bitonic_super_optimizer.py**: Main super-optimizer, registers instructions and enumerates templates
- **cost_model.py**: Instruction latency/throughput costs
- **asm_exporter.py**: Exports solutions to assembly with mnemonics
- **bitonic_compiler.py**: Entry point for generating solutions

## Z3 Determinism Fix (Feb 2026)

**Problem**: Runs were non-deterministic — identical inputs produced different outputs.

**Root cause**: Z3's `Solver.check()` returns non-deterministic models across process
invocations due to ASLR randomizing memory layout, which affects Z3's internal
pointer-based hash tables and SAT solver heuristics. Same constraints → same sat/unsat,
but different satisfying assignments (different concrete immediates → different output states).

**Key insight**: `imap_unordered` was a red herring. Even `Pool(processes=1)` and
sequential in-process execution were non-deterministic across Python invocations.

**Solution** (3 parts in `bitonic_super_optimizer.py`):
1. **Z3 `Optimize` with `minimize`**: For `max_solutions=1` (production default), use
   `Optimize` instead of `Solver`, with `minimize(term)` on all symbolic variables.
   Returns the unique lexicographic minimum — deterministic regardless of ASLR.
2. **Sort validated gadgets**: After `imap_unordered` collection, sort by
   `(parent_path, input_state, output_state)` for deterministic dict insertion order.
3. **Sort gadgets within groups + transition iteration**: Sort gadgets by canonical
   `sort_key()` (instruction names + args), iterate transitions in sorted order.

**Design decisions**:
- Removed `max_solutions_per_gadget` from production pipeline entirely (was never used)
- Worker hardcodes `max_solutions=1`; the parameter on `synthesize_gadget_with_symbolic`
  defaults to 1 but accepts >1 for tests (`test_symbolic_synthesis.py` uses 20)
- `z3_avx.py` can't use custom `Context()` — functions create bare `BitVecVal` literals
  without ctx, causing "Context mismatch" errors. Would need extensive refactoring.

**What NOT to try**:
- Fresh `Context()` per solve — z3_avx.py creates context-less literals, fails with mismatch
- `PYTHONHASHSEED=0` — doesn't affect Z3's C++ internals
- `imap` instead of `imap_unordered` — doesn't fix Z3 non-determinism, only return order
- `max_solutions=None` (enumerate all) — too slow for templates with many valid immediates

## Adding New Data Type Support (e.g., i64)

### 1. Z3 Semantics (z3_avx.py)
- Implement missing instruction semantics following existing patterns
- For i64: added `_unpack_epi64_generic()` and 6 wrapper functions
- Element width changes: i32=32bits (8/vec), i64=64bits (4/vec)

### 2. Register Instructions (bitonic_super_optimizer.py)
- Add conditional block to `_get_available_intrinsics()` for new VM+datatype combo
- Add templates to `_enumerate_single_input_instructions()` with symbolic immediates
- Add templates to `_enumerate_dual_input_instructions()` for dual-operand ops
- Use correct intrinsic variants: i32 uses `_ps/_epi32`, i64 uses `_pd/_epi64`

### 3. Instruction Costs (cost_model.py)
- Add entries to `_init_generic_costs()` with latency, throughput, ports
- Common pattern: permutes ~1-3 cycles, blends ~0.33-1 cycle, unpacks ~1 cycle

### 4. **CRITICAL: Assembly Mnemonics (asm_exporter.py)**
- **MUST update** `_intrinsic_to_asm_mnemonic()` mapping for new instructions
- i32 mnemonics: `vshufps`, `vpunpckldq`, `vblendps`, `vpermilps`, `vpermd`
- i64 mnemonics: `vshufpd`, `vpunpcklqdq`, `vblendpd`, `vpermilpd`, `vpermq`
- Without this, assembly export will use intrinsic names instead of mnemonics

### 5. Tests (test_z3_avx.py, test_super_vectorizer.py)
- Add unit tests for new Z3 semantics (test patterns with sat/unsat)
- Enable new datatype in parameterized integration tests

## Common Instruction Patterns

### Single-operand (lane permutations)
- **In-lane**: `permute_ps/pd` (128-bit lanes, 4-bit immediate per lane)
- **Cross-lane**: `permute4x64_epi64` (full register, 8-bit immediate)
- **Variable**: `permutexvar_epi32/64` (control vector in separate register)

### Dual-operand
- **Shuffle**: `shuffle_ps/pd` (select from both inputs within lanes)
- **Unpack**: `unpacklo/hi_epi32/64` (interleave elements, fixed pattern)
- **Blend**: `blend_ps/pd` (element-wise selection via immediate mask)
- **Align**: `alignr_epi32/64` (concatenate + shift right)

## Testing Strategy
1. Z3 unit tests: verify symbolic semantics with concrete values
2. Integration tests: verify synthesis finds valid solutions
3. Assembly generation: check mnemonics and element counts
4. Full pytest suite: ensure no regressions

## Path Selection Refactoring (Feb 2026)

**Problem**: Three different cost metrics created conflicts:
1. `best_gadget()` used instruction count for selection
2. `CostModel` used latency-based costs
3. Control vector (CV) penalty applied ad-hoc in 3 places

**Solution**: Created unified `PathSelector` system:

### New Module: path_selector.py
- **GadgetScore**: Unified scoring = latency + (CV_count × penalty_weight)
- **PathStep**: Explicit (node, gadget_index) tuple in path
- **CompletePath**: Full root-to-leaf path with total costs
- **PathSelector**: Modified A* that explores different gadgets at same node

### Cleanup (Removed Deprecated Code)
- **Removed**: `SolutionNode.best_gadget()`, `SolutionNode.cost`
- **Removed**: `CostModel.compute_costs()`, `CostModel.calculate_path_cost()`
- CostModel only created when PathSelector needs it
