# Multiplexer-Based Operand Selection for Multi-Instruction Gadgets

## Problem

Three bugs in the gadget synthesis pipeline:

1. **Broken data flow**: In 2-instruction gadgets, inst2 ignores inst1's output. `_apply_instructions` updates `current_reg` after inst1, but inst2's operand strings (`"top"`, `"bottom"`) resolve to original registers via `_substitute_register_names`, not `current_reg`. Every 2-instruction gadget degenerates to its last instruction alone.

2. **Missing combination**: `(single, dual)` template pairs are never generated -- only `(single, single)` and `(dual, single)`. Misses compositions like `permute(top) -> shuffle(result, bottom)`.

3. **Exporter register mapping bug**: `_format_instruction` uses `dest_reg if args["a"] == "top" else other_reg`, which gives wrong physical registers for bottom-side instructions called with `(dest_reg=bottom, other_reg=top)`.

## Solution

### Multiplexer Encoding

For inst2's register operands, create Z3 `select` variables (2-bit BitVec constrained to {0,1,2}) choosing between `{top_reg, bottom_reg, prev_output}`. Z3 discovers the optimal wiring. Inst1 resolves normally (no prev exists).

### Source Resolution at Extraction

When extracting a concrete gadget from a Z3 model, resolve each select variable to a source name string (`"top"`, `"bottom"`, `"prev"`) stored in `InstructionSpec.args`. Downstream consumers (exporters, cost model) see the resolved string.

### K-Per-Wiring Algorithm

Outer loop discovers unique wirings via `Optimize.minimize`, inner loop finds K smallest unique output states per wiring. Total solutions per template = `valid_wirings x K`. Backward compatible: depth-1 has no select vars, so outer loop runs once.

### Exporter Fix

Change `_format_instruction` to absolute register mapping `(top_reg, bottom_reg)` instead of relative `(dest_reg, other_reg)`. Handle `"prev"` as a new source name.

### Template Combinations

Add `(single, dual)` to depth-2 generation. Remove dead `dual_insts_bottom_top`.

## Implementation Order

1. Exporter fix (independent, testable alone)
2. Mux encoding in `_apply_instructions`
3. K-per-wiring algorithm in `synthesize_gadget_with_symbolic`
4. Add `(single, dual)` templates
5. Remove dead code
6. Integration test

## Files Changed

- `bitonic_super_optimizer.py`: mux encoding, K-per-wiring, new templates, dead code removal
- `asm_exporter.py`: absolute register mapping, `"prev"` handling
- `json_exporter.py`: no structural change (args dict unchanged)
- `cost_model.py`: no change
- `path_selector.py`: no change
