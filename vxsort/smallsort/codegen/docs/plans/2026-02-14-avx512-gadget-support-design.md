# AVX512 Gadget Support Design

## Summary

Add AVX512 (512-bit ZMM) gadget synthesis to the bitonic sort super-optimizer.
This enables sorting 16xi32 or 8xi64 elements per vector pair using the full
AVX512 instruction set including merge-masked variants.

## Decisions

- **Register width**: 512-bit ZMM registers (16xi32, 8xi64)
- **Instruction set**: Full core set + all mask variants already in z3_avx.py
- **K-mask modeling**: Separate intrinsics for masked vs unmasked (not optional params)
- **Implementation order**: i64 first (8 elements, simpler), then i32 (16 elements)
- **Sub-task structure**: 3 phases — ASM/cost layer, i64 gadgets, i32 gadgets

## Architecture

### Phase 1: ASM Encoding + Cost Model

**asm_exporter.py** — Add/verify mnemonic mappings for all AVX512 instructions:

| Intrinsic | Mnemonic | Notes |
|---|---|---|
| `_mm512_permutexvar_epi64` | `vpermq` | Already present |
| `_mm512_permutexvar_epi32` | `vpermd` | Already present |
| `_mm512_mask_permutexvar_epi32` | `vpermd` | Needs k-mask register syntax |
| `_mm512_mask_permutexvar_epi64` | `vpermq` | Needs k-mask register syntax |
| `_mm512_permutex2var_epi32` | `vpermi2d` or `vpermt2d` | New mapping needed |
| `_mm512_permutex2var_epi64` | `vpermi2q` or `vpermt2q` | New mapping needed |
| `_mm512_mask_permutex2var_epi32` | `vpermi2d` | New mapping + k-mask syntax |
| `_mm512_mask_permutex2var_epi64` | `vpermi2q` | New mapping + k-mask syntax |
| `_mm512_permute_ps` | `vpermilps` | Already present (via generic) |
| `_mm512_permute_pd` | `vpermilpd` | New mapping needed |
| `_mm512_mask_permute_ps` | `vpermilps` | Needs k-mask register syntax |
| `_mm512_mask_permute_pd` | `vpermilpd` | Needs k-mask register syntax |
| `_mm512_permutevar_ps` | `vpermilps` | New mapping needed |
| `_mm512_permutevar_pd` | `vpermilpd` | New mapping needed |
| `_mm512_mask_permutevar_ps` | `vpermilps` | Needs k-mask register syntax |
| `_mm512_mask_permutevar_pd` | `vpermilpd` | Needs k-mask register syntax |
| `_mm512_shuffle_ps` | `vshufps` | Already present |
| `_mm512_shuffle_pd` | `vshufpd` | New mapping needed |
| `_mm512_mask_shuffle_ps` | `vshufps` | Needs k-mask register syntax |
| `_mm512_mask_shuffle_pd` | `vshufpd` | Needs k-mask register syntax |
| `_mm512_unpacklo_epi32` | `vpunpckldq` | Already present |
| `_mm512_unpackhi_epi32` | `vpunpckhdq` | Already present |
| `_mm512_unpacklo_epi64` | `vpunpcklqdq` | New mapping needed |
| `_mm512_unpackhi_epi64` | `vpunpckhqdq` | New mapping needed |
| `_mm512_mask_unpacklo_epi32` | `vpunpckldq` | Needs k-mask register syntax |
| `_mm512_mask_unpackhi_epi32` | `vpunpckhdq` | Needs k-mask register syntax |
| `_mm512_mask_unpacklo_epi64` | `vpunpcklqdq` | Needs k-mask register syntax |
| `_mm512_mask_unpackhi_epi64` | `vpunpckhqdq` | Needs k-mask register syntax |
| `_mm512_shuffle_i32x4` | `vshufi32x4` | New mapping needed |
| `_mm512_alignr_epi32` | `valignd` | Already present |
| `_mm512_alignr_epi64` | `valignq` | Already present |
| `_mm512_mask_alignr_epi32` | `valignd` | Needs k-mask register syntax |
| `_mm512_mask_alignr_epi64` | `valignq` | Needs k-mask register syntax |

**K-mask assembly syntax**: AVX512 masked instructions use `{k1}` notation:
```asm
vpermq      zmm0{k1}, zmm1, zmm2    ; merge-masked permute
```
The `_format_instruction()` function needs to detect masked intrinsics and emit
the `{kN}` decorator on the destination register. A k-mask register must be
allocated from `RegisterAllocator` (k1-k7, k0 cannot be used as writemask).

**cost_model.py** — Add cost entries for all new AVX512 intrinsics. Masked
variants have the same latency/throughput as unmasked on modern CPUs.

**_get_instruction_metadata()** — Add new AVX512 intrinsics to the shuffle_type
and control_vector_width mappings for correct comment formatting.

### Phase 2: i64 Gadgets (8 elements per vector)

**bitonic_super_optimizer.py** changes:

#### `_get_available_intrinsics()`

Add `AVX512 + i64` block registering:

```python
if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i64:
    # Unmasked single-input
    intrinsics["_mm512_permutexvar_epi64"] = z3_avx._mm512_permutexvar_epi64
    intrinsics["_mm512_permute_pd"] = z3_avx._mm512_permute_pd
    intrinsics["_mm512_permutevar_pd"] = z3_avx._mm512_permutevar_pd

    # Masked single-input
    intrinsics["_mm512_mask_permutexvar_epi64"] = z3_avx._mm512_mask_permutexvar_epi64
    intrinsics["_mm512_mask_permute_pd"] = z3_avx._mm512_mask_permute_pd
    intrinsics["_mm512_mask_permutevar_pd"] = z3_avx._mm512_mask_permutevar_pd

    # Unmasked dual-input
    intrinsics["_mm512_permutex2var_epi64"] = z3_avx._mm512_permutex2var_epi64
    intrinsics["_mm512_shuffle_pd"] = z3_avx._mm512_shuffle_pd
    intrinsics["_mm512_unpacklo_epi64"] = z3_avx._mm512_unpacklo_epi64
    intrinsics["_mm512_unpackhi_epi64"] = z3_avx._mm512_unpackhi_epi64
    intrinsics["_mm512_shuffle_i32x4"] = z3_avx._mm512_shuffle_i32x4
    intrinsics["_mm512_alignr_epi64"] = z3_avx._mm512_alignr_epi64

    # Masked dual-input
    intrinsics["_mm512_mask_permutex2var_epi64"] = z3_avx._mm512_mask_permutex2var_epi64
    intrinsics["_mm512_mask_shuffle_pd"] = z3_avx._mm512_mask_shuffle_pd
    intrinsics["_mm512_mask_unpacklo_epi64"] = z3_avx._mm512_mask_unpacklo_epi64
    intrinsics["_mm512_mask_unpackhi_epi64"] = z3_avx._mm512_mask_unpackhi_epi64
    intrinsics["_mm512_mask_alignr_epi64"] = z3_avx._mm512_mask_alignr_epi64
```

#### `_enumerate_single_input_instructions()`

Add `AVX512 + i64` block. For each unmasked instruction, also emit a masked
variant with a symbolic k-mask:

```python
# Unmasked
permutexvar = InstructionSpec("_mm512_permutexvar_epi64", {
    "a": input_reg,
    "op_idx": SymbolicPlaceholder(f"ctrl_permutexvar_{uid}", 512),
})

# Masked variant
mask_permutexvar = InstructionSpec("_mm512_mask_permutexvar_epi64", {
    "src": input_reg,  # merge source (current register value)
    "k": SymbolicPlaceholder(f"k_mask_permutexvar_{uid}", 8),  # 8 elements
    "a": input_reg,
    "op_idx": SymbolicPlaceholder(f"ctrl_mask_permutexvar_{uid}", 512),
})
```

Note: For masked instructions, `src` is the merge-mask source — when k[j]=0, the
element comes from `src`. Setting `src` to the current register means "keep the
original value", which is the standard merge-mask behavior.

#### `_enumerate_dual_input_instructions()`

Add `AVX512 + i64` block with both unmasked and masked variants.

For `_mm512_permutex2var_epi64`, this is a 3-operand instruction (`a`, `op_idx`,
`b`) which is different from the standard 2-operand pattern. The dispatch logic
in `_apply_instructions()` needs a new branch:

```python
elif "a" in args and "op_idx" in args and "b" in args:
    # Two-source variable permute (permutex2var)
    current_reg = intrinsic(args["a"], args["op_idx"], args["b"])
```

#### `_apply_instructions()` — New dispatch branches

Masked instructions have different argument patterns. Add branches for:

```python
# Masked single-input with control vector: (src, k, op_idx, a)
elif "src" in args and "k" in args and "a" in args and "op_idx" in args:
    current_reg = intrinsic(args["src"], args["k"], args["op_idx"], args["a"])

# Masked single-input with immediate: (src, k, a, imm8)
elif "src" in args and "k" in args and "a" in args and "imm8" in args and "b" not in args:
    current_reg = intrinsic(args["src"], args["k"], args["a"], args["imm8"])

# Masked dual-input with immediate: (src, k, a, b, imm8)
elif "src" in args and "k" in args and "a" in args and "b" in args and "imm8" in args:
    current_reg = intrinsic(args["src"], args["k"], args["a"], args["b"], args["imm8"])

# Masked dual-input without immediate: (src, k, a, b)
elif "src" in args and "k" in args and "a" in args and "b" in args:
    current_reg = intrinsic(args["src"], args["k"], args["a"], args["b"])

# Masked permutex2var: (a, k, op_idx, b)
elif "a" in args and "k" in args and "op_idx" in args and "b" in args:
    current_reg = intrinsic(args["a"], args["k"], args["op_idx"], args["b"])
```

#### `_extract_solution_from_model()`

The k-mask SymbolicPlaceholders will be concretized like immediates — the model
gives a concrete integer value for each k-mask. No changes needed beyond what
already handles SymbolicPlaceholder resolution, though bit-size handling for
8-bit and 16-bit masks may need a check.

#### `maybe_resolve_symbolic_vars()`

Currently handles sizes 8, 256, and 512. Needs to also handle size 16 (for
i32's 16-element k-mask). For i64, the 8-bit k-mask is already handled.

### Phase 3: i32 Gadgets (16 elements per vector)

Same pattern as Phase 2 but with `_epi32`/`_ps` variants. Key differences:
- 16 elements per vector (vs 8 for i64)
- k-mask is 16 bits (vs 8)
- Control vectors use 512-bit with 32-bit elements (vs 64-bit)
- `SymbolicPlaceholder` for k-mask uses size 16

### Z3 Semantics Gaps

One missing Z3 function:
- `_mm512_mask_shuffle_i32x4` — needs to be added to z3_avx.py

All other required functions already exist.

### Testing

For each phase:
1. Unit tests in `test_z3_avx.py` for any new Z3 semantics
2. Integration test in `test_super_vectorizer.py` enabling `vector_machine.AVX512`
   with `primitive_type.i64` / `primitive_type.i32`
3. ASM export test verifying k-mask syntax and correct mnemonics

### Risk: Search Space Size

With 16 elements (i32) and masked instruction variants, the template count will
be significantly larger than AVX2's. The synthesis may be slow. Mitigations:
- Start with `--gadget-depth 1` for initial validation
- The K-smallest output state algorithm bounds per-template work
- Can limit `--max-gadget-solutions` to reduce diversity if needed
