# AVX512 Gadget Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add AVX512 (512-bit ZMM) gadget synthesis to the bitonic sort super-optimizer, supporting i64 (8 elements/vec) and i32 (16 elements/vec) with all merge-masked instruction variants.

**Architecture:** Three sequential phases. Phase 1 adds the ASM encoding / cost model / metadata layer that both data types need. Phase 2 wires up i64 instruction templates + dispatch + tests. Phase 3 does the same for i32. Phases 2 and 3 can be dispatched as parallel sub-agents since they touch separate code blocks (different `if` branches keyed on `primitive_type`).

**Tech Stack:** Python 3, Z3 SMT solver, pytest. All changes are in `vxsort/smallsort/codegen/src/` and `vxsort/smallsort/codegen/tests/`.

**Design doc:** `docs/plans/2026-02-14-avx512-gadget-support-design.md`

---

## Task 1: ASM Mnemonic Mappings

Add all missing AVX512 intrinsic-to-mnemonic mappings so the ASM exporter produces correct x86-64 output.

**Files:**
- Modify: `src/asm_exporter.py:68-115` (the `mapping` dict inside `_intrinsic_to_asm_mnemonic`)

**Step 1: Add the missing mappings**

In `_intrinsic_to_asm_mnemonic()`, add these entries to the `mapping` dict (after the existing AVX512 entries):

```python
# AVX512 permute operations (masked)
"_mm512_mask_permutexvar_epi32": "vpermd",
"_mm512_mask_permutexvar_epi64": "vpermq",
# AVX512 two-source permute
"_mm512_permutex2var_epi32": "vpermi2d",
"_mm512_permutex2var_epi64": "vpermi2q",
"_mm512_mask_permutex2var_epi32": "vpermi2d",
"_mm512_mask_permutex2var_epi64": "vpermi2q",
# AVX512 in-lane permute (masked)
"_mm512_permute_pd": "vpermilpd",
"_mm512_mask_permute_ps": "vpermilps",
"_mm512_mask_permute_pd": "vpermilpd",
# AVX512 in-lane variable permute
"_mm512_permutevar_ps": "vpermilps",
"_mm512_permutevar_pd": "vpermilpd",
"_mm512_mask_permutevar_ps": "vpermilps",
"_mm512_mask_permutevar_pd": "vpermilpd",
# AVX512 shuffle (masked + pd)
"_mm512_shuffle_pd": "vshufpd",
"_mm512_mask_shuffle_ps": "vshufps",
"_mm512_mask_shuffle_pd": "vshufpd",
# AVX512 unpack (64-bit + all masked)
"_mm512_unpacklo_epi64": "vpunpcklqdq",
"_mm512_unpackhi_epi64": "vpunpckhqdq",
"_mm512_mask_unpacklo_epi32": "vpunpckldq",
"_mm512_mask_unpackhi_epi32": "vpunpckhdq",
"_mm512_mask_unpacklo_epi64": "vpunpcklqdq",
"_mm512_mask_unpackhi_epi64": "vpunpckhqdq",
# AVX512 128-bit lane shuffle
"_mm512_shuffle_i32x4": "vshufi32x4",
"_mm512_mask_shuffle_i32x4": "vshufi32x4",
# AVX512 align (masked)
"_mm512_mask_alignr_epi32": "valignd",
"_mm512_mask_alignr_epi64": "valignq",
```

**Step 2: Run tests to verify no regressions**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/ -x -q`
Expected: All existing tests pass.

**Step 3: Commit**

```bash
git add src/asm_exporter.py
git commit -m "feat: add AVX512 ASM mnemonic mappings for all masked/unmasked instructions"
```

---

## Task 2: Instruction Metadata for Comment Formatting

Update `_get_instruction_metadata()` so shuffle and control-vector comments format correctly for AVX512 instructions.

**Files:**
- Modify: `src/asm_exporter.py:8-62` (`_get_instruction_metadata`)

**Step 1: Add AVX512 entries to the metadata sets**

In `_get_instruction_metadata()`:

Add to `shuffle2_instructions`:
```python
"_mm512_shuffle_pd",
"_mm512_mask_shuffle_pd",
"_mm512_permute_pd",
"_mm512_mask_permute_pd",
```

Add to `shuffle4_instructions`:
```python
"_mm512_shuffle_ps",      # already present check
"_mm512_mask_shuffle_ps",
"_mm512_permute_ps",      # already present check
"_mm512_mask_permute_ps",
"_mm512_shuffle_i32x4",
"_mm512_mask_shuffle_i32x4",
```

Add to `control_vector_instructions`:
```python
# 32-bit control vectors (masked)
"_mm512_mask_permutexvar_epi32": 32,
"_mm512_mask_permutevar_ps": 32,
"_mm512_mask_permutex2var_epi32": 32,
# 64-bit control vectors (masked)
"_mm512_mask_permutexvar_epi64": 64,
"_mm512_mask_permutevar_pd": 64,
"_mm512_mask_permutex2var_epi64": 64,
```

**Step 2: Run tests**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/ -x -q`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add src/asm_exporter.py
git commit -m "feat: add AVX512 instruction metadata for shuffle/control-vector comments"
```

---

## Task 3: K-Mask Assembly Syntax in RegisterAllocator + _format_instruction

AVX512 masked instructions need `{k1}` notation on the destination register and a k-mask register allocated.

**Files:**
- Modify: `src/asm_exporter.py:120-154` (`RegisterAllocator`)
- Modify: `src/asm_exporter.py:193-318` (`_format_instruction`)

**Step 1: Add k-mask register allocation to RegisterAllocator**

Add a method and state for k-mask registers:

```python
class RegisterAllocator:
    def __init__(self, vm, dtype, num_vecs):
        # ... existing code ...
        self.next_kmask_reg = 1  # k1-k7 (k0 cannot be writemask)
        self.kmask_regs_allocated = []

    def allocate_kmask(self) -> str:
        """Allocate a k-mask register (k1-k7)."""
        if self.next_kmask_reg > 7:
            raise RuntimeError("Exhausted k-mask registers (k1-k7)")
        reg = f"k{self.next_kmask_reg}"
        self.next_kmask_reg += 1
        self.kmask_regs_allocated.append(reg)
        return reg

    def reset_temps(self):
        """Reset temporary register allocation."""
        self.next_temp_reg = self.num_vecs
        self.temp_regs_allocated = []
        self.next_kmask_reg = 1
        self.kmask_regs_allocated = []
```

**Step 2: Add a helper to detect masked intrinsics**

```python
def _is_masked_intrinsic(intrinsic_name: str) -> bool:
    """Check if an intrinsic is a masked (merge-mask) variant."""
    return "_mask_" in intrinsic_name
```

**Step 3: Update _format_instruction to handle masked instructions**

In `_format_instruction()`, after building the `dest_reg`, detect masked intrinsics:

1. If `_is_masked_intrinsic(inst.intrinsic_name)` and `"k"` is in `args`:
   - Allocate a k-mask register
   - Append `{kN}` to the destination register string
   - The `k` arg value is the concrete mask integer — add it as a comment
2. If `"src"` is in `args`:
   - Resolve it to a physical register (merge source)
   - For masked instructions, `src` determines what unmasked lanes retain

The masked instruction dispatch in `_format_instruction` needs new branches to handle `"src"`, `"k"`, `"a"`, `"b"`, `"op_idx"`, `"imm8"` argument combinations. These mirror the dispatch patterns described in the design doc.

**Step 4: Run tests**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/ -x -q`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add src/asm_exporter.py
git commit -m "feat: add k-mask register allocation and masked instruction formatting"
```

---

## Task 4: Cost Model Entries for AVX512 Instructions

Add cost entries for all new AVX512 intrinsics.

**Files:**
- Modify: `src/cost_model.py:41-85` (`_init_generic_costs`)

**Step 1: Add masked and missing unmasked AVX512 costs**

In `_init_generic_costs()`, add to the AVX512 section:

```python
# AVX512 masked permute variants (same cost as unmasked)
"_mm512_mask_permutexvar_epi32": InstructionCost(3.0, 1.0, ["p5"]),
"_mm512_mask_permutexvar_epi64": InstructionCost(3.0, 1.0, ["p5"]),
# AVX512 two-source permute
"_mm512_permutex2var_epi64": InstructionCost(3.0, 1.0, ["p5"]),
"_mm512_mask_permutex2var_epi64": InstructionCost(3.0, 1.0, ["p5"]),
# AVX512 in-lane permute
"_mm512_permute_pd": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_mask_permute_ps": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_mask_permute_pd": InstructionCost(1.0, 1.0, ["p5"]),
# AVX512 in-lane variable permute
"_mm512_permutevar_ps": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_permutevar_pd": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_mask_permutevar_ps": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_mask_permutevar_pd": InstructionCost(1.0, 1.0, ["p5"]),
# AVX512 shuffle
"_mm512_shuffle_pd": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_mask_shuffle_ps": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_mask_shuffle_pd": InstructionCost(1.0, 1.0, ["p5"]),
# AVX512 unpack
"_mm512_unpacklo_epi64": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_unpackhi_epi64": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_mask_unpacklo_epi32": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_mask_unpackhi_epi32": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_mask_unpacklo_epi64": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_mask_unpackhi_epi64": InstructionCost(1.0, 1.0, ["p5"]),
# AVX512 128-bit lane shuffle
"_mm512_mask_shuffle_i32x4": InstructionCost(3.0, 1.0, ["p5"]),
# AVX512 align (masked)
"_mm512_mask_alignr_epi32": InstructionCost(1.0, 1.0, ["p5"]),
"_mm512_mask_alignr_epi64": InstructionCost(1.0, 1.0, ["p5"]),
# AVX512 alignr (unmasked, missing)
"_mm512_alignr_epi64": InstructionCost(1.0, 1.0, ["p5"]),
```

**Step 2: Run tests**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/ -x -q`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add src/cost_model.py
git commit -m "feat: add cost model entries for AVX512 masked/unmasked instructions"
```

---

## Task 5: Add _mm512_mask_shuffle_i32x4 to z3_avx.py

This is the one missing Z3 semantic function needed for AVX512.

**Files:**
- Modify: `src/z3_avx.py` (after `_mm512_shuffle_i32x4` at ~line 1555)
- Modify: `tests/test_z3_avx.py` (add test)

**Step 1: Write the failing test**

In `tests/test_z3_avx.py`, add import and test:

```python
from z3_avx import _mm512_mask_shuffle_i32x4

class TestMaskShuffleI32x4:
    def test_mm512_mask_shuffle_i32x4_identity_all_masked(self):
        """With all-ones mask, behaves like unmasked shuffle_i32x4."""
        ctx = main_ctx()
        s = Solver(ctx=ctx)
        a = zmm_reg_with_32b_values("a", s, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], ctx=ctx)
        b = zmm_reg_with_32b_values("b", s, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], ctx=ctx)
        src = zmm_reg_with_32b_values("src", s, [99]*16, ctx=ctx)
        k = BitVecVal(0xFFFF, 16)  # all ones
        imm8 = null_shuffle_i32x4_imm8  # 0xE4 = identity
        result = _mm512_mask_shuffle_i32x4(src, k, a, b, imm8)
        # With identity shuffle and all-ones mask, output should be [a_lane0, a_lane1, b_lane2, b_lane3]
        for i in range(16):
            lane_val = Extract(32*(i+1)-1, 32*i, result)
            s.add(lane_val == BitVecVal(i if i < 8 else i+8, 32, ctx=ctx))
        assert s.check() == sat

    def test_mm512_mask_shuffle_i32x4_zero_mask_keeps_src(self):
        """With zero mask, all elements come from src."""
        ctx = main_ctx()
        s = Solver(ctx=ctx)
        a = zmm_reg_with_32b_values("a", s, list(range(16)), ctx=ctx)
        b = zmm_reg_with_32b_values("b", s, list(range(16, 32)), ctx=ctx)
        src = zmm_reg_with_32b_values("src", s, [99]*16, ctx=ctx)
        k = BitVecVal(0x0000, 16)  # all zeros
        result = _mm512_mask_shuffle_i32x4(src, k, a, b, 0xE4)
        # All elements should come from src = 99
        for i in range(16):
            lane_val = Extract(32*(i+1)-1, 32*i, result)
            s.add(lane_val == BitVecVal(99, 32, ctx=ctx))
        assert s.check() == sat
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/test_z3_avx.py::TestMaskShuffleI32x4 -v`
Expected: ImportError — `_mm512_mask_shuffle_i32x4` not found

**Step 3: Implement _mm512_mask_shuffle_i32x4**

In `src/z3_avx.py`, after the `_mm512_shuffle_i32x4` function (~line 1555), add:

```python
def _mm512_mask_shuffle_i32x4(
    src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int
):
    """
    Shuffle 128-bit lanes from a and b using imm8, with merge masking.
    Implements __m512i _mm512_mask_shuffle_i32x4(__m512i src, __mmask16 k, __m512i a, __m512i b, const int imm8)
    Elements are copied from src when the corresponding mask bit is not set.
    """
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    # First do the unmasked shuffle
    lanes = [None] * 4
    for j in range(4):
        source = a if j < 2 else b
        ctrl = Extract(2 * j + 1, 2 * j, imm)
        lanes[j] = _select4_4x32b(source, ctrl)

    unmasked = simplify(Concat(lanes[::-1]))

    # Apply merge mask
    num_elements = 16
    elements = [None] * num_elements
    for j in range(num_elements):
        i = j * 32
        mask_bit = Extract(j, j, k)
        tmp_elem = Extract(i + 31, i, unmasked)
        src_elem = Extract(i + 31, i, src)
        elements[j] = simplify(If(mask_bit == 1, tmp_elem, src_elem))

    return simplify(Concat(elements[::-1]))
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/test_z3_avx.py::TestMaskShuffleI32x4 -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/ -x -q`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/z3_avx.py tests/test_z3_avx.py
git commit -m "feat: add _mm512_mask_shuffle_i32x4 Z3 semantics with tests"
```

---

## Task 6: Super-Optimizer Dispatch for AVX512 Instruction Patterns

The `_apply_instructions()` method needs new dispatch branches for AVX512 argument patterns (permutex2var 3-operand, masked with src/k, etc.). This is shared infrastructure for both i64 and i32.

**Files:**
- Modify: `src/bitonic_super_optimizer.py:920-1018` (`_apply_instructions`)
- Modify: `src/bitonic_super_optimizer.py:368-394` (`maybe_resolve_symbolic_vars`)

**Step 1: Add size 16 to maybe_resolve_symbolic_vars**

In `maybe_resolve_symbolic_vars()` (line ~377), the existing code handles sizes 8, 256, 512. Add size 16 for i32 k-masks:

```python
if isinstance(value, SymbolicPlaceholder):
    if value.size == 8:
        actual_val = BitVec(value.name, 8, ctx=ctx)
    elif value.size == 16:
        actual_val = BitVec(value.name, 16, ctx=ctx)
    elif value.size == 256:
        actual_val = z3_avx.ymm_reg(value.name, ctx=ctx)
    elif value.size == 512:
        actual_val = z3_avx.zmm_reg(value.name, ctx=ctx)
    else:
        actual_val = BitVec(value.name, value.size, ctx=ctx)
```

**Step 2: Add new dispatch branches to _apply_instructions**

In `_apply_instructions()` (line ~990), add new branches BEFORE the existing ones (more specific patterns must match first). The existing dispatch cascade is:

```python
if "a" in args and "op_idx" in args:        # permutexvar
elif "a" in args and "imm8" in args ...:     # single + imm8
elif "a" in args and "b" in args and "imm8": # dual + imm8
elif "a" in args and "b" in args and "mask": # blend with mask
elif "a" in args and "b" in args:            # dual no imm
else:                                        # generic fallback
```

Insert these new branches AT THE TOP of the dispatch chain (before the existing `if`):

```python
# --- AVX512 masked dispatch (most specific first) ---

# Masked permutex2var: (a, k, op_idx, b)
if "a" in args and "k" in args and "op_idx" in args and "b" in args:
    current_reg = intrinsic(args["a"], args["k"], args["op_idx"], args["b"])

# Masked single-input with control vector: (src, k, op_idx, a)
elif "src" in args and "k" in args and "op_idx" in args and "a" in args:
    current_reg = intrinsic(args["src"], args["k"], args["op_idx"], args["a"])

# Masked dual-input with immediate: (src, k, a, b, imm8)
elif "src" in args and "k" in args and "a" in args and "b" in args and "imm8" in args:
    current_reg = intrinsic(args["src"], args["k"], args["a"], args["b"], args["imm8"])

# Masked single-input with immediate: (src, k, a, imm8)
elif "src" in args and "k" in args and "a" in args and "imm8" in args and "b" not in args:
    current_reg = intrinsic(args["src"], args["k"], args["a"], args["imm8"])

# Masked dual-input without immediate: (src, k, a, b) — e.g., mask_unpack
elif "src" in args and "k" in args and "a" in args and "b" in args:
    current_reg = intrinsic(args["src"], args["k"], args["a"], args["b"])

# Unmasked permutex2var: (a, op_idx, b) — no k
elif "a" in args and "op_idx" in args and "b" in args:
    current_reg = intrinsic(args["a"], args["op_idx"], args["b"])

# --- Existing AVX2 dispatch (unchanged) ---
elif "a" in args and "op_idx" in args:
    ...  # existing code
```

**Step 3: Run existing tests to verify no regressions**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/ -x -q`
Expected: All existing tests still pass (new branches don't affect AVX2 paths since no AVX2 instructions have `"src"`, `"k"`, or 3-operand `"op_idx"+"b"` patterns).

**Step 4: Commit**

```bash
git add src/bitonic_super_optimizer.py
git commit -m "feat: add AVX512 dispatch branches for masked and permutex2var instructions"
```

---

## Task 7: AVX512 i64 — Register Intrinsics + Enumerate Templates

Wire up all AVX512 i64 instructions in the super-optimizer.

**Files:**
- Modify: `src/bitonic_super_optimizer.py:228-278` (`_get_available_intrinsics`)
- Modify: `src/bitonic_super_optimizer.py:1194-1290` (`_enumerate_single_input_instructions`)
- Modify: `src/bitonic_super_optimizer.py:1292-1419` (`_enumerate_dual_input_instructions`)

**Step 1: Add AVX512 i64 block to _get_available_intrinsics**

After the `AVX2 + i64` block (line ~277), add:

```python
# For AVX512 i64, we need ZMM (512-bit) operations on 64-bit elements
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
    intrinsics["_mm512_mask_shuffle_i32x4"] = z3_avx._mm512_mask_shuffle_i32x4
    intrinsics["_mm512_mask_alignr_epi64"] = z3_avx._mm512_mask_alignr_epi64
```

**Step 2: Add AVX512 i64 block to _enumerate_single_input_instructions**

After the `AVX2 + i64` block (line ~1286), add:

```python
if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i64:
    input_reg = reg_name
    unique_id = id(input_reg)

    # --- Unmasked ---
    permutexvar = InstructionSpec(
        "_mm512_permutexvar_epi64",
        {"a": input_reg, "op_idx": SymbolicPlaceholder(f"ctrl_permutexvar_{unique_id}", 512)},
    )
    permute_pd = InstructionSpec(
        "_mm512_permute_pd",
        {"a": input_reg, "imm8": SymbolicPlaceholder(f"imm8_permute_pd_{unique_id}", 8)},
    )
    permutevar_pd = InstructionSpec(
        "_mm512_permutevar_pd",
        {"a": input_reg, "b": SymbolicPlaceholder(f"ctrl_permutevar_pd_{unique_id}", 512)},
    )

    # --- Masked ---
    mask_permutexvar = InstructionSpec(
        "_mm512_mask_permutexvar_epi64",
        {
            "src": input_reg,
            "k": SymbolicPlaceholder(f"k_mask_permutexvar_{unique_id}", 8),
            "op_idx": SymbolicPlaceholder(f"ctrl_m_permutexvar_{unique_id}", 512),
            "a": input_reg,
        },
    )
    mask_permute_pd = InstructionSpec(
        "_mm512_mask_permute_pd",
        {
            "src": input_reg,
            "k": SymbolicPlaceholder(f"k_mask_permute_pd_{unique_id}", 8),
            "a": input_reg,
            "imm8": SymbolicPlaceholder(f"imm8_m_permute_pd_{unique_id}", 8),
        },
    )
    mask_permutevar_pd = InstructionSpec(
        "_mm512_mask_permutevar_pd",
        {
            "src": input_reg,
            "k": SymbolicPlaceholder(f"k_mask_permutevar_pd_{unique_id}", 8),
            "a": input_reg,
            "b": SymbolicPlaceholder(f"ctrl_m_permutevar_pd_{unique_id}", 512),
        },
    )

    return [permutexvar, permute_pd, permutevar_pd,
            mask_permutexvar, mask_permute_pd, mask_permutevar_pd]
```

**Step 3: Add AVX512 i64 block to _enumerate_dual_input_instructions**

After the `AVX2 + i64` block (line ~1415), add:

```python
if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i64:
    reg1 = reg1_name
    reg2 = reg2_name
    unique_id = f"{id(reg1)}_{id(reg2)}"

    # --- Unmasked ---
    permutex2var = InstructionSpec(
        "_mm512_permutex2var_epi64",
        {"a": reg1, "op_idx": SymbolicPlaceholder(f"ctrl_permutex2var_{unique_id}", 512), "b": reg2},
    )
    shuffle_pd = InstructionSpec(
        "_mm512_shuffle_pd",
        {"a": reg1, "b": reg2, "imm8": SymbolicPlaceholder(f"imm8_shuffle_pd_{unique_id}", 8)},
    )
    unpacklo = InstructionSpec("_mm512_unpacklo_epi64", {"a": reg1, "b": reg2})
    unpackhi = InstructionSpec("_mm512_unpackhi_epi64", {"a": reg1, "b": reg2})
    shuffle_i32x4 = InstructionSpec(
        "_mm512_shuffle_i32x4",
        {"a": reg1, "b": reg2, "imm8": SymbolicPlaceholder(f"imm8_shuf_i32x4_{unique_id}", 8)},
    )
    alignr = InstructionSpec(
        "_mm512_alignr_epi64",
        {"a": reg1, "b": reg2, "imm8": SymbolicPlaceholder(f"imm8_alignr_{unique_id}", 8)},
    )

    # --- Masked ---
    mask_permutex2var = InstructionSpec(
        "_mm512_mask_permutex2var_epi64",
        {
            "a": reg1,
            "k": SymbolicPlaceholder(f"k_mask_permutex2var_{unique_id}", 8),
            "op_idx": SymbolicPlaceholder(f"ctrl_m_permutex2var_{unique_id}", 512),
            "b": reg2,
        },
    )
    mask_shuffle_pd = InstructionSpec(
        "_mm512_mask_shuffle_pd",
        {
            "src": reg1,
            "k": SymbolicPlaceholder(f"k_mask_shuffle_pd_{unique_id}", 8),
            "a": reg1,
            "b": reg2,
            "imm8": SymbolicPlaceholder(f"imm8_m_shuffle_pd_{unique_id}", 8),
        },
    )
    mask_unpacklo = InstructionSpec(
        "_mm512_mask_unpacklo_epi64",
        {
            "src": reg1,
            "k": SymbolicPlaceholder(f"k_mask_unpacklo_{unique_id}", 8),
            "a": reg1,
            "b": reg2,
        },
    )
    mask_unpackhi = InstructionSpec(
        "_mm512_mask_unpackhi_epi64",
        {
            "src": reg1,
            "k": SymbolicPlaceholder(f"k_mask_unpackhi_{unique_id}", 8),
            "a": reg1,
            "b": reg2,
        },
    )
    mask_shuffle_i32x4 = InstructionSpec(
        "_mm512_mask_shuffle_i32x4",
        {
            "src": reg1,
            "k": SymbolicPlaceholder(f"k_mask_shuf_i32x4_{unique_id}", 8),
            "a": reg1,
            "b": reg2,
            "imm8": SymbolicPlaceholder(f"imm8_m_shuf_i32x4_{unique_id}", 8),
        },
    )
    mask_alignr = InstructionSpec(
        "_mm512_mask_alignr_epi64",
        {
            "src": reg1,
            "k": SymbolicPlaceholder(f"k_mask_alignr_{unique_id}", 8),
            "a": reg1,
            "b": reg2,
            "imm8": SymbolicPlaceholder(f"imm8_m_alignr_{unique_id}", 8),
        },
    )

    return [permutex2var, shuffle_pd, unpacklo, unpackhi, shuffle_i32x4, alignr,
            mask_permutex2var, mask_shuffle_pd, mask_unpacklo, mask_unpackhi,
            mask_shuffle_i32x4, mask_alignr]
```

**Step 4: Run tests**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/ -x -q`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add src/bitonic_super_optimizer.py
git commit -m "feat: register AVX512 i64 intrinsics and enumerate instruction templates"
```

---

## Task 8: AVX512 i64 Integration Test

Add an integration test verifying end-to-end synthesis for AVX512 i64.

**Files:**
- Modify: `tests/test_super_vectorizer.py`

**Step 1: Enable AVX512 in the parametrized test**

Change line 174 from:
```python
@pytest.mark.parametrize("vm", [vector_machine.AVX2])
```
to:
```python
@pytest.mark.parametrize("vm", [vector_machine.AVX2, vector_machine.AVX512])
```

This enables `test_first_stage_requires_no_permutation` for AVX512 with both i32 and i64. Since i32 templates are not yet registered, restrict:

```python
@pytest.mark.parametrize("vm", [vector_machine.AVX2, vector_machine.AVX512])
@pytest.mark.parametrize("dt", [primitive_type.i32, primitive_type.i64])
```

But guard against unimplemented combinations:

```python
def test_first_stage_requires_no_permutation(vm, dt):
    if vm == vector_machine.AVX512 and dt == primitive_type.i32:
        pytest.skip("AVX512 i32 not yet implemented")
    ...
```

**Step 2: Add a dedicated AVX512 i64 synthesis test**

```python
def test_avx512_i64_synthesis_depth1():
    """Test that AVX512 i64 synthesis finds solutions for first 2 stages at depth 1."""
    super_opt = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX512)
    solutions, all_complete = super_opt.build_solution_tree(
        depth_limit=2, gadget_depth=2, natural_order=False, max_unique_outputs=1
    )
    assert len(solutions) > 0, "Should find at least one solution for AVX512 i64"
    print(f"Found {len(solutions)} root solutions for AVX512 i64")
```

**Step 3: Run the new test**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/test_super_vectorizer.py::test_avx512_i64_synthesis_depth1 -v -s`
Expected: PASS with at least 1 solution found.

**Step 4: Run full test suite (should be under 60s)**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/ -x -q`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add tests/test_super_vectorizer.py
git commit -m "feat: add AVX512 i64 integration tests for gadget synthesis"
```

---

## Task 9: AVX512 i32 — Register Intrinsics + Enumerate Templates

Wire up all AVX512 i32 instructions. Same pattern as Task 7 but with `_epi32`/`_ps` variants and 16-bit k-masks.

**Files:**
- Modify: `src/bitonic_super_optimizer.py:228-278` (`_get_available_intrinsics`)
- Modify: `src/bitonic_super_optimizer.py:1194-1290` (`_enumerate_single_input_instructions`)
- Modify: `src/bitonic_super_optimizer.py:1292-1419` (`_enumerate_dual_input_instructions`)

**Step 1: Add AVX512 i32 block to _get_available_intrinsics**

```python
# For AVX512 i32, we need ZMM (512-bit) operations on 32-bit elements
if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i32:
    # Unmasked single-input
    intrinsics["_mm512_permutexvar_epi32"] = z3_avx._mm512_permutexvar_epi32
    intrinsics["_mm512_permute_ps"] = z3_avx._mm512_permute_ps
    intrinsics["_mm512_permutevar_ps"] = z3_avx._mm512_permutevar_ps
    # Masked single-input
    intrinsics["_mm512_mask_permutexvar_epi32"] = z3_avx._mm512_mask_permutexvar_epi32
    intrinsics["_mm512_mask_permute_ps"] = z3_avx._mm512_mask_permute_ps
    intrinsics["_mm512_mask_permutevar_ps"] = z3_avx._mm512_mask_permutevar_ps
    # Unmasked dual-input
    intrinsics["_mm512_permutex2var_epi32"] = z3_avx._mm512_permutex2var_epi32
    intrinsics["_mm512_shuffle_ps"] = z3_avx._mm512_shuffle_ps
    intrinsics["_mm512_unpacklo_epi32"] = z3_avx._mm512_unpacklo_epi32
    intrinsics["_mm512_unpackhi_epi32"] = z3_avx._mm512_unpackhi_epi32
    intrinsics["_mm512_shuffle_i32x4"] = z3_avx._mm512_shuffle_i32x4
    intrinsics["_mm512_alignr_epi32"] = z3_avx._mm512_alignr_epi32
    # Masked dual-input
    intrinsics["_mm512_mask_permutex2var_epi32"] = z3_avx._mm512_mask_permutex2var_epi32
    intrinsics["_mm512_mask_shuffle_ps"] = z3_avx._mm512_mask_shuffle_ps
    intrinsics["_mm512_mask_unpacklo_epi32"] = z3_avx._mm512_mask_unpacklo_epi32
    intrinsics["_mm512_mask_unpackhi_epi32"] = z3_avx._mm512_mask_unpackhi_epi32
    intrinsics["_mm512_mask_shuffle_i32x4"] = z3_avx._mm512_mask_shuffle_i32x4
    intrinsics["_mm512_mask_alignr_epi32"] = z3_avx._mm512_mask_alignr_epi32
```

**Step 2: Add AVX512 i32 block to _enumerate_single_input_instructions**

Same structure as Task 7 Step 2 but:
- Use `_epi32`/`_ps` intrinsic names
- K-mask size is 16 (not 8): `SymbolicPlaceholder(f"k_mask_...", 16)`
- Control vectors are still 512-bit

**Step 3: Add AVX512 i32 block to _enumerate_dual_input_instructions**

Same structure as Task 7 Step 3 but with i32/ps variants and k-mask size 16.

**Step 4: Remove the pytest.skip guard from Task 8**

In `test_first_stage_requires_no_permutation`, remove the `pytest.skip("AVX512 i32 not yet implemented")` guard.

**Step 5: Run tests**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/ -x -q`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/bitonic_super_optimizer.py tests/test_super_vectorizer.py
git commit -m "feat: register AVX512 i32 intrinsics and enumerate instruction templates"
```

---

## Task 10: AVX512 i32 Integration Test

**Files:**
- Modify: `tests/test_super_vectorizer.py`

**Step 1: Add AVX512 i32 synthesis test**

```python
def test_avx512_i32_synthesis_depth1():
    """Test that AVX512 i32 synthesis finds solutions for first 2 stages at depth 1."""
    super_opt = BitonicSuperVectorizer(2, primitive_type.i32, vector_machine.AVX512)
    solutions, all_complete = super_opt.build_solution_tree(
        depth_limit=2, gadget_depth=2, natural_order=False, max_unique_outputs=1
    )
    assert len(solutions) > 0, "Should find at least one solution for AVX512 i32"
    print(f"Found {len(solutions)} root solutions for AVX512 i32")
```

**Step 2: Run the test**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/test_super_vectorizer.py::test_avx512_i32_synthesis_depth1 -v -s`
Expected: PASS (may be slow due to 16 elements).

**Step 3: Run full suite and verify under 60s wall clock**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/ -x -q`

If the AVX512 i32 test is too slow (>30s), reduce `depth_limit=1` or add `@pytest.mark.slow` and skip in CI.

**Step 4: Commit**

```bash
git add tests/test_super_vectorizer.py
git commit -m "feat: add AVX512 i32 integration test for gadget synthesis"
```

---

## Task 11: Final Validation — Lint, Dead Code, Full Suite

**Step 1: Run ruff**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run ruff check .`
Fix any issues.

**Step 2: Run vulture**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run vulture`
Inspect output, remove dead code introduced by this work.

**Step 3: Full test suite**

Run: `cd /Users/dmg/projects/vxsort-cpp2/vxsort/smallsort/codegen && uv run pytest tests/ -x -q`
Expected: All tests pass in under 60s.

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: fix lint and remove dead code from AVX512 implementation"
```

---

## Parallelism Notes

- **Tasks 1-6** are sequential prerequisites (ASM layer, cost model, dispatch infrastructure).
- **Tasks 7-8** (i64) and **Tasks 9-10** (i32) can run as **parallel sub-agents** since they modify separate `if` blocks keyed on `primitive_type`.
- **Task 11** runs after both parallel tracks complete.
