# Intrinsic Table Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract the three per-(vm, prim_type) if-chains from `gadget_synthesizer.py` into a `src/intrinsics/` package with one module per target.

**Architecture:** Create `src/intrinsics/` with four target modules (`avx2_i32`, `avx2_i64`, `avx512_i32`, `avx512_i64`). Each module exports three functions: `z3_functions()` (name→callable map), `single_input_nodes(input_ref)`, and `dual_input_nodes(ref1, ref2)`. An `__init__.py` provides a registry that replaces the if-chains. The three functions being replaced are `get_available_intrinsics` (lines 247-357), `_enumerate_single_input_intrinsics` (lines 1252-1432), and `_enumerate_dual_input_intrinsics` (lines 1434-1712) — ~570 lines total in `gadget_synthesizer.py`.

**Tech Stack:** Python, z3 (for z3_avx references), existing `bitonic_types` (InputRef, Symbolic, IntrinsicNode)

---

## Context for the implementor

### File locations (all paths relative to `vxsort/smallsort/codegen/`)

- `src/gadget_synthesizer.py` — 1770 lines; contains the three if-chains being extracted
- `src/bitonic_super_optimizer.py` — re-exports `get_available_intrinsics` from gadget_synthesizer
- `src/bitonic_verifier.py` — imports `get_available_intrinsics` via bitonic_super_optimizer
- `src/bitonic_types.py` — defines `InputRef`, `Symbolic`, `IntrinsicNode` (read-only for this task)
- `src/z3_avx.py` — defines the Z3 intrinsic functions (read-only for this task)
- `src/utils.py` — defines `vector_machine`, `primitive_type` enums (read-only for this task)
- `tests/test_symbolic_synthesis.py` — calls `_enumerate_*` methods directly
- `tests/test_super_vectorizer.py` — calls `_enumerate_*` methods directly (lines 142, 148, 673, 674)

### What each exported function provides

1. **`z3_functions()`** — Returns `dict[str, callable]` mapping intrinsic name strings to their Z3 semantic implementations in `z3_avx`. Currently the body of `get_available_intrinsics()`.

2. **`single_input_nodes(input_ref: InputRef)`** — Returns `list[IntrinsicNode]` of graph-node templates for single-input permutations. Each node has a unique `Symbolic` immediate/control named with a tag derived from `input_ref.name`.

3. **`dual_input_nodes(ref1: InputRef, ref2: InputRef)`** — Same idea but for two-input shuffle/blend/unpack instructions.

### Import pattern

This codebase uses try/except import blocks to support both relative (`from .foo`) and bare (`import foo`) imports. Every new module must follow this convention.

---

### Task 1: Create the `src/intrinsics/` package with `__init__.py`

**Files:**
- Create: `src/intrinsics/__init__.py`

**Step 1: Write the registry module**

```python
"""Per-target intrinsic table registry.

Provides ``get_z3_functions``, ``get_single_input_nodes``, and
``get_dual_input_nodes`` dispatched by (vector_machine, primitive_type).
"""

from __future__ import annotations

try:
    from ..utils import vector_machine, primitive_type
    from ..bitonic_types import InputRef, IntrinsicNode
    from . import avx2_i32, avx2_i64, avx512_i32, avx512_i64
except ImportError:
    from utils import vector_machine, primitive_type  # type: ignore
    from bitonic_types import InputRef, IntrinsicNode  # type: ignore
    import intrinsics.avx2_i32 as avx2_i32  # type: ignore
    import intrinsics.avx2_i64 as avx2_i64  # type: ignore
    import intrinsics.avx512_i32 as avx512_i32  # type: ignore
    import intrinsics.avx512_i64 as avx512_i64  # type: ignore

_MODULES = {
    (vector_machine.AVX2, primitive_type.i32): avx2_i32,
    (vector_machine.AVX2, primitive_type.i64): avx2_i64,
    (vector_machine.AVX512, primitive_type.i32): avx512_i32,
    (vector_machine.AVX512, primitive_type.i64): avx512_i64,
}


def _get_module(vm: vector_machine, prim_type: primitive_type):
    key = (vm, prim_type)
    if key not in _MODULES:
        raise NotImplementedError(
            f"Intrinsics not implemented for {vm} and {prim_type}"
        )
    return _MODULES[key]


def get_z3_functions(
    vm: vector_machine, prim_type: primitive_type
) -> dict[str, callable]:
    return _get_module(vm, prim_type).z3_functions()


def get_single_input_nodes(
    vm: vector_machine, prim_type: primitive_type, input_ref: InputRef
) -> list[IntrinsicNode]:
    return _get_module(vm, prim_type).single_input_nodes(input_ref)


def get_dual_input_nodes(
    vm: vector_machine,
    prim_type: primitive_type,
    ref1: InputRef,
    ref2: InputRef,
) -> list[IntrinsicNode]:
    return _get_module(vm, prim_type).dual_input_nodes(ref1, ref2)
```

**Step 2: Run tests to verify it imports (will fail — modules don't exist yet)**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run python -c "from src.intrinsics import get_z3_functions"`
Expected: ImportError (avx2_i32 doesn't exist yet) — confirms the `__init__.py` is at least parse-valid.

---

### Task 2: Create `src/intrinsics/avx2_i32.py`

**Files:**
- Create: `src/intrinsics/avx2_i32.py`

**Step 1: Write the module**

Copy the AVX2+i32 blocks from all three if-chains in `gadget_synthesizer.py` into three functions. The content comes from:
- `get_available_intrinsics`, lines 254-274 (the `AVX2 and i32` block)
- `_enumerate_single_input_intrinsics`, lines 1258-1288 (the `AVX2 and i32` block)
- `_enumerate_dual_input_intrinsics`, lines 1440-1476 (the `AVX2 and i32` block)

```python
"""AVX2 i32 intrinsic tables."""

from __future__ import annotations

try:
    from .. import z3_avx
    from ..bitonic_types import InputRef, Symbolic, IntrinsicNode
except ImportError:
    import z3_avx  # type: ignore
    from bitonic_types import InputRef, Symbolic, IntrinsicNode  # type: ignore


def z3_functions() -> dict[str, callable]:
    """Map intrinsic names to Z3 semantic implementations for AVX2 i32."""
    return {
        "_mm256_permute4x64_epi64": z3_avx._mm256_permute4x64_epi64,
        "_mm256_permute_ps": z3_avx._mm256_permute_ps,
        "_mm256_permutexvar_epi32": z3_avx._mm256_permutexvar_epi32,
        "_mm256_permutevar_ps": z3_avx._mm256_permutevar_ps,
        "_mm256_shuffle_ps": z3_avx._mm256_shuffle_ps,
        "_mm256_unpacklo_epi32": z3_avx._mm256_unpacklo_epi32,
        "_mm256_unpackhi_epi32": z3_avx._mm256_unpackhi_epi32,
        "_mm256_permute2x128_si256": z3_avx._mm256_permute2x128_si256,
        "_mm256_blend_ps": z3_avx._mm256_blend_ps,
        "_mm256_blendv_ps": z3_avx._mm256_blendv_ps,
        "_mm256_alignr_epi32": z3_avx._mm256_alignr_epi32,
    }


def single_input_nodes(input_ref: InputRef) -> list[IntrinsicNode]:
    """Single-input IntrinsicNode templates for AVX2 i32."""
    tag = input_ref.name
    return [
        IntrinsicNode(
            "_mm256_permute_ps",
            {
                "a": input_ref,
                "imm8": Symbolic(f"imm8_permute_ps_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_permute4x64_epi64",
            {
                "a": input_ref,
                "imm8": Symbolic(f"imm8_permute4x64_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_permutexvar_epi32",
            {
                "a": input_ref,
                "op_idx": Symbolic(f"ctrl_permutexvar_{tag}", 256),
            },
        ),
        IntrinsicNode(
            "_mm256_permutevar_ps",
            {
                "a": input_ref,
                "b": Symbolic(f"ctrl_permutevar_ps_{tag}", 256),
            },
        ),
    ]


def dual_input_nodes(ref1: InputRef, ref2: InputRef) -> list[IntrinsicNode]:
    """Dual-input IntrinsicNode templates for AVX2 i32."""
    tag = f"{ref1.name}_{ref2.name}"
    return [
        IntrinsicNode(
            "_mm256_shuffle_ps",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_shuffle_{tag}", 8),
            },
        ),
        IntrinsicNode("_mm256_unpacklo_epi32", {"a": ref1, "b": ref2}),
        IntrinsicNode("_mm256_unpackhi_epi32", {"a": ref1, "b": ref2}),
        IntrinsicNode(
            "_mm256_permute2x128_si256",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_perm2x128_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_blend_ps",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_blend_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_alignr_epi32",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_alignr_{tag}", 8),
            },
        ),
    ]
```

**Step 2: Smoke-test the module imports**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run python -c "from src.intrinsics.avx2_i32 import z3_functions, single_input_nodes, dual_input_nodes; print('OK')"`
Expected: `OK`

---

### Task 3: Create `src/intrinsics/avx2_i64.py`

**Files:**
- Create: `src/intrinsics/avx2_i64.py`

**Step 1: Write the module**

Same pattern. Content comes from:
- `get_available_intrinsics`, lines 277-297 (the `AVX2 and i64` block)
- `_enumerate_single_input_intrinsics`, lines 1290-1320 (the `AVX2 and i64` block)
- `_enumerate_dual_input_intrinsics`, lines 1478-1514 (the `AVX2 and i64` block)

```python
"""AVX2 i64 intrinsic tables."""

from __future__ import annotations

try:
    from .. import z3_avx
    from ..bitonic_types import InputRef, Symbolic, IntrinsicNode
except ImportError:
    import z3_avx  # type: ignore
    from bitonic_types import InputRef, Symbolic, IntrinsicNode  # type: ignore


def z3_functions() -> dict[str, callable]:
    """Map intrinsic names to Z3 semantic implementations for AVX2 i64."""
    return {
        "_mm256_permute4x64_epi64": z3_avx._mm256_permute4x64_epi64,
        "_mm256_permute_pd": z3_avx._mm256_permute_pd,
        "_mm256_permutexvar_epi64": z3_avx._mm256_permutexvar_epi64,
        "_mm256_permutevar_pd": z3_avx._mm256_permutevar_pd,
        "_mm256_shuffle_pd": z3_avx._mm256_shuffle_pd,
        "_mm256_unpacklo_epi64": z3_avx._mm256_unpacklo_epi64,
        "_mm256_unpackhi_epi64": z3_avx._mm256_unpackhi_epi64,
        "_mm256_permute2x128_si256": z3_avx._mm256_permute2x128_si256,
        "_mm256_blend_pd": z3_avx._mm256_blend_pd,
        "_mm256_blendv_pd": z3_avx._mm256_blendv_pd,
        "_mm256_alignr_epi64": z3_avx._mm256_alignr_epi64,
    }


def single_input_nodes(input_ref: InputRef) -> list[IntrinsicNode]:
    """Single-input IntrinsicNode templates for AVX2 i64."""
    tag = input_ref.name
    return [
        IntrinsicNode(
            "_mm256_permute_pd",
            {
                "a": input_ref,
                "imm8": Symbolic(f"imm8_permute_pd_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_permute4x64_epi64",
            {
                "a": input_ref,
                "imm8": Symbolic(f"imm8_permute4x64_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_permutexvar_epi64",
            {
                "a": input_ref,
                "op_idx": Symbolic(f"ctrl_permutexvar_{tag}", 256),
            },
        ),
        IntrinsicNode(
            "_mm256_permutevar_pd",
            {
                "a": input_ref,
                "b": Symbolic(f"ctrl_permutevar_pd_{tag}", 256),
            },
        ),
    ]


def dual_input_nodes(ref1: InputRef, ref2: InputRef) -> list[IntrinsicNode]:
    """Dual-input IntrinsicNode templates for AVX2 i64."""
    tag = f"{ref1.name}_{ref2.name}"
    return [
        IntrinsicNode(
            "_mm256_shuffle_pd",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_shuffle_{tag}", 8),
            },
        ),
        IntrinsicNode("_mm256_unpacklo_epi64", {"a": ref1, "b": ref2}),
        IntrinsicNode("_mm256_unpackhi_epi64", {"a": ref1, "b": ref2}),
        IntrinsicNode(
            "_mm256_permute2x128_si256",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_perm2x128_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_blend_pd",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_blend_{tag}", 8),
            },
        ),
        IntrinsicNode(
            "_mm256_alignr_epi64",
            {
                "a": ref1,
                "b": ref2,
                "imm8": Symbolic(f"imm8_alignr_{tag}", 8),
            },
        ),
    ]
```

**Step 2: Smoke-test**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run python -c "from src.intrinsics.avx2_i64 import z3_functions; print('OK')"`
Expected: `OK`

---

### Task 4: Create `src/intrinsics/avx512_i64.py`

**Files:**
- Create: `src/intrinsics/avx512_i64.py`

**Step 1: Write the module**

Content comes from:
- `get_available_intrinsics`, lines 300-326 (the `AVX512 and i64` block)
- `_enumerate_single_input_intrinsics`, lines 1322-1374 (the `AVX512 and i64` block)
- `_enumerate_dual_input_intrinsics`, lines 1516-1611 (the `AVX512 and i64` block)

This is the largest module due to masked variants. Follow the exact same structure as the AVX2 modules but with the AVX512 i64 intrinsics. Copy the intrinsic lists verbatim from `gadget_synthesizer.py`.

**Step 2: Smoke-test**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run python -c "from src.intrinsics.avx512_i64 import z3_functions; print('OK')"`
Expected: `OK`

---

### Task 5: Create `src/intrinsics/avx512_i32.py`

**Files:**
- Create: `src/intrinsics/avx512_i32.py`

**Step 1: Write the module**

Content comes from:
- `get_available_intrinsics`, lines 329-355 (the `AVX512 and i32` block)
- `_enumerate_single_input_intrinsics`, lines 1376-1428 (the `AVX512 and i32` block)
- `_enumerate_dual_input_intrinsics`, lines 1613-1708 (the `AVX512 and i32` block)

Follow the exact same structure. Copy intrinsic lists verbatim.

**Step 2: Smoke-test**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run python -c "from src.intrinsics.avx512_i32 import z3_functions; print('OK')"`
Expected: `OK`

---

### Task 6: Write tests for the new registry

**Files:**
- Create: `tests/test_intrinsic_tables.py`

**Step 1: Write the test file**

```python
"""Tests for the src/intrinsics/ registry and per-target modules."""

import pytest

from utils import vector_machine, primitive_type
from bitonic_types import InputRef, IntrinsicNode, Symbolic
from intrinsics import get_z3_functions, get_single_input_nodes, get_dual_input_nodes


# All supported (vm, prim_type) combinations
ALL_TARGETS = [
    (vector_machine.AVX2, primitive_type.i32),
    (vector_machine.AVX2, primitive_type.i64),
    (vector_machine.AVX512, primitive_type.i32),
    (vector_machine.AVX512, primitive_type.i64),
]


@pytest.mark.parametrize("vm,pt", ALL_TARGETS)
def test_z3_functions_returns_nonempty_dict_of_callables(vm, pt):
    funcs = get_z3_functions(vm, pt)
    assert isinstance(funcs, dict)
    assert len(funcs) > 0
    for name, fn in funcs.items():
        assert isinstance(name, str)
        assert callable(fn), f"{name} is not callable"


@pytest.mark.parametrize("vm,pt", ALL_TARGETS)
def test_single_input_nodes_returns_intrinsic_nodes(vm, pt):
    ref = InputRef("top")
    nodes = get_single_input_nodes(vm, pt, ref)
    assert isinstance(nodes, list)
    assert len(nodes) > 0
    for node in nodes:
        assert isinstance(node, IntrinsicNode)
        assert isinstance(node.name, str)
        # Every intrinsic in the node list must have a z3 implementation
        funcs = get_z3_functions(vm, pt)
        assert node.name in funcs, f"{node.name} not in z3_functions for {vm}/{pt}"


@pytest.mark.parametrize("vm,pt", ALL_TARGETS)
def test_dual_input_nodes_returns_intrinsic_nodes(vm, pt):
    ref1 = InputRef("top")
    ref2 = InputRef("bottom")
    nodes = get_dual_input_nodes(vm, pt, ref1, ref2)
    assert isinstance(nodes, list)
    assert len(nodes) > 0
    for node in nodes:
        assert isinstance(node, IntrinsicNode)
        funcs = get_z3_functions(vm, pt)
        assert node.name in funcs, f"{node.name} not in z3_functions for {vm}/{pt}"


@pytest.mark.parametrize("vm,pt", ALL_TARGETS)
def test_symbolic_names_include_tag(vm, pt):
    """Symbolic immediates must include the input ref name for uniqueness."""
    ref = InputRef("mytag")
    nodes = get_single_input_nodes(vm, pt, ref)
    for node in nodes:
        for val in node.operands.values():
            if isinstance(val, Symbolic):
                assert "mytag" in val.name, (
                    f"Symbolic {val.name} in {node.name} doesn't include tag"
                )


def test_unsupported_target_raises():
    with pytest.raises(NotImplementedError):
        get_z3_functions(vector_machine.AVX2, primitive_type.i16)


# Exact counts to catch regressions (match current if-chain outputs)
EXPECTED_COUNTS = {
    (vector_machine.AVX2, primitive_type.i32): (4, 6),
    (vector_machine.AVX2, primitive_type.i64): (4, 6),
    (vector_machine.AVX512, primitive_type.i32): (6, 12),
    (vector_machine.AVX512, primitive_type.i64): (6, 12),
}


@pytest.mark.parametrize("vm,pt", ALL_TARGETS)
def test_node_counts_match_expected(vm, pt):
    """Guard against accidentally adding/removing intrinsic entries."""
    single = get_single_input_nodes(vm, pt, InputRef("top"))
    dual = get_dual_input_nodes(vm, pt, InputRef("top"), InputRef("bottom"))
    expected_single, expected_dual = EXPECTED_COUNTS[(vm, pt)]
    assert len(single) == expected_single, (
        f"{vm}/{pt}: expected {expected_single} single, got {len(single)}"
    )
    assert len(dual) == expected_dual, (
        f"{vm}/{pt}: expected {expected_dual} dual, got {len(dual)}"
    )
```

**Step 2: Run the new tests**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run pytest tests/test_intrinsic_tables.py -v`
Expected: All tests pass.

---

### Task 7: Wire `gadget_synthesizer.py` to use the registry

**Files:**
- Modify: `src/gadget_synthesizer.py`

**Step 1: Add import of the new registry**

In the `try/except` import block at the top of `gadget_synthesizer.py`, add:

```python
# In the try block:
    from .intrinsics import get_z3_functions, get_single_input_nodes, get_dual_input_nodes

# In the except block:
    from intrinsics import get_z3_functions, get_single_input_nodes, get_dual_input_nodes  # type: ignore
```

**Step 2: Replace `get_available_intrinsics` body**

Replace the entire function body (lines 247-357) with:

```python
def get_available_intrinsics(
    vm: vector_machine, prim_type: primitive_type
) -> dict[str, callable]:
    """Get available intrinsics for the given VM and primitive type."""
    return get_z3_functions(vm, prim_type)
```

**Step 3: Replace `_enumerate_single_input_intrinsics` body**

Replace the method body (lines 1252-1432) with:

```python
    def _enumerate_single_input_intrinsics(
        self, input_ref: InputRef
    ) -> list[IntrinsicNode]:
        """Generate single-input IntrinsicNode templates with Symbolic immediates."""
        return get_single_input_nodes(self.vm, self.prim_type, input_ref)
```

**Step 4: Replace `_enumerate_dual_input_intrinsics` body**

Replace the method body (lines 1434-1712) with:

```python
    def _enumerate_dual_input_intrinsics(
        self, ref1: InputRef, ref2: InputRef
    ) -> list[IntrinsicNode]:
        """Generate dual-input IntrinsicNode templates with Symbolic immediates."""
        return get_dual_input_nodes(self.vm, self.prim_type, ref1, ref2)
```

**Step 5: Run the full test suite**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run pytest -v`
Expected: All existing tests still pass.

---

### Task 8: Run quick integration test

**Step 1: Run the compiler on AVX2 i64**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run python src/bitonic_compiler.py --vector-machine AVX2 --datatype i64 --depth-limit 3 --gadget-depth 1 --top-k 5`
Expected: Synthesis completes successfully with no errors.

---

### Task 9: Clean up dead code and lint

**Files:**
- Modify: `src/gadget_synthesizer.py` (remove the old mermaid comment referencing old method names if stale)

**Step 1: Run vulture**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run vulture`
Inspect output for any newly-dead code introduced by this refactor.

**Step 2: Run ruff**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run ruff check .`
Expected: Clean.

**Step 3: Run full test suite one final time**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run pytest -v`
Expected: All pass, wall clock under 1 minute.
