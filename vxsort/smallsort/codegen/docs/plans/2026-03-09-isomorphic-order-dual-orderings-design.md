# `isomorphic_order` Tag and Dual-Ordering Completeness

**Date**: 2026-03-09
**Status**: Design

## Problem

All four intrinsic tables call `get_dual_input_nodes(top_ref, bottom_ref)`, producing
templates with a fixed `a=top, b=bottom` operand ordering.  For instructions where
swapping `a` and `b` cannot be compensated by adjusting the control variable (imm8 /
control-vector / k-mask), the current search misses half the reachable output space.

Affected shapes:

| Shape | Description | Currently covers both orderings? |
|---|---|:---:|
| B | depth-1 dual instruction | **No** |
| D | depth-2 dual→single | **No** |
| E | depth-2 single→dual (mux) | Yes — mux selects `{top, bottom, prev}` per operand |
| F | shared-prefix (Blacher) | Partially — unconditional loop covers all, even symmetric ones |

## `isomorphic_order` Classification

`isomorphic_order = True` means swapping `a` and `b` can always be compensated by
adjusting the instruction's control variable.  Only one operand ordering is needed.

`isomorphic_order = False` means the swap produces a genuinely different output space
that cannot be recovered by changing the control variable.  Both orderings must be
searched.

### Per-instruction table

#### AVX2 i32

| Instruction | `isomorphic_order` | Reason |
|---|:---:|---|
| `_mm256_shuffle_ps` | False | positions 0,1 always from `a`; 2,3 always from `b` |
| `_mm256_unpacklo_epi32` | False | `[a0,b0,a1,b1,…]` vs `[b0,a0,…]`; no imm8 |
| `_mm256_unpackhi_epi32` | False | same |
| `_mm256_permute2x128_si256` | True | imm8 explicitly selects from `{a[0],a[1],b[0],b[1]}`; remap 0↔2, 1↔3 |
| `_mm256_blend_ps` | True | `blend(a,b,m) = blend(b,a,~m)` |
| `_mm256_alignr_epi32` | False | concatenates `[b\|a]` then shifts; swap changes concat order |

#### AVX2 i64

| Instruction | `isomorphic_order` | Reason |
|---|:---:|---|
| `_mm256_shuffle_pd` | False | even output lanes always from `a`, odd always from `b` |
| `_mm256_unpacklo_epi64` | False | no imm8 |
| `_mm256_unpackhi_epi64` | False | no imm8 |
| `_mm256_permute2x128_si256` | True | same as i32 |
| `_mm256_blend_pd` | True | same as `blend_ps` |
| `_mm256_alignr_epi64` | False | same as `alignr_epi32` |

#### AVX512 i32

| Instruction | `isomorphic_order` | Reason |
|---|:---:|---|
| `_mm512_permutex2var_epi32` | True | source-select bit in `op_idx` per element; flip all bits |
| `_mm512_shuffle_ps` | False | same structure as `_mm256_shuffle_ps` |
| `_mm512_unpacklo_epi32` | False | no imm8 |
| `_mm512_unpackhi_epi32` | False | no imm8 |
| `_mm512_shuffle_i32x4` | True | imm8 selects 128-bit chunks from `{a[0],a[1],b[0],b[1]}`; remap 0↔2, 1↔3 |
| `_mm512_alignr_epi32` | False | same as AVX2 alignr |
| `_mm512_mask_permutex2var_epi32` | True | pass-through lanes re-expressible as permuted selections via full `op_idx` |
| `_mm512_mask_shuffle_ps` | False | shuffle asymmetry survives masking |
| `_mm512_mask_unpacklo_epi32` | False | interleaving structure survives masking |
| `_mm512_mask_unpackhi_epi32` | False | same |
| `_mm512_mask_shuffle_i32x4` | True | same reasoning as unmasked; pass-through re-expressible |
| `_mm512_mask_alignr_epi32` | False | concat structure survives masking |

#### AVX512 i64

| Instruction | `isomorphic_order` | Reason |
|---|:---:|---|
| `_mm512_permutex2var_epi64` | True | same as epi32 |
| `_mm512_shuffle_pd` | False | same as `_mm256_shuffle_pd` |
| `_mm512_unpacklo_epi64` | False | no imm8 |
| `_mm512_unpackhi_epi64` | False | no imm8 |
| `_mm512_shuffle_i32x4` | True | same as i32 table |
| `_mm512_alignr_epi64` | False | same reasoning as AVX2 |
| `_mm512_mask_permutex2var_epi64` | True | same as epi32 |
| `_mm512_mask_shuffle_pd` | False | shuffle asymmetry survives masking |
| `_mm512_mask_unpacklo_epi64` | False | same |
| `_mm512_mask_unpackhi_epi64` | False | same |
| `_mm512_mask_shuffle_i32x4` | True | same |
| `_mm512_mask_alignr_epi64` | False | same |

## Design

### 1. Tag representation

Add `isomorphic_order: bool = True` to `IntrinsicNode` in `bitonic_types.py`:

```python
@dataclass
class IntrinsicNode:
    name: str
    operands: dict
    isomorphic_order: bool = field(default=True, compare=False, hash=False)
```

`compare=False, hash=False` ensures the field does not affect equality checks or
hashing — `IntrinsicNode` identity semantics are unchanged.

Each `dual_input_nodes()` function in the four intrinsic table files sets
`isomorphic_order=False` for instructions marked above.

### 2. Per-shape behavior

**Shape B (depth-1 dual):** For each dual with `isomorphic_order=False`, append a
swapped-ref variant to the candidate list.  A new helper
`_swap_register_operands(node)` on `GadgetSynthesizer` exchanges the two `InputRef`
operands and copies the remaining operands unchanged.

**Shape D (depth-2 dual→single):** For each dual in `dual_intrinsics` with
`isomorphic_order=False`, iterate with both the original and swapped node as `inst0`.

**Shape E (depth-2 single→dual):** No change.  The mux on each register operand
of the dual instruction independently selects from `{top, bottom, prev_result}`,
covering both orderings implicitly.

**Shape F (shared-prefix):** Replace the unconditional `orderings` loop with
logic keyed on `isomorphic_order` and whether the dual has Symbolic operands:

| `isomorphic_order` | Has Symbolics | What to generate |
|:---:|:---:|---|
| True | Yes | one ordering; independent `_shared_top` / `_shared_bot` Symbolic names |
| False | Yes | two same-ordering variants `(A,B)` and `(B,A)` for both sides together |
| False | No | cross-ordering: `top=tail(A,B)`, `bottom=tail(B,A)` — only way to get distinct outputs without a CV |
| True | No | skip (not present in current tables; top ≡ bottom always) |

The cross-ordering case is what correctly includes `unpacklo` / `unpackhi` in Shape F,
fixing the over-aggressive `parametric_duals` filter.  After CSE, the two tail nodes
have different arg signatures (`a=result_0,b=prev` vs `a=prev,b=result_0`), so
`instruction_count()` = 4 and the existing filter passes.

### 3. What does NOT change

- `_evaluate`, `_concretize_graph`, `unified_instructions` — all correct as-is
- The `instruction_count() <= 4` filter in `_validate_gadget_worker`
- The super-optimizer pipeline
