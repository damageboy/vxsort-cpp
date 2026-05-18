# z3_avx Rust First Wave Design

**Goal:** Start the long-term Python-to-Rust port by creating a Rust workspace with a `z3_avx` crate that ports shared register utilities and one representative immediate-controlled intrinsic family.

**Scope:** This first wave ports only the common utilities needed to build and inspect AVX-sized Z3 bit-vector registers plus `_mm256_permute_ps`, `_mm512_permute_ps`, and `_mm512_mask_permute_ps`. Other instruction families remain in Python for now.

**Architecture:** The Rust crate uses native Z3 through the Rust `z3` crate for this milestone. The installed `z3` crate uses a thread-local context API, so public functions return bit-vector ASTs without taking an explicit `Context` parameter. Immediate operands use an `Imm8` enum with `Literal(u8)` and `Expr(BV)` variants so tests can cover both concrete immediates and solver-selected symbolic immediates.

**Modules:**

- `src/lib.rs`: public crate surface and module exports.
- `src/registers.rs`: YMM/ZMM constructors, register packing from concrete lane values, uniqueness constraints, reversed-register constraints, and construction from selected elements.
- `src/control.rs`: `_MM_SHUFFLE`, `_MM_SHUFFLE2`, mask decode helpers, string formatting helpers, and `Imm8`.
- `src/lanes.rs`: shared extraction, concatenation, and selector helpers.
- `src/permute.rs`: shared `permute_ps_generic` plus `_mm256_permute_ps`, `_mm512_permute_ps`, and `_mm512_mask_permute_ps`.
- `tests/registers.rs`: migrated tests for shared register construction behavior.
- `tests/permute_ps.rs`: migrated tests for `_mm256_permute_ps` and `_mm512_permute_ps`.

**Rust API Shape:**

```rust
pub enum Imm8 {
    Literal(u8),
    Expr(BV),
}

pub fn ymm_reg(name: &str) -> BV;
pub fn zmm_reg(name: &str) -> BV;

pub fn ymm_reg_with_32b_values(
    name: &str,
    solver: &Solver,
    values: &[u64],
) -> BV;

pub fn construct_ymm_reg_from_elements(
    bits: u32,
    element_specs: &[(&BV, usize)],
) -> BV;

pub fn mm256_permute_ps(a: &BV, imm8: Imm8) -> BV;
pub fn mm512_permute_ps(a: &BV, imm8: Imm8) -> BV;
pub fn mm512_mask_permute_ps(src: &BV, k: &BV, a: &BV, imm8: Imm8, solver: &Solver) -> BV;
```

**Behavioral Parity Notes:**

- Register packing must preserve the Python/Z3 bit ordering: lane 0 occupies the least significant bits, and concatenation is performed most-significant lane first.
- Concrete-valued register helpers must create fresh symbolic lane variables and add equality constraints to the supplied solver, matching Python. They should not collapse the whole register into one packed constant, because the tests and future synthesis workflows rely on named symbolic pieces and solver-carried facts.
- Selector helpers should build nested `ite` expressions so symbolic controls remain solver-searchable.
- Literal immediates should be converted to 8-bit bit-vector values internally, not handled by a separate semantic path.
- `_mm*_permute_ps` uses the same 8-bit control for every 128-bit lane.
- `_mm512_mask_permute_ps` uses the same generic permutation result and then applies an element writemask: mask bit 1 selects the permuted element, mask bit 0 preserves the corresponding `src` element.
- Masked instructions must be solver-aware when a symbolic mask is wider than the number of live element mask bits. For `_mm512_mask_permute_ps`, only 16 mask bits are live; bits above 15 are canonicalized to zero through the supplied solver.
- The unmasked `permute_ps` immediate has no dead bits. The masked wrapper does have canonicalization side effects for non-live high mask bits.

**Testing Strategy:**

- Write each Rust test before the implementation it validates.
- Compare Z3 expressions with satisfiability checks rather than string snapshots.
- Start with register width and packing tests, then helper tests, then literal identity permute, then symbolic identity immediate discovery.
- Keep this first Rust test suite small and fast.

**Out of Scope:**

- Full `src/z3_avx.py` parity.
- Python/Rust interop.
- WASM backend implementation.
- AVX512 masks, variable index permutes, blends, unpack, alignr, and min/max.

**Review Note:** The Superpowers brainstorming workflow normally asks for a spec-review subagent. This environment only permits spawning subagents when the user explicitly asks for subagents, so this spec is locally reviewed instead.
