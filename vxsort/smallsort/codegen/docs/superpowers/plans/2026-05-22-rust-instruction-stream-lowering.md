# Rust Instruction Stream Lowering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split Rust solution lowering from ASM text rendering so constant/materialization GVN is shared by both ASM export and future in-process UICA scoring.

**Architecture:** Introduce a Rust `instruction_stream` module that lowers solution paths into a canonical modeled instruction stream. The stream owns instruction mnemonics, operands, comments, rodata/constants, virtual/physical register choices, and constant materialization reuse. `asm_exporter` becomes a renderer over that stream. Future UICA integration will consume the same stream directly instead of parsing or dumping assembly.

**Tech Stack:** Rust workspace (`bitonic_codegen`, `gadget_synth`), release-mode cargo tests, clippy with warnings denied.

---

## Design Boundaries

### In scope

- Create modeled instruction stream types in Rust.
- Move current ASM-emission logic toward a lowering-then-rendering pipeline.
- Preserve existing `--asm-output-path` CLI behavior.
- Add the first shared constant/materialization layer for immediates and bit-vector constants.
- Preserve current ASM MVP behavior for every currently supported Rust exporter compare-swap case: AVX2 i64 emulation, i32 native min/max, and non-AVX2 i64 native min/max.
- Unsupported gadget operands must remain explicit in the stream/output rather than silently disappearing.

### Out of scope for this plan

- Implementing UICA scoring itself.
- Full AVX512 k-mask/vector-constant load parity.
- NASM validation.
- Python parity tests.
- Checkpointing.

## File Map

### Create

- `bitonic_codegen/src/instruction_stream.rs`
  - Defines `InstructionStream`, `InstructionBlock`, `ModeledInstruction`, `Operand`, `Register`, `ConstantData`, `LoweringOptions`.
  - Provides lowering entry points from `TransitionTable` + `CompletePath`.
  - Owns constant pooling/GVN decisions.

### Modify

- `bitonic_codegen/src/lib.rs`
  - Export `instruction_stream` module.
- `bitonic_codegen/src/asm_exporter.rs`
  - Replace direct string construction for instruction lines with stream rendering.
  - Keep `generate_solution_asm_for_paths` and `write_solution_asm_for_paths` public APIs stable.
- `bitonic_codegen/tests/asm_exporter.rs`
  - Keep existing behavior tests.
- Add `bitonic_codegen/tests/instruction_stream.rs`
  - Unit tests for lowering and constant reuse.

---

### Task 1: Add Modeled Instruction Stream Types

**Files:**
- Create: `bitonic_codegen/src/instruction_stream.rs`
- Modify: `bitonic_codegen/src/lib.rs`
- Test: `bitonic_codegen/tests/instruction_stream.rs`

- [ ] **Step 1: Write failing type/constructor tests**

Add tests proving a stream can hold:

```rust
InstructionStream {
    blocks: vec![InstructionBlock {
        label: Some("vxsort_solution_0".to_owned()),
        instructions: vec![ModeledInstruction {
            mnemonic: "vpermq".to_owned(),
            operands: vec![Operand::Register(Register::Ymm(0)), Operand::Immediate(78)],
            comment: Some("stage 0".to_owned()),
        }],
    }],
    constants: vec![],
}
```

Expected: compilation fails until the module/types exist.

- [ ] **Step 2: Implement minimal public types**

Keep types simple and serializable-by-inspection; do not add serde unless needed.

- [ ] **Step 3: Export the module from `lib.rs`**

Add:

```rust
pub mod instruction_stream;
```

- [ ] **Step 4: Verify focused tests**

```bash
cargo fmt --all --check
cargo test --release -q -p bitonic_codegen --test instruction_stream
```

### Task 2: Lower Complete Paths to InstructionStream

**Files:**
- Modify: `bitonic_codegen/src/instruction_stream.rs`
- Test: `bitonic_codegen/tests/instruction_stream.rs`
- Reference: current `bitonic_codegen/src/asm_exporter.rs`

- [ ] **Step 1: Add failing lowering test for a one-stage AVX2 i64 path**

Build a hand-authored `TransitionTable` containing `_mm256_permute4x64_epi64`, lower one `CompletePath`, and assert modeled instructions include:

- label block `vxsort_solution_0`
- `vpermq`
- AVX2 i64 compare-swap sequence: `vpcmpgtq`, `vblendvpd`, `vblendvpd`
- final `ret`

- [ ] **Step 2: Move register allocator concepts into lowering**

Lowering should decide source/destination registers and temporaries. ASM rendering must not allocate registers.

- [ ] **Step 3: Lower current supported instruction operands**

Support:

- `InstructionArg::Input("top" | "bottom" | "prev" | "input" | "result_N")`
- `InstructionArg::U64` immediate operands
- unsupported operands as explicit comments or `Operand::Unsupported`, not silently dropped

- [ ] **Step 4: Verify focused tests**

```bash
cargo fmt --all --check
cargo test --release -q -p bitonic_codegen --test instruction_stream
```

### Task 3: Make ASM Exporter Render InstructionStream

**Files:**
- Modify: `bitonic_codegen/src/asm_exporter.rs`
- Test: `bitonic_codegen/tests/asm_exporter.rs`
- Test: `bitonic_codegen/tests/cli.rs`

- [ ] **Step 1: Add/keep failing regression expectations**

Existing ASM tests should continue to assert readable text contains `section .text`, `global`, `vpermq`, `vpcmpgtq`, `vblendvpd`, and `ret`.

- [ ] **Step 2: Replace direct instruction construction with stream lowering**

`generate_solution_asm_for_paths` should:

```rust
let stream = lower_solution_paths(metadata, table, paths, LoweringOptions::default());
render_asm(&stream)
```

- [ ] **Step 3: Implement `render_asm` as pure formatting**

Renderer should not inspect `TransitionTable`, allocate temps, or perform constant pooling.

- [ ] **Step 4: Verify ASM and CLI tests**

```bash
cargo fmt --all --check
cargo test --release -q -p bitonic_codegen --test asm_exporter
cargo test --release -q -p bitonic_codegen --test cli
```

### Task 4: Add Constant Materialization / GVN Foundation

**Files:**
- Modify: `bitonic_codegen/src/instruction_stream.rs`
- Modify: `bitonic_codegen/src/asm_exporter.rs`
- Test: `bitonic_codegen/tests/instruction_stream.rs`
- Test: `bitonic_codegen/tests/asm_exporter.rs`

- [ ] **Step 1: Add failing tests for constant reuse**

Create two modeled constant requests for the same bit-vector payload and assert they map to one `ConstantData` entry and one label/id.

- [ ] **Step 2: Define materialization representation**

Add explicit stream types for constant data and materialized values:

```rust
enum ConstantKey {
    Vector { bits: u32, hex: String },
    KMask { bits: u32, value: u64 },
}

struct ConstantData {
    label: String,
    key: ConstantKey,
}

enum Operand {
    Register(Register),
    Immediate(u64),
    ConstantRef(String),
    Unsupported(String),
}
```

- [ ] **Step 3: Add deterministic GVN rules**

The lowering pass owns these rules, shared by ASM and future UICA:

- identical vector constants reuse the same `ConstantData` label;
- identical k-mask constants reuse the same `ConstantData` label/value;
- if a constant is already materialized into a register in the current solution stream, later uses reuse that register unless a later pass explicitly invalidates it;
- label allocation is deterministic (`LC0`, `LC1`, ... or equivalent);
- immediate operands remain immediates and are not pooled.

- [ ] **Step 4: Represent load/broadcast/k-mask materialization as modeled instructions**

Add modeled materialization op forms rather than ASM strings, for example:

```rust
ModeledInstruction { mnemonic: "vmovdqa64", operands: vec![Operand::Register(tmp), Operand::ConstantRef(label)] }
ModeledInstruction { mnemonic: "kmovw", operands: vec![Operand::KMask(1), Operand::Immediate(value)] }
```

ASM rendering turns these into text. Future UICA consumes the same modeled ops directly.

- [ ] **Step 5: Render rodata from `ConstantData`**

`asm_exporter` should render a deterministic rodata section when `InstructionStream.constants` is non-empty. It should not allocate or pool constants itself.

- [ ] **Step 6: Lower `InstructionArg::BitVec` through the pool**

For currently unsupported instruction forms, emit `Operand::Unsupported` with enough detail to debug. Do not silently drop constants.

- [ ] **Step 7: Verify focused tests**

```bash
cargo fmt --all --check
cargo test --release -q -p bitonic_codegen --test instruction_stream
cargo test --release -q -p bitonic_codegen --test asm_exporter
```

### Task 5: Preserve Current ASM Behavior Across Supported Configs

**Files:**
- Modify: `bitonic_codegen/src/instruction_stream.rs`
- Modify: `bitonic_codegen/src/asm_exporter.rs`
- Test: `bitonic_codegen/tests/asm_exporter.rs`

- [ ] **Step 1: Add compare-swap regression tests**

Cover all current Rust exporter compare-swap cases:

- AVX2 i64 emits `vpcmpgtq` + two `vblendvpd` instructions;
- AVX2 i32 emits `vpminsd` + `vpmaxsd`;
- AVX512 i64 emits `vpminsq` + `vpmaxsq`.

- [ ] **Step 2: Keep unsupported gadget instruction forms explicit**

If a gadget cannot be fully rendered yet, the stream/ASM must contain an `unsupported` comment/instruction marker instead of omitting it.

- [ ] **Step 3: Verify ASM tests**

```bash
cargo fmt --all --check
cargo test --release -q -p bitonic_codegen --test asm_exporter
```

### Task 6: Full Rust Verification Gate

**Files:**
- Rust workspace only.

- [ ] **Step 1: Run formatting**

```bash
cargo fmt --all --check
```

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --all-targets --all-features --release -- -D warnings
```

- [ ] **Step 3: Run Rust tests**

```bash
cargo test --release -q
```

Expected: no warnings, no clippy issues, all tests pass.

---

## Execution Notes for Subagents

- Do not implement UICA in this plan.
- Do not add Python parity tests.
- Keep ASM output behavior stable while moving logic behind the new stream.
- Treat compiler warnings as failures.
- Use release-mode Rust tests by default.
