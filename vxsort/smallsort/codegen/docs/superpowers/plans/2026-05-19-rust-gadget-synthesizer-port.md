# Rust Gadget Synthesizer Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Python `GadgetSynthesizer` behavior to Rust until Rust independently emits the same template and synthesized gadget/minigraph artifacts as Python.

**Architecture:** Use artifact comparison as the migration ratchet. Python and Rust each dump normalized gadget synthesis artifacts without running the full super-optimizer; a comparer reports structural and concrete synthesis differences. Rust reaches parity in layers: template enumeration, depth-1 synthesis, recursive graph/mux support, depth-2 synthesis, shared-prefix graphs, AVX512 unmasked, then AVX512 masked.

**Tech Stack:** Python `uv`, `z3`, existing `src/gadget_synthesizer.py`; Rust workspace crates `z3_avx` and `gadget_synth`; JSONL fixture dumps; `cargo test`; `pytest`; `ruff`; `vulture`.

---

## Boundaries

### In scope

- Python standalone gadget-synth dump tool.
- Rust standalone gadget-synth dump tool.
- Canonical JSONL schemas for template graphs and synthesized gadgets.
- Dump comparer.
- Rust parity with Python `GadgetSynthesizer` for:
  - candidate graph generation
  - depth-1 synthesis
  - recursive graph evaluation
  - mux constraints and pruning
  - depth-2 graph shapes
  - shared-prefix graph shapes
  - AVX2/AVX512 i32/i64 intrinsic dispatch coverage

### Out of scope for this plan

- Replacing Python orchestration.
- Porting `WaveEngine`.
- Porting multiprocessing/checkpointing/UI/path selection.
- PyO3 integration. Use standalone artifacts first.

---

## File Map

### New files

- `tools/dump_gadget_synth.py`
  - Python reference dumper.
  - Modes: `templates`, `synthesized`.
  - Uses current Python `GadgetSynthesizer` directly.

- `tools/compare_gadget_synth_dumps.py`
  - Compares canonical JSONL dumps.
  - Reports missing/extra/different records grouped by stable keys.

- `gadget_synth/src/bin/dump_gadget_synth.rs`
  - Rust dumper matching Python schema.
  - Modes: `templates`, `synthesized`.

- `gadget_synth/tests/dump_schema.rs`
  - Rust-side schema/normalization tests.

- `tests/test_gadget_synth_dump.py`
  - Python dumper/comparer tests.

### Likely modified files

- `gadget_synth/src/types.rs`
  - Add recursive graph node model.
  - Add `MuxNode` and mux pruning enum.
  - Add serialization-friendly normalization helpers if appropriate.

- `gadget_synth/src/intrinsics.rs`
  - Expand Rust intrinsic dispatch coverage.
  - Keep intrinsic template enumeration aligned with Python `src/intrinsics/*.py`.

- `gadget_synth/src/synthesizer.rs`
  - Add recursive evaluation.
  - Add mux constraints.
  - Add topological concretization.
  - Add K-per-wiring synthesis algorithm.
  - Add depth-2/shared-prefix graph generation.

- `gadget_synth/Cargo.toml`
  - Add CLI deps if needed, e.g. `serde`, `serde_json`, `clap`.

- Root `Cargo.toml`
  - Workspace dependency additions if needed.

---

## Canonical Dump Rules

### Template dump records

Each record represents one normalized `GadgetGraph` template. It must not include unstable object IDs or raw symbolic variable names.

Required fields:

```json
{
  "kind": "template",
  "arch": "avx2",
  "dtype": "i64",
  "gadget_depth": 2,
  "tier": "shallow",
  "graph": {
    "top": {"kind": "intrinsic", "name": "_mm256_permute_pd", "operands": {}},
    "bottom": null
  }
}
```

Normalization rules:

- Symbolics become `{ "kind": "symbolic", "role": "imm8", "bits": 8, "var_id": "v0" }` or equivalent role-based values.
- `var_id` is deterministic within one graph and must preserve aliasing/independence: two references to the same symbolic variable get the same `var_id`; independent variables get different `var_id`s.
- Mux select names are normalized to `{ "kind": "symbolic", "role": "mux_select", "bits": 2, "var_id": "vN" }`; never emit pointer-derived names.
- Intrinsic operands are recursively normalized.
- Operand key order is sorted for comparison.
- `isomorphic_order` is included for template comparison.
- Mux pruning mode is included.
- JSONL output is sorted deterministically.

### Synthesized dump records

Each record represents one concrete validated gadget result for a fixture.

Required fields:

```json
{
  "kind": "synthesized",
  "arch": "avx2",
  "dtype": "i64",
  "gadget_depth": 1,
  "fixture": "identity_pairs",
  "input_state": {"top": [0, 1, 2, 3], "bottom": [4, 5, 6, 7]},
  "target_pairs": [[0, 4], [1, 5], [2, 6], [3, 7]],
  "output_state": {"top": [0, 1, 2, 3], "bottom": [4, 5, 6, 7]},
  "top_instructions": [
    {"name": "_mm256_permute_pd", "args": {"a": "top", "imm8": 5}}
  ],
  "bottom_instructions": []
}
```

Normalization rules:

- Concrete register operands use only strings like `top`, `bottom`, `prev`, `result_N`.
- Small concrete immediates/masks may be JSON integers when they fit exactly and unambiguously.
- Wide control-vector operands must use a portable bit-vector encoding, e.g. `{ "kind": "bitvec", "bits": 256, "hex": "0x..." }`; do not force 256/512-bit values through JSON numbers or Rust `u64`.
- No Z3 expressions.
- Records are compared as multisets, not plain sets; duplicate canonical records are reported explicitly.
- Stable record keys must include enough context for actionable diffs: `kind`, `arch`, `dtype`, `gadget_depth`, `fixture` when present, `template_id` or `graph_key`, `output_state` for synthesized records, and instruction keys.

---

## Fixtures

Start with small fixed fixtures. Do not run the full optimizer.

### `identity_pairs`

AVX2 i64:

```python
input_state = VectorState(top=[0, 1, 2, 3], bottom=[4, 5, 6, 7])
target_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
```

AVX2 i32:

```python
input_state = VectorState(top=list(range(8)), bottom=list(range(8, 16)))
target_pairs = list(zip(range(8), range(8, 16)))
```

AVX512 i64:

```python
input_state = VectorState(top=list(range(8)), bottom=list(range(8, 16)))
target_pairs = list(zip(range(8), range(8, 16)))
```

AVX512 i32:

```python
input_state = VectorState(top=list(range(16)), bottom=list(range(16, 32)))
target_pairs = list(zip(range(16), range(16, 32)))
```

Add more fixtures only when needed to expose mismatches.

---

## Runtime Policy

Default test commands must stay fast. Unit tests and depth-1 parity checks can run under normal `pytest`/`cargo test`. Full synthesized depth-2 and AVX512 matrix parity should be gated behind an explicit command, slow marker, or verification script so the default Python test suite does not become a multi-minute synthesis run.

Both Python and Rust dumpers must support equivalent intrinsic filtering options before AVX512 staging:

- `--intrinsic-filter name1,name2,...` for exact allowlists.
- Or paired convenience flags such as `--include-masked` / `--exclude-masked`.

The comparer should fail if the two dumps were generated with incompatible filter metadata.

---

## Task 1: Python template dumper

**Files:**
- Create: `tools/dump_gadget_synth.py`
- Test: `tests/test_gadget_synth_dump.py`

- [ ] **Step 1: Write tests for template dump mode**

Add tests that run the dumper in-process or through `subprocess` for AVX2 i64 depth 1 and assert:

- JSONL parses.
- All records have `kind == "template"`.
- Records contain `arch`, `dtype`, `gadget_depth`, `graph`.
- No record contains Python object IDs.

Run:

```bash
uv run pytest tests/test_gadget_synth_dump.py -q
```

Expected: fail because tool does not exist.

- [ ] **Step 2: Implement minimal Python template dumper**

Implement:

```bash
uv run python tools/dump_gadget_synth.py templates \
  --vector-machine AVX2 --datatype i64 --gadget-depth 1 \
  --output /tmp/py-templates.jsonl
```

Also design the CLI so `--intrinsic-filter`, `--include-masked`, or `--exclude-masked` can be added without changing the schema later.

Use:

```python
synth = GadgetSynthesizer(vm, prim_type)
graphs = synth.precompute_all_candidates(gadget_depth)
```

Normalize `InputRef`, `Symbolic`, `Mux`, `IntrinsicNode`, and `None` recursively.

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_gadget_synth_dump.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add tools/dump_gadget_synth.py tests/test_gadget_synth_dump.py
git commit -m "test: add Python gadget synth template dumper"
```

---

## Task 2: Rust template dumper

**Files:**
- Create: `gadget_synth/src/bin/dump_gadget_synth.rs`
- Create/modify: `gadget_synth/tests/dump_schema.rs`
- Modify: `gadget_synth/Cargo.toml`

- [ ] **Step 1: Write Rust dump schema tests**

Test that AVX2 i64 depth 1 template dump can be produced and contains stable normalized JSON records.

Run:

```bash
cargo test -p gadget_synth dump_schema
```

Expected: fail because CLI/schema support does not exist.

- [ ] **Step 2: Add CLI dependencies if needed**

Prefer simple dependencies:

```toml
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

Use workspace dependencies if already present or preferred.

- [ ] **Step 3: Implement Rust `templates` mode**

Expose or call existing candidate generation:

```rust
let synth = GadgetSynthesizer::new(arch, dtype);
let graphs = synth.precompute_all_candidates(gadget_depth);
```

Keep CLI option names and filter metadata aligned with the Python dumper.

If `precompute_all_candidates` is private, make a small public method with a clear name rather than exposing internals casually.

- [ ] **Step 4: Run Rust tests**

```bash
cargo fmt --all --check
cargo test -p gadget_synth
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add gadget_synth/Cargo.toml gadget_synth/src/bin/dump_gadget_synth.rs gadget_synth/tests/dump_schema.rs
git commit -m "test: add Rust gadget synth template dumper"
```

---

## Task 3: Dump comparer

**Files:**
- Create: `tools/compare_gadget_synth_dumps.py`
- Modify: `tests/test_gadget_synth_dump.py`

- [ ] **Step 1: Write comparer tests**

Create tiny temporary JSONL files and assert:

- equal files return exit 0
- missing records return non-zero
- extra records return non-zero
- changed records return non-zero with useful key output

Run:

```bash
uv run pytest tests/test_gadget_synth_dump.py -q
```

Expected: fail.

- [ ] **Step 2: Implement comparer**

Comparer should:

- parse JSONL
- canonicalize each record via sorted JSON
- compute stable display keys for each record
- compare multisets, not plain sets
- report duplicate canonical records explicitly
- print concise grouped differences
- support `--max-diffs`

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_gadget_synth_dump.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add tools/compare_gadget_synth_dumps.py tests/test_gadget_synth_dump.py
git commit -m "test: add gadget synth dump comparer"
```

---

## Task 4: Template parity ratchet

**Files:**
- Modify: `tests/test_gadget_synth_dump.py`
- Modify as needed: `gadget_synth/src/synthesizer.rs`, `gadget_synth/src/intrinsics.rs`, `gadget_synth/src/types.rs`

- [ ] **Step 1: Add integration test for AVX2 i64 depth 1 template parity**

Test command flow:

```bash
uv run python tools/dump_gadget_synth.py templates --vector-machine AVX2 --datatype i64 --gadget-depth 1 --output /tmp/py.jsonl
cargo run -q -p gadget_synth --bin dump_gadget_synth -- templates --arch avx2 --dtype i64 --gadget-depth 1 --output /tmp/rust.jsonl
uv run python tools/compare_gadget_synth_dumps.py /tmp/py.jsonl /tmp/rust.jsonl
```

Expected initially: may fail if normalization or candidate shape differs.

- [ ] **Step 2: Fix Rust/Python normalization mismatches**

Do not change semantics to make tests pass. First inspect differences and align dump normalization.

- [ ] **Step 3: Extend depth-1 parity matrix only**

Add template parity checks in this order:

1. AVX2 i64 depth 1
2. AVX2 i32 depth 1
3. AVX512 i64 depth 1
4. AVX512 i32 depth 1

Do not add depth-2 template parity here. Depth-2 requires the recursive graph model and mux support from later tasks.

- [ ] **Step 4: Commit after each green expansion**

Use commits like:

```bash
git commit -m "test: match AVX2 i64 depth-1 gadget templates"
```

---

## Task 5: Python synthesized dumper mode

**Files:**
- Modify: `tools/dump_gadget_synth.py`
- Modify: `tests/test_gadget_synth_dump.py`

- [ ] **Step 1: Add tests for synthesized mode**

Test AVX2 i64 `identity_pairs`, `gadget_depth=1`, `max_unique_outputs=3`.

Assert:

- records parse
- `kind == "synthesized"`
- records include `input_state`, `target_pairs`, `output_state`, instruction lists
- concrete register args contain only strings
- concrete immediates/masks are integers only when narrow
- concrete control vectors use the portable bit-vector encoding

- [ ] **Step 2: Implement synthesized mode**

For each precomputed graph:

```python
results, _, _ = synth.synthesize_gadget_with_symbolic(
    graph,
    input_state,
    target_pairs,
    max_solutions=1,
    max_unique_outputs=max_unique_outputs,
)
```

Normalize concrete `PermutationGadget` records.

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_gadget_synth_dump.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add tools/dump_gadget_synth.py tests/test_gadget_synth_dump.py
git commit -m "test: add Python gadget synth synthesized dumper"
```

---

## Task 6: Rust synthesized dumper mode

**Files:**
- Modify: `gadget_synth/src/bin/dump_gadget_synth.rs`
- Modify: `gadget_synth/tests/dump_schema.rs`
- Modify as needed: `gadget_synth/src/synthesizer.rs`, `gadget_synth/src/types.rs`

- [ ] **Step 1: Add Rust tests for synthesized mode**

Use AVX2 i64 `identity_pairs`, `gadget_depth=1`.

Run:

```bash
cargo test -p gadget_synth dump_schema
```

Expected: may fail until CLI mode exists.

- [ ] **Step 2: Implement synthesized mode using existing Rust synthesis**

Loop graphs and call:

```rust
synth.synthesize_graph(&graph, &input_state, &target_pairs, options)
```

Normalize records to same schema as Python.

- [ ] **Step 3: Run tests**

```bash
cargo fmt --all --check
cargo test -p gadget_synth
```

Expected: pass for currently supported intrinsics or skip unsupported graphs with explicit reporting. Prefer completing dispatch in Task 7 before strict parity.

- [ ] **Step 4: Commit**

```bash
git add gadget_synth/src/bin/dump_gadget_synth.rs gadget_synth/tests/dump_schema.rs gadget_synth/src/synthesizer.rs gadget_synth/src/types.rs
git commit -m "test: add Rust gadget synth synthesized dumper"
```

---

## Task 7: AVX2 i64 depth-1 synthesis parity

**Files:**
- Modify: `gadget_synth/src/intrinsics.rs`
- Modify: `gadget_synth/src/synthesizer.rs` if dispatch shape requires it
- Modify: `tests/test_gadget_synth_dump.py`

- [ ] **Step 1: Add failing parity test**

Compare Python vs Rust synthesized dumps for:

```text
arch=AVX2 dtype=i64 gadget_depth=1 fixture=identity_pairs max_unique_outputs=3
```

Expected: fail because Rust dispatch is incomplete.

- [ ] **Step 2: Add missing AVX2 i64 dispatch**

Implement dispatch for:

- `_mm256_permute4x64_epi64`
- `_mm256_permutexvar_epi64`
- `_mm256_permutevar_pd`
- `_mm256_shuffle_pd`
- `_mm256_unpacklo_epi64`
- `_mm256_unpackhi_epi64`
- `_mm256_blend_pd`
- `_mm256_alignr_epi8`

Existing support:

- `_mm256_permute_pd`
- `_mm256_permute2x128_si256`

- [ ] **Step 3: Run targeted tests**

```bash
cargo test -p gadget_synth
uv run pytest tests/test_gadget_synth_dump.py -q
```

Expected: parity test passes.

- [ ] **Step 4: Commit**

```bash
git add gadget_synth/src/intrinsics.rs gadget_synth/src/synthesizer.rs tests/test_gadget_synth_dump.py
git commit -m "feat: match AVX2 i64 depth-1 gadget synthesis"
```

---

## Task 8: AVX2 i32 depth-1 synthesis parity

**Files:**
- Modify: `gadget_synth/src/intrinsics.rs`
- Modify: `tests/test_gadget_synth_dump.py`

- [ ] **Step 1: Add failing parity test for AVX2 i32 depth 1**

- [ ] **Step 2: Add missing AVX2 i32 dispatch**

Implement dispatch for:

- `_mm256_permute_ps`
- `_mm256_permute4x64_epi64`
- `_mm256_permutexvar_epi32`
- `_mm256_permutevar_ps`
- `_mm256_shuffle_ps`
- `_mm256_unpacklo_epi32`
- `_mm256_unpackhi_epi32`
- `_mm256_permute2x128_si256`
- `_mm256_blend_ps`
- `_mm256_alignr_epi8`

- [ ] **Step 3: Run tests**

```bash
cargo test -p gadget_synth
uv run pytest tests/test_gadget_synth_dump.py -q
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: match AVX2 i32 depth-1 gadget synthesis"
```

---

## Task 9: Recursive graph model in Rust

**Files:**
- Modify: `gadget_synth/src/types.rs`
- Modify: `gadget_synth/src/intrinsics.rs`
- Modify: `gadget_synth/src/synthesizer.rs`
- Modify: `gadget_synth/src/bin/dump_gadget_synth.rs`
- Modify: `gadget_synth/tests/dump_schema.rs`

- [ ] **Step 1: Add tests for nested graph normalization**

Create a hand-built graph equivalent to single→single and assert dump shape is stable.

- [ ] **Step 2: Replace shallow `Operand` with recursive graph node model**

Target shape:

```rust
enum GadgetNode {
    Input(InputRef),
    Symbolic(Symbolic),
    Mux(MuxNode),
    Intrinsic(Box<IntrinsicNode>),
}
```

`IntrinsicNode.operands` should hold `GadgetNode` values.

- [ ] **Step 3: Update depth-1 constructors**

Keep existing depth-1 behavior identical.

- [ ] **Step 4: Update evaluator for recursive nodes**

Add recursive evaluation cache keyed by node identity or stable traversal index.

- [ ] **Step 5: Update concretization**

Match Python `_concretize_graph()`:

- topological post-order
- nested intrinsic operand becomes `prev` or `result_N`
- mux operand becomes selected source string

- [ ] **Step 6: Run tests**

```bash
cargo fmt --all --check
cargo test -p gadget_synth
uv run pytest tests/test_gadget_synth_dump.py -q
```

- [ ] **Step 7: Commit**

```bash
git commit -m "refactor: support recursive gadget graph nodes in Rust"
```

---

## Task 10: Depth-2 graph shape parity

**Files:**
- Modify: `gadget_synth/src/synthesizer.rs`
- Modify: `tests/test_gadget_synth_dump.py`

- [ ] **Step 1: Add failing template parity test for AVX2 i64 depth 2**

- [ ] **Step 2: Port Python `_build_gadget_graphs(depth=2)` shapes**

Implement:

- single → single hardwired
- dual → single hardwired
- single → dual with muxes
- asymmetric dual operand swaps

- [ ] **Step 3: Add mux type and pruning metadata**

Support `MuxNode` with:

- select symbolic bits
- sources: top, bottom, inst0
- pruning: `at_least_one_last`

- [ ] **Step 4: Run template parity tests**

```bash
uv run pytest tests/test_gadget_synth_dump.py -q
```

- [ ] **Step 5: Expand depth-2 template parity after AVX2 i64 passes**

Add depth-2 template parity in this order, still excluding shared-prefix if it has not been ported yet:

1. AVX2 i64 depth 2
2. AVX2 i32 depth 2
3. AVX512 i64 depth 2 with the same masked/unmasked filter on both dumpers
4. AVX512 i32 depth 2 with the same masked/unmasked filter on both dumpers

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: match depth-2 gadget template shapes"
```

---

## Task 11: Mux evaluation and K-per-wiring parity

> Note: full unfiltered AVX2 i64 depth-2 synthesized parity is deferred because
> the Python reference dump currently times out. Task 11 uses representative
> intrinsic-filter parity (`_mm256_permute_pd,_mm256_shuffle_pd` and
> `_mm256_permutexvar_epi64,_mm256_shuffle_pd`) to ratchet mux/K-per-wiring
> behavior while staying runnable.

**Files:**
- Modify: `gadget_synth/src/synthesizer.rs`
- Modify: `gadget_synth/src/types.rs`
- Modify: `tests/test_gadget_synth_dump.py`

- [ ] **Step 1: Add failing synthesized parity test for AVX2 i64 depth 2 excluding shared-prefix if needed**

- [ ] **Step 2: Implement mux evaluation**

Match Python behavior:

- range constraint: select <= sources.len() - 1
- If-chain source selection
- symbolic select stored with `sel_` role/name

- [ ] **Step 3: Implement mux pruning**

Match Python `_apply_mux_pruning()`:

- `force_last`
- `at_least_one_last`
- sibling mux joint constraint

- [ ] **Step 4: Implement K-per-wiring algorithm**

Mirror Python:

1. collect assertions
2. split select vars from non-select symbolic vars
3. optimize output wiring first
4. fix wiring
5. enumerate up to `max_unique_outputs`
6. minimize immediate/control variables for fixed output
7. block outputs and continue
8. block wiring and continue

- [ ] **Step 5: Run parity tests**

```bash
cargo test -p gadget_synth
uv run pytest tests/test_gadget_synth_dump.py -q
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: match muxed depth-2 gadget synthesis"
```

---

## Task 12: Shared-prefix graph parity

**Files:**
- Modify: `gadget_synth/src/synthesizer.rs`
- Modify: `tests/test_gadget_synth_dump.py`

- [ ] **Step 1: Add failing template parity test for shared-prefix depth 2**

- [ ] **Step 2: Port `_build_shared_prefix_graphs()`**

Rules to match exactly:

- symmetric dual + symbolic tail: one ordering with independent symbolic names per side
- asymmetric dual + symbolic tail: two same-ordering variants
- asymmetric dual + non-symbolic tail: cross-ordering
- symmetric dual + non-symbolic tail: skip

- [ ] **Step 3: Run template parity**

- [ ] **Step 4: Add synthesized parity for AVX2 i64 full depth 2**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: match shared-prefix gadget graphs"
```

---

## Task 13: AVX512 unmasked synthesis parity

**Files:**
- Modify: `gadget_synth/src/intrinsics.rs`
- Modify: `tests/test_gadget_synth_dump.py`

- [ ] **Step 1: Add AVX512 unmasked parity tests with masked intrinsics excluded**

Start depth 1, then depth 2. Add matching `--exclude-masked` or `--intrinsic-filter` support to both dumpers before these tests, because Python AVX512 candidate enumeration includes masked templates by default.

- [ ] **Step 2: Add unmasked AVX512 dispatch**

Implement:

- `_mm512_permutexvar_epi32`
- `_mm512_permutexvar_epi64`
- `_mm512_permute_ps`
- `_mm512_permute_pd`
- `_mm512_permutevar_ps`
- `_mm512_permutevar_pd`
- `_mm512_permutex2var_epi32`
- `_mm512_permutex2var_epi64`
- `_mm512_shuffle_ps`
- `_mm512_shuffle_pd`
- `_mm512_unpacklo_epi32`
- `_mm512_unpackhi_epi32`
- `_mm512_unpacklo_epi64`
- `_mm512_unpackhi_epi64`
- `_mm512_shuffle_i32x4`
- `_mm512_alignr_epi32`
- `_mm512_alignr_epi64`

- [ ] **Step 3: Run tests**

```bash
cargo fmt --all --check
cargo test -p gadget_synth
uv run pytest tests/test_gadget_synth_dump.py -q
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: match AVX512 unmasked gadget synthesis"
```

---

## Task 14: AVX512 masked synthesis parity

**Files:**
- Modify: `gadget_synth/src/intrinsics.rs`
- Modify: `gadget_synth/src/synthesizer.rs` if mask canonicalization requires it
- Modify: `tests/test_gadget_synth_dump.py`

- [ ] **Step 1: Add masked AVX512 parity tests with explicit masked-intrinsic selection**

Use matching `--include-masked` or `--intrinsic-filter` settings in both dumpers so failures are about masked behavior, not mismatched candidate menus.

- [ ] **Step 2: Add masked AVX512 dispatch**

Implement:

- `_mm512_mask_permutexvar_epi32`
- `_mm512_mask_permutexvar_epi64`
- `_mm512_mask_permute_ps`
- `_mm512_mask_permute_pd`
- `_mm512_mask_permutevar_ps`
- `_mm512_mask_permutevar_pd`
- `_mm512_mask_permutex2var_epi32`
- `_mm512_mask_permutex2var_epi64`
- `_mm512_mask_shuffle_ps`
- `_mm512_mask_shuffle_pd`
- `_mm512_mask_unpacklo_epi32`
- `_mm512_mask_unpackhi_epi32`
- `_mm512_mask_unpacklo_epi64`
- `_mm512_mask_unpackhi_epi64`
- `_mm512_mask_shuffle_i32x4`
- `_mm512_mask_alignr_epi32`
- `_mm512_mask_alignr_epi64`

- [ ] **Step 3: Verify mask canonicalization**

Ensure dead mask bits and dead immediate/control bits are minimized/canonicalized consistently with Python/Rust `z3_avx` semantics.

- [ ] **Step 4: Run tests**

```bash
cargo fmt --all --check
cargo test -p gadget_synth
uv run pytest tests/test_gadget_synth_dump.py -q
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: match AVX512 masked gadget synthesis"
```

---

## Task 15: Final full-matrix verification

**Files:**
- Modify docs if needed.

- [ ] **Step 1: Run full Rust verification**

```bash
cargo fmt --all --check
cargo test -q
```

Expected: all Rust tests pass.

- [ ] **Step 2: Run full Python verification**

```bash
uv run pytest
uv run ruff check .
uv run vulture
```

Expected:

- `pytest` passes.
- `ruff` passes.
- `vulture` may exit 3 with existing unrelated findings; inspect and confirm no new relevant findings.

- [ ] **Step 3: Generate and compare full dump matrix via an explicit verification script**

Do not put the full depth-2/AVX512 synthesized matrix in default `pytest`; it is allowed to be slower and should be gated behind an explicit command or slow marker.

For each combo:

- AVX2 i64 depth 1/2
- AVX2 i32 depth 1/2
- AVX512 i64 depth 1/2
- AVX512 i32 depth 1/2

Run Python and Rust dumps and compare.

- [ ] **Step 4: Commit final verification/docs updates**

```bash
git commit -m "test: verify full gadget synthesizer parity"
```

---

## Execution Notes

- Use AVX2 i64 first for speed and easier debugging.
- Do not wire Rust into Python orchestration until dump parity is green.
- Keep commits small and revertible.
- When a parity mismatch occurs, classify it before fixing:
  1. dump normalization mismatch
  2. candidate enumeration mismatch
  3. intrinsic dispatch/semantics mismatch
  4. solver enumeration/canonicalization mismatch
  5. concrete instruction extraction mismatch
- Prefer adding a focused fixture over debugging against a full optimizer run.

---

## Completion Definition

The Rust synthesizer matches Python when:

```bash
uv run python tools/dump_gadget_synth.py templates ... > py-templates.jsonl
cargo run -p gadget_synth --bin dump_gadget_synth -- templates ... > rust-templates.jsonl
uv run python tools/compare_gadget_synth_dumps.py py-templates.jsonl rust-templates.jsonl

uv run python tools/dump_gadget_synth.py synthesized ... > py-synth.jsonl
cargo run -p gadget_synth --bin dump_gadget_synth -- synthesized ... > rust-synth.jsonl
uv run python tools/compare_gadget_synth_dumps.py py-synth.jsonl rust-synth.jsonl
```

passes for the full supported matrix:

- AVX2 i32/i64
- AVX512 i32/i64
- depth 1 and depth 2
- shared-prefix included
- masked AVX512 included

Only then should we add a Python adapter or replace the Python synthesis worker path.
