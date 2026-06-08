# Rust ID-Based Solution Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace path and assigned-path storage that copies full state tuples with ID-based storage over interned states and per-stage transition arrays.

**Architecture:** `TransitionTable` becomes an interned transition graph: states are stored once, transitions are stored once per stage, and complete paths store only transition indexes. Current synchronous rough scoring, full scoring, JSON export, ASM export, and verification keep the same behavior by resolving IDs through the table at the API boundary. Async rough/full scoring is explicitly out of scope for this plan, but this model adds the reverse indexes that async scoring will need later.

**Tech Stack:** Rust, `std::collections::{HashMap, HashSet}`, existing `gadget_synth::VectorState`, existing `TransitionTable`, existing scoring/export/verifier tests.

---

## File Structure

- Modify `bitonic_codegen/src/transition_table.rs`
  - Own state interning, per-stage transition arrays, transition lookup indexes, path discovery, and compatibility state-resolution helpers.
- Modify `bitonic_codegen/src/scoring.rs`
  - Convert `AssignedPath` and `AssignedPathKey` from copied `(stage, input, output)` tuples to transition/gadget IDs.
- Modify `bitonic_codegen/src/wave_engine.rs`
  - Carry `StateId` through synthesis jobs where the input state is already known, insert output states through the interner, and store scored paths by ID-based complete paths.
- Modify `bitonic_codegen/src/instruction_stream.rs`
  - Resolve ID-based complete/assigned paths against `TransitionTable` during lowering.
- Modify `bitonic_codegen/src/json_exporter.rs`
  - Export the existing JSON shape by resolving state IDs on demand.
- Modify `bitonic_codegen/src/json_importer.rs`
  - Import existing JSON into interned states and ID-based paths.
- Modify `bitonic_codegen/src/asm_exporter.rs`
  - Keep ASM export behavior while accepting ID-based paths.
- Modify `bitonic_codegen/src/verifier.rs`
  - Resolve ID-based paths to concrete states and gadgets for verification.
- Modify tests under `bitonic_codegen/tests/`
  - Update transition table, scoring, wave engine, lowering, JSON, ASM, verifier, and uiCA scoring tests.

## Representation Decisions

Use dynamic `State` storage for this work item:

```rust
pub type LaneLabel = u8;

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct State {
    // Invariant: lanes.len() == 2 * lanes_per_vector.
    // Layout: [top lanes..., bottom lanes...].
    lanes: Vec<LaneLabel>,
}
```

Do not make `TransitionTable` const-generic in this first migration. The current solver configuration is runtime-selected by CLI, and genericizing `WaveEngine`, importer/exporter, scoring, runtime summaries, and tests would make the storage refactor much larger. This plan gets the main win: paths and assigned paths no longer copy state vectors. A later cleanup can change `State` to `State<const LANES: usize> { top: [u8; LANES], bottom: [u8; LANES] }` once the ID model is stable.

Switch internal labels to zero-based `u8`:

```text
old internal labels: 1..=total_elements
new internal labels: 0..total_elements
```

This allows exactly 256 elements to fit in `u8` as `0..=255`. JSON and user-facing comments can remain compatible if needed by converting at the boundary.

All live synthesis inputs and target pairs must use the same zero-based convention. `gadget_synth::VectorState` can remain `Vec<u64>` for now, but the labels carried inside it during solve should be zero-based. JSON import/export, verifier setup, and human-readable state comments can convert to one-based labels if they need to preserve the current external shape.

Core IDs:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StateId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TransitionIndex(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PathId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct GadgetIndex(pub u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TransitionRef {
    pub stage: u16,
    pub transition: TransitionIndex,
}
```

Core storage:

```rust
pub struct StateInterner {
    lanes_per_vector: usize,
    states: Vec<State>,
    by_state: HashMap<State, StateId>,
}

pub struct TransitionRecord {
    pub input: StateId,
    pub output: StateId,
    pub gadgets: Vec<PermutationGadget>,
}

pub struct StageData {
    transitions: Vec<TransitionRecord>,
    transition_by_pair: HashMap<(StateId, StateId), TransitionIndex>,
    transitions_by_input: HashMap<StateId, Vec<TransitionIndex>>,
    transitions_by_output: HashMap<StateId, Vec<TransitionIndex>>,

    unique_inputs: HashSet<StateId>,
    unique_outputs: HashSet<StateId>,
    attempted_pairs: HashSet<(StateId, usize)>,
    forwarded_outputs: HashSet<StateId>,

    consecutive_zero_budgets: usize,
    unproductive_waves: usize,
    attempts: usize,
    dirty: bool,
}

pub struct TransitionTable {
    states: StateInterner,
    stages: Vec<StageData>,
}
```

Complete and assigned paths:

```rust
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct CompletePath {
    // transitions[stage] indexes TransitionTable::stages[stage].transitions.
    transitions: Vec<TransitionIndex>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct AssignedPath {
    // Same stage order as the complete path.
    path: CompletePath,
    gadgets: Vec<GadgetIndex>,
}

pub type AssignedPathKey = Vec<(TransitionRef, GadgetIndex)>;
```

Later async scoring will likely introduce a persistent `PathRegistry`:

```rust
pub struct PathRegistry {
    paths: Vec<CompletePath>,
    path_by_transitions: HashMap<CompletePath, PathId>,
    paths_by_transition: HashMap<TransitionRef, Vec<PathId>>,
}
```

For this plan, add the registry only if it simplifies `WaveEngine`; otherwise keep `Vec<ScoredPath>` ID-based and defer persistent path registry behavior to the async-scoring plan.

---

### Task 1: Add ID Types and State Interner Tests

**Files:**
- Modify: `bitonic_codegen/src/transition_table.rs`
- Test: `bitonic_codegen/tests/transition_table.rs`

- [ ] **Step 1: Write failing state interner tests**

Add tests for:

```rust
#[test]
fn interning_same_state_returns_same_id() {
    let mut interner = StateInterner::new(4);
    let state = State::from_top_bottom([0, 1, 2, 3], [4, 5, 6, 7]);

    let first = interner.intern(state.clone());
    let second = interner.intern(state);

    assert_eq!(first, second);
    assert_eq!(interner.len(), 1);
}

#[test]
fn state_rejects_label_outside_declared_element_range() {
    let err = State::try_from_one_based_u64(4, &[1, 2, 3, 9], &[5, 6, 7, 8])
        .unwrap_err();

    assert!(err.contains("out of range"));
}
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test transition_table
```

Expected: compile failure because `StateInterner` and `State` do not exist yet.

- [ ] **Step 3: Implement `LaneLabel`, `StateId`, `State`, and `StateInterner`**

Add conversion helpers:

```rust
impl State {
    pub fn try_from_zero_based_u64(
        lanes_per_vector: usize,
        top: &[u64],
        bottom: &[u64],
    ) -> Result<Self, String>;

    pub fn try_from_one_based_u64(
        lanes_per_vector: usize,
        top: &[u64],
        bottom: &[u64],
    ) -> Result<Self, String>;

    pub fn to_zero_based_tuple_u64(&self, lanes_per_vector: usize) -> StateTuple;
    pub fn to_one_based_tuple_u64(&self, lanes_per_vector: usize) -> StateTuple;
}
```

`try_from_one_based_u64` must reject `0`, reject labels greater than `2 * lanes_per_vector`, and store `label - 1`. `try_from_zero_based_u64` must reject labels greater than or equal to `2 * lanes_per_vector`.

Keep `StateTuple` temporarily as a compatibility alias:

```rust
pub type StateTuple = (Vec<u64>, Vec<u64>);
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the same focused command.

---

### Task 2: Refactor TransitionTable Storage Behind Existing Public Behavior

**Files:**
- Modify: `bitonic_codegen/src/transition_table.rs`
- Test: `bitonic_codegen/tests/transition_table.rs`

- [ ] **Step 1: Write failing transition insertion tests**

Add tests for:

```rust
#[test]
fn transition_table_returns_stable_transition_index_for_same_state_pair() {
    let mut table = TransitionTable::new_with_lanes(1, 4);
    let input = table.intern_state_from_one_based(&([1, 2, 3, 4], [5, 6, 7, 8]));
    let output = State::try_from_one_based_u64(4, &[1, 2, 3, 4], &[5, 6, 8, 7]).unwrap();

    let first = table.add_transition_by_id(0, input, output.clone(), empty_gadget());
    let second = table.add_transition_by_id(0, input, output, another_gadget());

    assert_eq!(first.transition, second.transition);
    assert!(first.transition_was_new);
    assert!(!second.transition_was_new);
    assert_eq!(second.gadget_index, Some(GadgetIndex(1)));
}
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test transition_table transition_table_returns_stable_transition_index_for_same_state_pair
```

- [ ] **Step 3: Replace `StageData::transitions` map with transition array plus indexes**

Replace:

```rust
transitions: HashMap<TransitionKey, Vec<PermutationGadget>>
```

with:

```rust
transitions: Vec<TransitionRecord>,
transition_by_pair: HashMap<(StateId, StateId), TransitionIndex>,
transitions_by_input: HashMap<StateId, Vec<TransitionIndex>>,
transitions_by_output: HashMap<StateId, Vec<TransitionIndex>>,
```

Add:

```rust
pub struct TransitionInsertResult {
    pub transition: TransitionRef,
    pub transition_was_new: bool,
    pub gadget_was_new: bool,
    pub gadget_index: Option<GadgetIndex>,
}
```

Add main insertion API:

```rust
pub fn add_transition_by_id(
    &mut self,
    stage: usize,
    input: StateId,
    output: State,
    gadget: PermutationGadget,
) -> TransitionInsertResult;
```

`add_transition_by_id` must not re-intern the input; only intern the output state.

- [ ] **Step 4: Keep temporary compatibility accessors**

Add or preserve methods needed by existing callers:

```rust
pub fn add_transition(
    &mut self,
    stage: usize,
    input_state: &VectorState,
    output_state: &VectorState,
    gadget: PermutationGadget,
) -> bool;

pub fn transition(&self, reference: TransitionRef) -> &TransitionRecord;
pub fn transition_gadgets(&self, reference: TransitionRef) -> &[PermutationGadget];
pub fn state(&self, id: StateId) -> &State;
pub fn state_as_one_based_tuple(&self, id: StateId) -> StateTuple;
pub fn state_as_vector_state(&self, id: StateId) -> VectorState;
```

The compatibility `add_transition` may re-intern the input; live wave-engine code will stop using it in a later task.

- [ ] **Step 5: Run existing transition table tests**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test transition_table
```

Expected: pass.

---

### Task 3: Convert Path Discovery to TransitionIndex-Based CompletePath

**Files:**
- Modify: `bitonic_codegen/src/transition_table.rs`
- Test: `bitonic_codegen/tests/transition_table.rs`

- [ ] **Step 1: Write failing ID-based path discovery tests**

Add tests for:

```rust
#[test]
fn complete_path_stores_transition_indexes_not_state_tuples() {
    let mut table = three_stage_table();
    let terminal = table.transition_ref_for_one_based(2, &[...], &[...]).unwrap();

    let paths = table.trace_paths_ending_at(terminal, None, None);

    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].transitions().len(), 3);
    assert_eq!(paths[0].transition_at_stage(2), terminal.transition);
}
```

Also add a continuity test that resolves the path through the table and checks:

```text
transition[i].output == transition[i + 1].input
```

- [ ] **Step 2: Run focused path tests and verify they fail**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test transition_table complete_path_stores_transition_indexes_not_state_tuples
```

- [ ] **Step 3: Replace path aliases**

Replace:

```rust
pub type PathStep = (usize, StateTuple, StateTuple);
pub type CompletePath = Vec<PathStep>;
```

with:

```rust
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct CompletePath {
    transitions: Vec<TransitionIndex>,
}
```

Add methods:

```rust
impl CompletePath {
    pub fn new(transitions: Vec<TransitionIndex>) -> Self;
    pub fn transitions(&self) -> &[TransitionIndex];
    pub fn transition_at_stage(&self, stage: usize) -> TransitionIndex;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

- [ ] **Step 4: Rewrite path enumeration using indexes**

Use `transitions_by_input` and `transitions_by_output` instead of scanning cloned transition keys:

```rust
pub fn trace_paths_ending_at(
    &self,
    anchor: TransitionRef,
    max_paths: Option<usize>,
    exclude_paths: Option<&HashSet<CompletePath>>,
) -> Vec<CompletePath>;
```

Preserve temporary compatibility wrapper:

```rust
pub fn trace_paths_ending_with(
    &self,
    stage: usize,
    input_tuple: &StateTuple,
    output_tuple: &StateTuple,
    max_paths: Option<usize>,
    exclude_paths: Option<&HashSet<CompletePath>>,
) -> Vec<CompletePath>;
```

The wrapper resolves the state tuples to IDs, looks up the transition index, then calls `trace_paths_ending_at`.

- [ ] **Step 5: Run transition table tests**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test transition_table
```

Expected: pass.

---

### Task 4: Convert Scoring AssignedPath to Transition/Gadget IDs

**Files:**
- Modify: `bitonic_codegen/src/scoring.rs`
- Test: `bitonic_codegen/tests/scoring.rs`
- Test: `bitonic_codegen/tests/uica_scoring.rs`

- [ ] **Step 1: Write failing scoring tests**

Add tests that assert:

```rust
let assigned = scorer.assign_path_gadgets_k_best(&path, &table, 3);
assert_eq!(assigned[0].path(), &path);
assert_eq!(assigned[0].gadget_at_stage(0), GadgetIndex(0));
assert_eq!(
    assigned[0].selection_key(&table),
    vec![(TransitionRef { stage: 0, transition: TransitionIndex(0) }, GadgetIndex(0))]
);
```

- [ ] **Step 2: Run focused scoring tests and verify they fail**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test scoring --test uica_scoring
```

- [ ] **Step 3: Replace copied-state AssignedStep storage**

Replace or retire:

```rust
pub struct AssignedStep {
    stage: usize,
    input: StateTuple,
    output: StateTuple,
    gadget_index: usize,
    gadget: PermutationGadget,
}
```

with ID-based assigned path storage:

```rust
pub struct AssignedPath {
    path: CompletePath,
    gadgets: Vec<GadgetIndex>,
}
```

Add resolver helpers for boundary code:

```rust
impl AssignedPath {
    pub fn path(&self) -> &CompletePath;
    pub fn gadgets(&self) -> &[GadgetIndex];
    pub fn gadget_at_stage(&self, stage: usize) -> GadgetIndex;
    pub fn selection_key(&self) -> AssignedPathKey;
}
```

- [ ] **Step 4: Update `Scorer::assign_path_gadgets_k_best`**

For each stage in `CompletePath`, resolve:

```rust
let transition = table.transition(TransitionRef { stage, transition: transition_index });
let gadgets = &transition.gadgets;
```

Build K-best assignments as `Vec<GadgetIndex>` instead of cloned gadgets/states.

- [ ] **Step 5: Update `transition_table_from_assigned_paths`**

This helper currently builds an export table from assigned paths. Keep it working by resolving transitions and selected gadgets through the source table:

```rust
pub fn transition_table_from_assigned_paths(
    source_table: &TransitionTable,
    assigned_paths: &[AssignedPath],
) -> TransitionTable
```

- [ ] **Step 6: Run focused scoring tests**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test scoring --test uica_scoring
```

Expected: pass.

---

### Task 5: Update WaveEngine to Carry StateId on the Hot Path

**Files:**
- Modify: `bitonic_codegen/src/wave_engine.rs`
- Test: `bitonic_codegen/tests/wave_engine.rs`
- Test: `bitonic_codegen/tests/runtime.rs`

- [ ] **Step 1: Write failing wave-engine tests for input ID preservation**

Add a test that records a transition from a known input state and asserts the inserted transition uses the same `StateId` as the job input, without re-interning through a raw tuple.

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test wave_engine input_state_id_is_preserved_when_recording_transition
```

- [ ] **Step 3: Change job/input selection to use StateId**

Update job structs so the engine owns IDs:

```rust
struct SynthesisJob {
    index: usize,
    stage: usize,
    input: StateId,
    input_state: VectorState,
    candidate_index: usize,
}

struct SynthesisJobOutput {
    job: SynthesisJob,
    results: Vec<(VectorState, PermutationGadget)>,
}
```

The worker can still receive/cloned `VectorState`; the engine should use `job.input` when recording results.

- [ ] **Step 4: Update candidate state collection**

Change APIs equivalent to `inputs_for_stage` and `get_unforwarded_outputs` to return `StateId`s plus resolved `VectorState` only when building worker jobs.

- [ ] **Step 5: Update transition recording**

Change `record_transition` to:

```rust
pub fn record_transition(
    &mut self,
    stage: usize,
    input: StateId,
    output_state: &VectorState,
    gadget: PermutationGadget,
) -> TransitionRecordResult
```

Use:

```rust
let insert = self.transition_table.add_transition_by_id(stage, input, output, gadget);
```

Preserve current scoring trigger behavior:

```rust
if insert.gadget_was_new && stage + 1 == self.stages.len() {
    self.discover_paths_for_transition(insert.transition, None)
}
```

- [ ] **Step 6: Update `discover_paths_for_transition`**

Change from tuple arguments to `TransitionRef`:

```rust
pub fn discover_paths_for_transition(
    &mut self,
    transition: TransitionRef,
    max_paths: Option<usize>,
) -> usize
```

Use `trace_paths_ending_at`.

- [ ] **Step 7: Run focused wave/runtime tests**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test wave_engine --test runtime
```

Expected: pass.

---

### Task 6: Update Lowering, Export, Import, and Verification Boundaries

**Files:**
- Modify: `bitonic_codegen/src/instruction_stream.rs`
- Modify: `bitonic_codegen/src/json_exporter.rs`
- Modify: `bitonic_codegen/src/json_importer.rs`
- Modify: `bitonic_codegen/src/asm_exporter.rs`
- Modify: `bitonic_codegen/src/verifier.rs`
- Test: `bitonic_codegen/tests/instruction_stream.rs`
- Test: `bitonic_codegen/tests/json_exporter.rs`
- Test: `bitonic_codegen/tests/json_importer.rs`
- Test: `bitonic_codegen/tests/asm_exporter.rs`

- [ ] **Step 1: Write failing boundary tests for ID-based paths**

For each boundary, add or update one test where:

```rust
let paths = table.trace_paths_ending_at(terminal_ref, None, None);
let assigned = scorer.assign_path_gadgets_k_best(&paths[0], &table, 1);
```

Then verify lowering/export/import/ASM/verifier still sees the expected concrete states and selected gadgets.

- [ ] **Step 2: Run focused boundary tests and verify they fail**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test instruction_stream --test json_exporter --test json_importer --test asm_exporter
```

- [ ] **Step 3: Add transition/path resolver helpers**

Add helpers to `TransitionTable`:

```rust
pub struct ResolvedPathStep<'a> {
    pub stage: usize,
    pub transition: TransitionRef,
    pub input: StateId,
    pub output: StateId,
    pub input_state: StateTuple,
    pub output_state: StateTuple,
    pub gadgets: &'a [PermutationGadget],
}

pub fn resolve_complete_path(&self, path: &CompletePath) -> Vec<ResolvedPathStep<'_>>;
pub fn resolve_assigned_path(&self, path: &AssignedPath) -> Vec<ResolvedAssignedStep<'_>>;
```

These helpers are allowed to allocate compatibility `StateTuple`s because they run at scoring/export/verify boundaries, not in path storage.

- [ ] **Step 4: Update instruction lowering**

Change lowering to iterate resolved steps instead of destructuring `(stage, input, output)` from the path.

- [ ] **Step 5: Update JSON and ASM export**

Keep the external JSON format unchanged for this work item. Convert zero-based internal labels back to the expected one-based JSON labels at export.

- [ ] **Step 6: Update JSON import**

Import one-based JSON labels into zero-based internal states through `State::try_from_one_based_u64`.

- [ ] **Step 7: Update verifier**

Verifier can keep using one-based `StateTuple` internally at first. Resolve ID paths to one-based tuples before building registers.

- [ ] **Step 8: Run focused boundary tests**

Run the same focused command. Then run:

```bash
cargo test --release -q -p bitonic_codegen --test verifier
```

Expected: pass.

---

### Task 7: Update Full Scoring and Final Export Flow

**Files:**
- Modify: `bitonic_codegen/src/lib.rs`
- Modify: `bitonic_codegen/src/uica_scoring.rs`
- Test: `bitonic_codegen/tests/uica_scoring.rs`
- Test: `bitonic_codegen/tests/cli.rs`

- [ ] **Step 1: Write failing final-scoring regression tests**

Add or update tests to assert that:

```text
rough candidate count is unchanged for an existing small solve
full scoring still lowers exactly the selected assigned path
exported JSON remains compatible with existing importer tests
```

- [ ] **Step 2: Run focused final-scoring tests and verify they fail**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test uica_scoring --test cli
```

- [ ] **Step 3: Update `UiPackScorer::score_assigned_path`**

Pass the source table into rough scoring if needed:

```rust
fn score_assigned_path(&self, table: &TransitionTable, assigned_path: &AssignedPath) -> PathCost;
```

If changing the `Scorer` trait is too broad, add a separate method on `UiPackScorer` and keep the trait method only where table-free dummy scoring is still possible.

- [ ] **Step 4: Update `full_score_and_prune_paths`**

Use ID-based `ScoredPath` and resolve selected gadgets through the table when building the one-path export table.

- [ ] **Step 5: Run focused final-scoring tests**

Run:

```bash
cargo test --release -q -p bitonic_codegen --test uica_scoring --test cli
```

Expected: pass.

---

### Task 8: Remove Legacy Tuple-Based Hot-Path APIs

**Files:**
- Modify: `bitonic_codegen/src/transition_table.rs`
- Modify: `bitonic_codegen/src/scoring.rs`
- Modify: `bitonic_codegen/src/wave_engine.rs`
- Test: all relevant Rust tests

- [ ] **Step 1: Search for remaining hot-path tuple APIs**

Run:

```bash
rg -n "PathStep|TransitionKey|StateTuple|trace_paths_ending_with|get_all_transitions|get_transitions|as_complete_path|selection_key\\(" bitonic_codegen/src bitonic_codegen/tests -g '*.rs'
```

- [ ] **Step 2: Decide which compatibility APIs stay**

Allowed compatibility uses:

```text
JSON import/export
instruction comments
verifier concrete-state construction
tests that explicitly check external JSON shape
```

Disallowed hot-path uses:

```text
WaveEngine transition recording
CompletePath storage
AssignedPath storage
AssignedPathKey dedupe
path discovery recursion
rough top-k retention
```

- [ ] **Step 3: Remove or narrow legacy APIs**

Remove `TransitionKey` if no longer needed. Keep `StateTuple` only as a boundary alias, preferably documented:

```rust
/// Compatibility state representation used only at JSON/verifier/comment boundaries.
pub type StateTuple = (Vec<u64>, Vec<u64>);
```

- [ ] **Step 4: Re-run the search**

The remaining matches should be explainable boundary uses.

---

### Task 9: Verification

**Files:**
- Rust workspace only.

- [ ] **Step 1: Format check**

Run:

```bash
cargo fmt --all --check
```

Expected: pass.

- [ ] **Step 2: Clippy**

Run:

```bash
cargo clippy --all-targets --all-features --release -- -D warnings
```

Expected: pass.

- [ ] **Step 3: Rust test suite**

Run:

```bash
cargo test --release -q -p bitonic_codegen
```

Expected: pass in about the same time as the current release-mode package tests, under the repository's one-minute guideline.

- [ ] **Step 4: Focused solve smoke test**

Run a small solve to verify runtime behavior:

```bash
./target/release/bitonic_codegen solve \
  --vector-machine AVX2 \
  --datatype i64 \
  --wave-attempts 20 \
  --wave-outputs 10 \
  --workers 2 \
  --max-gadget-solutions 3 \
  --top-k 2 \
  --target-cpu SKL \
  --runtime-trace /tmp/vxsort-id-storage-trace.jsonl \
  --output-path /tmp/vxsort-id-storage-output.json
```

Expected:

```text
solve completes
trace contains worker/job events
JSON output imports/verifies through existing tests or verifier command
```

Do not run Python tests/tooling for this Rust-only storage refactor.

---

## Explicit Non-Goals

- Do not make rough scoring asynchronous in this plan.
- Do not make full uiCA scoring asynchronous in this plan.
- Do not change the external JSON schema unless an existing test proves it is already incompatible.
- Do not hard-prune transitions or complete paths based on rough scores.
- Do not convert `TransitionTable` to const generics in this plan; keep that as a later, smaller cleanup once ID-based storage is passing.

## Follow-Up Plan

After this lands, write the async scoring plan around:

```rust
PathRegistry {
    paths,
    path_by_transitions,
    paths_by_transition,
}
```

and two work event types:

```rust
NewCompletePath(PathId)
NewGadgetForExistingTransition { transition: TransitionRef, gadget: GadgetIndex }
```

That follow-up should make online scoring best-effort and keep a final catch-up pass authoritative.
