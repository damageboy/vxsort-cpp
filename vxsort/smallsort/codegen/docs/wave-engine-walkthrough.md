# Wave Engine Walkthrough — Top to Bottom

## The Problem: Why This Engine Exists

You have a bitonic sorting network with N stages. Each stage has comparison pairs like `(1,5), (2,6), (3,7), (4,8)`. The goal is to find SIMD instruction sequences ("gadgets") that implement each stage's compare-swap pattern.

But there's a catch: the output permutation of stage S becomes the input to stage S+1, and a gadget at stage S might produce multiple valid outputs — the elements are correctly sorted, but lanes can be in different orders. So the search space is a **multi-stage state machine**: you need to find a complete path through all stages where outputs chain to inputs.

A naive sequential approach (solve stage 0 fully, then stage 1...) is terrible because:

1. You don't know which stage-0 outputs will lead to cheap stage-1 solutions
2. The bottleneck stage isn't always the first one
3. You want scored end-to-end paths as early as possible

The wave engine solves this with **iterative adaptive bottleneck targeting**.

**Source**: [`bitonic_sorter.py`](../src/bitonic_sorter.py)

---

## Initialization

> **Source**: [`wave_engine.py` — `__init__`](../src/wave_engine.py#L187-L266)

Sets up the entire synthesis infrastructure:

1. **Compute element layout**: `elements_per_vector = width / type_bits`, `total_elements = num_vecs × elements_per_vector`
2. **Build bitonic stages**: `BitonicSorter(total_elements)` generates comparison pairs per stage. If `natural_order` is set, appends a final reordering stage.
3. **Pre-compute candidate gadget graphs**: split into shallow (depth ≤ 1, fast to solve) and deep (depth ≥ 2, finds solutions shallow can't) tiers
4. **Initialize TransitionTable**: one `StageData` per bitonic stage — the central accumulator for all discoveries
5. **Retroactive input** (optional): pre-populate stage 0 with all N! permutations of comparison pairs as zero-cost identity transitions. This is a search space expansion trick — instead of starting from one fixed input, every possible lane arrangement is a valid starting point, and the engine retroactively picks the cheapest one.

### BitonicSorter — Stage Pairs

> **Source**: [`bitonic_sorter.py`](../src/bitonic_sorter.py)

Generates the bitonic sorting network's comparison pairs. For `total_elements` inputs, produces a sequence of stages where each stage contains `(i, j)` pairs indicating which positions must be compare-swapped. Example for 8 elements (2×4 AVX2 i64): produces ~6 stages.

The wave engine treats each stage as a separate synthesis target — it must find SIMD instruction sequences (gadgets) that implement the compare-swap pattern for that stage, given whatever input permutation the previous stage produced.

### GadgetSynthesizer — Candidate Graph Pre-computation

> **Source**: [`gadget_synthesizer.py`](../src/gadget_synthesizer.py) — [`precompute_candidates_stratified`](../src/gadget_synthesizer.py#L1409-L1421)

Pre-computes all candidate gadget graphs — instruction templates with symbolic (Z3 variable) immediates.

**Two Tiers**:
- **Shallow (depth ≤ 1)**: Identity + single-instruction per side. Fast Z3 solves. Tried first.
- **Deep (depth ≥ 2)**: Multi-instruction chains, mux-wired inputs, shared-prefix graphs where both sides share a common instruction then diverge. Slower but find solutions shallow can't.

**Graph Structure** — each `GadgetGraph` has a `top` and `bottom` `IntrinsicNode` tree:
- `InputRef`: register inputs (top_vec, bottom_vec)
- `Symbolic`: Z3 variables for immediates (imm8, masks, control indices)
- `Mux`: choice between multiple sources (enables depth-2 wiring)
- `IntrinsicNode`: SIMD intrinsic call with operands

Candidates depend only on `gadget_depth` and available intrinsics — they're computed once and reused across all stages and waves.

### TransitionTable — The Central Accumulator

> **Source**: [`transition_table.py`](../src/transition_table.py)

The core data structure that accumulates all discovered transitions across all waves. Think of it as a stage-indexed state machine being built incrementally.

```
TransitionTable
  └── stages[i]: StageData
        └── transitions: {(input_tuple, output_tuple) → [gadget1, gadget2, ...]}
        └── unique_outputs: {output_tuple → VectorState}
        └── attempted_pairs: {(input_tuple, candidate_idx)}  # dedup
        └── forwarded_outputs: {output_tuple}  # already sent downstream
```

**Why multiple gadgets per transition?** The same (input → output) transition may be achievable by different instruction sequences. They're all stored as alternatives for later **gadget assignment optimization** — hill climbing picks the best one per stage when scoring complete paths.

**Key insight**: The TransitionTable decouples discovery from path selection. Waves add transitions; path scoring happens separately by tracing complete chains through the table.

---

## The Wave Loop

> **Source**: [`wave_engine.py` — `run()`](../src/wave_engine.py#L966-L1239)

The main loop runs iterative waves until termination (Ctrl-C, max_waves, or all stages exhausted).

### What Happens Each Wave

```
1. Pick target stage (weakest / bottleneck)
2. Generate jobs for target stage
3. Run target stage with budget (attempts + outputs limits)
4. Track zero-output / unproductive counters
5. Depth-first propagation: cascade through ALL downstream stages
6. Drain remaining in-flight jobs
7. Checkpoint
8. Log summary (outputs, propagation, scored paths, best scores)
9. wave_count++
```

**Why not just run all stages equally?** Because stages have vastly different difficulty. Stage 3 might have 2 outputs while stage 1 has 200. Pouring more work into stage 1 is wasteful — stage 3 is the bottleneck. The wave loop concentrates compute where it matters most, then cascades results through the rest of the pipeline.

**Wave 0 special case**: Always targets stage 0 to seed the pipeline. After that, adaptive targeting takes over.

---

## Target Selection — Finding the Weakest Link

> **Source**: [`wave_engine.py` — `_select_target_stage`](../src/wave_engine.py#L764-L832)

Each wave targets the **weakest stage** — the bottleneck in the pipeline.

### Selection Criteria (priority order)

1. **Fewest distinct outputs** — the stage constraining the most complete paths
2. **Lowest success rate** (outputs / attempts) — tiebreaker
3. **Earliest stage index** — final tiebreaker

### The Thinking

Adding outputs to the bottleneck stage creates the most new complete paths. If stage 3 has only 4 outputs while every other stage has 50+, one new stage-3 output can combine with 50 × 50 × ... paths from other stages. That's maximum leverage.

### Two Override Mechanisms

#### Bubble-Up — When a Stage Needs Upstream Help

> **Source**: [`wave_engine.py`](../src/wave_engine.py#L783-L802)

Sometimes a stage is stuck not because it's inherently hard, but because it doesn't have enough **diverse inputs**. The compare-swap gadgets that exist for stage 3's current inputs might simply not produce new outputs — but if stage 2 found a new output permutation, that new input to stage 3 might unlock entirely new solutions.

**Trigger conditions**:
- `consecutive_zero_budgets ≥ 2` — two waves produced zero new outputs at this stage
- `unproductive_waves ≥ 3` — three waves with no new end-to-end paths

**Action**: Target the nearest non-exhausted **upstream** stage instead. The hope: new upstream outputs create fresh inputs that break the downstream logjam.

If all upstream stages are exhausted too, the stuck stage gets marked exhausted — there's genuinely nothing more to find.

#### Unproductive Penalty — Avoiding Global Dead Ends

> **Source**: [`wave_engine.py` — `_pick_target`](../src/wave_engine.py#L804-L832)

A stage might keep producing local outputs (new transitions) that **never lead to new end-to-end paths**. This is globally useless even though it looks locally productive.

A wave targeting stage S is "unproductive" if it produces:
- Zero new last-stage outputs, AND
- Zero new scored paths

Each unproductive wave inflates the stage's apparent output count by `+10`:

```python
effective_outputs = actual_outputs + (unproductive_waves × 10)
```

This makes the stage look less "weak" so other stages get targeted instead. The counter resets when the stage finally produces downstream progress.

**Why not just skip it?** Because the stage might become productive again after upstream stages produce new inputs (via bubble-up). The penalty is a soft deprioritization, not a hard exclusion.

---

## Budget System — Controlling Compute Per Wave

> **Source**: [`wave_engine.py` — `_run_stage_in_wave`](../src/wave_engine.py#L878-L945)

Each wave has **two per-stage budgets**, whichever is hit first ends the stage:

- **`wave_attempts`** (default 10,000): max Z3 solver invocations. Controls raw compute cost.
- **`wave_outputs`** (default 100): max new distinct outputs. Prevents one prolific stage from dominating a wave.

### Why Two Budgets?

Attempts alone would let a high-success-rate stage produce thousands of outputs in one wave, starving downstream stages of pool capacity. Output budgets ensure the wave moves on and cascades results downstream.

Attempt budgets alone would let a low-success-rate stage burn 10k Z3 calls finding nothing. Together they create a balanced "spend enough but don't over-invest" policy.

### Budget Enforcement

`_submit_jobs` checks `(attempts_done + in_flight - wave_start) < budget` before submitting each job. Jobs beyond budget stay in the pending deque for the next wave.

---

## Job Generation and Execution

### Job Generation — The Cross-Product

> **Source**: [`wave_engine.py` — `_make_jobs`](../src/wave_engine.py#L311-L357), [`_generate_jobs_for_stage`](../src/wave_engine.py#L363-L409)

For a target stage, jobs are the **cross-product** of:
- **Input states**: stage 0 uses `initial_state` (or N! permutations if retroactive); stage N uses `unique_outputs` from stage N-1
- **Candidate graphs**: all pre-computed gadget templates (shallow + deep)

Each `(input_state, candidate_graph)` pair becomes one Z3 solver job.

**Budget-aware generation**: With N! retroactive inputs × hundreds of candidates, the cross-product can be millions of jobs. `_make_jobs` accepts a `limit` parameter and stops collecting once the budget is reached.

**Special cases**:
- Final natural-order stage: `allow_any_lane_order = False` — output must be in exact positional order
- No input states available + predecessors not exhausted → stage is "stalled" (will retry when predecessors produce more)

### Attempted Pair Tracking — Never Repeat Work

> **Source**: [`transition_table.py` — `was_attempted`](../src/transition_table.py#L126-L130)

Every `(input_state_tuple, candidate_index)` combination that has been submitted to Z3 is recorded in the TransitionTable's `attempted_pairs` set. When generating jobs for a new wave, `_make_jobs` checks and skips already-tried combinations.

**Exhaustion detection**: A stage is "exhausted" when every (input × candidate) combination has been tried AND all predecessor stages are also exhausted (no new inputs will ever appear). This is how the search terminates naturally.

### Process Pool — Parallel Z3 Synthesis

> **Source**: [`wave_engine.py`](../src/wave_engine.py#L999-L1002)

A single long-lived `multiprocessing.Pool` shared across the entire run.

```python
pool = Pool(processes=num_workers, maxtasksperchild=max_tasks_per_child)
pool_target = num_workers × 2  # 2× over-subscription keeps pool fed
```

**Why a shared pool?** Creating/destroying pools per wave or per stage would waste process startup time. A single pool amortizes that cost. Jobs from different stages coexist — `_submit_jobs` prioritizes lower stage indices so upstream work gets pool slots first.

**Memory management**: `maxtasksperchild` recycles workers after N tasks, limiting Z3 memory growth in long runs.

### Z3 Worker — The Actual Synthesis

> **Source**: [`gadget_synthesizer.py` — `_validate_gadget_worker`](../src/gadget_synthesizer.py#L1444-L1502), [`z3_avx.py`](../src/z3_avx.py)

Each worker receives a job tuple and runs in its own subprocess with a fresh Z3 context.

1. Create a fresh `GadgetSynthesizer(vm, prim_type)` with its own Z3 context
2. Call `synthesize_gadget_with_symbolic(graph, input_state, target_pairs, ...)`:
   - Build Z3 bitvector expressions from the gadget graph
   - `InputRef` → concrete values from input_state
   - `Symbolic` → Z3 `BitVec` variables (the unknowns)
   - `IntrinsicNode` → Z3 function calls modeling SIMD semantics
   - Add sorting constraints: for each `(i,j)` pair, assert `output[i] ≤ output[j]`
   - `s.check()` — if SAT, extract concrete immediate values
3. Enumerate up to `max_unique_outputs` distinct output permutations via `_all_smt()`
4. Reject gadgets with > 4 unified instructions (after CSE deduplication)
5. Return `(gadget_results, input_state, metadata, construction_time, solver_time)`

---

## Result Processing

### Result Draining

> **Source**: [`wave_engine.py` — `_drain_results`](../src/wave_engine.py#L514-L593)

The main thread periodically drains the completion queue between job submissions.

For each successful result:
1. Record `(input_tuple, candidate_index)` as attempted
2. For each `(gadget, output_state)` in the results:
   - Is this output **new** (not in `unique_outputs`)?
   - `tt.add_transition(stage, input, output, gadget)` — deduplicated by gadget sort key
   - If new: **eagerly enqueue downstream** jobs
   - If new AND last stage: **trigger on-the-fly scoring**
3. Update progress bar

**Why drain frequently?** Keeps the pool fed. If we waited until all jobs complete, the pool would idle while we process results. Interleaving submission and draining maximizes throughput.

### Eager Downstream Feeding — Keep the Pool Busy

> **Source**: [`wave_engine.py` — `_enqueue_downstream`](../src/wave_engine.py#L411-L430)

When a new unique output appears at stage S during result draining, the engine **immediately** generates jobs for stage S+1 using that output as input and adds them to the pending deque.

This is a throughput optimization: while slow Z3 "straggler" jobs at the target stage are still running, the pool can work on downstream jobs from earlier discoveries. Without this, pool workers would sit idle waiting for the target stage to finish before propagation begins.

The formal depth-first propagation loop later gives each downstream stage its own budget — `_make_jobs` skips already-attempted pairs, so eagerly-processed work is naturally accounted for and never double-counted.

### Depth-First Propagation — Cascade Through the Pipeline

> **Source**: [`wave_engine.py`](../src/wave_engine.py#L1092-L1144)

After the target stage's budget is exhausted, the wave cascades results through **every downstream stage**, each with its own fresh budget.

```python
for stage_idx in range(target + 1, last_stage + 1):
    # Reset wave-start snapshot → fresh budget
    # Collect unforwarded outputs from stage_idx - 1
    # Generate jobs from those outputs
    # _run_stage_in_wave(stage_idx) with full budget
```

**Why depth-first?** So that new discoveries at the target stage cascade all the way to the last stage **within the same wave**. This enables immediate end-to-end scoring — you see whether the target stage's new outputs actually lead to good complete paths, right away.

**The forwarding mechanism**: Each stage tracks which outputs have been "forwarded" to the next stage. `get_unforwarded_outputs()` returns only new, unseen outputs. `mark_forwarded()` prevents re-processing. This ensures each output is propagated exactly once.

---

## On-the-Fly Scoring

> **Source**: [`wave_engine.py` — `_score_new_paths_for_transition`](../src/wave_engine.py#L599-L634)

Rather than waiting until the entire search is done, the engine scores complete paths **the moment they form**.

When a new last-stage transition appears during result draining:
1. `trace_paths_ending_with(stage, input, output)` — backwards DFS to find all complete paths through that transition
2. Filter out already-scored paths (tracked by path key)
3. For each new path: hill-climb gadget assignment, generate ASM, run LLVM-MCA
4. Update `best_scores` and display in progress bar

**Why on-the-fly?**
- **Early feedback**: you see if the search is converging or needs strategy changes
- **No wasted effort**: scoring only new paths avoids re-scoring the entire TransitionTable
- **Incremental**: each new last-stage transition only traces paths through *that* transition, not all possible paths

### Backwards DFS — Efficient Path Discovery

> **Source**: [`transition_table.py` — `trace_paths_ending_with`](../src/transition_table.py#L307-L372)

Given a terminal transition `(stage_last, input, output)`, traces backwards through the TransitionTable to find every complete path ending with that transition.

**Why backwards?** Forward enumeration (`enumerate_complete_paths`) is O(all paths). Backwards from a specific transition is O(paths through that transition) — vastly smaller. This makes on-the-fly scoring practical even when the TransitionTable has millions of possible paths.

### Hill Climbing — Gadget Assignment Optimization

> **Source**: [`wave_engine.py` — `hill_climb_path`](../src/wave_engine.py#L57-L105)

Each transition in a path may have multiple alternative gadgets. Hill climbing finds the best combination.

```python
def hill_climb_path(path, tt, scorer, max_passes=3):
    # 1. Initial: pick gadget with fewest instructions at each stage
    # 2. For each stage, try all alternative gadgets for that transition
    # 3. If any swap improves the total score → accept, restart from stage 0
    # 4. Converge when no improvement found in a full pass
    return (best_assignments, best_score)
```

**Why hill climbing?** Exhaustive search over all gadget combinations is exponential (∏ alternatives per stage). Hill climbing is fast and finds good solutions — the gadget alternatives are typically similar in cost, so the optimum is usually within 1-2 swaps of the greedy initial assignment.

### LLVM-MCA — Cycle-Accurate Performance Estimation

> **Source**: [`wave_engine.py` — `_score_complete_paths`](../src/wave_engine.py#L636-L758), [`llvm_mca_runner.py`](../src/llvm_mca_runner.py), [`perf_estimator.py`](../src/perf_estimator.py)

Estimates real hardware performance for scored paths:

1. `generate_solution_asm()` — convert gadget assignments to x86 assembly
2. `sanitize_asm_for_llvm_mca()` — clean up for MCA consumption
3. `run_llvm_mca(asm, mcpu, mca_bin)` — run `llvm-mca --mcpu=<target>` for each target CPU
4. Parse throughput (IPC) and simulated cycles

Runs multiple MCA invocations in parallel via `ThreadPoolExecutor`. Supports multiple target CPUs (e.g., ZEN4, ADL-P) simultaneously.

**Why MCA, not just instruction count?** Instruction count is a poor proxy for performance. A 4-instruction path using port-conflicting shuffles can be slower than a 5-instruction path with better port distribution. MCA simulates the actual microarchitecture.

### Best Scores — Live Per-CPU Tracking

Tracks the best-performing path for each target CPU across the entire run:

```python
self.best_scores = {
    "ZEN4": {"throughput": 0.42, "cycles": 18.5, "path_key": [...]},
    "ADL-P": {"throughput": 0.38, "cycles": 21.0, "path_key": [...]},
}
```

Displayed live in the progress bar:
```
Best: ZEN4: 18.5cy (throughput 0.42)  |  ADL-P: 21.0cy (throughput 0.38)
```

---

## Checkpoint & Resume — Never Lose Work

> **Source**: [`wave_engine.py` — `_save_checkpoint`](../src/wave_engine.py#L838-L872), [`wave_checkpoint.py`](../src/wave_checkpoint.py), [`wave_engine.py` — `resume`](../src/wave_engine.py#L1245-L1301)

After every stage completion within a wave, the engine saves state incrementally.

### What's Saved

```
checkpoint_dir/
    master.json              # Config, wave count, best scores, stage stats
    stage_00.json.zst        # Per-stage transitions + bookkeeping (only if dirty)
    stage_01.json.zst
    ...
    scored_paths.json.zst    # All LLVM-MCA scoring results
```

- **master.json** (plain JSON, always rewritten): config, wave count, best scores, stage stats, exhausted stages
- **stage_NN.json.zst** (only if dirty): all transitions, attempted pairs, forwarded outputs, bookkeeping counters
- **scored_paths.json.zst**: all LLVM-MCA results

### Atomic Writes

All files written via `temp file + os.replace()` — crash-safe, no corruption.

### Resume Flow

1. Load `master.json` → restore `wave_count`, `exhausted_stages`
2. Load each stage file → replay transitions via `tt.add_transition()` for proper bookkeeping
3. Load `scored_paths.json.zst` → rebuild `_scored_path_keys` and `best_scores`
4. Seed pending queues from unforwarded outputs
5. Continue from exactly where it left off

### Ctrl-C Handling

SIGINT is caught gracefully — the engine finishes current in-flight jobs, saves a final checkpoint, then exits.

---

## Summary

The wave engine is essentially a **multi-stage breadth-first search with adaptive bottleneck targeting**. Instead of solving stages sequentially, it repeatedly asks "which stage is the weakest?" and pours compute into that stage, then cascades the results downstream. The bubble-up heuristic handles the case where a stage needs upstream diversity rather than more local compute. The unproductive penalty prevents wasting effort on locally productive but globally useless stages. The result is that complete end-to-end paths emerge as early as possible, and the engine converges on the best-performing instruction sequences without wasting effort on non-bottleneck stages.
