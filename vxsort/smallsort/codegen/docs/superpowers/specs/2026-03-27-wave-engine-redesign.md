# Wave Engine Redesign: Continuous Pool Feeding + Incremental Checkpointing & Scoring

## Context

The wave-atomic engine at commit c159b2d gates checkpointing, scoring, and pool lifecycle on wave completion. For large configs (AVX512/i32, 24h+ per wave) this means: no crash safety, no visibility into solution quality, idle CPUs during Z3 stragglers, and broken score persistence (overwritten not accumulated, never reloaded on resume).

We reset to c159b2d (pre-pipelining) and redesign the wave engine to fix all four problems while keeping the wave concept as the unit of targeting and budgeting.

## Design

### 1. Wave Structure (unchanged)

Waves remain. Each wave:
1. Selects target stage (weakest, with bubble-up — existing `_select_target_stage`)
2. Generates jobs for the target stage
3. Submits to the shared pool, drains results
4. As results produce new outputs, downstream jobs are enqueued immediately (replaces the old batch `_forward_propagate`)
5. Wave ends when target stage and all downstream overflow work complete
6. Increments wave count

The pool is long-lived across waves. Checkpointing happens per-stage-completion within the wave. Scoring happens on the fly as last-stage transitions form complete paths.

### 2. Single Long-Lived Pool with Straggler Feeding

One `multiprocessing.Pool` created in `run()`, destroyed in `finally`. `maxtasksperchild` still recycles workers.

Within a wave:
- Jobs for the target stage are submitted in batches (`batch_size = workers * 2`)
- Results drained from a `queue.Queue` via callbacks
- When a result at stage N produces a new output, jobs for stage N+1 are generated immediately for that single new input and appended to a per-stage pending deque
- When topping up the pool: iterate stages in order — if the target stage's deque is empty (stragglers in-flight), submit from the next stage's deque
- When the target stage's last in-flight job completes, that stage is done. Downstream stages started as overflow continue until drained. Then the wave ends.

No geometric thresholds, no forwarding sets. Just "submit from the earliest stage with pending work."

### 3. Checkpoint on Stage Completion

When a stage finishes within a wave (pending deque empty AND in-flight == 0):
1. Save that stage to `stage_NN.json.zst` (transitions, attempted_pairs, forwarded_outputs, attempts)
2. Save `master.json` (stage stats, best_scores, wave_count)
3. Save accumulated scored paths to `scored_paths.json.zst`
4. Clear dirty flag

On Ctrl-C: save all dirty stages immediately.

### 4. Scoring on the Fly

When a result at the **last stage** produces a new transition:
1. Trace backwards from the new transition's input through stages N-1 → 0 to find complete paths ending with this transition
2. Filter out already-scored paths (`_scored_path_keys`)
3. For each new complete path:
   - Hill-climb gadget assignment (existing `hill_climb_path`)
   - Generate ASM (`generate_solution_asm`)
   - Run LLVM-MCA for each target CPU (subprocess, stdin/stdout)
   - Update `best_scores`, add to `_all_scored_paths`, add key to `_scored_path_keys`
   - Log the result

This is inline in the result drain loop. LLVM-MCA is a subprocess that doesn't compete with the Z3 worker pool. The completion queue accumulates while scoring runs.

### 5. Score Persistence

- `_all_scored_paths: list[dict]` accumulates across the entire run
- Each entry: `{path_key: <tuple of steps>, scores: {cpu: {throughput, cycles}}}`
- `path_key` is the stable `tuple((stage, input_tuple, output_tuple) ...)` — no positional indices
- `best_scores` references `path_key` instead of `path_index`
- On each checkpoint: `save_scored_paths(self._all_scored_paths)` — atomic overwrite of the complete accumulated set
- `path_index` is dropped entirely

### 6. Resume

`resume(checkpoint_dir)`:
1. Load `master.json` — wave_count, exhausted_stages, target_cpus
2. Load each `stage_NN.json.zst` — transitions, attempted_pairs, forwarded_outputs, attempts
3. Load `scored_paths.json.zst` → `_all_scored_paths` (return `[]` if missing for backwards compat)
4. Rebuild `_scored_path_keys` from loaded scored paths
5. Rebuild `best_scores` from `_all_scored_paths` (authoritative — ignore master.json's best_scores)
6. Restore `_stage_attempts` from TransitionTable

Then in `run()` before main loop:
7. Seed downstream pending deques from unforwarded outputs
8. Generate jobs for targeted stage
9. Enter normal wave loop

### 7. CLI Changes

- `--llvm-mca-path` renamed to `--llvm-mca-cmd` — accepts a full command line (e.g. `docker run -i silkeh/clang /usr/bin/llvm-mca`), split via `shlex.split`, invoked with stdin/stdout piping. Old flag kept as deprecated alias.

## Files to Modify

### `src/wave_engine.py`

**Delete:**
- `_run_stage_budget` — replaced by shared pool + drain loop
- `_forward_propagate` — replaced by immediate `_enqueue_downstream` on each result
- `run_wave` — absorbed into `run()`

**New methods:**
- `_submit_jobs(pool, queue, target)` — fill pool from earliest stage with pending work
- `_drain_results(queue, progress, task_ids)` — process results, enqueue downstream, trigger scoring
- `_enqueue_downstream(stage_idx, output_state)` — generate jobs for stage N+1 from new output
- `_generate_jobs_for_stage(target)` — budget-limited job generation
- `_enqueue_from_unforwarded(stage_idx)` — seed downstream jobs on resume
- `_score_new_paths_for_transition(stage_idx, input_tuple, output_tuple)` — backwards trace + score
- `_rebuild_best_scores_from_accumulated()` — rebuild best_scores from all scored paths

**Modified methods:**
- `__init__` — add `_pending_jobs` (per-stage deques), `_in_flight` (per-stage counters), `_all_scored_paths`, `_checkpoint_dirty`
- `run()` — rewritten: long-lived pool, wave loop with shared pool feeding
- `_save_checkpoint()` — includes accumulated scored paths
- `resume()` — loads scored paths, rebuilds `_scored_path_keys` and `best_scores`
- `_score_complete_paths()` — uses stable `path_key`, called from on-the-fly scoring

**Kept unchanged:**
- `_create_initial_state`, `_make_jobs`, `_select_target_stage`
- `hill_climb_path`, `_build_complete_path` (module-level)
- `export_to_solution_nodes`

### `src/wave_checkpoint.py`

- `load_scored_paths()`: return `[]` on missing file instead of raising
- No format change — caller passes full accumulated list

### `src/bitonic_compiler.py`

- `--llvm-mca-path` → `--llvm-mca-cmd` (with deprecated alias)

### `src/llvm_mca_runner.py`

- `find_llvm_mca` and `run_llvm_mca` updated to accept a command line string, split via `shlex.split`

### `src/transition_table.py`

- Add `trace_paths_ending_with(stage, input_tuple, output_tuple)` — backwards trace from a last-stage transition to enumerate complete paths

## Verification

1. `uv run pytest` — existing tests pass
2. Quick integration: `uv run python src/bitonic_compiler.py --vector-machine AVX2 --datatype i64 --depth-limit 3 --gadget-depth 1 --top-k 5`
   - Checkpoints appear after each stage completes within a wave
   - Scores appear as soon as last-stage transitions form complete paths
   - Ctrl-C saves all dirty stages
   - Resume from checkpoint continues correctly with scores intact
3. `uv run ruff check .`
4. `uv run vulture`
