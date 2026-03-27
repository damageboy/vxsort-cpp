# Wave Engine Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the wave-atomic engine with continuous pool feeding, per-stage-completion checkpointing, and on-the-fly LLVM-MCA scoring.

**Architecture:** A single long-lived `multiprocessing.Pool` shared across the entire run. Waves still drive stage targeting and budgeting. Per-stage pending deques feed the pool in stage-priority order, solving the straggler problem. Checkpoints fire when a stage completes. Scoring fires immediately when a new last-stage transition forms a complete path.

**Tech Stack:** Python 3.12+, multiprocessing.Pool, queue.Queue, zstandard, LLVM-MCA (subprocess via stdin/stdout)

**Spec:** `docs/superpowers/specs/2026-03-27-wave-engine-redesign.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/transition_table.py` | Modify | Add `trace_paths_ending_with()` for backwards path enumeration |
| `src/wave_checkpoint.py` | Modify | Fix `load_scored_paths` to return `[]` on missing file |
| `src/wave_engine.py` | Rewrite | New pool feeding, checkpointing, scoring, resume |
| `src/llvm_mca_runner.py` | Modify | Accept command string (for docker), pipe via stdin |
| `src/bitonic_compiler.py` | Modify | `--llvm-mca-path` → `--llvm-mca-cmd` |
| `tests/test_transition_table.py` | Create | Tests for `trace_paths_ending_with` |
| `tests/test_wave_engine.py` | Modify | Update tests for new API |
| `tests/test_wave_checkpoint.py` | Modify | Update scored paths tests, add missing-file test |

---

### Task 1: Backwards Path Trace in TransitionTable

**Files:**
- Modify: `src/transition_table.py`
- Create: `tests/test_transition_table.py`

- [ ] **Step 1: Write failing test for trace_paths_ending_with**

```python
# tests/test_transition_table.py
"""Tests for TransitionTable path tracing."""

from bitonic_types import InstructionSpec, PermutationGadget, VectorState
from transition_table import TransitionTable


def _vs(top, bottom):
    return VectorState(top=top, bottom=bottom)


def _gadget():
    return PermutationGadget(
        top_instructions=[InstructionSpec("test", {"a": "top"})],
        bottom_instructions=[],
        validated=True,
    )


class TestTracePathsEndingWith:
    """trace_paths_ending_with: backwards trace from last-stage transition."""

    def test_single_complete_path(self):
        """One path through 3 stages ending at a specific transition."""
        tt = TransitionTable(3)
        g = _gadget()

        inp0 = _vs([1, 2], [3, 4])
        out0 = _vs([1, 3], [2, 4])
        inp1 = out0
        out1 = _vs([1, 2], [4, 3])
        inp2 = out1
        out2 = _vs([4, 3], [2, 1])

        tt.add_transition(0, inp0, out0, g)
        tt.add_transition(1, inp1, out1, g)
        tt.add_transition(2, inp2, out2, g)

        paths = tt.trace_paths_ending_with(2, inp2.as_tuple(), out2.as_tuple())
        assert len(paths) == 1
        assert len(paths[0]) == 3
        assert paths[0][0] == (0, inp0.as_tuple(), out0.as_tuple())
        assert paths[0][2] == (2, inp2.as_tuple(), out2.as_tuple())

    def test_no_path_when_chain_broken(self):
        """No complete path if a middle stage has no matching transition."""
        tt = TransitionTable(3)
        g = _gadget()

        inp0 = _vs([1, 2], [3, 4])
        out0 = _vs([1, 3], [2, 4])
        # Stage 1 missing
        inp2 = _vs([9, 9], [9, 9])  # doesn't match out0
        out2 = _vs([4, 3], [2, 1])

        tt.add_transition(0, inp0, out0, g)
        tt.add_transition(2, inp2, out2, g)

        paths = tt.trace_paths_ending_with(2, inp2.as_tuple(), out2.as_tuple())
        assert paths == []

    def test_multiple_paths_branching_upstream(self):
        """Two paths to the same last-stage transition via different stage-0 outputs."""
        tt = TransitionTable(3)
        g = _gadget()

        # Two different stage 0 outputs
        inp0 = _vs([1, 2], [3, 4])
        out0a = _vs([1, 3], [2, 4])
        out0b = _vs([2, 1], [4, 3])

        tt.add_transition(0, inp0, out0a, g)
        tt.add_transition(0, inp0, out0b, g)

        # Both lead to same stage 1 output via different inputs
        out1 = _vs([5, 5], [5, 5])
        tt.add_transition(1, out0a, out1, g)
        tt.add_transition(1, out0b, out1, g)

        # Single stage 2 transition
        out2 = _vs([9, 9], [9, 9])
        tt.add_transition(2, out1, out2, g)

        paths = tt.trace_paths_ending_with(2, out1.as_tuple(), out2.as_tuple())
        assert len(paths) == 2

    def test_single_stage(self):
        """When there's only 1 stage, the trace returns the transition itself."""
        tt = TransitionTable(1)
        g = _gadget()

        inp = _vs([1, 2], [3, 4])
        out = _vs([2, 1], [4, 3])
        tt.add_transition(0, inp, out, g)

        paths = tt.trace_paths_ending_with(0, inp.as_tuple(), out.as_tuple())
        assert len(paths) == 1
        assert paths[0] == [(0, inp.as_tuple(), out.as_tuple())]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transition_table.py -v`
Expected: FAIL — `AttributeError: 'TransitionTable' object has no attribute 'trace_paths_ending_with'`

- [ ] **Step 3: Implement trace_paths_ending_with**

Add to `src/transition_table.py`, in the `TransitionTable` class, after the `enumerate_complete_paths` method:

```python
def trace_paths_ending_with(
    self,
    stage: int,
    input_tuple: StateTuple,
    output_tuple: StateTuple,
) -> list[list[tuple[int, StateTuple, StateTuple]]]:
    """Trace backwards to find all complete paths ending with this transition.

    Given a transition at ``stage`` with the given input/output, traces
    backwards through stages stage-1, stage-2, ..., 0 to find all
    complete paths (one step per stage) that end with this transition.

    Returns a list of paths, each path being a list of
    ``(stage_index, input_tuple, output_tuple)`` triples ordered
    stage 0 → stage N.
    """
    # Start with the given transition as the last step
    anchor = (stage, input_tuple, output_tuple)

    if stage == 0:
        return [[anchor]]

    # Trace backwards: for each stage from stage-1 down to 0,
    # find transitions whose output matches the required input
    # of the next stage.
    #
    # We build partial paths backwards, then reverse at the end.
    # partial_paths[i] = list of paths from stage i to `stage`,
    # stored in reverse order (stage, stage-1, ..., i).
    partial_paths: list[list[tuple[int, StateTuple, StateTuple]]] = [[anchor]]

    for s in range(stage - 1, -1, -1):
        sd = self.stages[s]
        new_partials: list[list[tuple[int, StateTuple, StateTuple]]] = []

        for partial in partial_paths:
            # The earliest step in this partial needs its input fed
            required_output = partial[-1][1]  # input_tuple of the next stage

            for (inp_t, out_t) in sd.transitions:
                if out_t == required_output:
                    new_partials.append([*partial, (s, inp_t, out_t)])

        if not new_partials:
            return []  # broken chain — no complete paths
        partial_paths = new_partials

    # Reverse each path so it's stage 0 → stage N
    return [list(reversed(p)) for p in partial_paths]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transition_table.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/transition_table.py tests/test_transition_table.py
git commit -m "feat: add trace_paths_ending_with to TransitionTable"
```

---

### Task 2: Fix Scored Paths Checkpoint Persistence

**Files:**
- Modify: `src/wave_checkpoint.py`
- Modify: `tests/test_wave_checkpoint.py`

- [ ] **Step 1: Write failing test for load_scored_paths returning [] on missing file**

Add to `tests/test_wave_checkpoint.py`, in `TestScoredPaths` class:

```python
def test_load_missing_returns_empty(self):
    """load_scored_paths returns [] when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = WaveCheckpoint(tmpdir)
        loaded = ckpt.load_scored_paths()
        assert loaded == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_wave_checkpoint.py::TestScoredPaths::test_load_missing_returns_empty -v`
Expected: FAIL — `FileNotFoundError`

- [ ] **Step 3: Fix load_scored_paths**

In `src/wave_checkpoint.py`, change `load_scored_paths`:

```python
def load_scored_paths(self) -> list[dict]:
    """Load scored paths from compressed JSON.

    Returns an empty list if the file does not exist.
    """
    path = self._scored_paths_path()
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        raw = f.read()
    dctx = zstd.ZstdDecompressor()
    return json.loads(dctx.decompress(raw))
```

- [ ] **Step 4: Update the old test that expected FileNotFoundError**

In `tests/test_wave_checkpoint.py`, class `TestCheckpointDirectory`, change `test_load_scored_paths_missing_raises`:

```python
def test_load_scored_paths_missing_returns_empty(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = WaveCheckpoint(tmpdir)
        assert ckpt.load_scored_paths() == []
```

- [ ] **Step 5: Run all checkpoint tests**

Run: `uv run pytest tests/test_wave_checkpoint.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/wave_checkpoint.py tests/test_wave_checkpoint.py
git commit -m "fix: load_scored_paths returns [] on missing file"
```

---

### Task 3: Update Scored Path Format to Use Stable path_key

**Files:**
- Modify: `src/wave_engine.py` (the `_score_complete_paths` method)
- Modify: `tests/test_wave_checkpoint.py`

- [ ] **Step 1: Update scored paths test to use path_key format**

In `tests/test_wave_checkpoint.py`, class `TestScoredPaths`, update `test_roundtrip`:

```python
def test_roundtrip(self):
    scored = [
        {
            "path_key": [
                [0, [[1, 2], [3, 4]], [[1, 3], [2, 4]]],
            ],
            "scores": {"znver3": {"throughput": 3.5, "cycles": 42.0}},
        },
        {
            "path_key": [
                [0, [[1, 2], [3, 4]], [[2, 1], [4, 3]]],
            ],
            "scores": {"znver3": {"throughput": 4.0, "cycles": 50.0}},
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = WaveCheckpoint(tmpdir)
        ckpt.save_scored_paths(scored)
        loaded = ckpt.load_scored_paths()
        assert loaded == scored
```

- [ ] **Step 2: Run test to verify it passes** (format-only change, checkpoint doesn't validate schema)

Run: `uv run pytest tests/test_wave_checkpoint.py::TestScoredPaths::test_roundtrip -v`
Expected: PASS (the checkpoint just serializes dicts)

- [ ] **Step 3: Update _score_complete_paths to use path_key**

In `src/wave_engine.py`, modify `_score_complete_paths`. Replace the `scored.append(...)` block and `best_scores` update:

```python
def _score_complete_paths(
    self, paths: list[list[tuple]] | None = None
) -> list[dict]:
    """Score complete paths with LLVM-MCA.

    If paths is None, enumerates all new complete paths via the
    TransitionTable (for batch scoring). If paths is provided,
    scores exactly those paths (for on-the-fly scoring).

    Returns a list of scored path dicts (newly scored only).
    """
    if paths is None:
        paths = self.tt.enumerate_complete_paths(
            max_paths=self.config.max_paths_per_wave,
            exclude_paths=self._scored_path_keys,
        )
    if not paths:
        return []

    # Register as scored
    for path in paths:
        self._scored_path_keys.add(tuple(tuple(step) for step in path))

    scored: list[dict] = []
    for path in paths:
        path_key = [
            [stage, [list(in_t[0]), list(in_t[1])], [list(out_t[0]), list(out_t[1])]]
            for stage, in_t, out_t in path
        ]
        scored.append({"path_key": path_key, "path": path, "scores": {}})

    if not self.config.target_cpus:
        return scored

    try:
        from perf_estimator import generate_solution_asm, sanitize_asm_for_llvm_mca
        from llvm_mca_runner import find_llvm_mca, run_llvm_mca
        from cost_model import resolve_llvm_mca_cpu
    except ImportError:
        from .perf_estimator import generate_solution_asm, sanitize_asm_for_llvm_mca
        from .llvm_mca_runner import find_llvm_mca, run_llvm_mca
        from .cost_model import resolve_llvm_mca_cpu

    mca_bin = find_llvm_mca(self.config.llvm_mca_path)
    if mca_bin is None:
        return scored

    def _instr_count_scorer(_path, assignments):
        return sum(g.instruction_count() for g in assignments)

    path_asm: dict[int, str] = {}
    for i, entry in enumerate(scored):
        path = entry["path"]
        assignments, _ = hill_climb_path(
            path, self.tt, _instr_count_scorer, max_passes=1
        )
        complete_path = _build_complete_path(path, assignments)
        try:
            asm = generate_solution_asm(
                complete_path,
                self.config.vm,
                self.config.prim_type,
                self.config.num_vecs,
                self.config.natural_order,
                i + 1,
                len(scored),
            )
            path_asm[i] = sanitize_asm_for_llvm_mca(asm)
        except Exception as exc:
            import sys
            print(
                f"Warning: ASM generation for path {i} failed: {exc}",
                file=sys.stderr,
            )

    if not path_asm:
        return scored

    mca_jobs: list[tuple[int, str, str, str]] = []
    for target_cpu in self.config.target_cpus:
        mcpu = resolve_llvm_mca_cpu(target_cpu)
        if mcpu is None:
            continue
        for idx, sanitized in path_asm.items():
            mca_jobs.append((idx, target_cpu, mcpu, sanitized))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_one(job):
        idx, target_cpu, mcpu, sanitized = job
        result = run_llvm_mca(sanitized, mcpu, mca_bin, idx + 1)
        return idx, target_cpu, result.throughput, result.simulated_cycles

    num_threads = min(len(mca_jobs), self.config.max_workers or os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(_run_one, job): job for job in mca_jobs}
        for future in as_completed(futures):
            try:
                idx, target_cpu, throughput, cycles = future.result()
                scored[idx]["scores"][target_cpu] = {
                    "throughput": throughput,
                    "cycles": cycles,
                }
                prev = self.best_scores.get(target_cpu)
                if prev is None or cycles < prev["cycles"]:
                    self.best_scores[target_cpu] = {
                        "throughput": throughput,
                        "cycles": cycles,
                        "path_key": scored[idx]["path_key"],
                    }
            except Exception as exc:
                import sys
                job = futures[future]
                print(
                    f"Warning: LLVM-MCA scoring path {job[0]} "
                    f"on {job[1]} failed: {exc}",
                    file=sys.stderr,
                )

    return scored
```

- [ ] **Step 4: Run existing tests**

Run: `uv run pytest tests/test_wave_engine.py tests/test_wave_checkpoint.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/wave_engine.py tests/test_wave_checkpoint.py
git commit -m "refactor: use stable path_key instead of positional path_index in scored paths"
```

---

### Task 4: New WaveEngine State + Helper Methods

**Files:**
- Modify: `src/wave_engine.py`

- [ ] **Step 1: Add new imports and state fields to __init__**

At the top of `src/wave_engine.py`, add `import time` and `from collections import deque` to the imports.

In `__init__`, after the existing `self._checkpoint` block (after line 245), add:

```python
        # Per-stage pending job deques (fed to the shared pool)
        self._pending_jobs: list[deque] = [
            deque() for _ in range(len(self.all_stages))
        ]

        # Per-stage in-flight job counters
        self._in_flight: list[int] = [0] * len(self.all_stages)

        # Accumulated scored paths across entire run
        self._all_scored_paths: list[dict] = []
```

- [ ] **Step 2: Implement _generate_jobs_for_stage**

Add to `WaveEngine` class, after `_make_jobs`:

```python
def _generate_jobs_for_stage(self, target_stage: int | None) -> None:
    """Generate up to wave_attempts jobs for a stage and enqueue them."""
    if target_stage is None:
        return

    if target_stage == 0:
        input_states = [self.initial_state]
    else:
        prev_outputs = self.tt.get_unique_outputs(target_stage - 1)
        input_states = list(prev_outputs.values()) if prev_outputs else []

    if not input_states:
        predecessors_exhausted = all(
            s in self.exhausted_stages for s in range(target_stage)
        )
        if predecessors_exhausted:
            self.exhausted_stages.add(target_stage)
        else:
            self._stalled_stages.add(target_stage)
        return

    new_jobs = self._make_jobs(target_stage, input_states)

    if not new_jobs:
        predecessors_exhausted = all(
            s in self.exhausted_stages for s in range(target_stage)
        )
        if predecessors_exhausted:
            self.exhausted_stages.add(target_stage)
            self._stalled_stages.discard(target_stage)
        else:
            self._stalled_stages.add(target_stage)
        return

    new_jobs = new_jobs[: self.config.wave_attempts]
    self._pending_jobs[target_stage].extend(new_jobs)
```

- [ ] **Step 3: Implement _enqueue_downstream**

Add after `_generate_jobs_for_stage`:

```python
def _enqueue_downstream(self, stage_idx: int, new_output_state: VectorState) -> None:
    """Generate jobs for stage_idx+1 using new_output_state as input."""
    next_stage = stage_idx + 1
    if next_stage >= len(self.all_stages):
        return

    out_t = new_output_state.as_tuple()
    self.tt.mark_forwarded(stage_idx, [out_t])

    new_jobs = self._make_jobs(next_stage, [new_output_state])
    self._pending_jobs[next_stage].extend(new_jobs)

    self._stalled_stages.discard(next_stage)
    for s in range(next_stage + 1, len(self.all_stages)):
        self._stalled_stages.discard(s)
```

- [ ] **Step 4: Implement _enqueue_from_unforwarded**

Add after `_enqueue_downstream`:

```python
def _enqueue_from_unforwarded(self, stage_idx: int) -> None:
    """Seed jobs for stage_idx from unforwarded outputs of stage_idx-1."""
    prev = stage_idx - 1
    if prev < 0:
        return
    unforwarded = self.tt.get_unforwarded_outputs(prev)
    if not unforwarded:
        return

    input_states = [vs for _tup, vs in unforwarded]
    output_tuples = [tup for tup, _vs in unforwarded]
    self.tt.mark_forwarded(prev, output_tuples)

    new_jobs = self._make_jobs(stage_idx, input_states)
    self._pending_jobs[stage_idx].extend(new_jobs)
```

- [ ] **Step 5: Implement _submit_jobs**

Add after `_enqueue_from_unforwarded`:

```python
def _submit_jobs(
    self,
    pool: Pool,
    completion_queue,
    pool_target: int,
) -> int:
    """Submit jobs to the pool until pool_target in-flight reached.

    Prioritizes lower stage indices. Returns number of jobs submitted.
    """
    total_in_flight = sum(self._in_flight)
    submitted = 0

    for stage_idx in range(len(self.all_stages)):
        pending = self._pending_jobs[stage_idx]
        while pending and total_in_flight < pool_target:
            _cand_idx, job = pending.popleft()

            pool.apply_async(
                _validate_gadget_worker,
                (job,),
                callback=lambda r, s=stage_idx: completion_queue.put(("ok", s, r)),
                error_callback=lambda e, s=stage_idx: completion_queue.put(
                    ("err", s, e)
                ),
            )

            self._in_flight[stage_idx] += 1
            total_in_flight += 1
            submitted += 1

    return submitted
```

- [ ] **Step 6: Implement _drain_results**

Add after `_submit_jobs`:

```python
def _drain_results(
    self,
    completion_queue,
    progress: SuccessProgress | None = None,
    stage_task_ids: dict[int, int] | None = None,
) -> int:
    """Process all available results from the completion queue.

    Records transitions, enqueues downstream jobs, triggers on-the-fly
    scoring for last-stage transitions. Returns count of results processed.
    """
    import queue as queue_mod

    count = 0
    last_stage = len(self.all_stages) - 1

    while True:
        try:
            item = completion_queue.get_nowait()
        except queue_mod.Empty:
            break

        count += 1

        if item[0] == "err":
            _, stage_idx, _exc = item
            self._in_flight[stage_idx] -= 1
            self._stage_attempts[stage_idx] += 1
            self.tt.record_attempt(stage_idx)
            if progress and stage_task_ids:
                tid = stage_task_ids.get(stage_idx)
                if tid is not None:
                    progress.update(tid, advance=1, success=0)
            continue

        _, stage_idx, payload = item
        self._in_flight[stage_idx] -= 1

        gadget_results, input_state, _metadata, _ct, _st = payload
        success = 1 if gadget_results else 0

        cand_index = _metadata.get("candidate_index", -1)
        self.tt.record_attempted_pair(
            stage_idx, input_state.as_tuple(), cand_index
        )
        self.tt.record_attempt(stage_idx)
        self._stage_attempts[stage_idx] += 1

        new_unique = 0
        for gadget, output_state in gadget_results:
            out_t = output_state.as_tuple()
            is_new = out_t not in self.tt.get_unique_outputs(stage_idx)
            self.tt.add_transition(stage_idx, input_state, output_state, gadget)

            if is_new:
                new_unique += 1
                self._enqueue_downstream(stage_idx, output_state)

                # On-the-fly scoring: new last-stage transition
                if stage_idx == last_stage:
                    self._score_new_paths_for_transition(
                        stage_idx,
                        input_state.as_tuple(),
                        out_t,
                        progress,
                    )

        if progress and stage_task_ids:
            tid = stage_task_ids.get(stage_idx)
            if tid is not None:
                progress.update(tid, advance=1, success=success, unique=new_unique)

    return count
```

- [ ] **Step 7: Implement _score_new_paths_for_transition**

Add after `_drain_results`:

```python
def _score_new_paths_for_transition(
    self,
    stage_idx: int,
    input_tuple: tuple,
    output_tuple: tuple,
    progress: SuccessProgress | None = None,
) -> None:
    """Score complete paths formed by a new transition at the last stage.

    Traces backwards to find all complete paths ending with this
    transition, filters already-scored, scores the rest.
    """
    paths = self.tt.trace_paths_ending_with(stage_idx, input_tuple, output_tuple)
    if not paths:
        return

    # Filter already-scored
    new_paths = []
    for path in paths:
        key = tuple(tuple(step) for step in path)
        if key not in self._scored_path_keys:
            new_paths.append(path)

    if not new_paths:
        return

    scored = self._score_complete_paths(paths=new_paths)
    if scored:
        self._all_scored_paths.extend(scored)

        if progress and self.best_scores:
            parts = [
                f"{cpu}: {s['cycles']:.1f}cy (throughput {s['throughput']:.2f})"
                for cpu, s in sorted(self.best_scores.items())
            ]
            progress.set_status("Best: " + "  |  ".join(parts))
```

- [ ] **Step 8: Implement _rebuild_best_scores_from_accumulated**

Add after `_score_new_paths_for_transition`:

```python
def _rebuild_best_scores_from_accumulated(self) -> None:
    """Rebuild best_scores from all accumulated scored paths."""
    self.best_scores = {}
    for entry in self._all_scored_paths:
        for cpu, scores in entry.get("scores", {}).items():
            cycles = scores.get("cycles", float("inf"))
            prev = self.best_scores.get(cpu)
            if prev is None or cycles < prev["cycles"]:
                self.best_scores[cpu] = {
                    "throughput": scores["throughput"],
                    "cycles": cycles,
                    "path_key": entry.get("path_key"),
                }
```

- [ ] **Step 9: Run existing tests to verify nothing is broken**

Run: `uv run pytest tests/ -v`
Expected: all PASS (new methods added but not yet called from run())

- [ ] **Step 10: Commit**

```bash
git add src/wave_engine.py
git commit -m "feat: add pool feeding, result draining, and on-the-fly scoring methods"
```

---

### Task 5: Update Checkpoint and Resume

**Files:**
- Modify: `src/wave_engine.py`

- [ ] **Step 1: Update _save_checkpoint to include accumulated scores**

Replace `_save_checkpoint` in `src/wave_engine.py`:

```python
def _save_checkpoint(self) -> None:
    """Save current state to checkpoint directory.

    Saves dirty stages, master config, and accumulated scored paths.
    """
    if self._checkpoint is None:
        return

    stage_stats = {}
    for i in range(len(self.all_stages)):
        stage_stats[i] = self.tt.stage_stats(i)

    config = WaveMasterConfig(
        num_vecs=self.config.num_vecs,
        vm=self.config.vm.name,
        prim_type=self.config.prim_type.name,
        gadget_depth=self.config.gadget_depth,
        natural_order=self.config.natural_order,
        retroactive_input=self.config.retroactive_input,
        num_stages=len(self.all_stages),
        wave_count=self.wave_count,
        best_scores=self.best_scores,
        stage_stats=stage_stats,
        exhausted_stages=sorted(self.exhausted_stages),
        target_cpus=self.config.target_cpus,
    )
    self._checkpoint.save_master(config)

    for i in range(len(self.all_stages)):
        if self.tt.stages[i].dirty:
            self._checkpoint.save_stage(i, self.tt)
            self.tt.stages[i].dirty = False

    if self._all_scored_paths:
        self._checkpoint.save_scored_paths(self._all_scored_paths)
```

- [ ] **Step 2: Update resume to load scores and rebuild state**

Replace `resume` in `src/wave_engine.py`:

```python
def resume(self, checkpoint_dir: str) -> None:
    """Load state from a checkpoint directory.

    Restores TransitionTable, wave_count, scores, and scored path keys.
    """
    ckpt = WaveCheckpoint(checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    master = ckpt.load_master()
    self.wave_count = master.wave_count
    self.exhausted_stages = set(master.exhausted_stages)

    for i in range(len(self.all_stages)):
        stage_path = os.path.join(checkpoint_dir, f"stage_{i:02d}.json.zst")
        if os.path.exists(stage_path):
            ckpt.load_stage(i, self.tt)

    # Restore cumulative progress counters from TransitionTable
    for i in range(len(self.all_stages)):
        self._stage_attempts[i] = self.tt.stages[i].attempts

    # Restore scored paths and rebuild derived state
    self._all_scored_paths = ckpt.load_scored_paths()
    for entry in self._all_scored_paths:
        path = entry.get("path")
        if path:
            self._scored_path_keys.add(tuple(tuple(step) for step in path))

    self._rebuild_best_scores_from_accumulated()
```

- [ ] **Step 3: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add src/wave_engine.py
git commit -m "feat: checkpoint includes accumulated scores, resume rebuilds scoring state"
```

---

### Task 6: Rewrite run() and Delete Old Methods

**Files:**
- Modify: `src/wave_engine.py`
- Modify: `tests/test_wave_engine.py`

- [ ] **Step 1: Update tests for new API**

The existing tests call `run_wave()` and check `summary["waves"]`. Update `tests/test_wave_engine.py`:

Replace `TestFirstWaveDiscoveries`:

```python
class TestFirstWaveDiscoveries:
    """First wave should discover transitions at stage 0."""

    def test_first_wave_discovers_transitions(self):
        config = _fast_config(wave_attempts=500, wave_outputs=50, max_waves=1)
        engine = WaveEngine(config)
        engine.run()

        stage0_outputs = engine.tt.unique_output_count(0)
        assert stage0_outputs > 0, (
            f"Expected stage 0 to have discoveries, got {stage0_outputs}"
        )
```

Replace `TestMultiWaveProgress`:

```python
class TestMultiWaveProgress:
    """Multiple waves should make progress."""

    def test_three_waves_make_progress(self):
        config = _fast_config(max_waves=3, wave_attempts=500, wave_outputs=50)
        engine = WaveEngine(config)

        summary = engine.run()

        assert summary["wave_count"] == 3
        assert not summary["interrupted"]

        stage0_outputs = engine.tt.unique_output_count(0)
        assert stage0_outputs > 0
```

- [ ] **Step 2: Rewrite run()**

Replace `run_wave` and `run` with a single new `run` method. Delete `_run_stage_budget`, `_forward_propagate`, and `run_wave` entirely.

New `run()` method (replaces everything from the old `run_wave` through the old `run`):

```python
def run(self) -> dict:
    """Main loop: long-lived pool, wave-based targeting, continuous feeding.

    Each wave:
    1. Selects target stage
    2. Generates jobs and enqueues them
    3. Drains the pool, feeding downstream stages as outputs appear
    4. Checkpoints when a stage completes (pending=0, in_flight=0)
    5. Scores on the fly as last-stage transitions form complete paths

    Termination: Ctrl-C, max_waves reached, or all stages exhausted.
    """
    import queue as queue_mod

    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(_signum, _frame):
        self._interrupted = True
        import sys
        print(
            "\nCtrl-C detected — finishing current work and saving checkpoint...",
            file=sys.stderr,
        )

    signal.signal(signal.SIGINT, _sigint_handler)

    num_workers = self.config.max_workers or os.cpu_count() or 4
    pool_target = num_workers * 2
    completion_queue: queue_mod.Queue = queue_mod.Queue()

    pool = Pool(
        processes=num_workers,
        maxtasksperchild=self.config.max_tasks_per_child,
    )

    progress = SuccessProgress.create()
    progress.enable_memory_monitor()

    try:
        with progress:
            stage_task_ids: dict[int, int] = {}
            for s in range(len(self.all_stages)):
                tid = progress.add_task(
                    f"Stage {s}",
                    total=None,
                    start=False,
                    successes=0,
                    unique=0,
                )
                stage_task_ids[s] = tid

            # Seed pending queues from any unforwarded outputs (resume case)
            for s in range(1, len(self.all_stages)):
                self._enqueue_from_unforwarded(s)

            while True:
                if self._interrupted:
                    self._save_checkpoint()
                    progress.console.print(
                        "[yellow]Interrupted — checkpoint saved.[/yellow]"
                    )
                    break

                if (
                    self.config.max_waves is not None
                    and self.wave_count >= self.config.max_waves
                ):
                    break

                if len(self.exhausted_stages) >= len(self.all_stages):
                    progress.console.print(
                        "[green]All stages exhausted — search complete.[/green]"
                    )
                    break

                # --- Start a new wave ---
                target = self._select_target_stage()
                if target is None:
                    break

                target_outputs_before = self.tt.unique_output_count(target)
                self._generate_jobs_for_stage(target)

                task_id = stage_task_ids.get(target)
                if task_id is not None:
                    progress.start_task(task_id)
                    progress.update(
                        task_id,
                        description=(
                            f"[bold]Stage {target}[/bold] "
                            f"(wave {self.wave_count})"
                        ),
                    )

                # Track which stages are active this wave (have pending or in-flight)
                wave_new_outputs = 0
                stages_active_this_wave: set[int] = {target}

                # --- Drain the wave ---
                while True:
                    if self._interrupted:
                        break

                    # Top up the pool
                    self._submit_jobs(pool, completion_queue, pool_target)

                    # Drain results
                    drained = self._drain_results(
                        completion_queue, progress, stage_task_ids
                    )

                    # Track active stages (any with pending or in-flight)
                    for s in range(len(self.all_stages)):
                        if self._pending_jobs[s] or self._in_flight[s] > 0:
                            stages_active_this_wave.add(s)

                    # Check for stage completions -> checkpoint
                    for s in list(stages_active_this_wave):
                        if (
                            not self._pending_jobs[s]
                            and self._in_flight[s] == 0
                        ):
                            stages_active_this_wave.discard(s)
                            self._save_checkpoint()

                    # Wave done when nothing pending and nothing in-flight
                    total_pending = sum(
                        len(q) for q in self._pending_jobs
                    )
                    total_in_flight = sum(self._in_flight)
                    if total_pending == 0 and total_in_flight == 0:
                        break

                    if drained == 0:
                        time.sleep(0.05)

                # --- Wave complete ---
                # Track consecutive zero-output budgets for target stage
                target_outputs_now = self.tt.unique_output_count(target)
                sd = self.tt.stages[target]
                # Compare to pre-wave count to get new outputs this wave
                if target_outputs_now == target_outputs_before:
                    sd.consecutive_zero_budgets += 1
                else:
                    sd.consecutive_zero_budgets = 0
                    for s in range(target + 1, len(self.all_stages)):
                        self._stalled_stages.discard(s)

                # Final checkpoint for the wave
                self._save_checkpoint()

                # Wave summary
                stats = self.tt.stage_stats(target)
                scored_total = len(self._all_scored_paths)
                progress.console.print(
                    f"Wave {self.wave_count}: "
                    f"stage {target} targeted, "
                    f"{stats['distinct_outputs']} outputs"
                    f" | {scored_total} paths scored total",
                )

                if self.best_scores:
                    parts = [
                        f"{cpu}: {s['cycles']:.1f}cy "
                        f"(throughput {s['throughput']:.2f})"
                        for cpu, s in sorted(self.best_scores.items())
                    ]
                    progress.set_status("Best: " + "  |  ".join(parts))

                self.wave_count += 1

    finally:
        pool.terminate()
        pool.join()
        signal.signal(signal.SIGINT, original_sigint)

    return {
        "wave_count": self.wave_count,
        "interrupted": self._interrupted,
    }
```

- [ ] **Step 3: Delete old methods**

Delete the following methods from `WaveEngine`:
- `_run_stage_budget` (lines 304-452)
- `_forward_propagate` (lines 458-505)
- `run_wave` (lines 729-814)

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Run quick integration test**

Run: `uv run python src/bitonic_compiler.py --vector-machine AVX2 --datatype i64 --depth-limit 3 --gadget-depth 1 --top-k 5 --max-waves 2`
Expected: completes without error, shows wave summaries

- [ ] **Step 6: Commit**

```bash
git add src/wave_engine.py tests/test_wave_engine.py
git commit -m "feat: rewrite wave engine with continuous pool feeding and on-the-fly scoring"
```

---

### Task 7: LLVM-MCA Command Support (stdin/stdout)

**Files:**
- Modify: `src/llvm_mca_runner.py`
- Modify: `src/wave_engine.py`
- Modify: `src/bitonic_compiler.py`

- [ ] **Step 1: Update run_llvm_mca to pipe via stdin**

Replace `run_llvm_mca` in `src/llvm_mca_runner.py`:

```python
def run_llvm_mca(
    asm_text: str,
    mcpu: str,
    llvm_mca_cmd: str | list[str],
    solution_index: int,
    output_dir: str | None = None,
) -> McaResult:
    """Run llvm-mca on Intel-syntax assembly and return parsed results.

    Args:
        asm_text: Intel-syntax assembly text (with LLVM-MCA markers).
        mcpu: CPU model for -mcpu= flag.
        llvm_mca_cmd: Command to run llvm-mca. Can be a string
            (split via shlex) or a list. Supports dockerized commands
            like "docker run -i silkeh/clang /usr/bin/llvm-mca".
        solution_index: 1-based solution index for reporting.
        output_dir: Optional directory for saving .s input files.
    """
    import shlex

    if isinstance(llvm_mca_cmd, str):
        base_parts = shlex.split(llvm_mca_cmd)
    else:
        base_parts = list(llvm_mca_cmd)

    # Save asm to file for debugging if output_dir specified
    asm_path = ""
    if output_dir:
        asm_path = str(Path(output_dir) / f"solution_{solution_index:03d}.s")
        Path(asm_path).write_text(asm_text)

    base_cmd = [
        *base_parts,
        "-march=x86-64",
        f"-mcpu={mcpu}",
        "-x86-asm-syntax=intel",
    ]

    # Pipe via stdin (works with docker, local binary, etc.)
    json_cmd = [*base_cmd, "--json"]

    try:
        result = subprocess.run(
            json_cmd,
            input=asm_text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return McaResult(
                solution_index=solution_index,
                asm_path=asm_path,
                throughput=-1.0,
                simulated_cycles=-1.0,
                warnings=[
                    f"llvm-mca failed (rc={result.returncode}): "
                    f"{result.stderr.strip()}"
                ],
            )

        mca_json = json.loads(result.stdout)
        mca_result = _parse_mca_json(mca_json, solution_index, asm_path)

        mca_result.analysis_path = _run_full_analysis(
            base_cmd, asm_text, output_dir, solution_index, mca_result.warnings
        )

        return mca_result

    except FileNotFoundError:
        return McaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            throughput=-1.0,
            simulated_cycles=-1.0,
            warnings=[f"llvm-mca command not found: '{base_parts[0]}'"],
        )
    except json.JSONDecodeError as e:
        return McaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            throughput=-1.0,
            simulated_cycles=-1.0,
            warnings=[f"Failed to parse llvm-mca JSON output: {e}"],
        )
    except subprocess.TimeoutExpired:
        return McaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            throughput=-1.0,
            simulated_cycles=-1.0,
            warnings=["llvm-mca timed out after 30s"],
        )
```

- [ ] **Step 2: Update _run_full_analysis to pipe via stdin**

Replace `_run_full_analysis` in `src/llvm_mca_runner.py`:

```python
def _run_full_analysis(
    base_cmd: list[str],
    asm_text: str,
    output_dir: str | None,
    solution_index: int,
    warnings: list[str],
) -> str:
    """Run llvm-mca without --json and save the full text analysis."""
    text_cmd = [*base_cmd, "-timeline"]
    try:
        result = subprocess.run(
            text_cmd,
            input=asm_text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            warnings.append(
                f"llvm-mca text analysis failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )
            return ""

        analysis_text = result.stdout
        if output_dir:
            analysis_path = str(
                Path(output_dir) / f"solution_{solution_index:03d}_analysis.txt"
            )
        else:
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".analysis.txt",
                prefix=f"vxsort_mca_{solution_index:03d}_",
                delete=False,
            )
            tmp.write(analysis_text)
            tmp.close()
            analysis_path = tmp.name
        if output_dir:
            Path(analysis_path).write_text(analysis_text)
        return analysis_path

    except (subprocess.TimeoutExpired, OSError) as e:
        warnings.append(f"llvm-mca text analysis error: {e}")
        return ""
```

- [ ] **Step 3: Update find_llvm_mca to handle command strings**

Replace `find_llvm_mca` in `src/llvm_mca_runner.py`:

```python
def find_llvm_mca(explicit_cmd: str | None) -> str | None:
    """Locate the llvm-mca binary or command.

    If explicit_cmd is provided, returns it directly (may be a full
    command like "docker run -i silkeh/clang /usr/bin/llvm-mca").
    Otherwise searches common locations and PATH.
    """
    if explicit_cmd:
        return explicit_cmd

    for homebrew_path in [
        "/opt/homebrew/opt/llvm/bin/llvm-mca",
        "/usr/local/opt/llvm/bin/llvm-mca",
    ]:
        if Path(homebrew_path).is_file():
            return homebrew_path

    return shutil.which("llvm-mca")
```

- [ ] **Step 4: Rename llvm_mca_path to llvm_mca_cmd in WaveConfig**

In `src/wave_engine.py`, in `WaveConfig`:

```python
    llvm_mca_cmd: str | None = None
```

And update references in `_score_complete_paths`:

```python
    mca_bin = find_llvm_mca(self.config.llvm_mca_cmd)
```

- [ ] **Step 5: Update bitonic_compiler.py CLI arg**

In `src/bitonic_compiler.py`, change the argparse definition:

```python
    parser.add_argument(
        "--llvm-mca-cmd",
        type=str,
        default=None,
        metavar="CMD",
        help="Command to run llvm-mca (e.g. 'docker run -i silkeh/clang /usr/bin/llvm-mca'). "
        "If not specified, searches Homebrew locations and PATH.",
    )
    # Deprecated alias
    parser.add_argument(
        "--llvm-mca-path",
        type=str,
        default=None,
        dest="llvm_mca_cmd_deprecated",
        help="Deprecated: use --llvm-mca-cmd instead.",
    )
```

And where the arg is used, resolve the deprecated alias:

```python
    llvm_mca_cmd = getattr(args, "llvm_mca_cmd", None) or getattr(
        args, "llvm_mca_cmd_deprecated", None
    )
```

Pass `llvm_mca_cmd` everywhere `llvm_mca_path` was passed. Update the `WaveConfig` construction and all `estimate_solutions_from_*` calls to use `llvm_mca_cmd`.

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: all PASS

- [ ] **Step 7: Lint and dead code check**

Run: `uv run ruff check . && uv run vulture`
Expected: clean

- [ ] **Step 8: Commit**

```bash
git add src/llvm_mca_runner.py src/wave_engine.py src/bitonic_compiler.py
git commit -m "feat: --llvm-mca-cmd accepts full command lines (docker support), pipe via stdin"
```

---

### Task 8: Final Integration Test and Cleanup

**Files:**
- Modify: `tests/test_wave_engine.py`

- [ ] **Step 1: Add end-to-end checkpoint + resume + scoring test**

Add to `tests/test_wave_engine.py`, in `TestEndToEnd`:

```python
def test_checkpoint_resume_with_scores(self, tmp_path):
    """Checkpoint saves scores, resume restores them."""
    config = _fast_config(
        wave_attempts=1000,
        wave_outputs=20,
        max_paths_per_wave=50,
        max_waves=2,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    engine = WaveEngine(config)
    engine.run()
    assert engine.wave_count == 2

    import os
    assert os.path.exists(str(tmp_path / "ckpt" / "master.json"))

    # Resume
    config2 = _fast_config(
        wave_attempts=1000,
        wave_outputs=20,
        max_paths_per_wave=50,
        max_waves=3,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    engine2 = WaveEngine(config2)
    engine2.resume(str(tmp_path / "ckpt"))

    # Scored path keys should be restored
    assert engine2._scored_path_keys == engine._scored_path_keys
    assert engine2._all_scored_paths == engine._all_scored_paths

    engine2.run()
    assert engine2.wave_count == 3

    solutions = engine2.export_to_solution_nodes()
    assert len(solutions) > 0
```

- [ ] **Step 2: Update existing end-to-end test**

Update `test_avx2_i64_full_pipeline` to match new `run()` return format (no `summary["waves"]`):

```python
def test_avx2_i64_full_pipeline(self, tmp_path):
    """Full pipeline: synthesis -> checkpoint -> resume -> export."""
    config = _fast_config(
        wave_attempts=1000,
        wave_outputs=20,
        max_paths_per_wave=50,
        max_waves=2,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    engine = WaveEngine(config)
    summary = engine.run()
    assert summary["wave_count"] == 2

    import os
    assert os.path.exists(str(tmp_path / "ckpt" / "master.json"))

    config2 = _fast_config(
        wave_attempts=1000,
        wave_outputs=20,
        max_paths_per_wave=50,
        max_waves=3,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    engine2 = WaveEngine(config2)
    engine2.resume(str(tmp_path / "ckpt"))
    engine2.run()
    assert engine2.wave_count == 3

    solutions = engine2.export_to_solution_nodes()
    assert len(solutions) > 0
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: all PASS

- [ ] **Step 4: Run integration test**

Run: `uv run python src/bitonic_compiler.py --vector-machine AVX2 --datatype i64 --depth-limit 3 --gadget-depth 1 --top-k 5 --max-waves 2`
Expected: completes with wave summaries, checkpoints saved

- [ ] **Step 5: Lint and dead code**

Run: `uv run ruff check . && uv run vulture`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add tests/test_wave_engine.py
git commit -m "test: add checkpoint+resume+scoring integration test"
```
