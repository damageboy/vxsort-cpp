"""WaveEngine: iterative wave-based synthesis orchestrator.

Each wave targets the weakest stage, runs a fixed budget of synthesis
jobs via a long-lived shared process pool, forward-propagates new
outputs to downstream stages as they appear, and checkpoints on each
stage completion within the wave.  Scoring happens on the fly as
last-stage transitions form complete paths.
"""

from __future__ import annotations

import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import Pool

try:
    from .bitonic_sorter import BitonicSorter
    from .bitonic_types import (
        PermutationGadget,
        SolutionNode,
        VectorState,
    )
    from .gadget_synthesizer import (
        GadgetSynthesizer,
        _validate_gadget_worker,
    )
    from .success_progress import SuccessProgress
    from .transition_table import TransitionTable
    from .utils import primitive_type, vector_machine, width_dict
    from .wave_checkpoint import WaveCheckpoint, WaveMasterConfig
except ImportError:
    from bitonic_sorter import BitonicSorter  # type: ignore[no-redef]
    from bitonic_types import (  # type: ignore[no-redef]
        PermutationGadget,
        SolutionNode,
        VectorState,
    )
    from gadget_synthesizer import (  # type: ignore[no-redef]
        GadgetSynthesizer,
        _validate_gadget_worker,
    )
    from success_progress import SuccessProgress  # type: ignore[no-redef]
    from transition_table import TransitionTable  # type: ignore[no-redef]
    from utils import primitive_type, vector_machine, width_dict  # type: ignore[no-redef]
    from wave_checkpoint import WaveCheckpoint, WaveMasterConfig  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Hill-climbing optimiser (module-level, no WaveEngine dependency)
# ---------------------------------------------------------------------------


def hill_climb_path(
    path: list[tuple],
    tt: TransitionTable,
    scorer: callable,
    max_passes: int = 3,
) -> tuple[list[PermutationGadget], float]:
    """First-improvement hill climbing for gadget assignment.

    Args:
        path: list of (stage, input_tuple, output_tuple) tuples
        tt: TransitionTable with gadget alternatives
        scorer: callable(path, gadget_assignments) -> float (lower is better)
        max_passes: maximum number of full passes through stages

    Returns:
        (best_gadget_assignments, best_score)
    """
    # 1. Initial assignment: pick the gadget with fewest instructions
    assignments: list[PermutationGadget] = []
    for stage, in_t, out_t in path:
        gadgets = tt.get_all_transitions(stage)[(in_t, out_t)]
        assignments.append(min(gadgets, key=lambda g: g.instruction_count()))

    best_score = scorer(path, assignments)

    # 2. First-improvement hill climbing, capped at max_passes
    for _pass in range(max_passes):
        improved = False
        for idx, (stage, in_t, out_t) in enumerate(path):
            all_gadgets = tt.get_all_transitions(stage)[(in_t, out_t)]
            current = assignments[idx]
            for candidate in all_gadgets:
                if candidate is current:
                    continue
                # Try the candidate
                assignments[idx] = candidate
                new_score = scorer(path, assignments)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                    break  # first improvement: restart from stage 0
                else:
                    assignments[idx] = current
            if improved:
                break  # restart outer loop from stage 0
        if not improved:
            break  # no improvement in a full pass, converged

    return assignments, best_score


def _build_complete_path(path, gadget_assignments):
    """Build a CompletePath from a TransitionTable path + gadget assignments.

    Bridges between the wave engine's (stage, input_tuple, output_tuple) format
    and the CompletePath/PathStep format expected by generate_solution_asm.
    """
    # Late imports to avoid circular deps
    try:
        from path_selector import CompletePath, GadgetScore, PathStep
    except ImportError:
        from .path_selector import CompletePath, GadgetScore, PathStep  # type: ignore[no-redef]

    dummy_score = GadgetScore(latency_cost=0.0, control_vector_count=0, total_score=0.0)

    steps = []
    for (stage, in_t, out_t), gadget in zip(path, gadget_assignments):
        node = SolutionNode(
            stage=stage,
            input_state=VectorState(top=list(in_t[0]), bottom=list(in_t[1])),
            output_state=VectorState(top=list(out_t[0]), bottom=list(out_t[1])),
            gadgets=[gadget],
            children=[],
        )
        steps.append(PathStep(node=node, gadget_index=0, score=dummy_score))

    return CompletePath(
        steps=steps,
        total_latency=0.0,
        total_cv_count=0,
        total_score=0.0,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WaveConfig:
    """Configuration for wave-based synthesis."""

    num_vecs: int
    vm: vector_machine
    prim_type: primitive_type
    gadget_depth: int
    natural_order: bool
    retroactive_input: bool
    wave_attempts: int = 10000
    wave_outputs: int = 100
    max_paths_per_wave: int = 1000
    target_cpus: list[str] = field(default_factory=list)
    max_workers: int | None = None
    max_tasks_per_child: int | None = None
    max_unique_outputs: int = 3
    depth2_threshold: float = 0.1
    smt2_dump_dir: str | None = None
    checkpoint_dir: str | None = None
    llvm_mca_path: str | None = None
    max_waves: int | None = None
    top_k: int = 10


# ---------------------------------------------------------------------------
# WaveEngine
# ---------------------------------------------------------------------------


class WaveEngine:
    """Iterative wave-based synthesis engine with continuous pool feeding.

    A single long-lived process pool is shared across the entire run.
    Waves drive stage targeting and budgeting.  Within a wave, per-stage
    pending deques feed the pool in stage-priority order so downstream
    stages fill idle capacity during straggler Z3 jobs.  Checkpoints fire
    when a stage completes.  Scoring fires immediately when a new
    last-stage transition forms a complete path.
    """

    def __init__(self, config: WaveConfig) -> None:
        self.config = config

        # Compute element counts
        self.elements_per_vector = width_dict[config.vm] // int(
            config.prim_type.value[0]
        )
        self.total_elements = config.num_vecs * self.elements_per_vector

        # Build bitonic stages
        sorter = BitonicSorter(self.total_elements)

        if config.natural_order:
            natural_pairs = [
                (i + 1, self.elements_per_vector + i + 1)
                for i in range(self.elements_per_vector)
            ]
            new_stage = max(sorter.stages.keys()) + 1
            sorter.stages[new_stage] = natural_pairs

        self.all_stages = [sorter.stages[k] for k in sorted(sorter.stages.keys())]

        # Pre-compute candidates split by depth tier
        synthesizer = GadgetSynthesizer(config.vm, config.prim_type)
        shallow_candidates, deep_candidates = (
            synthesizer.precompute_candidates_stratified(config.gadget_depth)
        )
        self.shallow_candidates = shallow_candidates
        self.deep_candidates = deep_candidates

        # Build combined candidate list with stable indices
        self._candidates_with_index: list[tuple[int, object]] = list(
            enumerate(self.shallow_candidates)
        )
        offset = len(self.shallow_candidates)
        self._candidates_with_index.extend(
            (offset + i, g) for i, g in enumerate(self.deep_candidates)
        )

        # TransitionTable and initial state
        self.tt = TransitionTable(num_stages=len(self.all_stages))
        self.initial_state = self._create_initial_state()

        # Wave tracking
        self.wave_count = 0
        self.exhausted_stages: set[int] = set()
        self.best_scores: dict[str, dict] = {}

        # Stages with no untried jobs right now, but not truly exhausted
        self._stalled_stages: set[int] = set()

        # Already-scored path keys (avoid re-scoring)
        self._scored_path_keys: set[tuple] = set()

        # Cumulative progress counters per stage (for progress bars)
        self._stage_attempts: list[int] = [0] * len(self.all_stages)

        # Ctrl-C handling
        self._interrupted = False

        # Checkpoint
        self._checkpoint: WaveCheckpoint | None = None
        if config.checkpoint_dir is not None:
            self._checkpoint = WaveCheckpoint(config.checkpoint_dir)

        # Per-stage pending job deques (fed to the shared pool)
        self._pending_jobs: list[deque] = [deque() for _ in range(len(self.all_stages))]

        # Per-stage in-flight job counters
        self._in_flight: list[int] = [0] * len(self.all_stages)

        # Accumulated scored paths across entire run
        self._all_scored_paths: list[dict] = []

        # Per-stage attempt snapshot at wave start.  Used by _submit_jobs
        # to enforce per-stage budgets within a wave.  None outside a wave.
        self._wave_start_attempts: list[int] | None = None

    def _create_initial_state(self) -> VectorState:
        """Create initial vector state from first stage comparison pairs."""
        first_stage_pairs = self.all_stages[0]
        top = []
        bottom = []
        for pair in first_stage_pairs:
            top.append(pair[0])
            bottom.append(pair[1])
        return VectorState(top=top, bottom=bottom)

    # ------------------------------------------------------------------
    # Job creation
    # ------------------------------------------------------------------

    def _make_jobs(
        self,
        stage_idx: int,
        input_states: list[VectorState],
    ) -> list[tuple[int, tuple]]:
        """Create (candidate_index, job_tuple) pairs, skipping already-attempted.

        Returns list of (candidate_index, job) where job is the tuple
        expected by ``_validate_gadget_worker``.
        """
        stage_pairs = self.all_stages[stage_idx]
        jobs: list[tuple[int, tuple]] = []

        for input_state in input_states:
            input_tuple = input_state.as_tuple()
            for cand_idx, graph in self._candidates_with_index:
                if self.tt.was_attempted(stage_idx, input_tuple, cand_idx):
                    continue

                metadata = {
                    "input_state": input_state,
                    "stage_idx": stage_idx,
                    "max_unique_outputs": self.config.max_unique_outputs,
                    "candidate_index": cand_idx,
                }
                # Final natural-order stage must enforce strict lane order
                if self.config.natural_order and stage_idx == len(self.all_stages) - 1:
                    metadata["allow_any_lane_order"] = False
                if self.config.smt2_dump_dir:
                    metadata["smt2_dump_dir"] = self.config.smt2_dump_dir

                job = (
                    graph,
                    input_state,
                    stage_pairs,
                    self.config.vm,
                    self.config.prim_type,
                    metadata,
                )
                jobs.append((cand_idx, job))

        return jobs

    # ------------------------------------------------------------------
    # Job generation and enqueueing
    # ------------------------------------------------------------------

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

        # Cap new jobs considering work already queued/in-flight/done this wave
        already_committed = (
            len(self._pending_jobs[target_stage]) + self._in_flight[target_stage]
        )
        if self._wave_start_attempts is not None:
            already_committed += (
                self._stage_attempts[target_stage]
                - self._wave_start_attempts[target_stage]
            )
        budget_remaining = max(0, self.config.wave_attempts - already_committed)
        new_jobs = new_jobs[:budget_remaining]
        self._pending_jobs[target_stage].extend(new_jobs)

    def _enqueue_downstream(
        self, stage_idx: int, new_output_state: VectorState
    ) -> None:
        """Generate jobs for stage_idx+1 using new_output_state as input.

        Used during result draining to keep the pool fed while stragglers
        from the current stage are still in-flight.  The depth-first
        propagation loop in ``run()`` later runs each downstream stage
        with its own budget; ``_make_jobs`` skips already-attempted pairs
        so overflow work is naturally accounted for.
        """
        next_stage = stage_idx + 1
        if next_stage >= len(self.all_stages):
            return

        out_t = new_output_state.as_tuple()
        self.tt.mark_forwarded(stage_idx, [out_t])

        new_jobs = self._make_jobs(next_stage, [new_output_state])
        self._pending_jobs[next_stage].extend(new_jobs)

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

    # ------------------------------------------------------------------
    # Pool feeding and result draining
    # ------------------------------------------------------------------

    def _submit_jobs(
        self,
        pool: Pool,
        completion_queue,
        pool_target: int,
    ) -> int:
        """Submit jobs to the pool until pool_target in-flight reached.

        Prioritizes lower stage indices.  Enforces per-stage wave budgets:
        a stage that has already consumed ``wave_attempts`` (attempts +
        in-flight) this wave will not have more jobs submitted — they stay
        in the pending deque for the next wave.

        Returns number of jobs submitted.
        """
        total_in_flight = sum(self._in_flight)
        submitted = 0
        budget = self.config.wave_attempts

        for stage_idx in range(len(self.all_stages)):
            pending = self._pending_jobs[stage_idx]
            if not pending:
                continue

            # Per-stage budget check: attempts done + in-flight this wave
            if self._wave_start_attempts is not None:
                used = (
                    self._stage_attempts[stage_idx]
                    + self._in_flight[stage_idx]
                    - self._wave_start_attempts[stage_idx]
                )
                if used >= budget:
                    continue

            while pending and total_in_flight < pool_target:
                # Re-check budget before each submission
                if self._wave_start_attempts is not None:
                    used = (
                        self._stage_attempts[stage_idx]
                        + self._in_flight[stage_idx]
                        - self._wave_start_attempts[stage_idx]
                    )
                    if used >= budget:
                        break

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
            self.tt.record_attempted_pair(stage_idx, input_state.as_tuple(), cand_index)
            self.tt.record_attempt(stage_idx)
            self._stage_attempts[stage_idx] += 1

            new_unique = 0
            for gadget, output_state in gadget_results:
                out_t = output_state.as_tuple()
                is_new = out_t not in self.tt.get_unique_outputs(stage_idx)
                self.tt.add_transition(stage_idx, input_state, output_state, gadget)

                if is_new:
                    new_unique += 1
                    # Feed downstream stage to keep pool busy during stragglers
                    self._enqueue_downstream(stage_idx, output_state)

                    # On-the-fly scoring: new last-stage transition
                    if stage_idx == last_stage:
                        try:
                            self._score_new_paths_for_transition(
                                stage_idx,
                                input_state.as_tuple(),
                                out_t,
                                progress,
                            )
                        except Exception as score_exc:
                            import sys

                            print(
                                f"Warning: on-the-fly scoring failed: {score_exc}",
                                file=sys.stderr,
                            )

            if progress and stage_task_ids:
                tid = stage_task_ids.get(stage_idx)
                if tid is not None:
                    progress.update(tid, advance=1, success=success, unique=new_unique)

        return count

    # ------------------------------------------------------------------
    # Scoring via LLVM-MCA
    # ------------------------------------------------------------------

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

    def _score_complete_paths(self, paths: list[list[tuple]]) -> list[dict]:
        """Score the given complete paths with LLVM-MCA.

        Returns a list of scored path dicts (newly scored only).
        """
        if not paths:
            return []

        # Register as scored
        for path in paths:
            self._scored_path_keys.add(tuple(tuple(step) for step in path))

        scored: list[dict] = []
        for path in paths:
            path_key = [
                [
                    stage,
                    [list(in_t[0]), list(in_t[1])],
                    [list(out_t[0]), list(out_t[1])],
                ]
                for stage, in_t, out_t in path
            ]
            scored.append({"path_key": path_key, "path": path, "scores": {}})

        if not self.config.target_cpus:
            return scored

        # Late imports to avoid circular deps and allow optional LLVM-MCA
        try:
            from perf_estimator import (
                generate_solution_asm,
                sanitize_asm_for_llvm_mca,
            )
            from llvm_mca_runner import find_llvm_mca, run_llvm_mca
            from cost_model import resolve_llvm_mca_cpu
        except ImportError:
            from .perf_estimator import (  # type: ignore[no-redef]
                generate_solution_asm,
                sanitize_asm_for_llvm_mca,
            )
            from .llvm_mca_runner import find_llvm_mca, run_llvm_mca  # type: ignore[no-redef]
            from .cost_model import resolve_llvm_mca_cpu  # type: ignore[no-redef]

        mca_bin = find_llvm_mca(self.config.llvm_mca_path)
        if mca_bin is None:
            return scored

        def _instr_count_scorer(_path, assignments):
            return sum(g.instruction_count() for g in assignments)

        # Pre-build ASM for each path
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
                    if cycles > 0 and (prev is None or cycles < prev["cycles"]):
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

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def _select_target_stage(self) -> int | None:
        """Select the stage to target for the next wave.

        - First wave always targets Stage 0.
        - Subsequent waves: weakest stage (by success rate).
        - Bubble-up: if a stage has consecutive_zero_budgets >= 2
          and is not stage 0, go upstream.
        """
        if self.wave_count == 0:
            return 0

        excluded = self.exhausted_stages | self._stalled_stages
        target = self.tt.weakest_stage(exclude_exhausted=excluded)
        if target is None:
            return None

        sd = self.tt.stages[target]
        if sd.consecutive_zero_budgets >= 2:
            upstream = target - 1
            while upstream >= 0:
                if upstream not in excluded:
                    return upstream
                upstream -= 1
            predecessors_exhausted = all(
                s in self.exhausted_stages for s in range(target)
            )
            if predecessors_exhausted:
                self.exhausted_stages.add(target)
            else:
                self._stalled_stages.add(target)
            excluded = self.exhausted_stages | self._stalled_stages
            return self.tt.weakest_stage(exclude_exhausted=excluded)

        return target

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_stage_in_wave(
        self,
        stage_idx: int,
        pool: Pool,
        completion_queue,
        pool_target: int,
        progress: SuccessProgress | None,
        stage_task_ids: dict[int, int] | None,
    ) -> int:
        """Run one stage within a wave, respecting budget limits.

        Submits jobs for *stage_idx* and drains results until the stage's
        budget is exhausted (wave_attempts attempts or wave_outputs new
        outputs) or all jobs are done.  The pool is shared so stragglers
        at this stage don't block — downstream jobs from earlier stages
        still get processed.

        Returns the number of new unique outputs discovered at this stage.
        """
        budget_attempts = self.config.wave_attempts
        budget_outputs = self.config.wave_outputs

        # Use wave-start snapshot for consistent budget tracking.
        # This ensures overflow work from _enqueue_downstream is counted.
        wave_start = (
            self._wave_start_attempts[stage_idx]
            if self._wave_start_attempts is not None
            else self._stage_attempts[stage_idx]
        )
        outputs_at_wave_start = self.tt.unique_output_count(stage_idx)

        while True:
            if self._interrupted:
                break

            self._submit_jobs(pool, completion_queue, pool_target)
            drained = self._drain_results(completion_queue, progress, stage_task_ids)

            # Update progress totals
            if drained > 0:
                self._sync_progress_totals(progress, stage_task_ids)

            # Check budget limits for this stage (since wave start, not call start)
            attempts_this_wave = self._stage_attempts[stage_idx] - wave_start
            new_outputs = self.tt.unique_output_count(stage_idx) - outputs_at_wave_start

            if attempts_this_wave >= budget_attempts:
                break
            if new_outputs >= budget_outputs:
                break

            # Stage done if nothing in-flight and either no pending work
            # or budget prevents further submission
            if self._in_flight[stage_idx] == 0:
                if not self._pending_jobs[stage_idx]:
                    break
                # Budget exhausted — leftover pending jobs stay for next wave
                used = attempts_this_wave + self._in_flight[stage_idx]
                if used >= budget_attempts:
                    break

            if drained == 0:
                time.sleep(0.05)

        # Checkpoint this stage
        self._save_checkpoint()

        return self.tt.unique_output_count(stage_idx) - outputs_at_wave_start

    def _sync_progress_totals(
        self,
        progress: SuccessProgress | None,
        stage_task_ids: dict[int, int] | None,
    ) -> None:
        """Update progress bar totals from actual counts."""
        if not progress or not stage_task_ids:
            return
        for s in range(len(self.all_stages)):
            total = (
                self._stage_attempts[s]
                + len(self._pending_jobs[s])
                + self._in_flight[s]
            )
            tid = stage_task_ids[s]
            if total > 0:
                progress.start_task(tid)
                progress.update(tid, total=total)

    def run(self) -> dict:
        """Main loop: long-lived pool, depth-first waves with per-stage budgets.

        Each wave:
        1. Selects target stage
        2. Runs target stage with budget (wave_attempts / wave_outputs)
        3. Propagates outputs depth-first: each downstream stage gets its
           own budget, continuing until the last stage or a dead end
        4. Checkpoints after each stage completes
        5. Scores on the fly as last-stage transitions form complete paths
        6. Logs wave summary, then selects the next wave target

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
                        total=self._stage_attempts[s],
                        start=False,
                        successes=0,
                        unique=0,
                    )
                    stage_task_ids[s] = tid

                # Seed pending queues from unforwarded outputs (resume case)
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

                    # Snapshot per-stage attempts so budget checks know
                    # how many attempts each stage has consumed this wave
                    self._wave_start_attempts = list(self._stage_attempts)

                    # Generate jobs for the target stage
                    self._generate_jobs_for_stage(target)
                    self._sync_progress_totals(progress, stage_task_ids)

                    task_id = stage_task_ids.get(target)
                    if task_id is not None:
                        progress.update(
                            task_id,
                            description=(
                                f"[bold]Stage {target}[/bold] "
                                f"(wave {self.wave_count})"
                            ),
                        )

                    # Run target stage with budget
                    new_outputs = self._run_stage_in_wave(
                        target,
                        pool,
                        completion_queue,
                        pool_target,
                        progress,
                        stage_task_ids,
                    )

                    # Track consecutive zero-output budgets
                    sd = self.tt.stages[target]
                    if new_outputs == 0:
                        sd.consecutive_zero_budgets += 1
                    else:
                        sd.consecutive_zero_budgets = 0
                        for s in range(target + 1, len(self.all_stages)):
                            self._stalled_stages.discard(s)

                    # Depth-first propagation: run each downstream stage
                    # with its own budget.  Overflow work from _enqueue_downstream
                    # may have already enqueued jobs and marked outputs as forwarded,
                    # so we check both unforwarded outputs AND pending/in-flight work.
                    propagation: dict[int, int] = {}
                    for stage_idx in range(target + 1, len(self.all_stages)):
                        if self._interrupted:
                            break

                        # Generate jobs from any new unforwarded outputs
                        unforwarded = self.tt.get_unforwarded_outputs(stage_idx - 1)
                        if unforwarded:
                            input_states = [vs for _tup, vs in unforwarded]
                            output_tuples = [tup for tup, _vs in unforwarded]
                            self.tt.mark_forwarded(stage_idx - 1, output_tuples)

                            downstream_jobs = self._make_jobs(stage_idx, input_states)
                            self._pending_jobs[stage_idx].extend(downstream_jobs)

                        # Skip if no work at all (neither from overflow nor new)
                        has_work = (
                            self._pending_jobs[stage_idx]
                            or self._in_flight[stage_idx] > 0
                        )
                        if not has_work:
                            continue

                        self._sync_progress_totals(progress, stage_task_ids)
                        tid = stage_task_ids.get(stage_idx)
                        if tid is not None:
                            progress.update(
                                tid,
                                description=(
                                    f"[bold]Stage {stage_idx}[/bold] "
                                    f"(wave {self.wave_count})"
                                ),
                            )

                        stage_new = self._run_stage_in_wave(
                            stage_idx,
                            pool,
                            completion_queue,
                            pool_target,
                            progress,
                            stage_task_ids,
                        )

                        if stage_new > 0:
                            propagation[stage_idx] = stage_new

                    # --- Drain all remaining in-flight jobs ---
                    # Before selecting the next wave target, all in-flight
                    # work must complete so the TransitionTable is in a
                    # consistent state for target selection.
                    while sum(self._in_flight) > 0:
                        if self._interrupted:
                            break
                        drained = self._drain_results(
                            completion_queue, progress, stage_task_ids
                        )
                        if drained > 0:
                            self._sync_progress_totals(progress, stage_task_ids)
                        else:
                            time.sleep(0.05)

                    # --- Wave complete ---
                    self._wave_start_attempts = None
                    self._save_checkpoint()

                    # Update progress for exhausted stages
                    for s in range(len(self.all_stages)):
                        if s in self.exhausted_stages:
                            tid = stage_task_ids.get(s)
                            if tid is not None:
                                cum = self._stage_attempts[s] or 1
                                progress.update(
                                    tid,
                                    description=f"Stage {s} (exhausted)",
                                    total=cum,
                                    completed=cum,
                                )

                    # Wave summary
                    stats = self.tt.stage_stats(target)
                    prop_str = ""
                    if propagation:
                        prop_parts = [
                            f"s{s}+{n}" for s, n in sorted(propagation.items())
                        ]
                        prop_str = f" | propagated: {', '.join(prop_parts)}"
                    scored_total = len(self._all_scored_paths)
                    progress.console.print(
                        f"Wave {self.wave_count}: "
                        f"stage {target} targeted, "
                        f"+{new_outputs} outputs "
                        f"({stats['distinct_outputs']} total)"
                        f"{prop_str}"
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

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------

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
            path_key = entry.get("path_key")
            if path_key:
                # path_key is JSON-safe: [[stage, [in_top, in_bot], [out_top, out_bot]], ...]
                # Convert to hashable nested tuples for _scored_path_keys
                key = tuple(
                    (
                        step[0],
                        (tuple(step[1][0]), tuple(step[1][1])),
                        (tuple(step[2][0]), tuple(step[2][1])),
                    )
                    for step in path_key
                )
                self._scored_path_keys.add(key)

        self._rebuild_best_scores_from_accumulated()

    def _rebuild_best_scores_from_accumulated(self) -> None:
        """Rebuild best_scores from all accumulated scored paths."""
        self.best_scores = {}
        for entry in self._all_scored_paths:
            for cpu, scores in entry.get("scores", {}).items():
                cycles = scores.get("cycles", float("inf"))
                if cycles <= 0:
                    continue  # skip LLVM-MCA failure sentinels
                prev = self.best_scores.get(cpu)
                if prev is None or cycles < prev["cycles"]:
                    self.best_scores[cpu] = {
                        "throughput": scores["throughput"],
                        "cycles": cycles,
                        "path_key": entry.get("path_key"),
                    }

    # ------------------------------------------------------------------
    # Export to legacy SolutionNode tree
    # ------------------------------------------------------------------

    def export_to_solution_nodes(self) -> list[SolutionNode]:
        """Convert TransitionTable to SolutionNode tree for legacy export.

        Builds a tree of SolutionNodes by walking the TransitionTable
        stage by stage, wiring children across stages.
        """
        num_stages = len(self.all_stages)
        if num_stages == 0:
            return []

        # Build nodes per stage, keyed by (input_tuple, output_tuple)
        nodes_by_stage: list[dict[tuple, SolutionNode]] = [
            {} for _ in range(num_stages)
        ]

        for stage_idx in range(num_stages):
            all_trans = self.tt.get_all_transitions(stage_idx)
            for (inp_t, out_t), gadgets in all_trans.items():
                input_state = VectorState(top=list(inp_t[0]), bottom=list(inp_t[1]))
                output_state = VectorState(top=list(out_t[0]), bottom=list(out_t[1]))
                node = SolutionNode(
                    stage=stage_idx,
                    input_state=input_state,
                    output_state=output_state,
                    gadgets=list(gadgets),
                    children=[],
                )
                nodes_by_stage[stage_idx][(inp_t, out_t)] = node

        # Wire children: a node at stage N with output O links to all
        # nodes at stage N+1 whose input matches O
        for stage_idx in range(num_stages - 1):
            next_nodes = nodes_by_stage[stage_idx + 1]
            for (_inp_t, out_t), node in nodes_by_stage[stage_idx].items():
                for (next_inp_t, _next_out_t), next_node in next_nodes.items():
                    if out_t == next_inp_t:
                        node.children.append(next_node)

        # Root nodes are all stage 0 nodes
        return list(nodes_by_stage[0].values())
