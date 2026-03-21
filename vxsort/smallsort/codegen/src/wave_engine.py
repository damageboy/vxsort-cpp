"""WaveEngine: iterative wave-based synthesis orchestrator.

Replaces the pipelined BFS in ``bitonic_super_optimizer.py`` with a
budget-controlled, wave-oriented approach.  Each wave targets the
weakest stage, runs a fixed budget of synthesis jobs, forward-propagates
new outputs to downstream stages, and optionally checkpoints.
"""

from __future__ import annotations

import os
import signal
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
    propagation_divisor: int = 10
    max_paths_per_wave: int = 1000
    target_cpus: list[str] = field(default_factory=list)
    max_workers: int | None = None
    max_tasks_per_child: int | None = None
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
    """Iterative wave-based synthesis engine.

    Each wave targets the weakest stage (by success rate), runs a fixed
    budget of synthesis jobs, forward-propagates new outputs through
    subsequent stages, and optionally checkpoints state.
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

        # Ctrl-C handling
        self._interrupted = False

        # Checkpoint
        self._checkpoint: WaveCheckpoint | None = None
        if config.checkpoint_dir is not None:
            self._checkpoint = WaveCheckpoint(config.checkpoint_dir)

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
    # Job creation and execution
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
                    "max_unique_outputs": 3,
                    "candidate_index": cand_idx,
                }
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

    def _run_stage_budget(
        self,
        stage_idx: int,
        input_states: list[VectorState],
        budget_attempts: int,
        budget_outputs: int,
        progress: SuccessProgress | None = None,
        progress_task_id: int | None = None,
    ) -> int:
        """Run synthesis jobs for a stage within the given budget.

        Returns the number of new distinct outputs discovered.
        """
        import queue as queue_mod

        jobs = self._make_jobs(stage_idx, input_states)

        if not jobs:
            # All candidates exhausted for this stage
            self.exhausted_stages.add(stage_idx)
            if progress is not None and progress_task_id is not None:
                task = progress._tasks.get(progress_task_id)
                total = task.total if task and task.total else 1
                progress.update(
                    progress_task_id,
                    description=f"Stage {stage_idx} (exhausted)",
                    total=total,
                    completed=total,
                )
            return 0

        # Limit to budget
        jobs = jobs[:budget_attempts]

        new_outputs = 0
        tt = self.tt

        raw_jobs = [job for _, job in jobs]

        # Record all attempts and mark pairs before submitting
        for cand_idx, job in jobs:
            input_state = job[1]
            tt.record_attempted_pair(stage_idx, input_state.as_tuple(), cand_idx)

        tt.record_attempt(stage_idx, count=len(raw_jobs))

        # Update progress bar total
        if progress is not None and progress_task_id is not None:
            progress.update(progress_task_id, total=len(raw_jobs))

        # Run via multiprocessing Pool with async callbacks for progress
        completion_queue: queue_mod.Queue = queue_mod.Queue()

        pool = Pool(
            processes=self.config.max_workers,
            maxtasksperchild=self.config.max_tasks_per_child,
        )
        try:
            pending = len(raw_jobs)
            for job in raw_jobs:
                pool.apply_async(
                    _validate_gadget_worker,
                    (job,),
                    callback=lambda r: completion_queue.put(("ok", r)),
                    error_callback=lambda e: completion_queue.put(("err", e)),
                )

            completed = 0
            while completed < pending:
                if new_outputs >= budget_outputs:
                    break
                if self._interrupted:
                    break

                try:
                    status, payload = completion_queue.get(timeout=0.5)
                except queue_mod.Empty:
                    continue
                completed += 1

                if status == "err":
                    if progress is not None and progress_task_id is not None:
                        progress.update(progress_task_id, advance=1, success=0)
                    continue

                gadget_results, input_state, _metadata, _ct, _st = payload
                success = 1 if gadget_results else 0

                if progress is not None and progress_task_id is not None:
                    progress.update(progress_task_id, advance=1, success=success)

                for gadget, output_state in gadget_results:
                    out_t = output_state.as_tuple()
                    is_new_output = out_t not in tt.get_unique_outputs(stage_idx)
                    tt.add_transition(stage_idx, input_state, output_state, gadget)
                    if is_new_output:
                        new_outputs += 1
        finally:
            pool.terminate()
            pool.join()

        return new_outputs

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------

    def _forward_propagate(
        self,
        from_stage: int,
        progress: SuccessProgress | None = None,
        stage_task_ids: dict[int, int] | None = None,
    ) -> None:
        """Propagate new outputs from from_stage through subsequent stages.

        Uses reduced budget (main / propagation_divisor).
        """
        reduced_attempts = max(
            1, self.config.wave_attempts // self.config.propagation_divisor
        )
        reduced_outputs = max(
            1, self.config.wave_outputs // self.config.propagation_divisor
        )

        for stage_idx in range(from_stage + 1, len(self.all_stages)):
            # Get unforwarded outputs from the previous stage
            prev_stage = stage_idx - 1
            unforwarded = self.tt.get_unforwarded_outputs(prev_stage)
            if not unforwarded:
                continue

            # Convert to VectorState list and mark as forwarded
            input_states = [vs for _tup, vs in unforwarded]
            output_tuples = [tup for tup, _vs in unforwarded]
            self.tt.mark_forwarded(prev_stage, output_tuples)

            task_id = stage_task_ids.get(stage_idx) if stage_task_ids else None
            if progress is not None and task_id is not None:
                progress.start_task(task_id)

            # Run with reduced budget
            self._run_stage_budget(
                stage_idx,
                input_states,
                reduced_attempts,
                reduced_outputs,
                progress=progress,
                progress_task_id=task_id,
            )

    # ------------------------------------------------------------------
    # Scoring via LLVM-MCA
    # ------------------------------------------------------------------

    def _score_complete_paths(self) -> list[dict]:
        """Score complete paths through all stages with LLVM-MCA.

        For each target CPU, builds CompletePath objects from the
        TransitionTable, runs LLVM-MCA via the multiprocessing pool,
        and updates best_scores.

        Returns a list of scored path dicts.
        """
        paths = self.tt.enumerate_complete_paths(
            max_paths=self.config.max_paths_per_wave
        )
        if not paths:
            return []

        scored: list[dict] = []
        for i, path in enumerate(paths):
            scored.append(
                {
                    "path_index": i,
                    "path": path,
                    "scores": {},
                }
            )

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

        for target_cpu in self.config.target_cpus:
            mcpu = resolve_llvm_mca_cpu(target_cpu)
            if mcpu is None:
                continue

            # Score each path: build CompletePath, generate ASM, run MCA
            for entry in scored:
                path = entry["path"]

                # Use hill_climb_path with a simple instruction-count scorer
                # to get the initial gadget assignment, then score with MCA
                def _instr_count_scorer(_path, assignments):
                    return sum(g.instruction_count() for g in assignments)

                assignments, _ = hill_climb_path(
                    path, self.tt, _instr_count_scorer, max_passes=1
                )

                # Build a CompletePath for generate_solution_asm
                complete_path = _build_complete_path(path, assignments)

                try:
                    asm = generate_solution_asm(
                        complete_path,
                        self.config.vm,
                        self.config.prim_type,
                        self.config.num_vecs,
                        self.config.natural_order,
                        entry["path_index"] + 1,
                        len(scored),
                    )
                    sanitized = sanitize_asm_for_llvm_mca(asm)
                    result = run_llvm_mca(
                        sanitized, mcpu, mca_bin, entry["path_index"] + 1
                    )
                    entry["scores"][target_cpu] = {
                        "throughput": result.throughput,
                        "cycles": result.simulated_cycles,
                    }

                    # Update best scores
                    prev = self.best_scores.get(target_cpu)
                    if prev is None or result.simulated_cycles < prev["cycles"]:
                        self.best_scores[target_cpu] = {
                            "throughput": result.throughput,
                            "cycles": result.simulated_cycles,
                            "path_index": entry["path_index"],
                        }
                except Exception:
                    pass  # Skip paths that fail ASM generation

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

        target = self.tt.weakest_stage(exclude_exhausted=self.exhausted_stages)
        if target is None:
            return None

        # Bubble-up: if stage is stuck after 2 consecutive zero-output budgets,
        # try going upstream to produce different inputs. Keep bubbling up
        # until we find a non-exhausted stage or reach stage 0.
        sd = self.tt.stages[target]
        if sd.consecutive_zero_budgets >= 2:
            upstream = target - 1
            while upstream >= 0:
                if upstream not in self.exhausted_stages:
                    return upstream
                upstream -= 1
            # All upstream stages exhausted too — mark target as exhausted
            self.exhausted_stages.add(target)
            # Re-select without the newly exhausted stage
            return self.tt.weakest_stage(exclude_exhausted=self.exhausted_stages)

        return target

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        """Save current state to checkpoint directory."""
        if self._checkpoint is None:
            return

        # Build stage stats
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
            self._checkpoint.save_stage(i, self.tt)

    # ------------------------------------------------------------------
    # Single wave
    # ------------------------------------------------------------------

    def run_wave(
        self,
        progress: SuccessProgress | None = None,
        stage_task_ids: dict[int, int] | None = None,
    ) -> dict:
        """Execute one wave: target -> budget -> propagate -> score -> checkpoint.

        Returns a dict with wave results:
          - wave_index: int
          - target_stage: int
          - new_outputs: int
          - total_paths: int
        """
        target = self._select_target_stage()
        if target is None:
            return {
                "wave_index": self.wave_count,
                "target_stage": None,
                "new_outputs": 0,
                "total_paths": 0,
            }

        # Gather input states for the target stage
        if target == 0:
            input_states = [self.initial_state]
        else:
            # Use outputs from the previous stage
            prev_outputs = self.tt.get_unique_outputs(target - 1)
            if prev_outputs:
                input_states = list(prev_outputs.values())
            else:
                input_states = []

        # Update progress description for targeted stage
        task_id = stage_task_ids.get(target) if stage_task_ids else None
        if progress is not None and task_id is not None:
            progress.start_task(task_id)
            progress.update(
                task_id,
                description=f"[bold]Stage {target}[/bold] (wave {self.wave_count})",
            )

        # Run the budget
        new_outputs = self._run_stage_budget(
            target,
            input_states,
            self.config.wave_attempts,
            self.config.wave_outputs,
            progress=progress,
            progress_task_id=task_id,
        )

        # Track consecutive zero-output budgets
        sd = self.tt.stages[target]
        if new_outputs == 0:
            sd.consecutive_zero_budgets += 1
        else:
            sd.consecutive_zero_budgets = 0

        # Forward propagate new outputs
        self._forward_propagate(
            target, progress=progress, stage_task_ids=stage_task_ids
        )

        # Score paths
        scored = self._score_complete_paths()

        # Checkpoint
        self._save_checkpoint()
        if self._checkpoint is not None and scored:
            self._checkpoint.save_scored_paths(scored)

        result = {
            "wave_index": self.wave_count,
            "target_stage": target,
            "new_outputs": new_outputs,
            "total_paths": len(scored),
        }

        self.wave_count += 1
        return result

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Main loop: run waves until termination condition.

        Termination:
          - Ctrl-C (checkpoint and exit)
          - max_waves reached
          - All stages exhausted

        Returns a summary dict with:
          - wave_count: int
          - interrupted: bool
          - waves: list[dict]
        """
        waves: list[dict] = []

        # Install Ctrl-C handler
        original_sigint = signal.getsignal(signal.SIGINT)

        def _sigint_handler(_signum, _frame):
            self._interrupted = True

        signal.signal(signal.SIGINT, _sigint_handler)

        progress = SuccessProgress.create()
        progress.enable_memory_monitor()

        try:
            with progress:
                # Create a progress task for each stage
                stage_task_ids: dict[int, int] = {}
                for s in range(len(self.all_stages)):
                    tid = progress.add_task(
                        f"Stage {s}",
                        total=None,
                        start=False,
                        successes=0,
                    )
                    stage_task_ids[s] = tid

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

                    # Check if all stages exhausted
                    if len(self.exhausted_stages) >= len(self.all_stages):
                        progress.console.print(
                            "[green]All stages exhausted — search complete.[/green]"
                        )
                        break

                    prev_exhausted = set(self.exhausted_stages)

                    result = self.run_wave(
                        progress=progress,
                        stage_task_ids=stage_task_ids,
                    )
                    waves.append(result)

                    # Update progress for any newly exhausted stages
                    for s in self.exhausted_stages - prev_exhausted:
                        tid = stage_task_ids.get(s)
                        if tid is not None:
                            task = progress._tasks.get(tid)
                            total = task.total if task and task.total else 1
                            progress.update(
                                tid,
                                description=f"Stage {s} (exhausted)",
                                total=total,
                                completed=total,
                            )

                    if result["target_stage"] is None:
                        break

                    # Print wave summary
                    target = result["target_stage"]
                    stats = self.tt.stage_stats(target)
                    progress.console.print(
                        f"Wave {result['wave_index']}: "
                        f"stage {target}, "
                        f"+{result['new_outputs']} outputs "
                        f"({stats['distinct_outputs']} total), "
                        f"{result['total_paths']} paths scored",
                    )

                    # Update persistent status line with best scores per CPU
                    if self.best_scores:
                        parts = [
                            f"{cpu}: {s['cycles']:.1f}cy (throughput {s['throughput']:.2f})"
                            for cpu, s in sorted(self.best_scores.items())
                        ]
                        progress.set_status("Best: " + "  |  ".join(parts))
        finally:
            signal.signal(signal.SIGINT, original_sigint)

        return {
            "wave_count": self.wave_count,
            "interrupted": self._interrupted,
            "waves": waves,
        }

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------

    def resume(self, checkpoint_dir: str) -> None:
        """Load state from a checkpoint directory.

        Restores the TransitionTable and wave_count from the checkpoint.
        """
        ckpt = WaveCheckpoint(checkpoint_dir)
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

        master = ckpt.load_master()
        self.wave_count = master.wave_count
        self.best_scores = master.best_scores
        self.exhausted_stages = set(master.exhausted_stages)

        for i in range(len(self.all_stages)):
            stage_path = os.path.join(checkpoint_dir, f"stage_{i:02d}.json.zst")
            if os.path.exists(stage_path):
                ckpt.load_stage(i, self.tt)

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
