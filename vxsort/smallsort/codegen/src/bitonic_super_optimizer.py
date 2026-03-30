from __future__ import annotations

import os
import queue as queue_mod
from dataclasses import dataclass, field
from multiprocessing import Pool

try:
    from .success_progress import SuccessProgress
    from .bitonic_sorter import BitonicSorter
    from .utils import vector_machine, primitive_type, width_dict
    from .bitonic_types import (
        VectorState,
        InstructionSpec,
        PermutationGadget,
        SolutionNode,
    )
    from .gadget_synthesizer import (
        GadgetSynthesizer,
        _dispatch_intrinsic_by_signature,
        _match_dispatch_rule,
        get_available_intrinsics,
        _validate_gadget_worker,
        graph_max_depth,
    )
except ImportError:
    from success_progress import SuccessProgress
    from bitonic_sorter import BitonicSorter
    from utils import vector_machine, primitive_type, width_dict
    from bitonic_types import (  # type: ignore
        VectorState,
        InstructionSpec,
        PermutationGadget,
        SolutionNode,
    )
    from gadget_synthesizer import (  # type: ignore
        GadgetSynthesizer,
        _dispatch_intrinsic_by_signature,
        _match_dispatch_rule,
        get_available_intrinsics,
        _validate_gadget_worker,
        graph_max_depth,
    )

# Re-export all public names so existing importers keep working.
__all__ = [
    "VectorState",
    "InstructionSpec",
    "PermutationGadget",
    "SolutionNode",
    "GadgetSynthesizer",
    "_dispatch_intrinsic_by_signature",
    "_match_dispatch_rule",
    "get_available_intrinsics",
    "_validate_gadget_worker",
    "BitonicSuperVectorizer",
    "StageTracker",
    "_BATCH_THRESHOLDS",
    "graph_max_depth",
]


# ---------------------------------------------------------------------------
# Pipeline support: geometric-series thresholds
# ---------------------------------------------------------------------------

_BATCH_THRESHOLDS = (0.5, 0.75, 0.875, 1.0)


@dataclass
class StageTracker:
    """Tracks progress and accumulated results for one pipelined stage."""

    stage_idx: int
    stage_pairs: list  # comparison pairs from bitonic_sorter

    # Job tracking
    total_jobs_submitted: int = 0
    completed_jobs: int = 0

    # Result accumulation (raw results awaiting batch processing)
    unprocessed_results: list = field(default_factory=list)

    # Incremental Phase 3/4 state
    gadgets_by_transition: dict = field(default_factory=dict)
    nodes_by_parent: dict = field(default_factory=dict)
    unique_outputs: dict = field(default_factory=dict)

    # Prevent duplicate forwarding to next stage
    forwarded_output_tuples: set = field(default_factory=set)

    # Threshold tracking
    next_threshold_index: int = 0

    # Lifecycle
    all_inputs_received: bool = False  # True when prior stage finalizes
    finalized: bool = False  # True when all results processed

    # Total expected jobs (may increase as prior stage discovers more outputs)
    total_jobs_expected: int | None = None

    # Progress
    progress_task_id: int | None = None

    # Depth-stratified search
    shallow_complete: bool = False
    depth_escalated: bool = False
    all_input_states: list = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True when all submitted jobs are done AND no more inputs coming."""
        return (
            self.all_inputs_received
            and self.total_jobs_expected is not None
            and self.completed_jobs >= self.total_jobs_expected
        )

    @property
    def completion_fraction(self) -> float:
        if self.total_jobs_submitted == 0:
            return 0.0
        return self.completed_jobs / self.total_jobs_submitted

    def should_process_batch(self) -> bool:
        """Check if we've crossed the next geometric threshold."""
        if self.next_threshold_index >= len(_BATCH_THRESHOLDS):
            return False
        return self.completion_fraction >= _BATCH_THRESHOLDS[self.next_threshold_index]


class BitonicSuperVectorizer:
    """Super-optimizer for bitonic sorting networks using Z3-based gadget synthesis."""

    def __init__(
        self,
        num_vecs: int,
        prim_type: primitive_type,
        vm: vector_machine,
        smt2_dump_dir: str | None = None,
    ):
        self.num_vecs = num_vecs
        self.prim_type = prim_type
        self.vm = vm
        self.smt2_dump_dir = smt2_dump_dir

        # Calculate total elements and elements per vector
        self.elements_per_vector = width_dict[vm] // int(prim_type.value[0])
        self.total_elements = num_vecs * self.elements_per_vector

        # Initialize bitonic sorter to get comparison pairs per stage
        self.bitonic_sorter = BitonicSorter(self.total_elements)

        # Initialize gadget synthesizer
        self.synthesizer = GadgetSynthesizer(vm, prim_type)
        # Single-use contract: one synthesis run per instance to avoid mutable
        # run-state carryover between calls.
        self._synthesis_started = False

        print(
            f"BitonicSuperVectorizer: {num_vecs} x {vm.name} vectors, {self.elements_per_vector} x {prim_type.name} elements per vector, {self.total_elements} total elements, {len(self.bitonic_sorter.stages)} stages"
        )

    def _create_initial_state(self) -> VectorState:
        """
        Create initial vector state based on first stage pairs.

        The initial state is constructed from the FIRST stage's comparison pairs.
        For each pair (a, b), element a goes to top vector and element b goes to bottom.
        This ensures the first stage requires no permutation (0-instruction gadget).
        """
        first_stage_pairs = self.bitonic_sorter.stages[0]

        # Initialize empty lists for top and bottom vectors
        top = []
        bottom = []

        # For each comparison pair in the first stage:
        # - First element goes to top vector
        # - Second element goes to bottom vector
        for pair in first_stage_pairs:
            top.append(pair[0])
            bottom.append(pair[1])

        # Verify we have the expected number of elements
        assert (
            len(top) == self.elements_per_vector
        ), f"Expected {self.elements_per_vector} elements in top, got {len(top)}"
        assert (
            len(bottom) == self.elements_per_vector
        ), f"Expected {self.elements_per_vector} elements in bottom, got {len(bottom)}"

        return VectorState(top=top, bottom=bottom)

    def build_solution_tree(
        self,
        depth_limit: int | None = None,
        gadget_depth: int = 1,
        natural_order: bool = False,
        max_unique_outputs: int = 3,
        checkpoint_dir: str | None = None,
        resume_data: dict | None = None,
        max_workers: int | None = None,
        max_tasks_per_child: int | None = 1000,
        depth2_threshold: float = 0.1,
    ) -> tuple[list[SolutionNode], bool]:
        """
        Iteratively explore all stage transitions to build solution tree.
        Returns (root_nodes, all_stages_complete).

        Args:
            depth_limit: Maximum stage depth to explore (inclusive). If None, all stages are explored.
            gadget_depth: Maximum instruction depth per gadget (1-3, default 1).
            natural_order: If True, append a final stage that restores natural
                element order (1..N in top, N+1..2N in bottom) for memory writeback.
            max_unique_outputs: Number of smallest unique output states to enumerate
                per template for deterministic diversity. Default: 3.
            checkpoint_dir: Directory to save checkpoints after each stage completes.
                If None, no checkpoints are saved.
            resume_data: If provided, resume from a previous checkpoint. Expected
                keys: ``per_stage_data`` and ``last_completed_stage``.
            max_workers: Maximum number of worker processes. If None, defaults
                to os.cpu_count().
            max_tasks_per_child: Maximum tasks per worker process before recycling.
                Limits memory growth in long runs. None disables recycling.
            depth2_threshold: Coverage threshold for depth-2 escalation.
                After depth-1 completes for a stage, coverage is computed as
                unique_outputs / 2^K (K = number of comparison pairs).
                If coverage >= threshold, depth-2 is skipped for that stage.
                Default: 0.1 (10%).

        Returns:
            Tuple of (root nodes, all_stages_complete) where all_stages_complete
            is True if every expected stage was solved successfully.

        Note:
            This method is single-use per ``BitonicSuperVectorizer`` instance.
            Create a new instance for each synthesis run.
        """
        if self._synthesis_started:
            raise RuntimeError(
                "BitonicSuperVectorizer instances are single-use. "
                "Create a new instance for each synthesis run."
            )
        self._synthesis_started = True
        self._max_workers = max_workers
        self._max_tasks_per_child = max_tasks_per_child

        # Inject natural-order stage if requested
        self._natural_order_stage = None
        if natural_order:
            natural_pairs = [
                (i + 1, self.elements_per_vector + i + 1)
                for i in range(self.elements_per_vector)
            ]
            new_stage = max(self.bitonic_sorter.stages.keys()) + 1
            self.bitonic_sorter.stages[new_stage] = natural_pairs
            self._natural_order_stage = new_stage

        # Track which stages produce at least one valid gadget
        self._stages_completed: set[int] = set()

        initial_state = self._create_initial_state()
        # Pre-compute candidates split by depth tier
        shallow_candidates, deep_candidates = (
            self.synthesizer.precompute_candidates_stratified(gadget_depth)
        )

        # Build checkpoint config if checkpoint_dir is provided
        checkpoint_config = None
        if checkpoint_dir is not None:
            try:
                from checkpoint import CheckpointConfig
            except ImportError:
                from .checkpoint import CheckpointConfig  # type: ignore[no-redef]
            checkpoint_config = CheckpointConfig(
                num_vecs=self.num_vecs,
                vm=self.vm.name,
                prim_type=self.prim_type.name,
                gadget_depth=gadget_depth,
                natural_order=natural_order,
                max_unique_outputs=max_unique_outputs,
                depth_limit=depth_limit,
            )

        # Start with empty parent path for the root
        input_states_with_context = [(initial_state, ())]

        nodes_by_path = self._build_tree_pipelined(
            input_states_with_context,
            depth_limit,
            shallow_candidates,
            deep_candidates,
            depth2_threshold=depth2_threshold,
            max_unique_outputs=max_unique_outputs,
            checkpoint_dir=checkpoint_dir,
            checkpoint_config=checkpoint_config,
            resume_data=resume_data,
        )

        # Determine expected stages
        total_stages = len(self.bitonic_sorter.stages)
        expected_stages = total_stages
        if depth_limit is not None:
            expected_stages = min(depth_limit, total_stages)

        all_stages_complete = self._stages_completed == set(range(expected_stages))

        # Return root nodes (those with empty parent path)
        return nodes_by_path.get((), []), all_stages_complete

    # ------------------------------------------------------------------
    # Pipelined helpers
    # ------------------------------------------------------------------

    def _make_jobs_for_inputs(
        self,
        input_states_with_context: list[tuple[VectorState, tuple]],
        stage_idx: int,
        stage_pairs: list,
        perm_gadget_candidates: list[tuple],
        max_unique_outputs: int,
        exclude_outputs: list[tuple] | None = None,
    ) -> list[tuple]:
        """Create validation jobs for a set of input states (Phase 1).

        Args:
            exclude_outputs: Optional list of (top_tuple, bottom_tuple) output
                states to exclude from solver enumeration. Workers will skip
                these outputs, increasing diversity of newly-found solutions.
        """
        jobs = []
        for input_state, parent_path in input_states_with_context:
            metadata = {
                "input_state": input_state,
                "parent_path": parent_path,
                "stage_idx": stage_idx,
                "max_unique_outputs": max_unique_outputs,
            }
            if exclude_outputs:
                metadata["exclude_outputs"] = exclude_outputs
            if (
                self._natural_order_stage is not None
                and stage_idx == self._natural_order_stage
            ):
                metadata["allow_any_lane_order"] = False
            if self.smt2_dump_dir:
                metadata["smt2_dump_dir"] = self.smt2_dump_dir

            for graph in perm_gadget_candidates:
                jobs.append(
                    (
                        graph,
                        input_state,
                        stage_pairs,
                        self.vm,
                        self.prim_type,
                        metadata,
                    )
                )
        return jobs

    @staticmethod
    def _submit_stage_jobs(
        pool: Pool,
        tracker: StageTracker,
        jobs: list[tuple],
        pending_count: list[int],
        completion_queue: queue_mod.Queue,
    ) -> None:
        """Submit jobs to the pool and register callbacks for completion."""
        for job in jobs:
            pool.apply_async(
                _validate_gadget_worker,
                (job,),
                callback=lambda result, s=tracker.stage_idx: completion_queue.put(
                    (s, result, None)
                ),
                error_callback=lambda exc, s=tracker.stage_idx: completion_queue.put(
                    (s, None, exc)
                ),
            )
        tracker.total_jobs_submitted += len(jobs)
        pending_count[0] += len(jobs)

    @staticmethod
    def _process_batch(tracker: StageTracker) -> list[tuple[VectorState, tuple]]:
        """Drain unprocessed results, group by transition, return NEW unique outputs.

        Results are accumulated into the tracker's gadgets_by_transition,
        nodes_by_parent, and unique_outputs dicts.

        Returns list of (output_state, canonical_parent_path) for outputs
        not yet forwarded to the next stage.
        """
        results = tracker.unprocessed_results
        tracker.unprocessed_results = []

        # Phase 3 (incremental): group by transition
        for gadget_results, job_input_state, job_metadata, _ct, _st in results:
            parent_path = job_metadata["parent_path"]
            for gadget, output_state in gadget_results:
                key = (
                    parent_path,
                    job_input_state.as_tuple(),
                    output_state.as_tuple(),
                )
                if key not in tracker.gadgets_by_transition:
                    tracker.gadgets_by_transition[key] = []
                tracker.gadgets_by_transition[key].append(gadget)

                # Phase 4 (incremental): track unique outputs
                out_tuple = output_state.as_tuple()
                if out_tuple not in tracker.unique_outputs:
                    tracker.unique_outputs[out_tuple] = output_state

        # Identify NEW outputs not yet forwarded
        new_outputs = []
        for out_tuple, state in tracker.unique_outputs.items():
            if out_tuple not in tracker.forwarded_output_tuples:
                tracker.forwarded_output_tuples.add(out_tuple)
                new_outputs.append((state, ("canonical", out_tuple)))

        tracker.next_threshold_index += 1
        return new_outputs

    @staticmethod
    def _finalize_stage(tracker: StageTracker) -> None:
        """Sort transitions and gadgets for deterministic output.

        Must be called exactly once after all results are processed.
        Rebuilds nodes_by_parent from the accumulated gadgets_by_transition
        in sorted order, matching the sequential path's output.
        """
        tracker.nodes_by_parent = {}

        sorted_transitions = sorted(
            tracker.gadgets_by_transition.items(), key=lambda x: x[0]
        )

        for (parent_path, input_tuple, output_tuple), gadgets in sorted_transitions:
            input_state = VectorState(
                top=list(input_tuple[0]), bottom=list(input_tuple[1])
            )
            output_state = VectorState(
                top=list(output_tuple[0]), bottom=list(output_tuple[1])
            )

            gadgets.sort(key=lambda g: g.sort_key())

            node = SolutionNode(
                stage=tracker.stage_idx,
                input_state=input_state,
                output_state=output_state,
                gadgets=gadgets,
                children=[],
            )

            if parent_path not in tracker.nodes_by_parent:
                tracker.nodes_by_parent[parent_path] = []
            tracker.nodes_by_parent[parent_path].append(node)

        tracker.finalized = True

    @staticmethod
    def _cancel_downstream_stages(
        from_stage: int,
        effective_limit: int,
        cancelled_stages: set[int],
        pending_count: list[int],
        trackers: dict,
        progress: SuccessProgress,
        stage_task_ids: dict[int, int],
    ) -> None:
        """Mark downstream stages as cancelled and adjust pending count."""
        for s in range(from_stage + 1, effective_limit):
            if s in trackers:
                tracker = trackers[s]
                remaining = tracker.total_jobs_submitted - tracker.completed_jobs
                pending_count[0] -= remaining
                cancelled_stages.add(s)
            if s in stage_task_ids:
                progress.update(stage_task_ids[s], visible=False)

    def _build_tree_pipelined(
        self,
        initial_inputs: list[tuple[VectorState, tuple]],
        depth_limit: int | None,
        shallow_candidates: list[tuple],
        deep_candidates: list[tuple],
        depth2_threshold: float = 0.1,
        max_unique_outputs: int = 3,
        checkpoint_dir: str | None = None,
        checkpoint_config=None,
        resume_data: dict | None = None,
    ) -> dict[tuple, list[SolutionNode]]:
        """Build the solution tree with pipelined stage processing.

        Uses an event-driven pipeline where stage N+1 jobs are submitted as
        soon as unique outputs from stage N are discovered (at geometric-series
        thresholds), allowing overlap.

        Depth-stratified search: shallow_candidates (depth <= 1) are tried
        first. If coverage after depth-1 is below depth2_threshold,
        deep_candidates (depth >= 2) are submitted for that stage.

        The final output is deterministic: ``_finalize_stage`` sorts transitions
        and gadgets in canonical order.
        """
        per_stage_data: dict[
            int, tuple[dict[tuple, list[SolutionNode]], dict[tuple, VectorState]]
        ] = {}

        effective_limit = len(self.bitonic_sorter.stages)
        if depth_limit is not None:
            effective_limit = min(depth_limit, effective_limit)

        # Handle resume
        start_stage = 0
        current_inputs = initial_inputs
        if resume_data is not None:
            per_stage_data = resume_data["per_stage_data"]
            start_stage = resume_data["last_completed_stage"] + 1
            last_stage = resume_data["last_completed_stage"]
            _, unique_outputs = per_stage_data[last_stage]
            current_inputs = [
                (state, ("canonical", out_tuple))
                for out_tuple, state in unique_outputs.items()
            ]

        # Create persistent multi-stage progress display
        progress = SuccessProgress.create(
            width=60,
            success_style="green",
            attempt_style="yellow",
            success_label="Valid",
        )
        progress.enable_memory_monitor()

        # Pre-create one task per stage
        stage_task_ids: dict[int, int] = {}
        for s in range(effective_limit):
            if s < start_stage:
                tid = progress.add_task(
                    f"Stage {s} (resumed)",
                    total=1,
                    completed=1,
                    successes=1,
                    unique=0,
                )
            else:
                tid = progress.add_task(
                    f"Stage {s}",
                    total=None,
                    start=False,
                    successes=0,
                    unique=0,
                )
            stage_task_ids[s] = tid

        trackers: dict[int, StageTracker] = {}
        completion_queue: queue_mod.Queue = queue_mod.Queue()
        pending_count = [0]
        cancelled_stages: set[int] = set()
        highest_checkpointed_stage = start_stage - 1

        def _maybe_checkpoint() -> None:
            """Save checkpoint if a contiguous prefix of stages is finalized."""
            nonlocal highest_checkpointed_stage
            if checkpoint_dir is None or checkpoint_config is None:
                return
            try:
                from checkpoint import save_checkpoint, checkpoint_filename
            except ImportError:
                from .checkpoint import save_checkpoint, checkpoint_filename  # type: ignore[no-redef]

            s = highest_checkpointed_stage + 1
            while s in trackers and trackers[s].finalized:
                t = trackers[s]
                per_stage_data[s] = (t.nodes_by_parent, t.unique_outputs)
                s += 1
            new_watermark = s - 1
            if new_watermark > highest_checkpointed_stage:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    checkpoint_filename(
                        checkpoint_config.num_vecs,
                        checkpoint_config.vm,
                        checkpoint_config.prim_type,
                        checkpoint_config.natural_order,
                        new_watermark,
                    ),
                )
                save_checkpoint(
                    ckpt_path,
                    checkpoint_config,
                    per_stage_data,
                    sorted(
                        s_idx
                        for s_idx in range(new_watermark + 1)
                        if s_idx in per_stage_data
                    ),
                    new_watermark,
                )
                highest_checkpointed_stage = new_watermark

        def _finalize_and_cascade(stage_idx: int) -> None:
            """Finalize a stage and cascade to downstream stages if ready."""
            tracker = trackers[stage_idx]

            # Process any remaining unprocessed results
            if tracker.unprocessed_results:
                new_outputs = self._process_batch(tracker)
                if new_outputs and stage_idx + 1 < effective_limit:
                    _forward_outputs(stage_idx + 1, new_outputs)

            self._finalize_stage(tracker)

            if tracker.nodes_by_parent:
                self._stages_completed.add(stage_idx)
                per_stage_data[stage_idx] = (
                    tracker.nodes_by_parent,
                    tracker.unique_outputs,
                )
            else:
                per_stage_data[stage_idx] = ({}, {})
                # Empty stage: cancel everything downstream
                self._cancel_downstream_stages(
                    stage_idx,
                    effective_limit,
                    cancelled_stages,
                    pending_count,
                    trackers,
                    progress,
                    stage_task_ids,
                )

            progress.console.print(
                f"Stage {stage_idx}: Finalized — "
                f"{len(tracker.gadgets_by_transition)} transitions, "
                f"{len(tracker.unique_outputs)} unique outputs"
            )

            # Cascade: mark next stage as having all inputs
            if stage_idx + 1 in trackers:
                nxt = trackers[stage_idx + 1]
                nxt.all_inputs_received = True
                nxt.total_jobs_expected = nxt.total_jobs_submitted
                if nxt.is_complete and not nxt.finalized:
                    _finalize_and_cascade(stage_idx + 1)

            _maybe_checkpoint()

        def _forward_outputs(
            next_stage_idx: int,
            new_outputs: list[tuple[VectorState, tuple]],
        ) -> None:
            """Submit shallow jobs for newly-discovered outputs to the next stage."""
            if next_stage_idx >= effective_limit:
                return

            stage_pairs = self.bitonic_sorter.stages[next_stage_idx]
            is_new = next_stage_idx not in trackers

            if is_new:
                tracker = StageTracker(
                    stage_idx=next_stage_idx, stage_pairs=stage_pairs
                )
                tracker.progress_task_id = stage_task_ids.get(next_stage_idx)
                trackers[next_stage_idx] = tracker
            else:
                tracker = trackers[next_stage_idx]

            # Common: create jobs, submit, record inputs
            jobs = self._make_jobs_for_inputs(
                new_outputs,
                next_stage_idx,
                stage_pairs,
                shallow_candidates,
                max_unique_outputs,
            )
            self._submit_stage_jobs(
                pool, tracker, jobs, pending_count, completion_queue
            )
            tracker.all_input_states.extend(new_outputs)

            if is_new:
                if tracker.progress_task_id is not None:
                    progress.start_task(tracker.progress_task_id)
                    progress.update(
                        tracker.progress_task_id,
                        total=tracker.total_jobs_submitted,
                    )
                progress.console.print(
                    f"Stage {next_stage_idx}: Launched {tracker.total_jobs_submitted} jobs "
                    f"(pipelined from stage {next_stage_idx - 1})"
                )
            else:
                if tracker.progress_task_id is not None:
                    progress.update(
                        tracker.progress_task_id,
                        total=tracker.total_jobs_submitted,
                    )
                progress.console.print(
                    f"Stage {next_stage_idx}: Added {len(jobs)} jobs "
                    f"(total: {tracker.total_jobs_submitted})"
                )

        # Main event loop
        with progress:
            if resume_data is not None:
                progress.console.print(
                    f"Resuming from stage {start_stage} with {len(current_inputs)} inputs"
                )

            pool = Pool(
                processes=self._max_workers,
                maxtasksperchild=self._max_tasks_per_child,
            )
            try:
                # Launch stage 0 (or first stage after resume)
                if start_stage < effective_limit:
                    new_tracker = StageTracker(
                        stage_idx=start_stage,
                        stage_pairs=self.bitonic_sorter.stages[start_stage],
                    )
                    new_tracker.progress_task_id = stage_task_ids.get(start_stage)
                    new_tracker.all_inputs_received = True  # first stage has all inputs
                    trackers[start_stage] = new_tracker

                    jobs = self._make_jobs_for_inputs(
                        current_inputs,
                        start_stage,
                        new_tracker.stage_pairs,
                        shallow_candidates,
                        max_unique_outputs,
                    )
                    self._submit_stage_jobs(
                        pool, new_tracker, jobs, pending_count, completion_queue
                    )
                    new_tracker.total_jobs_expected = new_tracker.total_jobs_submitted
                    new_tracker.all_input_states = list(current_inputs)

                    if new_tracker.progress_task_id is not None:
                        progress.start_task(new_tracker.progress_task_id)
                        progress.update(
                            new_tracker.progress_task_id,
                            total=new_tracker.total_jobs_submitted,
                        )

                    progress.console.print(
                        f"Stage {start_stage}: Launched {new_tracker.total_jobs_submitted} jobs "
                        f"for {len(current_inputs)} inputs"
                    )

                def _escalate_stage(stage_idx: int) -> None:
                    """Escalate a stage to depth-2 if coverage is below threshold."""
                    if not deep_candidates:
                        return
                    tracker = trackers[stage_idx]
                    if tracker.depth_escalated:
                        return
                    tracker.depth_escalated = True

                    # Coverage check: unique_outputs / 2^K
                    k = len(tracker.stage_pairs)
                    comparison_space = 2**k
                    n_outputs = len(tracker.unique_outputs)
                    coverage = n_outputs / comparison_space

                    if coverage >= depth2_threshold:
                        progress.console.print(
                            f"Stage {stage_idx}: {n_outputs} unique outputs = "
                            f"{coverage:.1%} of 2^{k} ({comparison_space}) — "
                            f"above {depth2_threshold:.0%} threshold, skipping depth-2"
                        )
                        return

                    progress.console.print(
                        f"Stage {stage_idx}: {n_outputs} unique outputs = "
                        f"{coverage:.1%} of 2^{k} ({comparison_space}) — "
                        f"below {depth2_threshold:.0%} threshold, escalating to depth-2 "
                        f"for {len(tracker.all_input_states)} inputs "
                        f"({len(deep_candidates)} deep candidates)"
                    )

                    # Pass known outputs so depth-2 workers skip them,
                    # increasing diversity of newly-found solutions.
                    known_outputs = list(tracker.unique_outputs.keys())

                    jobs = self._make_jobs_for_inputs(
                        tracker.all_input_states,
                        stage_idx,
                        tracker.stage_pairs,
                        deep_candidates,
                        max_unique_outputs,
                        exclude_outputs=known_outputs,
                    )
                    shallow_jobs = tracker.total_jobs_submitted
                    self._submit_stage_jobs(
                        pool, tracker, jobs, pending_count, completion_queue
                    )
                    deep_jobs = tracker.total_jobs_submitted - shallow_jobs
                    tracker.total_jobs_expected = tracker.total_jobs_submitted
                    tracker.next_threshold_index = 0

                    progress.console.print(
                        f"Stage {stage_idx}: Submitted {deep_jobs} depth-2 jobs "
                        f"(total: {tracker.total_jobs_submitted}, "
                        f"shallow: {shallow_jobs}, deep: {deep_jobs})"
                    )

                    if tracker.progress_task_id is not None:
                        progress.update(
                            tracker.progress_task_id,
                            description=f"Stage {stage_idx} (depth-2)",
                            total=tracker.total_jobs_submitted,
                        )

                    # Free accumulated inputs — no longer needed after job creation
                    tracker.all_input_states = []

                # Event loop: process completed results via callback queue
                while pending_count[0] > 0:
                    stage_idx, result, error = completion_queue.get()
                    pending_count[0] -= 1

                    if stage_idx in cancelled_stages:
                        continue

                    if error is not None:
                        raise error

                    tracker = trackers[stage_idx]
                    tracker.unprocessed_results.append(result)
                    tracker.completed_jobs += 1

                    # Update progress
                    gadget_results = result[0]
                    success_inc = 1 if gadget_results else 0
                    if tracker.progress_task_id is not None:
                        progress.update(
                            tracker.progress_task_id,
                            advance=1,
                            success=success_inc,
                        )

                    # Check threshold
                    if tracker.should_process_batch():
                        threshold_pct = (
                            _BATCH_THRESHOLDS[tracker.next_threshold_index] * 100
                        )
                        new_outputs = self._process_batch(tracker)
                        if new_outputs and stage_idx + 1 < effective_limit:
                            progress.console.print(
                                f"Stage {stage_idx} → {stage_idx + 1}: "
                                f"forwarding {len(new_outputs)} new unique outputs "
                                f"at {threshold_pct:.0f}% completion "
                                f"({tracker.completed_jobs}/{tracker.total_jobs_submitted} jobs)"
                            )
                            _forward_outputs(stage_idx + 1, new_outputs)

                    # Check completion
                    if tracker.is_complete and not tracker.finalized:
                        if (
                            not tracker.shallow_complete
                            and deep_candidates
                            and tracker.all_inputs_received
                        ):
                            # Shallow phase done. Process remaining, then escalate.
                            tracker.shallow_complete = True
                            if tracker.unprocessed_results:
                                new_outputs = self._process_batch(tracker)
                                if new_outputs and stage_idx + 1 < effective_limit:
                                    _forward_outputs(stage_idx + 1, new_outputs)
                            _escalate_stage(stage_idx)
                            # After escalation, re-check: finalize unless deep
                            # jobs made is_complete False again.
                            if tracker.is_complete:
                                _finalize_and_cascade(stage_idx)
                        else:
                            _finalize_and_cascade(stage_idx)

                # Finalize any stages that haven't been finalized yet
                # (e.g., stages with 0 jobs due to no inputs)
                for s in range(start_stage, effective_limit):
                    if s in trackers and not trackers[s].finalized:
                        trackers[s].all_inputs_received = True
                        trackers[s].total_jobs_expected = trackers[
                            s
                        ].total_jobs_submitted
                        _finalize_and_cascade(s)
            finally:
                pool.terminate()
                pool.join()

            # Hide stages that were never launched
            for s in range(start_stage, effective_limit):
                if s not in trackers:
                    progress.update(stage_task_ids[s], visible=False)

        # Ensure per_stage_data is populated from all finalized trackers
        for s_idx, tracker in trackers.items():
            if tracker.finalized and s_idx not in per_stage_data:
                per_stage_data[s_idx] = (
                    tracker.nodes_by_parent,
                    tracker.unique_outputs,
                )

        # Backward pass: wire children (identical to sequential path)
        sorted_stages = sorted(per_stage_data.keys())
        for i in range(len(sorted_stages) - 1):
            stage_idx = sorted_stages[i]
            next_stage_idx = sorted_stages[i + 1]

            nodes_by_parent, _ = per_stage_data[stage_idx]
            next_nodes_by_parent, _ = per_stage_data[next_stage_idx]

            for _parent_path, nodes in nodes_by_parent.items():
                for node in nodes:
                    out_key = node.output_state.as_tuple()
                    canonical_path = ("canonical", out_key)
                    node.children = next_nodes_by_parent.get(canonical_path, [])

        # Collect root stage nodes
        all_nodes = {}
        if sorted_stages:
            all_nodes = per_stage_data[sorted_stages[0]][0]

        return all_nodes

    def synthesize_all_stages(
        self,
        depth_limit: int | None = None,
        gadget_depth: int = 1,
        natural_order: bool = False,
        max_unique_outputs: int = 3,
        checkpoint_dir: str | None = None,
        resume_data: dict | None = None,
        max_workers: int | None = None,
        max_tasks_per_child: int | None = 1000,
        depth2_threshold: float = 0.1,
    ) -> tuple[list[SolutionNode], bool]:
        """Entry point: builds solution tree for all stages.

        Args:
            depth_limit: Maximum stage depth to explore (inclusive). If None, all stages are explored.
            gadget_depth: Maximum instruction depth per gadget (1-3, default 1).
            natural_order: If True, append a final stage restoring natural element order.
            max_unique_outputs: Number of smallest unique output states to enumerate
                per template for deterministic diversity. Default: 3.
            checkpoint_dir: Directory to save checkpoints after each stage completes.
                If None, no checkpoints are saved.
            resume_data: If provided, resume from a previous checkpoint. Expected
                keys: ``per_stage_data`` and ``last_completed_stage``.
            max_workers: Maximum number of worker processes. If None, defaults
                to os.cpu_count().
            max_tasks_per_child: Maximum tasks per worker process before recycling.
                Limits memory growth in long runs. None disables recycling.

        Returns:
            Tuple of (root nodes, all_stages_complete) where all_stages_complete
            is True if every expected stage was solved successfully.

        Note:
            This method is single-use per ``BitonicSuperVectorizer`` instance.
            Create a new instance for each synthesis run.
        """
        return self.build_solution_tree(
            depth_limit=depth_limit,
            gadget_depth=gadget_depth,
            natural_order=natural_order,
            max_unique_outputs=max_unique_outputs,
            checkpoint_dir=checkpoint_dir,
            resume_data=resume_data,
            max_workers=max_workers,
            max_tasks_per_child=max_tasks_per_child,
            depth2_threshold=depth2_threshold,
        )
