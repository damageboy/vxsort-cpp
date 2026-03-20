"""WaveEngine: iterative wave-based synthesis orchestrator.

Replaces the pipelined BFS in ``bitonic_super_optimizer.py`` with a
budget-controlled, wave-oriented approach.  Each wave targets the
weakest stage, runs a fixed budget of synthesis jobs, forward-propagates
new outputs to downstream stages, and optionally checkpoints.
"""

from __future__ import annotations

import os
import signal
import sys
from dataclasses import dataclass, field
from multiprocessing import Pool

try:
    from .bitonic_sorter import BitonicSorter
    from .bitonic_types import (
        SolutionNode,
        VectorState,
    )
    from .gadget_synthesizer import (
        GadgetSynthesizer,
        _validate_gadget_worker,
    )
    from .transition_table import TransitionTable
    from .utils import primitive_type, vector_machine, width_dict
    from .wave_checkpoint import WaveCheckpoint, WaveMasterConfig
except ImportError:
    from bitonic_sorter import BitonicSorter  # type: ignore[no-redef]
    from bitonic_types import (  # type: ignore[no-redef]
        SolutionNode,
        VectorState,
    )
    from gadget_synthesizer import (  # type: ignore[no-redef]
        GadgetSynthesizer,
        _validate_gadget_worker,
    )
    from transition_table import TransitionTable  # type: ignore[no-redef]
    from utils import primitive_type, vector_machine, width_dict  # type: ignore[no-redef]
    from wave_checkpoint import WaveCheckpoint, WaveMasterConfig  # type: ignore[no-redef]


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
    ) -> int:
        """Run synthesis jobs for a stage within the given budget.

        Returns the number of new distinct outputs discovered.
        """
        jobs = self._make_jobs(stage_idx, input_states)

        # Limit to budget
        jobs = jobs[:budget_attempts]
        if not jobs:
            return 0

        new_outputs = 0
        tt = self.tt

        raw_jobs = [job for _, job in jobs]

        # Record all attempts and mark pairs before submitting
        for cand_idx, job in jobs:
            input_state = job[1]
            tt.record_attempted_pair(stage_idx, input_state.as_tuple(), cand_idx)

        tt.record_attempt(stage_idx, count=len(raw_jobs))

        # Run via multiprocessing Pool
        pool = Pool(
            processes=self.config.max_workers,
            maxtasksperchild=self.config.max_tasks_per_child,
        )
        try:
            results = pool.map(_validate_gadget_worker, raw_jobs)
        finally:
            pool.terminate()
            pool.join()

        # Process results
        for result in results:
            gadget_results, input_state, _metadata, _ct, _st = result
            for gadget, output_state in gadget_results:
                out_t = output_state.as_tuple()
                is_new_output = out_t not in tt.get_unique_outputs(stage_idx)
                tt.add_transition(stage_idx, input_state, output_state, gadget)
                if is_new_output:
                    new_outputs += 1

                if new_outputs >= budget_outputs:
                    break
            if new_outputs >= budget_outputs:
                break

        return new_outputs

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------

    def _forward_propagate(self, from_stage: int) -> None:
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

            # Run with reduced budget
            self._run_stage_budget(
                stage_idx, input_states, reduced_attempts, reduced_outputs
            )

    # ------------------------------------------------------------------
    # Scoring (stub)
    # ------------------------------------------------------------------

    def _score_complete_paths(self) -> list[dict]:
        """Score complete paths through all stages.

        Returns a list of scored path dicts.  Currently a stub --
        LLVM-MCA integration comes in Task 5/6.
        """
        paths = self.tt.enumerate_complete_paths(
            max_paths=self.config.max_paths_per_wave
        )
        scored: list[dict] = []
        for i, path in enumerate(paths):
            scored.append(
                {
                    "path_index": i,
                    "path": path,
                    "scores": {},
                }
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

        target = self.tt.weakest_stage(exclude_exhausted=self.exhausted_stages)
        if target is None:
            return None

        # Bubble-up: if stage is stuck, try the stage before it
        sd = self.tt.stages[target]
        if sd.consecutive_zero_budgets >= 2 and target > 0:
            upstream = target - 1
            if upstream not in self.exhausted_stages:
                return upstream

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

    def run_wave(self) -> dict:
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

        # Run the budget
        new_outputs = self._run_stage_budget(
            target,
            input_states,
            self.config.wave_attempts,
            self.config.wave_outputs,
        )

        # Track consecutive zero-output budgets
        sd = self.tt.stages[target]
        if new_outputs == 0:
            sd.consecutive_zero_budgets += 1
        else:
            sd.consecutive_zero_budgets = 0

        # Forward propagate new outputs
        self._forward_propagate(target)

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

        try:
            while True:
                if self._interrupted:
                    self._save_checkpoint()
                    break

                if (
                    self.config.max_waves is not None
                    and self.wave_count >= self.config.max_waves
                ):
                    break

                # Check if all stages exhausted
                if len(self.exhausted_stages) >= len(self.all_stages):
                    break

                result = self.run_wave()
                waves.append(result)

                if result["target_stage"] is None:
                    # No targetable stage
                    break

                # Print status
                target = result["target_stage"]
                stats = self.tt.stage_stats(target)
                print(
                    f"Wave {result['wave_index']}: "
                    f"stage {target}, "
                    f"+{result['new_outputs']} outputs "
                    f"({stats['distinct_outputs']} total), "
                    f"{result['total_paths']} paths",
                    file=sys.stderr,
                )
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
