"""WaveEngine: iterative wave-based synthesis orchestrator.

Each wave targets the weakest stage, runs a fixed budget of synthesis
jobs via a long-lived shared process pool, forward-propagates new
outputs to downstream stages as they appear, and checkpoints on each
stage completion within the wave.  Scoring happens on the fly as
last-stage transitions form complete paths.
"""

from __future__ import annotations

import os
import queue as queue_mod
import signal
import sys
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import permutations
from multiprocessing import Pool

try:
    from .bitonic_sorter import BitonicSorter
    from .bitonic_types import (
        IntrinsicNode,
        InstructionSpec,
        Mux,
        PermutationGadget,
        SolutionNode,
        VectorState,
    )
    from .instruction_stats import (
        InstructionStatsCollector,
        normalize_instruction_stats_key,
    )
    from .gadget_synthesizer import (
        GadgetSynthesizer,
        _validate_gadget_worker,
    )
    from .util.memory_monitor import collect_memory_snapshot
    from .transition_table import TransitionTable
    from .util.enums import primitive_type, vector_machine, width_dict
    from .wave_checkpoint import WaveCheckpoint, WaveMasterConfig
except ImportError:
    from bitonic_sorter import BitonicSorter  # type: ignore[no-redef]
    from bitonic_types import (  # type: ignore[no-redef]
        IntrinsicNode,
        InstructionSpec,
        Mux,
        PermutationGadget,
        SolutionNode,
        VectorState,
    )
    from instruction_stats import (  # type: ignore[no-redef]
        InstructionStatsCollector,
        normalize_instruction_stats_key,
    )
    from gadget_synthesizer import (  # type: ignore[no-redef]
        GadgetSynthesizer,
        _validate_gadget_worker,
    )
    from util.memory_monitor import collect_memory_snapshot  # type: ignore[no-redef]
    from transition_table import TransitionTable  # type: ignore[no-redef]
    from util.enums import primitive_type, vector_machine, width_dict  # type: ignore[no-redef]
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
    llvm_mca_workers: int | None = None
    max_waves: int | None = None
    top_k: int = 10
    runtime_ui: str = "auto"
    runtime_session_factory: Callable[[int], object] | None = None


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

        # Candidate lookup and scoring caches
        self._candidate_by_index: dict[int, object] = dict(self._candidates_with_index)
        self._candidate_instruction_keys: dict[int, list[str]] = {
            candidate_idx: self._collect_instruction_stats_keys(graph)
            for candidate_idx, graph in self._candidates_with_index
        }
        self._candidate_instruction_counts: dict[int, int] = {
            candidate_idx: self._candidate_instruction_count(graph)
            for candidate_idx, graph in self._candidates_with_index
        }
        self._candidate_priority_order: list[int] = (
            self._build_candidate_priority_order()
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

        # Complete path keys already discovered (avoid duplicate reporting/scoring)
        self._scored_path_keys: set[tuple] = set()

        # Cumulative progress counters per stage (for progress bars)
        self._stage_attempts: list[int] = [0] * len(self.all_stages)
        self._stage_successes: list[int] = [0] * len(self.all_stages)

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

        # Dedicated async LLVM-MCA scoring lane (separate from synthesis pool)
        self._mca_executor: ThreadPoolExecutor | None = None
        self._mca_future_sizes: dict[object, int] = {}
        self._mca_result_queue: queue_mod.Queue = queue_mod.Queue()
        self._mca_pending_paths: int = 0

        # Backlog of terminal transitions awaiting path discovery/chunking.
        # We process this incrementally to keep UI/search responsive.
        self._pending_score_anchors: deque[tuple[int, tuple, tuple]] = deque()
        self._queued_score_anchors: set[tuple[int, tuple, tuple]] = set()

        # Runtime instruction-usage telemetry
        self.instruction_stats = InstructionStatsCollector()
        menu_instruction_keys = self._collect_menu_instruction_stats_keys()
        for stage_idx in range(len(self.all_stages)):
            self.instruction_stats.prepopulate_stage(stage_idx, menu_instruction_keys)
        self._menu_instruction_count = sum(self._candidate_instruction_counts.values())
        (
            self._instruction_latency_arches,
            self._instruction_latency_by_key,
        ) = self._build_instruction_latency_table(menu_instruction_keys)

        # Per-stage attempt snapshot at wave start.  Used by _submit_jobs
        # to enforce per-stage budgets within a wave.  None outside a wave.
        self._wave_start_attempts: list[int] | None = None

        if config.retroactive_input:
            self._prepopulate_retroactive_stage0()

    def _create_initial_state(self) -> VectorState:
        """Create initial vector state from first stage comparison pairs."""
        first_stage_pairs = self.all_stages[0]
        top = []
        bottom = []
        for pair in first_stage_pairs:
            top.append(pair[0])
            bottom.append(pair[1])
        return VectorState(top=top, bottom=bottom)

    def _prepopulate_retroactive_stage0(self) -> None:
        """Pre-populate stage 0 with all N! lane permutations of the comparison pairs.

        Each permutation is a valid initial state with a zero-cost identity gadget.
        The input and output states are identical (identity transition).
        """
        first_stage_pairs = self.all_stages[0]
        identity_gadget = PermutationGadget(
            top_instructions=[], bottom_instructions=[], validated=True
        )

        for perm in permutations(first_stage_pairs):
            top = [p[0] for p in perm]
            bottom = [p[1] for p in perm]
            state = VectorState(top=top, bottom=bottom)
            self.tt.add_transition(
                stage=0,
                input_state=state,
                output_state=state,
                gadget=identity_gadget,
            )

        # Mark all stage-0 outputs as forwarded so downstream stages can consume them
        unforwarded = self.tt.get_unforwarded_outputs(0)
        self.tt.mark_forwarded(0, [t for t, _vs in unforwarded])

        # Stage 0 is fully solved — mark exhausted
        self.exhausted_stages.add(0)

    # ------------------------------------------------------------------
    # Job creation
    # ------------------------------------------------------------------

    def _candidate_throughput_score_for_cpu(
        self,
        candidate_idx: int,
        cost_model,
    ) -> float:
        """Return summed reciprocal throughput for one candidate on one CPU."""
        score = 0.0
        for key in self._candidate_instruction_keys.get(candidate_idx, []):
            intrinsic_name = key.split("/", 1)[0]
            score += cost_model.get_instruction_cost(intrinsic_name).throughput
        return score

    def _rank_candidates_for_cpu(self, target_cpu: str) -> list[int]:
        """Rank candidate indices for one CPU by (throughput, length, index)."""
        try:
            from cost_model import CostModel
        except ImportError:
            from .cost_model import CostModel  # type: ignore[no-redef]

        cost_model = CostModel(target_cpu)

        scored: list[tuple[float, int, int]] = []
        for candidate_idx, _graph in self._candidates_with_index:
            scored.append(
                (
                    self._candidate_throughput_score_for_cpu(candidate_idx, cost_model),
                    self._candidate_instruction_counts.get(candidate_idx, 0),
                    candidate_idx,
                )
            )

        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        return [candidate_idx for _score, _length, candidate_idx in scored]

    @staticmethod
    def _interleave_candidate_rankings(
        rankings: list[list[int]], fallback_order: list[int]
    ) -> list[int]:
        """Interleave per-CPU candidate rankings with stable deduplication."""
        if not rankings:
            return list(fallback_order)

        merged: list[int] = []
        seen: set[int] = set()

        max_len = max((len(ranking) for ranking in rankings), default=0)
        for offset in range(max_len):
            for ranking in rankings:
                if offset >= len(ranking):
                    continue
                candidate_idx = ranking[offset]
                if candidate_idx in seen:
                    continue
                seen.add(candidate_idx)
                merged.append(candidate_idx)

        for candidate_idx in fallback_order:
            if candidate_idx not in seen:
                merged.append(candidate_idx)

        return merged

    def _build_candidate_priority_order(self) -> list[int]:
        """Build a single candidate priority order for job scheduling."""
        fallback_order = [
            candidate_idx for candidate_idx, _ in self._candidates_with_index
        ]
        arches = [cpu for cpu in self.config.target_cpus if cpu]

        self._candidate_rankings_by_cpu: dict[str, list[int]] = {}
        if not arches:
            return fallback_order

        rankings: list[list[int]] = []
        for arch in arches:
            try:
                ranked = self._rank_candidates_for_cpu(arch)
                self._candidate_rankings_by_cpu[arch] = ranked
                rankings.append(ranked)
            except Exception as exc:  # pragma: no cover - defensive fallback
                print(
                    f"Warning: failed to rank candidates for {arch}: {exc}",
                    file=sys.stderr,
                )

        return self._interleave_candidate_rankings(rankings, fallback_order)

    @staticmethod
    def _format_candidate_head(ordering: list[int], top_n: int) -> str:
        """Render a compact candidate index head for runtime logs."""
        head = ordering[:top_n]
        tail = " ..." if len(ordering) > top_n else ""
        return f"[{', '.join(str(i) for i in head)}]{tail}"

    def _log_candidate_priority_startup(self, runtime_session, top_n: int = 12) -> None:
        """Emit startup logs describing candidate ordering in the runtime UI."""
        if runtime_session is None:
            return

        runtime_session.log(
            "candidate-ordering: windowed tiers enabled "
            "(window ~= wave_attempts / input_count)"
        )

        if self._candidate_rankings_by_cpu:
            ordered_arches = [
                arch
                for arch in self.config.target_cpus
                if arch in self._candidate_rankings_by_cpu
            ]
            ordered_arches.extend(
                arch
                for arch in self._candidate_rankings_by_cpu
                if arch not in ordered_arches
            )
            for arch in ordered_arches:
                ranked = self._candidate_rankings_by_cpu.get(arch)
                if not ranked:
                    continue
                runtime_session.log(
                    f"candidate-ordering: {arch} top{top_n}: "
                    f"{self._format_candidate_head(ranked, top_n)}"
                )
        else:
            runtime_session.log(
                "candidate-ordering: no target CPUs selected, "
                "using default candidate order"
            )

        runtime_session.log(
            f"candidate-ordering: interleaved top{top_n}: "
            f"{self._format_candidate_head(self._candidate_priority_order, top_n)}"
        )

    def _ordered_inputs_for_stage(
        self, stage_idx: int, input_states: list[VectorState]
    ) -> list[VectorState]:
        """Prioritize uncovered input states before previously-attempted ones."""
        if stage_idx == 0:
            return list(input_states)

        attempted_inputs = {
            input_tuple
            for input_tuple, _candidate_idx in self.tt.stages[stage_idx].attempted_pairs
        }

        uncovered: list[VectorState] = []
        covered: list[VectorState] = []
        for state in input_states:
            (covered if state.as_tuple() in attempted_inputs else uncovered).append(
                state
            )

        return uncovered + covered

    def _candidate_window_size(self, input_count: int) -> int:
        """Window size derived from per-wave attempt budget."""
        if input_count <= 0:
            return 1
        return max(1, self.config.wave_attempts // input_count)

    def _make_jobs(
        self,
        stage_idx: int,
        input_states: list[VectorState],
        limit: int | None = None,
    ) -> list[tuple[int, tuple]]:
        """Create (candidate_index, job_tuple) pairs, skipping already-attempted.

        Uses windowed candidate tiers: for each candidate window, scan all
        (uncovered-first) inputs before moving to the next window.
        """
        stage_pairs = self.all_stages[stage_idx]
        jobs: list[tuple[int, tuple]] = []

        ordered_inputs = self._ordered_inputs_for_stage(stage_idx, input_states)
        candidate_order = self._candidate_priority_order
        if not ordered_inputs or not candidate_order:
            return jobs

        window_size = min(
            len(candidate_order),
            self._candidate_window_size(len(ordered_inputs)),
        )

        for window_start in range(0, len(candidate_order), window_size):
            window = candidate_order[window_start : window_start + window_size]
            for cand_idx in window:
                graph = self._candidate_by_index.get(cand_idx)
                if graph is None:
                    continue

                instruction_keys = self._candidate_instruction_keys.get(cand_idx, [])

                for input_state in ordered_inputs:
                    input_tuple = input_state.as_tuple()
                    if self.tt.was_attempted(stage_idx, input_tuple, cand_idx):
                        continue

                    metadata = {
                        "input_state": input_state,
                        "stage_idx": stage_idx,
                        "max_unique_outputs": self.config.max_unique_outputs,
                        "candidate_index": cand_idx,
                        "instruction_keys": instruction_keys,
                    }
                    # Final natural-order stage must enforce strict lane order
                    if (
                        self.config.natural_order
                        and stage_idx == len(self.all_stages) - 1
                    ):
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
                    if limit is not None and len(jobs) >= limit:
                        return jobs

        return jobs

    def _collect_instruction_stats_keys(self, graph) -> list[str]:
        """Collect normalized instruction keys for one candidate graph."""
        visited: set[int] = set()
        instruction_keys: list[str] = []

        def visit(node) -> None:
            if node is None:
                return

            if isinstance(node, IntrinsicNode):
                node_id = id(node)
                if node_id in visited:
                    return
                visited.add(node_id)
                for operand in node.operands.values():
                    visit(operand)
                instruction_keys.append(
                    normalize_instruction_stats_key(
                        InstructionSpec(
                            node.name,
                            {key: None for key in node.operands},
                        )
                    )
                )
                return

            if isinstance(node, Mux):
                for source in node.sources:
                    visit(source)

        visit(getattr(graph, "top", None))
        visit(getattr(graph, "bottom", None))
        return instruction_keys

    def _collect_menu_instruction_stats_keys(self) -> list[str]:
        """Flatten intrinsic keys from the full precomputed candidate menu."""
        keys: set[str] = set()
        for candidate_idx, _graph in self._candidates_with_index:
            keys.update(self._candidate_instruction_keys.get(candidate_idx, []))
        return sorted(keys)

    def _candidate_instruction_count(self, graph) -> int:
        """Count intrinsic nodes in one candidate graph (deduping shared nodes)."""
        visited: set[int] = set()
        count = 0

        def visit(node) -> None:
            nonlocal count
            if node is None:
                return
            if isinstance(node, IntrinsicNode):
                node_id = id(node)
                if node_id in visited:
                    return
                visited.add(node_id)
                count += 1
                for operand in node.operands.values():
                    visit(operand)
                return
            if isinstance(node, Mux):
                for source in node.sources:
                    visit(source)

        visit(getattr(graph, "top", None))
        visit(getattr(graph, "bottom", None))
        return count

    @staticmethod
    def _format_latency_cell(latency: float) -> str:
        """Format latency values for compact table display."""
        return f"{latency:g}"

    def _build_instruction_latency_table(
        self, instruction_keys: list[str]
    ) -> tuple[list[str], dict[str, dict[str, str]]]:
        """Build instruction-latency lookup per selected target architecture."""
        arches = list(self.config.target_cpus)
        if not arches:
            return [], {}

        try:
            from intrinsic_registry import get_intrinsic_registry, xml_string_key
            from cost_model import resolve_arch_name
            from util.uops_parser import find_uops_database_path, parse_uops_database
        except ImportError:
            from .intrinsic_registry import get_intrinsic_registry, xml_string_key  # type: ignore[no-redef]
            from .cost_model import resolve_arch_name  # type: ignore[no-redef]
            from .util.uops_parser import (  # type: ignore[no-redef]
                find_uops_database_path,
                parse_uops_database,
            )

        database_path = find_uops_database_path(
            base_dir=os.path.join(os.path.dirname(__file__), "..")
        )
        if database_path is None:
            print(
                "Warning: no uops database found, latency columns will show '-'",
                file=sys.stderr,
            )
            return arches, {}

        registry = get_intrinsic_registry()
        xml_costs_by_arch: dict[str, dict[str, object]] = {}

        for arch in arches:
            canonical = resolve_arch_name(arch)
            if canonical is None:
                continue
            try:
                xml_costs_by_arch[arch] = parse_uops_database(database_path, canonical)
            except Exception as exc:
                print(
                    f"Warning: failed loading uops.info costs for {arch}: {exc}",
                    file=sys.stderr,
                )

        latency_by_key: dict[str, dict[str, str]] = {}
        for instruction_key in instruction_keys:
            intrinsic_name = instruction_key.split("/", 1)[0]
            info = registry.get(intrinsic_name)
            if info is None:
                continue

            xml_key = xml_string_key(info)
            evex_key = xml_key.replace(
                info.asm_mnemonic, f"{info.asm_mnemonic}_EVEX", 1
            )

            for arch in arches:
                arch_costs = xml_costs_by_arch.get(arch)
                if not arch_costs:
                    continue
                cost = arch_costs.get(xml_key) or arch_costs.get(evex_key)
                if cost is None:
                    continue
                latency = getattr(cost, "latency", None)
                if latency is None:
                    continue
                latency_by_key.setdefault(instruction_key, {})[arch] = (
                    self._format_latency_cell(float(latency))
                )

        return arches, latency_by_key

    def _resolve_runtime_ui(self) -> str:
        """Resolve runtime UI mode, defaulting to Textual only on a TTY."""
        if self.config.runtime_ui != "auto":
            return self.config.runtime_ui
        if sys.stdin.isatty() and sys.stdout.isatty():
            return "textual"
        return "none"

    def _request_interrupt_from_ui(self) -> None:
        """Handle a user-initiated stop request from the runtime UI."""
        if self._interrupted:
            return
        self._interrupted = True
        print(
            "\n'q' pressed in runtime UI - stopping and saving checkpoint...",
            file=sys.stderr,
        )

    def _create_runtime_session(self):
        """Create the configured runtime UI session."""
        if self.config.runtime_session_factory is not None:
            return self.config.runtime_session_factory(len(self.all_stages))

        try:
            from .textual_progress_ui import (
                NullWaveRuntimeSession,
                TextualWaveRuntimeSession,
            )
        except ImportError:
            from textual_progress_ui import (  # type: ignore[no-redef]
                NullWaveRuntimeSession,
                TextualWaveRuntimeSession,
            )

        if self._resolve_runtime_ui() == "textual":
            return TextualWaveRuntimeSession(
                stage_count=len(self.all_stages),
                instruction_top_k=self.config.top_k,
                on_quit_requested=self._request_interrupt_from_ui,
                instruction_latency_arches=self._instruction_latency_arches,
                instruction_latency_by_key=self._instruction_latency_by_key,
            )
        return NullWaveRuntimeSession(stage_count=len(self.all_stages))

    def _stage_progress_snapshots(self):
        """Build a render-ready snapshot of current stage metrics."""
        try:
            from .textual_progress_ui import StageProgressSnapshot
        except ImportError:
            from textual_progress_ui import StageProgressSnapshot  # type: ignore[no-redef]

        snapshots = []
        for stage_idx in range(len(self.all_stages)):
            total = (
                self._stage_attempts[stage_idx]
                + len(self._pending_jobs[stage_idx])
                + self._in_flight[stage_idx]
            )
            covered_inputs, total_inputs, uncovered_input_pct = (
                self._stage_input_domain_coverage(stage_idx)
            )
            snapshots.append(
                StageProgressSnapshot(
                    stage_idx=stage_idx,
                    attempts=self._stage_attempts[stage_idx],
                    total=total,
                    valid=self._stage_successes[stage_idx],
                    unique=self.tt.unique_output_count(stage_idx),
                    covered_inputs=covered_inputs,
                    total_inputs=total_inputs,
                    uncovered_input_pct=uncovered_input_pct,
                )
            )
        return snapshots

    def _stage_input_domain_coverage(
        self, stage_idx: int
    ) -> tuple[int | None, int | None, float | None]:
        """Return (covered_inputs, total_inputs, uncovered_pct) for one stage."""
        if stage_idx == 0:
            return None, None, None

        total_inputs = self.tt.unique_output_count(stage_idx - 1)
        if total_inputs == 0:
            return 0, 0, None

        covered_inputs = len(
            {
                input_tuple
                for input_tuple, _cand_idx in self.tt.stages[stage_idx].attempted_pairs
            }
        )
        uncovered_pct = max(0.0, 100.0 * (1.0 - (covered_inputs / total_inputs)))
        return covered_inputs, total_inputs, uncovered_pct

    def _runtime_status_text(self) -> str | None:
        """Build footer status text for path discovery/scoring + best LLVM-MCA."""
        parts = [f"Menu instructions: {self._menu_instruction_count}"]

        discovered = len(self._scored_path_keys)
        scored = len(self._all_scored_paths)
        pending = self._mca_pending_paths
        pending_anchors = len(self._pending_score_anchors)

        parts.append(
            f"Paths: {discovered} discovered, {scored} scored"
            + (f", {pending} pending LLVM-MCA" if pending > 0 else "")
            + (f", {pending_anchors} pending discovery" if pending_anchors > 0 else "")
        )

        if self.best_scores:
            best = [
                f"{cpu}: {s['cycles']:.1f}cy (throughput {s['throughput']:.2f})"
                for cpu, s in sorted(self.best_scores.items())
            ]
            parts.append("Best: " + "  |  ".join(best))
        return " | ".join(parts)

    def _sync_runtime_session(self, runtime_session) -> None:
        """Push the latest runtime snapshot into the active UI session."""
        if runtime_session is None:
            return
        runtime_session.update_stage_metrics(self._stage_progress_snapshots())
        runtime_session.update_instruction_stats(
            self.instruction_stats.snapshot_by_stage()
        )
        snapshot = collect_memory_snapshot()
        runtime_session.update_memory_line(
            snapshot.format() if snapshot is not None else "Memory: unavailable"
        )
        runtime_session.set_status(self._runtime_status_text())

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

        # Compute budget before _make_jobs so it can stop early
        # (avoids building millions of tuples with N! retroactive inputs)
        already_committed = (
            len(self._pending_jobs[target_stage]) + self._in_flight[target_stage]
        )
        if self._wave_start_attempts is not None:
            already_committed += (
                self._stage_attempts[target_stage]
                - self._wave_start_attempts[target_stage]
            )
        budget_remaining = max(0, self.config.wave_attempts - already_committed)

        new_jobs = self._make_jobs(target_stage, input_states, limit=budget_remaining)

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

                cand_idx, job = pending.popleft()
                _graph, input_state, _stage_pairs, _vm, _prim_type, _metadata = job
                self.tt.record_attempted_pair(
                    stage_idx, input_state.as_tuple(), cand_idx
                )

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
        runtime_session=None,
    ) -> int:
        """Process all available results from the synthesis completion queue.

        Records transitions, enqueues downstream jobs, and queues complete-path
        scoring for last-stage discoveries. Returns count of synthesis results
        processed (not LLVM-MCA completions).
        """
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
                continue

            _, stage_idx, payload = item
            self._in_flight[stage_idx] -= 1

            gadget_results, input_state, _metadata, _ct, _st = payload
            valid_output_count = len(gadget_results)

            instruction_keys = _metadata.get("instruction_keys", [])
            self.tt.record_attempt(stage_idx)
            self._stage_attempts[stage_idx] += 1
            self._stage_successes[stage_idx] += valid_output_count

            new_unique = 0
            for gadget, output_state in gadget_results:
                out_t = output_state.as_tuple()
                is_new = out_t not in self.tt.get_unique_outputs(stage_idx)
                self.tt.add_transition(stage_idx, input_state, output_state, gadget)

                if is_new:
                    new_unique += 1
                    # Feed downstream stage to keep pool busy during stragglers
                    self._enqueue_downstream(stage_idx, output_state)

                    # Queue on-the-fly scoring for newly discovered complete paths
                    if stage_idx == last_stage:
                        try:
                            self._score_new_paths_for_transition(
                                stage_idx,
                                input_state.as_tuple(),
                                out_t,
                                runtime_session,
                            )
                        except Exception as score_exc:
                            print(
                                f"Warning: on-the-fly scoring failed: {score_exc}",
                                file=sys.stderr,
                            )

            if gadget_results:
                outcome = "unique" if new_unique > 0 else "valid"
            else:
                outcome = "no_valid"
            if instruction_keys:
                self.instruction_stats.record_attempt(
                    stage_idx, instruction_keys, outcome
                )

        discovered_count = self._drain_score_anchor_queue(
            runtime_session,
            max_anchors=2,
        )
        mca_count = self._drain_scoring_results(runtime_session)

        if count > 0 or discovered_count > 0 or mca_count > 0:
            self._sync_runtime_session(runtime_session)

        return count

    # ------------------------------------------------------------------
    # Scoring via LLVM-MCA
    # ------------------------------------------------------------------

    def _path_discovery_chunk_size(self) -> int:
        """Bound per-anchor path discovery work to keep the UI responsive."""
        return max(1, self.config.max_paths_per_wave)

    def _queue_score_anchor(
        self,
        stage_idx: int,
        input_tuple: tuple,
        output_tuple: tuple,
    ) -> None:
        """Queue one terminal transition for incremental complete-path discovery."""
        anchor = (stage_idx, input_tuple, output_tuple)
        if anchor in self._queued_score_anchors:
            return
        self._queued_score_anchors.add(anchor)
        self._pending_score_anchors.append(anchor)

    def _enqueue_scoring_batch(self, paths: list[list[tuple]]) -> None:
        """Queue complete paths on the dedicated LLVM-MCA lane in small batches."""
        if not paths:
            return

        chunk_size = self._path_discovery_chunk_size()
        for start in range(0, len(paths), chunk_size):
            batch = paths[start : start + chunk_size]
            if self._mca_executor is None:
                self._integrate_scored_paths(self._score_complete_paths(batch))
                continue

            future = self._mca_executor.submit(self._score_complete_paths, batch)
            self._mca_future_sizes[future] = len(batch)
            self._mca_pending_paths += len(batch)
            future.add_done_callback(lambda done: self._mca_result_queue.put(done))

    def _integrate_scored_paths(self, scored: list[dict]) -> None:
        """Merge scored paths and update per-CPU best-score summary."""
        if not scored:
            return

        self._all_scored_paths.extend(scored)

        for entry in scored:
            for cpu, scores in entry.get("scores", {}).items():
                cycles = scores.get("cycles", float("inf"))
                if cycles <= 0:
                    continue
                prev = self.best_scores.get(cpu)
                if prev is None or cycles < prev["cycles"]:
                    self.best_scores[cpu] = {
                        "throughput": scores["throughput"],
                        "cycles": cycles,
                        "path_key": entry.get("path_key"),
                    }

    def _drain_score_anchor_queue(
        self, runtime_session=None, max_anchors: int = 1
    ) -> int:
        """Incrementally discover complete paths from queued terminal anchors."""
        discovered_paths = 0
        anchors_processed = 0
        chunk_size = self._path_discovery_chunk_size()

        while (
            anchors_processed < max_anchors
            and len(self._pending_score_anchors) > 0
            and not self._interrupted
        ):
            anchor = self._pending_score_anchors.popleft()
            self._queued_score_anchors.discard(anchor)
            stage_idx, input_tuple, output_tuple = anchor

            new_paths = self.tt.trace_paths_ending_with(
                stage_idx,
                input_tuple,
                output_tuple,
                max_paths=chunk_size,
                exclude_paths=self._scored_path_keys,
            )
            anchors_processed += 1

            if not new_paths:
                continue

            for path in new_paths:
                self._scored_path_keys.add(tuple(tuple(step) for step in path))

            discovered_paths += len(new_paths)
            if runtime_session is not None:
                runtime_session.log(
                    f"Stage {stage_idx}: +{len(new_paths)} complete path(s) discovered "
                    f"({len(self._scored_path_keys)} total discovered)"
                )

            self._enqueue_scoring_batch(new_paths)

            # Discovery chunk was full; likely more paths remain for this anchor.
            if len(new_paths) >= chunk_size:
                self._queue_score_anchor(stage_idx, input_tuple, output_tuple)

        return discovered_paths

    def _drain_scoring_results(self, runtime_session=None) -> int:
        """Drain finished LLVM-MCA batches and merge them into engine state."""
        completed_paths = 0

        while True:
            try:
                future = self._mca_result_queue.get_nowait()
            except queue_mod.Empty:
                break

            batch_size = self._mca_future_sizes.pop(future, 0)
            self._mca_pending_paths = max(0, self._mca_pending_paths - batch_size)

            try:
                scored = future.result()
            except Exception as exc:
                print(f"Warning: async LLVM-MCA batch failed: {exc}", file=sys.stderr)
                continue

            completed_paths += len(scored)
            self._integrate_scored_paths(scored)

        if completed_paths > 0 and runtime_session is not None:
            runtime_session.log(
                f"LLVM-MCA: scored {completed_paths} complete path(s) "
                f"({len(self._all_scored_paths)} total scored)"
            )

        return completed_paths

    def _wait_for_pending_scoring(self, runtime_session=None) -> None:
        """Wait for queued path discovery + LLVM-MCA work before shutdown."""
        while self._mca_pending_paths > 0 or len(self._pending_score_anchors) > 0:
            discovered = self._drain_score_anchor_queue(runtime_session, max_anchors=4)
            drained = self._drain_scoring_results(runtime_session)
            if discovered == 0 and drained == 0:
                time.sleep(0.05)

    def _score_new_paths_for_transition(
        self,
        stage_idx: int,
        input_tuple: tuple,
        output_tuple: tuple,
        runtime_session=None,
    ) -> None:
        """Queue complete-path discovery for one newly found last-stage transition."""
        self._queue_score_anchor(stage_idx, input_tuple, output_tuple)

        # Do a tiny amount of discovery immediately so the UI reflects progress
        # as soon as transitions are found, then continue incrementally.
        self._drain_score_anchor_queue(runtime_session, max_anchors=1)

    def _score_complete_paths(self, paths: list[list[tuple]]) -> list[dict]:
        """Score the given complete paths with LLVM-MCA.

        Returns a list of scored path dicts (newly scored only).
        """
        if not paths:
            return []

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
            from util.llvm_mca_runner import find_llvm_mca, run_llvm_mca
            from cost_model import resolve_llvm_mca_cpu
        except ImportError:
            from .perf_estimator import (  # type: ignore[no-redef]
                generate_solution_asm,
                sanitize_asm_for_llvm_mca,
            )
            from .util.llvm_mca_runner import find_llvm_mca, run_llvm_mca  # type: ignore[no-redef]
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
            result = run_llvm_mca(
                sanitized,
                mcpu,
                mca_bin,
                idx + 1,
                include_timeline=False,
            )
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
                except Exception as exc:
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
        - Subsequent waves: weakest stage (by success rate), but penalized
          by ``unproductive_waves`` to avoid hammering stages that produce
          local outputs without downstream progress.
        - Bubble-up: if a stage has consecutive_zero_budgets >= 2 or
          unproductive_waves >= 3, go upstream to create input diversity.
        """
        if self.wave_count == 0 and 0 not in self.exhausted_stages:
            return 0

        excluded = self.exhausted_stages | self._stalled_stages
        target = self._pick_target(excluded)
        if target is None:
            return None

        sd = self.tt.stages[target]
        needs_bubble_up = sd.consecutive_zero_budgets >= 2 or sd.unproductive_waves >= 3

        if needs_bubble_up:
            upstream = target - 1
            while upstream >= 0:
                if upstream not in excluded:
                    return upstream
                upstream -= 1
            # All upstream exhausted/stalled.
            # If this stage still has untried pairs, keep targeting it
            # instead of exhausting it prematurely.
            if self._stage_has_untried_work(target):
                return target

            predecessors_exhausted = all(
                s in self.exhausted_stages for s in range(target)
            )
            if predecessors_exhausted:
                self.exhausted_stages.add(target)
            else:
                self._stalled_stages.add(target)
            excluded = self.exhausted_stages | self._stalled_stages
            return self._pick_target(excluded)

        return target

    def _stage_has_untried_work(self, stage_idx: int) -> bool:
        """Return True if a stage still has at least one untried job pair."""
        if stage_idx == 0:
            input_states = [self.initial_state]
        else:
            prev_outputs = self.tt.get_unique_outputs(stage_idx - 1)
            input_states = list(prev_outputs.values()) if prev_outputs else []

        if not input_states:
            return False

        return bool(self._make_jobs(stage_idx, input_states, limit=1))

    def _pick_target(self, excluded: set[int]) -> int | None:
        """Pick the weakest stage, penalizing unproductive ones.

        Stages with high ``unproductive_waves`` are deprioritized:
        their output count is inflated so other stages look weaker
        and get targeted instead.
        """
        best_idx: int | None = None
        best_key = (float("inf"), float("inf"))

        for i, sd in enumerate(self.tt.stages):
            if i in excluded:
                continue

            n_outputs = len(sd.unique_outputs)
            rate = n_outputs / sd.attempts if sd.attempts > 0 else 0.0

            # Penalize unproductive stages: pretend they have more outputs
            # so they look less "weak" and other stages get priority.
            # Each unproductive wave adds a virtual output count penalty.
            penalty = sd.unproductive_waves * 10
            effective_outputs = n_outputs + penalty

            key = (effective_outputs, rate)
            if key < best_key:
                best_key = key
                best_idx = i

        return best_idx

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
        runtime_session,
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

            submitted = self._submit_jobs(pool, completion_queue, pool_target)
            if submitted > 0:
                self._sync_runtime_session(runtime_session)
            drained = self._drain_results(completion_queue, runtime_session)

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

        runtime_session = self._create_runtime_session()

        if self.config.target_cpus:
            mca_workers = self.config.llvm_mca_workers
            if mca_workers is None:
                mca_workers = max(1, min(4, num_workers))
            self._mca_executor = ThreadPoolExecutor(
                max_workers=mca_workers,
                thread_name_prefix="llvm-mca",
            )

        try:
            with runtime_session:
                if self._mca_executor is not None:
                    runtime_session.log(
                        "LLVM-MCA async lane enabled "
                        f"({self.config.llvm_mca_workers or max(1, min(4, num_workers))} worker(s))"
                    )

                # Seed pending queues from unforwarded outputs (resume case)
                for s in range(1, len(self.all_stages)):
                    self._enqueue_from_unforwarded(s)
                self._sync_runtime_session(runtime_session)
                self._log_candidate_priority_startup(runtime_session)

                while True:
                    if self._interrupted:
                        self._save_checkpoint()
                        runtime_session.log("Interrupted - checkpoint saved.")
                        break

                    if (
                        self.config.max_waves is not None
                        and self.wave_count >= self.config.max_waves
                    ):
                        break

                    if len(self.exhausted_stages) >= len(self.all_stages):
                        runtime_session.log("All stages exhausted - search complete.")
                        break

                    # --- Start a new wave ---
                    target = self._select_target_stage()
                    if target is None:
                        break

                    # Snapshot per-stage attempts so budget checks know
                    # how many attempts each stage has consumed this wave
                    self._wave_start_attempts = list(self._stage_attempts)
                    last_stage_idx = len(self.all_stages) - 1
                    last_stage_outputs_before = self.tt.unique_output_count(
                        last_stage_idx
                    )
                    scored_before = len(self._all_scored_paths)

                    # Generate jobs for the target stage
                    self._generate_jobs_for_stage(target)
                    self._sync_runtime_session(runtime_session)

                    # Run target stage with budget
                    new_outputs = self._run_stage_in_wave(
                        target,
                        pool,
                        completion_queue,
                        pool_target,
                        runtime_session,
                    )

                    # Track consecutive zero-output budgets
                    sd = self.tt.stages[target]
                    if new_outputs == 0:
                        sd.consecutive_zero_budgets += 1
                    else:
                        sd.consecutive_zero_budgets = 0
                        for s in range(target + 1, len(self.all_stages)):
                            self._stalled_stages.discard(s)

                    # Depth-first propagation: run EVERY downstream stage
                    # from target+1 to the last stage.  A wave is not
                    # complete until every stage has had its turn.
                    #
                    # Each downstream stage gets a fresh budget.  We reset
                    # _wave_start_attempts for the stage so that eagerly-
                    # processed work from _enqueue_downstream doesn't
                    # consume the propagation budget.
                    propagation: dict[int, int] = {}
                    for stage_idx in range(target + 1, len(self.all_stages)):
                        if self._interrupted:
                            break

                        # Fresh budget: reset wave-start snapshot for this
                        # stage so _run_stage_in_wave sees a full budget
                        self._wave_start_attempts[stage_idx] = self._stage_attempts[
                            stage_idx
                        ]

                        # Generate jobs from unforwarded outputs
                        unforwarded = self.tt.get_unforwarded_outputs(stage_idx - 1)
                        if unforwarded:
                            input_states = [vs for _tup, vs in unforwarded]
                            output_tuples = [tup for tup, _vs in unforwarded]
                            self.tt.mark_forwarded(stage_idx - 1, output_tuples)

                            downstream_jobs = self._make_jobs(
                                stage_idx, input_states, limit=self.config.wave_attempts
                            )
                            self._pending_jobs[stage_idx].extend(downstream_jobs)

                        self._sync_runtime_session(runtime_session)

                        stage_new = self._run_stage_in_wave(
                            stage_idx,
                            pool,
                            completion_queue,
                            pool_target,
                            runtime_session,
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
                        drained = self._drain_results(completion_queue, runtime_session)
                        if drained == 0:
                            time.sleep(0.05)

                    # --- Wave complete ---
                    self._wave_start_attempts = None

                    # Track downstream productivity: did this wave produce
                    # new last-stage outputs or new scored paths?
                    last_stage_outputs_after = self.tt.unique_output_count(
                        last_stage_idx
                    )
                    scored_after = len(self._all_scored_paths)
                    downstream_progress = (
                        last_stage_outputs_after > last_stage_outputs_before
                        or scored_after > scored_before
                    )
                    sd = self.tt.stages[target]
                    if downstream_progress:
                        sd.unproductive_waves = 0
                    else:
                        sd.unproductive_waves += 1

                    self._save_checkpoint()
                    self._sync_runtime_session(runtime_session)

                    # Wave summary
                    stats = self.tt.stage_stats(target)
                    prop_str = ""
                    if propagation:
                        prop_parts = [
                            f"s{s}+{n}" for s, n in sorted(propagation.items())
                        ]
                        prop_str = f" | propagated: {', '.join(prop_parts)}"
                    discovered_total = len(self._scored_path_keys)
                    scored_total = len(self._all_scored_paths)
                    pending_total = self._mca_pending_paths
                    pending_discovery = len(self._pending_score_anchors)
                    unprod = self.tt.stages[target].unproductive_waves
                    unprod_str = f" | unproductive x{unprod}" if unprod > 0 else ""
                    runtime_session.log(
                        f"Wave {self.wave_count}: "
                        f"stage {target} targeted, "
                        f"+{new_outputs} outputs "
                        f"({stats['distinct_outputs']} total)"
                        f"{prop_str}"
                        f" | paths: {discovered_total} discovered, "
                        f"{scored_total} scored"
                        + (
                            f", {pending_total} pending LLVM-MCA"
                            if pending_total > 0
                            else ""
                        )
                        + (
                            f", {pending_discovery} pending discovery"
                            if pending_discovery > 0
                            else ""
                        )
                        + unprod_str
                    )
                    self._sync_runtime_session(runtime_session)

                    self.wave_count += 1

                # Ensure queued LLVM-MCA work is integrated before exit.
                self._wait_for_pending_scoring(runtime_session)
                self._save_checkpoint()
                self._sync_runtime_session(runtime_session)

        finally:
            if self._mca_executor is not None:
                self._mca_executor.shutdown(wait=False)
                self._mca_executor = None

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
            # Valid-attempt counts were previously UI-only and are not
            # checkpointed, so resume from the known unique-output floor.
            self._stage_successes[i] = self.tt.unique_output_count(i)

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
