"""End-to-end Z3 verification for bitonic sort paths.

Given a CompletePath (sequence of stages with specific gadget choices),
wire all stages together as a single Z3 expression and prove the output
is sorted for ALL possible inputs.

Supports chunked verification that exploits the recursive structure of
bitonic sorting networks: instead of one monolithic proof over all stages,
splits at recursion boundaries and verifies each chunk with preconditions
asserting that sub-lists from the previous level are already sorted.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from z3 import (
    Solver,
    BitVec,
    BitVecVal,
    Extract,
    Concat,
    And,
    Not,
    unsat,
    simplify,
)

try:
    from . import z3_avx
    from .bitonic_super_optimizer import (
        VectorState,
        InstructionSpec,
        PermutationGadget,
        SolutionNode,
        _dispatch_intrinsic_by_signature,
        get_available_intrinsics,
    )
    from .path_selector import CompletePath
    from .utils import vector_machine, primitive_type, width_dict
except ImportError:
    import z3_avx  # type: ignore
    from bitonic_super_optimizer import (  # type: ignore
        VectorState,
        InstructionSpec,
        PermutationGadget,
        SolutionNode,
        _dispatch_intrinsic_by_signature,
        get_available_intrinsics,
    )
    from path_selector import CompletePath  # type: ignore
    from utils import vector_machine, primitive_type, width_dict  # type: ignore


@dataclass
class SolutionBundle:
    """Solutions loaded from a JSON file together with metadata."""

    roots: list[SolutionNode]
    natural_order: bool
    vm_name: str | None
    prim_type_name: str | None
    num_vecs: int | None


def load_solutions_from_json(path: str) -> SolutionBundle:
    """Load a solution tree from a JSON file produced by json_exporter.

    Reconstructs the SolutionNode DAG with shared children references
    and returns metadata stored alongside the solutions.
    """
    with open(path) as f:
        data = json.load(f)

    nodes_data = data["nodes"]
    built: dict[str, SolutionNode] = {}

    def build_node(node_id: str) -> SolutionNode:
        if node_id in built:
            return built[node_id]
        nd = nodes_data[node_id]
        children = [build_node(cid) for cid in nd["children"]]
        gadgets = []
        for gd in nd["gadgets"]:
            top_insts = [
                InstructionSpec(i["name"], dict(i["args"]))
                for i in gd["top_instructions"]
            ]
            bottom_insts = [
                InstructionSpec(i["name"], dict(i["args"]))
                for i in gd["bottom_instructions"]
            ]
            gadgets.append(
                PermutationGadget(
                    top_instructions=top_insts,
                    bottom_instructions=bottom_insts,
                    validated=True,
                )
            )
        node = SolutionNode(
            stage=nd["stage"],
            input_state=VectorState(
                top=nd["input_state"]["top"],
                bottom=nd["input_state"]["bottom"],
            ),
            output_state=VectorState(
                top=nd["output_state"]["top"],
                bottom=nd["output_state"]["bottom"],
            ),
            gadgets=gadgets,
            children=children,
        )
        built[node_id] = node
        return node

    roots = [build_node(rid) for rid in data["roots"]]
    return SolutionBundle(
        roots=roots,
        natural_order=data.get("natural_order", False),
        vm_name=data.get("vector_machine"),
        prim_type_name=data.get("primitive_type"),
        num_vecs=data.get("num_vecs"),
    )


@dataclass
class VerificationResult:
    """Result of end-to-end path verification.

    Attributes:
        verified: True if the path provably sorts all inputs.
        counterexample: If verified is False, a concrete input that breaks sorting.
        solver_time: Wall-clock seconds spent in Z3.
    """

    verified: bool
    counterexample: list[int] | None = None
    solver_time: float = 0.0


_MIN_GROUP_SIZE = 4  # Smallest sub-problem for chunked verification


def compute_recursion_boundaries(
    n: int, min_group_size: int = _MIN_GROUP_SIZE
) -> list[int]:
    """Compute the stage boundaries of the bitonic sort recursion.

    For n elements, the bitonic sort recursion sorts sub-lists of increasing
    size.  sort(2^k) completes after stage T(2^k)-1 where T(n) = T(n/2) + log2(n).

    Returns the stage indices *after which* sorted sub-lists of the given
    minimum size exist.  For n=16, min_group_size=4 this returns [2, 5]
    meaning:
      - after stage 2: four 4-element sorted groups
      - after stage 5: two 8-element sorted groups
    The final boundary (full sort) is NOT included — the caller handles it
    as the postcondition of the last chunk.
    """
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")

    # T(n) = total stages for sort(n)
    # T(1) = 0, T(n) = T(n/2) + log2(n)
    def total_stages(k: int) -> int:
        if k <= 1:
            return 0
        return total_stages(k // 2) + int(math.log2(k))

    boundaries = []
    group_size = min_group_size
    while group_size < n:
        boundary_stage = total_stages(group_size) - 1  # last stage of sort(group_size)
        boundaries.append(boundary_stage)
        group_size *= 2

    return boundaries


def num_verify_chunks(total_elements: int) -> int:
    """Return the number of verification steps for a given element count.

    For N elements with min_group_size=4:
      N=4:  1 (just verify the 4-element sort)
      N=8:  3 (2×4-element + 1×8-element merge)
      N=16: 7 (4×4-element + 2×8-element merge + 1×16-element merge)
      N=32: 15 (8×4 + 4×8 + 2×16 + 1×32)
    General: N/4 + N/8 + ... + 1 = N/2 - 1  (for N >= 8)
    """
    boundaries = compute_recursion_boundaries(total_elements)
    if not boundaries:
        return 1
    # Each level has N/post_group_size groups to verify, plus 1 for the final merge
    total = 0
    for level_idx in range(len(boundaries)):
        post_gs = _MIN_GROUP_SIZE * (2**level_idx)
        total += total_elements // post_gs
    total += 1  # final merge
    return total


@dataclass
class VerifyStep:
    """Lightweight step data for verification (picklable, no tree references).

    Extracted from PathStep to avoid pickling the entire SolutionNode DAG
    when sending jobs to worker processes.
    """

    gadget: PermutationGadget
    input_state: VectorState
    output_state: VectorState


def extract_verify_steps(path: CompletePath) -> list[VerifyStep]:
    """Extract picklable step data from a CompletePath."""
    return [
        VerifyStep(
            gadget=step.gadget,
            input_state=step.node.input_state,
            output_state=step.node.output_state,
        )
        for step in path.steps
    ]


_chunk_progress_queue = None


def _init_verify_worker(queue):
    """Pool initializer: stash the shared progress queue in each worker."""
    global _chunk_progress_queue
    _chunk_progress_queue = queue


def _verify_path_worker(
    job: tuple,
) -> tuple[int, VerificationResult]:
    """Top-level worker for multiprocessing path verification.

    Args:
        job: (path_index, steps, vm, prim_type, natural_order)
            where steps is a list of VerifyStep.

    Returns:
        (path_index, VerificationResult)
    """
    path_index, steps, vm, prim_type, natural_order = job

    def _on_chunk_done():
        if _chunk_progress_queue is not None:
            _chunk_progress_queue.put(path_index)

    verifier = BitonicPathVerifier(vm, prim_type)
    result = verifier.verify_steps(steps, natural_order, chunk_callback=_on_chunk_done)
    return path_index, result


class BitonicPathVerifier:
    """Proves that a complete bitonic sort path sorts correctly for all inputs."""

    def __init__(self, vm: vector_machine, prim_type: primitive_type):
        self.vm = vm
        self.prim_type = prim_type
        self.elements_per_vector = width_dict[vm] // int(prim_type.value[0])
        self.lane_width = int(prim_type.value[0]) * 8
        self.total_bits = width_dict[vm] * 8
        self.intrinsics = get_available_intrinsics(vm, prim_type)

    def verify_path(
        self, path: CompletePath, natural_order: bool = False
    ) -> VerificationResult:
        """Verify that a complete path sorts all inputs correctly.

        Args:
            path: A CompletePath from root to leaf through the solution tree.
            natural_order: If True, the last stage restores natural element
                order and has no min/max compare-swap.

        Returns:
            VerificationResult with verified=True if the path is a correct
            sorting network, or verified=False with a counterexample.
        """
        return self.verify_steps(extract_verify_steps(path), natural_order)

    def verify_steps(
        self,
        steps: list[VerifyStep],
        natural_order: bool = False,
        chunk_callback: Callable[[], None] | None = None,
    ) -> VerificationResult:
        """Verify that a sequence of gadget steps sorts all inputs correctly.

        Automatically uses chunked verification when the element count is
        large enough to benefit (>= 16 elements).  For smaller networks the
        monolithic proof is fast enough.

        Args:
            steps: List of VerifyStep (gadget + input/output state per stage).
            natural_order: If True, the last stage restores natural element
                order and has no min/max compare-swap.
            chunk_callback: Optional callable invoked after each chunk completes
                (used for progress reporting in chunked mode).

        Returns:
            VerificationResult with verified=True if the path is a correct
            sorting network, or verified=False with a counterexample.
        """
        N = 2 * self.elements_per_vector
        # Use chunked verification when recursion boundaries exist
        boundaries = compute_recursion_boundaries(N)
        if boundaries:
            return self._verify_steps_chunked(steps, natural_order, chunk_callback)
        result = self._verify_steps_monolithic(steps, natural_order)
        if chunk_callback:
            chunk_callback()
        return result

    def _simulate_steps(
        self, steps: list[VerifyStep], natural_order: bool
    ) -> tuple[list, list]:
        """Run gadget steps symbolically, returning (elems, sorted_elems).

        Creates N symbolic BitVec elements, wires them through the given
        stages (applying gadget permutations + min/max), and extracts the
        output elements in label order.

        Returns:
            elems: The N unconstrained symbolic input elements.
            sorted_elems: The N output elements after all stages.
        """
        N = 2 * self.elements_per_vector
        num_stages = len(steps)

        elems = [BitVec(f"e_{i}", self.lane_width) for i in range(N)]

        initial_state = steps[0].input_state
        top_vec = self._build_register(elems, initial_state.top)
        bottom_vec = self._build_register(elems, initial_state.bottom)

        for step_idx, step in enumerate(steps):
            gadget = step.gadget
            is_last_stage = step_idx == num_stages - 1

            new_top = self._apply_concrete_instructions(
                top_vec, bottom_vec, gadget.top_instructions, is_top=True
            )
            new_bottom = self._apply_concrete_instructions(
                top_vec, bottom_vec, gadget.bottom_instructions, is_top=False
            )

            if natural_order and is_last_stage:
                top_vec = new_top
                bottom_vec = new_bottom
            else:
                top_vec = z3_avx.generic_min(
                    new_top, new_bottom, self.total_bits, self.lane_width
                )
                bottom_vec = z3_avx.generic_max(
                    new_top, new_bottom, self.total_bits, self.lane_width
                )

        final_state = steps[-1].output_state
        sorted_elems = self._extract_output_elements(
            top_vec, bottom_vec, final_state, N
        )
        return elems, sorted_elems

    def _verify_steps_monolithic(
        self, steps: list[VerifyStep], natural_order: bool = False
    ) -> VerificationResult:
        """Original monolithic verification — one Z3 proof across all stages."""
        N = 2 * self.elements_per_vector
        elems, sorted_elems = self._simulate_steps(steps, natural_order)

        start = time.perf_counter()
        solver = Solver()
        sorted_ok = simplify(
            And([sorted_elems[i] <= sorted_elems[i + 1] for i in range(N - 1)])
        )
        solver.add(Not(sorted_ok))
        result = solver.check()
        solver_time = time.perf_counter() - start

        if result == unsat:
            return VerificationResult(verified=True, solver_time=solver_time)

        model = solver.model()
        ce = [model.evaluate(e, model_completion=True).as_long() for e in elems]
        return VerificationResult(
            verified=False, counterexample=ce, solver_time=solver_time
        )

    def _verify_steps_chunked(
        self,
        steps: list[VerifyStep],
        natural_order: bool = False,
        chunk_callback: Callable[[], None] | None = None,
    ) -> VerificationResult:
        """Chunked verification exploiting bitonic recursion structure.

        Splits the stage sequence at recursion boundaries and verifies each
        sub-group independently.  Each sub-group gets preconditions asserting
        that sub-lists from the previous recursion level are already sorted,
        and must prove that one specific group at the next level is sorted.

        For 16 elements this produces 7 verification steps:
          Level 0 (stages 0-2): 4 × prove one 4-element group sorted
          Level 1 (stages 3-5): 2 × prove one 8-element group sorted
          Level 2 (stages 6-9): 1 × prove full 16-element sort
        """
        N = 2 * self.elements_per_vector
        num_stages = len(steps)

        # Determine how many stages belong to the "real" sort (excluding
        # a possible appended natural-order stage)
        sort_stages = num_stages - 1 if natural_order else num_stages

        boundaries = compute_recursion_boundaries(N)

        # Build levels: [(stage_start, stage_end, pre_group_size, post_group_size)]
        levels: list[tuple[int, int, int, int]] = []
        prev = 0
        for level_idx, boundary_stage in enumerate(boundaries):
            end = boundary_stage + 1
            if end > sort_stages:
                break
            if end > prev:
                pre_gs = (
                    0 if level_idx == 0 else _MIN_GROUP_SIZE * (2 ** (level_idx - 1))
                )
                post_gs = _MIN_GROUP_SIZE * (2**level_idx)
                levels.append((prev, end, pre_gs, post_gs))
                prev = end

        # Final level: remaining stages (+ natural-order stage if any)
        if prev < num_stages:
            pre_gs = levels[-1][3] if levels else 0
            levels.append((prev, num_stages, pre_gs, N))

        total_solver_time = 0.0

        for level_idx, (start, end, pre_gs, post_gs) in enumerate(levels):
            is_last_level = level_idx == len(levels) - 1
            chunk_steps = steps[start:end]
            chunk_natural_order = natural_order and is_last_level

            # Verify each sub-group at this level independently
            num_groups = N // post_gs
            for group_idx in range(num_groups):
                group_elem_start = group_idx * post_gs
                group_elem_end = group_elem_start + post_gs

                result = self._verify_chunk(
                    chunk_steps,
                    precondition_group_size=pre_gs,
                    postcondition_range=(group_elem_start, group_elem_end),
                    natural_order=chunk_natural_order,
                )

                total_solver_time += result.solver_time

                if chunk_callback:
                    chunk_callback()

                if not result.verified:
                    return VerificationResult(
                        verified=False,
                        counterexample=result.counterexample,
                        solver_time=total_solver_time,
                    )

        return VerificationResult(verified=True, solver_time=total_solver_time)

    def _verify_chunk(
        self,
        steps: list[VerifyStep],
        precondition_group_size: int,
        postcondition_range: tuple[int, int],
        natural_order: bool = False,
    ) -> VerificationResult:
        """Verify a single chunk of stages for one output group.

        Args:
            steps: The subset of VerifyStep for this chunk.
            precondition_group_size: Elements are pre-sorted in groups of
                this size (0 = unconstrained).  E.g. 4 means [1-4] sorted,
                [5-8] sorted, etc.
            postcondition_range: (start, end) 0-based element indices for the
                group that must be sorted after this chunk.
            natural_order: If True, the last step in this chunk skips min/max.
        """
        N = 2 * self.elements_per_vector
        elems, sorted_elems = self._simulate_steps(steps, natural_order)

        start = time.perf_counter()
        solver = Solver()

        # Precondition: sub-lists of precondition_group_size are sorted
        if precondition_group_size > 0:
            pre_constraints = []
            for gs in range(0, N, precondition_group_size):
                ge = gs + precondition_group_size
                for i in range(gs, ge - 1):
                    pre_constraints.append(elems[i] <= elems[i + 1])
            solver.add(And(pre_constraints))

        # Postcondition: the specific group range is sorted
        post_start, post_end = postcondition_range
        post_constraints = [
            sorted_elems[i] <= sorted_elems[i + 1]
            for i in range(post_start, post_end - 1)
        ]

        sorted_ok = simplify(And(post_constraints))
        solver.add(Not(sorted_ok))
        result = solver.check()
        solver_time = time.perf_counter() - start

        if result == unsat:
            return VerificationResult(verified=True, solver_time=solver_time)

        model = solver.model()
        ce = [model.evaluate(e, model_completion=True).as_long() for e in elems]
        return VerificationResult(
            verified=False, counterexample=ce, solver_time=solver_time
        )

    def _build_register(self, elems: list, labels: list[int]):
        """Build a Z3 register from symbolic elements placed by label.

        Labels are 1-based: label k maps to elems[k-1].
        """
        lane_elems = [elems[label - 1] for label in labels]
        return simplify(Concat(lane_elems[::-1]))

    def _extract_output_elements(
        self, top_vec, bottom_vec, state: VectorState, n: int
    ) -> list:
        """Extract output elements in label order 1..N.

        Builds a mapping from element label -> (vector, lane) using the
        final output state, then reads elements in sorted label order.
        """
        label_to_pos: dict[int, tuple[str, int]] = {}
        for lane, label in enumerate(state.top):
            label_to_pos[label] = ("top", lane)
        for lane, label in enumerate(state.bottom):
            label_to_pos[label] = ("bottom", lane)

        sorted_elems = []
        for i in range(1, n + 1):
            vec_name, lane = label_to_pos[i]
            vec = top_vec if vec_name == "top" else bottom_vec
            lo = lane * self.lane_width
            hi = lo + self.lane_width - 1
            sorted_elems.append(simplify(Extract(hi, lo, vec)))

        return sorted_elems

    def _apply_concrete_instructions(
        self, top_reg, bottom_reg, instructions: list[InstructionSpec], is_top: bool
    ):
        """Apply concrete (non-symbolic) instructions to compute an output register.

        Simplified version of GadgetSynthesizer._apply_instructions without
        mux encoding or symbolic variables. All args are concrete.

        Instruction arguments can reference:
          - "top" / "bottom": original input registers (unchanged)
          - "input" / "a": current running register (latest output)
          - "prev": alias for current running register (same as "input")
          - "result_N": output of the Nth instruction (0-indexed), allowing
            3+ instruction chains to reference any earlier intermediate.
        """
        current_reg = top_reg if is_top else bottom_reg
        prev_output = None
        results: list = []

        for inst in instructions:
            intrinsic = self.intrinsics.get(inst.intrinsic_name)
            if intrinsic is None:
                raise ValueError(f"Unknown intrinsic: {inst.intrinsic_name}")

            args = {}
            for key, value in inst.args.items():
                args[key] = self._resolve_arg(
                    key,
                    value,
                    top_reg,
                    bottom_reg,
                    current_reg,
                    prev_output,
                    results,
                )

            current_reg = self._dispatch_intrinsic(intrinsic, args)
            prev_output = current_reg
            results.append(current_reg)

        return current_reg

    def _resolve_arg(
        self,
        key,
        value,
        top_reg,
        bottom_reg,
        current_reg,
        prev_output,
        results=None,
    ):
        """Resolve a concrete instruction argument to a Z3 value."""
        if isinstance(value, str):
            register_names = {
                "top": top_reg,
                "bottom": bottom_reg,
                "input": current_reg,
                "a": current_reg,
            }
            if value in register_names:
                return register_names[value]
            if value == "prev":
                if prev_output is None:
                    raise ValueError("'prev' referenced but no previous output")
                return prev_output
            if value.startswith("result_") and results is not None:
                idx = int(value[len("result_") :])
                if idx < 0 or idx >= len(results):
                    raise ValueError(
                        f"'{value}' out of range (only {len(results)} results available)"
                    )
                return results[idx]

        if isinstance(value, int):
            bit_widths = {
                "imm8": 8,
                "k": self.elements_per_vector,
            }
            return BitVecVal(value, bit_widths.get(key, self.total_bits))

        return value

    def _dispatch_intrinsic(self, intrinsic, args: dict):
        """Dispatch an intrinsic call using shared signature patterns."""
        return _dispatch_intrinsic_by_signature(intrinsic, args)
