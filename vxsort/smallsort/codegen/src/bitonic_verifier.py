"""End-to-end Z3 verification for bitonic sort paths.

Given a CompletePath (sequence of stages with specific gadget choices),
wire all stages together as a single Z3 expression and prove the output
is sorted for ALL possible inputs.
"""

from __future__ import annotations

import json
import time
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
    verifier = BitonicPathVerifier(vm, prim_type)
    result = verifier.verify_steps(steps, natural_order)
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
        self, steps: list[VerifyStep], natural_order: bool = False
    ) -> VerificationResult:
        """Verify that a sequence of gadget steps sorts all inputs correctly.

        Args:
            steps: List of VerifyStep (gadget + input/output state per stage).
            natural_order: If True, the last stage restores natural element
                order and has no min/max compare-swap.

        Returns:
            VerificationResult with verified=True if the path is a correct
            sorting network, or verified=False with a counterexample.
        """
        N = 2 * self.elements_per_vector
        num_stages = len(steps)

        # Step 1: Create N unconstrained symbolic elements
        elems = [BitVec(f"e_{i}", self.lane_width) for i in range(N)]

        # Step 2: Build initial registers from the first stage's input state
        initial_state = steps[0].input_state
        top_vec = self._build_register(elems, initial_state.top)
        bottom_vec = self._build_register(elems, initial_state.bottom)

        # Step 3: Process each stage
        for step_idx, step in enumerate(steps):
            gadget = step.gadget
            is_last_stage = step_idx == num_stages - 1

            # Apply gadget instructions to get permuted registers
            new_top = self._apply_concrete_instructions(
                top_vec, bottom_vec, gadget.top_instructions, is_top=True
            )
            new_bottom = self._apply_concrete_instructions(
                top_vec, bottom_vec, gadget.bottom_instructions, is_top=False
            )

            # Apply min/max compare-swap (unless natural_order final stage)
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

        # Step 4: Extract output elements in label order
        final_state = steps[-1].output_state
        sorted_elems = self._extract_output_elements(
            top_vec, bottom_vec, final_state, N
        )

        # Step 5: Assert NOT sorted, check UNSAT
        # Use signed <= (bvsle) since the sorting network uses signed min/max
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

        # SAT means there's a counterexample
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
