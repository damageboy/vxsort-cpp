from __future__ import annotations
import time
import copy
import os
import tarfile
import io
import zstandard as zstd
from dataclasses import dataclass
from tabulate import tabulate
from multiprocessing import Pool
from z3 import (
    Solver,
    Optimize,
    Context,
    main_ctx,
    Extract,
    BitVecVal,
    sat,
    BitVec,
    Distinct,
    Or,
    And,
    If,
    ULE,
    simplify,
)

try:
    from .success_progress import SuccessProgress
    from . import z3_avx
    from .bitonic_sorter import BitonicSorter
    from .utils import vector_machine, primitive_type, width_dict
except ImportError:
    from success_progress import SuccessProgress
    import z3_avx  # type: ignore
    from bitonic_sorter import BitonicSorter
    from utils import vector_machine, primitive_type, width_dict


def _all_smt(s, terms, max_results=None):
    """Enumerate all satisfying models over the given terms.

    Uses the corrected all_smt algorithm from 'Programming Z3'
    (Z3 issue #5765). Implemented iteratively with an explicit
    work-stack to avoid hitting Python's recursion limit when a
    variable has many valid assignments.

    Args:
        s: Z3 Solver instance.
        terms: Sequence of Z3 terms to enumerate over.
        max_results: Optional cap on the number of models returned.
            When ``None`` (the default) all models are enumerated.
    """
    terms = list(terms)
    count = 0

    # Each stack frame mirrors one call to the recursive algorithm:
    #   all_smt_rec(terms) would check sat, yield model, then iterate
    #   over terms[i:] pushing/blocking/fixing.
    # We represent the work as (terms_slice, model, next_i) where:
    #   terms_slice: the terms list for this "call"
    #   model: the model found at this level (None if not yet checked)
    #   next_i: the next index in the for-loop to process
    # A sentinel value of model=None means "need to check sat first".
    stack = [(terms, None, 0)]

    while stack:
        if max_results is not None and count >= max_results:
            # Unwind all pushed solver scopes
            for _ in range(len(stack) - 1):
                s.pop()
            break

        cur_terms, model, next_i = stack[-1]

        # First visit to this frame: check satisfiability
        if model is None:
            if sat == s.check():
                model = s.model()
                count += 1
                stack[-1] = (cur_terms, model, 0)
                yield model
                continue
            else:
                # Unsatisfiable — pop this frame
                stack.pop()
                if stack:
                    s.pop()
                continue

        # Advance the for-loop over terms
        if next_i >= len(cur_terms):
            # Done with all sub-partitions at this level
            stack.pop()
            if stack:
                s.pop()
            continue

        # Process partition i = next_i
        i = next_i
        stack[-1] = (cur_terms, model, next_i + 1)

        # Push solver scope and add blocking/fixing constraints
        s.push()
        s.add(cur_terms[i] != model.eval(cur_terms[i], model_completion=True))
        for j in range(i):
            s.add(cur_terms[j] == model.eval(cur_terms[j], model_completion=True))

        # "Recurse" into all_smt_rec(cur_terms[i:])
        stack.append((cur_terms[i:], None, 0))


@dataclass
class VectorState:
    """Tracks which elements are in which lanes of top/bottom vectors."""

    top: list[int]  # element indices in top vector
    bottom: list[int]  # element indices in bottom vector

    def __repr__(self):
        # Create transposed table: Top and Bottom as rows, lanes as columns
        max_len = max(len(self.top), len(self.bottom))

        # Pad vectors if needed
        top_vals = self.top + [""] * (max_len - len(self.top))
        bottom_vals = self.bottom + [""] * (max_len - len(self.bottom))

        # Create table data: each row is [label, val0, val1, val2, ...]
        table_data = [["Top"] + top_vals, ["Bottom"] + bottom_vals]

        # Headers are lane indices
        headers = [""] + list(range(max_len))
        return tabulate(table_data, headers=headers, tablefmt="rounded_outline")

    def copy(self):
        return VectorState(top=self.top.copy(), bottom=self.bottom.copy())

    def as_tuple(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return a hashable tuple representation for grouping/deduplication."""
        return (tuple(self.top), tuple(self.bottom))


@dataclass
class SymbolicPlaceholder:
    """Represents a symbolic Z3 variable that can be safely pickled."""

    name: str
    size: int


@dataclass
class InstructionSpec:
    """Represents a single AVX instruction with its arguments."""

    intrinsic_name: str
    args: dict  # operands, immediates, masks, etc.

    def sort_key(self) -> tuple:
        """Return a deterministic sort key for canonical ordering."""
        return (
            self.intrinsic_name,
            tuple(sorted((k, str(v)) for k, v in self.args.items())),
        )

    def __repr__(self):
        return f"{self.intrinsic_name}({self.args})"


@dataclass
class PermutationGadget:
    """Encapsulates instruction sequence for top/bottom vectors."""

    top_instructions: list[InstructionSpec]  # 0-3 instructions
    bottom_instructions: list[InstructionSpec]  # 0-3 instructions
    validated: bool = False

    def instruction_count(self) -> int:
        """Total number of instructions in this gadget."""
        return len(self.top_instructions) + len(self.bottom_instructions)

    def sort_key(self) -> tuple:
        """Return a deterministic sort key for canonical ordering."""
        return (
            tuple(i.sort_key() for i in self.top_instructions),
            tuple(i.sort_key() for i in self.bottom_instructions),
        )

    def __repr__(self):
        return f"Gadget(top={len(self.top_instructions)}, bottom={len(self.bottom_instructions)}, validated={self.validated})"


@dataclass
class SolutionNode:
    """Tree node for one stage's solutions.

    Each node represents a unique (input_state, output_state) transition.
    Multiple gadgets that achieve the same transition are stored together,
    allowing pruning of semantically equivalent paths.
    """

    stage: int
    input_state: VectorState
    output_state: VectorState
    gadgets: list[PermutationGadget]  # All gadgets that produce this transition
    children: list["SolutionNode"]

    def __repr__(self):
        return f"SolutionNode(stage={self.stage}, gadgets={len(self.gadgets)}, children={len(self.children)})"


def get_available_intrinsics(
    vm: vector_machine, prim_type: primitive_type
) -> dict[str, callable]:
    """Get available intrinsics for the given VM and primitive type."""
    intrinsics = {}

    # For AVX2 i32, we need YMM (256-bit) operations on 32-bit elements
    if vm == vector_machine.AVX2 and prim_type == primitive_type.i32:
        # Single input permutes (with immediates)
        intrinsics["_mm256_permute4x64_epi64"] = z3_avx._mm256_permute4x64_epi64
        intrinsics["_mm256_permute_ps"] = z3_avx._mm256_permute_ps

        # Single input permutes (with control vectors)
        intrinsics["_mm256_permutexvar_epi32"] = z3_avx._mm256_permutexvar_epi32
        intrinsics["_mm256_permutevar_ps"] = z3_avx._mm256_permutevar_ps

        # Two input permutes/shuffles
        intrinsics["_mm256_shuffle_ps"] = z3_avx._mm256_shuffle_ps
        intrinsics["_mm256_unpacklo_epi32"] = z3_avx._mm256_unpacklo_epi32
        intrinsics["_mm256_unpackhi_epi32"] = z3_avx._mm256_unpackhi_epi32
        intrinsics["_mm256_permute2x128_si256"] = z3_avx._mm256_permute2x128_si256

        # Blends
        intrinsics["_mm256_blend_ps"] = z3_avx._mm256_blend_ps
        intrinsics["_mm256_blendv_ps"] = z3_avx._mm256_blendv_ps

        # Align operations
        intrinsics["_mm256_alignr_epi32"] = z3_avx._mm256_alignr_epi32

    # For AVX2 i64, we need YMM (256-bit) operations on 64-bit elements
    if vm == vector_machine.AVX2 and prim_type == primitive_type.i64:
        # Single input permutes (with immediates)
        intrinsics["_mm256_permute4x64_epi64"] = z3_avx._mm256_permute4x64_epi64
        intrinsics["_mm256_permute_pd"] = z3_avx._mm256_permute_pd

        # Single input permutes (with control vectors)
        intrinsics["_mm256_permutexvar_epi64"] = z3_avx._mm256_permutexvar_epi64
        intrinsics["_mm256_permutevar_pd"] = z3_avx._mm256_permutevar_pd

        # Two input permutes/shuffles
        intrinsics["_mm256_shuffle_pd"] = z3_avx._mm256_shuffle_pd
        intrinsics["_mm256_unpacklo_epi64"] = z3_avx._mm256_unpacklo_epi64
        intrinsics["_mm256_unpackhi_epi64"] = z3_avx._mm256_unpackhi_epi64
        intrinsics["_mm256_permute2x128_si256"] = z3_avx._mm256_permute2x128_si256

        # Blends
        intrinsics["_mm256_blend_pd"] = z3_avx._mm256_blend_pd
        intrinsics["_mm256_blendv_pd"] = z3_avx._mm256_blendv_pd

        # Align operations
        intrinsics["_mm256_alignr_epi64"] = z3_avx._mm256_alignr_epi64

    # For AVX512 i64, we need ZMM (512-bit) operations on 64-bit elements
    if vm == vector_machine.AVX512 and prim_type == primitive_type.i64:
        # Unmasked single-input
        intrinsics["_mm512_permutexvar_epi64"] = z3_avx._mm512_permutexvar_epi64
        intrinsics["_mm512_permute_pd"] = z3_avx._mm512_permute_pd
        intrinsics["_mm512_permutevar_pd"] = z3_avx._mm512_permutevar_pd
        # Masked single-input
        intrinsics["_mm512_mask_permutexvar_epi64"] = (
            z3_avx._mm512_mask_permutexvar_epi64
        )
        intrinsics["_mm512_mask_permute_pd"] = z3_avx._mm512_mask_permute_pd
        intrinsics["_mm512_mask_permutevar_pd"] = z3_avx._mm512_mask_permutevar_pd
        # Unmasked dual-input
        intrinsics["_mm512_permutex2var_epi64"] = z3_avx._mm512_permutex2var_epi64
        intrinsics["_mm512_shuffle_pd"] = z3_avx._mm512_shuffle_pd
        intrinsics["_mm512_unpacklo_epi64"] = z3_avx._mm512_unpacklo_epi64
        intrinsics["_mm512_unpackhi_epi64"] = z3_avx._mm512_unpackhi_epi64
        intrinsics["_mm512_shuffle_i32x4"] = z3_avx._mm512_shuffle_i32x4
        intrinsics["_mm512_alignr_epi64"] = z3_avx._mm512_alignr_epi64
        # Masked dual-input
        intrinsics["_mm512_mask_permutex2var_epi64"] = (
            z3_avx._mm512_mask_permutex2var_epi64
        )
        intrinsics["_mm512_mask_shuffle_pd"] = z3_avx._mm512_mask_shuffle_pd
        intrinsics["_mm512_mask_unpacklo_epi64"] = z3_avx._mm512_mask_unpacklo_epi64
        intrinsics["_mm512_mask_unpackhi_epi64"] = z3_avx._mm512_mask_unpackhi_epi64
        intrinsics["_mm512_mask_shuffle_i32x4"] = z3_avx._mm512_mask_shuffle_i32x4
        intrinsics["_mm512_mask_alignr_epi64"] = z3_avx._mm512_mask_alignr_epi64

    # For AVX512 i32, we need ZMM (512-bit) operations on 32-bit elements
    if vm == vector_machine.AVX512 and prim_type == primitive_type.i32:
        # Unmasked single-input
        intrinsics["_mm512_permutexvar_epi32"] = z3_avx._mm512_permutexvar_epi32
        intrinsics["_mm512_permute_ps"] = z3_avx._mm512_permute_ps
        intrinsics["_mm512_permutevar_ps"] = z3_avx._mm512_permutevar_ps
        # Masked single-input
        intrinsics["_mm512_mask_permutexvar_epi32"] = (
            z3_avx._mm512_mask_permutexvar_epi32
        )
        intrinsics["_mm512_mask_permute_ps"] = z3_avx._mm512_mask_permute_ps
        intrinsics["_mm512_mask_permutevar_ps"] = z3_avx._mm512_mask_permutevar_ps
        # Unmasked dual-input
        intrinsics["_mm512_permutex2var_epi32"] = z3_avx._mm512_permutex2var_epi32
        intrinsics["_mm512_shuffle_ps"] = z3_avx._mm512_shuffle_ps
        intrinsics["_mm512_unpacklo_epi32"] = z3_avx._mm512_unpacklo_epi32
        intrinsics["_mm512_unpackhi_epi32"] = z3_avx._mm512_unpackhi_epi32
        intrinsics["_mm512_shuffle_i32x4"] = z3_avx._mm512_shuffle_i32x4
        intrinsics["_mm512_alignr_epi32"] = z3_avx._mm512_alignr_epi32
        # Masked dual-input
        intrinsics["_mm512_mask_permutex2var_epi32"] = (
            z3_avx._mm512_mask_permutex2var_epi32
        )
        intrinsics["_mm512_mask_shuffle_ps"] = z3_avx._mm512_mask_shuffle_ps
        intrinsics["_mm512_mask_unpacklo_epi32"] = z3_avx._mm512_mask_unpacklo_epi32
        intrinsics["_mm512_mask_unpackhi_epi32"] = z3_avx._mm512_mask_unpackhi_epi32
        intrinsics["_mm512_mask_shuffle_i32x4"] = z3_avx._mm512_mask_shuffle_i32x4
        intrinsics["_mm512_mask_alignr_epi32"] = z3_avx._mm512_mask_alignr_epi32

    return intrinsics


class GadgetSynthesizer:
    """Synthesizes permutation gadgets using Z3."""

    def __init__(self, vm: vector_machine, prim_type: primitive_type):
        self.vm = vm
        self.prim_type = prim_type
        self.elements_per_vector = width_dict[vm] // int(prim_type.value[0])
        self.lane_width = int(prim_type.value[0]) * 8  # 16, 32, or 64 bits per element
        self.available_intrinsics = get_available_intrinsics(vm, prim_type)

        # Memoize instruction templates - these are reused across all gadget generation
        self.single_insts_top = self._enumerate_single_input_instructions("top")
        self.single_insts_bottom = self._enumerate_single_input_instructions("bottom")
        self.dual_insts_top_bottom = self._enumerate_dual_input_instructions(
            "top", "bottom"
        )

    def _create_input_registers(
        self,
        solver: Solver,
        ctx: Context,
        input_state: VectorState,
    ) -> tuple:
        """
        Create Z3 symbolic registers with actual element values.
        Returns (top_reg, bottom_reg).
        """
        # Create registers based on VM type
        if self.vm == vector_machine.AVX2:
            top_reg = z3_avx.ymm_reg("top_input", ctx=ctx)
            bottom_reg = z3_avx.ymm_reg("bottom_input", ctx=ctx)
        elif self.vm == vector_machine.AVX512:
            top_reg = z3_avx.zmm_reg("top_input", ctx=ctx)
            bottom_reg = z3_avx.zmm_reg("bottom_input", ctx=ctx)
        else:
            raise NotImplementedError(
                f"Register creation not implemented for VM: {self.vm}"
            )

        # Set up constraints: each lane contains the actual element value
        for lane_idx in range(self.elements_per_vector):
            top_elem = input_state.top[lane_idx]
            bottom_elem = input_state.bottom[lane_idx]

            lane_start = lane_idx * self.lane_width
            lane_end = lane_start + self.lane_width - 1

            top_lane = Extract(lane_end, lane_start, top_reg)
            bottom_lane = Extract(lane_end, lane_start, bottom_reg)

            solver.add(top_lane == BitVecVal(top_elem, self.lane_width, ctx=ctx))
            solver.add(bottom_lane == BitVecVal(bottom_elem, self.lane_width, ctx=ctx))

        return top_reg, bottom_reg

    def synthesize_gadget_with_symbolic(
        self,
        top_instructions_template: list[InstructionSpec],
        bottom_instructions_template: list[InstructionSpec],
        input_state: VectorState,
        target_pairs: list[tuple[int, int]],
        max_solutions: int = 1,
        max_unique_outputs: int = 3,
        solver_callback: callable | None = None,
        allow_any_lane_order: bool = True,
    ) -> tuple[list[tuple[PermutationGadget, VectorState]], float, float]:
        """
        Synthesize gadgets using symbolic immediates in Z3.

        Takes instruction templates with symbolic values (e.g., BitVec("imm8", 8))
        and lets Z3 find concrete immediate values that satisfy constraints.

        The symbolic values are represented using SymbolicPlaceholder for pickling.
        The pickling is required for multiprocessing.

        Returns (results, construction_time, solver_time) where results is
        list of (gadget, output_state) tuples. The output state is computed
        directly from the satisfying model, avoiding a redundant Z3 solve. The output
        state is in canonical form: for each pair, the lower element index goes to
        the top vector and the higher index goes to bottom, reflecting the
        compare-and-exchange operation.

        Args:
            max_solutions: Maximum number of solutions to return. Defaults to 1,
                which enumerates unique output states deterministically.
                Values > 1 use Z3 Solver with enumeration (non-deterministic ordering).
            max_unique_outputs: When max_solutions == 1, the number of smallest
                unique output states to enumerate per template. Higher values
                increase search diversity at the cost of speed. Default: 3.
            solver_callback: Optional callback receiving the Solver instance.
            allow_any_lane_order: When True (default), any target pair can land
                in any lane and the output state is canonicalized (min to top,
                max to bottom). When False, pair[i] is pinned to lane i with
                strict top/bottom assignment and no canonicalization.
        """
        start_construction = time.perf_counter()
        ctx = main_ctx()
        solver = Solver(ctx=ctx)

        # Create input registers with actual element values
        top_reg, bottom_reg = self._create_input_registers(solver, ctx, input_state)

        # Collect all symbolic variables from instruction templates and resolve them
        symbolic_vars = {}

        def maybe_resolve_symbolic_vars(instructions: list[InstructionSpec]):
            """Resolve SymbolicPlaceholder to actual Z3 variables."""

            # SymbolicPlaceholder is used to communicate with multiprocessing
            # z3 runs in "production"
            # In testing code, the z3 expressions are passed directly
            # So we need to handle both cases here
            for inst in instructions:
                for key, value in inst.args.items():
                    if isinstance(value, SymbolicPlaceholder):
                        if value.size == 8:
                            actual_val = BitVec(value.name, 8, ctx=ctx)
                        elif value.size == 16:
                            actual_val = BitVec(value.name, 16, ctx=ctx)
                        elif value.size == 256:
                            actual_val = z3_avx.ymm_reg(value.name, ctx=ctx)
                        elif value.size == 512:
                            actual_val = z3_avx.zmm_reg(value.name, ctx=ctx)
                        else:
                            actual_val = BitVec(value.name, value.size, ctx=ctx)

                        inst.args[key] = actual_val
                        symbolic_vars[id(actual_val)] = actual_val
                    elif hasattr(value, "decl") and callable(
                        getattr(value, "decl", None)
                    ):
                        # Already a Z3 expression (e.g. from tests)
                        symbolic_vars[id(value)] = value

        maybe_resolve_symbolic_vars(top_instructions_template)
        maybe_resolve_symbolic_vars(bottom_instructions_template)

        # Apply gadget instructions to get output registers
        top_output, top_mux_constraints = self._apply_instructions(
            top_reg,
            bottom_reg,
            top_instructions_template,
            is_top=True,
            solver=solver,
            symbolic_vars=symbolic_vars,
        )
        bottom_output, bottom_mux_constraints = self._apply_instructions(
            top_reg,
            bottom_reg,
            bottom_instructions_template,
            is_top=False,
            solver=solver,
            symbolic_vars=symbolic_vars,
        )

        # Add mux range constraints (select vars must be in {0, 1, 2})
        for c in top_mux_constraints:
            solver.add(c)
        for c in bottom_mux_constraints:
            solver.add(c)

        # Add constraints: for each output lane, the top/bottom elements must
        # form a valid target pair (in either orientation).
        all_output_lanes = []
        for lane_idx in range(self.elements_per_vector):
            lane_start = lane_idx * self.lane_width
            lane_end = lane_start + self.lane_width - 1

            top_lane = Extract(lane_end, lane_start, top_output)
            bottom_lane = Extract(lane_end, lane_start, bottom_output)

            all_output_lanes.append(top_lane)
            all_output_lanes.append(bottom_lane)

            if allow_any_lane_order:
                # Any target pair can land in any lane (existing behavior)
                pair_options = []
                for elem_a, elem_b in target_pairs:
                    val_a = BitVecVal(elem_a, self.lane_width, ctx=ctx)
                    val_b = BitVecVal(elem_b, self.lane_width, ctx=ctx)
                    pair_options.append(
                        Or(
                            And(top_lane == val_a, bottom_lane == val_b),
                            And(top_lane == val_b, bottom_lane == val_a),
                        )
                    )
                solver.add(Or(*pair_options))
            else:
                # Strict: pair[lane_idx] pinned to this lane, first->top, second->bottom
                elem_a, elem_b = target_pairs[lane_idx]
                solver.add(top_lane == BitVecVal(elem_a, self.lane_width, ctx=ctx))
                solver.add(bottom_lane == BitVecVal(elem_b, self.lane_width, ctx=ctx))

        # All output elements must be distinct — prevents element duplication
        # where an instruction copies the same element to both top and bottom.
        solver.add(Distinct(*all_output_lanes))

        if solver_callback:
            solver_callback(solver)

        # When strict lane order, the output state is deterministic from pairs
        fixed_output_state = None
        if not allow_any_lane_order:
            fixed_output_state = VectorState(
                top=[p[0] for p in target_pairs],
                bottom=[p[1] for p in target_pairs],
            )

        # Partition symbolic vars: select variables (mux wiring) vs
        # immediate/control vector variables
        select_vars = {
            k: v
            for k, v in symbolic_vars.items()
            if isinstance(k, str) and k.startswith("sel_")
        }
        imm_vars = {k: v for k, v in symbolic_vars.items() if k not in select_vars}
        imm_terms = list(imm_vars.values())

        construction_time = time.perf_counter() - start_construction
        solver_start = time.perf_counter()

        # If no symbolic variables at all, single check suffices
        if not imm_terms and not select_vars:
            result = solver.check()
            if result != sat:
                solver_time = time.perf_counter() - solver_start
                return [], construction_time, solver_time
            model = solver.model()
            gadget, output_state = self._extract_solution_from_model(
                model,
                top_output,
                bottom_output,
                symbolic_vars,
                top_instructions_template,
                bottom_instructions_template,
                fixed_output_state=fixed_output_state,
            )
            solver_time = time.perf_counter() - solver_start
            return [(gadget, output_state)], construction_time, solver_time

        if max_solutions == 1:
            # K-per-wiring algorithm: deterministic synthesis with diversity.
            #
            # Outer loop: discover unique operand wirings (which source each
            # inst2 operand reads from: top, bottom, or prev). The wiring
            # space is small (at most 3^num_select_vars) and many wirings
            # are UNSAT, so this terminates quickly.
            #
            # Inner loop: for each valid wiring, find the K smallest unique
            # output states using Optimize.minimize with output-blocking.
            # For each unique output, pin the output + wiring and minimize
            # immediates for a deterministic gadget.
            #
            # For depth-1 templates (no select vars), the outer loop runs
            # once and the inner loop behaves identically to the previous
            # K-smallest algorithm — fully backward compatible.
            original_assertions = list(solver.assertions())
            results = []
            wiring_blocks = []

            # Outer loop: discover unique wirings
            while True:
                opt_wiring = Optimize(ctx=ctx)
                for a in original_assertions:
                    opt_wiring.add(a)
                for block in wiring_blocks:
                    opt_wiring.add(block)
                # Minimize output to get a deterministic first hit
                opt_wiring.minimize(top_output)
                opt_wiring.minimize(bottom_output)

                if opt_wiring.check() != sat:
                    break  # No more valid wirings

                model = opt_wiring.model()

                # Extract the wiring from the model
                wiring = {}
                for name, var in select_vars.items():
                    wiring[name] = model.evaluate(var, model_completion=True).as_long()

                # Pin this wiring for the inner loop
                wiring_constraints = [
                    var == BitVecVal(wiring[name], 2)
                    for name, var in select_vars.items()
                ]

                # Inner loop: K smallest unique outputs for this wiring
                output_blocks = []
                for _ in range(max_unique_outputs):
                    opt_k = Optimize(ctx=ctx)
                    for a in original_assertions:
                        opt_k.add(a)
                    for c in wiring_constraints:
                        opt_k.add(c)
                    for block in output_blocks:
                        opt_k.add(block)
                    opt_k.minimize(top_output)
                    opt_k.minimize(bottom_output)

                    if opt_k.check() != sat:
                        break

                    model_k = opt_k.model()
                    top_val = simplify(model_k.eval(top_output))
                    bot_val = simplify(model_k.eval(bottom_output))

                    # Pin output + wiring, minimize immediates only
                    opt_imm = Optimize(ctx=ctx)
                    for a in original_assertions:
                        opt_imm.add(a)
                    for c in wiring_constraints:
                        opt_imm.add(c)
                    opt_imm.add(top_output == top_val)
                    opt_imm.add(bottom_output == bot_val)
                    for term in imm_terms:
                        opt_imm.minimize(term)

                    if opt_imm.check() == sat:
                        gadget, output_state = self._extract_solution_from_model(
                            opt_imm.model(),
                            top_output,
                            bottom_output,
                            symbolic_vars,
                            top_instructions_template,
                            bottom_instructions_template,
                            fixed_output_state=fixed_output_state,
                        )
                        results.append((gadget, output_state))

                    # Block this output for the next inner iteration
                    output_blocks.append(
                        Or(top_output != top_val, bottom_output != bot_val)
                    )

                # Block this entire wiring for the next outer iteration
                if select_vars:
                    wiring_blocks.append(
                        Or(
                            *(
                                var != BitVecVal(wiring[name], 2)
                                for name, var in select_vars.items()
                            )
                        )
                    )
                else:
                    # No select vars (depth-1): only run outer loop once
                    break

            # Sort by output state for deterministic ordering
            results.sort(key=lambda x: x[1].as_tuple())
            solver_time = time.perf_counter() - solver_start
            return results, construction_time, solver_time

        # Enumerate multiple solutions over symbolic variables
        all_terms = list(symbolic_vars.values())
        results = []
        for model in _all_smt(solver, all_terms, max_results=max_solutions):
            gadget, output_state = self._extract_solution_from_model(
                model,
                top_output,
                bottom_output,
                symbolic_vars,
                top_instructions_template,
                bottom_instructions_template,
                fixed_output_state=fixed_output_state,
            )
            results.append((gadget, output_state))
        solver_time = time.perf_counter() - solver_start
        return results, construction_time, solver_time

    def _extract_solution_from_model(
        self,
        model,
        top_output,
        bottom_output,
        symbolic_vars: dict,
        top_instructions_template: list[InstructionSpec],
        bottom_instructions_template: list[InstructionSpec],
        fixed_output_state: VectorState | None = None,
    ) -> tuple[PermutationGadget, VectorState]:
        """Extract a concrete gadget and output state from a Z3 model.

        Reads element values from the symbolic output registers, builds the
        canonical output state (min to top, max to bottom per lane), and
        concretizes all symbolic variables in the instruction templates
        using the model's assignments.

        When ``fixed_output_state`` is provided (strict lane order mode), the
        output state is used directly without reading from the model or
        applying min/max canonicalization.
        """
        if fixed_output_state is not None:
            output_state = fixed_output_state
        else:
            # Extract element values from output registers to build canonical output state
            output_top = []
            output_bottom = []
            for lane_idx in range(self.elements_per_vector):
                lane_start = lane_idx * self.lane_width
                lane_end = lane_start + self.lane_width - 1

                top_lane = Extract(lane_end, lane_start, top_output)
                bottom_lane = Extract(lane_end, lane_start, bottom_output)

                top_val = model.evaluate(top_lane, model_completion=True)
                bottom_val = model.evaluate(bottom_lane, model_completion=True)

                top_elem = top_val.as_long() if hasattr(top_val, "as_long") else top_val
                bottom_elem = (
                    bottom_val.as_long()
                    if hasattr(bottom_val, "as_long")
                    else bottom_val
                )

                # Canonical ordering: lower element index to top, higher to bottom
                output_top.append(min(top_elem, bottom_elem))
                output_bottom.append(max(top_elem, bottom_elem))

            output_state = VectorState(top=output_top, bottom=output_bottom)

        # Create concrete instructions by substituting symbolic values.
        # Select variables (from mux encoding) are resolved to source name
        # strings ("top", "bottom", "prev") via _SELECT_TO_SOURCE.
        def concretize_instructions(
            instructions: list[InstructionSpec],
        ) -> list[InstructionSpec]:
            concrete_insts = []
            for inst in instructions:
                concrete_args = {}
                for key, value in inst.args.items():
                    # Check if this arg has a corresponding select variable
                    select_name = f"sel_{key}_{inst.intrinsic_name}_{id(inst)}"
                    if select_name in symbolic_vars:
                        select_val = model.evaluate(
                            symbolic_vars[select_name], model_completion=True
                        ).as_long()
                        concrete_args[key] = self._SELECT_TO_SOURCE[select_val]
                    elif id(value) in symbolic_vars:
                        if hasattr(value, "size") and callable(
                            getattr(value, "size", None)
                        ):
                            bit_size = value.size()
                            if bit_size == 256:
                                concrete_bitvec = model.evaluate(
                                    value, model_completion=True
                                )
                                if hasattr(concrete_bitvec, "as_long"):
                                    concrete_value = concrete_bitvec.as_long()
                                else:
                                    concrete_value = concrete_bitvec
                                concrete_args[key] = concrete_value
                            elif bit_size == 8:
                                concrete_value = model.evaluate(
                                    value, model_completion=True
                                ).as_long()
                                concrete_args[key] = concrete_value
                            else:
                                concrete_value = model.evaluate(
                                    value, model_completion=True
                                ).as_long()
                                concrete_args[key] = concrete_value
                        else:
                            concrete_value = model.evaluate(
                                value, model_completion=True
                            ).as_long()
                            concrete_args[key] = concrete_value
                    else:
                        concrete_args[key] = value
                concrete_insts.append(
                    InstructionSpec(inst.intrinsic_name, concrete_args)
                )
            return concrete_insts

        concrete_top = concretize_instructions(top_instructions_template)
        concrete_bottom = concretize_instructions(bottom_instructions_template)

        gadget = PermutationGadget(
            top_instructions=concrete_top,
            bottom_instructions=concrete_bottom,
            validated=True,
        )

        return gadget, output_state

    def compute_output_state(
        self, input_state: VectorState, gadget: PermutationGadget
    ) -> VectorState:
        """
        Compute the output state after applying a gadget to an input state.

        Uses Z3 to symbolically execute the gadget and determine element positions.

        Args:
            input_state: Input element positions
            gadget: Gadget to apply

        Returns:
            Output state with new element positions
        """
        ctx = main_ctx()
        solver = Solver(ctx=ctx)

        # Create input registers where each lane contains the element index
        if self.vm == vector_machine.AVX2:
            top_reg = z3_avx.ymm_reg("top_input", ctx=ctx)
            bottom_reg = z3_avx.ymm_reg("bottom_input", ctx=ctx)
        elif self.vm == vector_machine.AVX512:
            top_reg = z3_avx.zmm_reg("top_input", ctx=ctx)
            bottom_reg = z3_avx.zmm_reg("bottom_input", ctx=ctx)
        else:
            raise NotImplementedError(
                f"Register creation not implemented for VM: {self.vm}"
            )

        # Constrain input registers to contain element indices
        for lane_idx in range(self.elements_per_vector):
            top_elem = input_state.top[lane_idx]
            bottom_elem = input_state.bottom[lane_idx]

            lane_start = lane_idx * self.lane_width
            lane_end = lane_start + self.lane_width - 1

            top_lane = Extract(lane_end, lane_start, top_reg)
            bottom_lane = Extract(lane_end, lane_start, bottom_reg)

            solver.add(top_lane == BitVecVal(top_elem, self.lane_width, ctx=ctx))
            solver.add(bottom_lane == BitVecVal(bottom_elem, self.lane_width, ctx=ctx))

        # Apply gadget instructions (mux constraints ignored here —
        # concrete gadgets have resolved args, no select variables)
        top_output, _ = self._apply_instructions(
            top_reg, bottom_reg, gadget.top_instructions, is_top=True
        )
        bottom_output, _ = self._apply_instructions(
            top_reg, bottom_reg, gadget.bottom_instructions, is_top=False
        )

        # Solve to get concrete output values
        if solver.check() != sat:
            # If unsatisfiable, return input state (shouldn't happen with valid gadgets)
            print("Warning: Could not compute output state, returning input")
            return input_state.copy()

        model = solver.model()

        # Extract output element positions from the model
        output_top = []
        output_bottom = []

        for lane_idx in range(self.elements_per_vector):
            lane_start = lane_idx * self.lane_width
            lane_end = lane_start + self.lane_width - 1

            top_lane = Extract(lane_end, lane_start, top_output)
            bottom_lane = Extract(lane_end, lane_start, bottom_output)

            # Evaluate the output lanes in the model
            top_val = model.evaluate(top_lane, model_completion=True)
            bottom_val = model.evaluate(bottom_lane, model_completion=True)

            # Convert Z3 bit-vector values to Python integers
            top_val_long = top_val.as_long() if hasattr(top_val, "as_long") else top_val
            bottom_val_long = (
                bottom_val.as_long() if hasattr(bottom_val, "as_long") else bottom_val
            )

            output_top.append(top_val_long)
            output_bottom.append(bottom_val_long)

        return VectorState(top=output_top, bottom=output_bottom)

    def _substitute_register_names(
        self, arg, top_reg, bottom_reg, current_reg, symbolic_vars=None, key=None
    ):
        """
        Substitute register name strings with actual Z3 register variables.

        Args:
            arg: Argument value (could be string register name, immediate, Z3 symbolic var, etc.)
            top_reg: Top Z3 register
            bottom_reg: Bottom Z3 register
            current_reg: Current Z3 register being computed
            symbolic_vars: Optional dict to track symbolic variables by name

        Returns:
            Actual register if arg is a register name, otherwise returns arg unchanged
        """
        # If it's a Z3 expression (BitVec), track it in symbolic_vars if it has a name
        if hasattr(arg, "decl") and callable(getattr(arg, "decl", None)):
            # This is a Z3 expression
            if symbolic_vars is not None:
                # Try to extract the variable name
                symbolic_vars[id(arg)] = arg
            return arg

        if isinstance(arg, int):
            # Check if this is a 256-bit (or 512-bit) control vector or an 8-bit immediate
            if key == "imm8":
                return BitVecVal(arg, 8)
            else:
                return BitVecVal(arg, width_dict[self.vm] * 8)

        if isinstance(arg, str):
            if arg == "top":
                return top_reg
            elif arg == "bottom":
                return bottom_reg
            elif arg in ["input", "a"]:  # Generic input register
                return current_reg
        return arg

    # Map select variable values to source name strings for extraction
    _SELECT_TO_SOURCE = {0: "top", 1: "bottom", 2: "prev"}

    def _create_operand_mux(
        self,
        key: str,
        inst: InstructionSpec,
        top_reg,
        bottom_reg,
        prev_output,
        symbolic_vars: dict | None,
        mux_constraints: list,
    ):
        """Create a Z3 If-else chain selecting between {top, bottom, prev}.

        For inst2+ in a multi-instruction chain, register operands ("a", "b")
        get a 2-bit select variable that Z3 uses to pick the optimal source.

        Args:
            key: Argument key ("a" or "b")
            inst: The instruction spec (used for unique naming)
            top_reg: Z3 register for original top input
            bottom_reg: Z3 register for original bottom input
            prev_output: Z3 register from the previous instruction's output
            symbolic_vars: Dict to track the select variable for later extraction
            mux_constraints: List to append range constraints to

        Returns:
            Z3 If-expression selecting between the three sources
        """
        select_name = f"sel_{key}_{inst.intrinsic_name}_{id(inst)}"
        select_var = BitVec(select_name, 2)

        if symbolic_vars is not None:
            symbolic_vars[select_name] = select_var

        # Constrain to exactly {0, 1, 2} — a 2-bit var can be 0-3
        mux_constraints.append(ULE(select_var, BitVecVal(2, 2)))

        mux = If(
            select_var == 0,
            top_reg,
            If(select_var == 1, bottom_reg, If(select_var == 2, prev_output, top_reg)),
        )  # unreachable due to constraint

        return mux

    def _apply_instructions(
        self,
        top_reg,
        bottom_reg,
        instructions: list[InstructionSpec],
        is_top: bool,
        solver=None,
        symbolic_vars=None,
    ):
        """
        Apply a sequence of instructions to compute output register.

        For multi-instruction sequences, inst2+ register operands get
        multiplexer select variables so Z3 can choose whether each operand
        reads from the original top/bottom registers or from the previous
        instruction's output.

        Args:
            top_reg: Input top register
            bottom_reg: Input bottom register
            instructions: List of instructions to apply
            is_top: True if computing top output, False for bottom output
            solver: Optional Z3 solver (unused but kept for API compatibility)
            symbolic_vars: Optional dict to track symbolic variables

        Returns:
            (output_register, mux_constraints) tuple. The caller must add
            mux_constraints to the solver before solving.
        """
        # Start with the appropriate input register
        current_reg = top_reg if is_top else bottom_reg
        prev_output = None
        mux_constraints = []

        for inst_idx, inst in enumerate(instructions):
            intrinsic = self.available_intrinsics.get(inst.intrinsic_name)
            if intrinsic is None:
                raise ValueError(f"Unknown intrinsic: {inst.intrinsic_name}")

            # Substitute register names in arguments with actual Z3 registers
            args = {}
            for key, value in inst.args.items():
                if (
                    inst_idx > 0
                    and prev_output is not None
                    and key in ("a", "b")
                    and isinstance(value, str)
                    and value in ("top", "bottom")
                ):
                    # inst2+: create mux for register operands
                    args[key] = self._create_operand_mux(
                        key,
                        inst,
                        top_reg,
                        bottom_reg,
                        prev_output,
                        symbolic_vars,
                        mux_constraints,
                    )
                else:
                    # First instruction or non-register arg: resolve normally
                    args[key] = self._substitute_register_names(
                        value,
                        top_reg,
                        bottom_reg,
                        current_reg,
                        symbolic_vars,
                        key=key,
                    )

            # Determine how to call the intrinsic based on its signature.
            # AVX512 masked patterns (most specific) come first, then AVX2 patterns.
            if (
                "k" in args
                and "src" in args
                and "op_idx" in args
                and "a" in args
                and "b" not in args
            ):
                # Masked single-input with control vector
                # e.g., _mm512_mask_permutexvar_epi64(src, k, op_idx, a)
                current_reg = intrinsic(
                    args["src"], args["k"], args["op_idx"], args["a"]
                )
            elif (
                "k" in args
                and "src" in args
                and "a" in args
                and "b" in args
                and "imm8" in args
            ):
                # Masked dual-input with immediate
                # e.g., _mm512_mask_shuffle_pd(src, k, a, b, imm8)
                current_reg = intrinsic(
                    args["src"], args["k"], args["a"], args["b"], args["imm8"]
                )
            elif (
                "k" in args
                and "src" in args
                and "a" in args
                and "imm8" in args
                and "b" not in args
            ):
                # Masked single-input with immediate
                # e.g., _mm512_mask_permute_pd(src, k, a, imm8)
                current_reg = intrinsic(args["src"], args["k"], args["a"], args["imm8"])
            elif (
                "k" in args
                and "src" in args
                and "a" in args
                and "b" in args
                and "imm8" not in args
            ):
                # Masked dual-input without immediate
                # e.g., _mm512_mask_unpacklo_epi64(src, k, a, b)
                current_reg = intrinsic(args["src"], args["k"], args["a"], args["b"])
            elif (
                "k" in args
                and "a" in args
                and "op_idx" in args
                and "b" in args
                and "src" not in args
            ):
                # Masked permutex2var: a is merge source
                # e.g., _mm512_mask_permutex2var_epi64(a, k, op_idx, b)
                current_reg = intrinsic(args["a"], args["k"], args["op_idx"], args["b"])
            elif "a" in args and "op_idx" in args and "b" in args and "k" not in args:
                # Unmasked permutex2var
                # e.g., _mm512_permutex2var_epi64(a, op_idx, b)
                current_reg = intrinsic(args["a"], args["op_idx"], args["b"])
            elif "a" in args and "op_idx" in args and "b" not in args:
                # Single source with control index (e.g., _mm256_permutexvar_epi32)
                current_reg = intrinsic(args["a"], args["op_idx"])
            elif "a" in args and "imm8" in args and "b" not in args:
                # Single source with immediate (e.g., _mm256_permute_ps)
                current_reg = intrinsic(args["a"], args["imm8"])
            elif "a" in args and "b" in args and "imm8" in args:
                # Two sources with immediate (e.g., _mm256_shuffle_ps)
                current_reg = intrinsic(args["a"], args["b"], args["imm8"])
            elif "a" in args and "b" in args and "mask" in args:
                # Blend with variable mask
                current_reg = intrinsic(args["a"], args["b"], args["mask"])
            elif (
                "a" in args
                and "b" in args
                and "imm8" not in args
                and "mask" not in args
            ):
                # Two sources without immediate (e.g., unpack, permutevar_ps)
                current_reg = intrinsic(args["a"], args["b"])
            else:
                # Generic fallback
                arg_values = list(args.values())
                current_reg = intrinsic(*arg_values)

            prev_output = current_reg

        return current_reg, mux_constraints

    def _generate_candidate_gadgets(
        self, top_depth: int, bottom_depth: int
    ) -> list[tuple[list[InstructionSpec], list[InstructionSpec]]]:
        """
        Generate candidate instruction sequences without validation.
        Returns list of (top_sequence, bottom_sequence) tuples.
        """
        # Build instruction sequences for top
        top_sequences = []
        if top_depth > 0:
            if top_depth == 1:
                top_sequences = [[inst] for inst in self.single_insts_top]
                top_sequences.extend([[inst] for inst in self.dual_insts_top_bottom])
            elif top_depth == 2:
                # Try pairs: (single, single), (dual, single), (single, dual)
                for inst1 in self.single_insts_top:
                    for inst2 in self.single_insts_top:
                        top_sequences.append([inst1, inst2])
                for inst1 in self.dual_insts_top_bottom:
                    for inst2 in self.single_insts_top:
                        top_sequences.append([inst1, inst2])
                for inst1 in self.single_insts_top:
                    for inst2 in self.dual_insts_top_bottom:
                        top_sequences.append([inst1, inst2])
            elif top_depth == 3:
                # Try triples: (single, single, single)
                for inst1 in self.single_insts_top:
                    for inst2 in self.single_insts_top:
                        for inst3 in self.single_insts_top:
                            top_sequences.append([inst1, inst2, inst3])
        else:
            top_sequences = [[]]  # Empty sequence for depth 0

        # Build instruction sequences for bottom
        bottom_sequences = []
        if bottom_depth > 0:
            if bottom_depth == 1:
                bottom_sequences = [[inst] for inst in self.single_insts_bottom]
                bottom_sequences.extend([[inst] for inst in self.dual_insts_top_bottom])
            elif bottom_depth == 2:
                for inst1 in self.single_insts_bottom:
                    for inst2 in self.single_insts_bottom:
                        bottom_sequences.append([inst1, inst2])
                for inst1 in self.dual_insts_top_bottom:
                    for inst2 in self.single_insts_bottom:
                        bottom_sequences.append([inst1, inst2])
                for inst1 in self.single_insts_bottom:
                    for inst2 in self.dual_insts_top_bottom:
                        bottom_sequences.append([inst1, inst2])
            elif bottom_depth == 3:
                for inst1 in self.single_insts_bottom:
                    for inst2 in self.single_insts_bottom:
                        for inst3 in self.single_insts_bottom:
                            bottom_sequences.append([inst1, inst2, inst3])
        else:
            bottom_sequences = [[]]  # Empty sequence for depth 0

        # Generate all combinations
        candidates = []
        for top_seq in top_sequences:
            for bottom_seq in bottom_sequences:
                candidates.append((top_seq, bottom_seq))

        return candidates

    def precompute_all_candidates(
        self, gadget_depth: int
    ) -> list[tuple[list[InstructionSpec], list[InstructionSpec]]]:
        """Pre-compute all candidate instruction sequences for all depth combinations.

        The candidates depend only on gadget_depth and the available intrinsics,
        not on input state or stage pairs. This allows computing them once and
        reusing across all stages and input states.
        """
        all_candidates = []
        for top_depth in range(gadget_depth + 1):
            for bottom_depth in range(gadget_depth + 1):
                candidates = self._generate_candidate_gadgets(top_depth, bottom_depth)
                all_candidates.extend(candidates)
        return all_candidates

    def _validate_gadgets(
        self,
        jobs: list[tuple],
        show_progress: bool = True,
    ) -> list[tuple[PermutationGadget, VectorState, VectorState, dict]]:
        """
        Validate candidate gadgets using synthesis in parallel.

        Returns list of (gadget, input_state, output_state, metadata) tuples.
        The output_state is computed directly during validation, avoiding
        a redundant Z3 solve.
        """
        validated_gadgets = []

        progress = None
        task_id = None
        if show_progress:
            progress = SuccessProgress.create(
                width=60,
                success_style="green",
                attempt_style="yellow",
                success_label="Valid",
            )
            progress.start()
            task_id = progress.add_task(
                "Validating gadgets", total=len(jobs), successes=0
            )

        try:
            pool = Pool()
            total_construct_time = 0.0
            total_solve_time = 0.0

            # Use imap_unordered for streaming results and progress updates
            for (
                gadget_results,
                job_input_state,
                job_metadata,
                construct_time,
                solver_time,
            ) in pool.imap_unordered(_validate_gadget_worker, jobs):
                total_construct_time += construct_time
                total_solve_time += solver_time

                success_inc = 0
                for gadget, output_state in gadget_results:
                    validated_gadgets.append(
                        (gadget, job_input_state, output_state, job_metadata)
                    )
                    success_inc = 1

                if progress:
                    progress.update(task_id, advance=1, success=success_inc)

            pool.close()
            pool.join()

            if jobs:
                print(
                    f"TOTAL construction time: {total_construct_time:.2f}s, TOTAL solver time: {total_solve_time:.2f}s"
                )

            # Phase 2.5: Compress tar files if smt2_dump_dir is set
            if jobs and "smt2_dump_dir" in jobs[0][6]:
                smt2_dump_dir = jobs[0][6]["smt2_dump_dir"]
                stage_idx = jobs[0][6]["stage_idx"]
                for filename in os.listdir(smt2_dump_dir):
                    if filename.startswith(f"stage{stage_idx}_") and filename.endswith(
                        ".tar"
                    ):
                        tar_path = os.path.join(smt2_dump_dir, filename)
                        zst_path = tar_path + ".zst"
                        cctx = zstd.ZstdCompressor()
                        with open(tar_path, "rb") as f_in:
                            with open(zst_path, "wb") as f_out:
                                cctx.copy_stream(f_in, f_out)
                        os.remove(tar_path)
        finally:
            if progress:
                progress.stop()

        # Sort validated gadgets by a deterministic key to ensure consistent
        # dict insertion order downstream, regardless of imap_unordered return order.
        validated_gadgets.sort(
            key=lambda x: (
                x[3]["parent_path"],
                x[1].as_tuple(),
                x[2].as_tuple(),
            )
        )

        return validated_gadgets

    def _enumerate_single_input_instructions(
        self, reg_name: str = "input"
    ) -> list[InstructionSpec]:
        """
        Generate single-input instruction templates with symbolic immediates or control vectors.
        Z3 will solve for the concrete values.

        Single-input means: operates on ONE of our vectors (top OR bottom),
        even if it takes additional operands like control vectors.
        """
        if self.vm == vector_machine.AVX2 and self.prim_type == primitive_type.i32:
            input_reg = reg_name
            unique_id = id(input_reg)

            # Permute within 128-bit lanes using immediate
            permute_ps = InstructionSpec(
                "_mm256_permute_ps",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute_ps_{unique_id}", 8),
                },
            )

            # Permute 64-bit chunks across full register
            permute4x64 = InstructionSpec(
                "_mm256_permute4x64_epi64",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute4x64_{unique_id}", 8),
                },
            )

            # Variable permute across all lanes (most powerful)
            permutexvar = InstructionSpec(
                "_mm256_permutexvar_epi32",
                {
                    "a": input_reg,
                    "op_idx": SymbolicPlaceholder(f"ctrl_permutexvar_{unique_id}", 256),
                },
            )

            # Variable permute within 128-bit lanes
            permutevar_ps = InstructionSpec(
                "_mm256_permutevar_ps",
                {
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_permutevar_ps_{unique_id}", 256),
                },
            )

            return [permute_ps, permute4x64, permutexvar, permutevar_ps]

        if self.vm == vector_machine.AVX2 and self.prim_type == primitive_type.i64:
            input_reg = reg_name
            unique_id = id(input_reg)

            # Permute within 128-bit lanes using immediate (for pd/64-bit doubles)
            permute_pd = InstructionSpec(
                "_mm256_permute_pd",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute_pd_{unique_id}", 8),
                },
            )

            # Permute 64-bit elements across full register
            permute4x64 = InstructionSpec(
                "_mm256_permute4x64_epi64",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute4x64_{unique_id}", 8),
                },
            )

            # Variable permute across all lanes (most powerful for 64-bit)
            permutexvar = InstructionSpec(
                "_mm256_permutexvar_epi64",
                {
                    "a": input_reg,
                    "op_idx": SymbolicPlaceholder(f"ctrl_permutexvar_{unique_id}", 256),
                },
            )

            # Variable permute within 128-bit lanes (for pd)
            permutevar_pd = InstructionSpec(
                "_mm256_permutevar_pd",
                {
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_permutevar_pd_{unique_id}", 256),
                },
            )

            return [permute_pd, permute4x64, permutexvar, permutevar_pd]

        if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i64:
            input_reg = reg_name
            unique_id = id(input_reg)

            # --- Unmasked ---
            # Cross-lane variable permute (most powerful)
            permutexvar = InstructionSpec(
                "_mm512_permutexvar_epi64",
                {
                    "a": input_reg,
                    "op_idx": SymbolicPlaceholder(f"ctrl_permutexvar_{unique_id}", 512),
                },
            )
            # In-lane permute with immediate
            permute_pd = InstructionSpec(
                "_mm512_permute_pd",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute_pd_{unique_id}", 8),
                },
            )
            # In-lane variable permute
            permutevar_pd = InstructionSpec(
                "_mm512_permutevar_pd",
                {
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_permutevar_pd_{unique_id}", 512),
                },
            )

            # --- Masked ---
            mask_permutexvar = InstructionSpec(
                "_mm512_mask_permutexvar_epi64",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permutexvar_{unique_id}", 8),
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_m_permutexvar_{unique_id}", 512
                    ),
                    "a": input_reg,
                },
            )
            mask_permute_pd = InstructionSpec(
                "_mm512_mask_permute_pd",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permute_pd_{unique_id}", 8),
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_m_permute_pd_{unique_id}", 8),
                },
            )
            mask_permutevar_pd = InstructionSpec(
                "_mm512_mask_permutevar_pd",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permutevar_pd_{unique_id}", 8),
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_m_permutevar_pd_{unique_id}", 512),
                },
            )

            return [
                permutexvar,
                permute_pd,
                permutevar_pd,
                mask_permutexvar,
                mask_permute_pd,
                mask_permutevar_pd,
            ]

        if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i32:
            input_reg = reg_name
            unique_id = id(input_reg)

            # --- Unmasked ---
            permutexvar = InstructionSpec(
                "_mm512_permutexvar_epi32",
                {
                    "a": input_reg,
                    "op_idx": SymbolicPlaceholder(f"ctrl_permutexvar_{unique_id}", 512),
                },
            )
            permute_ps = InstructionSpec(
                "_mm512_permute_ps",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute_ps_{unique_id}", 8),
                },
            )
            permutevar_ps = InstructionSpec(
                "_mm512_permutevar_ps",
                {
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_permutevar_ps_{unique_id}", 512),
                },
            )

            # --- Masked (16-bit k-mask for 16 elements) ---
            mask_permutexvar = InstructionSpec(
                "_mm512_mask_permutexvar_epi32",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permutexvar_{unique_id}", 16),
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_m_permutexvar_{unique_id}", 512
                    ),
                    "a": input_reg,
                },
            )
            mask_permute_ps = InstructionSpec(
                "_mm512_mask_permute_ps",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permute_ps_{unique_id}", 16),
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_m_permute_ps_{unique_id}", 8),
                },
            )
            mask_permutevar_ps = InstructionSpec(
                "_mm512_mask_permutevar_ps",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permutevar_ps_{unique_id}", 16),
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_m_permutevar_ps_{unique_id}", 512),
                },
            )

            return [
                permutexvar,
                permute_ps,
                permutevar_ps,
                mask_permutexvar,
                mask_permute_ps,
                mask_permutevar_ps,
            ]

        raise NotImplementedError(
            f"Single-input instructions not implemented for {self.vm} and {self.prim_type}"
        )

    def _enumerate_dual_input_instructions(
        self, reg1_name: str = "top", reg2_name: str = "bottom"
    ) -> list[InstructionSpec]:
        """
        Generate dual-input instruction templates with symbolic immediates.
        Z3 will solve for the concrete immediate values.
        """
        if self.vm == vector_machine.AVX2 and self.prim_type == primitive_type.i32:
            reg1 = reg1_name
            reg2 = reg2_name
            unique_id = f"{id(reg1)}_{id(reg2)}"

            # Shuffle: select elements from both inputs within 128-bit lanes
            shuffle_ps = InstructionSpec(
                "_mm256_shuffle_ps",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuffle_{unique_id}", 8),
                },
            )

            # Unpack low: interleave low elements from both inputs
            unpacklo = InstructionSpec(
                "_mm256_unpacklo_epi32",
                {"a": reg1, "b": reg2},
            )

            # Unpack high: interleave high elements from both inputs
            unpackhi = InstructionSpec(
                "_mm256_unpackhi_epi32",
                {"a": reg1, "b": reg2},
            )

            # Permute 128-bit lanes between two registers
            permute2x128 = InstructionSpec(
                "_mm256_permute2x128_si256",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_perm2x128_{unique_id}", 8),
                },
            )

            # Blend: select elements from either input based on mask
            blend_ps = InstructionSpec(
                "_mm256_blend_ps",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_blend_{unique_id}", 8),
                },
            )

            # Align right: concatenate and shift
            alignr = InstructionSpec(
                "_mm256_alignr_epi32",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_alignr_{unique_id}", 8),
                },
            )

            return [shuffle_ps, unpacklo, unpackhi, permute2x128, blend_ps, alignr]

        if self.vm == vector_machine.AVX2 and self.prim_type == primitive_type.i64:
            reg1 = reg1_name
            reg2 = reg2_name
            unique_id = f"{id(reg1)}_{id(reg2)}"

            # Shuffle: select 64-bit elements from both inputs within 128-bit lanes (pd variant)
            shuffle_pd = InstructionSpec(
                "_mm256_shuffle_pd",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuffle_{unique_id}", 8),
                },
            )

            # Unpack low: interleave low 64-bit elements from both inputs
            unpacklo = InstructionSpec(
                "_mm256_unpacklo_epi64",
                {"a": reg1, "b": reg2},
            )

            # Unpack high: interleave high 64-bit elements from both inputs
            unpackhi = InstructionSpec(
                "_mm256_unpackhi_epi64",
                {"a": reg1, "b": reg2},
            )

            # Permute 128-bit lanes between two registers (works for any element size)
            permute2x128 = InstructionSpec(
                "_mm256_permute2x128_si256",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_perm2x128_{unique_id}", 8),
                },
            )

            # Blend: select 64-bit elements from either input based on mask (pd variant)
            blend_pd = InstructionSpec(
                "_mm256_blend_pd",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_blend_{unique_id}", 8),
                },
            )

            # Align right: concatenate and shift by 64-bit elements
            alignr = InstructionSpec(
                "_mm256_alignr_epi64",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_alignr_{unique_id}", 8),
                },
            )

            return [shuffle_pd, unpacklo, unpackhi, permute2x128, blend_pd, alignr]

        if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i64:
            reg1 = reg1_name
            reg2 = reg2_name
            unique_id = f"{id(reg1)}_{id(reg2)}"

            # --- Unmasked ---
            # Two-source variable permute (most powerful AVX512 instruction)
            permutex2var = InstructionSpec(
                "_mm512_permutex2var_epi64",
                {
                    "a": reg1,
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_permutex2var_{unique_id}", 512
                    ),
                    "b": reg2,
                },
            )
            # Shuffle pd within 128-bit lanes
            shuffle_pd = InstructionSpec(
                "_mm512_shuffle_pd",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuffle_pd_{unique_id}", 8),
                },
            )
            # Unpack low/high
            unpacklo = InstructionSpec("_mm512_unpacklo_epi64", {"a": reg1, "b": reg2})
            unpackhi = InstructionSpec("_mm512_unpackhi_epi64", {"a": reg1, "b": reg2})
            # 128-bit lane shuffle
            shuffle_i32x4 = InstructionSpec(
                "_mm512_shuffle_i32x4",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuf_i32x4_{unique_id}", 8),
                },
            )
            # Align right
            alignr = InstructionSpec(
                "_mm512_alignr_epi64",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_alignr_{unique_id}", 8),
                },
            )

            # --- Masked ---
            mask_permutex2var = InstructionSpec(
                "_mm512_mask_permutex2var_epi64",
                {
                    "a": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_permutex2var_{unique_id}", 8),
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_m_permutex2var_{unique_id}", 512
                    ),
                    "b": reg2,
                },
            )
            mask_shuffle_pd = InstructionSpec(
                "_mm512_mask_shuffle_pd",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_shuffle_pd_{unique_id}", 8),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_shuffle_pd_{unique_id}", 8),
                },
            )
            mask_unpacklo = InstructionSpec(
                "_mm512_mask_unpacklo_epi64",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_unpacklo_{unique_id}", 8),
                    "a": reg1,
                    "b": reg2,
                },
            )
            mask_unpackhi = InstructionSpec(
                "_mm512_mask_unpackhi_epi64",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_unpackhi_{unique_id}", 8),
                    "a": reg1,
                    "b": reg2,
                },
            )
            mask_shuffle_i32x4 = InstructionSpec(
                "_mm512_mask_shuffle_i32x4",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_shuf_i32x4_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_shuf_i32x4_{unique_id}", 8),
                },
            )
            mask_alignr = InstructionSpec(
                "_mm512_mask_alignr_epi64",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_alignr_{unique_id}", 8),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_alignr_{unique_id}", 8),
                },
            )

            return [
                permutex2var,
                shuffle_pd,
                unpacklo,
                unpackhi,
                shuffle_i32x4,
                alignr,
                mask_permutex2var,
                mask_shuffle_pd,
                mask_unpacklo,
                mask_unpackhi,
                mask_shuffle_i32x4,
                mask_alignr,
            ]

        if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i32:
            reg1 = reg1_name
            reg2 = reg2_name
            unique_id = f"{id(reg1)}_{id(reg2)}"

            # --- Unmasked ---
            permutex2var = InstructionSpec(
                "_mm512_permutex2var_epi32",
                {
                    "a": reg1,
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_permutex2var_{unique_id}", 512
                    ),
                    "b": reg2,
                },
            )
            shuffle_ps = InstructionSpec(
                "_mm512_shuffle_ps",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuffle_ps_{unique_id}", 8),
                },
            )
            unpacklo = InstructionSpec("_mm512_unpacklo_epi32", {"a": reg1, "b": reg2})
            unpackhi = InstructionSpec("_mm512_unpackhi_epi32", {"a": reg1, "b": reg2})
            shuffle_i32x4 = InstructionSpec(
                "_mm512_shuffle_i32x4",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuf_i32x4_{unique_id}", 8),
                },
            )
            alignr = InstructionSpec(
                "_mm512_alignr_epi32",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_alignr_{unique_id}", 8),
                },
            )

            # --- Masked (16-bit k-mask for 16 elements) ---
            mask_permutex2var = InstructionSpec(
                "_mm512_mask_permutex2var_epi32",
                {
                    "a": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_permutex2var_{unique_id}", 16),
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_m_permutex2var_{unique_id}", 512
                    ),
                    "b": reg2,
                },
            )
            mask_shuffle_ps = InstructionSpec(
                "_mm512_mask_shuffle_ps",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_shuffle_ps_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_shuffle_ps_{unique_id}", 8),
                },
            )
            mask_unpacklo = InstructionSpec(
                "_mm512_mask_unpacklo_epi32",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_unpacklo_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                },
            )
            mask_unpackhi = InstructionSpec(
                "_mm512_mask_unpackhi_epi32",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_unpackhi_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                },
            )
            mask_shuffle_i32x4 = InstructionSpec(
                "_mm512_mask_shuffle_i32x4",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_shuf_i32x4_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_shuf_i32x4_{unique_id}", 8),
                },
            )
            mask_alignr = InstructionSpec(
                "_mm512_mask_alignr_epi32",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_alignr_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_alignr_{unique_id}", 8),
                },
            )

            return [
                permutex2var,
                shuffle_ps,
                unpacklo,
                unpackhi,
                shuffle_i32x4,
                alignr,
                mask_permutex2var,
                mask_shuffle_ps,
                mask_unpacklo,
                mask_unpackhi,
                mask_shuffle_i32x4,
                mask_alignr,
            ]

        raise NotImplementedError(
            f"Dual-input instructions not implemented for {self.vm} and {self.prim_type}"
        )


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

        Returns:
            Tuple of (root nodes, all_stages_complete) where all_stages_complete
            is True if every expected stage was solved successfully.
        """
        if self._synthesis_started:
            raise RuntimeError(
                "BitonicSuperVectorizer instances are single-use. "
                "Create a new instance for each synthesis run."
            )
        self._synthesis_started = True

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
        # Pre-compute all candidates once - they're independent of stage/input state
        all_candidates = self.synthesizer.precompute_all_candidates(gadget_depth)

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
        nodes_by_path = self._build_tree_iterative(
            input_states_with_context,
            depth_limit,
            all_candidates,
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

    def _process_single_stage(
        self,
        input_states_with_context: list[tuple[VectorState, tuple]],
        stage_idx: int,
        all_candidates: list[tuple],
        max_unique_outputs: int = 3,
    ) -> tuple[dict[tuple, list[SolutionNode]], dict[tuple, VectorState]]:
        """Process one stage: validate gadgets, group, create nodes.

        Includes Phase 1 (collect candidates), Phase 2 (validate in parallel),
        Phase 3 (group by transition), and Phase 4 (create nodes, track unique
        outputs). Children are NOT wired by this method.

        Args:
            input_states_with_context: List of (input_state, parent_path) tuples
                where parent_path tracks the chain of previous gadgets leading
                to this state.
            stage_idx: Current stage index.
            all_candidates: Pre-computed list of (top_seq, bottom_seq) instruction
                templates, independent of stage/input state.
            max_unique_outputs: Number of smallest unique output states to
                enumerate per template for deterministic diversity.

        Returns:
            Tuple of (nodes_by_parent, unique_outputs) where nodes_by_parent maps
            parent_path to list of SolutionNodes and unique_outputs maps output
            state tuples to VectorState instances. Children on the nodes are empty.
        """
        stage_pairs = self.bitonic_sorter.stages[stage_idx]

        print(
            f"Stage {stage_idx}: Collecting candidates for {len(input_states_with_context)} nodes from the previous stage"
        )

        # Phase 1: Collect all candidate gadgets for entire stage
        all_jobs = []

        for input_state, parent_path in input_states_with_context:
            metadata = {
                "input_state": input_state,
                "parent_path": parent_path,
                "stage_idx": stage_idx,
                "max_unique_outputs": max_unique_outputs,
            }
            if (
                self._natural_order_stage is not None
                and stage_idx == self._natural_order_stage
            ):
                metadata["allow_any_lane_order"] = False
            if self.smt2_dump_dir:
                metadata["smt2_dump_dir"] = self.smt2_dump_dir

            # Enrich pre-computed candidates with per-state metadata
            for top_seq, bottom_seq in all_candidates:
                all_jobs.append(
                    (
                        top_seq,
                        bottom_seq,
                        input_state,
                        stage_pairs,
                        self.vm,
                        self.prim_type,
                        metadata,
                    )
                )

        print(f"Stage {stage_idx}: Generated {len(all_jobs)} candidates to validate")

        # Phase 2: Validate all candidates in parallel with progress reporting
        validated_gadgets = self.synthesizer._validate_gadgets(
            all_jobs, show_progress=True
        )

        print(
            f"Stage {stage_idx}: Validated {len(validated_gadgets)}/{len(all_jobs)} gadgets"
        )

        # Phase 3: Group gadgets by (parent_path, input_state, output_state)
        # This prunes semantically equivalent gadgets - we only need one path per unique state transition
        # Key: (parent_path, input_state_tuple, output_state_tuple) -> list of gadgets
        gadgets_by_transition: dict[tuple, list[PermutationGadget]] = {}

        # Collect all validated gadgets
        for gadget, input_state, output_state, metadata in validated_gadgets:
            parent_path = metadata["parent_path"]
            key = (parent_path, input_state.as_tuple(), output_state.as_tuple())
            if key not in gadgets_by_transition:
                gadgets_by_transition[key] = []
            gadgets_by_transition[key].append(gadget)

        print(
            f"Stage {stage_idx}: Pruned to {len(gadgets_by_transition)} unique state transitions"
        )

        # Post-validation statistics: analyze output_state sharing across input_states
        output_to_inputs: dict[tuple, set[tuple]] = {}
        for _parent_path, input_tuple, output_tuple in gadgets_by_transition.keys():
            if output_tuple not in output_to_inputs:
                output_to_inputs[output_tuple] = set()
            output_to_inputs[output_tuple].add(input_tuple)

        # Phase 4: Create nodes from grouped gadgets
        nodes_by_parent: dict[tuple, list[SolutionNode]] = {}
        # Deduplicate: track one representative per unique output_state
        unique_outputs: dict[tuple, VectorState] = {}

        # Sort transition keys for deterministic iteration order
        sorted_transitions = sorted(gadgets_by_transition.items(), key=lambda x: x[0])

        for (
            parent_path,
            input_tuple,
            output_tuple,
        ), gadgets in sorted_transitions:
            # Reconstruct VectorState from tuples
            input_state = VectorState(
                top=list(input_tuple[0]), bottom=list(input_tuple[1])
            )
            output_state = VectorState(
                top=list(output_tuple[0]), bottom=list(output_tuple[1])
            )

            # Sort gadgets within each group for deterministic ordering
            gadgets.sort(key=lambda g: g.sort_key())

            # Create node with all equivalent gadgets
            node = SolutionNode(
                stage=stage_idx,
                input_state=input_state,
                output_state=output_state,
                gadgets=gadgets,
                children=[],
            )

            # Add to nodes_by_parent
            if parent_path not in nodes_by_parent:
                nodes_by_parent[parent_path] = []
            nodes_by_parent[parent_path].append(node)

            # Track unique output_states for deduplication
            if output_tuple not in unique_outputs:
                unique_outputs[output_tuple] = output_state

        return nodes_by_parent, unique_outputs

    def _build_tree_iterative(
        self,
        initial_inputs: list[tuple[VectorState, tuple]],
        depth_limit: int | None,
        all_candidates: list[tuple],
        max_unique_outputs: int = 3,
        checkpoint_dir: str | None = None,
        checkpoint_config=None,
        resume_data: dict | None = None,
    ) -> dict[tuple, list[SolutionNode]]:
        """Build the solution tree iteratively using a forward pass then backward wiring.

        The forward pass iterates through stages sequentially, processing each
        stage with ``_process_single_stage``. The backward pass wires children
        from later stages back into earlier stage nodes.

        Args:
            initial_inputs: List of (input_state, parent_path) tuples for stage 0.
            depth_limit: Maximum stage depth to explore. If None, all stages.
            all_candidates: Pre-computed instruction templates.
            max_unique_outputs: Diversity parameter per template.
            checkpoint_dir: Directory to save checkpoints after each stage.
            checkpoint_config: Configuration object for checkpoint saving.
            resume_data: If provided, resume from a previous checkpoint. Expected
                keys: ``per_stage_data`` and ``last_completed_stage``.

        Returns:
            Dictionary mapping parent_path to list of SolutionNodes for stage 0.
        """
        per_stage_data = {}  # stage_idx -> (nodes_by_parent, unique_outputs)

        # Determine effective limit
        effective_limit = len(self.bitonic_sorter.stages)
        if depth_limit is not None:
            effective_limit = min(depth_limit, effective_limit)

        # Handle resume: skip already-completed stages
        start_stage = 0
        current_inputs = initial_inputs
        if resume_data is not None:
            per_stage_data = resume_data["per_stage_data"]
            start_stage = resume_data["last_completed_stage"] + 1
            # Reconstruct current_inputs from last completed stage
            last_stage = resume_data["last_completed_stage"]
            _, unique_outputs = per_stage_data[last_stage]
            current_inputs = [
                (state, ("canonical", out_tuple))
                for out_tuple, state in unique_outputs.items()
            ]
            print(
                f"Resuming from stage {start_stage} with {len(current_inputs)} inputs"
            )

        # Forward pass: iterate through stages
        for stage_idx in range(start_stage, effective_limit):
            nodes_by_parent, unique_outputs = self._process_single_stage(
                current_inputs, stage_idx, all_candidates, max_unique_outputs
            )
            per_stage_data[stage_idx] = (nodes_by_parent, unique_outputs)

            if nodes_by_parent:
                self._stages_completed.add(stage_idx)

            # Checkpoint save point
            if checkpoint_dir is not None and checkpoint_config is not None:
                try:
                    from checkpoint import save_checkpoint, checkpoint_filename
                except ImportError:
                    from .checkpoint import save_checkpoint, checkpoint_filename  # type: ignore[no-redef]
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    checkpoint_filename(
                        checkpoint_config.num_vecs,
                        checkpoint_config.vm,
                        checkpoint_config.prim_type,
                        checkpoint_config.natural_order,
                        stage_idx,
                    ),
                )
                save_checkpoint(
                    ckpt_path,
                    checkpoint_config,
                    per_stage_data,
                    sorted(self._stages_completed),
                    stage_idx,
                )

            # Log deduplication and prepare inputs for next stage
            if unique_outputs:
                total_next = sum(len(nodes) for nodes in nodes_by_parent.values())
                print(
                    f"Stage {stage_idx}: Deduplicated {total_next} -> {len(unique_outputs)} next-stage inputs"
                )
                current_inputs = [
                    (state, ("canonical", out_tuple))
                    for out_tuple, state in unique_outputs.items()
                ]
            else:
                break  # No valid outputs, can't continue

        # Backward pass: wire children by iterating stages in reverse
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

        Returns:
            Tuple of (root nodes, all_stages_complete) where all_stages_complete
            is True if every expected stage was solved successfully.
        """
        return self.build_solution_tree(
            depth_limit=depth_limit,
            gadget_depth=gadget_depth,
            natural_order=natural_order,
            max_unique_outputs=max_unique_outputs,
            checkpoint_dir=checkpoint_dir,
            resume_data=resume_data,
        )


_worker_tar = None
_worker_job_count = 0


def _validate_gadget_worker(job):
    """Worker function for parallel gadget validation.

    Returns (gadget_results, input_state, metadata, construction_time, solver_time)
    where gadget_results is a list of (gadget, output_state) tuples.
    """
    global _worker_tar, _worker_job_count
    top_seq, bottom_seq, input_state, target_pairs, vm, prim_type, metadata = job

    # Create clones of sequences to avoid modifying the ones in the main process
    top_seq_clone = copy.deepcopy(top_seq)
    bottom_seq_clone = copy.deepcopy(bottom_seq)

    # Create a local synthesizer and synthesis in its own Z3 context
    synthesizer = GadgetSynthesizer(vm, prim_type)

    smt2_dump_dir = metadata.get("smt2_dump_dir")
    stage_idx = metadata.get("stage_idx")

    def dump_smt2_to_tar(solver):
        global _worker_tar, _worker_job_count
        if smt2_dump_dir is None:
            return

        if _worker_tar is None:
            pid = os.getpid()
            # Write to an uncompressed tar file first; we'll compress it in the main process
            tar_filename = f"stage{stage_idx}_pid_{pid}.tar"
            tar_path = os.path.join(smt2_dump_dir, tar_filename)
            _worker_tar = tarfile.open(tar_path, mode="a")

        _worker_job_count += 1
        smt2_text = "(reset)\n" + solver.sexpr() + "\n(check-sat)\n"
        smt2_bytes = smt2_text.encode("utf-8")

        tar_info = tarfile.TarInfo(name=f"job_{_worker_job_count}.smt2")
        tar_info.size = len(smt2_bytes)
        _worker_tar.addfile(tar_info, io.BytesIO(smt2_bytes))
        # Ensure it's written to disk
        _worker_tar.fileobj.flush()

    allow_any_lane_order = metadata.get("allow_any_lane_order", True)
    max_unique_outputs = metadata.get("max_unique_outputs", 3)

    gadget_results, construction_time, solver_time = (
        synthesizer.synthesize_gadget_with_symbolic(
            top_seq_clone,
            bottom_seq_clone,
            input_state,
            target_pairs,
            max_solutions=1,
            max_unique_outputs=max_unique_outputs,
            solver_callback=dump_smt2_to_tar,
            allow_any_lane_order=allow_any_lane_order,
        )
    )

    metadata["worker_pid"] = os.getpid()
    return gadget_results, input_state, metadata, construction_time, solver_time
