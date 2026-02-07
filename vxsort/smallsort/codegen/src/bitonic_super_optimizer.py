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
    Context,
    main_ctx,
    Extract,
    BitVecVal,
    sat,
    BitVec,
    Distinct,
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
    cost: float = 0.0

    def __repr__(self):
        return f"SolutionNode(stage={self.stage}, gadgets={len(self.gadgets)}, cost={self.cost}, children={len(self.children)})"

    def best_gadget(self) -> PermutationGadget:
        """Return the gadget with fewest instructions."""
        return min(self.gadgets, key=lambda g: g.instruction_count())


class GadgetSynthesizer:
    """Synthesizes permutation gadgets using Z3."""

    def __init__(self, vm: vector_machine, prim_type: primitive_type):
        self.vm = vm
        self.prim_type = prim_type
        self.elements_per_vector = width_dict[vm] // int(prim_type.value[0])
        self.lane_width = int(prim_type.value[0]) * 8  # 16, 32, or 64 bits per element
        self.available_intrinsics = self._get_available_intrinsics()

        # Memoize instruction templates - these are reused across all gadget generation
        self.single_insts_top = self._enumerate_single_input_instructions("top")
        self.single_insts_bottom = self._enumerate_single_input_instructions("bottom")
        self.dual_insts_top_bottom = self._enumerate_dual_input_instructions(
            "top", "bottom"
        )
        self.dual_insts_bottom_top = self._enumerate_dual_input_instructions(
            "bottom", "top"
        )

    def _get_available_intrinsics(self) -> dict[str, callable]:
        """Get available intrinsics for the current VM and primitive type."""
        intrinsics = {}

        # For AVX2 i32, we need YMM (256-bit) operations on 32-bit elements
        if self.vm == vector_machine.AVX2 and self.prim_type == primitive_type.i32:
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

        return intrinsics

    def _create_pair_id_mapping(
        self, target_pairs: list[tuple[int, int]]
    ) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
        """
        Create mappings for pair IDs.

        Returns:
            pair_id_map: element index -> pair_id (both elements in a pair map to same ID)
            pair_id_reverse_map: pair_id -> (low_elem, high_elem) in canonical order
        """
        pair_id_map = {}
        pair_id_reverse_map = {}
        for pair_id, (elem1, elem2) in enumerate(target_pairs, start=1):
            pair_id_map[elem1] = pair_id
            pair_id_map[elem2] = pair_id
            # Canonical ordering: lower index in top, higher in bottom
            pair_id_reverse_map[pair_id] = (min(elem1, elem2), max(elem1, elem2))
        return pair_id_map, pair_id_reverse_map

    def _create_input_registers_with_pair_ids(
        self,
        solver: Solver,
        ctx: Context,
        input_state: VectorState,
        pair_id_map: dict[int, int],
    ) -> tuple:
        """
        Create Z3 symbolic registers with pair IDs as values.
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

        # Set up constraints: each lane should have the pair_id of the element at that position
        for lane_idx in range(self.elements_per_vector):
            top_elem = input_state.top[lane_idx]
            bottom_elem = input_state.bottom[lane_idx]

            top_pair_id = pair_id_map.get(top_elem, 0)
            bottom_pair_id = pair_id_map.get(bottom_elem, 0)

            # Extract the lane from the register and constrain it to the pair_id
            lane_start = lane_idx * self.lane_width
            lane_end = lane_start + self.lane_width - 1

            top_lane = Extract(lane_end, lane_start, top_reg)
            bottom_lane = Extract(lane_end, lane_start, bottom_reg)

            solver.add(top_lane == BitVecVal(top_pair_id, self.lane_width, ctx=ctx))
            solver.add(
                bottom_lane == BitVecVal(bottom_pair_id, self.lane_width, ctx=ctx)
            )

        return top_reg, bottom_reg

    def synthesize_gadget_with_symbolic(
        self,
        top_instructions_template: list[InstructionSpec],
        bottom_instructions_template: list[InstructionSpec],
        input_state: VectorState,
        target_pairs: list[tuple[int, int]],
        max_solutions: int | None = None,
        solver_callback: callable | None = None,
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
            max_solutions: Optional cap on the number of solutions returned.
                When ``None`` (the default) all solutions are enumerated.
            solver_callback: Optional callback receiving the Solver instance.
        """
        start_construction = time.perf_counter()
        ctx = main_ctx()
        solver = Solver(ctx=ctx)

        # Create pair_id mappings:
        # - pair_id_map: element index -> pair_id
        # - pair_id_reverse_map: pair_id -> (low_elem, high_elem) in canonical order
        pair_id_map, pair_id_reverse_map = self._create_pair_id_mapping(target_pairs)

        # Create input registers with pair IDs
        top_reg, bottom_reg = self._create_input_registers_with_pair_ids(
            solver, ctx, input_state, pair_id_map
        )

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
        top_output = self._apply_instructions(
            top_reg,
            bottom_reg,
            top_instructions_template,
            is_top=True,
            solver=solver,
            symbolic_vars=None,
        )
        bottom_output = self._apply_instructions(
            top_reg,
            bottom_reg,
            bottom_instructions_template,
            is_top=False,
            solver=solver,
            symbolic_vars=None,
        )

        # Add constraints: for each lane, top_output[lane] == bottom_output[lane] (same pair_id)
        output_lanes = []
        for lane_idx in range(self.elements_per_vector):
            lane_start = lane_idx * self.lane_width
            lane_end = lane_start + self.lane_width - 1

            top_lane = Extract(lane_end, lane_start, top_output)
            bottom_lane = Extract(lane_end, lane_start, bottom_output)

            solver.add(top_lane == bottom_lane)
            output_lanes.append(top_lane)

        # Add constraint: all lanes must have distinct pair_ids (no duplicate pairs)
        # Without this, the solver could find degenerate solutions where multiple lanes
        # have the same pair_id, losing elements in the process.
        solver.add(Distinct(*output_lanes))

        if solver_callback:
            solver_callback(solver)

        # Collect symbolic variable terms for enumeration
        terms = list(symbolic_vars.values())
        construction_time = time.perf_counter() - start_construction
        solver_start = time.perf_counter()
        # If no symbolic variables, single check suffices
        if not terms:
            result = solver.check()
            if result != sat:
                solver_time = time.perf_counter() - solver_start
                return [], construction_time, solver_time
            model = solver.model()
            gadget, output_state = self._extract_solution_from_model(
                model,
                top_output,
                bottom_output,
                pair_id_reverse_map,
                symbolic_vars,
                top_instructions_template,
                bottom_instructions_template,
            )
            solver_time = time.perf_counter() - solver_start
            return [(gadget, output_state)], construction_time, solver_time

        # Enumerate all solutions over symbolic variables
        results = []
        for model in _all_smt(solver, terms, max_results=max_solutions):
            gadget, output_state = self._extract_solution_from_model(
                model,
                top_output,
                bottom_output,
                pair_id_reverse_map,
                symbolic_vars,
                top_instructions_template,
                bottom_instructions_template,
            )
            results.append((gadget, output_state))
        solver_time = time.perf_counter() - solver_start
        return results, construction_time, solver_time

    def _extract_solution_from_model(
        self,
        model,
        top_output,
        bottom_output,
        pair_id_reverse_map: dict[int, tuple[int, int]],
        symbolic_vars: dict,
        top_instructions_template: list[InstructionSpec],
        bottom_instructions_template: list[InstructionSpec],
    ) -> tuple[PermutationGadget, VectorState]:
        """Extract a concrete gadget and output state from a Z3 model.

        Reads pair IDs from the symbolic output registers, builds the
        canonical output state, and concretizes all symbolic variables
        in the instruction templates using the model's assignments.
        """
        # Extract pair ordering from output registers to build canonical output state
        output_top = []
        output_bottom = []
        for lane_idx in range(self.elements_per_vector):
            lane_start = lane_idx * self.lane_width
            lane_end = lane_start + self.lane_width - 1

            top_lane = Extract(lane_end, lane_start, top_output)
            pair_id_val = model.evaluate(top_lane, model_completion=True)
            pair_id = (
                pair_id_val.as_long()
                if hasattr(pair_id_val, "as_long")
                else pair_id_val
            )

            if pair_id not in pair_id_reverse_map:
                raise ValueError(
                    f"Invalid pair_id {pair_id} at lane {lane_idx} not found in reverse map"
                )
            low_elem, high_elem = pair_id_reverse_map[pair_id]
            output_top.append(low_elem)
            output_bottom.append(high_elem)

        output_state = VectorState(top=output_top, bottom=output_bottom)

        # Create concrete instructions by substituting symbolic values
        def concretize_instructions(
            instructions: list[InstructionSpec],
        ) -> list[InstructionSpec]:
            concrete_insts = []
            for inst in instructions:
                concrete_args = {}
                for key, value in inst.args.items():
                    if id(value) in symbolic_vars:
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

        # Apply gadget instructions
        top_output = self._apply_instructions(
            top_reg, bottom_reg, gadget.top_instructions, is_top=True
        )
        bottom_output = self._apply_instructions(
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

        Args:
            top_reg: Input top register
            bottom_reg: Input bottom register
            instructions: List of instructions to apply
            is_top: True if computing top output, False for bottom output
            solver: Optional Z3 solver (unused but kept for API compatibility)
            symbolic_vars: Optional dict to track symbolic variables

        Returns:
            Output register after applying instructions
        """
        # Start with the appropriate input register
        current_reg = top_reg if is_top else bottom_reg

        for inst in instructions:
            intrinsic = self.available_intrinsics.get(inst.intrinsic_name)
            if intrinsic is None:
                raise ValueError(f"Unknown intrinsic: {inst.intrinsic_name}")

            # Substitute register names in arguments with actual Z3 registers
            args = {}
            for key, value in inst.args.items():
                args[key] = self._substitute_register_names(
                    value, top_reg, bottom_reg, current_reg, symbolic_vars, key=key
                )

            # Determine how to call the intrinsic based on its signature
            if "a" in args and "op_idx" in args:
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

        return current_reg

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
                # Try pairs: (single, single) and (dual, single)
                for inst1 in self.single_insts_top:
                    for inst2 in self.single_insts_top:
                        top_sequences.append([inst1, inst2])
                for inst1 in self.dual_insts_top_bottom:
                    for inst2 in self.single_insts_top:
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
        for top_depth in range(gadget_depth):
            for bottom_depth in range(gadget_depth):
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
        assert len(top) == self.elements_per_vector, (
            f"Expected {self.elements_per_vector} elements in top, got {len(top)}"
        )
        assert len(bottom) == self.elements_per_vector, (
            f"Expected {self.elements_per_vector} elements in bottom, got {len(bottom)}"
        )

        return VectorState(top=top, bottom=bottom)

    def build_solution_tree(
        self,
        depth_limit: int | None = None,
        gadget_depth: int = 3,
        max_solutions_per_gadget: int | None = None,
    ) -> list[SolutionNode]:
        """
        Recursively explore all stage transitions to build solution tree.
        Returns root nodes (first stage solutions).

        Args:
            depth_limit: Maximum stage depth to explore (inclusive). If None, all stages are explored.
            gadget_depth: Maximum instruction depth per gadget (1-3, default 3).
            max_solutions_per_gadget: Optional cap on solutions per gadget template.
                When ``None`` all solutions are enumerated.
        """
        initial_state = self._create_initial_state()
        # Pre-compute all candidates once - they're independent of stage/input state
        all_candidates = self.synthesizer.precompute_all_candidates(gadget_depth)
        # Start with empty parent path for the root
        input_states_with_context = [(initial_state, ())]
        nodes_by_path = self._build_tree_recursive(
            input_states_with_context,
            0,
            depth_limit,
            all_candidates,
            max_solutions_per_gadget=max_solutions_per_gadget,
        )

        # Return root nodes (those with empty parent path)
        return nodes_by_path.get((), [])

    def _build_tree_recursive(
        self,
        input_states_with_context: list[tuple[VectorState, list]],
        stage_idx: int,
        depth_limit: int | None = None,
        all_candidates: list[tuple] | None = None,
        max_solutions_per_gadget: int | None = None,
    ) -> dict[tuple, list[SolutionNode]]:
        """
        Build solution tree collecting all candidates for entire stage before validation.

        Args:
            input_states_with_context: List of (input_state, parent_path) tuples where parent_path
                                      tracks the chain of previous gadgets leading to this state
            stage_idx: Current stage index
            depth_limit: Maximum stage depth to explore
            all_candidates: Pre-computed list of (top_seq, bottom_seq) instruction templates,
                           independent of stage/input state
            max_solutions_per_gadget: Optional cap forwarded to each worker.

        Returns:
            Dictionary mapping parent_path to list of SolutionNodes for that path
        """
        if stage_idx >= len(self.bitonic_sorter.stages) or (
            depth_limit is not None and stage_idx >= depth_limit
        ):
            # No more stages or reached depth limit, return empty dict
            print(f"Reached depth limit {stage_idx}")
            return {}

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
            }
            if self.smt2_dump_dir:
                metadata["smt2_dump_dir"] = self.smt2_dump_dir

            if max_solutions_per_gadget is not None:
                metadata["max_solutions"] = max_solutions_per_gadget

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

        # Early bailout: Check if next stage would exceed depth limit or total stages
        # No point building nodes if there are no more stages to process
        next_stage_idx = stage_idx + 1
        has_next_stage = next_stage_idx < len(self.bitonic_sorter.stages) and (
            depth_limit is None or next_stage_idx < depth_limit
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

        # Phase 4: Create nodes from grouped gadgets
        nodes_by_parent: dict[tuple, list[SolutionNode]] = {}
        next_stage_inputs = [] if has_next_stage else None

        for (
            parent_path,
            input_tuple,
            output_tuple,
        ), gadgets in gadgets_by_transition.items():
            # Reconstruct VectorState from tuples
            input_state = VectorState(
                top=list(input_tuple[0]), bottom=list(input_tuple[1])
            )
            output_state = VectorState(
                top=list(output_tuple[0]), bottom=list(output_tuple[1])
            )

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

            # Prepare for next stage (only if there is one)
            # Only one entry per unique output_state per parent_path
            if has_next_stage:
                new_path = parent_path + (id(node),)
                next_stage_inputs.append((output_state, new_path))

        # Phase 4: Recursively process next stage
        if has_next_stage and next_stage_inputs:
            children_by_path = self._build_tree_recursive(
                next_stage_inputs,
                stage_idx + 1,
                depth_limit,
                all_candidates,
                max_solutions_per_gadget=max_solutions_per_gadget,
            )

            # Attach children to their parent nodes
            for parent_path, nodes in nodes_by_parent.items():
                for node in nodes:
                    node_path = parent_path + (id(node),)
                    if node_path in children_by_path:
                        node.children = children_by_path[node_path]

        return nodes_by_parent

    def synthesize_all_stages(
        self,
        depth_limit: int | None = None,
        gadget_depth: int = 3,
        max_solutions_per_gadget: int | None = None,
    ) -> list[SolutionNode]:
        """Entry point: builds solution tree for all stages.

        Args:
            depth_limit: Maximum stage depth to explore (inclusive). If None, all stages are explored.
            gadget_depth: Maximum instruction depth per gadget (1-3, default 3).
            max_solutions_per_gadget: Optional cap on solutions per gadget template.
        """
        return self.build_solution_tree(
            depth_limit=depth_limit,
            gadget_depth=gadget_depth,
            max_solutions_per_gadget=max_solutions_per_gadget,
        )

    def compute_costs(self, roots: list[SolutionNode], cost_model):
        """Traverse tree and compute cumulative costs for each path."""
        for root in roots:
            self._compute_costs_recursive(root, cost_model)

    def _compute_costs_recursive(self, node: SolutionNode, cost_model):
        """Recursively compute costs for node and its children.

        Uses the best (lowest instruction count) gadget for cost calculation.
        """
        node.cost = cost_model.calculate_gadget_cost(node.best_gadget())
        for child in node.children:
            self._compute_costs_recursive(child, cost_model)
            # Add parent cost to child for cumulative cost
            child.cost += node.cost

    def export_solutions_to_json(self, roots: list[SolutionNode], output_path: str):
        """Generate JSON with all solutions and costs."""
        import json

        def gadget_to_dict(gadget: PermutationGadget) -> dict:
            return {
                "top_instructions": [
                    {"name": inst.intrinsic_name, "args": inst.args}
                    for inst in gadget.top_instructions
                ],
                "bottom_instructions": [
                    {"name": inst.intrinsic_name, "args": inst.args}
                    for inst in gadget.bottom_instructions
                ],
                "instruction_count": gadget.instruction_count(),
            }

        def node_to_dict(node: SolutionNode) -> dict:
            best = node.best_gadget()
            return {
                "stage": node.stage,
                "input_state": {
                    "top": node.input_state.top,
                    "bottom": node.input_state.bottom,
                },
                "output_state": {
                    "top": node.output_state.top,
                    "bottom": node.output_state.bottom,
                },
                "gadget": gadget_to_dict(
                    best
                ),  # Best gadget for backward compatibility
                "gadgets": [
                    gadget_to_dict(g) for g in node.gadgets
                ],  # All equivalent gadgets
                "gadget_count": len(node.gadgets),
                "cost": node.cost,
                "children": [node_to_dict(child) for child in node.children],
            }

        solutions = [node_to_dict(root) for root in roots]

        with open(output_path, "w") as f:
            json.dump(solutions, f, indent=2)

        print(f"Exported {len(roots)} solution trees to {output_path}")


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
    # Default to 1 for backward compatibility; callers opt in to more via
    # build_solution_tree(max_solutions_per_gadget=N).
    max_solutions = metadata.get("max_solutions", 1)

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

    gadget_results, construction_time, solver_time = (
        synthesizer.synthesize_gadget_with_symbolic(
            top_seq_clone,
            bottom_seq_clone,
            input_state,
            target_pairs,
            max_solutions=max_solutions,
            solver_callback=dump_smt2_to_tar,
        )
    )

    metadata["worker_pid"] = os.getpid()
    return gadget_results, input_state, metadata, construction_time, solver_time
