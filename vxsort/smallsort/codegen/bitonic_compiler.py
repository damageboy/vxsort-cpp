#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import override

from functional import seq
from tabulate import tabulate
from z3 import Solver, BitVecVal, BitVec, Extract, sat

# Handle both relative and absolute imports
try:
    from . import z3_avx
    from .cost_model import CostModel
except ImportError:
    import z3_avx  # type: ignore
    from cost_model import CostModel  # type: ignore


class top_bottom_ind(Enum):
    Top = (0,)
    Bottom = (1,)


class vector_machine(Enum):
    AVX2 = (1,)
    AVX512 = (2,)


class primitive_type(Enum):
    i16 = (2,)
    i32 = (4,)
    i64 = (8,)
    f32 = (4,)
    f64 = 8


width_dict = {
    vector_machine.AVX2: 32,
    vector_machine.AVX512: 64,
}


@dataclass
class InstructionSpec:
    """Represents a single AVX instruction with its arguments."""

    intrinsic_name: str
    args: dict  # operands, immediates, masks, etc.

    def __repr__(self):
        return f"{self.intrinsic_name}({self.args})"


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
        table_str = tabulate(table_data, headers=headers, tablefmt="rounded_outline")
        return f"\nVectorState:\n{table_str}"

    def copy(self):
        return VectorState(top=self.top.copy(), bottom=self.bottom.copy())


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
    """Tree node for one stage's solutions."""

    stage: int
    input_state: VectorState
    output_state: VectorState
    gadget: PermutationGadget
    children: list["SolutionNode"]
    cost: float = 0.0

    def __repr__(self):
        return f"SolutionNode(stage={self.stage}, cost={self.cost}, children={len(self.children)})"


class BitonicStage:
    def __init__(self, stage: int, pairs: list[tuple[int, int]]):
        self.stage = stage
        self.pairs = pairs

    @override
    def __repr__(self):
        return f"S{self.stage}: {self.pairs}"


class BitonicSorter:
    stages: dict[int, list[tuple[int, int]]]

    def __init__(self, n: int):
        self.stages = {}
        _ = self.generate_bitonic_sorter(n)

    # Bitonic sorters are recursive in nature, where we sort both halves of the input
    # and proceed to merge to two halves via a bitonic merge operation.
    def generate_bitonic_sorter(self, n: int, stage: int = 0, i: int = 0) -> int:
        if n == 1:
            return stage

        k = n // 2
        _ = self.generate_bitonic_sorter(k, stage, i)
        stage = self.generate_bitonic_sorter(k, stage, i + k)
        return self.generate_bitonic_merge(n, stage, i, True)

    def generate_bitonic_merge(self, n: int, stage: int, i: int, initial_merge: bool) -> int:
        if n == 1:
            return stage
        k = n // 2

        if initial_merge:
            stage_pairs = seq.range(i, i + k).zip(seq.range(i + k, i + n).reverse()).to_list()
        else:
            stage_pairs = seq.range(i, i + k).map(lambda x: (x, x + k)).to_list()

        self.add_ops(BitonicStage(stage, stage_pairs))

        _ = self.generate_bitonic_merge(k, stage + 1, i, False)
        return self.generate_bitonic_merge(k, stage + 1, i + k, False)

    def add_ops(self, bs: BitonicStage):
        if bs.stage not in self.stages:
            self.stages[bs.stage] = bs.pairs
        else:
            self.stages[bs.stage].extend(bs.pairs)


class GadgetSynthesizer:
    """Synthesizes permutation gadgets using Z3."""

    def __init__(self, vm: vector_machine, prim_type: primitive_type):
        self.vm = vm
        self.prim_type = prim_type
        self.elements_per_vector = width_dict[vm] // int(prim_type.value[0])
        self.available_intrinsics = self._get_available_intrinsics()

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

    def _create_pair_id_mapping(self, target_pairs: list[tuple[int, int]]) -> dict[int, int]:
        """
        Create mapping from element indices to pair IDs.
        Each pair gets a unique ID, and both elements in the pair map to that ID.
        """
        pair_id_map = {}
        for pair_id, (elem1, elem2) in enumerate(target_pairs, start=1):
            pair_id_map[elem1] = pair_id
            pair_id_map[elem2] = pair_id
        return pair_id_map

    def _create_input_registers_with_pair_ids(self, solver: Solver, input_state: VectorState,
                                              pair_id_map: dict[int, int]) -> tuple:
        """
        Create Z3 symbolic registers with pair IDs as values.
        Returns (top_reg, bottom_reg).
        """
        # Create registers based on VM type
        if self.vm == vector_machine.AVX2:
            top_reg = z3_avx.ymm_reg("top_input")
            bottom_reg = z3_avx.ymm_reg("bottom_input")
        else:
            top_reg = z3_avx.zmm_reg("top_input")
            bottom_reg = z3_avx.zmm_reg("bottom_input")

        # Set up constraints: each lane should have the pair_id of the element at that position
        for lane_idx in range(self.elements_per_vector):
            top_elem = input_state.top[lane_idx]
            bottom_elem = input_state.bottom[lane_idx]

            top_pair_id = pair_id_map.get(top_elem, 0)
            bottom_pair_id = pair_id_map.get(bottom_elem, 0)

            # Extract the lane from the register and constrain it to the pair_id
            lane_start = lane_idx * 32  # 32 bits per i32 element
            lane_end = lane_start + 31

            top_lane = Extract(lane_end, lane_start, top_reg)
            bottom_lane = Extract(lane_end, lane_start, bottom_reg)

            solver.add(top_lane == BitVecVal(top_pair_id, 32))
            solver.add(bottom_lane == BitVecVal(bottom_pair_id, 32))

        return top_reg, bottom_reg

    def validate_gadget(self, gadget: PermutationGadget, input_state: VectorState,
                        target_pairs: list[tuple[int, int]]) -> bool:
        """
        Validate that the gadget correctly aligns target pairs using Z3.

        Returns True if the gadget places all pairs on the same lanes.
        """
        solver = Solver()

        # Create pair_id mapping
        pair_id_map = self._create_pair_id_mapping(target_pairs)

        # Create input registers with pair IDs
        top_reg, bottom_reg = self._create_input_registers_with_pair_ids(solver, input_state, pair_id_map)

        # Apply gadget instructions to get output registers
        top_output = self._apply_instructions(top_reg, bottom_reg, gadget.top_instructions, is_top=True, solver=solver)
        bottom_output = self._apply_instructions(top_reg, bottom_reg, gadget.bottom_instructions, is_top=False,
                                                 solver=solver)

        # Add constraints: for each lane, top_output[lane] == bottom_output[lane] (same pair_id)
        for lane_idx in range(self.elements_per_vector):
            lane_start = lane_idx * 32
            lane_end = lane_start + 31

            top_lane = Extract(lane_end, lane_start, top_output)
            bottom_lane = Extract(lane_end, lane_start, bottom_output)

            solver.add(top_lane == bottom_lane)

        # Check if constraints are satisfiable
        result = solver.check()
        return result == sat

    def synthesize_gadget_with_symbolic(self, top_instructions_template: list[InstructionSpec],
                                        bottom_instructions_template: list[InstructionSpec], input_state: VectorState,
                                        target_pairs: list[tuple[int, int]]) -> list[PermutationGadget]:
        """
        Synthesize gadgets using symbolic immediates in Z3.

        Takes instruction templates with symbolic values (e.g., BitVec("imm8", 8))
        and lets Z3 find concrete immediate values that satisfy constraints.

        Returns list of valid gadgets with concrete immediate values extracted from Z3 model.
        """
        solver = Solver()

        # Create pair_id mapping
        pair_id_map = self._create_pair_id_mapping(target_pairs)

        # Create input registers with pair IDs
        top_reg, bottom_reg = self._create_input_registers_with_pair_ids(solver, input_state, pair_id_map)

        # Collect all symbolic variables from instruction templates
        symbolic_vars = {}

        def collect_symbolic_vars(instructions: list[InstructionSpec]):
            """Extract all Z3 symbolic variables from instruction arguments."""
            for inst in instructions:
                for key, value in inst.args.items():
                    # Check if this is a Z3 expression (has decl method)
                    if hasattr(value, "decl") and callable(getattr(value, "decl", None)):
                        # Store the actual Z3 variable for later extraction
                        var_name = str(value)
                        symbolic_vars[id(value)] = value

        collect_symbolic_vars(top_instructions_template)
        collect_symbolic_vars(bottom_instructions_template)

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
        for lane_idx in range(self.elements_per_vector):
            lane_start = lane_idx * 32
            lane_end = lane_start + 31

            top_lane = Extract(lane_end, lane_start, top_output)
            bottom_lane = Extract(lane_end, lane_start, bottom_output)

            solver.add(top_lane == bottom_lane)

        # Check if constraints are satisfiable
        result = solver.check()
        if result != sat:
            return []

        # Extract concrete values from model
        model = solver.model()

        # Create concrete instructions by substituting symbolic values
        def concretize_instructions(instructions: list[InstructionSpec]) -> list[InstructionSpec]:
            concrete_insts = []
            for inst in instructions:
                concrete_args = {}
                for key, value in inst.args.items():
                    # Check if this is a symbolic variable (Z3 expression)
                    if id(value) in symbolic_vars:
                        # Check if it's a control vector (256-bit register) or a scalar immediate
                        if hasattr(value, "size") and callable(getattr(value, "size", None)):
                            bit_size = value.size()
                            if bit_size == 256:  # Control vector (YMM register)
                                # Extract the entire 256-bit value
                                concrete_bitvec = model.evaluate(value, model_completion=True)
                                # Convert to a list of 8 32-bit elements for display
                                # For now, keep as integer representation
                                concrete_value = concrete_bitvec.as_long() if hasattr(concrete_bitvec, "as_long") else 0
                                concrete_args[key] = concrete_value
                            elif bit_size == 8:  # Immediate (imm8)
                                concrete_value = model.evaluate(value, model_completion=True).as_long()
                                concrete_args[key] = concrete_value
                            else:
                                # Unknown size, try to extract as scalar
                                concrete_value = model.evaluate(value, model_completion=True).as_long()
                                concrete_args[key] = concrete_value
                        else:
                            # Fallback: extract as scalar
                            concrete_value = model.evaluate(value, model_completion=True).as_long()
                            concrete_args[key] = concrete_value
                    else:
                        concrete_args[key] = value
                concrete_insts.append(InstructionSpec(inst.intrinsic_name, concrete_args))
            return concrete_insts

        concrete_top = concretize_instructions(top_instructions_template)
        concrete_bottom = concretize_instructions(bottom_instructions_template)

        gadget = PermutationGadget(top_instructions=concrete_top, bottom_instructions=concrete_bottom, validated=True)

        return [gadget]

    def compute_output_state(self, input_state: VectorState, gadget: PermutationGadget) -> VectorState:
        """
        Compute the output state after applying a gadget to an input state.

        Uses Z3 to symbolically execute the gadget and determine element positions.

        Args:
            input_state: Input element positions
            gadget: Gadget to apply

        Returns:
            Output state with new element positions
        """
        solver = Solver()

        # Create input registers where each lane contains the element index
        if self.vm == vector_machine.AVX2:
            top_reg = z3_avx.ymm_reg("top_input")
            bottom_reg = z3_avx.ymm_reg("bottom_input")
        else:
            top_reg = z3_avx.zmm_reg("top_input")
            bottom_reg = z3_avx.zmm_reg("bottom_input")

        # Constrain input registers to contain element indices
        for lane_idx in range(self.elements_per_vector):
            top_elem = input_state.top[lane_idx]
            bottom_elem = input_state.bottom[lane_idx]

            lane_start = lane_idx * 32
            lane_end = lane_start + 31

            top_lane = Extract(lane_end, lane_start, top_reg)
            bottom_lane = Extract(lane_end, lane_start, bottom_reg)

            solver.add(top_lane == BitVecVal(top_elem, 32))
            solver.add(bottom_lane == BitVecVal(bottom_elem, 32))

        # Apply gadget instructions
        top_output = self._apply_instructions(top_reg, bottom_reg, gadget.top_instructions, is_top=True)
        bottom_output = self._apply_instructions(top_reg, bottom_reg, gadget.bottom_instructions, is_top=False)

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
            lane_start = lane_idx * 32
            lane_end = lane_start + 31

            top_lane = Extract(lane_end, lane_start, top_output)
            bottom_lane = Extract(lane_end, lane_start, bottom_output)

            # Evaluate the output lanes in the model
            top_val = model.evaluate(top_lane, model_completion=True)
            bottom_val = model.evaluate(bottom_lane, model_completion=True)

            # Convert Z3 bit-vector values to Python integers
            output_top.append(top_val.as_long())
            output_bottom.append(bottom_val.as_long())

        return VectorState(top=output_top, bottom=output_bottom)

    def _substitute_register_names(self, arg, top_reg, bottom_reg, current_reg, symbolic_vars=None):
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
                try:
                    var_name = str(arg)
                    symbolic_vars[var_name] = arg
                except:
                    pass
            return arg

        if isinstance(arg, str):
            if arg == "top":
                return top_reg
            elif arg == "bottom":
                return bottom_reg
            elif arg in ["input", "a"]:  # Generic input register
                return current_reg
        return arg

    def _apply_instructions(self, top_reg, bottom_reg, instructions: list[InstructionSpec], is_top: bool, solver=None,
                            symbolic_vars=None):
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
                args[key] = self._substitute_register_names(value, top_reg, bottom_reg, current_reg, symbolic_vars)

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
            elif "a" in args and "b" in args and "imm8" not in args and "mask" not in args:
                # Two sources without immediate (e.g., unpack, permutevar_ps)
                current_reg = intrinsic(args["a"], args["b"])
            else:
                # Generic fallback
                arg_values = list(args.values())
                current_reg = intrinsic(*arg_values)

        return current_reg

    def enumerate_gadgets(self, input_state: VectorState, target_pairs: list[tuple[int, int]], max_depth: int = 3) -> \
            tuple[list[PermutationGadget], int]:
        """
        Generate candidate gadgets up to max_depth instructions per vector.
        Returns validated gadgets.
        """
        valid_gadgets = []
        total_combinations_tried = 0

        # Try all combinations of (top_depth, bottom_depth) from (0,0) to (max_depth, max_depth)
        for top_depth in range(max_depth + 1):
            for bottom_depth in range(max_depth + 1):
                # Skip (0, 0) - no instructions means no change
                if top_depth == 0 and bottom_depth == 0:
                    # Check if input already satisfies target
                    if self._check_input_matches_target(input_state, target_pairs):
                        gadget = PermutationGadget([], [], validated=True)
                        valid_gadgets.append(gadget)
                    continue

                # Generate gadgets for this depth combination
                gadgets, combinations_tried = self._generate_gadgets_at_depth(input_state, target_pairs, top_depth,
                                                                              bottom_depth)
                valid_gadgets.extend(gadgets)
                total_combinations_tried += combinations_tried

        return valid_gadgets, total_combinations_tried

    def _check_input_matches_target(self, input_state: VectorState, target_pairs: list[tuple[int, int]]) -> bool:
        """Check if input state already matches target pairs (all pairs aligned on same lanes)."""
        for i in range(len(input_state.top)):
            top_elem = input_state.top[i]
            bottom_elem = input_state.bottom[i]
            # Check if this forms a valid pair
            pair_found = False
            for pair in target_pairs:
                if (top_elem == pair[0] and bottom_elem == pair[1]) or (top_elem == pair[1] and bottom_elem == pair[0]):
                    pair_found = True
                    break
            if not pair_found:
                return False
        return True

    def _generate_gadgets_at_depth(self, input_state: VectorState, target_pairs: list[tuple[int, int]], top_depth: int,
                                   bottom_depth: int) -> tuple[list[PermutationGadget], int]:
        """Generate and validate gadgets with specific instruction depths."""
        valid_gadgets = []
        max_combinations_to_try = 50  # Reduced since symbolic synthesis is more powerful
        # Get instruction template pools (now with symbolic immediates)
        single_insts_top = self._enumerate_single_input_instructions("top")
        single_insts_bottom = self._enumerate_single_input_instructions("bottom")
        dual_insts_top_bottom = self._enumerate_dual_input_instructions("top", "bottom")
        dual_insts_bottom_top = self._enumerate_dual_input_instructions("bottom", "top")
        # Build instruction sequences for top
        top_sequences = []
        if top_depth > 0:
            if top_depth == 1:
                # Try single instructions
                top_sequences = [[inst] for inst in single_insts_top[:3]]
                top_sequences.extend([[inst] for inst in dual_insts_top_bottom[:2]])
            elif top_depth == 2:
                # Try pairs: (single, single) and (dual, single)
                for inst1 in single_insts_top[:2]:  # Try first 2 of each type
                    for inst2 in single_insts_top[:2]:
                        top_sequences.append([inst1, inst2])
                for inst1 in dual_insts_top_bottom[:2]:
                    for inst2 in single_insts_top[:2]:
                        top_sequences.append([inst1, inst2])
            elif top_depth == 3:
                # Try triples: (single, single, single)
                for inst1 in single_insts_top[:2]:
                    for inst2 in single_insts_top[:2]:
                        for inst3 in single_insts_top[:2]:
                            top_sequences.append([inst1, inst2, inst3])
        else:
            top_sequences = [[]]  # Empty sequence for depth 0
        # Build instruction sequences for bottom
        bottom_sequences = []
        if bottom_depth > 0:
            if bottom_depth == 1:
                bottom_sequences = [[inst] for inst in single_insts_bottom[:3]]
                bottom_sequences.extend([[inst] for inst in dual_insts_bottom_top[:2]])
            elif bottom_depth == 2:
                for inst1 in single_insts_bottom[:2]:
                    for inst2 in single_insts_bottom[:2]:
                        bottom_sequences.append([inst1, inst2])
            elif bottom_depth == 3:
                for inst1 in single_insts_bottom[:2]:
                    for inst2 in single_insts_bottom[:2]:
                        for inst3 in single_insts_bottom[:2]:
                            bottom_sequences.append([inst1, inst2, inst3])
        else:
            bottom_sequences = [[]]  # Empty sequence for depth 0
        # Try combinations (limited)
        combinations_tried = 0
        for top_seq in top_sequences:
            if combinations_tried >= max_combinations_to_try:
                break
            for bottom_seq in bottom_sequences:
                if combinations_tried >= max_combinations_to_try:
                    break

                # Use symbolic synthesis instead of validation
                gadgets = self.synthesize_gadget_with_symbolic(top_seq, bottom_seq, input_state, target_pairs)
                valid_gadgets.extend(gadgets)

                combinations_tried += 1
        return valid_gadgets, combinations_tried

    def _enumerate_single_input_instructions(self, reg_name: str = "input") -> list[InstructionSpec]:
        """
        Generate single-input instruction templates with symbolic immediates or control vectors.
        Z3 will solve for the concrete values.

        Single-input means: operates on ONE of our vectors (top OR bottom),
        even if it takes additional operands like control vectors.
        """
        instructions = []

        if self.vm == vector_machine.AVX2 and self.prim_type == primitive_type.i32:
            input_reg = reg_name
            unique_id = id(input_reg)

            # Create instruction templates with symbolic immediates/controls
            # Z3 will find the concrete values that satisfy the constraints

            # Instructions with symbolic immediates
            # _mm256_permute_ps: permute within 128-bit lanes
            instructions.append(InstructionSpec("_mm256_permute_ps",
                                                {"a": input_reg, "imm8": BitVec(f"imm8_permute_ps_{unique_id}", 8)}))

            # _mm256_permute4x64_epi64: permute 64-bit chunks (affects i32 grouping)
            instructions.append(InstructionSpec("_mm256_permute4x64_epi64",
                                                {"a": input_reg, "imm8": BitVec(f"imm8_permute4x64_{unique_id}", 8)}))

            # Instructions with symbolic control vectors
            # _mm256_permutexvar_epi32: variable permute across all lanes (most powerful!)
            instructions.append(InstructionSpec("_mm256_permutexvar_epi32", {"a": input_reg, "op_idx": z3_avx.ymm_reg(
                f"ctrl_permutexvar_{unique_id}")}))

            # _mm256_permutevar_ps: variable permute within 128-bit lanes
            instructions.append(InstructionSpec("_mm256_permutevar_ps", {"a": input_reg, "b": z3_avx.ymm_reg(
                f"ctrl_permutevar_ps_{unique_id}")}))

        return instructions

    def _enumerate_dual_input_instructions(self, reg1_name: str = "top", reg2_name: str = "bottom") -> list[
        InstructionSpec]:
        """
        Generate dual-input instruction templates with symbolic immediates.
        Z3 will solve for the concrete immediate values.
        """
        instructions = []

        if self.vm == vector_machine.AVX2 and self.prim_type == primitive_type.i32:
            reg1 = reg1_name
            reg2 = reg2_name

            # Generate unique IDs for symbolic variables
            unique_id = f"{id(reg1)}_{id(reg2)}"

            # Shuffle instructions with symbolic immediate
            instructions.append(InstructionSpec("_mm256_shuffle_ps",
                                                {"a": reg1, "b": reg2, "imm8": BitVec(f"imm8_shuffle_{unique_id}", 8)}))

            # Unpack instructions (no immediates needed)
            instructions.append(InstructionSpec("_mm256_unpacklo_epi32", {"a": reg1, "b": reg2}))
            instructions.append(InstructionSpec("_mm256_unpackhi_epi32", {"a": reg1, "b": reg2}))

            # Permute2x128 with symbolic immediate
            instructions.append(InstructionSpec("_mm256_permute2x128_si256", {"a": reg1, "b": reg2, "imm8": BitVec(
                f"imm8_perm2x128_{unique_id}", 8)}))

            # Blend instructions with symbolic immediate
            instructions.append(InstructionSpec("_mm256_blend_ps",
                                                {"a": reg1, "b": reg2, "imm8": BitVec(f"imm8_blend_{unique_id}", 8)}))

            # Alignr with symbolic immediate
            instructions.append(InstructionSpec("_mm256_alignr_epi32",
                                                {"a": reg1, "b": reg2, "imm8": BitVec(f"imm8_alignr_{unique_id}", 8)}))

        return instructions


class BitonicSuperVectorizer:
    """Super-optimizer for bitonic sorting networks using Z3-based gadget synthesis."""

    def __init__(self, num_vecs: int, prim_type: primitive_type, vm: vector_machine):
        self.num_vecs = num_vecs
        self.prim_type = prim_type
        self.vm = vm

        # Calculate total elements and elements per vector
        self.elements_per_vector = width_dict[vm] // int(prim_type.value[0])
        self.total_elements = num_vecs * self.elements_per_vector

        # Initialize bitonic sorter to get comparison pairs per stage
        self.bitonic_sorter = BitonicSorter(self.total_elements)

        # Initialize gadget synthesizer
        self.synthesizer = GadgetSynthesizer(vm, prim_type)

        print(
            f"BitonicSuperVectorizer: {num_vecs} x {vm.name} vectors, {self.elements_per_vector} x {prim_type.name} elements per vector, {self.total_elements} total elements, {len(self.bitonic_sorter.stages)} stages")

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
        assert len(
            top) == self.elements_per_vector, f"Expected {self.elements_per_vector} elements in top, got {len(top)}"
        assert len(
            bottom) == self.elements_per_vector, f"Expected {self.elements_per_vector} elements in bottom, got {len(bottom)}"

        return VectorState(top=top, bottom=bottom)

    def synthesize_stage(self, input_state: VectorState, target_pairs: list[tuple[int, int]]) -> tuple[
        list[PermutationGadget], int]:
        """
        For given input state, find all valid gadgets that align target pairs.
        """
        return self.synthesizer.enumerate_gadgets(input_state, target_pairs, max_depth=3)

    def _apply_min_max_exchange(self, state: VectorState, pairs: list[tuple[int, int]]) -> VectorState:
        """
        Apply min-max exchange: for each pair (a, b) where a < b,
        ensure a is in top and b is in bottom at the same lane.
        """
        output = state.copy()
        # Min-max exchange doesn't change positions, it just ensures proper ordering
        # For our purposes, we assume the permutation gadget already aligned pairs correctly
        # The min-max just swaps values, not positions
        return output

    def build_solution_tree(self) -> list[SolutionNode]:
        """
        Recursively explore all stage transitions to build solution tree.
        Returns root nodes (first stage solutions).
        """
        initial_state = self._create_initial_state()
        return self._build_tree_recursive(initial_state, 0)

    def _build_tree_recursive(self, input_state: VectorState, stage_idx: int) -> list[SolutionNode]:
        """Recursively build solution tree from given state and stage."""
        if stage_idx >= len(self.bitonic_sorter.stages):
            # No more stages, return empty list
            return []

        stage_pairs = self.bitonic_sorter.stages[stage_idx]

        # Find all valid gadgets for this stage
        gadgets, _ = self.synthesize_stage(input_state, stage_pairs)

        if not gadgets:
            print(f"Warning: No gadgets found for stage {stage_idx}")
            return []

        nodes = []
        for gadget in gadgets:
            # Compute output state after applying gadget and min-max exchange
            next_input_state = self._compute_output_state(input_state, gadget, stage_pairs)

            # Recursively build children for next stage
            children = self._build_tree_recursive(next_input_state, stage_idx + 1)

            node = SolutionNode(stage=stage_idx, input_state=input_state, output_state=next_input_state, gadget=gadget,
                                children=children)
            nodes.append(node)

        return nodes

    def _compute_output_state(self, input_state: VectorState, gadget: PermutationGadget,
                              stage_pairs: list[tuple[int, int]]) -> VectorState:
        """
        Compute output state after applying gadget.

        Uses Z3 to symbolically execute the gadget and determine where each element ends up.
        """
        # Use the synthesizer to compute the output state
        return self.synthesizer.compute_output_state(input_state, gadget)

    def synthesize_all_stages(self) -> list[SolutionNode]:
        """Entry point: builds solution tree for all stages."""
        return self.build_solution_tree()

    def compute_costs(self, roots: list[SolutionNode], cost_model):
        """Traverse tree and compute cumulative costs for each path."""
        for root in roots:
            self._compute_costs_recursive(root, cost_model)

    def _compute_costs_recursive(self, node: SolutionNode, cost_model):
        """Recursively compute costs for node and its children."""
        node.cost = cost_model.calculate_gadget_cost(node.gadget)
        for child in node.children:
            self._compute_costs_recursive(child, cost_model)
            # Add parent cost to child for cumulative cost
            child.cost += node.cost

    def export_solutions(self, roots: list[SolutionNode], output_path: str):
        """Generate JSON with all solutions and costs."""
        import json

        def node_to_dict(node: SolutionNode) -> dict:
            return {
                "stage": node.stage,
                "input_state": {"top": node.input_state.top, "bottom": node.input_state.bottom},
                "output_state": {"top": node.output_state.top, "bottom": node.output_state.bottom},
                "gadget": {
                    "top_instructions": [{"name": inst.intrinsic_name, "args": inst.args} for inst in
                                         node.gadget.top_instructions],
                    "bottom_instructions": [{"name": inst.intrinsic_name, "args": inst.args} for inst in
                                            node.gadget.bottom_instructions],
                    "instruction_count": node.gadget.instruction_count(),
                },
                "cost": node.cost,
                "children": [node_to_dict(child) for child in node.children],
            }

        solutions = [node_to_dict(root) for root in roots]

        with open(output_path, "w") as f:
            json.dump(solutions, f, indent=2)

        print(f"Exported {len(roots)} solution trees to {output_path}")


def generate_bitonic_sorter(num_vecs: int, type: primitive_type, vm: vector_machine):
    """
    Generate bitonic sorter with super-optimized permutation sequences.

    Args:
        num_vecs: Number of SIMD vectors to sort
        type: Primitive type (i32, f32, i64, f64)
        vm: Vector machine (AVX2, AVX512)

    Returns:
        List of SolutionNode trees representing different optimized solutions
    """
    total_elements = int(num_vecs * (width_dict[vm] / int(type.value[0])))

    print(f"Building {vm.name} sorter for {total_elements} elements ({num_vecs} vectors)")

    # Create super-vectorizer
    super_opt = BitonicSuperVectorizer(num_vecs, type, vm)

    # Synthesize all stages to build solution tree
    print("Synthesizing permutation gadgets...")
    solutions = super_opt.synthesize_all_stages()

    print(f"Found {len(solutions)} root solutions")

    # Compute costs
    print("Computing costs...")
    cost_model = CostModel("generic")
    super_opt.compute_costs(solutions, cost_model)

    # Export solutions to JSON
    output_path = f"bitonic_solutions_{num_vecs}x{vm.name}_{type.name}.json"
    super_opt.export_solutions(solutions, output_path)

    return solutions


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # Start with 2 vectors, i32, AVX2 as per plan
    generate_bitonic_sorter(2, primitive_type.i32, vector_machine.AVX2)
