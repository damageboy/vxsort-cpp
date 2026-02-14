from __future__ import annotations

from utils import vector_machine, primitive_type
from utils import width_dict
from z3_avx import mm_shuffle_str, mm_shuffle2_str


def _get_instruction_metadata(intrinsic_name: str) -> dict:
    """
    Get metadata about an instruction for proper comment formatting.

    Returns:
        dict with:
        - 'shuffle_type': 'shuffle2' for 2-element, 'shuffle4' for 4-element, None for not a shuffle
        - 'control_vector_width': element width in bits for control vector operations, None otherwise
    """
    # Instructions that use shuffle with 2 elements (shuffle_pd variants)
    shuffle2_instructions = {
        "_mm256_shuffle_pd",
        "_mm512_shuffle_pd",
        "_mm256_permute_pd",
        "_mm512_permute_pd",
    }

    # Instructions that use shuffle with 4 elements (shuffle_ps variants)
    shuffle4_instructions = {
        "_mm256_shuffle_ps",
        "_mm512_shuffle_ps",
        "_mm256_permute_ps",
        "_mm512_permute_ps",
    }

    # Instructions that use control vectors with their element widths
    control_vector_instructions = {
        # 32-bit control vectors
        "_mm256_permutexvar_epi32": 32,
        "_mm512_permutexvar_epi32": 32,
        "_mm256_permutevar_ps": 32,
        "_mm512_permutevar_ps": 32,
        "_mm512_permutex2var_epi32": 32,
        # 64-bit control vectors
        "_mm256_permutexvar_epi64": 64,
        "_mm512_permutexvar_epi64": 64,
        "_mm256_permutevar_pd": 64,
        "_mm512_permutevar_pd": 64,
        "_mm512_permutex2var_epi64": 64,
    }

    metadata = {
        "shuffle_type": None,
        "control_vector_width": None,
    }

    if intrinsic_name in shuffle2_instructions:
        metadata["shuffle_type"] = "shuffle2"
    elif intrinsic_name in shuffle4_instructions:
        metadata["shuffle_type"] = "shuffle4"

    if intrinsic_name in control_vector_instructions:
        metadata["control_vector_width"] = control_vector_instructions[intrinsic_name]

    return metadata


def _intrinsic_to_asm_mnemonic(intrinsic_name: str) -> str:
    """Map intrinsic name to assembly mnemonic."""
    # Common AVX2/AVX512 intrinsics to assembly mnemonics
    mapping = {
        # Permute operations
        "_mm256_permute4x64_epi64": "vpermq",
        "_mm256_permute_ps": "vpermilps",
        "_mm256_permute_pd": "vpermilpd",
        "_mm256_permutexvar_epi32": "vpermd",
        "_mm256_permutexvar_epi64": "vpermq",
        "_mm256_permutevar_ps": "vpermilps",
        "_mm256_permutevar_pd": "vpermilpd",
        "_mm512_permutexvar_epi64": "vpermq",
        "_mm512_permutexvar_epi32": "vpermd",
        # Shuffle operations
        "_mm256_shuffle_ps": "vshufps",
        "_mm256_shuffle_pd": "vshufpd",
        "_mm256_shuffle_epi32": "vpshufd",
        "_mm512_shuffle_ps": "vshufps",
        "_mm512_shuffle_epi32": "vpshufd",
        # Unpack operations
        "_mm256_unpacklo_epi32": "vpunpckldq",
        "_mm256_unpackhi_epi32": "vpunpckhdq",
        "_mm256_unpacklo_epi64": "vpunpcklqdq",
        "_mm256_unpackhi_epi64": "vpunpckhqdq",
        "_mm256_unpacklo_ps": "vunpcklps",
        "_mm256_unpackhi_ps": "vunpckhps",
        "_mm512_unpacklo_epi32": "vpunpckldq",
        "_mm512_unpackhi_epi32": "vpunpckhdq",
        # Permute2 operations
        "_mm256_permute2x128_si256": "vperm2i128",
        "_mm256_permute2f128_ps": "vperm2f128",
        # Blend operations
        "_mm256_blend_ps": "vblendps",
        "_mm256_blend_pd": "vblendpd",
        "_mm256_blendv_ps": "vblendvps",
        "_mm256_blendv_pd": "vblendvpd",
        "_mm256_blend_epi32": "vpblendd",
        "_mm512_mask_blend_ps": "vblendmps",
        "_mm512_mask_blend_epi32": "vpblendmd",
        # Align operations
        "_mm256_alignr_epi32": "valignd",
        "_mm512_alignr_epi32": "valignd",
        "_mm256_alignr_epi64": "valignq",
        "_mm512_alignr_epi64": "valignq",
        # Min/Max operations
        "_mm256_min_ps": "vminps",
        "_mm256_max_ps": "vmaxps",
        "_mm512_min_ps": "vminps",
        "_mm512_max_ps": "vmaxps",
    }

    return mapping.get(intrinsic_name, intrinsic_name)


class RegisterAllocator:
    """Manages register allocation for assembly output."""

    def __init__(self, vm: vector_machine, dtype: primitive_type, num_vecs: int):
        self.vm = vm
        self.dtype = dtype
        self.num_vecs = num_vecs
        # Determine register prefix based on architecture
        if vm == vector_machine.AVX2:
            self.reg_prefix = "ymm"
        elif vm == vector_machine.AVX512:
            self.reg_prefix = "zmm"
        else:
            self.reg_prefix = "v"  # fallback

        # First num_vecs registers are for data
        self.next_temp_reg = num_vecs
        self.temp_regs_allocated = []

    def get_data_reg(self, vec_idx: int) -> str:
        """Get the register name for a data vector."""
        return f"{self.reg_prefix}{vec_idx}"

    def allocate_temp(self) -> str:
        """Allocate a temporary register."""
        reg = f"{self.reg_prefix}{self.next_temp_reg}"
        self.next_temp_reg += 1
        self.temp_regs_allocated.append(reg)
        return reg

    def reset_temps(self):
        """Reset temporary register allocation."""
        self.next_temp_reg = self.num_vecs
        self.temp_regs_allocated = []


def _format_control_vector(val: int, vm: vector_machine, dtype: primitive_type) -> str:
    """Format a 256/512-bit control vector as a list of elements."""

    total_bytes = width_dict[vm]
    total_bits = total_bytes * 8
    element_bytes = dtype.value[0]
    element_bits = element_bytes * 8
    num_lanes = total_bits // element_bits

    elements = []
    for i in range(num_lanes):
        element = (val >> (i * element_bits)) & ((1 << element_bits) - 1)
        elements.append(element)

    return "[" + ", ".join(str(e) for e in elements) + "]"


def _resolve_source(name: str, top_reg: str, bottom_reg: str, is_top: bool) -> str:
    """Map a source name to a physical register.

    Args:
        name: Source name ("top", "bottom", or "prev")
        top_reg: Physical register for the top vector
        bottom_reg: Physical register for the bottom vector
        is_top: Whether we're formatting a top-side instruction chain
    """
    if name == "top":
        return top_reg
    elif name == "bottom":
        return bottom_reg
    elif name == "prev":
        # "prev" = output of prior instruction in this chain,
        # which was written to the destination register
        return top_reg if is_top else bottom_reg
    return name  # not a register name (shouldn't happen for a/b args)


def _format_instruction(
    inst,
    reg_allocator: RegisterAllocator,
    top_reg: str,
    bottom_reg: str,
    is_top: bool,
) -> str:
    """
    Format a single instruction as assembly.

    Args:
        inst: InstructionSpec with intrinsic_name and args
        reg_allocator: RegisterAllocator for temporary registers
        top_reg: Physical register for the top vector (absolute)
        bottom_reg: Physical register for the bottom vector (absolute)
        is_top: Whether this instruction is in the top-side chain
    """
    mnemonic = _intrinsic_to_asm_mnemonic(inst.intrinsic_name)
    metadata = _get_instruction_metadata(inst.intrinsic_name)
    args = inst.args
    dest_reg = top_reg if is_top else bottom_reg

    # Build operand list - Intel syntax: dest, src1, [src2], [imm]
    operands = [dest_reg]  # Destination is always first

    ctrl_val = None
    ctrl_element_width = None  # Track element width for control vectors

    # Handle different instruction patterns based on arguments
    if "a" in args and "b" in args:
        # Two-input instruction (shuffle, blend, unpack, etc.)
        src1 = _resolve_source(args["a"], top_reg, bottom_reg, is_top)
        src2_raw = args["b"]

        # Check if 'b' is a control vector (not a register name)
        if src2_raw not in ["top", "bottom", "prev"]:
            # It's a control vector - allocate a temp register for it
            ctrl_val = src2_raw
            ctrl_element_width = metadata["control_vector_width"]
            ctrl_reg = reg_allocator.allocate_temp()
            operands.append(src1)
            operands.append(ctrl_reg)
        else:
            src2 = _resolve_source(src2_raw, top_reg, bottom_reg, is_top)
            operands.append(src1)
            operands.append(src2)
    elif "a" in args:
        # Single-input instruction
        src = _resolve_source(args["a"], top_reg, bottom_reg, is_top)

        # Check for control/index operand
        if "op_idx" in args:
            # Variable permute with control vector
            ctrl_val = args["op_idx"]
            ctrl_element_width = metadata["control_vector_width"]
            ctrl_reg = reg_allocator.allocate_temp()
            operands.append(ctrl_reg)
            operands.append(src)
        else:
            operands.append(src)
    elif "input" in args:
        # Old-style single-input
        src = _resolve_source(args["input"], top_reg, bottom_reg, is_top)
        operands.append(src)

    # Add immediate values at the end
    comment = None
    if "imm8" in args:
        imm_val = args["imm8"]
        if isinstance(imm_val, int):
            operands.append(f"0x{imm_val:02x}")
            # Use appropriate shuffle formatter based on instruction type
            if metadata["shuffle_type"] == "shuffle2":
                comment = mm_shuffle2_str(imm_val)
            elif metadata["shuffle_type"] == "shuffle4":
                comment = mm_shuffle_str(imm_val)
            else:
                # Default to 4-element for backward compatibility
                comment = mm_shuffle_str(imm_val)
        else:
            operands.append(f"<{imm_val}>")  # Symbolic value
    elif "imm" in args:
        imm_val = args["imm"]
        if isinstance(imm_val, int):
            operands.append(f"0x{imm_val:x}")
        else:
            operands.append(f"<{imm_val}>")  # Symbolic value
    elif "mask" in args:
        mask_val = args["mask"]
        if isinstance(mask_val, int):
            if mask_val > 0xFF:
                # 256/512-bit mask for blendv
                ctrl_val = mask_val
                ctrl_reg = reg_allocator.allocate_temp()
                operands.append(ctrl_reg)
            else:
                operands.append(f"0x{mask_val:02x}")
        else:
            operands.append(f"<{mask_val}>")

    if ctrl_val is not None:
        # Format control vector with correct element width
        if ctrl_element_width is not None:
            # Use the instruction's element width instead of the data type's width

            total_bits = width_dict[reg_allocator.vm] * 8
            num_elements = total_bits // ctrl_element_width

            elements = []
            for i in range(num_elements):
                element = (ctrl_val >> (i * ctrl_element_width)) & (
                    (1 << ctrl_element_width) - 1
                )
                elements.append(element)

            comment = "[" + ", ".join(str(e) for e in elements) + "]"
        else:
            # Fall back to data type's element width
            comment = _format_control_vector(
                ctrl_val, reg_allocator.vm, reg_allocator.dtype
            )

    asm_line = f"    {mnemonic:20s} {', '.join(operands)}"
    if comment:
        asm_line += f"  ; {comment}"
    return asm_line


def _format_vector_state_as_comment(state, prefix: str = "") -> str:
    """Format VectorState __repr__ output as assembly comments."""
    state_str = repr(state)
    lines = state_str.split("\n")
    comment_lines = [f"{prefix}; {line}" for line in lines]
    return "\n".join(comment_lines)


def _print_solution_step_as_assembly(
    step,
    reg_allocator: RegisterAllocator,
    cumulative_cost: float = 0.0,
    indent: int = 0,
):
    """Print a single PathStep as assembly code.

    Args:
        step: PathStep with node and selected gadget_index
        reg_allocator: RegisterAllocator for register management
        cumulative_cost: Running total cost up to this step
        indent: Indentation level for nested output
    """
    prefix = "  " * indent

    # Print stage header
    print(f"{prefix}; Stage {step.node.stage}")
    print(f"{prefix}; Cost: {cumulative_cost:.2f} (step: {step.score.total_score:.2f})")
    if step.score.control_vector_count > 0:
        print(f"{prefix}; Control vectors: {step.score.control_vector_count}")

    # Get register names
    top_reg = reg_allocator.get_data_reg(0)
    bottom_reg = reg_allocator.get_data_reg(1) if reg_allocator.num_vecs > 1 else None

    # Use the explicitly selected gadget from the step
    gadget = step.gadget

    # Print top vector instructions
    if gadget.top_instructions:
        print(f"{prefix}; Top vector ({top_reg}) operations:")
        for inst in gadget.top_instructions:
            effective_bottom = bottom_reg if bottom_reg else top_reg
            asm_line = _format_instruction(
                inst, reg_allocator, top_reg, effective_bottom, is_top=True
            )
            print(f"{prefix}{asm_line}")

    # Print bottom vector instructions
    if gadget.bottom_instructions and bottom_reg:
        print(f"{prefix}; Bottom vector ({bottom_reg}) operations:")
        for inst in gadget.bottom_instructions:
            asm_line = _format_instruction(
                inst, reg_allocator, top_reg, bottom_reg, is_top=False
            )
            print(f"{prefix}{asm_line}")

    # Print output state
    print(f"{prefix}; Output State:")
    print(_format_vector_state_as_comment(step.node.output_state, prefix))

    print()

    # Reset temporary registers for next stage
    reg_allocator.reset_temps()


_MAX_ASM_PATHS = 10_000


def export_solutions_to_asm(
    solutions,
    num_vecs: int,
    dtype: primitive_type,
    vm: vector_machine,
    output_path: str,
    selected_paths=None,
    natural_order: bool = False,
):
    """Export solutions as readable assembly code.

    Args:
        solutions: List of root SolutionNode objects (may be pruned)
        num_vecs: Number of SIMD vectors
        dtype: Primitive data type
        vm: Vector machine (AVX2/AVX512)
        output_path: Path to write assembly output
        selected_paths: Optional list of CompletePath from PathSelector.
                       If provided, uses these explicit paths instead of
                       enumerating all paths through the DAG.
    """
    import sys
    from cost_model import CostModel
    from path_selector import PathSelector

    # If selected_paths is provided, use them directly
    if selected_paths is not None:
        all_paths = selected_paths
        total_path_count = len(all_paths)
        truncated = False
    else:
        # Fall back to old behavior: enumerate paths through DAG
        # This is needed when top_k is None (no pruning)
        print(
            "Warning: No selected_paths provided, enumerating all paths (may be slow)"
        )

        # Use PathSelector to enumerate paths with cap
        cost_model = CostModel("generic")
        cost_model.compute_costs(solutions)
        path_selector = PathSelector(cost_model)

        # Select paths with a large limit to get "all" paths, capped at _MAX_ASM_PATHS
        all_paths = path_selector.select_top_k_paths(solutions, _MAX_ASM_PATHS)
        total_path_count = len(all_paths)
        truncated = total_path_count >= _MAX_ASM_PATHS

    with open(output_path, "w") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            print("; Bitonic Sort Assembly Output")
            print(f"; Architecture: {vm.name}")
            print(f"; Data Type: {dtype.name}")
            print(f"; Number of vectors: {num_vecs}")
            print(f"; Natural order: {'yes' if natural_order else 'no'}")
            print(f"; Root solutions: {len(solutions)}")
            if truncated:
                print(f"; NOTE: Showing {len(all_paths)} paths (may be capped)")
            print(";")
            print("; Registers:")
            reg_prefix = "ymm" if vm == vector_machine.AVX2 else "zmm"
            for i in range(num_vecs):
                print(f";   {reg_prefix}{i}: Input/output vector {i}")
            print(f";   {reg_prefix}{num_vecs}+: Temporary registers as needed")
            print()
            print("=" * 80)
            print()

            print(f"; Total paths: {len(all_paths)}")
            print()

            for i, path in enumerate(all_paths):
                print(f"; ========== SOLUTION {i + 1} of {len(all_paths)} ==========")
                print(f"; Total latency: {path.total_latency:.2f}")
                print(f"; Total score: {path.total_score:.2f}")
                print(f"; Control vectors: {path.total_cv_count}")
                print()

                # Print initial input state (before first stage)
                if path.steps:
                    print("; Initial Input State:")
                    print(
                        _format_vector_state_as_comment(
                            path.steps[0].node.input_state, ""
                        )
                    )
                    print()

                reg_allocator = RegisterAllocator(vm, dtype, num_vecs)

                # Print each step in the path, accumulating cost
                running_cost = 0.0
                for step in path.steps:
                    running_cost += step.score.total_score
                    _print_solution_step_as_assembly(
                        step, reg_allocator, cumulative_cost=running_cost
                    )

                print("=" * 80)
                print()
        finally:
            sys.stdout = old_stdout

    print(f"Exported {len(all_paths)} solutions to {output_path}")
