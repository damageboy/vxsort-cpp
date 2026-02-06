from __future__ import annotations

from utils import vector_machine, primitive_type
from z3_avx import mm_shuffle_str


def _intrinsic_to_asm_mnemonic(intrinsic_name: str) -> str:
    """Map intrinsic name to assembly mnemonic."""
    # Common AVX2/AVX512 intrinsics to assembly mnemonics
    mapping = {
        # Permute operations
        "_mm256_permute4x64_epi64": "vpermq",
        "_mm256_permute_ps": "vpermilps",
        "_mm256_permutexvar_epi32": "vpermd",
        "_mm256_permutevar_ps": "vpermilps",
        "_mm512_permutexvar_epi64": "vpermq",
        "_mm512_permutexvar_epi32": "vpermd",
        # Shuffle operations
        "_mm256_shuffle_ps": "vshufps",
        "_mm256_shuffle_epi32": "vpshufd",
        "_mm512_shuffle_ps": "vshufps",
        "_mm512_shuffle_epi32": "vpshufd",
        # Unpack operations
        "_mm256_unpacklo_epi32": "vpunpckldq",
        "_mm256_unpackhi_epi32": "vpunpckhdq",
        "_mm256_unpacklo_ps": "vunpcklps",
        "_mm256_unpackhi_ps": "vunpckhps",
        "_mm512_unpacklo_epi32": "vpunpckldq",
        "_mm512_unpackhi_epi32": "vpunpckhdq",
        # Permute2 operations
        "_mm256_permute2x128_si256": "vperm2i128",
        "_mm256_permute2f128_ps": "vperm2f128",
        # Blend operations
        "_mm256_blend_ps": "vblendps",
        "_mm256_blendv_ps": "vblendvps",
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
    from utils import width_dict

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


def _format_instruction(
    inst, reg_allocator: RegisterAllocator, dest_reg: str, other_reg: str
) -> str:
    """
    Format a single instruction as assembly.

    Args:
        inst: InstructionSpec with intrinsic_name and args
        reg_allocator: RegisterAllocator for temporary registers
        dest_reg: The destination register (top or bottom)
        other_reg: The other data register
    """
    mnemonic = _intrinsic_to_asm_mnemonic(inst.intrinsic_name)
    args = inst.args

    # Build operand list - Intel syntax: dest, src1, [src2], [imm]
    operands = [dest_reg]  # Destination is always first

    ctrl_val = None

    # Handle different instruction patterns based on arguments
    if "a" in args and "b" in args:
        # Two-input instruction (shuffle, blend, unpack, etc.)
        src1 = dest_reg if args["a"] == "top" else other_reg
        src2 = dest_reg if args["b"] == "top" else other_reg

        # Check if 'b' is a control vector (not a register name)
        if args["b"] not in ["top", "bottom"]:
            # It's a control vector - allocate a temp register for it
            ctrl_val = args["b"]
            ctrl_reg = reg_allocator.allocate_temp()
            operands.append(src1)
            operands.append(ctrl_reg)
        else:
            operands.append(src1)
            operands.append(src2)
    elif "a" in args:
        # Single-input instruction
        src = dest_reg if args["a"] == "top" else other_reg

        # Check for control/index operand
        if "op_idx" in args:
            # Variable permute with control vector
            ctrl_val = args["op_idx"]
            ctrl_reg = reg_allocator.allocate_temp()
            operands.append(ctrl_reg)
            operands.append(src)
        else:
            operands.append(src)
    elif "input" in args:
        # Old-style single-input
        src = dest_reg if args["input"] == "top" else other_reg
        operands.append(src)

    # Add immediate values at the end
    comment = None
    if "imm8" in args:
        imm_val = args["imm8"]
        if isinstance(imm_val, int):
            operands.append(f"0x{imm_val:02x}")
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
        comment = _format_control_vector(
            ctrl_val, reg_allocator.vm, reg_allocator.dtype
        )

    asm_line = f"    {mnemonic:20s} {', '.join(operands)}"
    if comment:
        asm_line += f"  ; {comment}"
    return asm_line


def _collect_all_paths(node, current_path=None) -> list[list]:
    """
    Collect all root-to-leaf paths in the tree.
    Returns a list of paths, where each path is a list of nodes.
    """
    if current_path is None:
        current_path = []
    current_path = current_path + [node]

    if not node.children:
        return [current_path]

    paths = []
    for child in node.children:
        paths.extend(_collect_all_paths(child, current_path))
    return paths


def _format_vector_state_as_comment(state, prefix: str = "") -> str:
    """Format VectorState __repr__ output as assembly comments."""
    state_str = repr(state)
    lines = state_str.split("\n")
    comment_lines = [f"{prefix}; {line}" for line in lines]
    return "\n".join(comment_lines)


def _print_solution_as_assembly(
    solution_node, reg_allocator: RegisterAllocator, path_num: int = 1, indent: int = 0
):
    """Print a single solution path as assembly code."""
    prefix = "  " * indent

    # Print stage header
    print(f"{prefix}; Stage {solution_node.stage}")
    print(f"{prefix}; Cost: {solution_node.cost:.2f}")

    # Get register names
    top_reg = reg_allocator.get_data_reg(0)
    bottom_reg = reg_allocator.get_data_reg(1) if reg_allocator.num_vecs > 1 else None

    # Use the best gadget (lowest instruction count) from the node
    gadget = solution_node.best_gadget()

    # Print top vector instructions
    if gadget.top_instructions:
        print(f"{prefix}; Top vector ({top_reg}) operations:")
        for inst in gadget.top_instructions:
            if bottom_reg:
                asm_line = _format_instruction(inst, reg_allocator, top_reg, bottom_reg)
            else:
                asm_line = _format_instruction(inst, reg_allocator, top_reg, top_reg)
            print(f"{prefix}{asm_line}")

    # Print bottom vector instructions
    if gadget.bottom_instructions and bottom_reg:
        print(f"{prefix}; Bottom vector ({bottom_reg}) operations:")
        for inst in gadget.bottom_instructions:
            asm_line = _format_instruction(inst, reg_allocator, bottom_reg, top_reg)
            print(f"{prefix}{asm_line}")

    # Print output state
    print(f"{prefix}; Output State:")
    print(_format_vector_state_as_comment(solution_node.output_state, prefix))

    print()

    # Reset temporary registers for next stage
    reg_allocator.reset_temps()


def export_solutions_as_assembly(
    solutions,
    num_vecs: int,
    dtype: primitive_type,
    vm: vector_machine,
    output_path: str,
):
    """Export solutions as readable assembly code."""
    # Lazy import to avoid circular dependency

    with open(output_path, "w") as f:
        import sys

        old_stdout = sys.stdout
        sys.stdout = f

        print("; Bitonic Sort Assembly Output")
        print(f"; Architecture: {vm.name}")
        print(f"; Data Type: {dtype.name}")
        print(f"; Number of vectors: {num_vecs}")
        print(f"; Root solutions: {len(solutions)}")
        print(";")
        print("; Registers:")
        reg_prefix = "ymm" if vm == vector_machine.AVX2 else "zmm"
        for i in range(num_vecs):
            print(f";   {reg_prefix}{i}: Input/output vector {i}")
        print(f";   {reg_prefix}{num_vecs}+: Temporary registers as needed")
        print()
        print("=" * 80)
        print()

        # Collect all root-to-leaf paths across all roots, sorted by leaf cost
        all_paths = []
        for solution in solutions:
            all_paths.extend(_collect_all_paths(solution))
        all_paths.sort(key=lambda path: path[-1].cost)

        print(f"; Total paths: {len(all_paths)}")
        print()

        for i, path in enumerate(all_paths):
            leaf_cost = path[-1].cost

            print(f"; ========== SOLUTION {i + 1} of {len(all_paths)} ==========")
            print(f"; Total cost: {leaf_cost:.2f}")
            print()

            # Print initial input state (before first stage)
            if path:
                print("; Initial Input State:")
                print(_format_vector_state_as_comment(path[0].input_state, ""))
                print()

            reg_allocator = RegisterAllocator(vm, dtype, num_vecs)

            # Print each stage in the path
            for node in path:
                _print_solution_as_assembly(node, reg_allocator)

            print("=" * 80)
            print()

        sys.stdout = old_stdout

    print(f"Exported {len(solutions)} solutions to {output_path}")
