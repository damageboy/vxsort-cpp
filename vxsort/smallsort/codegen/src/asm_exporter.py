from __future__ import annotations

from intrinsic_registry import get_intrinsic_registry
from utils import vector_machine, primitive_type
from utils import width_dict
from z3_avx import mm_shuffle_str, mm_shuffle2_str


def _get_instruction_metadata(intrinsic_name: str) -> dict:
    """Get metadata about an instruction for proper comment formatting.

    Delegates to the shared intrinsic registry.

    Returns:
        dict with:
        - 'imm_type': 'shuffle2', 'shuffle4', 'binary', or None
        - 'control_vector_width': element width in bits for control vector ops, None
    """
    registry = get_intrinsic_registry()
    info = registry.get(intrinsic_name)
    if info is not None:
        return {
            "imm_type": info.imm_type,
            "control_vector_width": info.control_vector_width,
        }
    return {"imm_type": None, "control_vector_width": None}


def _intrinsic_to_asm_mnemonic(intrinsic_name: str) -> str:
    """Map intrinsic name to assembly mnemonic.

    Delegates to the shared intrinsic registry.
    """
    registry = get_intrinsic_registry()
    info = registry.get(intrinsic_name)
    if info is not None:
        return info.asm_mnemonic.lower()
    return intrinsic_name


def _is_masked_intrinsic(intrinsic_name: str) -> bool:
    """Check if an intrinsic is a masked (merge-mask) variant."""
    return "_mask_" in intrinsic_name


def _get_compare_swap_mnemonics(
    dtype: primitive_type, vm: vector_machine
) -> tuple[str, str, str] | None:
    """Get (mov, min, max) assembly mnemonics for a compare-swap operation.

    Returns None for AVX2 i64/u64 which need emulation (no native min/max).

    Note: primitive_type uses Python Enum with duplicate values, so i32/u32/f32
    are aliases (same object), as are i64/u64/f64 and i16/u16. We use signed
    integer semantics as the default since that's the primary use case.
    """
    element_size = dtype.value[0]

    # AVX2 64-bit: no native min/max, needs cmpgt+blend emulation
    if vm == vector_machine.AVX2 and element_size == 8:
        return None

    # mov mnemonic depends on element size and ISA
    if vm == vector_machine.AVX512 and element_size == 8:
        mov = "vmovdqa64"
    elif vm == vector_machine.AVX512:
        mov = "vmovdqa32"
    else:
        mov = "vmovdqa"

    # min/max mnemonics by element size (signed integer semantics)
    minmax_by_size = {
        2: ("vpminsw", "vpmaxsw"),
        4: ("vpminsd", "vpmaxsd"),
        8: ("vpminsq", "vpmaxsq"),
    }

    min_mn, max_mn = minmax_by_size[element_size]
    return (mov, min_mn, max_mn)


class RegisterAllocator:
    """Manages register allocation for assembly output.

    Supports alternating register pairs to eliminate vmovdqa copies in
    compare-swap stages.  Two pairs are reserved: pair 0 = (reg0, reg1),
    pair 1 = (reg2, reg3).  Each compare-swap reads from the *source*
    pair and writes to the *destination* pair, then the caller calls
    ``swap_pairs()`` to flip roles.
    """

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

        # Reserve num_vecs * 2 registers for data (two pairs)
        self._pair_index: int = 0
        self.next_temp_reg = num_vecs * 2
        self.temp_regs_allocated = []

        # K-mask registers (k1-k7; k0 cannot be used as writemask)
        self.next_kmask_reg = 1
        self.kmask_regs_allocated = []

    # -- Pair-alternation properties --

    @property
    def src_top(self) -> str:
        """Source top register for the current pair."""
        offset = self._pair_index * self.num_vecs
        return f"{self.reg_prefix}{offset}"

    @property
    def src_bottom(self) -> str:
        """Source bottom register for the current pair."""
        offset = self._pair_index * self.num_vecs + 1
        return f"{self.reg_prefix}{offset}"

    @property
    def dst_top(self) -> str:
        """Destination top register (alternate pair)."""
        offset = (1 - self._pair_index) * self.num_vecs
        return f"{self.reg_prefix}{offset}"

    @property
    def dst_bottom(self) -> str:
        """Destination bottom register (alternate pair)."""
        offset = (1 - self._pair_index) * self.num_vecs + 1
        return f"{self.reg_prefix}{offset}"

    def swap_pairs(self):
        """Toggle which register pair is the source vs destination."""
        self._pair_index = 1 - self._pair_index

    def get_data_reg(self, vec_idx: int) -> str:
        """Get the register name for a data vector."""
        return f"{self.reg_prefix}{vec_idx}"

    def allocate_temp(self) -> str:
        """Allocate a temporary register."""
        reg = f"{self.reg_prefix}{self.next_temp_reg}"
        self.next_temp_reg += 1
        self.temp_regs_allocated.append(reg)
        return reg

    def allocate_kmask(self) -> str:
        """Allocate a k-mask register (k1-k7)."""
        if self.next_kmask_reg > 7:
            raise RuntimeError("Exhausted k-mask registers (k1-k7)")
        reg = f"k{self.next_kmask_reg}"
        self.next_kmask_reg += 1
        self.kmask_regs_allocated.append(reg)
        return reg

    def reset_temps(self):
        """Reset temporary register allocation (preserves pair state)."""
        self.next_temp_reg = self.num_vecs * 2
        self.temp_regs_allocated = []
        self.next_kmask_reg = 1
        self.kmask_regs_allocated = []


def _format_control_vector_bits(val: int, total_bits: int, element_bits: int) -> str:
    """Format a packed control vector as a bracketed list of lane values.

    Args:
        val: Packed integer containing all lane values.
        total_bits: Total width of the vector register in bits.
        element_bits: Width of each lane element in bits.
    """
    num_lanes = total_bits // element_bits
    mask = (1 << element_bits) - 1
    elements = [(val >> (i * element_bits)) & mask for i in range(num_lanes)]
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

    # Handle k-mask for masked intrinsics
    is_masked = _is_masked_intrinsic(inst.intrinsic_name)
    kmask_reg = None
    kmask_val = None
    if is_masked and "k" in args:
        kmask_reg = reg_allocator.allocate_kmask()
        kmask_val = args["k"]

    # Build operand list - Intel syntax: dest, src1, [src2], [imm]
    # For masked instructions, dest gets {kN} suffix
    dest_str = f"{dest_reg}{{{kmask_reg}}}" if kmask_reg else dest_reg
    operands = [dest_str]  # Destination is always first

    ctrl_val = None
    ctrl_element_width = None  # Track element width for control vectors

    # Handle different instruction patterns based on arguments.
    # More specific masked patterns come first, then fall through to
    # existing unmasked patterns.

    if (
        "k" in args
        and "src" in args
        and "op_idx" in args
        and "a" in args
        and "b" not in args
    ):
        # Masked single-input with control vector
        # e.g., _mm512_mask_permutexvar_epi64(src, k, op_idx, a)
        ctrl_val = args["op_idx"]
        ctrl_element_width = metadata["control_vector_width"]
        ctrl_reg = reg_allocator.allocate_temp()
        src = _resolve_source(args["a"], top_reg, bottom_reg, is_top)
        operands.append(ctrl_reg)
        operands.append(src)

    elif "k" in args and "src" in args and "a" in args and "b" in args:
        # Masked dual-input (with or without immediate)
        # e.g., _mm512_mask_shuffle_pd(src, k, a, b, imm8)
        # e.g., _mm512_mask_unpacklo_epi64(src, k, a, b)
        src1 = _resolve_source(args["a"], top_reg, bottom_reg, is_top)
        src2 = _resolve_source(args["b"], top_reg, bottom_reg, is_top)
        operands.append(src1)
        operands.append(src2)

    elif "k" in args and "src" in args and "a" in args and "b" not in args:
        # Masked single-input with immediate (no control vector, no b)
        # e.g., _mm512_mask_permute_pd(src, k, a, imm8)
        src = _resolve_source(args["a"], top_reg, bottom_reg, is_top)
        operands.append(src)

    elif (
        "k" in args
        and "a" in args
        and "op_idx" in args
        and "b" in args
        and "src" not in args
    ):
        # Masked permutex2var: a is merge source, op_idx is control, b is second input
        # e.g., _mm512_mask_permutex2var_epi64(a, k, op_idx, b)
        ctrl_val = args["op_idx"]
        ctrl_element_width = metadata["control_vector_width"]
        ctrl_reg = reg_allocator.allocate_temp()
        src2 = _resolve_source(args["b"], top_reg, bottom_reg, is_top)
        operands.append(ctrl_reg)
        operands.append(src2)

    elif "a" in args and "op_idx" in args and "b" in args and "k" not in args:
        # Unmasked permutex2var: a is first input, op_idx is control, b is second input
        # e.g., _mm512_permutex2var_epi64(a, op_idx, b)
        ctrl_val = args["op_idx"]
        ctrl_element_width = metadata["control_vector_width"]
        ctrl_reg = reg_allocator.allocate_temp()
        src2 = _resolve_source(args["b"], top_reg, bottom_reg, is_top)
        operands.append(ctrl_reg)
        operands.append(src2)

    elif "a" in args and "b" in args:
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
    comment_parts = []
    if "imm8" in args:
        imm_val = args["imm8"]
        if isinstance(imm_val, int):
            imm_type = metadata["imm_type"]
            if imm_type == "binary":
                operands.append(f"0x{imm_val:02x}")
                comment_parts.append(f"0b{imm_val:08b}")
            elif imm_type == "shuffle2":
                operands.append(f"0x{imm_val:02x}")
                comment_parts.append(mm_shuffle2_str(imm_val))
            elif imm_type == "shuffle4":
                operands.append(f"0x{imm_val:02x}")
                comment_parts.append(mm_shuffle_str(imm_val))
            else:
                operands.append(f"0x{imm_val:02x}")
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
        total_bits = width_dict[reg_allocator.vm] * 8
        if ctrl_element_width is not None:
            comment_parts.append(
                _format_control_vector_bits(ctrl_val, total_bits, ctrl_element_width)
            )
        else:
            element_bits = reg_allocator.dtype.value[0] * 8
            comment_parts.append(
                _format_control_vector_bits(ctrl_val, total_bits, element_bits)
            )

    # Add k-mask value as a comment
    if kmask_val is not None:
        if isinstance(kmask_val, int):
            comment_parts.insert(0, f"k=0x{kmask_val:02x}")
        else:
            comment_parts.insert(0, f"k={kmask_val}")

    comment = ", ".join(comment_parts) if comment_parts else None
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


def _emit_compare_swap_lines(
    reg_allocator: RegisterAllocator,
) -> list[str]:
    """Emit min/max compare-swap instructions using alternating register pairs.

    Reads from the allocator's *source* pair and writes to the *destination*
    pair, eliminating the vmovdqa copy that was previously needed.

    Does NOT call ``swap_pairs()`` — the caller is responsible for that
    (because the final natural-order stage skips compare-swap entirely).

    For types with native min/max (all except AVX2 64-bit):
        min  dst_top, src_top, src_bottom
        max  dst_bottom, src_top, src_bottom

    For AVX2 64-bit (no native min/max):
        cmpgt     tmp, src_top, src_bottom
        blendvpd  dst_top, src_bottom, src_top, tmp      (min)
        blendvpd  dst_bottom, src_top, src_bottom, tmp    (max)
    """
    lines: list[str] = []
    src_top = reg_allocator.src_top
    src_bot = reg_allocator.src_bottom
    dst_top = reg_allocator.dst_top
    dst_bot = reg_allocator.dst_bottom

    mnemonics = _get_compare_swap_mnemonics(reg_allocator.dtype, reg_allocator.vm)

    if mnemonics is not None:
        # Native min/max path — 2 instructions, no copy
        _mov, min_mn, max_mn = mnemonics
        lines.append(f"    {min_mn:20s} {dst_top}, {src_top}, {src_bot}")
        lines.append(f"    {max_mn:20s} {dst_bot}, {src_top}, {src_bot}")
    else:
        # AVX2 64-bit emulation: cmpgt + 2x blendv — 3 instructions, no copy
        tmp = reg_allocator.allocate_temp()
        lines.append(f"    {'vpcmpgtq':20s} {tmp}, {src_top}, {src_bot}")
        lines.append(f"    {'vblendvpd':20s} {dst_top}, {src_bot}, {src_top}, {tmp}")
        lines.append(f"    {'vblendvpd':20s} {dst_bot}, {src_top}, {src_bot}, {tmp}")

    return lines


def _print_solution_step_as_assembly(
    step,
    reg_allocator: RegisterAllocator,
    cumulative_cost: float = 0.0,
    indent: int = 0,
    emit_compare_swap: bool = True,
):
    """Print a single PathStep as assembly code.

    Args:
        step: PathStep with node and selected gadget_index
        reg_allocator: RegisterAllocator for register management
        cumulative_cost: Running total cost up to this step
        indent: Indentation level for nested output
        emit_compare_swap: Whether to emit min/max compare-swap after permutations.
            Set to False for the natural-order stage (pure reorder, no comparison).
    """
    prefix = "  " * indent

    # Print stage header with register mapping
    print(f"{prefix}; Stage {step.node.stage}")
    print(f"{prefix}; Cost: {cumulative_cost:.2f} (step: {step.score.total_score:.2f})")
    if step.score.control_vector_count > 0:
        print(f"{prefix}; Control vectors: {step.score.control_vector_count}")

    # Get register names from current source pair
    top_reg = reg_allocator.src_top
    bottom_reg = reg_allocator.src_bottom

    # Show register mapping: which register is top/bottom for this stage
    print(
        f"{prefix}; Registers: {top_reg} = top, {bottom_reg} = bottom"
        f"  →  {reg_allocator.dst_top} = top, {reg_allocator.dst_bottom} = bottom"
    )

    # Use the explicitly selected gadget from the step
    gadget = step.gadget

    # Print top vector instructions (gadget operates in-place on source pair)
    if gadget.top_instructions:
        print(f"{prefix}; Top vector ({top_reg}) operations:")
        for inst in gadget.top_instructions:
            asm_line = _format_instruction(
                inst, reg_allocator, top_reg, bottom_reg, is_top=True
            )
            print(f"{prefix}{asm_line}")

    # Print bottom vector instructions
    if gadget.bottom_instructions:
        print(f"{prefix}; Bottom vector ({bottom_reg}) operations:")
        for inst in gadget.bottom_instructions:
            asm_line = _format_instruction(
                inst, reg_allocator, top_reg, bottom_reg, is_top=False
            )
            print(f"{prefix}{asm_line}")

    # Emit compare-swap (min/max) after permutation instructions
    if emit_compare_swap:
        cs_lines = _emit_compare_swap_lines(reg_allocator)
        print(
            f"{prefix}; Compare-swap: "
            f"min → {reg_allocator.dst_top} (top), "
            f"max → {reg_allocator.dst_bottom} (bottom)"
        )
        for line in cs_lines:
            print(f"{prefix}{line}")
        reg_allocator.swap_pairs()

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
        path_selector = PathSelector(cost_model)

        # Select paths with a large limit to get "all" paths, capped at _MAX_ASM_PATHS
        all_paths = path_selector.select_top_k_paths(solutions, _MAX_ASM_PATHS)
        total_path_count = len(all_paths)
        truncated = total_path_count >= _MAX_ASM_PATHS

    with open(output_path, "w") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            print("; Bitonic Sort Assembly Output (with min/max compare-swap)")
            print(f"; Architecture: {vm.name}")
            print(f"; Data Type: {dtype.name}")
            print(f"; Number of vectors: {num_vecs}")
            print(f"; Natural order: {'yes' if natural_order else 'no'}")
            print(f"; Root solutions: {len(solutions)}")
            if truncated:
                print(f"; NOTE: Showing {len(all_paths)} paths (may be capped)")
            print(";")
            print(
                "; Registers (alternating pairs — roles swap after each compare-swap):"
            )
            reg_prefix = "ymm" if vm == vector_machine.AVX2 else "zmm"
            vec_labels = ["top", "bottom"] + [f"vec{i}" for i in range(2, num_vecs)]
            for i in range(num_vecs):
                label = vec_labels[i] if i < len(vec_labels) else f"vec{i}"
                print(f";   {reg_prefix}{i}: Pair 0 — {label}")
            for i in range(num_vecs):
                label = vec_labels[i] if i < len(vec_labels) else f"vec{i}"
                print(f";   {reg_prefix}{num_vecs + i}: Pair 1 — {label}")
            print(f";   {reg_prefix}{num_vecs * 2}+: Temporary registers as needed")
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
                last_idx = len(path.steps) - 1
                for step_idx, step in enumerate(path.steps):
                    running_cost += step.score.total_score
                    # Skip compare-swap on the final natural-order stage
                    # (it's a pure reorder, not a comparison stage)
                    is_natural_order_step = natural_order and step_idx == last_idx
                    _print_solution_step_as_assembly(
                        step,
                        reg_allocator,
                        cumulative_cost=running_cost,
                        emit_compare_swap=not is_natural_order_step,
                    )

                # If results ended up in pair 1, move back to pair 0
                if reg_allocator._pair_index != 0:
                    mnemonics = _get_compare_swap_mnemonics(dtype, vm)
                    mov = mnemonics[0] if mnemonics else "vmovdqa"
                    src_t = reg_allocator.src_top
                    src_b = reg_allocator.src_bottom
                    dst_t = reg_allocator.dst_top
                    dst_b = reg_allocator.dst_bottom
                    print("; Move results back to pair 0:")
                    print(f"    {mov:20s} {dst_t}, {src_t}")
                    print(f"    {mov:20s} {dst_b}, {src_b}")
                    print()

                print("=" * 80)
                print()
        finally:
            sys.stdout = old_stdout

    print(f"Exported {len(all_paths)} solutions to {output_path}")
