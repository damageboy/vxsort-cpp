from __future__ import annotations

from intrinsic_registry import get_intrinsic_registry
from utils import vector_machine, primitive_type


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


def export_solutions_to_asm(
    solutions,
    num_vecs: int,
    dtype: primitive_type,
    vm: vector_machine,
    output_path: str,
    selected_paths=None,
    natural_order: bool = False,
    nasm_path: str | None = None,
):
    """Export solutions as NASM-valid assembly code.

    Uses the canonical emitter from osaca_estimator (generate_solution_asm)
    for each solution, optionally verifies each with NASM, then concatenates
    all solutions into a single output file with separators.

    Args:
        solutions: List of root SolutionNode objects (may be pruned)
        num_vecs: Number of SIMD vectors
        dtype: Primitive data type
        vm: Vector machine (AVX2/AVX512)
        output_path: Path to write assembly output
        selected_paths: Optional list of CompletePath from PathSelector.
                       If provided, uses these explicit paths instead of
                       enumerating all paths through the DAG.
        natural_order: Whether the final stage is a natural-order reorder
        nasm_path: Optional path to nasm binary for per-solution verification
    """
    import tempfile

    from cost_model import CostModel
    from perf_estimator import generate_solution_asm, _verify_with_nasm
    from path_selector import PathSelector

    # If selected_paths is provided, use them directly
    if selected_paths is not None:
        all_paths = selected_paths
    else:
        print(
            "Warning: No selected_paths provided, enumerating all paths (may be slow)"
        )
        cost_model = CostModel("generic")
        path_selector = PathSelector(cost_model)
        all_paths = path_selector.select_top_k_paths(solutions, 10_000)

    total = len(all_paths)
    nasm_failures = 0

    with open(output_path, "w") as f:
        # File header
        f.write("; Bitonic Sort Assembly Output — NASM-valid x86-64\n")
        f.write(f"; Architecture: {vm.name}\n")
        f.write(f"; Data Type: {dtype.name}\n")
        f.write(f"; Number of vectors: {num_vecs}\n")
        f.write(f"; Natural order: {'yes' if natural_order else 'no'}\n")
        f.write(f"; Total solutions: {total}\n")
        f.write("\n")

        # Create temp dir for NASM verification
        tmp_dir = tempfile.mkdtemp(prefix="vxsort_asm_") if nasm_path else None

        for i, path in enumerate(all_paths):
            solution_index = i + 1

            # Generate NASM-valid assembly for this solution
            asm_content = generate_solution_asm(
                path, vm, dtype, num_vecs, natural_order, solution_index, total
            )

            # Optionally verify with NASM before including
            if nasm_path is not None and tmp_dir is not None:
                tmp_path = f"{tmp_dir}/solution_{solution_index:03d}.asm"
                with open(tmp_path, "w") as tmp_f:
                    tmp_f.write(asm_content)
                if not _verify_with_nasm(tmp_path, nasm_path):
                    nasm_failures += 1

            # Write separator + solution
            f.write(f"; {'=' * 78}\n")
            f.write(f"; SOLUTION {solution_index} of {total}\n")
            f.write(f"; {'=' * 78}\n\n")
            f.write(asm_content)
            f.write("\n")

    if nasm_failures:
        print(f"Warning: {nasm_failures} of {total} solutions failed NASM verification")
    print(f"Exported {total} solutions to {output_path}")
