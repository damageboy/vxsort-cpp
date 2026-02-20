"""OSACA-based performance estimation for bitonic sort solutions.

Generates per-solution .asm files with OSACA markers, optionally verifies
them with NASM, runs OSACA analysis, and prints a comparison table.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from tabulate import tabulate

from asm_exporter import (
    RegisterAllocator,
    _format_control_vector_bits,
    _get_compare_swap_mnemonics,
    _get_instruction_metadata,
    _intrinsic_to_asm_mnemonic,
    _is_masked_intrinsic,
    _resolve_source,
)
from cost_model import resolve_arch_name
from utils import primitive_type, vector_machine, width_dict

# Maps canonical arch names (from cost_model.resolve_arch_name) to OSACA arch codes.
_OSACA_ARCH_MAP: dict[str, str] = {
    "SNB": "SNB",
    "IVB": "IVB",
    "HSW": "HSW",
    "BDW": "BDW",
    "SKL": "SKX",
    "KBL": "SKX",
    "CFL": "SKX",  # client Skylake -> SKX
    "CNL": "ICL",
    "ICL": "ICL",
    "TGL": "ICL",
    "RKL": "ICL",
    "SKX": "SKX",
    "CLX": "CSX",  # OSACA calls Cascade Lake "CSX"
    "ADL-P": "SPR",
    "ADL-E": "SPR",
    "ZEN+": "ZEN1",
    "ZEN1": "ZEN1",
    "ZEN2": "ZEN2",
    "ZEN3": "ZEN3",
    "ZEN4": "ZEN4",
    "ZEN5": "ZEN5",
}


def resolve_osaca_arch(target_cpu: str) -> str | None:
    """Resolve a --target-cpu value to an OSACA architecture code.

    Returns None if the architecture is not supported by OSACA.
    """
    canonical = resolve_arch_name(target_cpu)
    if canonical is None:
        return None
    return _OSACA_ARCH_MAP.get(canonical)


@dataclass
class OsacaResult:
    """Result from running OSACA on a single solution."""

    solution_index: int
    asm_path: str
    yaml_path: str
    critical_path: float
    lcd: float
    warnings: list[str] = field(default_factory=list)


def _control_vector_elements(val: int, total_bits: int, element_bits: int) -> list[int]:
    """Unpack a packed control vector into individual lane values."""
    num_lanes = total_bits // element_bits
    mask = (1 << element_bits) - 1
    return [(val >> (i * element_bits)) & mask for i in range(num_lanes)]


def _format_cv_as_data_directive(val: int, total_bits: int, element_bits: int) -> str:
    """Format a control vector as a NASM data directive (dd/dq)."""
    elements = _control_vector_elements(val, total_bits, element_bits)
    if element_bits <= 32:
        directive = "dd"
    else:
        directive = "dq"
    return f"    {directive} " + ", ".join(str(e) for e in elements)


def _emit_gadget_asm(
    gadget,
    reg_allocator: RegisterAllocator,
    top_reg: str,
    bottom_reg: str | None,
    stage_idx: int,
    rodata_entries: list[tuple[str, str]],
) -> list[str]:
    """Emit assembly lines for a gadget's instructions, including CV/k-mask loads.

    Returns a list of assembly lines. Appends (label, directive) pairs to
    rodata_entries for any control vectors that need .rodata definitions.
    """
    lines: list[str] = []
    total_bits = width_dict[reg_allocator.vm] * 8
    cv_counter = [0]  # mutable counter for closure

    def _emit_instructions(instructions, is_top: bool, side_label: str):
        dest_reg = top_reg if is_top else bottom_reg
        for inst in instructions:
            mnemonic = _intrinsic_to_asm_mnemonic(inst.intrinsic_name)
            metadata = _get_instruction_metadata(inst.intrinsic_name)
            args = inst.args

            # Collect control vector value and element width if present
            ctrl_val = None
            ctrl_element_width = None
            kmask_val = None
            kmask_reg = None

            # Detect k-mask
            is_masked = _is_masked_intrinsic(inst.intrinsic_name)
            if is_masked and "k" in args:
                kmask_val = args["k"]
                kmask_reg = reg_allocator.allocate_kmask()

            # Detect control vector from various arg patterns
            if "op_idx" in args:
                ctrl_val = args["op_idx"]
                ctrl_element_width = metadata["control_vector_width"]
            elif "b" in args and args["b"] not in ("top", "bottom", "prev"):
                # 'b' is a control vector value, not a register
                ctrl_val = args["b"]
                ctrl_element_width = metadata["control_vector_width"]
            elif (
                "mask" in args and isinstance(args["mask"], int) and args["mask"] > 0xFF
            ):
                ctrl_val = args["mask"]
                ctrl_element_width = None  # use dtype default

            # Emit k-mask load if needed
            if kmask_val is not None and kmask_reg is not None:
                if isinstance(kmask_val, int):
                    if kmask_val <= 0xFFFFFFFF:
                        lines.append(f"    {'mov':20s} eax, 0x{kmask_val:x}")
                    else:
                        lines.append(f"    {'mov':20s} rax, 0x{kmask_val:x}")
                    lines.append(f"    {'kmovw':20s} {kmask_reg}, eax")

            # Emit control vector load if needed
            ctrl_reg = None
            if ctrl_val is not None:
                ctrl_reg = reg_allocator.allocate_temp()
                cv_label = f"cv_s{stage_idx}_{side_label[0]}{cv_counter[0]}"
                cv_counter[0] += 1

                element_bits = ctrl_element_width or (reg_allocator.dtype.value[0] * 8)
                data_directive = _format_cv_as_data_directive(
                    ctrl_val, total_bits, element_bits
                )
                rodata_entries.append((cv_label, data_directive))

                load_mnemonic = (
                    "vmovdqa"
                    if reg_allocator.vm == vector_machine.AVX2
                    else "vmovdqa32"
                )
                lines.append(f"    {load_mnemonic:20s} {ctrl_reg}, [rel {cv_label}]")
                # Add comment showing CV contents
                comment = _format_control_vector_bits(
                    ctrl_val, total_bits, element_bits
                )
                lines[-1] += f"  ; {comment}"

            # Build operand list
            dest_str = f"{dest_reg}{{{kmask_reg}}}" if kmask_reg else dest_reg
            operands: list[str] = [dest_str]

            # Build the rest of operands based on arg patterns
            # (mirrors _format_instruction logic from asm_exporter)
            if (
                "k" in args
                and "src" in args
                and "op_idx" in args
                and "a" in args
                and "b" not in args
            ):
                src = _resolve_source(args["a"], top_reg, bottom_reg or top_reg, is_top)
                assert ctrl_reg is not None
                operands.append(ctrl_reg)
                operands.append(src)

            elif "k" in args and "src" in args and "a" in args and "b" in args:
                src1 = _resolve_source(
                    args["a"], top_reg, bottom_reg or top_reg, is_top
                )
                src2 = _resolve_source(
                    args["b"], top_reg, bottom_reg or top_reg, is_top
                )
                operands.append(src1)
                operands.append(src2)

            elif "k" in args and "src" in args and "a" in args and "b" not in args:
                src = _resolve_source(args["a"], top_reg, bottom_reg or top_reg, is_top)
                operands.append(src)

            elif (
                "k" in args
                and "a" in args
                and "op_idx" in args
                and "b" in args
                and "src" not in args
            ):
                src2 = _resolve_source(
                    args["b"], top_reg, bottom_reg or top_reg, is_top
                )
                assert ctrl_reg is not None
                operands.append(ctrl_reg)
                operands.append(src2)

            elif "a" in args and "op_idx" in args and "b" in args and "k" not in args:
                src2 = _resolve_source(
                    args["b"], top_reg, bottom_reg or top_reg, is_top
                )
                assert ctrl_reg is not None
                operands.append(ctrl_reg)
                operands.append(src2)

            elif "a" in args and "b" in args:
                src1 = _resolve_source(
                    args["a"], top_reg, bottom_reg or top_reg, is_top
                )
                if args["b"] not in ("top", "bottom", "prev"):
                    # 'b' is a control vector
                    assert ctrl_reg is not None
                    operands.append(src1)
                    operands.append(ctrl_reg)
                else:
                    src2 = _resolve_source(
                        args["b"], top_reg, bottom_reg or top_reg, is_top
                    )
                    operands.append(src1)
                    operands.append(src2)

            elif "a" in args:
                src = _resolve_source(args["a"], top_reg, bottom_reg or top_reg, is_top)
                if "op_idx" in args:
                    assert ctrl_reg is not None
                    operands.append(ctrl_reg)
                    operands.append(src)
                else:
                    operands.append(src)

            elif "input" in args:
                src = _resolve_source(
                    args["input"], top_reg, bottom_reg or top_reg, is_top
                )
                operands.append(src)

            # Immediate values
            if "imm8" in args:
                imm_val = args["imm8"]
                if isinstance(imm_val, int):
                    operands.append(f"0x{imm_val:02x}")
            elif "imm" in args:
                imm_val = args["imm"]
                if isinstance(imm_val, int):
                    operands.append(f"0x{imm_val:x}")
            elif "mask" in args:
                mask_val = args["mask"]
                if isinstance(mask_val, int) and mask_val <= 0xFF:
                    operands.append(f"0x{mask_val:02x}")
                elif ctrl_reg is not None:
                    operands.append(ctrl_reg)

            lines.append(f"    {mnemonic:20s} {', '.join(operands)}")

    # Emit top instructions
    if gadget.top_instructions:
        _emit_instructions(gadget.top_instructions, is_top=True, side_label="top")

    # Emit bottom instructions
    if gadget.bottom_instructions and bottom_reg:
        _emit_instructions(gadget.bottom_instructions, is_top=False, side_label="bot")

    return lines


def _emit_compare_swap_lines(
    reg_allocator: RegisterAllocator,
    top_reg: str,
    bottom_reg: str,
) -> list[str]:
    """Emit compare-swap (min/max) instructions, returning lines."""
    lines: list[str] = []
    mnemonics = _get_compare_swap_mnemonics(reg_allocator.dtype, reg_allocator.vm)

    if mnemonics is not None:
        mov, min_mn, max_mn = mnemonics
        tmp = reg_allocator.allocate_temp()
        lines.append(f"    {mov:20s} {tmp}, {top_reg}")
        lines.append(f"    {min_mn:20s} {top_reg}, {top_reg}, {bottom_reg}")
        lines.append(f"    {max_mn:20s} {bottom_reg}, {bottom_reg}, {tmp}")
    else:
        # AVX2 64-bit emulation: cmpgt + blendv
        tmp1 = reg_allocator.allocate_temp()
        tmp2 = reg_allocator.allocate_temp()
        lines.append(f"    {'vmovdqa':20s} {tmp1}, {top_reg}")
        lines.append(f"    {'vpcmpgtq':20s} {tmp2}, {top_reg}, {bottom_reg}")
        lines.append(
            f"    {'vblendvpd':20s} {top_reg}, {top_reg}, {bottom_reg}, {tmp2}"
        )
        lines.append(
            f"    {'vblendvpd':20s} {bottom_reg}, {bottom_reg}, {tmp1}, {tmp2}"
        )

    return lines


def _generate_osaca_asm(
    path,
    vm: vector_machine,
    dtype: primitive_type,
    num_vecs: int,
    natural_order: bool,
    solution_index: int,
    total_solutions: int,
) -> str:
    """Generate a complete, NASM-valid .asm file for OSACA analysis.

    Args:
        path: CompletePath with steps
        vm: Vector machine (AVX2/AVX512)
        dtype: Primitive data type
        num_vecs: Number of SIMD vectors
        natural_order: Whether the final stage is a natural-order reorder
        solution_index: 1-based solution index
        total_solutions: Total number of solutions

    Returns:
        Complete .asm file content as a string.
    """
    header_lines = [
        f"; OSACA bitonic sort solution -- {vm.name} {dtype.name} {num_vecs}-vec, "
        f"solution {solution_index} of {total_solutions}",
        "bits 64",
        "default rel",
        "",
    ]

    # Collect .rodata entries as we emit instructions
    rodata_entries: list[tuple[str, str]] = []

    # Build .text section
    text_lines = [
        "section .text",
        "global _osaca_kernel",
        "_osaca_kernel:",
        "    ; OSACA-BEGIN",
    ]

    reg_allocator = RegisterAllocator(vm, dtype, num_vecs)
    top_reg = reg_allocator.get_data_reg(0)
    bottom_reg = reg_allocator.get_data_reg(1) if num_vecs > 1 else None

    last_idx = len(path.steps) - 1
    for step_idx, step in enumerate(path.steps):
        text_lines.append(f"    ; -- Stage {step.node.stage} --")

        gadget = step.gadget
        gadget_lines = _emit_gadget_asm(
            gadget,
            reg_allocator,
            top_reg,
            bottom_reg,
            stage_idx=step_idx,
            rodata_entries=rodata_entries,
        )
        text_lines.extend(gadget_lines)

        # Compare-swap (skip on final natural-order stage)
        is_natural_order_step = natural_order and step_idx == last_idx
        if not is_natural_order_step and num_vecs > 1 and bottom_reg:
            cs_lines = _emit_compare_swap_lines(reg_allocator, top_reg, bottom_reg)
            text_lines.extend(cs_lines)

        reg_allocator.reset_temps()

    text_lines.append("    ; OSACA-END")
    text_lines.append("    ret")
    text_lines.append("")

    # Build .rodata section
    rodata_lines = []
    if rodata_entries:
        align_bytes = width_dict[vm]
        rodata_lines.append("section .rodata")
        for label, directive in rodata_entries:
            rodata_lines.append(f"align {align_bytes}")
            rodata_lines.append(f"{label}:")
            rodata_lines.append(directive)
        rodata_lines.append("")

    # Assemble the file: header, rodata, text
    all_lines = header_lines + rodata_lines + text_lines
    return "\n".join(all_lines) + "\n"


def _write_asm_file(asm_content: str, output_dir: str, solution_index: int) -> str:
    """Write assembly content to a file in the output directory.

    Returns the file path.
    """
    filename = f"solution_{solution_index:03d}.asm"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write(asm_content)
    return filepath


def _verify_with_nasm(asm_path: str, nasm_path: str) -> bool:
    """Verify assembly file with NASM.

    Returns True if assembly is valid, False otherwise.
    """
    obj_format = "macho64" if platform.system() == "Darwin" else "elf64"
    obj_path = asm_path + ".o"
    try:
        result = subprocess.run(
            [nasm_path, "-f", obj_format, "-o", obj_path, asm_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(
                f"  NASM warning: {Path(asm_path).name} failed: {result.stderr.strip()}",
                file=sys.stderr,
            )
            return False
        return True
    except FileNotFoundError:
        print(f"  NASM not found at '{nasm_path}'", file=sys.stderr)
        return False
    finally:
        if os.path.exists(obj_path):
            os.unlink(obj_path)


def _split_kernel_by_stages(kernel) -> list[list]:
    """Split a parsed OSACA kernel into per-stage chunks.

    Splits at comment lines containing "-- Stage N --". Each chunk includes
    all instructions from one stage comment to the next (exclusive).
    """
    stages: list[list] = []
    current: list = []

    for line in kernel:
        if (
            line.comment is not None
            and line.mnemonic is None
            and "-- Stage " in line.comment
        ):
            if current:
                stages.append(current)
            current = []
        else:
            current.append(line)

    if current:
        stages.append(current)
    return stages


def _run_osaca(
    asm_path: str,
    arch: str,
    output_dir: str,
    solution_index: int,
) -> OsacaResult:
    """Run OSACA on an assembly file using the Python API.

    Analyzes each stage independently to avoid OSACA's KernelDG hanging
    on large straight-line kernels (exponential in dependency graph size).
    Sums per-stage critical paths and takes the max LCD.

    Returns an OsacaResult with critical path and LCD values.
    """
    from osaca.osaca import get_asm_parser
    from osaca.semantics import ArchSemantics, KernelDG, MachineModel
    from osaca.semantics.marker_utils import reduce_to_section

    output_path = os.path.join(output_dir, f"solution_{solution_index:03d}.txt")
    warnings: list[str] = []

    try:
        # Parse the assembly file
        code = Path(asm_path).read_text()
        parser = get_asm_parser(arch, "intel")
        parsed = parser.parse_file(code)

        # Reduce to marked section
        kernel = reduce_to_section(parsed, parser)

        # Build machine model and add semantics
        mm = MachineModel(arch=arch)
        sem = ArchSemantics(parser, mm)
        sem.normalize_instruction_forms(kernel)
        sem.add_semantics(kernel)
        sem.assign_optimal_throughput(kernel)

        # Split into per-stage chunks and analyze each independently
        stages = _split_kernel_by_stages(kernel)
        if not stages:
            warnings.append("No stages found in kernel")
            stages = [kernel]

        total_cp = 0.0
        max_lcd = 0.0
        stage_details: list[str] = []

        for stage_idx, stage_kernel in enumerate(stages):
            if not stage_kernel:
                continue

            kdg = KernelDG(stage_kernel, parser, mm, sem, timeout=30)

            # Extract critical path length for this stage
            cp_chain = kdg.get_critical_path()
            stage_cp = 0.0
            if cp_chain:
                for instr in cp_chain:
                    if instr.latency is not None:
                        stage_cp += instr.latency
            total_cp += stage_cp

            # Extract LCD for this stage
            lcd_deps = kdg.get_loopcarried_dependencies()
            stage_lcd = 0.0
            if lcd_deps:
                for dep in lcd_deps:
                    if hasattr(dep, "latency") and dep.latency is not None:
                        stage_lcd = max(stage_lcd, dep.latency)
            max_lcd = max(max_lcd, stage_lcd)

            stage_details.append(
                f"Stage {stage_idx}: CP={stage_cp:.1f}, LCD={stage_lcd:.1f}, "
                f"instrs={len(stage_kernel)}"
            )

        # Write per-stage details to output file
        with open(output_path, "w") as f:
            f.write(f"OSACA per-stage analysis for {Path(asm_path).name}\n")
            f.write(f"Architecture: {arch}\n\n")
            for detail in stage_details:
                f.write(detail + "\n")
            f.write(f"\nTotal CP: {total_cp:.1f}\n")
            f.write(f"Max LCD: {max_lcd:.1f}\n")

    except Exception as e:
        warnings.append(f"OSACA error: {e}")
        return OsacaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            yaml_path=output_path,
            critical_path=-1.0,
            lcd=-1.0,
            warnings=warnings,
        )

    return OsacaResult(
        solution_index=solution_index,
        asm_path=asm_path,
        yaml_path=output_path,
        critical_path=total_cp,
        lcd=max_lcd,
        warnings=warnings,
    )


def estimate_solutions(
    paths,
    vm: vector_machine,
    dtype: primitive_type,
    num_vecs: int,
    natural_order: bool,
    target_cpu: str,
    nasm_path: str | None = None,
) -> list[OsacaResult]:
    """Run OSACA estimation on a list of CompletePath solutions.

    Args:
        paths: List of CompletePath objects
        vm: Vector machine (AVX2/AVX512)
        dtype: Primitive data type
        num_vecs: Number of SIMD vectors
        natural_order: Whether the final stage is natural-order
        target_cpu: Target CPU string (same as --target-cpu)
        nasm_path: Optional path to nasm binary for verification

    Returns:
        List of OsacaResult objects, one per path.
    """
    arch = resolve_osaca_arch(target_cpu)
    if arch is None:
        print(
            f"Warning: Cannot resolve OSACA architecture for '{target_cpu}'. "
            f"Supported architectures: {', '.join(sorted(_OSACA_ARCH_MAP.keys()))}",
            file=sys.stderr,
        )
        return []

    # Create temp directory
    output_dir = tempfile.mkdtemp(prefix="vxsort_osaca_", dir="/tmp")
    print(f"OSACA estimation: arch={arch}, output_dir={output_dir}")

    results: list[OsacaResult] = []
    total = len(paths)

    for i, path in enumerate(paths):
        solution_index = i + 1

        # Generate assembly
        asm_content = _generate_osaca_asm(
            path,
            vm,
            dtype,
            num_vecs,
            natural_order,
            solution_index,
            total,
        )
        asm_path = _write_asm_file(asm_content, output_dir, solution_index)

        # Optionally verify with NASM
        if nasm_path is not None:
            _verify_with_nasm(asm_path, nasm_path)

        # Run OSACA
        result = _run_osaca(asm_path, arch, output_dir, solution_index)
        results.append(result)

    return results


def print_estimation_table(results: list[OsacaResult], paths) -> None:
    """Print a table comparing OSACA results across solutions."""
    if not results:
        print("No OSACA results to display.")
        return

    headers = ["Solution", "ASM File", "CP", "LCD", "Stages", "CVs", "Our Cost"]
    rows = []

    for result, path in zip(results, paths):
        cp_str = f"{result.critical_path:.2f}" if result.critical_path >= 0 else "err"
        lcd_str = f"{result.lcd:.2f}" if result.lcd >= 0 else "err"
        rows.append(
            [
                result.solution_index,
                Path(result.asm_path).name,
                cp_str,
                lcd_str,
                len(path.steps),
                path.total_cv_count,
                f"{path.total_score:.2f}",
            ]
        )

    print()
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # Print any warnings
    any_warnings = False
    for result in results:
        if result.warnings:
            if not any_warnings:
                print("\nWarnings:")
                any_warnings = True
            for w in result.warnings:
                print(f"  Solution {result.solution_index}: {w}")
