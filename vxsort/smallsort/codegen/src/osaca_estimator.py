"""OSACA-based performance estimation for bitonic sort solutions.

Generates per-solution .asm files with OSACA markers, optionally verifies
them with NASM, runs OSACA analysis, and prints a comparison table.
"""

from __future__ import annotations

import os
import platform
import re
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
from cost_model import get_supported_cpus, resolve_osaca_arch
from utils import primitive_type, vector_machine, width_dict


@dataclass
class OsacaResult:
    """Result from running OSACA on a single solution."""

    solution_index: int
    asm_path: str
    yaml_path: str
    critical_path: float
    throughput: float
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

                if reg_allocator.vm == vector_machine.AVX512:
                    load_mnemonic = "vmovdqa64"
                else:
                    load_mnemonic = "vmovdqa32"
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
        lines.append(f"    {'vmovdqa32':20s} {tmp1}, {top_reg}")
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


def _build_mnemonic_fixups(arch: str) -> dict[str, str]:
    """Build a mnemonic substitution table based on what the OSACA DB has.

    OSACA databases are inconsistent about ``vmovdqa`` vs the EVEX-suffixed
    ``vmovdqa32`` / ``vmovdqa64``:

    - Older DBs (e.g. SKX) have 2-operand forms only under ``vmovdqa``;
      their ``vmovdqa32`` entries are 3-operand masked forms we don't use.
    - Newer DBs (e.g. ZEN5) have 2-operand forms only under ``vmovdqa32``
      and lack ``vmovdqa`` entirely.

    We probe the machine model to decide which direction to normalize.
    """
    from osaca.semantics import MachineModel

    mm = MachineModel(arch=arch)

    # Check which mnemonics have usable 2-operand (reg,reg or reg,mem) forms
    has_bare_2op = False
    has_32_2op = False

    for e in mm["instruction_forms"]:
        name = e.get("name", "").lower()
        nops = len(e.get("operands", []))
        if nops == 2:
            if name == "vmovdqa":
                has_bare_2op = True
            elif name == "vmovdqa32":
                has_32_2op = True

    fixups: dict[str, str] = {}

    if has_bare_2op and not has_32_2op:
        # SKX-style: normalize suffixed → bare
        fixups["vmovdqa32"] = "vmovdqa"
        fixups["vmovdqa64"] = "vmovdqa"
    elif has_32_2op and not has_bare_2op:
        # ZEN5-style: normalize bare → suffixed
        fixups["vmovdqa"] = "vmovdqa32"

    return fixups


def _sanitize_asm_for_osaca(code: str, arch: str) -> str:
    """Pre-process NASM assembly text into forms OSACA's Intel parser accepts.

    OSACA cannot parse NASM-specific ``[rel label]`` memory operands.
    We replace them with ``[rdi]`` (register-indirect), which is
    semantically equivalent for throughput / latency analysis.  Inline
    comments are also stripped so that bracket characters inside comments
    (e.g. ``; [7, 6, 5, 4, 3, 2, 1, 0]``) don't confuse the tokenizer,
    while OSACA marker comments (``; OSACA-BEGIN`` / ``; OSACA-END``) are
    preserved.

    Mnemonic fixups (e.g. ``vmovdqa32`` → ``vmovdqa``) are applied only
    when the target arch's OSACA database lacks the original form.
    """
    fixups = _build_mnemonic_fixups(arch)

    out_lines: list[str] = []
    for line in code.splitlines():
        # Replace [rel <label>] with [rdi]
        line = re.sub(r"\[rel\s+\w+\]", "[rdi]", line)

        # Apply arch-specific mnemonic fixups
        for src, dst in fixups.items():
            line = re.sub(rf"\b{src}\b", dst, line)

        # Strip inline comments, but keep OSACA markers
        semi_pos = line.find(";")
        if semi_pos >= 0:
            comment = line[semi_pos:]
            if "OSACA-BEGIN" in comment or "OSACA-END" in comment:
                pass  # keep marker comments
            else:
                line = line[:semi_pos].rstrip()

        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _run_osaca(
    asm_path: str,
    arch: str,
    output_dir: str,
    solution_index: int,
) -> OsacaResult:
    """Run OSACA on an assembly file using the Python API.

    Computes only the critical path (CP) and port pressure — skips the
    loop-carried dependency (LCD) analysis entirely.  OSACA's LCD uses
    nx.all_simple_paths on a doubled dependency graph, which is exponential
    in the number of register fan-out paths.  For our straight-line bitonic
    sort kernels (not loop bodies) LCD is meaningless anyway.

    We bypass KernelDG.__init__ and call create_DG + dag_longest_path
    directly on the full kernel.
    """
    import networkx as nx
    from osaca.osaca import get_asm_parser
    from osaca.semantics import INSTR_FLAGS, ArchSemantics, MachineModel
    from osaca.semantics.kernel_dg import KernelDG
    from osaca.semantics.marker_utils import reduce_to_section

    output_path = os.path.join(output_dir, f"solution_{solution_index:03d}.txt")
    warnings: list[str] = []

    try:
        # Parse the assembly file, sanitizing NASM constructs OSACA can't handle
        code = Path(asm_path).read_text()
        code = _sanitize_asm_for_osaca(code, arch)
        parser = get_asm_parser(arch, "intel")
        parsed = parser.parse_file(code)

        # Reduce to marked section
        kernel = reduce_to_section(parsed, parser)

        # Build machine model and add semantics
        mm = MachineModel(arch=arch)
        sem = ArchSemantics(parser, mm)
        sem.normalize_instruction_forms(kernel)
        sem.add_semantics(kernel)

        # Bail out if any instruction is missing from OSACA's database
        unknown_mnemonics: list[str] = []
        for instr in kernel:
            if not instr.mnemonic:
                continue
            flags = getattr(instr, "flags", [])
            if INSTR_FLAGS.TP_UNKWN in flags or INSTR_FLAGS.LT_UNKWN in flags:
                unknown_mnemonics.append(instr.mnemonic)
        if unknown_mnemonics:
            unique = sorted(set(unknown_mnemonics))
            raise ValueError(
                f"OSACA {arch} database is missing {len(unique)} instruction(s): "
                + ", ".join(unique)
            )

        sem.assign_optimal_throughput(kernel)

        # Build dependency graph directly — skip LCD (exponential cost)
        kdg = object.__new__(KernelDG)
        kdg.timed_out = False
        kdg.kernel = kernel
        kdg.parser = parser
        kdg.model = mm
        kdg.arch_sem = sem
        kdg.dg = kdg.create_DG(kernel)
        kdg.loopcarried_deps = {}

        # Compute critical path via dag_longest_path on the DG
        total_cp = 0.0
        if nx.algorithms.dag.is_directed_acyclic_graph(kdg.dg):
            longest = nx.algorithms.dag.dag_longest_path(kdg.dg, weight="latency")
            for s, d in nx.utils.pairwise(longest):
                total_cp += kdg.dg.edges[(s, d)]["latency"]
            # Add latency of the last instruction in the path
            if longest:
                last_node = kdg.dg.nodes[longest[-1]]["instruction_form"]
                if last_node.latency is not None:
                    total_cp += last_node.latency

        # Compute throughput (port pressure sum)
        tp_sum = ArchSemantics.get_throughput_sum(kernel)
        tp_max = max(tp_sum) if tp_sum else 0.0

        # Write details to output file
        with open(output_path, "w") as f:
            f.write(f"OSACA analysis for {Path(asm_path).name}\n")
            f.write(f"Architecture: {arch}\n\n")
            f.write(f"Critical Path: {total_cp:.1f} cycles\n")
            f.write(f"Throughput (port bottleneck): {tp_max:.2f} cycles\n")
            f.write(f"Instructions: {len(kernel)}\n")
            f.write(f"DG nodes: {kdg.dg.number_of_nodes()}, ")
            f.write(f"edges: {kdg.dg.number_of_edges()}\n")

    except Exception as e:
        warnings.append(f"OSACA error: {e}")
        return OsacaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            yaml_path=output_path,
            critical_path=-1.0,
            throughput=-1.0,
            warnings=warnings,
        )

    return OsacaResult(
        solution_index=solution_index,
        asm_path=asm_path,
        yaml_path=output_path,
        critical_path=total_cp,
        throughput=tp_max,
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
        supported = [info.canonical for info in get_supported_cpus() if info.osaca_code]
        print(
            f"Warning: Cannot resolve OSACA architecture for '{target_cpu}'. "
            f"Supported architectures: {', '.join(supported)}",
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

    headers = ["Solution", "ASM File", "CP", "TP", "Stages", "CVs", "Our Cost"]
    rows = []

    for result, path in zip(results, paths):
        cp_str = f"{result.critical_path:.2f}" if result.critical_path >= 0 else "err"
        tp_str = f"{result.throughput:.2f}" if result.throughput >= 0 else "err"
        rows.append(
            [
                result.solution_index,
                Path(result.asm_path).name,
                cp_str,
                tp_str,
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
