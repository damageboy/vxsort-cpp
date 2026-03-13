"""Performance estimation for bitonic sort solutions.

Generates per-solution .asm files with analysis markers, optionally verifies
them with NASM, runs performance analysis, and prints a comparison table.
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
    _emit_compare_swap_lines,
    _format_control_vector_bits,
    _format_vector_state_as_comment,
    _get_instruction_metadata,
    _intrinsic_to_asm_mnemonic,
    _is_masked_intrinsic,
    _resolve_source,
)
from z3_avx import mm_shuffle2_str, mm_shuffle_str
from cost_model import get_supported_cpus
from utils import primitive_type, vector_machine, width_dict


@dataclass
class EstimationResult:
    """Result from performance estimation on a single solution."""

    solution_index: int
    asm_path: str
    throughput: float
    simulated_cycles: float = -1.0
    analysis_path: str = ""
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
    dst_top_reg: str,
    dst_bottom_reg: str | None,
    stage_idx: int,
    rodata_entries: list[tuple[str, str]],
    cv_cache: dict[str, str] | None = None,
) -> tuple[list[str], bool]:
    """Emit assembly lines for a gadget's instructions, including CV/k-mask loads.

    Returns (lines, is_dual_side):
      - lines: assembly lines
      - is_dual_side: True when both sides have instructions and the gadget
        writes to the destination register pair (caller must adjust COEX flow)

    For dual-side gadgets, each side writes to its corresponding dst register
    instead of modifying the src register in-place.  This eliminates the
    cross-side save (vmovdqa copy) because source registers are never modified.

    When both sides share a common instruction prefix (detected via
    unified_instruction_map), the shared instructions are emitted once to
    temp registers, then each side emits only its unique suffix.

    Appends (label, directive) pairs to rodata_entries for control vectors.
    cv_cache deduplicates identical control vector entries across stages.
    """
    lines: list[str] = []
    total_bits = width_dict[reg_allocator.vm] * 8
    cv_counter = [0]  # mutable counter for closure
    if cv_cache is None:
        cv_cache = {}
    # Register-level CV cache: if the same ctrl_val is already in a register
    # (from an earlier instruction or the other side), reuse it instead of
    # allocating a fresh temp and emitting another load.
    cv_reg_cache: dict[int, str] = {}

    # Detect dual-side: both top and bottom have instructions.
    # In dual-side mode, gadget writes to dst pair so src is never modified.
    is_dual_side = (
        bool(gadget.top_instructions)
        and bool(gadget.bottom_instructions)
        and bottom_reg is not None
    )

    # Detect shared instruction prefix for deduplication.
    # When both sides share instructions at the same chain positions
    # (same signature per unified_instruction_map), emit them once.
    prefix_len = 0
    if is_dual_side:
        _, top_indices, bottom_indices, _, _ = gadget.unified_instruction_map()
        for i in range(min(len(top_indices), len(bottom_indices))):
            if top_indices[i] == bottom_indices[i]:
                prefix_len = i + 1
            else:
                break

    def _emit_instructions(
        instructions,
        is_top: bool,
        side_label: str,
        *,
        shared_mode: bool = False,
        skip_prefix: int = 0,
        precomputed_results: dict[int, str] | None = None,
    ) -> dict[int, str]:
        """Emit instructions for one chain.

        Args:
            shared_mode: Each instruction writes to a fresh temp register.
                Returns {inst_idx: register} for all emitted instructions.
            skip_prefix: Skip the first N instructions (already emitted shared).
            precomputed_results: Maps inst_idx → register for pre-emitted results.

        Returns:
            In shared_mode: {inst_idx: register} for emitted instructions.
            Otherwise: empty dict.
        """
        source_reg = top_reg if is_top else bottom_reg
        if shared_mode:
            dest_reg = None  # set per-instruction below
        elif is_dual_side:
            dest_reg = dst_top_reg if is_top else dst_bottom_reg
        else:
            dest_reg = source_reg  # in-place

        result_re = re.compile(r"^result_(\d+)$")

        # Track what "prev" resolves to (updated after each emitted instruction).
        # When resuming after a shared prefix, "prev" initially points to the
        # last shared result so the first suffix instruction chains correctly.
        prev_target = dest_reg
        if (
            skip_prefix > 0
            and precomputed_results
            and (skip_prefix - 1) in precomputed_results
        ):
            prev_target = precomputed_results[skip_prefix - 1]

        shared_results: dict[int, str] = {}

        # Pre-scan for input/result saves (only suffix instructions need this).
        # In shared_mode, each instruction writes to its own temp, so no saves.
        needs_input_save = False
        needed_result_saves: set[int] = set()
        if not shared_mode:
            for inst_idx_s, inst in enumerate(instructions):
                if inst_idx_s < skip_prefix:
                    continue
                for v in inst.args.values():
                    if isinstance(v, str):
                        if v == "input":
                            needs_input_save = True
                        m = result_re.match(v)
                        if m:
                            idx = int(m.group(1))
                            if not (precomputed_results and idx in precomputed_results):
                                needed_result_saves.add(idx)

        mov_mnemonic = (
            "vmovdqa64" if reg_allocator.vm == vector_machine.AVX512 else "vmovdqa32"
        )

        input_save_reg = None
        if needs_input_save:
            if is_dual_side:
                # Source register is never overwritten → "input" = source_reg
                input_save_reg = source_reg
            else:
                input_save_reg = reg_allocator.allocate_temp()
                lines.append(
                    f"    {mov_mnemonic:20s} {input_save_reg}, {dest_reg}"
                    f"  ; save input"
                )

        result_save_regs: dict[int, str] = {}

        def resolve(name: str) -> str:
            """Resolve source name including multi-instruction refs."""
            if name == "input":
                assert input_save_reg is not None
                return input_save_reg
            m = result_re.match(name)
            if m:
                idx = int(m.group(1))
                if precomputed_results and idx in precomputed_results:
                    return precomputed_results[idx]
                if idx in shared_results:
                    return shared_results[idx]
                assert idx in result_save_regs, f"result_{idx} not yet computed"
                return result_save_regs[idx]
            if name == "prev":
                return prev_target
            # "top" / "bottom" always resolve to src registers (unmodified)
            return _resolve_source(name, top_reg, bottom_reg or top_reg, is_top)

        for inst_idx, inst in enumerate(instructions):
            if inst_idx < skip_prefix:
                continue

            if shared_mode:
                dest_reg = reg_allocator.allocate_temp()
                shared_results[inst_idx] = dest_reg

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

            # Detect control vector from various arg patterns.
            # Use isinstance(int) to distinguish numeric control vectors
            # from string source references ("input", "result_N", etc.)
            if "op_idx" in args:
                ctrl_val = args["op_idx"]
                ctrl_element_width = metadata["control_vector_width"]
            elif "b" in args and isinstance(args["b"], int):
                # 'b' is a numeric control vector, not a register reference
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

            # Emit control vector load if needed, reusing a register if the
            # same value was already loaded earlier in this gadget.
            ctrl_reg = None
            if ctrl_val is not None:
                if ctrl_val in cv_reg_cache:
                    # Already loaded into a register — reuse it
                    ctrl_reg = cv_reg_cache[ctrl_val]
                else:
                    ctrl_reg = reg_allocator.allocate_temp()
                    cv_reg_cache[ctrl_val] = ctrl_reg

                    element_bits = ctrl_element_width or (
                        reg_allocator.dtype.value[0] * 8
                    )
                    data_directive = _format_cv_as_data_directive(
                        ctrl_val, total_bits, element_bits
                    )

                    # Deduplicate .rodata: reuse an existing label if the
                    # same directive was already emitted.
                    if data_directive in cv_cache:
                        cv_label = cv_cache[data_directive]
                    else:
                        cv_label = f"cv_s{stage_idx}_{side_label[0]}{cv_counter[0]}"
                        cv_counter[0] += 1
                        cv_cache[data_directive] = cv_label
                        rodata_entries.append((cv_label, data_directive))

                    lines.append(f"    {mov_mnemonic:20s} {ctrl_reg}, [rel {cv_label}]")
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
                src = resolve(args["a"])
                assert ctrl_reg is not None
                operands.append(ctrl_reg)
                operands.append(src)

            elif "k" in args and "src" in args and "a" in args and "b" in args:
                src1 = resolve(args["a"])
                src2 = resolve(args["b"])
                operands.append(src1)
                operands.append(src2)

            elif "k" in args and "src" in args and "a" in args and "b" not in args:
                src = resolve(args["a"])
                operands.append(src)

            elif (
                "k" in args
                and "a" in args
                and "op_idx" in args
                and "b" in args
                and "src" not in args
            ):
                src2 = resolve(args["b"])
                assert ctrl_reg is not None
                operands.append(ctrl_reg)
                operands.append(src2)

            elif "a" in args and "op_idx" in args and "b" in args and "k" not in args:
                src2 = resolve(args["b"])
                assert ctrl_reg is not None
                operands.append(ctrl_reg)
                operands.append(src2)

            elif "a" in args and "b" in args:
                src1 = resolve(args["a"])
                if isinstance(args["b"], int):
                    # 'b' is a numeric control vector
                    assert ctrl_reg is not None
                    operands.append(src1)
                    operands.append(ctrl_reg)
                else:
                    src2 = resolve(args["b"])
                    operands.append(src1)
                    operands.append(src2)

            elif "a" in args:
                src = resolve(args["a"])
                if "op_idx" in args:
                    assert ctrl_reg is not None
                    operands.append(ctrl_reg)
                    operands.append(src)
                else:
                    operands.append(src)

            elif "input" in args:
                src = resolve(args["input"])
                operands.append(src)

            # Immediate values and their comments
            comment_parts: list[str] = []
            if "imm8" in args:
                imm_val = args["imm8"]
                if isinstance(imm_val, int):
                    imm_type = metadata["imm_type"]
                    operands.append(f"0x{imm_val:02x}")
                    if imm_type == "binary":
                        comment_parts.append(f"0b{imm_val:08b}")
                    elif imm_type == "shuffle2":
                        comment_parts.append(mm_shuffle2_str(imm_val))
                    elif imm_type == "shuffle4":
                        comment_parts.append(mm_shuffle_str(imm_val))
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

            # Add control vector comment on the main instruction
            # (CV load line already has its own comment)
            if ctrl_val is not None:
                element_bits = ctrl_element_width or (reg_allocator.dtype.value[0] * 8)
                comment_parts.append(
                    _format_control_vector_bits(ctrl_val, total_bits, element_bits)
                )

            # Add k-mask value comment
            if kmask_val is not None:
                if isinstance(kmask_val, int):
                    comment_parts.insert(0, f"k=0x{kmask_val:02x}")
                else:
                    comment_parts.insert(0, f"k={kmask_val}")

            comment = ", ".join(comment_parts) if comment_parts else None
            inst_line = f"    {mnemonic:20s} {', '.join(operands)}"
            if comment:
                inst_line += f"  ; {comment}"
            lines.append(inst_line)

            # Update prev_target so next instruction's "prev" resolves here
            prev_target = dest_reg

            # Save result to temp if later instructions reference it.
            # In shared_mode each instruction already has its own register.
            if not shared_mode and inst_idx in needed_result_saves:
                result_save_regs[inst_idx] = reg_allocator.allocate_temp()
                lines.append(
                    f"    {mov_mnemonic:20s} {result_save_regs[inst_idx]}, {dest_reg}"
                    f"  ; save result_{inst_idx}"
                )

        return shared_results

    # --- Emission logic ---

    shared_results: dict[int, str] = {}

    if prefix_len > 0:
        # Emit shared instructions once to temp registers
        lines.append(
            f"    ; Shared instructions ({prefix_len}"
            f" of {len(gadget.top_instructions)}):"
        )
        shared_results = _emit_instructions(
            gadget.top_instructions[:prefix_len],
            is_top=True,  # doesn't matter: shared uses absolute refs
            side_label="shared",
            shared_mode=True,
        )

    # Emit top-side suffix (or all top instructions if no shared prefix)
    if gadget.top_instructions and len(gadget.top_instructions) > prefix_len:
        lines.append(f"    ; Top vector ({top_reg}) operations:")
        _emit_instructions(
            gadget.top_instructions,
            is_top=True,
            side_label="top",
            skip_prefix=prefix_len,
            precomputed_results=shared_results,
        )

    # Emit bottom-side suffix (or all bottom instructions if no shared prefix)
    if (
        gadget.bottom_instructions
        and bottom_reg
        and len(gadget.bottom_instructions) > prefix_len
    ):
        lines.append(f"    ; Bottom vector ({bottom_reg}) operations:")
        _emit_instructions(
            gadget.bottom_instructions,
            is_top=False,
            side_label="bot",
            skip_prefix=prefix_len,
            precomputed_results=shared_results,
        )

    # Edge case: if ALL instructions were shared, copy the last shared result
    # to the appropriate destination registers.
    if prefix_len > 0:
        mov_mnemonic = (
            "vmovdqa64" if reg_allocator.vm == vector_machine.AVX512 else "vmovdqa32"
        )
        last_shared = shared_results[prefix_len - 1]
        if prefix_len == len(gadget.top_instructions) and is_dual_side:
            lines.append(
                f"    {mov_mnemonic:20s} {dst_top_reg}, {last_shared}"
                f"  ; shared → top"
            )
        if (
            prefix_len == len(gadget.bottom_instructions)
            and is_dual_side
            and bottom_reg
        ):
            lines.append(
                f"    {mov_mnemonic:20s} {dst_bottom_reg}, {last_shared}"
                f"  ; shared → bottom"
            )

    return lines, is_dual_side


def generate_solution_asm(
    path,
    vm: vector_machine,
    dtype: primitive_type,
    num_vecs: int,
    natural_order: bool,
    solution_index: int,
    total_solutions: int,
) -> str:
    """Generate a complete, NASM-valid .asm file for performance analysis.

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
        f"; Total latency: {path.total_latency:.2f}",
        f"; Total score: {path.total_score:.2f}",
        f"; Control vectors: {path.total_cv_count}",
        f"; Natural order: {'yes' if natural_order else 'no'}",
        "bits 64",
        "default rel",
        "",
    ]

    # Collect .rodata entries as we emit instructions
    rodata_entries: list[tuple[str, str]] = []
    cv_cache: dict[str, str] = {}  # data_directive -> label, for dedup

    # Build .text section
    text_lines = [
        "section .text",
        "global _osaca_kernel",
        "_osaca_kernel:",
        "    ; OSACA-BEGIN",
    ]

    reg_allocator = RegisterAllocator(vm, dtype, num_vecs)

    # Print initial input state
    if path.steps:
        text_lines.append("    ; Initial Input State:")
        state_comment = _format_vector_state_as_comment(
            path.steps[0].node.input_state, "    "
        )
        text_lines.extend(state_comment.split("\n"))
        text_lines.append("")

    running_cost = 0.0
    last_idx = len(path.steps) - 1
    for step_idx, step in enumerate(path.steps):
        running_cost += step.score.total_score

        # Stage header with cost and register mapping
        text_lines.append(f"    ; Stage {step.node.stage}")
        text_lines.append(
            f"    ; Cost: {running_cost:.2f} (step: {step.score.total_score:.2f})"
        )
        if step.score.control_vector_count > 0:
            text_lines.append(
                f"    ; Control vectors: {step.score.control_vector_count}"
            )

        # Gadget operates on source pair
        top_reg = reg_allocator.src_top
        bottom_reg = reg_allocator.src_bottom
        dst_top_reg = reg_allocator.dst_top
        dst_bottom_reg = reg_allocator.dst_bottom
        is_natural_order_step = natural_order and step_idx == last_idx
        has_coex = not is_natural_order_step and num_vecs > 1

        gadget = step.gadget
        gadget_lines, is_dual_side = _emit_gadget_asm(
            gadget,
            reg_allocator,
            top_reg,
            bottom_reg,
            dst_top_reg,
            dst_bottom_reg,
            stage_idx=step_idx,
            rodata_entries=rodata_entries,
            cv_cache=cv_cache,
        )

        # Compute output registers for the transition comment.
        # Dual-side + COEX: 2 swaps → back to src pair.
        # Non-dual + COEX: 1 swap → moves to dst pair.
        # Dual-side + natural: 1 swap → moves to dst pair.
        # Non-dual + natural: 0 swaps → stays in src pair.
        output_in_dst = is_dual_side != has_coex  # XOR
        if output_in_dst:
            out_top, out_bottom = dst_top_reg, dst_bottom_reg
        else:
            out_top, out_bottom = top_reg, bottom_reg

        # Register transition
        text_lines.append(
            f"    ; Registers: {top_reg} = top, {bottom_reg} = bottom"
            f"  →  {out_top} = top, "
            f"{out_bottom} = bottom"
        )

        text_lines.extend(gadget_lines)

        # Compare-swap (skip on final natural-order stage)
        if has_coex:
            if is_dual_side:
                # Gadget wrote to dst pair; swap so COEX reads it as src
                reg_allocator.swap_pairs()
            text_lines.append(
                f"    ; Compare-swap: "
                f"min → {reg_allocator.dst_top} (top), "
                f"max → {reg_allocator.dst_bottom} (bottom)"
            )
            cs_lines = _emit_compare_swap_lines(reg_allocator)
            text_lines.extend(cs_lines)
            reg_allocator.swap_pairs()
        elif is_dual_side:
            # Natural-order dual-side: gadget wrote to dst, swap once
            reg_allocator.swap_pairs()

        # Output state
        text_lines.append("    ; Output State:")
        state_comment = _format_vector_state_as_comment(step.node.output_state, "    ")
        text_lines.extend(state_comment.split("\n"))
        text_lines.append("")

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


# Lines to strip entirely (NASM-specific directives, data, scaffolding)
_LLVM_STRIP_PATTERNS = [
    re.compile(r"^\s*bits\s+\d+"),
    re.compile(r"^\s*default\s+rel"),
    re.compile(r"^\s*section\s+"),
    re.compile(r"^\s*global\s+"),
    re.compile(r"^\s*align\s+\d+"),
    re.compile(r"^\s*\w+:$"),  # labels
    re.compile(r"^\s*d[dqbw]\s+"),  # data directives
    re.compile(r"^\s*ret\b"),
]


def sanitize_asm_for_llvm_mca(asm_text: str) -> str:
    """Sanitize NASM-syntax assembly for LLVM-MCA's Intel parser.

    LLVM-MCA with -x86-asm-syntax=intel accepts Intel operand ordering
    natively. Only NASM-specific constructs need transformation:
    - [rel label] → [rip + label]
    - ; comments → # comments
    - OSACA markers → LLVM-MCA markers
    - NASM directives, data sections, labels → stripped
    """
    output_lines: list[str] = []

    for line in asm_text.splitlines():
        stripped = line.strip()

        # OSACA markers → LLVM-MCA markers
        if "; OSACA-BEGIN" in line:
            output_lines.append("# LLVM-MCA-BEGIN")
            continue
        if "; OSACA-END" in line:
            output_lines.append("# LLVM-MCA-END")
            continue

        # Strip NASM-only directives
        if any(p.match(stripped) for p in _LLVM_STRIP_PATTERNS):
            continue

        # Empty lines
        if not stripped:
            continue

        # Comment-only lines
        if stripped.startswith(";"):
            output_lines.append(f"# {stripped[1:].strip()}")
            continue

        # Instruction lines: fix memory operands and comments
        # [rel label] → [rip + label]
        line = re.sub(r"\[rel\s+(\w+)\]", r"[rip + \1]", line)

        # ; inline comment → # inline comment
        semi_pos = line.find(";")
        if semi_pos >= 0:
            line = line[:semi_pos].rstrip() + " # " + line[semi_pos + 1 :].strip()

        output_lines.append(line.rstrip())

    result = "\n".join(output_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip() + "\n" if result.strip() else ""


def estimate_solutions(
    paths,
    vm: vector_machine,
    dtype: primitive_type,
    num_vecs: int,
    natural_order: bool,
    target_cpu: str,
    nasm_path: str | None = None,
    llvm_mca_path: str | None = None,
) -> list[EstimationResult]:
    """Run LLVM-MCA estimation on a list of CompletePath solutions.

    Args:
        paths: List of CompletePath objects
        vm: Vector machine (AVX2/AVX512)
        dtype: Primitive data type
        num_vecs: Number of SIMD vectors
        natural_order: Whether the final stage is natural-order
        target_cpu: Target CPU string (same as --target-cpu)
        nasm_path: Optional path to nasm binary for verification
        llvm_mca_path: Optional explicit path to llvm-mca binary

    Returns:
        List of EstimationResult objects, one per path.
    """
    from cost_model import resolve_llvm_mca_cpu
    from llvm_mca_runner import find_llvm_mca, run_llvm_mca

    mcpu = resolve_llvm_mca_cpu(target_cpu)
    if mcpu is None:
        supported = [
            info.canonical for info in get_supported_cpus() if info.llvm_mca_cpu
        ]
        print(
            f"Warning: Cannot resolve llvm-mca CPU for '{target_cpu}'. "
            f"Supported: {', '.join(supported)}",
            file=sys.stderr,
        )
        return []

    mca_bin = find_llvm_mca(llvm_mca_path)
    if mca_bin is None:
        print(
            "Warning: llvm-mca not found. Install LLVM or use --llvm-mca-path.",
            file=sys.stderr,
        )
        return []

    # Create temp directory
    output_dir = tempfile.mkdtemp(prefix="vxsort_mca_", dir="/tmp")
    print(f"LLVM-MCA estimation: mcpu={mcpu}, output_dir={output_dir}")

    results: list[EstimationResult] = []
    total = len(paths)

    for i, path in enumerate(paths):
        solution_index = i + 1

        # Generate Intel/NASM assembly
        asm_content = generate_solution_asm(
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

        # Sanitize for LLVM-MCA and run
        sanitized = sanitize_asm_for_llvm_mca(asm_content)
        mca_result = run_llvm_mca(sanitized, mcpu, mca_bin, solution_index, output_dir)

        results.append(
            EstimationResult(
                solution_index=mca_result.solution_index,
                asm_path=asm_path,
                throughput=mca_result.throughput,
                simulated_cycles=mca_result.simulated_cycles,
                analysis_path=mca_result.analysis_path,
                warnings=mca_result.warnings,
            )
        )

    return results


def print_estimation_table(results: list[EstimationResult], paths) -> None:
    """Print a table comparing LLVM-MCA results across solutions."""
    if not results:
        print("No estimation results to display.")
        return

    headers = [
        "Solution",
        "ASM File",
        "Analysis",
        "Sim.Cy",
        "TP",
        "Stages",
        "CVs",
        "Our Cost",
    ]
    rows = []

    for result, path in zip(results, paths):
        sc_str = (
            f"{result.simulated_cycles:.2f}" if result.simulated_cycles >= 0 else "err"
        )
        tp_str = f"{result.throughput:.2f}" if result.throughput >= 0 else "err"
        if result.analysis_path:
            analysis_name = Path(result.analysis_path).name
            analysis_link = (
                f"\033]8;;subl://open?url=file://{result.analysis_path}\033\\"
                f"{analysis_name}\033]8;;\033\\"
            )
        else:
            analysis_link = "-"
        rows.append(
            [
                result.solution_index,
                f"\033]8;;subl://open?url=file://{result.asm_path}\033\\{Path(result.asm_path).name}\033]8;;\033\\",
                analysis_link,
                sc_str,
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
