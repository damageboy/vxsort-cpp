//! ASM renderer for bitonic sort solution paths.
//!
//! The public API (`generate_solution_asm_for_paths`,
//! `write_solution_asm_for_paths`) is intentionally stable: callers pass a
//! `TransitionTable` + `CompletePath` list; this module lowers to an
//! [`InstructionStream`] via [`lower_solution_paths`] and then renders the
//! stream to NASM/Intel-syntax assembly text.
//!
//! All lowering decisions (register allocation, gadget argument resolution,
//! compare-swap encoding) live in [`crate::instruction_stream`].  This module
//! is a **pure formatter** over the modeled stream.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use crate::instruction_stream::{
    ConstantData, ConstantKey, InstructionStream, LoweringOptions, ModeledInstruction, Operand,
    control_vector_elements, lower_assigned_paths, lower_solution_paths,
};
use crate::json_exporter::SolutionJsonMetadata;
use crate::scoring::AssignedPath;
use crate::transition_table::{CompletePath, TransitionTable};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Write NASM/Intel-syntax assembly for `paths` to `output_path`.
pub fn write_solution_asm_for_paths(
    output_path: impl AsRef<Path>,
    metadata: &SolutionJsonMetadata,
    table: &TransitionTable,
    paths: &[CompletePath],
) -> Result<(), io::Error> {
    let mut file = File::create(output_path)?;
    file.write_all(generate_solution_asm_for_paths(metadata, table, paths).as_bytes())
}

pub fn write_solution_asm_for_assigned_paths(
    output_path: impl AsRef<Path>,
    metadata: &SolutionJsonMetadata,
    paths: &[AssignedPath],
) -> Result<(), io::Error> {
    let mut file = File::create(output_path)?;
    file.write_all(generate_solution_asm_for_assigned_paths(metadata, paths).as_bytes())
}

/// Generate NASM/Intel-syntax assembly for `paths` and return it as a
/// `String`.
///
/// The function lowers the paths to an [`InstructionStream`] and then renders
/// that stream to text.  Conceptually:
///
/// ```text
/// let stream = lower_solution_paths(metadata, table, paths, LoweringOptions::default());
/// render_asm(&stream, metadata)
/// ```
pub fn generate_solution_asm_for_paths(
    metadata: &SolutionJsonMetadata,
    table: &TransitionTable,
    paths: &[CompletePath],
) -> String {
    let stream = lower_solution_paths(metadata, table, paths, LoweringOptions::default());
    render_asm(&stream, metadata)
}

pub fn generate_solution_asm_for_assigned_paths(
    metadata: &SolutionJsonMetadata,
    paths: &[AssignedPath],
) -> String {
    let stream = lower_assigned_paths(metadata, paths, LoweringOptions::default());
    render_asm(&stream, metadata)
}

// ---------------------------------------------------------------------------
// Renderer (pure formatting, no lowering decisions)
// ---------------------------------------------------------------------------

/// Render an [`InstructionStream`] to NASM/Intel-syntax assembly text.
///
/// This function is a **pure formatter**: it does not inspect the
/// `TransitionTable`, allocate registers, pool constants, or resolve gadget
/// arguments.  All of that is already encoded in the stream.
fn render_asm(stream: &InstructionStream, metadata: &SolutionJsonMetadata) -> String {
    let mut lines = vec![
        "; vxsort_codegen Rust assembly export".to_owned(),
        format!("; vector_machine: {}", metadata.arch.cli_name()),
        format!("; primitive_type: {}", metadata.dtype.cli_name()),
        format!("; num_vecs: {}", metadata.num_vecs),
        "default rel".to_owned(),
    ];

    if !stream.constants.is_empty() {
        lines.push("section .rodata".to_owned());
        for constant in &stream.constants {
            lines.extend(render_constant_data(constant));
        }
    }

    lines.push("section .text".to_owned());

    if stream.blocks.is_empty() {
        lines.push("; no complete solutions exported".to_owned());
        lines.push(String::new());
        return lines.join("\n");
    }

    for block in &stream.blocks {
        lines.push(String::new());
        if let Some(label) = &block.label {
            lines.push(format!("global {label}"));
            lines.push(format!("{label}:"));
        }
        for instruction in &block.instructions {
            lines.push(render_instruction(instruction));
        }
    }

    lines.push(String::new());
    lines.join("\n")
}

/// Format a single [`ModeledInstruction`] as an indented assembly line.
///
/// Operands are rendered in Intel syntax (destination first) and joined with
/// `", "`.  An inline comment is appended when `instruction.comment` is set.
fn render_instruction(instruction: &ModeledInstruction) -> String {
    if instruction.mnemonic.is_empty() {
        return match &instruction.comment {
            Some(comment) => format!("    ; {comment}"),
            None => String::new(),
        };
    }

    let operand_str = instruction
        .operands
        .iter()
        .map(render_operand)
        .collect::<Vec<_>>()
        .join(", ");

    let body = if operand_str.is_empty() {
        format!("    {}", instruction.mnemonic)
    } else {
        format!("    {:<20} {}", instruction.mnemonic, operand_str)
    };

    match &instruction.comment {
        Some(comment) => format!("{body}  ; {comment}"),
        None => body,
    }
}

/// Format a single [`Operand`] as an assembly token.
fn render_operand(operand: &Operand) -> String {
    match operand {
        Operand::Register(r) => r.name(),
        Operand::Immediate(i) => format_immediate(*i),
        Operand::ConstantRef(label) => format!("[rel {label}]"),
        Operand::KMask(n) => format!("k{n}"),
        Operand::MaskedRegister(register, mask) => format!("{}{{k{mask}}}", register.name()),
        Operand::Gpr(register) => (*register).to_owned(),
        Operand::Unsupported(s) => format!("; unsupported: {s}"),
    }
}

fn render_constant_data(constant: &ConstantData) -> Vec<String> {
    let (bits, bytes) = match &constant.key {
        ConstantKey::Vector { bits, hex } => (*bits, little_endian_bytes_from_hex(hex, *bits)),
        ConstantKey::KMask { bits, value } => (*bits, little_endian_bytes_from_u64(*value, *bits)),
        ConstantKey::ControlVector {
            bits,
            element_bits,
            value,
        } => return render_control_vector_constant(constant, *bits, *element_bits, *value),
    };

    let mut lines = vec![format!("{}:", constant.label)];
    if bytes.is_empty() {
        lines.push(format!("    db                 0  ; {bits}-bit constant"));
        return lines;
    }

    for chunk in bytes.chunks(16) {
        let rendered = chunk
            .iter()
            .map(|byte| format!("0x{byte:02x}"))
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!(
            "    db                 {rendered}  ; {bits}-bit constant"
        ));
    }
    lines
}

fn render_control_vector_constant(
    constant: &ConstantData,
    bits: u32,
    element_bits: u32,
    value: u64,
) -> Vec<String> {
    let directive = if element_bits <= 32 { "dd" } else { "dq" };
    let elements = control_vector_elements(value, bits, element_bits)
        .into_iter()
        .map(|element| element.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    vec![
        format!("{}:", constant.label),
        format!("    {directive:<18} {elements}  ; {bits}-bit control vector"),
    ]
}

fn format_immediate(value: u64) -> String {
    if value <= 0xff {
        format!("0x{value:02x}")
    } else {
        format!("0x{value:x}")
    }
}

fn little_endian_bytes_from_hex(hex: &str, bits: u32) -> Vec<u8> {
    let byte_len = bits.div_ceil(8) as usize;
    let mut digits = hex
        .strip_prefix("0x")
        .or_else(|| hex.strip_prefix("0X"))
        .unwrap_or(hex)
        .to_owned();
    if digits.len() % 2 == 1 {
        digits.insert(0, '0');
    }

    let mut bytes = digits
        .as_bytes()
        .rchunks(2)
        .filter_map(|pair| std::str::from_utf8(pair).ok())
        .filter_map(|pair| u8::from_str_radix(pair, 16).ok())
        .collect::<Vec<_>>();
    bytes.resize(byte_len, 0);
    bytes.truncate(byte_len);
    bytes
}

fn little_endian_bytes_from_u64(value: u64, bits: u32) -> Vec<u8> {
    let byte_len = bits.div_ceil(8) as usize;
    let mut bytes = value.to_le_bytes().to_vec();
    bytes.truncate(byte_len);
    bytes
}
