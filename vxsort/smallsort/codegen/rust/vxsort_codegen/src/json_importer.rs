use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget, VectorState};
use serde_json::Value;

use crate::json_exporter::SolutionJsonMetadata;
use crate::scoring::{AssignedPath, AssignedStep};
use crate::transition_table::{CompletePath, PathStep, StateTuple, TransitionTable};
use crate::{ArchArg, DTypeArg};

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedSolutionJson {
    pub metadata: SolutionJsonMetadata,
    pub transition_table: TransitionTable,
    pub paths: Vec<CompletePath>,
    pub assigned_paths: Vec<AssignedPath>,
}

pub fn read_solution_json(input_path: impl AsRef<Path>) -> Result<ImportedSolutionJson, String> {
    let bytes = fs::read(input_path.as_ref())
        .map_err(|error| format!("failed to read {}: {error}", input_path.as_ref().display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to parse {}: {error}", input_path.as_ref().display()))?;
    solution_json_from_value(&value)
}

pub fn solution_json_from_value(value: &Value) -> Result<ImportedSolutionJson, String> {
    let metadata = parse_metadata(value)?;
    let nodes = value
        .get("nodes")
        .and_then(Value::as_object)
        .ok_or_else(|| "solution JSON must contain an object `nodes` field".to_owned())?;

    let mut parsed_nodes = BTreeMap::new();
    let mut max_stage = None;
    for (id, node) in nodes {
        let parsed = parse_node(id, node, &metadata)?;
        max_stage = Some(max_stage.map_or(parsed.step.0, |stage: usize| stage.max(parsed.step.0)));
        parsed_nodes.insert(id.clone(), parsed);
    }

    let mut transition_table = TransitionTable::new(max_stage.map_or(0, |stage| stage + 1));
    for parsed in parsed_nodes.values() {
        let input = VectorState::new(parsed.step.1.0.clone(), parsed.step.1.1.clone());
        let output = VectorState::new(parsed.step.2.0.clone(), parsed.step.2.1.clone());
        for gadget in &parsed.gadgets {
            transition_table.add_transition(parsed.step.0, &input, &output, gadget.clone());
        }
    }

    let roots = value
        .get("roots")
        .and_then(Value::as_array)
        .ok_or_else(|| "solution JSON must contain an array `roots` field".to_owned())?;
    let mut graph_paths = Vec::new();
    for root in roots {
        let root_id = root
            .as_str()
            .ok_or_else(|| "solution JSON root ids must be strings".to_owned())?;
        collect_paths(root_id, &parsed_nodes, &mut Vec::new(), &mut graph_paths)?;
    }
    let assigned_paths = parse_assigned_paths(value.get("paths"), &parsed_nodes)?;
    let paths = if assigned_paths.is_empty() {
        graph_paths
    } else {
        assigned_paths
            .iter()
            .map(AssignedPath::as_complete_path)
            .collect()
    };

    Ok(ImportedSolutionJson {
        metadata,
        transition_table,
        paths,
        assigned_paths,
    })
}

#[derive(Clone, Debug, PartialEq)]
struct ParsedNode {
    step: PathStep,
    gadgets: Vec<PermutationGadget>,
    children: Vec<String>,
}

fn parse_metadata(value: &Value) -> Result<SolutionJsonMetadata, String> {
    Ok(SolutionJsonMetadata {
        natural_order: value
            .get("natural_order")
            .and_then(Value::as_bool)
            .ok_or_else(|| "solution JSON must contain boolean `natural_order`".to_owned())?,
        arch: parse_arch(
            value
                .get("vector_machine")
                .and_then(Value::as_str)
                .ok_or_else(|| "solution JSON must contain string `vector_machine`".to_owned())?,
        )?,
        dtype: parse_dtype(
            value
                .get("primitive_type")
                .and_then(Value::as_str)
                .ok_or_else(|| "solution JSON must contain string `primitive_type`".to_owned())?,
        )?,
        num_vecs: value
            .get("num_vecs")
            .and_then(Value::as_u64)
            .ok_or_else(|| "solution JSON must contain integer `num_vecs`".to_owned())?
            as usize,
    })
}

fn parse_arch(value: &str) -> Result<ArchArg, String> {
    match value.to_ascii_lowercase().as_str() {
        "avx2" => Ok(ArchArg::Avx2),
        "avx512" => Ok(ArchArg::Avx512),
        _ => Err(format!("unsupported vector_machine `{value}`")),
    }
}

fn parse_dtype(value: &str) -> Result<DTypeArg, String> {
    match value.to_ascii_lowercase().as_str() {
        "i16" => Ok(DTypeArg::I16),
        "u16" => Ok(DTypeArg::U16),
        "i32" => Ok(DTypeArg::I32),
        "u32" => Ok(DTypeArg::U32),
        "f32" => Ok(DTypeArg::F32),
        "i64" => Ok(DTypeArg::I64),
        "u64" => Ok(DTypeArg::U64),
        "f64" => Ok(DTypeArg::F64),
        _ => Err(format!("unsupported primitive_type `{value}`")),
    }
}

fn parse_node(
    id: &str,
    value: &Value,
    metadata: &SolutionJsonMetadata,
) -> Result<ParsedNode, String> {
    let stage = value
        .get("stage")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("node `{id}` must contain integer `stage`"))?
        as usize;
    let input = parse_state(
        value
            .get("input_state")
            .ok_or_else(|| format!("node `{id}` must contain `input_state`"))?,
        id,
        "input_state",
    )?;
    let output = parse_state(
        value
            .get("output_state")
            .ok_or_else(|| format!("node `{id}` must contain `output_state`"))?,
        id,
        "output_state",
    )?;
    let gadgets = value
        .get("gadgets")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("node `{id}` must contain array `gadgets`"))?
        .iter()
        .map(|gadget| parse_gadget(gadget, metadata))
        .collect::<Result<Vec<_>, _>>()?;
    let mut children = value
        .get("children")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("node `{id}` must contain array `children`"))?
        .iter()
        .map(|child| {
            child
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("node `{id}` children must be strings"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    children.sort();

    Ok(ParsedNode {
        step: (stage, input, output),
        gadgets,
        children,
    })
}

fn parse_state(value: &Value, node_id: &str, field: &str) -> Result<StateTuple, String> {
    let top = parse_u64_array(
        value
            .get("top")
            .ok_or_else(|| format!("node `{node_id}` {field} must contain `top`"))?,
        &format!("node `{node_id}` {field}.top"),
    )?;
    let bottom = parse_u64_array(
        value
            .get("bottom")
            .ok_or_else(|| format!("node `{node_id}` {field} must contain `bottom`"))?,
        &format!("node `{node_id}` {field}.bottom"),
    )?;
    Ok((top, bottom))
}

fn parse_u64_array(value: &Value, label: &str) -> Result<Vec<u64>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .ok_or_else(|| format!("{label} entries must be unsigned integers"))
        })
        .collect()
}

fn parse_gadget(
    value: &Value,
    metadata: &SolutionJsonMetadata,
) -> Result<PermutationGadget, String> {
    let top = parse_instruction_array(value, "top_instructions", metadata)?;
    let bottom = parse_instruction_array(value, "bottom_instructions", metadata)?;
    Ok(PermutationGadget::new(top, bottom))
}

fn parse_instruction_array(
    value: &Value,
    field: &str,
    metadata: &SolutionJsonMetadata,
) -> Result<Vec<InstructionSpec>, String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("gadget must contain array `{field}`"))?
        .iter()
        .map(|instruction| parse_instruction(instruction, metadata))
        .collect()
}

fn parse_instruction(
    value: &Value,
    metadata: &SolutionJsonMetadata,
) -> Result<InstructionSpec, String> {
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "instruction must contain string `name`".to_owned())?;
    let args = value
        .get("args")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("instruction `{name}` must contain object `args`"))?;
    let mut parsed_args = BTreeMap::new();
    for (key, value) in args {
        let key = static_str(key);
        parsed_args.insert(key, parse_instruction_arg(key, value, metadata)?);
    }
    Ok(InstructionSpec::new(static_str(name), parsed_args))
}

fn parse_instruction_arg(
    key: &'static str,
    value: &Value,
    metadata: &SolutionJsonMetadata,
) -> Result<InstructionArg, String> {
    if let Some(input) = value.as_str() {
        return Ok(InstructionArg::Input(input.to_owned()));
    }
    if let Some(immediate) = value.as_u64() {
        return Ok(InstructionArg::U64(immediate));
    }
    if let Some(number) = value.as_number() {
        let decimal = number.to_string();
        return Ok(InstructionArg::BitVec {
            bits: instruction_arg_bits(key, metadata),
            hex: decimal_to_hex(&decimal)?,
        });
    }
    let Some(object) = value.as_object() else {
        return Err("instruction argument must be string, integer, or bitvec object".to_owned());
    };
    let kind = object
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| "instruction argument object must contain string `kind`".to_owned())?;
    if kind != "bitvec" {
        return Err(format!("unsupported instruction argument kind `{kind}`"));
    }
    let bits = object
        .get("bits")
        .and_then(Value::as_u64)
        .ok_or_else(|| "bitvec argument must contain integer `bits`".to_owned())?
        as u32;
    let hex = object
        .get("hex")
        .and_then(Value::as_str)
        .ok_or_else(|| "bitvec argument must contain string `hex`".to_owned())?
        .to_owned();
    Ok(InstructionArg::BitVec { bits, hex })
}

fn instruction_arg_bits(key: &'static str, metadata: &SolutionJsonMetadata) -> u32 {
    match key {
        "imm8" => 8,
        "k" => (total_bits(metadata.arch) / metadata.dtype.element_bits()) as u32,
        _ => total_bits(metadata.arch) as u32,
    }
}

fn total_bits(arch: ArchArg) -> usize {
    match arch {
        ArchArg::Avx2 => 256,
        ArchArg::Avx512 => 512,
    }
}

fn decimal_to_hex(decimal: &str) -> Result<String, String> {
    if decimal.is_empty() || !decimal.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "unsupported numeric instruction argument `{decimal}`"
        ));
    }
    let mut digits = decimal
        .trim_start_matches('0')
        .bytes()
        .map(|byte| byte - b'0')
        .collect::<Vec<_>>();
    if digits.is_empty() {
        return Ok("0".to_owned());
    }

    let mut hex_digits = Vec::new();
    while !digits.is_empty() {
        let mut quotient = Vec::new();
        let mut remainder = 0u8;
        for digit in digits {
            let value = remainder as u16 * 10 + digit as u16;
            let q = (value / 16) as u8;
            remainder = (value % 16) as u8;
            if !quotient.is_empty() || q != 0 {
                quotient.push(q);
            }
        }
        hex_digits.push(char::from_digit(remainder as u32, 16).expect("remainder is base16"));
        digits = quotient;
    }

    Ok(hex_digits.into_iter().rev().collect())
}

fn collect_paths(
    node_id: &str,
    nodes: &BTreeMap<String, ParsedNode>,
    current: &mut CompletePath,
    paths: &mut Vec<CompletePath>,
) -> Result<(), String> {
    let node = nodes
        .get(node_id)
        .ok_or_else(|| format!("solution JSON references missing node `{node_id}`"))?;
    current.push(node.step.clone());
    if node.children.is_empty() {
        paths.push(current.clone());
    } else {
        let mut seen = BTreeSet::new();
        for child in &node.children {
            if seen.insert(child) {
                collect_paths(child, nodes, current, paths)?;
            }
        }
    }
    current.pop();
    Ok(())
}

fn parse_assigned_paths(
    value: Option<&Value>,
    nodes: &BTreeMap<String, ParsedNode>,
) -> Result<Vec<AssignedPath>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let paths = value
        .as_array()
        .ok_or_else(|| "solution JSON `paths` field must be an array".to_owned())?;
    paths
        .iter()
        .enumerate()
        .map(|(path_index, path)| parse_assigned_path(path_index, path, nodes))
        .collect()
}

fn parse_assigned_path(
    path_index: usize,
    value: &Value,
    nodes: &BTreeMap<String, ParsedNode>,
) -> Result<AssignedPath, String> {
    let steps = value
        .get("steps")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("paths[{path_index}] must contain array `steps`"))?;
    steps
        .iter()
        .enumerate()
        .map(|(step_index, step)| parse_assigned_step(path_index, step_index, step, nodes))
        .collect::<Result<Vec<_>, _>>()
        .map(AssignedPath::new)
}

fn parse_assigned_step(
    path_index: usize,
    step_index: usize,
    value: &Value,
    nodes: &BTreeMap<String, ParsedNode>,
) -> Result<AssignedStep, String> {
    let node_id = value
        .get("node_id")
        .or_else(|| value.get("node"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            format!("paths[{path_index}].steps[{step_index}] must contain string `node_id`")
        })?;
    let gadget_index = value
        .get("gadget_index")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            format!("paths[{path_index}].steps[{step_index}] must contain integer `gadget_index`")
        })? as usize;
    let node = nodes.get(node_id).ok_or_else(|| {
        format!("paths[{path_index}].steps[{step_index}] references missing node `{node_id}`")
    })?;
    let gadget = node.gadgets.get(gadget_index).ok_or_else(|| {
        format!(
            "paths[{path_index}].steps[{step_index}] references gadget_index {gadget_index}, but node `{node_id}` has {} gadgets",
            node.gadgets.len()
        )
    })?;

    Ok(AssignedStep::new(
        node.step.0,
        node.step.1.clone(),
        node.step.2.clone(),
        gadget_index,
        gadget.clone(),
    ))
}

fn static_str(value: &str) -> &'static str {
    Box::leak(value.to_owned().into_boxed_str())
}
