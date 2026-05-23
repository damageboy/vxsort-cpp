use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io;
use std::path::Path;

use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget};
use serde_json::{Map, Value, json};

use crate::transition_table::{CompletePath, StateTuple, TransitionKey, TransitionTable};
use crate::{ArchArg, DTypeArg};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SolutionJsonMetadata {
    pub natural_order: bool,
    pub arch: ArchArg,
    pub dtype: DTypeArg,
    pub num_vecs: usize,
}

type StepKey = (usize, StateTuple, StateTuple);

pub fn write_solution_json_for_paths(
    output_path: impl AsRef<Path>,
    metadata: &SolutionJsonMetadata,
    table: &TransitionTable,
    paths: &[CompletePath],
) -> Result<(), io::Error> {
    let file = File::create(output_path)?;
    serde_json::to_writer_pretty(file, &solution_json_for_paths(metadata, table, paths))?;
    Ok(())
}

pub fn solution_json_for_paths(
    metadata: &SolutionJsonMetadata,
    table: &TransitionTable,
    paths: &[CompletePath],
) -> Value {
    let mut node_ids = BTreeMap::<StepKey, String>::new();
    let mut node_order = Vec::<StepKey>::new();
    let mut children = BTreeMap::<StepKey, BTreeSet<String>>::new();
    let mut root_ids = Vec::new();

    for path in paths {
        if path.is_empty() {
            continue;
        }

        for step in path {
            ensure_node_id(step, &mut node_ids, &mut node_order);
        }

        root_ids.push(
            node_ids
                .get(&path[0])
                .expect("path root should have an assigned node id")
                .clone(),
        );

        for window in path.windows(2) {
            let parent = &window[0];
            let child_id = node_ids
                .get(&window[1])
                .expect("path child should have an assigned node id")
                .clone();
            children.entry(parent.clone()).or_default().insert(child_id);
        }
    }

    let mut nodes = Map::new();
    for key in node_order {
        let id = node_ids
            .get(&key)
            .expect("node order should reference assigned ids");
        nodes.insert(id.clone(), node_json(&key, table, &children));
    }

    json!({
        "natural_order": metadata.natural_order,
        "vector_machine": metadata.arch.cli_name(),
        "primitive_type": metadata.dtype.cli_name(),
        "num_vecs": metadata.num_vecs,
        "roots": root_ids,
        "nodes": nodes,
    })
}

fn ensure_node_id(
    key: &StepKey,
    node_ids: &mut BTreeMap<StepKey, String>,
    node_order: &mut Vec<StepKey>,
) {
    if node_ids.contains_key(key) {
        return;
    }

    let id = format!("n{}", node_ids.len());
    node_ids.insert(key.clone(), id);
    node_order.push(key.clone());
}

fn node_json(
    key: &StepKey,
    table: &TransitionTable,
    children_by_key: &BTreeMap<StepKey, BTreeSet<String>>,
) -> Value {
    let (stage, input, output) = key;
    let transition_key: TransitionKey = (input.clone(), output.clone());
    let empty_gadgets = Vec::new();
    let gadgets = table
        .get_all_transitions(*stage)
        .get(&transition_key)
        .unwrap_or(&empty_gadgets);
    let child_ids = children_by_key
        .get(key)
        .map(|children| children.iter().cloned().collect::<Vec<_>>())
        .unwrap_or_default();

    json!({
        "stage": stage,
        "input_state": state_json(input),
        "output_state": state_json(output),
        "gadgets": gadgets.iter().map(gadget_json).collect::<Vec<_>>(),
        "gadget_count": gadgets.len(),
        "children": child_ids,
    })
}

fn state_json(state: &StateTuple) -> Value {
    json!({
        "top": state.0,
        "bottom": state.1,
    })
}

fn gadget_json(gadget: &PermutationGadget) -> Value {
    let (unified, top_output_index, bottom_output_index) = gadget.unified_instructions();
    json!({
        "top_instructions": instructions_json(gadget.top_instructions()),
        "bottom_instructions": instructions_json(gadget.bottom_instructions()),
        "unified_instructions": instructions_json(&unified),
        "top_output_index": top_output_index,
        "bottom_output_index": bottom_output_index,
    })
}

fn instructions_json(instructions: &[InstructionSpec]) -> Vec<Value> {
    instructions.iter().map(instruction_json).collect()
}

fn instruction_json(instruction: &InstructionSpec) -> Value {
    let mut args = Map::new();
    for (name, value) in instruction.args() {
        args.insert((*name).to_owned(), instruction_arg_json(value));
    }

    json!({
        "name": instruction.intrinsic_name(),
        "args": args,
    })
}

fn instruction_arg_json(value: &InstructionArg) -> Value {
    match value {
        InstructionArg::Input(name) => Value::String(name.clone()),
        InstructionArg::U64(value) => json!(value),
        InstructionArg::BitVec { bits, hex } => json!({
            "kind": "bitvec",
            "bits": bits,
            "hex": hex,
        }),
    }
}
