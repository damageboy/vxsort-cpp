use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io;
use std::path::Path;

use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget};
use serde_json::{Map, Value, json};

use crate::scoring::AssignedPath;
use crate::transition_table::{
    CompletePath, GadgetIndex, StateTuple, TransitionRef, TransitionTable,
};
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

pub fn write_solution_json_for_assigned_paths(
    output_path: impl AsRef<Path>,
    metadata: &SolutionJsonMetadata,
    table: &TransitionTable,
    paths: &[AssignedPath],
) -> Result<(), io::Error> {
    let file = File::create(output_path)?;
    serde_json::to_writer_pretty(
        file,
        &solution_json_for_assigned_paths(metadata, table, paths),
    )?;
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
    let mut node_gadgets = BTreeMap::<StepKey, Vec<PermutationGadget>>::new();
    let mut root_ids = Vec::new();

    for path in paths {
        if path.is_empty() {
            continue;
        }

        let keys = complete_path_keys(table, path);
        for (key, gadgets) in &keys {
            ensure_node_id(key, &mut node_ids, &mut node_order);
            node_gadgets
                .entry(key.clone())
                .or_default()
                .extend(gadgets.iter().cloned());
        }

        root_ids.push(
            node_ids
                .get(&keys[0].0)
                .expect("path root should have an assigned node id")
                .clone(),
        );

        for window in keys.windows(2) {
            let parent = &window[0].0;
            let child_id = node_ids
                .get(&window[1].0)
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
        let gadgets = node_gadgets.get(&key).cloned().unwrap_or_default();
        nodes.insert(
            id.clone(),
            node_json_with_gadgets(&key, &gadgets, &children),
        );
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

pub fn solution_json_for_assigned_paths(
    metadata: &SolutionJsonMetadata,
    table: &TransitionTable,
    paths: &[AssignedPath],
) -> Value {
    let mut node_ids = BTreeMap::<StepKey, String>::new();
    let mut node_order = Vec::<StepKey>::new();
    let mut children = BTreeMap::<StepKey, BTreeSet<String>>::new();
    let mut node_gadgets = BTreeMap::<StepKey, Vec<PermutationGadget>>::new();
    let mut node_gadget_indices = BTreeMap::<(StepKey, GadgetIndex), usize>::new();
    let mut root_ids = Vec::new();

    for path in paths {
        if path.path().is_empty() {
            continue;
        }

        let keys = assigned_path_keys(table, path);
        for (key, original_gadget_index, gadget) in &keys {
            ensure_node_id(key, &mut node_ids, &mut node_order);
            let gadgets = node_gadgets.entry(key.clone()).or_default();
            let exported_gadget_index = gadgets.iter().position(|existing| existing == *gadget);
            let exported_gadget_index = if let Some(index) = exported_gadget_index {
                index
            } else {
                gadgets.push((*gadget).clone());
                gadgets.len() - 1
            };
            node_gadget_indices
                .insert((key.clone(), *original_gadget_index), exported_gadget_index);
        }

        root_ids.push(
            node_ids
                .get(&keys[0].0)
                .expect("assigned path root should have an assigned node id")
                .clone(),
        );

        for window in keys.windows(2) {
            let parent_key = window[0].0.clone();
            let child_key = window[1].0.clone();
            let child_id = node_ids
                .get(&child_key)
                .expect("assigned path child should have an assigned node id")
                .clone();
            children.entry(parent_key).or_default().insert(child_id);
        }
    }

    let mut nodes = Map::new();
    for key in node_order {
        let id = node_ids
            .get(&key)
            .expect("node order should reference assigned ids");
        let gadgets = node_gadgets.get(&key).cloned().unwrap_or_default();
        nodes.insert(
            id.clone(),
            node_json_with_gadgets(&key, &gadgets, &children),
        );
    }

    let concrete_paths = paths
        .iter()
        .filter(|path| !path.path().is_empty())
        .map(|path| {
            let steps = path
                .path()
                .iter()
                .zip(path.gadgets())
                .map(|((stage, transition), gadget_index)| {
                    let key = step_key_for_transition(
                        table,
                        TransitionRef {
                            stage: stage
                                .try_into()
                                .expect("stage index should fit in transition ref"),
                            transition,
                        },
                    );
                    let node_id = node_ids
                        .get(&key)
                        .expect("assigned path step should have a node id");
                    let exported_gadget_index = node_gadget_indices
                        .get(&(key, *gadget_index))
                        .copied()
                        .expect("assigned path step should have an exported gadget index");
                    json!({
                        "node_id": node_id,
                        "gadget_index": exported_gadget_index,
                    })
                })
                .collect::<Vec<_>>();
            json!({ "steps": steps })
        })
        .collect::<Vec<_>>();

    json!({
        "natural_order": metadata.natural_order,
        "vector_machine": metadata.arch.cli_name(),
        "primitive_type": metadata.dtype.cli_name(),
        "num_vecs": metadata.num_vecs,
        "roots": root_ids,
        "nodes": nodes,
        "paths": concrete_paths,
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

fn node_json_with_gadgets(
    key: &StepKey,
    gadgets: &[PermutationGadget],
    children_by_key: &BTreeMap<StepKey, BTreeSet<String>>,
) -> Value {
    let (stage, input, output) = key;
    let child_ids = children_by_key
        .get(key)
        .map(|children| children.iter().cloned().collect::<Vec<_>>())
        .unwrap_or_default();

    node_json_payload(stage, input, output, gadgets, child_ids)
}

fn complete_path_keys(
    table: &TransitionTable,
    path: &CompletePath,
) -> Vec<(StepKey, Vec<PermutationGadget>)> {
    path.iter()
        .map(|(stage, transition)| {
            let transition_ref = TransitionRef {
                stage: stage
                    .try_into()
                    .expect("stage index should fit in transition ref"),
                transition,
            };
            let key = step_key_for_transition(table, transition_ref);
            let gadgets = table.transition_gadgets(transition_ref).to_vec();
            (key, gadgets)
        })
        .collect()
}

fn assigned_path_keys<'a>(
    table: &'a TransitionTable,
    path: &'a AssignedPath,
) -> Vec<(StepKey, GadgetIndex, &'a PermutationGadget)> {
    path.path()
        .iter()
        .zip(path.gadgets())
        .map(|((stage, transition), gadget_index)| {
            let transition_ref = TransitionRef {
                stage: stage
                    .try_into()
                    .expect("stage index should fit in transition ref"),
                transition,
            };
            (
                step_key_for_transition(table, transition_ref),
                *gadget_index,
                table.transition(transition_ref).gadget(*gadget_index),
            )
        })
        .collect()
}

fn step_key_for_transition(table: &TransitionTable, transition: TransitionRef) -> StepKey {
    let record = table.transition(transition);
    (
        usize::from(transition.stage),
        table.state_as_one_based_tuple(record.input()),
        table.state_as_one_based_tuple(record.output()),
    )
}

fn node_json_payload(
    stage: &usize,
    input: &StateTuple,
    output: &StateTuple,
    gadgets: &[PermutationGadget],
    child_ids: Vec<String>,
) -> Value {
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
