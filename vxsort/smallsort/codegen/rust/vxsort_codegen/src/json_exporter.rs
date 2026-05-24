use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io;
use std::path::Path;

use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget};
use serde_json::{Map, Value, json};

use crate::scoring::AssignedPath;
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

pub fn write_solution_json_for_assigned_paths(
    output_path: impl AsRef<Path>,
    metadata: &SolutionJsonMetadata,
    paths: &[AssignedPath],
) -> Result<(), io::Error> {
    let file = File::create(output_path)?;
    serde_json::to_writer_pretty(file, &solution_json_for_assigned_paths(metadata, paths))?;
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

pub fn solution_json_for_assigned_paths(
    metadata: &SolutionJsonMetadata,
    paths: &[AssignedPath],
) -> Value {
    let mut node_ids = BTreeMap::<StepKey, String>::new();
    let mut node_order = Vec::<StepKey>::new();
    let mut children = BTreeMap::<StepKey, BTreeSet<String>>::new();
    let mut node_gadgets = BTreeMap::<StepKey, Vec<PermutationGadget>>::new();
    let mut root_ids = Vec::new();

    for path in paths {
        if path.steps().is_empty() {
            continue;
        }

        for step in path.steps() {
            let key = (step.stage(), step.input().clone(), step.output().clone());
            ensure_node_id(&key, &mut node_ids, &mut node_order);
            let gadgets = node_gadgets.entry(key).or_default();
            if !gadgets.contains(step.gadget()) {
                gadgets.push(step.gadget().clone());
            }
        }

        let root_key = (
            path.steps()[0].stage(),
            path.steps()[0].input().clone(),
            path.steps()[0].output().clone(),
        );
        root_ids.push(
            node_ids
                .get(&root_key)
                .expect("assigned path root should have an assigned node id")
                .clone(),
        );

        for window in path.steps().windows(2) {
            let parent_key = (
                window[0].stage(),
                window[0].input().clone(),
                window[0].output().clone(),
            );
            let child_key = (
                window[1].stage(),
                window[1].input().clone(),
                window[1].output().clone(),
            );
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
        .filter(|path| !path.steps().is_empty())
        .map(|path| {
            let steps = path
                .steps()
                .iter()
                .map(|step| {
                    let key = (step.stage(), step.input().clone(), step.output().clone());
                    let node_id = node_ids
                        .get(&key)
                        .expect("assigned path step should have a node id");
                    let gadgets = node_gadgets
                        .get(&key)
                        .expect("assigned path step should have node gadgets");
                    let gadget_index = gadgets
                        .iter()
                        .position(|gadget| gadget == step.gadget())
                        .expect("assigned path gadget should be present in node gadgets");
                    json!({
                        "node_id": node_id,
                        "gadget_index": gadget_index,
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

    node_json_payload(stage, input, output, gadgets, child_ids)
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
