use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use bitonic_codegen::verifier::{
    VerifyJsonOptions, compute_recursion_boundaries, num_verify_chunks, verify_solution_json,
};

#[test]
fn recursion_chunk_counts_match_python_verifier() {
    assert_eq!(
        compute_recursion_boundaries(4).unwrap(),
        Vec::<usize>::new()
    );
    assert_eq!(compute_recursion_boundaries(8).unwrap(), vec![2]);
    assert_eq!(compute_recursion_boundaries(16).unwrap(), vec![2, 5]);

    assert_eq!(num_verify_chunks(4).unwrap(), 1);
    assert_eq!(num_verify_chunks(8).unwrap(), 3);
    assert_eq!(num_verify_chunks(16).unwrap(), 7);
}

#[test]
fn verifies_unique_gadget_avx2_i64_json() {
    let input_path = write_unique_first_gadget_fixture("fixture_2xAVX2_i64.json");

    let report = verify_solution_json(
        &input_path,
        VerifyJsonOptions {
            top_k: Some(1),
            workers: 1,
        },
    )
    .expect("fixture should verify");

    assert_eq!(report.path_count, 1);
    assert_eq!(report.failure_count(), 0);
    assert_eq!(report.chunks_per_path, 3);

    let _ = fs::remove_file(input_path);
}

#[test]
fn incomplete_json_path_fails_with_counterexample() {
    let input_path = temp_json_path("incomplete-path");
    fs::write(&input_path, incomplete_sort_json()).expect("fixture should be written");

    let report = verify_solution_json(
        &input_path,
        VerifyJsonOptions {
            top_k: Some(1),
            workers: 1,
        },
    )
    .expect("verification should complete");

    assert_eq!(report.path_count, 1);
    assert_eq!(report.failure_count(), 1);
    assert_eq!(report.failures[0].path_index, 0);
    assert_eq!(report.failures[0].counterexample.len(), 8);

    let _ = fs::remove_file(input_path);
}

#[test]
fn rejects_unsupported_verifier_dtype() {
    let input_path = temp_json_path("unsupported-dtype");
    fs::write(
        &input_path,
        incomplete_sort_json().replace(r#""primitive_type": "i64""#, r#""primitive_type": "u64""#),
    )
    .expect("fixture should be written");

    let error = verify_solution_json(
        &input_path,
        VerifyJsonOptions {
            top_k: Some(1),
            workers: 1,
        },
    )
    .expect_err("u64 verify should be rejected for now");

    assert!(error.contains("supports only i32 and i64"));

    let _ = fs::remove_file(input_path);
}

#[test]
fn rejects_ambiguous_multi_gadget_json() {
    let input_path = temp_json_path("ambiguous-gadget");
    fs::write(
        &input_path,
        incomplete_sort_json().replace(
            r#"{"top_instructions": [], "bottom_instructions": []}"#,
            r#"{"top_instructions": [], "bottom_instructions": []},
        {"top_instructions": [
          {"name": "_mm256_permute4x64_epi64", "args": {"a": "top", "imm8": 228}}
        ], "bottom_instructions": []}"#,
        ),
    )
    .expect("fixture should be written");

    let error = verify_solution_json(
        &input_path,
        VerifyJsonOptions {
            top_k: Some(1),
            workers: 1,
        },
    )
    .expect_err("python fixture has multiple gadgets per transition");

    assert!(error.contains("requires concrete one-gadget JSON"));

    let _ = fs::remove_file(input_path);
}

#[test]
fn verifies_concrete_paths_with_multiple_gadgets_per_node() {
    let input_path =
        write_concrete_first_gadget_fixture_with_extra_gadget("fixture_2xAVX2_i64.json");

    let report = verify_solution_json(
        &input_path,
        VerifyJsonOptions {
            top_k: None,
            workers: 1,
        },
    )
    .expect("explicit concrete path should verify despite extra node gadget");

    assert_eq!(report.path_count, 1);
    assert_eq!(report.failure_count(), 0);

    let _ = fs::remove_file(input_path);
}

#[test]
fn rejects_disconnected_json_path() {
    let input_path = write_unique_first_gadget_fixture("fixture_2xAVX2_i64.json");
    let mut json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&input_path).expect("fixture should be readable"))
            .expect("fixture should be JSON");
    json["nodes"]["n1"]["input_state"] =
        serde_json::json!({"top": [8, 6, 4, 2], "bottom": [7, 5, 3, 1]});
    fs::write(
        &input_path,
        serde_json::to_string_pretty(&json).expect("fixture should serialize"),
    )
    .expect("fixture should be written");

    let error = verify_solution_json(
        &input_path,
        VerifyJsonOptions {
            top_k: Some(1),
            workers: 1,
        },
    )
    .expect_err("disconnected path should be rejected before chunked proof");

    assert!(error.contains("path is disconnected"), "error was: {error}");

    let _ = fs::remove_file(input_path);
}

fn python_fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests")
        .join(name)
}

fn temp_json_path(name: &str) -> PathBuf {
    static NEXT_TEMP_ID: AtomicUsize = AtomicUsize::new(0);
    let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "vxsort-verify-{}-{id}-{name}.json",
        std::process::id()
    ))
}

fn write_unique_first_gadget_fixture(name: &str) -> PathBuf {
    let mut json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(python_fixture_path(name)).expect("python fixture should be readable"),
    )
    .expect("python fixture should be JSON");
    for node in json["nodes"]
        .as_object_mut()
        .expect("nodes should be an object")
        .values_mut()
    {
        let gadgets = node["gadgets"]
            .as_array_mut()
            .expect("gadgets should be an array");
        gadgets.truncate(1);
        node["gadget_count"] = serde_json::json!(gadgets.len());
    }

    let input_path = temp_json_path(name.trim_end_matches(".json"));
    fs::write(
        &input_path,
        serde_json::to_string_pretty(&json).expect("fixture should serialize"),
    )
    .expect("fixture should be written");
    input_path
}

fn write_concrete_first_gadget_fixture_with_extra_gadget(name: &str) -> PathBuf {
    let input_path = write_unique_first_gadget_fixture(name);
    let mut json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&input_path).expect("fixture should be readable"))
            .expect("fixture should be JSON");
    let mut steps = Vec::new();
    let mut node_id = json["roots"][0]
        .as_str()
        .expect("fixture should contain a root")
        .to_owned();
    loop {
        steps.push(serde_json::json!({
            "node_id": node_id,
            "gadget_index": 0,
        }));
        let children = json["nodes"][&node_id]["children"]
            .as_array()
            .expect("fixture node should contain children");
        let Some(child) = children.first() else {
            break;
        };
        node_id = child
            .as_str()
            .expect("child id should be a string")
            .to_owned();
    }
    let root_id = json["roots"][0]
        .as_str()
        .expect("fixture should contain a root")
        .to_owned();
    let gadgets = json["nodes"][&root_id]["gadgets"]
        .as_array_mut()
        .expect("root node should contain gadgets");
    gadgets.push(serde_json::json!({
        "top_instructions": [
            {"name": "_mm256_permute4x64_epi64", "args": {"a": "top", "imm8": 228}}
        ],
        "bottom_instructions": []
    }));
    json["nodes"][&root_id]["gadget_count"] = serde_json::json!(gadgets.len());
    json["paths"] = serde_json::json!([{ "steps": steps }]);
    fs::write(
        &input_path,
        serde_json::to_string_pretty(&json).expect("fixture should serialize"),
    )
    .expect("fixture should be written");
    input_path
}

fn incomplete_sort_json() -> &'static str {
    r#"{
  "natural_order": false,
  "vector_machine": "AVX2",
  "primitive_type": "i64",
  "num_vecs": 2,
  "roots": ["n0"],
  "nodes": {
    "n0": {
      "stage": 0,
      "input_state": {"top": [1, 3, 5, 7], "bottom": [2, 4, 6, 8]},
      "output_state": {"top": [1, 3, 5, 7], "bottom": [2, 4, 6, 8]},
      "gadgets": [
        {"top_instructions": [], "bottom_instructions": []}
      ],
      "gadget_count": 1,
      "children": []
    }
  }
}"#
}
