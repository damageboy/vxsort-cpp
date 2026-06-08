use bitonic_codegen::{
    ArchArg, CliArgs, CliCommand, DTypeArg, FetchUicaDataConfig, PruneScoreArg, RunConfig,
    RuntimeUiArg, WorkerBackendArg, build_dry_run_summary, fetch_uica_data, parse_run_config,
    run_solver_with_session, runtime::NullRuntimeSession,
};
use clap::Parser;
use serde_json::Value;
use std::fs;
use std::process::Command;

#[test]
fn parses_python_compatible_solve_flags() {
    let args = CliArgs::try_parse_from([
        "bitonic-codegen",
        "solve",
        "--vector-machine",
        "AVX2",
        "--datatype",
        "i64",
        "--depth-limit",
        "3",
        "--gadget-depth",
        "2",
        "--top-k",
        "5",
        "--max-waves",
        "1",
        "--wave-attempts",
        "100",
        "--wave-outputs",
        "7",
        "--workers",
        "3",
        "--runtime-ui",
        "none",
        "--runtime-trace",
        "trace.jsonl",
        "--target-cpu",
        "ZEN4,ADL-P",
        "--max-gadget-solutions",
        "7",
        "--estimate",
        "--natural-order",
        "--retroactive-input",
    ])
    .expect("python-compatible flags should parse");

    let config = parse_run_config(args).expect("run config should be built");

    assert_eq!(
        config,
        RunConfig {
            num_vecs: 2,
            arch: ArchArg::Avx2,
            dtype: DTypeArg::I64,
            depth_limit: Some(3),
            top_k: Some(5),
            prune_score: PruneScoreArg::Rough,
            gadget_depth: 2,
            natural_order: true,
            retroactive_input: true,
            max_gadget_solutions: 7,
            target_cpus: vec!["ZEN4".to_owned(), "ADL-P".to_owned()],
            estimate: true,
            output_path: None,
            uica_data_dir: std::path::PathBuf::from("uica-data"),
            max_waves: Some(1),
            wave_attempts: 100,
            wave_outputs: 7,
            worker_count: 3,
            worker_backend: WorkerBackendArg::Auto,
            runtime_ui: RuntimeUiArg::None,
            runtime_trace_path: Some(std::path::PathBuf::from("trace.jsonl")),
            dry_run: false,
        }
    );
}

#[test]
fn verify_parses_json_input_top_k_and_workers() {
    let args = CliArgs::try_parse_from([
        "bitonic-codegen",
        "verify",
        "--input",
        "fixture.json",
        "--top-k",
        "1",
        "--workers",
        "1",
    ])
    .expect("verify flags should parse");

    let CliCommand::Verify(verify_args) = args.command else {
        panic!("expected verify command");
    };

    assert_eq!(verify_args.input, std::path::PathBuf::from("fixture.json"));
    assert_eq!(verify_args.top_k, Some(1));
    assert_eq!(verify_args.workers, 1);
}

#[test]
fn solve_parses_uica_prune_score_mode() {
    let args = CliArgs::try_parse_from([
        "bitonic-codegen",
        "solve",
        "--vector-machine",
        "AVX2",
        "--datatype",
        "i64",
        "--target-cpu",
        "SKL",
        "--top-k",
        "1",
        "--prune-score",
        "uica",
    ])
    .expect("uica prune score mode should parse");

    let config = parse_run_config(args).expect("run config should be built");

    assert_eq!(config.prune_score, PruneScoreArg::Uica);
}

#[test]
fn verify_defaults_workers_to_zero() {
    let args = CliArgs::try_parse_from(["bitonic-codegen", "verify", "--input", "fixture.json"])
        .expect("verify input should parse");

    let CliCommand::Verify(verify_args) = args.command else {
        panic!("expected verify command");
    };

    assert_eq!(verify_args.top_k, None);
    assert_eq!(verify_args.workers, 0);
}

#[test]
fn estimate_parses_json_input_target_top_k_and_output_dir() {
    let args = CliArgs::try_parse_from([
        "bitonic-codegen",
        "estimate",
        "--input",
        "solutions.json",
        "--target-cpu",
        "SKL",
        "--top-k",
        "5",
        "--output-dir",
        "/tmp/vxsort-uica-estimate",
    ])
    .expect("estimate flags should parse");

    let CliCommand::Estimate(estimate_args) = args.command else {
        panic!("expected estimate command");
    };

    assert_eq!(
        estimate_args.input,
        std::path::PathBuf::from("solutions.json")
    );
    assert_eq!(estimate_args.target_cpu, "SKL");
    assert_eq!(
        estimate_args.uica_data_dir,
        std::path::PathBuf::from("uica-data")
    );
    assert_eq!(estimate_args.top_k, Some(5));
    assert_eq!(
        estimate_args.output_dir,
        Some(std::path::PathBuf::from("/tmp/vxsort-uica-estimate"))
    );
    assert!(!estimate_args.no_traces);
}

#[test]
fn binary_verify_requires_input() {
    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .arg("verify")
        .output()
        .expect("binary should run");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf-8");
    assert!(stderr.contains("--input"), "stderr was:\n{stderr}");
}

#[test]
fn binary_verify_succeeds_for_unique_gadget_json() {
    let input_path = write_unique_first_gadget_fixture("fixture_2xAVX2_i64.json");

    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .args([
            "verify",
            "--input",
            input_path.to_str().expect("fixture path should be utf-8"),
            "--top-k",
            "1",
            "--workers",
            "1",
        ])
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    assert!(stdout.contains("All 1 paths verified correct."));

    let _ = fs::remove_file(input_path);
}

#[test]
fn binary_verify_fails_for_incomplete_json_path() {
    let input_path = temp_named_output_path("verify-incomplete", "json");
    fs::write(&input_path, incomplete_sort_json()).expect("fixture should be written");

    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .args([
            "verify",
            "--input",
            input_path.to_str().expect("fixture path should be utf-8"),
            "--top-k",
            "1",
            "--workers",
            "1",
        ])
        .output()
        .expect("binary should run");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf-8");
    assert!(
        stderr.contains("VERIFICATION FAILED"),
        "stderr was:\n{stderr}"
    );
    assert!(stderr.contains("Path 1"), "stderr was:\n{stderr}");

    let _ = fs::remove_file(input_path);
}

#[test]
fn solve_rejects_export_only_primitive_aliases() {
    let args = CliArgs::try_parse_from([
        "bitonic-codegen",
        "solve",
        "--vector-machine",
        "AVX2",
        "--datatype",
        "u64",
        "--gadget-depth",
        "1",
        "--dry-run",
    ])
    .expect("u64 should be accepted by CLI metadata parser");

    let error = parse_run_config(args).expect_err("u64 solve is not supported yet");

    assert!(error.contains("supports only i32 and i64"));
    assert!(error.contains("u64"));
}

#[test]
fn dry_run_summary_reports_stages_initial_state_and_candidate_counts() {
    let args = CliArgs::try_parse_from([
        "bitonic-codegen",
        "solve",
        "--vector-machine",
        "AVX2",
        "--datatype",
        "i64",
        "--gadget-depth",
        "2",
        "--dry-run",
    ])
    .expect("dry-run flags should parse");
    let config = parse_run_config(args).expect("run config should be built");

    let summary = build_dry_run_summary(&config).expect("dry-solver should run");

    assert_eq!(summary.elements_per_vector, 4);
    assert_eq!(summary.total_elements, 8);
    assert_eq!(summary.initial_top, vec![1, 3, 5, 7]);
    assert_eq!(summary.initial_bottom, vec![2, 4, 6, 8]);
    assert_eq!(summary.stages.len(), 6);
    assert_eq!(
        summary.stages[0].pairs,
        vec![(1, 2), (3, 4), (5, 6), (7, 8)]
    );
    assert_eq!(
        summary.stages[3].pairs,
        vec![(1, 8), (2, 7), (3, 6), (4, 5)]
    );
    assert_eq!(summary.shallow_candidate_count, 224);
    assert_eq!(summary.deep_candidate_count, 8928);
}

#[test]
fn binary_dry_run_prints_summary() {
    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .args([
            "solve",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--gadget-depth",
            "2",
            "--dry-run",
        ])
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    assert!(stdout.contains("arch: AVX2"));
    assert!(stdout.contains("dtype: i64"));
    assert!(stdout.contains("elements per vector: 4"));
    assert!(stdout.contains("stages: 6"));
    assert!(stdout.contains("candidates: 224 shallow, 8928 deep"));
}

#[test]
fn binary_help_uses_ansi_color() {
    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .arg("--help")
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    assert!(
        stdout.contains("\x1b[92m") || stdout.contains("\x1b[1;92m"),
        "expected visible ANSI foreground color in help output, got:\n{stdout}"
    );
}

#[test]
fn run_summary_drives_sync_wave_loop() {
    let data_dir = fixture_uica_data_dir("empty-summary");
    let args = CliArgs::try_parse_from([
        "bitonic-codegen",
        "solve",
        "--vector-machine",
        "AVX2",
        "--datatype",
        "i64",
        "--target-cpu",
        "SKL",
        "--top-k",
        "1",
        "--uica-data-dir",
        data_dir.to_str().expect("fixture path should be utf-8"),
        "--max-waves",
        "1",
        "--wave-attempts",
        "0",
        "--wave-outputs",
        "1",
        "--runtime-ui",
        "none",
    ])
    .expect("run flags should parse");
    let mut config = parse_run_config(args).expect("run config should be built");
    config.worker_backend = WorkerBackendArg::InProcess;

    let mut session = NullRuntimeSession;
    let summary = run_solver_with_session(&config, &mut session).expect("solver should run");

    assert_eq!(summary.wave_count, 1);
    assert_eq!(summary.scored_paths, 0);
    assert_eq!(summary.best_score, None);
    assert_eq!(summary.waves.len(), 1);
    assert_eq!(summary.waves[0].target_stage, 0);
    assert_eq!(summary.waves[0].attempts, 0);
    assert_eq!(summary.waves[0].scored_paths, 0);
}

#[test]
fn run_summary_writes_runtime_trace_file() {
    let data_dir = fixture_uica_data_dir("runtime-trace");
    let trace_path = std::env::temp_dir().join(format!(
        "vxsort-runtime-trace-cli-{}.jsonl",
        std::process::id()
    ));
    let _ = fs::remove_file(&trace_path);
    let args = CliArgs::try_parse_from([
        "bitonic-codegen",
        "solve",
        "--vector-machine",
        "AVX2",
        "--datatype",
        "i64",
        "--target-cpu",
        "SKL",
        "--top-k",
        "1",
        "--uica-data-dir",
        data_dir.to_str().expect("fixture path should be utf-8"),
        "--max-waves",
        "1",
        "--wave-attempts",
        "5",
        "--wave-outputs",
        "100",
        "--workers",
        "2",
        "--runtime-ui",
        "none",
        "--runtime-trace",
        trace_path.to_str().expect("trace path should be utf-8"),
    ])
    .expect("run flags should parse");
    let mut config = parse_run_config(args).expect("run config should be built");
    config.worker_backend = WorkerBackendArg::InProcess;

    let mut session = NullRuntimeSession;
    run_solver_with_session(&config, &mut session).expect("solver should run");

    let trace_contents = fs::read_to_string(&trace_path).expect("trace file should be readable");
    let events = trace_contents
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("trace line should be JSON"))
        .collect::<Vec<_>>();
    let _ = fs::remove_file(&trace_path);

    assert!(events.iter().any(|event| event["event"] == "run_started"));
    assert!(events.iter().any(|event| event["event"] == "job_submitted"));
    assert!(
        events
            .iter()
            .any(|event| event["event"] == "worker_counters")
    );
    assert!(
        events
            .iter()
            .any(|event| event["event"] == "live_full_scoring_snapshot")
    );
}

#[test]
fn binary_non_dry_run_prints_sync_summary() {
    let data_dir = fixture_uica_data_dir("binary-summary");
    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .args([
            "solve",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--target-cpu",
            "SKL",
            "--top-k",
            "1",
            "--uica-data-dir",
            data_dir.to_str().expect("fixture path should be utf-8"),
            "--max-waves",
            "1",
            "--wave-attempts",
            "0",
            "--wave-outputs",
            "1",
            "--runtime-ui",
            "none",
        ])
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    assert!(stdout.contains("bitonic-codegen sync run"));
    assert!(stdout.contains("waves: 1"));
    assert!(stdout.contains("scored paths: 0"));
    assert!(stdout.contains("wave 0: target stage 0, attempts 0"));
}

#[test]
fn binary_writes_python_compatible_json_to_output_path() {
    let data_dir = fixture_uica_data_dir("json-output");
    let output_path = temp_output_path("json");
    let _ = fs::remove_file(&output_path);

    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .args([
            "solve",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--target-cpu",
            "SKL",
            "--top-k",
            "1",
            "--uica-data-dir",
            data_dir.to_str().expect("fixture path should be utf-8"),
            "--max-waves",
            "1",
            "--wave-attempts",
            "0",
            "--wave-outputs",
            "1",
            "--runtime-ui",
            "none",
            "--output-path",
            output_path.to_str().expect("temp path should be utf-8"),
        ])
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let contents = fs::read_to_string(&output_path).expect("output json should be written");
    let json: serde_json::Value = serde_json::from_str(&contents).expect("output should be json");
    assert_eq!(json["vector_machine"], "AVX2");
    assert_eq!(json["primitive_type"], "i64");
    assert_eq!(json["num_vecs"], 2);
    assert_eq!(json["roots"], serde_json::json!([]));
    assert_eq!(json["nodes"], serde_json::json!({}));
    assert_eq!(json["paths"], serde_json::json!([]));

    let _ = fs::remove_file(output_path);
}

#[test]
fn binary_solve_rejects_direct_asm_output_path() {
    let output_path = temp_output_path("asm");
    let _ = fs::remove_file(&output_path);

    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .args([
            "solve",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--target-cpu",
            "SKL",
            "--top-k",
            "1",
            "--asm-output-path",
            output_path.to_str().expect("temp path should be utf-8"),
        ])
        .output()
        .expect("binary should run");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf-8");
    assert!(
        stderr.contains("--asm-output-path"),
        "stderr was:\n{stderr}"
    );

    let _ = fs::remove_file(output_path);
}

#[test]
fn binary_converts_solution_json_to_assembly() {
    let input_path = temp_named_output_path("json-to-asm-input", "json");
    let output_path = temp_named_output_path("json-to-asm-output", "asm");
    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&output_path);
    fs::write(
        &input_path,
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
      "output_state": {"top": [1, 2, 5, 6], "bottom": [3, 4, 7, 8]},
      "gadgets": [
        {
          "top_instructions": [],
          "bottom_instructions": [],
          "unified_instructions": [],
          "top_output_index": -1,
          "bottom_output_index": -1
        },
        {
          "top_instructions": [
            {
              "name": "_mm256_permute4x64_epi64",
              "args": {"a": "top", "imm8": 78}
            }
          ],
          "bottom_instructions": [],
          "unified_instructions": [
            {
              "name": "_mm256_permute4x64_epi64",
              "args": {"a": "top", "imm8": 78}
            }
          ],
          "top_output_index": 0,
          "bottom_output_index": -1
        }
      ],
      "gadget_count": 2,
      "children": []
    }
  },
  "paths": [
    {
      "steps": [
        {"node_id": "n0", "gadget_index": 1}
      ]
    }
  ]
}"#,
    )
    .expect("input JSON should be written");

    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .args([
            "json-to-asm",
            "--input",
            input_path.to_str().expect("input path should be utf-8"),
            "--output",
            output_path.to_str().expect("output path should be utf-8"),
        ])
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let contents = fs::read_to_string(&output_path).expect("output asm should be written");
    assert!(contents.contains("section .text"));
    assert!(contents.contains("vpermq"));
    assert!(contents.contains("ret"));

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_file(output_path);
}

#[test]
fn binary_estimate_prints_ranked_table_and_writes_trace() {
    let input_path = temp_named_output_path("estimate-input", "json");
    let output_dir =
        std::env::temp_dir().join(format!("vxsort estimate output {}", std::process::id()));
    let data_dir = fixture_uica_data_dir_with_vpermq("estimate");
    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_dir_all(&output_dir);
    fs::write(&input_path, estimate_vpermq_json()).expect("estimate JSON should be written");

    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .args([
            "estimate",
            "--input",
            input_path.to_str().expect("input path should be utf-8"),
            "--target-cpu",
            "SKL",
            "--uica-data-dir",
            data_dir.to_str().expect("fixture path should be utf-8"),
            "--output-dir",
            output_dir.to_str().expect("output path should be utf-8"),
        ])
        .output()
        .expect("binary should run");

    assert!(
        output.status.success(),
        "stderr was:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    assert!(stdout.contains("uiCA estimate"), "stdout was:\n{stdout}");
    assert!(stdout.contains("target cpu: SKL"), "stdout was:\n{stdout}");
    assert!(stdout.contains("output dir:"), "stdout was:\n{stdout}");
    assert!(stdout.contains("Rank"), "stdout was:\n{stdout}");
    assert!(stdout.contains("TP"), "stdout was:\n{stdout}");
    assert!(stdout.contains("Trace"), "stdout was:\n{stdout}");
    assert!(stdout.contains("╭"), "stdout was:\n{stdout}");
    assert!(
        !stdout.contains("Rank  Path         TP"),
        "stdout was:\n{stdout}"
    );
    assert!(
        stdout.contains("solution_000_trace.html"),
        "stdout was:\n{stdout}"
    );
    assert!(
        stdout.contains("\x1b]8;;file://localhost/"),
        "stdout was:\n{stdout}"
    );
    assert!(stdout.contains("%20"), "stdout was:\n{stdout}");
    assert!(
        stdout.contains("\x1b\\solution_000_trace.html\x1b]8;;"),
        "stdout was:\n{stdout}"
    );
    assert!(
        !stdout.contains("subl://open?url=file://"),
        "stdout was:\n{stdout}"
    );
    assert!(output_dir.join("solution_000_trace.html").is_file());

    let _ = fs::remove_file(input_path);
    let _ = fs::remove_dir_all(output_dir);
    let _ = fs::remove_dir_all(data_dir);
}

#[test]
fn binary_multi_target_writes_cpu_suffixed_outputs() {
    let data_dir = fixture_uica_data_dir_with_arches("multi-target-output", &["SKL", "TGL"]);
    let json_path = temp_named_output_path("multi-target", "json");
    let expected_json = [
        suffixed_output_path(&json_path, "SKL"),
        suffixed_output_path(&json_path, "TGL"),
    ];
    let _ = fs::remove_file(&json_path);
    for path in &expected_json {
        let _ = fs::remove_file(path);
    }

    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .args([
            "solve",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--target-cpu",
            "SKL,TGL",
            "--top-k",
            "1",
            "--uica-data-dir",
            data_dir.to_str().expect("fixture path should be utf-8"),
            "--max-waves",
            "1",
            "--wave-attempts",
            "0",
            "--wave-outputs",
            "1",
            "--runtime-ui",
            "none",
            "--output-path",
            json_path.to_str().expect("json path should be utf-8"),
        ])
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    assert!(!json_path.exists());
    for path in &expected_json {
        assert!(path.exists(), "expected suffixed output {}", path.display());
        let _ = fs::remove_file(path);
    }
}

#[test]
fn binary_textual_runtime_prints_progress_events() {
    let data_dir = fixture_uica_data_dir("textual-runtime");
    let output = Command::new(env!("CARGO_BIN_EXE_bitonic_codegen"))
        .args([
            "solve",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--target-cpu",
            "SKL",
            "--top-k",
            "1",
            "--uica-data-dir",
            data_dir.to_str().expect("fixture path should be utf-8"),
            "--max-waves",
            "1",
            "--wave-attempts",
            "0",
            "--wave-outputs",
            "1",
            "--runtime-ui",
            "textual",
        ])
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    assert!(stdout.contains("runtime: run started (6 stages)"));
    assert!(stdout.contains("runtime: wave 0 started (target stage 0)"));
    assert!(stdout.contains("runtime: wave 0 finished"));
}

fn temp_output_path(extension: &str) -> std::path::PathBuf {
    temp_named_output_path("empty-run", extension)
}

fn temp_named_output_path(name: &str, extension: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "vxsort-rust-solutions-{}-{name}.{extension}",
        std::process::id()
    ))
}

fn suffixed_output_path(path: &std::path::Path, target_cpu: &str) -> std::path::PathBuf {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("test path should have a utf-8 stem");
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .expect("test path should have a utf-8 extension");
    path.with_file_name(format!("{stem}.{target_cpu}.{extension}"))
}

fn python_fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests")
        .join(name)
}

fn write_unique_first_gadget_fixture(name: &str) -> std::path::PathBuf {
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

    let input_path = temp_named_output_path(name.trim_end_matches(".json"), "json");
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

fn estimate_vpermq_json() -> &'static str {
    r#"{
  "natural_order": true,
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
        {
          "top_instructions": [
            {
              "name": "_mm256_permute4x64_epi64",
              "args": {"a": "top", "imm8": 78}
            }
          ],
          "bottom_instructions": [],
          "unified_instructions": [
            {
              "name": "_mm256_permute4x64_epi64",
              "args": {"a": "top", "imm8": 78}
            }
          ],
          "top_output_index": 0,
          "bottom_output_index": -1
        }
      ],
      "gadget_count": 1,
      "children": []
    }
  }
}"#
}

#[test]
fn run_config_can_be_cloned_for_background_engine_thread() {
    let args = CliArgs::try_parse_from([
        "bitonic-codegen",
        "solve",
        "--vector-machine",
        "AVX2",
        "--datatype",
        "i64",
        "--target-cpu",
        "SKL",
        "--top-k",
        "1",
        "--runtime-ui",
        "textual",
    ])
    .expect("run flags should parse");
    let config = parse_run_config(args).expect("run config should be built");

    let cloned = config.clone();

    assert_eq!(cloned, config);
}

#[test]
fn run_config_requires_target_cpu_for_scored_solving() {
    let args = CliArgs::try_parse_from([
        "bitonic-codegen",
        "solve",
        "--vector-machine",
        "AVX2",
        "--datatype",
        "i64",
        "--top-k",
        "1",
    ])
    .expect("run flags should parse");

    let error = parse_run_config(args).expect_err("target CPU should be required");

    assert!(error.contains("--target-cpu"));
}

#[test]
fn run_config_requires_top_k_for_scored_solving() {
    let args = CliArgs::try_parse_from([
        "bitonic-codegen",
        "solve",
        "--vector-machine",
        "AVX2",
        "--datatype",
        "i64",
        "--target-cpu",
        "SKL",
    ])
    .expect("run flags should parse");

    let error = parse_run_config(args).expect_err("top-k should be required");

    assert!(error.contains("--top-k"));
}

#[test]
fn fetch_uica_data_copies_manifest_and_requested_arch_pack_from_base_url() {
    let source = std::env::temp_dir().join(format!("vxsort-fetch-source-{}", std::process::id()));
    let destination =
        std::env::temp_dir().join(format!("vxsort-fetch-dest-{}", std::process::id()));
    let _ = fs::remove_dir_all(&source);
    let _ = fs::remove_dir_all(&destination);
    fs::create_dir_all(source.join("arch")).expect("source arch dir should be created");
    fs::write(
        source.join("manifest.json"),
        r#"{
  "schema_version": "uica-datapack-manifest-v2",
  "uipack_version": 9,
  "architectures": {
    "SKL": {
      "path": "arch/SKL.uipack",
      "size": 4,
      "checksum_kind": "fnv1a64",
      "checksum": "0000000000000000",
      "record_count": 0
    }
  }
}"#,
    )
    .expect("manifest should be written");
    fs::write(source.join("arch/SKL.uipack"), b"pack").expect("pack should be written");

    let report = fetch_uica_data(&FetchUicaDataConfig {
        target_cpus: vec!["skl".to_owned()],
        data_dir: destination.clone(),
        base_url: format!("file://{}", source.display()),
    })
    .expect("fetch should succeed");

    assert_eq!(report.arches, vec!["SKL"]);
    assert_eq!(
        fs::read_to_string(destination.join("manifest.json")).expect("manifest should exist"),
        fs::read_to_string(source.join("manifest.json")).expect("source manifest should exist")
    );
    assert_eq!(
        fs::read(destination.join("arch/SKL.uipack")).expect("pack should exist"),
        b"pack"
    );

    let _ = fs::remove_dir_all(source);
    let _ = fs::remove_dir_all(destination);
}

fn fixture_uica_data_dir(name: &str) -> std::path::PathBuf {
    fixture_uica_data_dir_with_arches(name, &["SKL"])
}

fn fixture_uica_data_dir_with_vpermq(name: &str) -> std::path::PathBuf {
    use std::collections::BTreeMap;
    use uica_data::{
        DATAPACK_MANIFEST_SCHEMA_VERSION, DATAPACK_SCHEMA_VERSION, DataPack, InstructionRecord,
        PerfRecord, UIPACK_VERSION, encode_uipack, read_uipack_header,
    };

    let dir = std::env::temp_dir().join(format!(
        "vxsort-uica-data-{name}-vpermq-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("arch")).expect("fixture arch dir should be created");
    let ports = vec![
        "0".to_owned(),
        "1".to_owned(),
        "5".to_owned(),
        "6".to_owned(),
    ];
    let pack = DataPack {
        schema_version: DATAPACK_SCHEMA_VERSION.to_owned(),
        all_ports: ports.clone(),
        alu_ports: ports.clone(),
        instructions: vec![InstructionRecord {
            arch: "SKL".to_owned(),
            iform: "VPERMQ_YMMqq_YMMqq_IMMb".to_owned(),
            string: "VPERMQ (YMM, YMM, I8)".to_owned(),
            all_ports: ports.clone(),
            alu_ports: ports,
            locked: false,
            xml_attrs: BTreeMap::new(),
            imm_zero: false,
            perf: PerfRecord {
                operands: vec![],
                latencies: vec![],
                uops: 1,
                retire_slots: 1,
                uops_mite: 1,
                uops_ms: 0,
                tp: Some(1.0),
                ports: BTreeMap::from([("5".to_owned(), 1)]),
                variants: BTreeMap::new(),
                div_cycles: 0,
                may_be_eliminated: false,
                complex_decoder: false,
                n_available_simple_decoders: 0,
                lcp_stall: false,
                implicit_rsp_change: 0,
                can_be_used_by_lsd: true,
                cannot_be_in_dsb_due_to_jcc_erratum: false,
                no_micro_fusion: false,
                no_macro_fusion: false,
                macro_fusible_with: vec![],
            },
        }],
    };
    let bytes = encode_uipack(&pack, "SKL").expect("fixture uipack should encode");
    let header = read_uipack_header(&bytes).expect("fixture uipack header should decode");
    fs::write(dir.join("arch/SKL.uipack"), &bytes).expect("fixture pack should be written");
    fs::write(
        dir.join("manifest.json"),
        format!(
            r#"{{
  "schema_version": "{}",
  "uipack_version": {},
  "architectures": {{
    "SKL": {{
      "path": "arch/SKL.uipack",
      "size": {},
      "checksum_kind": "fnv1a64",
      "checksum": "{:016x}",
      "record_count": {}
    }}
  }}
}}"#,
            DATAPACK_MANIFEST_SCHEMA_VERSION,
            UIPACK_VERSION,
            header.file_len,
            header.checksum,
            header.records_count
        ),
    )
    .expect("fixture manifest should be written");
    dir
}

fn fixture_uica_data_dir_with_arches(name: &str, arches: &[&str]) -> std::path::PathBuf {
    use uica_data::{
        DATAPACK_MANIFEST_SCHEMA_VERSION, DATAPACK_SCHEMA_VERSION, DataPack, UIPACK_VERSION,
        encode_uipack, read_uipack_header,
    };

    let dir = std::env::temp_dir().join(format!("vxsort-uica-data-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("arch")).expect("fixture arch dir should be created");
    let mut manifest_entries = Vec::new();
    for arch in arches {
        let pack = DataPack {
            schema_version: DATAPACK_SCHEMA_VERSION.to_owned(),
            all_ports: vec![
                "0".to_owned(),
                "1".to_owned(),
                "5".to_owned(),
                "6".to_owned(),
            ],
            alu_ports: vec![
                "0".to_owned(),
                "1".to_owned(),
                "5".to_owned(),
                "6".to_owned(),
            ],
            instructions: vec![],
        };
        let bytes = encode_uipack(&pack, arch).expect("fixture uipack should encode");
        let header = read_uipack_header(&bytes).expect("fixture uipack header should decode");
        fs::write(dir.join(format!("arch/{arch}.uipack")), &bytes)
            .expect("fixture pack should be written");
        manifest_entries.push(format!(
            r#"    "{arch}": {{
      "path": "arch/{arch}.uipack",
      "size": {},
      "checksum_kind": "fnv1a64",
      "checksum": "{:016x}",
      "record_count": {}
    }}"#,
            header.file_len, header.checksum, header.records_count
        ));
    }
    fs::write(
        dir.join("manifest.json"),
        format!(
            r#"{{
  "schema_version": "{}",
  "uipack_version": {},
  "architectures": {{
{}
  }}
}}"#,
            DATAPACK_MANIFEST_SCHEMA_VERSION,
            UIPACK_VERSION,
            manifest_entries.join(",\n")
        ),
    )
    .expect("fixture manifest should be written");
    dir
}
