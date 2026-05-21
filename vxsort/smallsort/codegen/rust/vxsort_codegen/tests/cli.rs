use clap::Parser;
use std::process::Command;
use vxsort_codegen::{
    ArchArg, CliArgs, DTypeArg, RunConfig, RuntimeUiArg, build_dry_run_summary, build_run_summary,
    parse_run_config,
};

#[test]
fn parses_python_compatible_solve_flags() {
    let args = CliArgs::try_parse_from([
        "vxsort-codegen",
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
        "--target-cpu",
        "ZEN4,ADL-P",
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
            gadget_depth: 2,
            natural_order: true,
            retroactive_input: true,
            max_gadget_solutions: 3,
            target_cpus: vec!["ZEN4".to_owned(), "ADL-P".to_owned()],
            estimate: true,
            max_waves: Some(1),
            wave_attempts: 100,
            wave_outputs: 7,
            worker_count: 3,
            runtime_ui: RuntimeUiArg::None,
            dry_run: false,
        }
    );
}

#[test]
fn dry_run_summary_reports_stages_initial_state_and_candidate_counts() {
    let args = CliArgs::try_parse_from([
        "vxsort-codegen",
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

    let summary = build_dry_run_summary(&config).expect("dry-run summary should be built");

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
    let output = Command::new(env!("CARGO_BIN_EXE_vxsort_codegen"))
        .args([
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
fn run_summary_drives_sync_wave_loop() {
    let args = CliArgs::try_parse_from([
        "vxsort-codegen",
        "--vector-machine",
        "AVX2",
        "--datatype",
        "i64",
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
    let config = parse_run_config(args).expect("run config should be built");

    let summary = build_run_summary(&config).expect("run summary should be built");

    assert_eq!(summary.wave_count, 1);
    assert_eq!(summary.scored_paths, 0);
    assert_eq!(summary.best_score, None);
    assert_eq!(summary.waves.len(), 1);
    assert_eq!(summary.waves[0].target_stage, 0);
    assert_eq!(summary.waves[0].attempts, 0);
    assert_eq!(summary.waves[0].scored_paths, 0);
}

#[test]
fn binary_non_dry_run_prints_sync_summary() {
    let output = Command::new(env!("CARGO_BIN_EXE_vxsort_codegen"))
        .args([
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
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
    assert!(stdout.contains("vxsort-codegen sync run"));
    assert!(stdout.contains("waves: 1"));
    assert!(stdout.contains("scored paths: 0"));
    assert!(stdout.contains("wave 0: target stage 0, attempts 0"));
}

#[test]
fn binary_textual_runtime_prints_progress_events() {
    let output = Command::new(env!("CARGO_BIN_EXE_vxsort_codegen"))
        .args([
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
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

#[test]
fn run_config_can_be_cloned_for_background_engine_thread() {
    let args = CliArgs::try_parse_from([
        "vxsort-codegen",
        "--vector-machine",
        "AVX2",
        "--datatype",
        "i64",
        "--runtime-ui",
        "textual",
    ])
    .expect("run flags should parse");
    let config = parse_run_config(args).expect("run config should be built");

    let cloned = config.clone();

    assert_eq!(cloned, config);
}
