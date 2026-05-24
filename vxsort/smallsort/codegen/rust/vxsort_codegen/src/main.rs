use std::{
    io::IsTerminal,
    io::Write,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Instant,
};

use clap::Parser;
use vxsort_codegen::{
    CliArgs, CliCommand, DryRunSummary, EstimateReport, FetchUicaDataConfig, RunConfig, RunSummary,
    RuntimeUiArg, build_dry_run_summary, build_run_summary, build_run_summary_with_session,
    convert_solution_json_to_asm, estimate_solution_json, fetch_uica_data,
    instruction_stream::{
        InstructionBlock, LoweringOptions, ModeledInstruction, lower_assigned_paths,
        lower_solution_paths,
    },
    parse_run_config, read_solution_json,
    runtime::{ChannelRuntimeSession, LineRuntimeSession},
    runtime_tui::run_tui_event_loop,
    uica_scoring::{UiPackScorer, uops_key_for_instruction},
    verify_solution_json,
};

fn main() {
    let args = CliArgs::parse();
    match &args.command {
        CliCommand::Solve(_) => {}
        CliCommand::Verify(verify_args) => {
            let report = exit_on_error(verify_solution_json(
                &verify_args.input,
                verify_args.top_k,
                verify_args.workers,
            ));
            println!("All {} paths verified correct.", report.path_count);
            return;
        }
        CliCommand::Estimate(estimate_args) => {
            let report = exit_on_error(estimate_solution_json(
                &estimate_args.input,
                vxsort_codegen::EstimateOptions {
                    target_cpu: estimate_args.target_cpu.clone(),
                    uica_data_dir: estimate_args.uica_data_dir.clone(),
                    top_k: estimate_args.top_k,
                    output_dir: estimate_args.output_dir.clone(),
                    write_traces: !estimate_args.no_traces,
                },
            ));
            print_estimate_report(&report);
            return;
        }
        CliCommand::FetchUicaData(fetch_args) => {
            let report = exit_on_error(fetch_uica_data(&FetchUicaDataConfig {
                target_cpus: split_target_cpus(&fetch_args.target_cpu),
                data_dir: fetch_args.data_dir.clone(),
                base_url: fetch_args.base_url.clone(),
            }));
            println!("fetched uiCA manifest: {}", report.manifest_path.display());
            for arch in report.arches {
                println!("fetched uiCA arch pack: {arch}");
            }
            return;
        }
        CliCommand::JsonToAsm(json_to_asm_args) => {
            exit_on_error(convert_solution_json_to_asm(
                &json_to_asm_args.input,
                &json_to_asm_args.output,
            ));
            println!(
                "wrote solution assembly: {}",
                json_to_asm_args.output.display()
            );
            return;
        }
        CliCommand::ScoreJson(score_args) => {
            exit_on_error(score_solution_json(score_args));
            return;
        }
    }

    let config = exit_on_error(parse_run_config(args));

    if config.dry_run {
        let summary = exit_on_error(build_dry_run_summary(&config));
        print_dry_run_summary(&config, &summary);
        return;
    }

    let summary = run_with_runtime_ui(&config);
    print_run_summary(&config, &summary);
}

fn score_solution_json(args: &vxsort_codegen::ScoreJsonArgs) -> Result<(), String> {
    let imported = read_solution_json(&args.input)?;
    let stream = if imported.assigned_paths.is_empty() {
        lower_solution_paths(
            &imported.metadata,
            &imported.transition_table,
            &imported.paths,
            LoweringOptions::default(),
        )
    } else {
        lower_assigned_paths(
            &imported.metadata,
            &imported.assigned_paths,
            LoweringOptions::default(),
        )
    };
    let scorer = UiPackScorer::from_data_dir(&args.uica_data_dir, &args.target_cpu)?;
    let path_indices = match args.path_index {
        Some(index) => vec![index],
        None => (0..stream.blocks.len()).collect(),
    };

    println!(
        "score-json: {} paths imported from {}",
        stream.blocks.len(),
        args.input.display()
    );
    for path_index in path_indices {
        let block = stream
            .blocks
            .get(path_index)
            .ok_or_else(|| format!("path index {path_index} out of range"))?;
        let block = scoring_block(block, args.prefix_len);
        let scored_instruction_count = block
            .instructions
            .iter()
            .filter(|instruction| !instruction.mnemonic.eq_ignore_ascii_case("ret"))
            .filter(|instruction| !instruction.mnemonic.is_empty())
            .count();

        println!(
            "path {path_index}: {} modeled instructions{}",
            scored_instruction_count,
            args.prefix_len
                .map(|prefix_len| format!(" (prefix {prefix_len})"))
                .unwrap_or_default()
        );
        for (instruction_index, instruction) in block
            .instructions
            .iter()
            .filter(|instruction| !instruction.mnemonic.is_empty())
            .filter(|instruction| !instruction.mnemonic.eq_ignore_ascii_case("ret"))
            .enumerate()
        {
            let key =
                uops_key_for_instruction(instruction).unwrap_or_else(|| "unsupported".to_owned());
            println!(
                "  {instruction_index:02}: {} {:?} -> {key}",
                instruction.mnemonic, instruction.operands
            );
        }

        let rough = scorer.rough_score_block(&block);
        println!(
            "path {path_index}: rough score {}, instructions {}",
            rough.score(),
            rough.instruction_count()
        );
        println!("path {path_index}: starting full uiCA simulation");
        let _ = std::io::stdout().flush();
        let started = Instant::now();
        let full = scorer.full_score_block(&block)?;
        println!(
            "path {path_index}: full uiCA score {}, instructions {}, elapsed {:.3}s",
            full.score(),
            full.instruction_count(),
            started.elapsed().as_secs_f64()
        );
    }

    Ok(())
}

fn scoring_block(block: &InstructionBlock, prefix_len: Option<usize>) -> InstructionBlock {
    let Some(prefix_len) = prefix_len else {
        return block.clone();
    };
    let instructions = block
        .instructions
        .iter()
        .filter(|instruction| !instruction.mnemonic.is_empty())
        .filter(|instruction| !instruction.mnemonic.eq_ignore_ascii_case("ret"))
        .take(prefix_len)
        .cloned()
        .collect::<Vec<ModeledInstruction>>();
    InstructionBlock {
        label: block.label.clone(),
        instructions,
    }
}

fn split_target_cpus(target_cpu: &str) -> Vec<String> {
    target_cpu
        .split(',')
        .map(str::trim)
        .filter(|cpu| !cpu.is_empty())
        .map(str::to_owned)
        .collect()
}

fn print_estimate_report(report: &EstimateReport) {
    println!("uiCA estimate");
    println!("input: {}", report.input_path.display());
    println!("target cpu: {}", report.target_cpu);
    println!(
        "paths: {} imported, {} estimated",
        report.imported_path_count,
        report.results.len()
    );
    println!("output dir: {}", report.output_dir.display());

    if report.results.is_empty() {
        println!("No paths to estimate.");
        return;
    }

    println!();
    println!(
        "{:<5} {:<6} {:>8} {:>8} {:>8} {:>8} {:<24} Status",
        "Rank", "Path", "TP", "Cycles", "Iters", "Instr", "Trace"
    );
    for result in &report.results {
        let rank = result
            .rank
            .map(|rank| rank.to_string())
            .unwrap_or_else(|| "-".to_owned());
        let throughput = result
            .throughput
            .filter(|value| value.is_finite())
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "err".to_owned());
        let trace = result
            .trace_path
            .as_deref()
            .map(terminal_file_link)
            .unwrap_or_else(|| "-".to_owned());
        let status = if result.error.is_some() { "err" } else { "ok" };
        println!(
            "{:<5} {:<6} {:>8} {:>8} {:>8} {:>8} {:<24} {}",
            rank,
            result.path_index,
            throughput,
            result.cycles_simulated,
            result.iterations_simulated,
            result.instruction_count,
            trace,
            status
        );
    }

    let warnings = report
        .results
        .iter()
        .filter_map(|result| {
            result
                .error
                .as_ref()
                .map(|error| (result.path_index, error))
        })
        .collect::<Vec<_>>();
    if !warnings.is_empty() {
        println!();
        println!("Warnings:");
        for (path_index, error) in warnings {
            println!("  path {path_index}: {error}");
        }
    }
}

fn terminal_file_link(path: &Path) -> String {
    let Some(label) = path.file_name().and_then(|name| name.to_str()) else {
        return path.display().to_string();
    };
    let Ok(absolute) = path.canonicalize() else {
        return label.to_owned();
    };
    format!(
        "\x1b]8;;file://{}\x1b\\{}\x1b]8;;\x1b\\",
        absolute.display(),
        label
    )
}

fn run_with_runtime_ui(config: &RunConfig) -> RunSummary {
    match config.runtime_ui {
        RuntimeUiArg::None => exit_on_error(build_run_summary(config)),
        RuntimeUiArg::Textual => run_with_textual_runtime(config),
        RuntimeUiArg::Auto => {
            if std::io::stdout().is_terminal() {
                run_with_threaded_tui(config)
            } else {
                run_with_line_runtime(config)
            }
        }
    }
}

fn run_with_textual_runtime(config: &RunConfig) -> RunSummary {
    run_with_line_runtime(config)
}

fn run_with_line_runtime(config: &RunConfig) -> RunSummary {
    let mut session = LineRuntimeSession::stdout();
    exit_on_error(build_run_summary_with_session(config, &mut session))
}

fn run_with_threaded_tui(config: &RunConfig) -> RunSummary {
    let (sender, receiver) = mpsc::channel();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let engine_cancel_flag = Arc::clone(&cancel_flag);
    let tui_error_cancel_flag = Arc::clone(&cancel_flag);
    let engine_config = config.clone();

    let engine = thread::spawn(move || {
        let mut session = ChannelRuntimeSession::new(sender, engine_cancel_flag);
        build_run_summary_with_session(&engine_config, &mut session)
    });

    let tui_result = run_tui_event_loop(receiver, cancel_flag)
        .map_err(|error| format!("failed to run TUI: {error}"));
    if tui_result.is_err() {
        tui_error_cancel_flag.store(true, Ordering::SeqCst);
    }

    let engine_result = match engine.join() {
        Ok(result) => result,
        Err(_) => Err("engine thread panicked".to_owned()),
    };

    if let Err(error) = tui_result {
        eprintln!("warning: {error}");
    }

    exit_on_error(engine_result)
}

fn exit_on_error<T>(result: Result<T, String>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2);
        }
    }
}

fn print_dry_run_summary(config: &RunConfig, summary: &DryRunSummary) {
    println!("vxsort-codegen dry run");
    println!("arch: {}", config.arch.cli_name());
    println!("dtype: {}", config.dtype.cli_name());
    println!("num vecs: {}", config.num_vecs);
    println!("elements per vector: {}", summary.elements_per_vector);
    println!("total elements: {}", summary.total_elements);
    println!("stages: {}", summary.stages.len());
    println!(
        "initial state: top={:?} bottom={:?}",
        summary.initial_top, summary.initial_bottom
    );
    println!(
        "candidates: {} shallow, {} deep",
        summary.shallow_candidate_count, summary.deep_candidate_count
    );
}

fn print_run_summary(config: &RunConfig, summary: &RunSummary) {
    println!("vxsort-codegen sync run");
    println!("arch: {}", config.arch.cli_name());
    println!("dtype: {}", config.dtype.cli_name());
    println!("waves: {}", summary.wave_count);
    println!("search exhausted: {}", summary.search_exhausted);
    println!("scored paths: {}", summary.scored_paths);
    println!(
        "best score: {}",
        summary
            .best_score
            .map(|score| score.to_string())
            .unwrap_or_else(|| "none".to_owned())
    );
    for wave in &summary.waves {
        println!(
            "wave {}: target stage {}, attempts {}, valid outputs {}, new outputs {}, scored paths {}, propagated {:?}",
            wave.wave,
            wave.target_stage,
            wave.attempts,
            wave.valid_outputs,
            wave.new_outputs,
            wave.scored_paths,
            wave.propagation_stages
        );
    }
}
