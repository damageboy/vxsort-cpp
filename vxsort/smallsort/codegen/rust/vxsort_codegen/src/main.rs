use std::{
    io::IsTerminal,
    io::Write,
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
    CliArgs, CliCommand, DryRunSummary, FetchUicaDataConfig, RunConfig, RunSummary, RuntimeUiArg,
    build_dry_run_summary, build_run_summary, build_run_summary_with_session,
    convert_solution_json_to_asm, fetch_uica_data,
    instruction_stream::{
        InstructionBlock, LoweringOptions, ModeledInstruction, lower_solution_paths,
    },
    parse_run_config, read_solution_json,
    runtime::{ChannelRuntimeSession, LineRuntimeSession},
    runtime_tui::run_tui_event_loop,
    uica_scoring::{UiPackScorer, uops_key_for_instruction},
};

fn main() {
    let args = CliArgs::parse();
    match &args.command {
        CliCommand::Solve(_) => {}
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
    let stream = lower_solution_paths(
        &imported.metadata,
        &imported.transition_table,
        &imported.paths,
        LoweringOptions::default(),
    );
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
