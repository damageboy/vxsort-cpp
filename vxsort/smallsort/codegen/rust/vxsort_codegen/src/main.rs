use std::{
    io::IsTerminal,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
};

use clap::Parser;
use vxsort_codegen::{
    CliArgs, DryRunSummary, RunConfig, RunSummary, RuntimeUiArg, build_dry_run_summary,
    build_run_summary, build_run_summary_with_session, parse_run_config,
    runtime::{ChannelRuntimeSession, LineRuntimeSession},
    runtime_tui::run_tui_event_loop,
};

fn main() {
    let args = CliArgs::parse();
    let config = exit_on_error(parse_run_config(args));

    if config.dry_run {
        let summary = exit_on_error(build_dry_run_summary(&config));
        print_dry_run_summary(&config, &summary);
        return;
    }

    let summary = run_with_runtime_ui(&config);
    print_run_summary(&config, &summary);
}

fn run_with_runtime_ui(config: &RunConfig) -> RunSummary {
    match config.runtime_ui {
        RuntimeUiArg::None => exit_on_error(build_run_summary(config)),
        RuntimeUiArg::Textual => run_with_textual_runtime(config),
        RuntimeUiArg::Auto => {
            if std::io::stdout().is_terminal() {
                run_with_textual_runtime(config)
            } else {
                exit_on_error(build_run_summary(config))
            }
        }
    }
}

fn run_with_textual_runtime(config: &RunConfig) -> RunSummary {
    if std::io::stdout().is_terminal() {
        return run_with_threaded_tui(config);
    }

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
