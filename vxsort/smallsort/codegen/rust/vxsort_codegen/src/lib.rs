use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
};

use asm_exporter::{write_solution_asm_for_assigned_paths, write_solution_asm_for_paths};
use clap::{Args, Parser, Subcommand, ValueEnum};
use instruction_stream::{InstructionBlock, LoweringOptions, lower_assigned_paths};
use json_exporter::{SolutionJsonMetadata, write_solution_json_for_assigned_paths};
pub use json_importer::{ImportedSolutionJson, read_solution_json, solution_json_from_value};
use runtime::{FullScoringProgressSnapshot, RuntimeEvent, RuntimeSession};
use runtime_trace::RuntimeTrace;
use scoring::PathCost;
use serde::{Deserialize, Serialize};
pub use uica_data_fetch::{
    FetchUicaDataConfig, FetchUicaDataReport, default_uica_data_base_url, fetch_uica_data,
};
pub use uica_estimator::{EstimateOptions, EstimateReport, EstimateResult, estimate_solution_json};
use uica_scoring::UiPackScorer;
use wave_engine::{ScoredPath, WaveConfig, WaveEngine, WaveProgressObserver};

pub mod asm_exporter;
pub mod bitonic_sorter;
pub mod instruction_stream;
pub mod json_exporter;
pub mod json_importer;
pub mod runtime;
pub mod runtime_trace;
pub mod runtime_tui;
pub mod scoring;
pub mod stable_vec;
pub mod transition_table;
pub mod uica_data_fetch;
pub mod uica_estimator;
pub mod uica_scoring;
pub mod verifier;
pub mod wave_engine;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
pub enum ArchArg {
    Avx2,
    Avx512,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
pub enum DTypeArg {
    I16,
    U16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
pub enum WorkerBackendArg {
    Auto,
    InProcess,
    Process,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RuntimeUiArg {
    Auto,
    Textual,
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
pub enum PruneScoreArg {
    Rough,
    Uica,
}

impl ArchArg {
    pub fn cli_name(self) -> &'static str {
        match self {
            ArchArg::Avx2 => "AVX2",
            ArchArg::Avx512 => "AVX512",
        }
    }
}

impl DTypeArg {
    pub fn cli_name(self) -> &'static str {
        match self {
            DTypeArg::I16 => "i16",
            DTypeArg::U16 => "u16",
            DTypeArg::I32 => "i32",
            DTypeArg::U32 => "u32",
            DTypeArg::F32 => "f32",
            DTypeArg::I64 => "i64",
            DTypeArg::U64 => "u64",
            DTypeArg::F64 => "f64",
        }
    }

    pub fn element_bits(self) -> usize {
        match self {
            DTypeArg::I16 | DTypeArg::U16 => 16,
            DTypeArg::I32 | DTypeArg::U32 | DTypeArg::F32 => 32,
            DTypeArg::I64 | DTypeArg::U64 | DTypeArg::F64 => 64,
        }
    }

    pub fn is_synthesis_supported(self) -> bool {
        matches!(self, DTypeArg::I32 | DTypeArg::I64)
    }
}

impl WorkerBackendArg {
    pub fn cli_name(self) -> &'static str {
        match self {
            WorkerBackendArg::Auto => "auto",
            WorkerBackendArg::InProcess => "in-process",
            WorkerBackendArg::Process => "process",
        }
    }
}

impl PruneScoreArg {
    pub fn cli_name(self) -> &'static str {
        match self {
            PruneScoreArg::Rough => "rough",
            PruneScoreArg::Uica => "uica",
        }
    }
}

const HELP_STYLES: clap::builder::Styles = clap::builder::Styles::styled()
    .header(
        clap::builder::styling::AnsiColor::BrightGreen
            .on_default()
            .bold(),
    )
    .usage(
        clap::builder::styling::AnsiColor::BrightGreen
            .on_default()
            .bold(),
    )
    .literal(
        clap::builder::styling::AnsiColor::BrightCyan
            .on_default()
            .bold(),
    )
    .placeholder(clap::builder::styling::AnsiColor::BrightYellow.on_default())
    .valid(clap::builder::styling::AnsiColor::BrightGreen.on_default())
    .invalid(clap::builder::styling::AnsiColor::BrightRed.on_default());

#[derive(Debug, Parser)]
#[command(name = "vxsort-codegen")]
#[command(about = "Rust shell for vxsort bitonic code generation")]
#[command(color = clap::ColorChoice::Always)]
#[command(styles = HELP_STYLES)]
#[command(subcommand_required = true, arg_required_else_help = true)]
pub struct CliArgs {
    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Debug, Args)]
pub struct SolveArgs {
    #[arg(long = "num-vecs", default_value_t = 2)]
    pub num_vecs: usize,

    #[arg(long = "vector-machine", value_enum, ignore_case = true)]
    pub arch: Option<ArchArg>,

    #[arg(long = "datatype", value_enum, ignore_case = true)]
    pub dtype: Option<DTypeArg>,

    #[arg(long = "depth-limit")]
    pub depth_limit: Option<usize>,

    #[arg(long = "top-k")]
    pub top_k: Option<usize>,

    #[arg(long = "prune-score", value_enum, ignore_case = true, default_value_t = PruneScoreArg::Rough)]
    pub prune_score: PruneScoreArg,

    #[arg(long = "gadget-depth", default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=3))]
    pub gadget_depth: u8,

    #[arg(long = "natural-order")]
    pub natural_order: bool,

    #[arg(long = "retroactive-input")]
    pub retroactive_input: bool,

    #[arg(long = "max-gadget-solutions", default_value_t = 3)]
    pub max_gadget_solutions: usize,

    #[arg(long = "target-cpu", default_value = "generic")]
    pub target_cpu: String,

    #[arg(long = "estimate")]
    pub estimate: bool,

    #[arg(long = "output-path")]
    pub output_path: Option<PathBuf>,

    #[arg(long = "uica-data-dir", default_value = "uica-data")]
    pub uica_data_dir: PathBuf,

    #[arg(long = "max-waves")]
    pub max_waves: Option<usize>,

    #[arg(long = "wave-attempts", default_value_t = 10_000)]
    pub wave_attempts: usize,

    #[arg(long = "wave-outputs", default_value_t = 100)]
    pub wave_outputs: usize,

    #[arg(long = "workers", default_value_t = 0)]
    pub workers: usize,

    #[arg(long = "worker-backend", value_enum, ignore_case = true, default_value_t = WorkerBackendArg::Auto)]
    pub worker_backend: WorkerBackendArg,

    #[arg(long = "runtime-ui", value_enum, ignore_case = true, default_value_t = RuntimeUiArg::Auto)]
    pub runtime_ui: RuntimeUiArg,

    #[arg(long = "runtime-trace")]
    pub runtime_trace_path: Option<PathBuf>,

    #[arg(long = "dry-run")]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct VerifyJsonArgs {
    #[arg(long = "input")]
    pub input: PathBuf,

    #[arg(long = "top-k")]
    pub top_k: Option<usize>,

    #[arg(long = "workers", default_value_t = 0)]
    pub workers: usize,
}

#[derive(Debug, Args)]
pub struct EstimateJsonArgs {
    #[arg(long = "input")]
    pub input: PathBuf,

    #[arg(long = "target-cpu")]
    pub target_cpu: String,

    #[arg(long = "uica-data-dir", default_value = "uica-data")]
    pub uica_data_dir: PathBuf,

    #[arg(long = "top-k")]
    pub top_k: Option<usize>,

    #[arg(long = "output-dir")]
    pub output_dir: Option<PathBuf>,

    #[arg(long = "no-traces")]
    pub no_traces: bool,
}

#[derive(Debug, Subcommand)]
pub enum CliCommand {
    Solve(SolveArgs),
    #[command(name = "__synthesis-worker", hide = true)]
    SynthesisWorker(SynthesisWorkerArgs),
    Verify(VerifyJsonArgs),
    Estimate(EstimateJsonArgs),
    FetchUicaData(FetchUicaDataArgs),
    JsonToAsm(JsonToAsmArgs),
    ScoreJson(ScoreJsonArgs),
}

#[derive(Debug, Args)]
pub struct SynthesisWorkerArgs {}

#[derive(Debug, Args)]
pub struct FetchUicaDataArgs {
    #[arg(long = "target-cpu", required = true)]
    pub target_cpu: String,

    #[arg(long = "data-dir", default_value = "uica-data")]
    pub data_dir: PathBuf,

    #[arg(long = "base-url", default_value = "https://uica.houmus.org/data")]
    pub base_url: String,
}

#[derive(Debug, Args)]
pub struct JsonToAsmArgs {
    #[arg(long = "input")]
    pub input: PathBuf,

    #[arg(long = "output")]
    pub output: PathBuf,
}

#[derive(Debug, Args)]
pub struct ScoreJsonArgs {
    #[arg(long = "input")]
    pub input: PathBuf,

    #[arg(long = "target-cpu")]
    pub target_cpu: String,

    #[arg(long = "uica-data-dir", default_value = "uica-data")]
    pub uica_data_dir: PathBuf,

    #[arg(long = "path-index")]
    pub path_index: Option<usize>,

    #[arg(long = "prefix-len")]
    pub prefix_len: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunConfig {
    pub num_vecs: usize,
    pub arch: ArchArg,
    pub dtype: DTypeArg,
    pub depth_limit: Option<usize>,
    pub top_k: Option<usize>,
    pub prune_score: PruneScoreArg,
    pub gadget_depth: u8,
    pub natural_order: bool,
    pub retroactive_input: bool,
    pub max_gadget_solutions: usize,
    pub target_cpus: Vec<String>,
    pub estimate: bool,
    pub output_path: Option<PathBuf>,
    pub uica_data_dir: PathBuf,
    pub max_waves: Option<usize>,
    pub wave_attempts: usize,
    pub wave_outputs: usize,
    pub worker_count: usize,
    pub worker_backend: WorkerBackendArg,
    pub runtime_ui: RuntimeUiArg,
    pub runtime_trace_path: Option<PathBuf>,
    pub dry_run: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageSummary {
    pub stage: usize,
    pub pairs: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DryRunSummary {
    pub elements_per_vector: usize,
    pub total_elements: usize,
    pub initial_top: Vec<usize>,
    pub initial_bottom: Vec<usize>,
    pub stages: Vec<StageSummary>,
    pub shallow_candidate_count: usize,
    pub deep_candidate_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WaveSummary {
    pub wave: usize,
    pub target_stage: usize,
    pub attempts: usize,
    pub valid_outputs: usize,
    pub new_outputs: usize,
    pub discovered_paths: usize,
    pub scored_paths: usize,
    pub propagation_stages: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunSummary {
    pub wave_count: usize,
    pub search_exhausted: bool,
    pub scored_paths: usize,
    pub best_score: Option<f64>,
    pub waves: Vec<WaveSummary>,
}

pub fn parse_run_config(args: CliArgs) -> Result<RunConfig, String> {
    let solve_args = match args.command {
        CliCommand::Solve(solve_args) => solve_args,
        _ => return Err("utility subcommands are not solver run configurations".to_owned()),
    };
    parse_solve_run_config(solve_args)
}

pub fn parse_solve_run_config(solve_args: SolveArgs) -> Result<RunConfig, String> {
    let arch = solve_args
        .arch
        .ok_or_else(|| "--vector-machine is required".to_owned())?;
    let dtype = solve_args
        .dtype
        .ok_or_else(|| "--datatype is required".to_owned())?;
    if !dtype.is_synthesis_supported() {
        return Err(format!(
            "Rust solve currently supports only i32 and i64; {} is supported for JSON/ASM export only",
            dtype.cli_name()
        ));
    }

    let target_cpus: Vec<String> = solve_args
        .target_cpu
        .split(',')
        .map(str::trim)
        .filter(|cpu| !cpu.is_empty() && *cpu != "generic")
        .map(str::to_owned)
        .collect();

    if !solve_args.dry_run {
        if target_cpus.is_empty() {
            return Err("--target-cpu is required for scored solving".to_owned());
        }
        if solve_args.top_k.is_none() {
            return Err("--top-k is required for scored solving".to_owned());
        }
    }

    Ok(RunConfig {
        num_vecs: solve_args.num_vecs,
        arch,
        dtype,
        depth_limit: solve_args.depth_limit,
        top_k: solve_args.top_k,
        prune_score: solve_args.prune_score,
        gadget_depth: solve_args.gadget_depth,
        natural_order: solve_args.natural_order,
        retroactive_input: solve_args.retroactive_input,
        max_gadget_solutions: solve_args.max_gadget_solutions,
        target_cpus,
        estimate: solve_args.estimate,
        output_path: solve_args.output_path,
        uica_data_dir: solve_args.uica_data_dir,
        max_waves: solve_args.max_waves,
        wave_attempts: solve_args.wave_attempts,
        wave_outputs: solve_args.wave_outputs,
        worker_count: resolve_worker_count(solve_args.workers),
        worker_backend: solve_args.worker_backend,
        runtime_ui: solve_args.runtime_ui,
        runtime_trace_path: solve_args.runtime_trace_path,
        dry_run: solve_args.dry_run,
    })
}

pub fn resolve_worker_count(requested_workers: usize) -> usize {
    if requested_workers > 0 {
        return requested_workers;
    }

    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .max(1)
}

pub fn convert_solution_json_to_asm(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<(), String> {
    let imported = read_solution_json(input_path)?;
    if imported.assigned_paths.is_empty() {
        write_solution_asm_for_paths(
            output_path,
            &imported.metadata,
            &imported.transition_table,
            &imported.paths,
        )
        .map_err(|error| format!("failed to write solution assembly: {error}"))
    } else {
        write_solution_asm_for_assigned_paths(
            output_path,
            &imported.metadata,
            &imported.transition_table,
            &imported.assigned_paths,
        )
        .map_err(|error| format!("failed to write solution assembly: {error}"))
    }
}

pub fn verify_solution_json(
    input_path: impl AsRef<Path>,
    top_k: Option<usize>,
    workers: usize,
) -> Result<verifier::VerificationReport, String> {
    let report =
        verifier::verify_solution_json(input_path, verifier::VerifyJsonOptions { top_k, workers })?;
    if report.failure_count() > 0 {
        return Err(verifier::format_verification_failures(&report));
    }
    Ok(report)
}

/// Runs the solver while emitting progress events to `session`.
///
/// Line logging, the threaded TUI, tests, and other runtime observers all use
/// this entry point so the core solve remains independent of presentation.
pub fn run_solver_with_session(
    config: &RunConfig,
    session: &mut impl RuntimeSession,
) -> Result<RunSummary, String> {
    if !config.dtype.is_synthesis_supported() {
        return Err(format!(
            "Rust solve currently supports only i32 and i64; {} is supported for JSON/ASM export only",
            config.dtype.cli_name()
        ));
    }
    let metadata = SolutionJsonMetadata {
        natural_order: config.natural_order,
        arch: config.arch,
        dtype: config.dtype,
        num_vecs: config.num_vecs,
    };
    let final_top_k = config
        .top_k
        .ok_or_else(|| "--top-k is required for scored solving".to_owned())?;
    let rough_top_k = rough_candidate_limit(final_top_k);
    if config.target_cpus.is_empty() {
        return Err("--target-cpu is required for scored solving".to_owned());
    }
    let mut trace = RuntimeTrace::from_path(config.runtime_trace_path.as_deref())?;
    crate::trace!(trace, "run_started", {
            "num_vecs": config.num_vecs,
            "arch": config.arch.cli_name(),
            "dtype": config.dtype.cli_name(),
            "gadget_depth": config.gadget_depth,
            "natural_order": config.natural_order,
            "retroactive_input": config.retroactive_input,
            "max_gadget_solutions": config.max_gadget_solutions,
            "target_cpus": config.target_cpus.clone(),
            "final_top_k": final_top_k,
            "rough_top_k": rough_top_k,
            "prune_score": config.prune_score.cli_name(),
            "max_waves": config.max_waves,
            "wave_attempts": config.wave_attempts,
            "wave_outputs": config.wave_outputs,
            "worker_count": config.worker_count,
            "worker_backend": config.worker_backend.cli_name(),
    });

    let mut total_wave_count = 0;
    let mut search_exhausted = true;
    let mut total_scored_paths = 0;
    let mut best_score = None;
    let mut waves = Vec::new();
    let multi_target = config.target_cpus.len() > 1;

    for target_cpu in &config.target_cpus {
        crate::trace!(trace, "target_started", {
                "target_cpu": target_cpu,
                "rough_top_k": rough_top_k,
                "final_top_k": final_top_k,
                "prune_score": config.prune_score.cli_name(),
        });
        let rough_scorer = UiPackScorer::from_data_dir(&config.uica_data_dir, target_cpu)?
            .with_solution_metadata(metadata.clone());
        let mut engine = WaveEngine::with_scorer(
            WaveConfig {
                num_vecs: config.num_vecs,
                arch: config.arch,
                dtype: config.dtype,
                gadget_depth: config.gadget_depth,
                natural_order: config.natural_order,
                retroactive_input: config.retroactive_input,
                top_k: Some(rough_top_k),
                worker_count: config.worker_count,
                worker_backend: config.worker_backend,
                max_unique_outputs: config.max_gadget_solutions,
            },
            rough_scorer,
        )
        .map_err(|error| error.to_string())?;

        let mut live_full_scorer = LiveFullScorer::new(
            &metadata,
            FinalScoringOptions {
                final_top_k,
                target_cpu,
                uica_data_dir: &config.uica_data_dir,
                worker_count: config.worker_count,
            },
            match config.prune_score {
                PruneScoreArg::Rough => LiveFullScoringInput::RetainedRough,
                PruneScoreArg::Uica => LiveFullScoringInput::RecentRough,
            },
        )?;
        let result = {
            let mut target_session =
                TargetRuntimeSession::new(session, target_cpu, rough_top_k, final_top_k);
            engine
                .run_sync_with_session_trace_and_observer(
                    config.max_waves,
                    config.wave_attempts,
                    config.wave_outputs,
                    &mut target_session,
                    &mut trace,
                    &mut live_full_scorer,
                )
                .map_err(|error| error.to_string())?
        };
        crate::trace!(trace, "target_rough_finished", {
                "target_cpu": target_cpu,
                "wave_count": result.wave_count(),
                "search_exhausted": result.search_exhausted(),
                "rough_candidate_count": engine.scored_paths().len(),
                "best_rough_score": engine.best_score(),
        });

        let final_paths = match config.prune_score {
            PruneScoreArg::Rough => {
                drop(live_full_scorer);
                full_score_and_prune_paths(
                    &metadata,
                    engine.transition_table(),
                    engine.scored_paths(),
                    FinalScoringOptions {
                        final_top_k,
                        target_cpu,
                        uica_data_dir: &config.uica_data_dir,
                        worker_count: config.worker_count,
                    },
                    session,
                )?
            }
            PruneScoreArg::Uica => {
                live_full_scorer.finish(session, &mut trace, "target_finished")?
            }
        };
        let assigned_paths = final_paths
            .iter()
            .map(|scored_path| scored_path.assigned_path().clone())
            .collect::<Vec<_>>();

        if let Some(output_path) = &config.output_path {
            let output_path = output_path_for_target(output_path, target_cpu, multi_target);
            session.on_event(RuntimeEvent::ExportStarted {
                target_cpu: target_cpu.clone(),
                kind: "JSON".to_owned(),
                path: output_path.display().to_string(),
            });
            write_solution_json_for_assigned_paths(
                &output_path,
                &metadata,
                engine.transition_table(),
                &assigned_paths,
            )
            .map_err(|error| format!("failed to write solution JSON: {error}"))?;
            session.on_event(RuntimeEvent::ExportFinished {
                target_cpu: target_cpu.clone(),
                kind: "JSON".to_owned(),
                path: output_path.display().to_string(),
            });
        }

        total_wave_count += result.wave_count();
        search_exhausted &= result.search_exhausted();
        total_scored_paths += final_paths.len();
        if let Some(target_best_score) = final_paths
            .iter()
            .map(|scored_path| scored_path.cost().score())
            .min_by(f64::total_cmp)
        {
            best_score = Some(
                best_score
                    .map(|score: f64| score.min(target_best_score))
                    .unwrap_or(target_best_score),
            );
        }
        waves.extend(result.waves().iter().map(|wave| {
            WaveSummary {
                wave: wave.wave(),
                target_stage: wave.target_stage(),
                attempts: wave.target().attempts(),
                valid_outputs: wave.target().valid_outputs(),
                new_outputs: wave.target().new_outputs(),
                discovered_paths: wave.discovered_paths(),
                scored_paths: wave.scored_paths(),
                propagation_stages: wave
                    .propagation()
                    .iter()
                    .map(|stage| stage.stage())
                    .collect(),
            }
        }));
    }

    session.on_event(RuntimeEvent::RunFinished {
        wave_count: total_wave_count,
        search_exhausted,
        scored_paths: total_scored_paths,
        best_score,
    });
    crate::trace!(trace, "run_finished", {
            "wave_count": total_wave_count,
            "search_exhausted": search_exhausted,
            "scored_paths": total_scored_paths,
            "best_score": best_score,
    });
    trace.flush()?;

    Ok(RunSummary {
        wave_count: total_wave_count,
        search_exhausted,
        scored_paths: total_scored_paths,
        best_score,
        waves,
    })
}

fn rough_candidate_limit(final_top_k: usize) -> usize {
    final_top_k.saturating_mul(10).max(final_top_k).max(1)
}

struct TargetRuntimeSession<'a, S> {
    inner: &'a mut S,
    target_cpu: &'a str,
    rough_candidate_limit: usize,
    final_top_k: usize,
}

impl<'a, S> TargetRuntimeSession<'a, S> {
    fn new(
        inner: &'a mut S,
        target_cpu: &'a str,
        rough_candidate_limit: usize,
        final_top_k: usize,
    ) -> Self {
        Self {
            inner,
            target_cpu,
            rough_candidate_limit,
            final_top_k,
        }
    }
}

impl<S: RuntimeSession> RuntimeSession for TargetRuntimeSession<'_, S> {
    fn on_event(&mut self, event: RuntimeEvent) {
        match event {
            RuntimeEvent::RunStarted { stage_count } => {
                self.inner
                    .on_event(RuntimeEvent::RunStarted { stage_count });
                self.inner.on_event(RuntimeEvent::TargetStarted {
                    target_cpu: self.target_cpu.to_owned(),
                    rough_candidate_limit: self.rough_candidate_limit,
                    final_top_k: self.final_top_k,
                });
            }
            RuntimeEvent::RunFinished {
                wave_count,
                search_exhausted,
                scored_paths,
                best_score,
            } => self.inner.on_event(RuntimeEvent::RoughSearchFinished {
                target_cpu: self.target_cpu.to_owned(),
                wave_count,
                search_exhausted,
                rough_candidate_count: scored_paths,
                best_rough_score: best_score,
            }),
            other => self.inner.on_event(other),
        }
    }

    fn should_stop(&self) -> bool {
        self.inner.should_stop()
    }
}

#[derive(Clone, Copy)]
struct FinalScoringOptions<'a> {
    final_top_k: usize,
    target_cpu: &'a str,
    uica_data_dir: &'a Path,
    worker_count: usize,
}

#[derive(Clone, Debug)]
struct FinalScoringJob {
    rough_path: ScoredPath,
    block: InstructionBlock,
}

#[derive(Clone, Debug)]
struct FinalScoringResult {
    scored_path: ScoredPath,
    rough_cost: PathCost,
    selection_key: scoring::AssignedPathKey,
}

fn compare_final_scoring_results(
    left: &FinalScoringResult,
    right: &FinalScoringResult,
) -> std::cmp::Ordering {
    left.scored_path
        .cost()
        .score()
        .total_cmp(&right.scored_path.cost().score())
        .then_with(|| left.rough_cost.score().total_cmp(&right.rough_cost.score()))
        .then_with(|| left.selection_key.cmp(&right.selection_key))
        .then_with(|| {
            left.scored_path
                .discovery_order()
                .cmp(&right.scored_path.discovery_order())
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LiveFullScoringInput {
    RetainedRough,
    RecentRough,
}

struct LiveFullScorer {
    target_cpu: String,
    metadata: SolutionJsonMetadata,
    input: LiveFullScoringInput,
    final_top_k: usize,
    sender: mpsc::Sender<Option<FinalScoringJob>>,
    receiver: mpsc::Receiver<Result<FinalScoringResult, String>>,
    workers: Vec<thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    submitted: std::collections::HashSet<scoring::AssignedPathKey>,
    completed: std::collections::HashMap<scoring::AssignedPathKey, FinalScoringResult>,
    retained: Vec<FinalScoringResult>,
    submitted_count: usize,
    completed_count: usize,
    pending_count: usize,
    best_retained: Option<FinalScoringResult>,
}

impl LiveFullScorer {
    fn new(
        metadata: &SolutionJsonMetadata,
        options: FinalScoringOptions<'_>,
        input: LiveFullScoringInput,
    ) -> Result<Self, String> {
        let worker_count = options
            .worker_count
            .max(1)
            .min(options.final_top_k.max(1) * 10);
        let (sender, job_receiver) = mpsc::channel::<Option<FinalScoringJob>>();
        let job_receiver = Arc::new(Mutex::new(job_receiver));
        let (result_sender, receiver) = mpsc::channel::<Result<FinalScoringResult, String>>();
        let metadata = metadata.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let scorer = Arc::new(UiPackScorer::from_data_dir(
            options.uica_data_dir,
            options.target_cpu,
        )?);
        let mut workers = Vec::with_capacity(worker_count);

        for _ in 0..worker_count {
            let job_receiver = Arc::clone(&job_receiver);
            let result_sender = result_sender.clone();
            let stop = Arc::clone(&stop);
            let scorer = Arc::clone(&scorer);
            workers.push(thread::spawn(move || {
                loop {
                    let message = {
                        let receiver = job_receiver
                            .lock()
                            .expect("live full scoring job receiver should not be poisoned");
                        receiver.recv()
                    };
                    match message {
                        Ok(Some(job)) => {
                            if stop.load(Ordering::Relaxed) {
                                return;
                            }
                            let result = score_final_job(job, scorer.as_ref());
                            if result_sender.send(Ok(result)).is_err() {
                                return;
                            }
                        }
                        Ok(None) | Err(_) => return,
                    }
                }
            }));
        }
        drop(result_sender);

        Ok(Self {
            target_cpu: options.target_cpu.to_owned(),
            metadata,
            input,
            final_top_k: options.final_top_k,
            sender,
            receiver,
            workers,
            stop,
            submitted: std::collections::HashSet::new(),
            completed: std::collections::HashMap::new(),
            retained: Vec::new(),
            submitted_count: 0,
            completed_count: 0,
            pending_count: 0,
            best_retained: None,
        })
    }

    fn enqueue_retained_paths(
        &mut self,
        table: &transition_table::TransitionTable,
        rough_paths: &[ScoredPath],
        trace: &mut RuntimeTrace,
        reason: &str,
    ) -> Result<(), String> {
        for rough_path in rough_paths {
            let key = rough_path.assigned_path().selection_key();
            if !self.submitted.insert(key.clone()) {
                continue;
            }
            let block = final_scoring_block(&self.metadata, table, rough_path);
            self.sender
                .send(Some(FinalScoringJob {
                    rough_path: rough_path.clone(),
                    block,
                }))
                .map_err(|error| format!("failed to queue live full scoring job: {error}"))?;
            self.submitted_count += 1;
            self.pending_count += 1;
            crate::trace!(trace, "live_full_scoring_job_queued", {
                        "reason": reason,
                        "target_cpu": self.target_cpu,
                        "submitted": self.submitted_count,
                        "pending": self.pending_count,
                        "rough_score": rough_path.cost().score(),
            });
        }
        Ok(())
    }

    fn handle_completed_result(
        &mut self,
        result: FinalScoringResult,
        trace: &mut RuntimeTrace,
        reason: &str,
    ) {
        self.pending_count = self.pending_count.saturating_sub(1);
        self.completed_count += 1;
        match self.input {
            LiveFullScoringInput::RetainedRough => {
                self.completed.insert(result.selection_key.clone(), result);
            }
            LiveFullScoringInput::RecentRough => {
                self.retain_final_result(result);
            }
        }
        crate::trace!(trace, "live_full_scoring_job_completed", {
                    "reason": reason,
                    "target_cpu": self.target_cpu,
                    "completed": self.completed_count,
                    "pending": self.pending_count,
        });
    }

    fn retain_final_result(&mut self, result: FinalScoringResult) {
        self.retained.push(result);
        self.retained.sort_by(compare_final_scoring_results);
        self.retained.truncate(self.final_top_k);
        self.best_retained = self.retained.first().cloned();
    }

    fn poll(
        &mut self,
        retained_paths: &[ScoredPath],
        session: &mut dyn RuntimeSession,
        trace: &mut RuntimeTrace,
        reason: &str,
    ) -> Result<(), String> {
        while let Ok(result) = self.receiver.try_recv() {
            let result = result?;
            self.handle_completed_result(result, trace, reason);
        }

        if self.input == LiveFullScoringInput::RetainedRough {
            self.refresh_best_retained(retained_paths);
        }
        let snapshot = self.snapshot();
        session.on_event(RuntimeEvent::LiveFullScoringProgress(snapshot.clone()));
        crate::trace!(trace, "live_full_scoring_snapshot", {
                    "reason": reason,
                    "target_cpu": self.target_cpu,
                    "submitted": self.submitted_count,
                    "completed": snapshot.completed_paths(),
                    "pending": snapshot.pending_paths(),
                    "scored": snapshot.scored_paths(),
                    "best_score": snapshot.best_score(),
                    "best_estimated_cycles": snapshot.best_estimated_cycles(),
        });
        Ok(())
    }

    fn finish(
        &mut self,
        session: &mut dyn RuntimeSession,
        trace: &mut RuntimeTrace,
        reason: &str,
    ) -> Result<Vec<ScoredPath>, String> {
        while self.pending_count > 0 {
            let result = self
                .receiver
                .recv()
                .map_err(|error| format!("live full scoring worker stopped early: {error}"))??;
            self.handle_completed_result(result, trace, reason);
            let snapshot = self.snapshot();
            session.on_event(RuntimeEvent::LiveFullScoringProgress(snapshot.clone()));
        }

        self.retained.sort_by(compare_final_scoring_results);
        session.on_event(RuntimeEvent::FullScoringFinished {
            target_cpu: self.target_cpu.clone(),
            scored_count: self.completed_count,
            kept_count: self.retained.len(),
            best_score: self
                .retained
                .first()
                .map(|result| result.scored_path.cost().score()),
        });
        Ok(self
            .retained
            .iter()
            .map(|result| result.scored_path.clone())
            .collect())
    }

    fn refresh_best_retained(&mut self, retained_paths: &[ScoredPath]) {
        let retained_keys = retained_paths
            .iter()
            .map(|path| path.assigned_path().selection_key())
            .collect::<std::collections::HashSet<_>>();
        self.best_retained = self
            .completed
            .values()
            .filter(|result| retained_keys.contains(&result.selection_key))
            .min_by(|left, right| {
                left.scored_path
                    .cost()
                    .score()
                    .total_cmp(&right.scored_path.cost().score())
                    .then_with(|| left.rough_cost.score().total_cmp(&right.rough_cost.score()))
                    .then_with(|| left.selection_key.cmp(&right.selection_key))
                    .then_with(|| {
                        left.scored_path
                            .discovery_order()
                            .cmp(&right.scored_path.discovery_order())
                    })
            })
            .cloned();
    }

    fn snapshot(&self) -> FullScoringProgressSnapshot {
        let best_score = self
            .best_retained
            .as_ref()
            .map(|result| result.scored_path.cost().score());
        let best_estimated_cycles = self
            .best_retained
            .as_ref()
            .map(|result| result.scored_path.cost().estimated_cycles());
        FullScoringProgressSnapshot::new(
            self.completed_count,
            self.submitted_count,
            self.pending_count,
            match self.input {
                LiveFullScoringInput::RetainedRough => self.completed.len(),
                LiveFullScoringInput::RecentRough => self.retained.len(),
            },
            best_score,
            best_estimated_cycles,
        )
    }
}

impl Drop for LiveFullScorer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for _ in &self.workers {
            let _ = self.sender.send(None);
        }
        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }
    }
}

impl WaveProgressObserver for LiveFullScorer {
    fn on_wave_progress(
        &mut self,
        engine: &WaveEngine,
        recent_scored_paths: &[ScoredPath],
        session: &mut dyn RuntimeSession,
        trace: &mut RuntimeTrace,
        reason: &str,
    ) -> Result<(), String> {
        let candidate_paths = match self.input {
            LiveFullScoringInput::RetainedRough => engine.scored_paths(),
            LiveFullScoringInput::RecentRough => recent_scored_paths,
        };
        self.enqueue_retained_paths(engine.transition_table(), candidate_paths, trace, reason)?;
        self.poll(engine.scored_paths(), session, trace, reason)
    }
}

fn full_score_and_prune_paths(
    metadata: &SolutionJsonMetadata,
    table: &transition_table::TransitionTable,
    rough_paths: &[ScoredPath],
    options: FinalScoringOptions<'_>,
    session: &mut impl RuntimeSession,
) -> Result<Vec<ScoredPath>, String> {
    let worker_count = final_scoring_worker_count(options.worker_count, rough_paths.len());
    session.on_event(RuntimeEvent::FullScoringStarted {
        target_cpu: options.target_cpu.to_owned(),
        candidate_count: rough_paths.len(),
        final_top_k: options.final_top_k,
    });

    if rough_paths.is_empty() {
        session.on_event(RuntimeEvent::FullScoringFinished {
            target_cpu: options.target_cpu.to_owned(),
            scored_count: 0,
            kept_count: 0,
            best_score: None,
        });
        return Ok(Vec::new());
    }

    for (index, rough_path) in rough_paths.iter().enumerate() {
        session.on_event(RuntimeEvent::FullScoringCandidateStarted {
            target_cpu: options.target_cpu.to_owned(),
            candidate_index: index + 1,
            candidate_count: rough_paths.len(),
            rough_score: rough_path.cost().score(),
        });
    }

    let mut best_full_score: Option<f64> = None;
    let mut completed_count = 0;
    let mut scored = score_final_paths_parallel(
        metadata,
        table,
        rough_paths,
        options,
        worker_count,
        |result| {
            best_full_score = Some(
                best_full_score
                    .map(|score| score.min(result.scored_path.cost().score()))
                    .unwrap_or_else(|| result.scored_path.cost().score()),
            );
            completed_count += 1;
            session.on_event(RuntimeEvent::FullScoringProgress {
                target_cpu: options.target_cpu.to_owned(),
                scored_count: completed_count,
                candidate_count: rough_paths.len(),
                best_score: best_full_score,
            });
        },
    )?;

    scored.sort_by(compare_final_scoring_results);
    scored.truncate(options.final_top_k);
    let kept_count = scored.len();
    session.on_event(RuntimeEvent::FullScoringFinished {
        target_cpu: options.target_cpu.to_owned(),
        scored_count: rough_paths.len(),
        kept_count,
        best_score: best_full_score,
    });
    Ok(scored
        .into_iter()
        .map(|result| result.scored_path)
        .collect())
}

fn final_scoring_worker_count(requested_workers: usize, candidate_count: usize) -> usize {
    if candidate_count == 0 {
        return requested_workers.max(1);
    }
    requested_workers.max(1).min(candidate_count)
}

fn score_final_paths_parallel(
    metadata: &SolutionJsonMetadata,
    table: &transition_table::TransitionTable,
    rough_paths: &[ScoredPath],
    options: FinalScoringOptions<'_>,
    worker_count: usize,
    mut on_result: impl FnMut(&FinalScoringResult),
) -> Result<Vec<FinalScoringResult>, String> {
    let scorer = Arc::new(UiPackScorer::from_data_dir(
        options.uica_data_dir,
        options.target_cpu,
    )?);

    if worker_count == 1 {
        let mut results = Vec::with_capacity(rough_paths.len());
        for rough_path in rough_paths {
            let block = final_scoring_block(metadata, table, rough_path);
            let result = score_final_job(
                FinalScoringJob {
                    rough_path: rough_path.clone(),
                    block,
                },
                scorer.as_ref(),
            );
            on_result(&result);
            results.push(result);
        }
        return Ok(results);
    }

    let (job_sender, job_receiver) = mpsc::channel::<Option<FinalScoringJob>>();
    let job_receiver = Arc::new(Mutex::new(job_receiver));
    let (result_sender, result_receiver) = mpsc::channel::<Result<FinalScoringResult, String>>();
    let mut workers = Vec::with_capacity(worker_count);

    for _ in 0..worker_count {
        let job_receiver = Arc::clone(&job_receiver);
        let result_sender = result_sender.clone();
        let scorer = Arc::clone(&scorer);
        workers.push(thread::spawn(move || {
            loop {
                let message = {
                    let receiver = job_receiver
                        .lock()
                        .expect("final scoring job receiver should not be poisoned");
                    receiver.recv()
                };
                match message {
                    Ok(Some(job)) => {
                        let result = score_final_job(job, scorer.as_ref());
                        if result_sender.send(Ok(result)).is_err() {
                            return;
                        }
                    }
                    Ok(None) | Err(_) => return,
                }
            }
        }));
    }
    drop(result_sender);

    for rough_path in rough_paths {
        let block = final_scoring_block(metadata, table, rough_path);
        job_sender
            .send(Some(FinalScoringJob {
                rough_path: rough_path.clone(),
                block,
            }))
            .map_err(|error| format!("failed to queue final scoring job: {error}"))?;
    }
    for _ in 0..worker_count {
        job_sender
            .send(None)
            .map_err(|error| format!("failed to stop final scoring worker: {error}"))?;
    }
    drop(job_sender);

    let mut results = Vec::with_capacity(rough_paths.len());
    while results.len() < rough_paths.len() {
        match result_receiver
            .recv()
            .map_err(|error| format!("final scoring worker stopped early: {error}"))?
        {
            Ok(result) => {
                on_result(&result);
                results.push(result);
            }
            Err(error) => {
                for worker in workers {
                    let _ = worker.join();
                }
                return Err(error);
            }
        }
    }

    for worker in workers {
        worker
            .join()
            .map_err(|_| "final scoring worker panicked".to_owned())?;
    }
    Ok(results)
}

fn final_scoring_block(
    metadata: &SolutionJsonMetadata,
    table: &transition_table::TransitionTable,
    rough_path: &ScoredPath,
) -> InstructionBlock {
    let assigned_path = rough_path.assigned_path().clone();
    let mut stream = lower_assigned_paths(
        metadata,
        table,
        std::slice::from_ref(&assigned_path),
        LoweringOptions::default(),
    );
    stream.blocks.pop().unwrap_or(InstructionBlock {
        label: None,
        instructions: Vec::new(),
    })
}

fn score_final_job(job: FinalScoringJob, scorer: &UiPackScorer) -> FinalScoringResult {
    let full_cost = full_score_block(&job.block, scorer);
    FinalScoringResult {
        scored_path: job.rough_path.with_cost(full_cost),
        rough_cost: job.rough_path.cost().clone(),
        selection_key: job.rough_path.assigned_path().selection_key(),
    }
}

fn full_score_block(block: &InstructionBlock, scorer: &UiPackScorer) -> PathCost {
    scorer
        .full_score_block(block)
        .unwrap_or_else(|_| unsupported_block_cost(block))
}

fn unsupported_block_cost(block: &instruction_stream::InstructionBlock) -> PathCost {
    let instruction_count = block
        .instructions
        .iter()
        .filter(|instruction| !instruction.mnemonic.eq_ignore_ascii_case("ret"))
        .count() as u32;
    PathCost::new(instruction_count, f64::INFINITY, f64::INFINITY)
}

fn output_path_for_target(path: &Path, target_cpu: &str, multi_target: bool) -> PathBuf {
    if !multi_target {
        return path.to_path_buf();
    }

    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("solutions");
    let file_name = match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if !extension.is_empty() => format!("{stem}.{target_cpu}.{extension}"),
        _ => format!("{stem}.{target_cpu}"),
    };
    path.with_file_name(file_name)
}

pub fn build_dry_run_summary(config: &RunConfig) -> Result<DryRunSummary, String> {
    let engine = WaveEngine::new(WaveConfig {
        num_vecs: config.num_vecs,
        arch: config.arch,
        dtype: config.dtype,
        gadget_depth: config.gadget_depth,
        natural_order: config.natural_order,
        retroactive_input: config.retroactive_input,
        top_k: config.top_k,
        worker_count: config.worker_count,
        worker_backend: config.worker_backend,
        max_unique_outputs: config.max_gadget_solutions,
    })
    .map_err(|error| error.to_string())?;

    let stages = engine
        .stages()
        .iter()
        .map(|stage| StageSummary {
            stage: stage.index(),
            pairs: stage.pairs().to_vec(),
        })
        .collect::<Vec<_>>();

    Ok(DryRunSummary {
        elements_per_vector: engine.elements_per_vector(),
        total_elements: engine.total_elements(),
        initial_top: engine
            .initial_state()
            .top()
            .iter()
            .map(|value| *value as usize + 1)
            .collect(),
        initial_bottom: engine
            .initial_state()
            .bottom()
            .iter()
            .map(|value| *value as usize + 1)
            .collect(),
        stages,
        shallow_candidate_count: engine.shallow_candidates().len(),
        deep_candidate_count: engine.deep_candidates().len(),
    })
}

#[cfg(test)]
mod tests {
    use uica_data::{
        DATAPACK_SCHEMA_VERSION, DataPack, UIPACK_VERSION, encode_uipack, read_uipack_header,
    };

    use super::*;
    use crate::runtime::RuntimeSession;
    use crate::scoring::AssignedPath;
    use crate::transition_table::{CompletePath, GadgetIndex, PathId, TransitionIndex};

    #[derive(Default)]
    struct RecordingRuntimeSession {
        events: Vec<RuntimeEvent>,
    }

    impl RuntimeSession for RecordingRuntimeSession {
        fn on_event(&mut self, event: RuntimeEvent) {
            self.events.push(event);
        }
    }

    fn empty_uica_data_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vxsort-final-scoring-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("arch")).expect("fixture arch dir should be created");
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
        let bytes = encode_uipack(&pack, "SKL").expect("fixture uipack should encode");
        let header = read_uipack_header(&bytes).expect("fixture uipack header should decode");
        std::fs::write(dir.join("arch/SKL.uipack"), &bytes).expect("fixture pack should write");
        std::fs::write(
            dir.join("manifest.json"),
            format!(
                r#"{{
  "schema_version": "uica-datapack-manifest-v2",
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
                UIPACK_VERSION, header.file_len, header.checksum, header.records_count
            ),
        )
        .expect("fixture manifest should write");
        dir
    }

    fn empty_rough_path(discovery_order: usize, rough_score: f64) -> ScoredPath {
        let path = CompletePath::new(Vec::new());
        let assigned_path = AssignedPath::new(path.clone(), Vec::new());
        ScoredPath::new(
            PathId(discovery_order as u32),
            path,
            assigned_path,
            PathCost::new(0, rough_score, rough_score),
            discovery_order,
        )
    }

    fn keyed_rough_path(discovery_order: usize, rough_score: f64) -> ScoredPath {
        let path = CompletePath::new(vec![TransitionIndex(discovery_order as u32)]);
        let assigned_path = AssignedPath::new(path.clone(), vec![GadgetIndex(0)]);
        ScoredPath::new(
            PathId(discovery_order as u32),
            path,
            assigned_path,
            PathCost::new(0, rough_score, rough_score),
            discovery_order,
        )
    }

    #[test]
    fn final_scoring_parallelizes_locally_and_keeps_final_events_separate_from_rough_scoring() {
        let data_dir = empty_uica_data_dir("worker-capacity");
        let metadata = SolutionJsonMetadata {
            natural_order: false,
            arch: ArchArg::Avx2,
            dtype: DTypeArg::I64,
            num_vecs: 2,
        };
        let table = transition_table::TransitionTable::new(0);
        let rough_paths = vec![
            empty_rough_path(0, 3.0),
            empty_rough_path(1, 2.0),
            empty_rough_path(2, 1.0),
        ];
        let mut session = RecordingRuntimeSession::default();

        let final_paths = full_score_and_prune_paths(
            &metadata,
            &table,
            &rough_paths,
            FinalScoringOptions {
                final_top_k: 2,
                target_cpu: "SKL",
                uica_data_dir: &data_dir,
                worker_count: 2,
            },
            &mut session,
        )
        .expect("final scoring should complete");

        assert_eq!(final_paths.len(), 2);
        assert_eq!(final_scoring_worker_count(8, 3), 3);
        assert_eq!(final_scoring_worker_count(0, 3), 1);
        assert!(matches!(
            session.events.first(),
            Some(RuntimeEvent::FullScoringStarted {
                candidate_count: 3,
                final_top_k: 2,
                ..
            })
        ));
        assert!(session.events.iter().any(|event| matches!(
            event,
            RuntimeEvent::FullScoringProgress {
                scored_count: 3,
                candidate_count: 3,
                best_score: Some(0.0),
                ..
            }
        )));
        assert!(
            session
                .events
                .iter()
                .all(|event| !matches!(event, RuntimeEvent::ScoringProgress(_)))
        );

        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[test]
    fn live_full_scoring_deduplicates_retained_rough_paths_and_reports_progress() {
        let data_dir = empty_uica_data_dir("live-full");
        let metadata = SolutionJsonMetadata {
            natural_order: false,
            arch: ArchArg::Avx2,
            dtype: DTypeArg::I64,
            num_vecs: 2,
        };
        let table = transition_table::TransitionTable::new(0);
        let rough_path = empty_rough_path(0, 1.0);
        let mut scorer = LiveFullScorer::new(
            &metadata,
            FinalScoringOptions {
                final_top_k: 1,
                target_cpu: "SKL",
                uica_data_dir: &data_dir,
                worker_count: 1,
            },
            LiveFullScoringInput::RetainedRough,
        )
        .expect("live full scorer should initialize");
        let mut trace = RuntimeTrace::disabled();
        let mut session = RecordingRuntimeSession::default();

        scorer
            .enqueue_retained_paths(
                &table,
                std::slice::from_ref(&rough_path),
                &mut trace,
                "test",
            )
            .expect("first retained rough path should enqueue");
        scorer
            .enqueue_retained_paths(
                &table,
                std::slice::from_ref(&rough_path),
                &mut trace,
                "test",
            )
            .expect("duplicate retained rough path should be ignored");

        assert_eq!(scorer.submitted_count, 1);
        assert_eq!(scorer.pending_count, 1);

        for _ in 0..50 {
            scorer
                .poll(
                    std::slice::from_ref(&rough_path),
                    &mut session,
                    &mut trace,
                    "test",
                )
                .expect("live full scorer poll should succeed");
            if scorer.completed_count == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert_eq!(scorer.submitted_count, 1);
        assert_eq!(scorer.completed_count, 1);
        assert_eq!(scorer.pending_count, 0);
        assert!(session.events.iter().any(|event| matches!(
            event,
            RuntimeEvent::LiveFullScoringProgress(snapshot)
                if snapshot.completed_paths() == 1
                    && snapshot.total_paths() == 1
                    && snapshot.pending_paths() == 0
                    && snapshot.best_score() == Some(0.0)
        )));

        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[test]
    fn live_full_scoring_recent_mode_retains_global_final_top_k() {
        let data_dir = empty_uica_data_dir("live-full-global");
        let metadata = SolutionJsonMetadata {
            natural_order: false,
            arch: ArchArg::Avx2,
            dtype: DTypeArg::I64,
            num_vecs: 2,
        };
        let mut scorer = LiveFullScorer::new(
            &metadata,
            FinalScoringOptions {
                final_top_k: 2,
                target_cpu: "SKL",
                uica_data_dir: &data_dir,
                worker_count: 1,
            },
            LiveFullScoringInput::RecentRough,
        )
        .expect("live full scorer should initialize");

        for (discovery_order, rough_score, full_score) in
            [(0, 1.0, 30.0), (1, 100.0, 20.0), (2, 2.0, 40.0)]
        {
            let rough_path = keyed_rough_path(discovery_order, rough_score);
            scorer.retain_final_result(FinalScoringResult {
                scored_path: rough_path.with_cost(PathCost::new(0, full_score, full_score)),
                rough_cost: rough_path.cost().clone(),
                selection_key: rough_path.assigned_path().selection_key(),
            });
        }

        let retained_scores = scorer
            .retained
            .iter()
            .map(|result| result.scored_path.cost().score())
            .collect::<Vec<_>>();

        assert_eq!(retained_scores, vec![20.0, 30.0]);
        assert_eq!(
            scorer
                .best_retained
                .as_ref()
                .map(|result| result.rough_cost.score()),
            Some(100.0)
        );

        let _ = std::fs::remove_dir_all(data_dir);
    }
}
