use std::path::{Path, PathBuf};

use asm_exporter::{write_solution_asm_for_assigned_paths, write_solution_asm_for_paths};
use clap::{Args, Parser, Subcommand, ValueEnum};
use instruction_stream::{LoweringOptions, lower_assigned_paths};
use json_exporter::{SolutionJsonMetadata, write_solution_json_for_assigned_paths};
pub use json_importer::{ImportedSolutionJson, read_solution_json, solution_json_from_value};
use runtime::{NullRuntimeSession, RuntimeEvent, RuntimeSession};
use runtime_trace::RuntimeTrace;
use scoring::PathCost;
use serde_json::json;
pub use uica_data_fetch::{
    FetchUicaDataConfig, FetchUicaDataReport, default_uica_data_base_url, fetch_uica_data,
};
pub use uica_estimator::{EstimateOptions, EstimateReport, EstimateResult, estimate_solution_json};
use uica_scoring::UiPackScorer;
use wave_engine::{ScoredPath, WaveConfig, WaveEngine};

pub mod asm_exporter;
pub mod bitonic_sorter;
pub mod instruction_stream;
pub mod json_exporter;
pub mod json_importer;
pub mod runtime;
pub mod runtime_trace;
pub mod runtime_tui;
pub mod scoring;
pub mod transition_table;
pub mod uica_data_fetch;
pub mod uica_estimator;
pub mod uica_scoring;
pub mod verifier;
pub mod wave_engine;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ArchArg {
    Avx2,
    Avx512,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RuntimeUiArg {
    Auto,
    Textual,
    None,
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
    Verify(VerifyJsonArgs),
    Estimate(EstimateJsonArgs),
    FetchUicaData(FetchUicaDataArgs),
    JsonToAsm(JsonToAsmArgs),
    ScoreJson(ScoreJsonArgs),
}

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

pub fn build_run_summary(config: &RunConfig) -> Result<RunSummary, String> {
    let mut session = NullRuntimeSession;
    build_run_summary_with_session(config, &mut session)
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

pub fn build_run_summary_with_session(
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
    trace.event(
        "run_started",
        json!({
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
            "max_waves": config.max_waves,
            "wave_attempts": config.wave_attempts,
            "wave_outputs": config.wave_outputs,
            "worker_count": config.worker_count,
        }),
    );

    let mut total_wave_count = 0;
    let mut search_exhausted = true;
    let mut total_scored_paths = 0;
    let mut best_score = None;
    let mut waves = Vec::new();
    let multi_target = config.target_cpus.len() > 1;

    for target_cpu in &config.target_cpus {
        trace.event(
            "target_started",
            json!({
                "target_cpu": target_cpu,
                "rough_top_k": rough_top_k,
                "final_top_k": final_top_k,
            }),
        );
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
                max_unique_outputs: config.max_gadget_solutions,
            },
            Box::new(rough_scorer),
        )
        .map_err(|error| error.to_string())?;

        let result = {
            let mut target_session =
                TargetRuntimeSession::new(session, target_cpu, rough_top_k, final_top_k);
            engine
                .run_sync_with_session_and_trace(
                    config.max_waves,
                    config.wave_attempts,
                    config.wave_outputs,
                    &mut target_session,
                    &mut trace,
                )
                .map_err(|error| error.to_string())?
        };
        trace.event(
            "target_rough_finished",
            json!({
                "target_cpu": target_cpu,
                "wave_count": result.wave_count(),
                "search_exhausted": result.search_exhausted(),
                "rough_candidate_count": engine.scored_paths().len(),
                "best_rough_score": engine.best_score(),
            }),
        );

        let full_scorer = UiPackScorer::from_data_dir(&config.uica_data_dir, target_cpu)?;
        let final_paths = full_score_and_prune_paths(
            &metadata,
            engine.transition_table(),
            engine.scored_paths(),
            &full_scorer,
            FinalScoringOptions {
                final_top_k,
                target_cpu,
            },
            session,
        );
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
    trace.event(
        "run_finished",
        json!({
            "wave_count": total_wave_count,
            "search_exhausted": search_exhausted,
            "scored_paths": total_scored_paths,
            "best_score": best_score,
        }),
    );
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
}

fn full_score_and_prune_paths(
    metadata: &SolutionJsonMetadata,
    table: &transition_table::TransitionTable,
    rough_paths: &[ScoredPath],
    scorer: &UiPackScorer,
    options: FinalScoringOptions<'_>,
    session: &mut impl RuntimeSession,
) -> Vec<ScoredPath> {
    session.on_event(RuntimeEvent::FullScoringStarted {
        target_cpu: options.target_cpu.to_owned(),
        candidate_count: rough_paths.len(),
        final_top_k: options.final_top_k,
    });

    let mut best_full_score: Option<f64> = None;
    let mut scored = rough_paths
        .iter()
        .enumerate()
        .map(|(index, rough_path)| {
            session.on_event(RuntimeEvent::FullScoringCandidateStarted {
                target_cpu: options.target_cpu.to_owned(),
                candidate_index: index + 1,
                candidate_count: rough_paths.len(),
                rough_score: rough_path.cost().score(),
            });
            let full_cost = full_score_assigned_path(metadata, table, rough_path, scorer);
            best_full_score = Some(
                best_full_score
                    .map(|score| score.min(full_cost.score()))
                    .unwrap_or_else(|| full_cost.score()),
            );
            session.on_event(RuntimeEvent::FullScoringProgress {
                target_cpu: options.target_cpu.to_owned(),
                scored_count: index + 1,
                candidate_count: rough_paths.len(),
                best_score: best_full_score,
            });
            (
                rough_path.with_cost(full_cost),
                rough_path.cost().clone(),
                rough_path.assigned_path().selection_key(),
            )
        })
        .collect::<Vec<_>>();

    scored.sort_by(|left, right| {
        left.0
            .cost()
            .score()
            .total_cmp(&right.0.cost().score())
            .then_with(|| left.1.score().total_cmp(&right.1.score()))
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.0.discovery_order().cmp(&right.0.discovery_order()))
    });
    scored.truncate(options.final_top_k);
    let kept_count = scored.len();
    session.on_event(RuntimeEvent::FullScoringFinished {
        target_cpu: options.target_cpu.to_owned(),
        scored_count: rough_paths.len(),
        kept_count,
        best_score: best_full_score,
    });
    scored
        .into_iter()
        .map(|(scored_path, _, _)| scored_path)
        .collect()
}

fn full_score_assigned_path(
    metadata: &SolutionJsonMetadata,
    table: &transition_table::TransitionTable,
    rough_path: &ScoredPath,
    scorer: &UiPackScorer,
) -> PathCost {
    let assigned_path = rough_path.assigned_path().clone();
    let stream = lower_assigned_paths(
        metadata,
        table,
        std::slice::from_ref(&assigned_path),
        LoweringOptions::default(),
    );
    let Some(block) = stream.blocks.first() else {
        return PathCost::new(0, f64::INFINITY, f64::INFINITY);
    };

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
