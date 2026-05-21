use clap::{Parser, ValueEnum};
use runtime::{NullRuntimeSession, RuntimeSession};

pub mod bitonic_sorter;
pub mod runtime;
pub mod runtime_tui;
pub mod scoring;
pub mod transition_table;
pub mod wave_engine;

use wave_engine::{WaveConfig, WaveEngine};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ArchArg {
    Avx2,
    Avx512,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum DTypeArg {
    I32,
    I64,
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
            DTypeArg::I32 => "i32",
            DTypeArg::I64 => "i64",
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "vxsort-codegen")]
#[command(about = "Rust shell for vxsort bitonic code generation")]
pub struct CliArgs {
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

    #[arg(long = "dry-run")]
    pub dry_run: bool,
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
    pub max_waves: Option<usize>,
    pub wave_attempts: usize,
    pub wave_outputs: usize,
    pub worker_count: usize,
    pub runtime_ui: RuntimeUiArg,
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
    let arch = args
        .arch
        .ok_or_else(|| "--vector-machine is required".to_owned())?;
    let dtype = args
        .dtype
        .ok_or_else(|| "--datatype is required".to_owned())?;

    let target_cpus = args
        .target_cpu
        .split(',')
        .map(str::trim)
        .filter(|cpu| !cpu.is_empty() && *cpu != "generic")
        .map(str::to_owned)
        .collect();

    Ok(RunConfig {
        num_vecs: args.num_vecs,
        arch,
        dtype,
        depth_limit: args.depth_limit,
        top_k: args.top_k,
        gadget_depth: args.gadget_depth,
        natural_order: args.natural_order,
        retroactive_input: args.retroactive_input,
        max_gadget_solutions: args.max_gadget_solutions,
        target_cpus,
        estimate: args.estimate,
        max_waves: args.max_waves,
        wave_attempts: args.wave_attempts,
        wave_outputs: args.wave_outputs,
        worker_count: resolve_worker_count(args.workers),
        runtime_ui: args.runtime_ui,
        dry_run: args.dry_run,
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

pub fn build_run_summary_with_session(
    config: &RunConfig,
    session: &mut impl RuntimeSession,
) -> Result<RunSummary, String> {
    let mut engine = WaveEngine::new(WaveConfig {
        num_vecs: config.num_vecs,
        arch: config.arch,
        dtype: config.dtype,
        gadget_depth: config.gadget_depth,
        natural_order: config.natural_order,
        retroactive_input: config.retroactive_input,
        top_k: config.top_k,
        worker_count: config.worker_count,
    })
    .map_err(|error| error.to_string())?;

    let result = engine
        .run_sync_with_session(
            config.max_waves,
            config.wave_attempts,
            config.wave_outputs,
            session,
        )
        .map_err(|error| error.to_string())?;

    Ok(RunSummary {
        wave_count: result.wave_count(),
        search_exhausted: result.search_exhausted(),
        scored_paths: engine.scored_path_count(),
        best_score: engine.best_score(),
        waves: result
            .waves()
            .iter()
            .map(|wave| WaveSummary {
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
            })
            .collect(),
    })
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
            .map(|value| *value as usize)
            .collect(),
        initial_bottom: engine
            .initial_state()
            .bottom()
            .iter()
            .map(|value| *value as usize)
            .collect(),
        stages,
        shallow_candidate_count: engine.shallow_candidates().len(),
        deep_candidate_count: engine.deep_candidates().len(),
    })
}
