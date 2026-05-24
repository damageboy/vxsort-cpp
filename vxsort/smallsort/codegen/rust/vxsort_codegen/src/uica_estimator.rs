use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::instruction_stream::{
    InstructionBlock, LoweringOptions, lower_assigned_paths, lower_solution_paths,
};
use crate::uica_scoring::{UiBlockSimulation, UiPackScorer};
use crate::{ImportedSolutionJson, read_solution_json};

#[derive(Clone, Debug)]
pub struct EstimateOptions {
    pub target_cpu: String,
    pub uica_data_dir: PathBuf,
    pub top_k: Option<usize>,
    pub output_dir: Option<PathBuf>,
    pub write_traces: bool,
}

#[derive(Clone, Debug)]
pub struct EstimateReport {
    pub input_path: PathBuf,
    pub target_cpu: String,
    pub output_dir: PathBuf,
    pub imported_path_count: usize,
    pub results: Vec<EstimateResult>,
}

#[derive(Clone, Debug)]
pub struct EstimateResult {
    pub path_index: usize,
    pub rank: Option<usize>,
    pub throughput: Option<f64>,
    pub cycles_simulated: u32,
    pub iterations_simulated: u32,
    pub instruction_count: usize,
    pub trace_path: Option<PathBuf>,
    pub error: Option<String>,
}

pub fn estimate_solution_json(
    input_path: impl AsRef<Path>,
    options: EstimateOptions,
) -> Result<EstimateReport, String> {
    let imported = read_solution_json(input_path.as_ref())?;
    let imported_path_count = imported_path_count(&imported);
    let selected_blocks = select_blocks_from_imported(&imported, options.top_k);
    let output_dir = resolve_output_dir(options.output_dir.as_ref())?;
    let mut results = Vec::new();

    if !selected_blocks.is_empty() {
        let scorer = UiPackScorer::from_data_dir(&options.uica_data_dir, &options.target_cpu)?;
        for (path_index, block) in selected_blocks {
            results.push(estimate_block(
                &scorer,
                &block,
                path_index,
                &output_dir,
                options.write_traces,
            ));
        }
    }

    rank_results(&mut results);

    Ok(EstimateReport {
        input_path: input_path.as_ref().to_path_buf(),
        target_cpu: options.target_cpu,
        output_dir,
        imported_path_count,
        results,
    })
}

fn resolve_output_dir(output_dir: Option<&PathBuf>) -> Result<PathBuf, String> {
    let dir = output_dir.cloned().unwrap_or_else(default_output_dir);
    fs::create_dir_all(&dir)
        .map_err(|error| format!("failed to create output dir {}: {error}", dir.display()))?;
    Ok(dir)
}

fn default_output_dir() -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "vxsort_uica_estimate_{}_{}",
        std::process::id(),
        millis
    ))
}

fn estimate_block(
    scorer: &UiPackScorer,
    block: &InstructionBlock,
    path_index: usize,
    output_dir: &Path,
    write_traces: bool,
) -> EstimateResult {
    let fallback_instruction_count = scored_instruction_count(block);
    match scorer.simulate_block(block, write_traces) {
        Ok(simulation) => successful_result(path_index, output_dir, write_traces, simulation),
        Err(error) => EstimateResult {
            path_index,
            rank: None,
            throughput: None,
            cycles_simulated: 0,
            iterations_simulated: 0,
            instruction_count: fallback_instruction_count,
            trace_path: None,
            error: Some(error),
        },
    }
}

fn successful_result(
    path_index: usize,
    output_dir: &Path,
    write_traces: bool,
    simulation: UiBlockSimulation,
) -> EstimateResult {
    let mut trace_path = None;
    let mut error = None;
    if write_traces {
        match write_trace_html(path_index, output_dir, &simulation) {
            Ok(path) => trace_path = Some(path),
            Err(trace_error) => error = Some(trace_error),
        }
    }

    EstimateResult {
        path_index,
        rank: None,
        throughput: Some(simulation.throughput_cycles_per_iteration),
        cycles_simulated: simulation.cycles_simulated,
        iterations_simulated: simulation.iterations_simulated,
        instruction_count: simulation.instruction_count,
        trace_path,
        error,
    }
}

fn write_trace_html(
    path_index: usize,
    output_dir: &Path,
    simulation: &UiBlockSimulation,
) -> Result<PathBuf, String> {
    let reports = simulation
        .reports
        .as_ref()
        .ok_or_else(|| "uiCA simulation did not produce trace reports".to_owned())?;
    let html = uica_core::report::render_trace_html(&reports.trace)?;
    let trace_path = output_dir.join(format!("solution_{path_index:03}_trace.html"));
    fs::write(&trace_path, html)
        .map_err(|error| format!("failed to write trace {}: {error}", trace_path.display()))?;
    Ok(trace_path)
}

fn rank_results(results: &mut Vec<EstimateResult>) {
    results.sort_by(
        |left, right| match (finite_throughput(left), finite_throughput(right)) {
            (Some(left_throughput), Some(right_throughput)) => left_throughput
                .total_cmp(&right_throughput)
                .then_with(|| left.instruction_count.cmp(&right.instruction_count))
                .then_with(|| left.path_index.cmp(&right.path_index)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => left.path_index.cmp(&right.path_index),
        },
    );

    let mut rank = 1;
    for result in results {
        if finite_throughput(result).is_some() {
            result.rank = Some(rank);
            rank += 1;
        }
    }
}

fn finite_throughput(result: &EstimateResult) -> Option<f64> {
    if result.error.is_some() {
        return None;
    }
    result
        .throughput
        .filter(|throughput| throughput.is_finite())
}

fn imported_path_count(imported: &ImportedSolutionJson) -> usize {
    if imported.assigned_paths.is_empty() {
        imported.paths.len()
    } else {
        imported.assigned_paths.len()
    }
}

fn select_blocks_from_imported(
    imported: &ImportedSolutionJson,
    top_k: Option<usize>,
) -> Vec<(usize, InstructionBlock)> {
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
    let selected_count = top_k
        .unwrap_or(stream.blocks.len())
        .min(stream.blocks.len());
    stream
        .blocks
        .into_iter()
        .take(selected_count)
        .enumerate()
        .collect()
}

fn scored_instruction_count(block: &InstructionBlock) -> usize {
    block
        .instructions
        .iter()
        .filter(|instruction| !instruction.mnemonic.is_empty())
        .filter(|instruction| !instruction.mnemonic.eq_ignore_ascii_case("ret"))
        .count()
}
