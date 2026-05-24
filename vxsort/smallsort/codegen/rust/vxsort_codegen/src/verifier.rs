use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget, dispatch_intrinsic};
use z3::ast::{Ast, BV, Bool};
use z3::{SatResult, Solver};
use z3_avx::lanes::{concat_msb_first, extract_element};
use z3_avx::minmax::{generic_max, generic_min};

use crate::json_exporter::SolutionJsonMetadata;
use crate::json_importer::{ImportedSolutionJson, read_solution_json};
use crate::scoring::AssignedPath;
use crate::transition_table::{CompletePath, StateTuple, TransitionTable};
use crate::{ArchArg, DTypeArg};

const MIN_GROUP_SIZE: usize = 4;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifyJsonOptions {
    pub top_k: Option<usize>,
    pub workers: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerificationReport {
    pub path_count: usize,
    pub chunks_per_path: usize,
    pub failures: Vec<VerificationFailure>,
    pub solver_time: Duration,
}

impl VerificationReport {
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerificationFailure {
    pub path_index: usize,
    pub counterexample: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerificationResult {
    pub verified: bool,
    pub counterexample: Option<Vec<u64>>,
    pub solver_time: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VerifyStep {
    gadget: PermutationGadget,
    input_state: StateTuple,
    output_state: StateTuple,
}

#[derive(Clone, Debug)]
struct VerifyJob {
    path_index: usize,
    steps: Vec<VerifyStep>,
}

#[derive(Clone, Debug)]
struct PathVerifier {
    metadata: SolutionJsonMetadata,
    elements_per_vector: usize,
    lane_width: u32,
    total_bits: u32,
}

struct ArgResolveContext<'a> {
    top_reg: &'a BV,
    bottom_reg: &'a BV,
    current_reg: &'a BV,
    previous_output: Option<&'a BV>,
    results: &'a [BV],
}

pub fn verify_solution_json(
    input_path: impl AsRef<Path>,
    options: VerifyJsonOptions,
) -> Result<VerificationReport, String> {
    let imported = read_solution_json(input_path)?;
    verify_imported_solution_json(&imported, &options)
}

pub fn verify_imported_solution_json(
    imported: &ImportedSolutionJson,
    options: &VerifyJsonOptions,
) -> Result<VerificationReport, String> {
    validate_metadata(&imported.metadata)?;
    let total_elements = total_elements(&imported.metadata)?;
    let chunks_per_path = num_verify_chunks(total_elements)?;
    let jobs = build_verify_jobs(imported, options.top_k)?;
    let path_count = jobs.len();
    let worker_count = resolve_worker_count(options.workers, path_count);

    let results = if worker_count <= 1 {
        run_verify_jobs_sequential(&imported.metadata, jobs)?
    } else {
        run_verify_jobs_parallel(&imported.metadata, jobs, worker_count)?
    };

    let mut failures = Vec::new();
    let mut solver_time = Duration::ZERO;
    for (path_index, result) in results {
        solver_time += result.solver_time;
        if !result.verified {
            failures.push(VerificationFailure {
                path_index,
                counterexample: result.counterexample.unwrap_or_default(),
            });
        }
    }
    failures.sort_by_key(|failure| failure.path_index);

    Ok(VerificationReport {
        path_count,
        chunks_per_path,
        failures,
        solver_time,
    })
}

pub fn compute_recursion_boundaries(n: usize) -> Result<Vec<usize>, String> {
    if n < 2 || !n.is_power_of_two() {
        return Err(format!("n must be a power of 2, got {n}"));
    }

    let mut boundaries = Vec::new();
    let mut group_size = MIN_GROUP_SIZE;
    while group_size < n {
        boundaries.push(total_stages(group_size) - 1);
        group_size *= 2;
    }
    Ok(boundaries)
}

pub fn num_verify_chunks(total_elements: usize) -> Result<usize, String> {
    let boundaries = compute_recursion_boundaries(total_elements)?;
    if boundaries.is_empty() {
        return Ok(1);
    }

    let mut total = 0;
    for level_idx in 0..boundaries.len() {
        let post_group_size = MIN_GROUP_SIZE * (1 << level_idx);
        total += total_elements / post_group_size;
    }
    Ok(total + 1)
}

pub fn format_verification_failures(report: &VerificationReport) -> String {
    let mut message = format!(
        "VERIFICATION FAILED: {}/{} paths failed",
        report.failure_count(),
        report.path_count
    );
    for failure in &report.failures {
        message.push_str(&format!(
            "\n  Path {}: counterexample={:?}",
            failure.path_index + 1,
            failure.counterexample
        ));
    }
    message
}

fn total_stages(k: usize) -> usize {
    if k <= 1 {
        return 0;
    }
    total_stages(k / 2) + k.ilog2() as usize
}

fn validate_metadata(metadata: &SolutionJsonMetadata) -> Result<(), String> {
    if metadata.num_vecs != 2 {
        return Err(format!(
            "verify currently supports exactly 2 vectors, got {}",
            metadata.num_vecs
        ));
    }
    if !matches!(metadata.dtype, DTypeArg::I32 | DTypeArg::I64) {
        return Err(format!(
            "verify currently supports only i32 and i64, got {}",
            metadata.dtype.cli_name()
        ));
    }
    Ok(())
}

fn total_elements(metadata: &SolutionJsonMetadata) -> Result<usize, String> {
    Ok(2 * elements_per_vector(metadata)?)
}

fn elements_per_vector(metadata: &SolutionJsonMetadata) -> Result<usize, String> {
    let total_bits = total_bits(metadata.arch);
    let element_bits = metadata.dtype.element_bits();
    if !total_bits.is_multiple_of(element_bits) {
        return Err(format!(
            "{}-bit vectors cannot be split into {}-bit elements",
            total_bits, element_bits
        ));
    }
    Ok(total_bits / element_bits)
}

fn total_bits(arch: ArchArg) -> usize {
    match arch {
        ArchArg::Avx2 => 256,
        ArchArg::Avx512 => 512,
    }
}

fn build_verify_jobs(
    imported: &ImportedSolutionJson,
    top_k: Option<usize>,
) -> Result<Vec<VerifyJob>, String> {
    if imported.assigned_paths.is_empty() {
        let limit = top_k.unwrap_or(imported.paths.len());
        imported
            .paths
            .iter()
            .take(limit)
            .enumerate()
            .map(|(path_index, path)| {
                Ok(VerifyJob {
                    path_index,
                    steps: extract_verify_steps(path, &imported.transition_table)?,
                })
            })
            .collect()
    } else {
        let limit = top_k.unwrap_or(imported.assigned_paths.len());
        imported
            .assigned_paths
            .iter()
            .take(limit)
            .enumerate()
            .map(|(path_index, path)| {
                Ok(VerifyJob {
                    path_index,
                    steps: extract_assigned_verify_steps(path)?,
                })
            })
            .collect()
    }
}

fn extract_verify_steps(
    path: &CompletePath,
    table: &TransitionTable,
) -> Result<Vec<VerifyStep>, String> {
    validate_path_continuity(path)?;
    path.iter()
        .map(|(stage, input_state, output_state)| {
            let transition = (input_state.clone(), output_state.clone());
            let gadgets = table
                .get_all_transitions(*stage)
                .get(&transition)
                .ok_or_else(|| format!("path references missing stage {stage} transition"))?;
            let [gadget] = gadgets.as_slice() else {
                return Err(format!(
                    "stage {stage} transition has {} gadgets; verify requires concrete one-gadget JSON",
                    gadgets.len()
                ));
            };
            Ok(VerifyStep {
                gadget: gadget.clone(),
                input_state: input_state.clone(),
                output_state: output_state.clone(),
            })
        })
        .collect()
}

fn extract_assigned_verify_steps(path: &AssignedPath) -> Result<Vec<VerifyStep>, String> {
    let complete_path = path.as_complete_path();
    validate_path_continuity(&complete_path)?;
    Ok(path
        .steps()
        .iter()
        .map(|step| VerifyStep {
            gadget: step.gadget().clone(),
            input_state: step.input().clone(),
            output_state: step.output().clone(),
        })
        .collect())
}

fn validate_path_continuity(path: &CompletePath) -> Result<(), String> {
    for window in path.windows(2) {
        let (stage, _, output_state) = &window[0];
        let (next_stage, next_input_state, _) = &window[1];
        if output_state != next_input_state {
            return Err(format!(
                "path is disconnected between stage {stage} and stage {next_stage}: output_state does not match next input_state"
            ));
        }
    }
    Ok(())
}

fn resolve_worker_count(requested_workers: usize, job_count: usize) -> usize {
    if job_count <= 1 {
        return 1;
    }
    let workers = if requested_workers > 0 {
        requested_workers
    } else {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    };
    workers.max(1).min(job_count)
}

fn run_verify_jobs_sequential(
    metadata: &SolutionJsonMetadata,
    jobs: Vec<VerifyJob>,
) -> Result<Vec<(usize, VerificationResult)>, String> {
    let verifier = PathVerifier::new(metadata)?;
    jobs.into_iter()
        .map(|job| {
            verifier
                .verify_steps(&job.steps)
                .map(|result| (job.path_index, result))
        })
        .collect()
}

fn run_verify_jobs_parallel(
    metadata: &SolutionJsonMetadata,
    jobs: Vec<VerifyJob>,
    worker_count: usize,
) -> Result<Vec<(usize, VerificationResult)>, String> {
    let job_count = jobs.len();
    let (job_sender, job_receiver) = mpsc::channel::<Option<VerifyJob>>();
    let job_receiver = Arc::new(Mutex::new(job_receiver));
    let (result_sender, result_receiver) =
        mpsc::channel::<Result<(usize, VerificationResult), String>>();

    thread::scope(|scope| {
        for _ in 0..worker_count {
            let job_receiver = Arc::clone(&job_receiver);
            let result_sender = result_sender.clone();
            let metadata = metadata.clone();
            scope.spawn(move || {
                let verifier = match PathVerifier::new(&metadata) {
                    Ok(verifier) => verifier,
                    Err(error) => {
                        let _ = result_sender.send(Err(error));
                        return;
                    }
                };
                loop {
                    let message = {
                        let receiver = job_receiver
                            .lock()
                            .expect("verify job receiver lock should not be poisoned");
                        receiver.recv()
                    };
                    let Ok(Some(job)) = message else {
                        break;
                    };
                    let result = verifier
                        .verify_steps(&job.steps)
                        .map(|result| (job.path_index, result));
                    if result_sender.send(result).is_err() {
                        break;
                    }
                }
            });
        }
        drop(result_sender);

        for job in jobs {
            job_sender
                .send(Some(job))
                .map_err(|error| format!("failed to send verify job: {error}"))?;
        }
        for _ in 0..worker_count {
            let _ = job_sender.send(None);
        }

        let mut results = Vec::with_capacity(job_count);
        for _ in 0..job_count {
            let result = result_receiver
                .recv()
                .map_err(|error| format!("verify worker stopped unexpectedly: {error}"))??;
            results.push(result);
        }
        Ok(results)
    })
}

impl PathVerifier {
    fn new(metadata: &SolutionJsonMetadata) -> Result<Self, String> {
        validate_metadata(metadata)?;
        Ok(Self {
            metadata: metadata.clone(),
            elements_per_vector: elements_per_vector(metadata)?,
            lane_width: metadata.dtype.element_bits() as u32,
            total_bits: total_bits(metadata.arch) as u32,
        })
    }

    fn verify_steps(&self, steps: &[VerifyStep]) -> Result<VerificationResult, String> {
        if steps.is_empty() {
            return Err("cannot verify an empty path".to_owned());
        }

        let boundaries = compute_recursion_boundaries(2 * self.elements_per_vector)?;
        if boundaries.is_empty() {
            return self.verify_steps_monolithic(steps, self.metadata.natural_order);
        }
        self.verify_steps_chunked(steps)
    }

    fn verify_steps_monolithic(
        &self,
        steps: &[VerifyStep],
        natural_order: bool,
    ) -> Result<VerificationResult, String> {
        let n = 2 * self.elements_per_vector;
        let (input_elems, sorted_elems) = self.simulate_steps(steps, natural_order)?;
        let started = Instant::now();
        let solver = Solver::new();
        solver.assert(sorted_constraints(&sorted_elems, 0, n).not());
        let result = solver.check();
        let solver_time = started.elapsed();

        if result == SatResult::Unsat {
            return Ok(VerificationResult {
                verified: true,
                counterexample: None,
                solver_time,
            });
        }

        Ok(VerificationResult {
            verified: false,
            counterexample: Some(counterexample(&solver, &input_elems)?),
            solver_time,
        })
    }

    fn verify_steps_chunked(&self, steps: &[VerifyStep]) -> Result<VerificationResult, String> {
        let n = 2 * self.elements_per_vector;
        let sort_stages = if self.metadata.natural_order {
            steps.len().saturating_sub(1)
        } else {
            steps.len()
        };
        let boundaries = compute_recursion_boundaries(n)?;
        let mut levels = Vec::new();
        let mut previous = 0;
        for (level_idx, boundary_stage) in boundaries.into_iter().enumerate() {
            let end = boundary_stage + 1;
            if end > sort_stages {
                break;
            }
            if end > previous {
                let pre_group_size = if level_idx == 0 {
                    0
                } else {
                    MIN_GROUP_SIZE * (1 << (level_idx - 1))
                };
                let post_group_size = MIN_GROUP_SIZE * (1 << level_idx);
                levels.push((previous, end, pre_group_size, post_group_size));
                previous = end;
            }
        }

        if previous < steps.len() {
            let pre_group_size = levels.last().map(|level| level.3).unwrap_or(0);
            levels.push((previous, steps.len(), pre_group_size, n));
        }

        let mut total_solver_time = Duration::ZERO;
        for (level_idx, (start, end, pre_group_size, post_group_size)) in
            levels.iter().copied().enumerate()
        {
            let is_last_level = level_idx + 1 == levels.len();
            let natural_order = self.metadata.natural_order && is_last_level;
            let chunk_steps = &steps[start..end];
            let num_groups = n / post_group_size;
            for group_idx in 0..num_groups {
                let group_start = group_idx * post_group_size;
                let group_end = group_start + post_group_size;
                let result = self.verify_chunk(
                    chunk_steps,
                    pre_group_size,
                    group_start,
                    group_end,
                    natural_order,
                )?;
                total_solver_time += result.solver_time;
                if !result.verified {
                    return Ok(VerificationResult {
                        verified: false,
                        counterexample: result.counterexample,
                        solver_time: total_solver_time,
                    });
                }
            }
        }

        Ok(VerificationResult {
            verified: true,
            counterexample: None,
            solver_time: total_solver_time,
        })
    }

    fn verify_chunk(
        &self,
        steps: &[VerifyStep],
        precondition_group_size: usize,
        post_start: usize,
        post_end: usize,
        natural_order: bool,
    ) -> Result<VerificationResult, String> {
        let n = 2 * self.elements_per_vector;
        let (input_elems, sorted_elems) = self.simulate_steps(steps, natural_order)?;
        let started = Instant::now();
        let solver = Solver::new();

        if precondition_group_size > 0 {
            let mut preconditions = Vec::new();
            for group_start in (0..n).step_by(precondition_group_size) {
                let group_end = group_start + precondition_group_size;
                for idx in group_start..group_end - 1 {
                    preconditions.push(input_elems[idx].bvsle(&input_elems[idx + 1]));
                }
            }
            solver.assert(Bool::and(&preconditions));
        }

        solver.assert(sorted_constraints(&sorted_elems, post_start, post_end).not());
        let result = solver.check();
        let solver_time = started.elapsed();
        if result == SatResult::Unsat {
            return Ok(VerificationResult {
                verified: true,
                counterexample: None,
                solver_time,
            });
        }

        Ok(VerificationResult {
            verified: false,
            counterexample: Some(counterexample(&solver, &input_elems)?),
            solver_time,
        })
    }

    fn simulate_steps(
        &self,
        steps: &[VerifyStep],
        natural_order: bool,
    ) -> Result<(Vec<BV>, Vec<BV>), String> {
        let n = 2 * self.elements_per_vector;
        let input_elems = (0..n)
            .map(|idx| BV::new_const(format!("e_{idx}"), self.lane_width))
            .collect::<Vec<_>>();

        let first_state = &steps[0].input_state;
        let mut top_vec = self.build_register(&input_elems, &first_state.0)?;
        let mut bottom_vec = self.build_register(&input_elems, &first_state.1)?;
        let solver = Solver::new();

        for (step_idx, step) in steps.iter().enumerate() {
            let is_last_stage = step_idx + 1 == steps.len();
            let new_top = self.apply_concrete_instructions(
                &top_vec,
                &bottom_vec,
                step.gadget.top_instructions(),
                true,
                &solver,
            )?;
            let new_bottom = self.apply_concrete_instructions(
                &top_vec,
                &bottom_vec,
                step.gadget.bottom_instructions(),
                false,
                &solver,
            )?;

            if natural_order && is_last_stage {
                top_vec = new_top;
                bottom_vec = new_bottom;
            } else {
                top_vec = generic_min(&new_top, &new_bottom, self.total_bits, self.lane_width);
                bottom_vec = generic_max(&new_top, &new_bottom, self.total_bits, self.lane_width);
            }
        }

        let sorted_elems = self.extract_output_elements(
            &top_vec,
            &bottom_vec,
            &steps[steps.len() - 1].output_state,
        )?;
        Ok((input_elems, sorted_elems))
    }

    fn build_register(&self, elems: &[BV], labels: &[u64]) -> Result<BV, String> {
        if labels.len() != self.elements_per_vector {
            return Err(format!(
                "expected {} labels per vector, got {}",
                self.elements_per_vector,
                labels.len()
            ));
        }
        let mut lane_elems = Vec::with_capacity(labels.len());
        for &label in labels {
            if label == 0 || label as usize > elems.len() {
                return Err(format!(
                    "element label {label} out of range 1..={}",
                    elems.len()
                ));
            }
            lane_elems.push(elems[label as usize - 1].clone());
        }
        lane_elems.reverse();
        Ok(concat_msb_first(&lane_elems).simplify())
    }

    fn extract_output_elements(
        &self,
        top_vec: &BV,
        bottom_vec: &BV,
        state: &StateTuple,
    ) -> Result<Vec<BV>, String> {
        let n = 2 * self.elements_per_vector;
        let mut label_to_pos = HashMap::new();
        for (lane, label) in state.0.iter().copied().enumerate() {
            label_to_pos.insert(label, (true, lane));
        }
        for (lane, label) in state.1.iter().copied().enumerate() {
            label_to_pos.insert(label, (false, lane));
        }

        (1..=n as u64)
            .map(|label| {
                let (is_top, lane) = label_to_pos
                    .get(&label)
                    .copied()
                    .ok_or_else(|| format!("final state does not contain label {label}"))?;
                let reg = if is_top { top_vec } else { bottom_vec };
                Ok(extract_element(reg, self.lane_width, lane).simplify())
            })
            .collect()
    }

    fn apply_concrete_instructions(
        &self,
        top_reg: &BV,
        bottom_reg: &BV,
        instructions: &[InstructionSpec],
        is_top: bool,
        solver: &Solver,
    ) -> Result<BV, String> {
        let mut current_reg = if is_top {
            top_reg.clone()
        } else {
            bottom_reg.clone()
        };
        let mut previous_output = None;
        let mut results = Vec::new();

        for instruction in instructions {
            let mut args = BTreeMap::new();
            for (key, value) in instruction.args() {
                let context = ArgResolveContext {
                    top_reg,
                    bottom_reg,
                    current_reg: &current_reg,
                    previous_output: previous_output.as_ref(),
                    results: &results,
                };
                args.insert(*key, self.resolve_arg(key, value, &context)?);
            }
            current_reg = dispatch_intrinsic(instruction.intrinsic_name(), &args, solver)
                .map_err(|error| error.to_string())?;
            previous_output = Some(current_reg.clone());
            results.push(current_reg.clone());
        }

        Ok(current_reg)
    }

    fn resolve_arg(
        &self,
        key: &'static str,
        value: &InstructionArg,
        context: &ArgResolveContext<'_>,
    ) -> Result<BV, String> {
        match value {
            InstructionArg::Input(name) => match name.as_str() {
                "top" => Ok(context.top_reg.clone()),
                "bottom" => Ok(context.bottom_reg.clone()),
                "input" | "a" => Ok(context.current_reg.clone()),
                "prev" => context
                    .previous_output
                    .cloned()
                    .ok_or_else(|| "'prev' referenced but no previous output".to_owned()),
                name if name.starts_with("result_") => {
                    let index = name["result_".len()..]
                        .parse::<usize>()
                        .map_err(|_| format!("invalid result reference `{name}`"))?;
                    context
                        .results
                        .get(index)
                        .cloned()
                        .ok_or_else(|| format!("'{name}' out of range"))
                }
                other => Err(format!("unsupported instruction input reference `{other}`")),
            },
            InstructionArg::U64(value) => Ok(BV::from_u64(*value, self.arg_bit_width(key))),
            InstructionArg::BitVec { bits, hex } => bv_from_hex(*bits, hex),
        }
    }

    fn arg_bit_width(&self, key: &'static str) -> u32 {
        match key {
            "imm8" => 8,
            "k" => self.elements_per_vector as u32,
            _ => self.total_bits,
        }
    }
}

fn sorted_constraints(elems: &[BV], start: usize, end: usize) -> Bool {
    let constraints = (start..end - 1)
        .map(|idx| elems[idx].bvsle(&elems[idx + 1]))
        .collect::<Vec<_>>();
    Bool::and(&constraints).simplify()
}

fn counterexample(solver: &Solver, elems: &[BV]) -> Result<Vec<u64>, String> {
    let model = solver
        .get_model()
        .ok_or_else(|| "solver returned no counterexample model".to_owned())?;
    elems
        .iter()
        .enumerate()
        .map(|(idx, elem)| {
            model
                .eval(elem, true)
                .and_then(|value| value.as_u64())
                .ok_or_else(|| format!("model has no value for e_{idx}"))
        })
        .collect()
}

fn bv_from_hex(bits: u32, hex: &str) -> Result<BV, String> {
    if bits == 0 {
        return Err("bitvec width must be nonzero".to_owned());
    }
    let cleaned = hex
        .strip_prefix("0x")
        .or_else(|| hex.strip_prefix("#x"))
        .unwrap_or(hex);
    let expected_digits = bits.div_ceil(4) as usize;
    if cleaned.len() > expected_digits {
        return Err(format!(
            "bitvec hex value has {} digits for {bits}-bit value",
            cleaned.len()
        ));
    }
    let padded = format!("{cleaned:0>expected_digits$}");
    let first_chunk_bits = {
        let rem = bits % 64;
        if rem == 0 { 64 } else { rem }
    };
    let mut offset = 0usize;
    let mut chunk_bits = first_chunk_bits;
    let mut chunks = Vec::new();
    while offset < padded.len() {
        let hex_digits = chunk_bits.div_ceil(4) as usize;
        let end = offset + hex_digits;
        let value = u64::from_str_radix(&padded[offset..end], 16)
            .map_err(|error| format!("invalid bitvec hex `{hex}`: {error}"))?;
        chunks.push(BV::from_u64(value, chunk_bits));
        offset = end;
        chunk_bits = 64;
    }
    let first = chunks
        .first()
        .cloned()
        .ok_or_else(|| "bitvec hex produced no chunks".to_owned())?;
    Ok(chunks
        .iter()
        .skip(1)
        .fold(first, |acc, chunk| acc.concat(chunk))
        .simplify())
}
