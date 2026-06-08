use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use uica_core::{
    SimulationInput, SimulationOptions, SimulationRequest, UipackSource, compute_issue_limit,
    get_micro_arch,
};
use uica_data::{
    DataPackManifest, InstructionRecord, MappedUiPackRuntime, load_manifest_runtime,
    record_view_to_instruction_record,
};
use uica_decode_ir::DecodedInstruction;
use uica_model::{Invocation, ReportBundle};

use crate::instruction_stream::{InstructionBlock, ModeledInstruction, Operand, Register};
use crate::instruction_stream::{
    LoweringOptions, lower_assigned_path_snapshot, lower_assigned_paths,
};
use crate::json_exporter::SolutionJsonMetadata;
use crate::scoring::{AssignedPath, GadgetCost, PathCost, PathScoringSnapshot, Scorer};
use crate::transition_table::{TransitionRef, TransitionTable};

pub struct UiPackScorer {
    arch: String,
    issue_width: i32,
    runtime: MappedUiPackRuntime,
    catalog: BTreeMap<String, InstructionRecord>,
    solution_metadata: Option<SolutionJsonMetadata>,
}

pub struct UiBlockSimulation {
    pub instruction_count: usize,
    pub throughput_cycles_per_iteration: f64,
    pub cycles_simulated: u32,
    pub iterations_simulated: u32,
    pub reports: Option<ReportBundle>,
}

impl UiPackScorer {
    pub fn from_data_dir(data_dir: impl AsRef<Path>, arch: &str) -> Result<Self, String> {
        let manifest = data_dir.as_ref().join("manifest.json");
        let manifest_bytes = fs::read(&manifest)
            .map_err(|error| format!("failed to read {}: {error}", manifest.display()))?;
        let manifest_data: DataPackManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|error| format!("failed to parse {}: {error}", manifest.display()))?;
        let canonical_arch = manifest_data
            .architectures
            .keys()
            .find(|candidate| candidate.eq_ignore_ascii_case(arch))
            .cloned()
            .ok_or_else(|| {
                format!(
                    "uiCA data for {arch} is not present in {}. Run `bitonic-codegen fetch-uica-data --target-cpu {arch}` first.",
                    data_dir.as_ref().display()
                )
            })?;
        let runtime = load_manifest_runtime(&manifest, &canonical_arch).map_err(|error| {
            format!(
                "failed to load uiCA data for {arch} from {}: {error}. Run `bitonic-codegen fetch-uica-data --target-cpu {arch}` first.",
                data_dir.as_ref().display()
            )
        })?;
        Self::from_runtime(&canonical_arch, runtime)
    }

    pub fn from_runtime(arch: &str, runtime: MappedUiPackRuntime) -> Result<Self, String> {
        let (arch_name, catalog) = {
            let view = runtime.view().map_err(|error| error.to_string())?;
            let arch_name = view.arch().to_owned();
            let mut catalog = BTreeMap::new();
            for index in 0..view.record_count() {
                let record = view.record(index).map_err(|error| error.to_string())?;
                let owned = record_view_to_instruction_record(record).map_err(|error| {
                    format!(
                        "failed to decode uiCA record {} for {arch_name}: {error}",
                        index
                    )
                })?;
                catalog.entry(owned.string.clone()).or_insert(owned);
            }
            (arch_name, catalog)
        };
        let micro_arch = get_micro_arch(&arch_name)
            .ok_or_else(|| format!("uiCA microarchitecture is not supported: {arch}"))?;

        Ok(Self {
            arch: arch_name,
            issue_width: micro_arch.issue_width as i32,
            runtime,
            catalog,
            solution_metadata: None,
        })
    }

    pub fn with_solution_metadata(mut self, metadata: SolutionJsonMetadata) -> Self {
        self.solution_metadata = Some(metadata);
        self
    }

    pub fn rough_score_block(&self, block: &InstructionBlock) -> PathCost {
        let mut instruction_count = 0;
        let mut retire_slots = 0;
        let mut port_usage = Vec::new();

        for instruction in &block.instructions {
            if instruction.mnemonic.is_empty() {
                continue;
            }
            if instruction.mnemonic.eq_ignore_ascii_case("ret") {
                continue;
            }
            instruction_count += 1;
            let Some(record) = self.record_for_instruction(instruction) else {
                return PathCost::new(instruction_count, f64::INFINITY, f64::INFINITY);
            };
            retire_slots += record.perf.retire_slots.max(1);
            add_port_usage(&mut port_usage, &record.perf.ports);
        }

        let issue = compute_issue_limit(retire_slots, self.issue_width);
        let ports = compute_aggregated_port_usage_limit(&port_usage);
        let score = issue.max(ports);
        PathCost::new(instruction_count, score, score)
    }

    pub fn simulate_block(
        &self,
        block: &InstructionBlock,
        include_reports: bool,
    ) -> Result<UiBlockSimulation, String> {
        let mut decoded = Vec::new();
        let mut ip = 0u64;
        for instruction in &block.instructions {
            if instruction.mnemonic.is_empty() {
                continue;
            }
            if instruction.mnemonic.eq_ignore_ascii_case("ret") {
                continue;
            }
            let decoded_instruction = self.decoded_instruction(instruction, ip)?;
            ip += u64::from(decoded_instruction.len);
            decoded.push(decoded_instruction);
        }

        if decoded.is_empty() {
            return Ok(UiBlockSimulation {
                instruction_count: 0,
                throughput_cycles_per_iteration: 0.0,
                cycles_simulated: 0,
                iterations_simulated: 0,
                reports: None,
            });
        }

        let invocation = Invocation {
            arch: self.arch.clone(),
            ..Invocation::default()
        };
        let output = uica_core::simulate(SimulationRequest {
            input: SimulationInput::Decoded(&decoded),
            invocation: &invocation,
            uipack: UipackSource::Runtime(&self.runtime),
            options: SimulationOptions {
                include_reports,
                include_trace: false,
                ..SimulationOptions::default()
            },
        })?;
        let throughput_cycles_per_iteration = output
            .result
            .summary
            .throughput_cycles_per_iteration
            .unwrap_or(f64::INFINITY);
        Ok(UiBlockSimulation {
            instruction_count: decoded.len(),
            throughput_cycles_per_iteration,
            cycles_simulated: output.result.summary.cycles_simulated,
            iterations_simulated: output.result.summary.iterations_simulated,
            reports: output.reports,
        })
    }

    pub fn full_score_block(&self, block: &InstructionBlock) -> Result<PathCost, String> {
        let simulation = self.simulate_block(block, false)?;
        Ok(PathCost::new(
            simulation.instruction_count as u32,
            simulation.throughput_cycles_per_iteration,
            simulation.throughput_cycles_per_iteration,
        ))
    }

    fn decoded_instruction(
        &self,
        instruction: &ModeledInstruction,
        ip: u64,
    ) -> Result<DecodedInstruction, String> {
        let key = uops_key_for_instruction(instruction).ok_or_else(|| {
            format!(
                "unsupported instruction for uiCA scoring: {} {:?}",
                instruction.mnemonic, instruction.operands
            )
        })?;
        let record = self
            .catalog
            .get(&key)
            .ok_or_else(|| format!("uiCA data does not contain instruction key: {key}"))?;
        let explicit_reg_operands = explicit_register_operands(instruction);
        let output_regs = output_registers(instruction);
        let mut input_regs = input_registers(instruction);
        if key == "KMOVW (K, R32)" && input_regs.is_empty() {
            input_regs.push("EAX".to_owned());
        }
        let len = approximate_instruction_len(instruction);

        Ok(DecodedInstruction {
            ip,
            len,
            mnemonic: instruction.mnemonic.clone(),
            disasm: key,
            bytes: vec![0; len as usize],
            pos_nominal_opcode: 0,
            input_regs,
            output_regs,
            reads_flags: false,
            writes_flags: false,
            has_memory_read: instruction
                .operands
                .iter()
                .any(|operand| matches!(operand, Operand::ConstantRef(_))),
            has_memory_write: false,
            mem_addrs: vec![],
            implicit_rsp_change: 0,
            immediate: immediate_value(instruction),
            immediate_width_bits: immediate_value(instruction).map_or(0, |_| 8),
            has_66_prefix: false,
            iform: record.iform.clone(),
            iform_signature: String::new(),
            max_op_size_bytes: max_op_size_bytes(instruction),
            uses_high8_reg: false,
            explicit_reg_operands,
            agen: None,
            xml_attrs: BTreeMap::new(),
        })
    }

    fn record_for_instruction(
        &self,
        instruction: &ModeledInstruction,
    ) -> Option<&InstructionRecord> {
        let key = uops_key_for_instruction(instruction)?;
        self.catalog.get(&key)
    }
}

impl Scorer for UiPackScorer {
    fn score_gadget(&self, gadget: &gadget_synth::PermutationGadget) -> GadgetCost {
        let instruction_count =
            gadget.top_instructions().len() + gadget.bottom_instructions().len();
        let mut retire_slots = 0;
        let mut port_usage = Vec::new();
        let mut synthetic_unknowns = 0;

        for instruction in gadget
            .top_instructions()
            .iter()
            .chain(gadget.bottom_instructions())
        {
            let Some(key) = uops_key_for_intrinsic(instruction.intrinsic_name()) else {
                if instruction.intrinsic_name().starts_with("_mm") {
                    return GadgetCost::new(
                        instruction_count as u32,
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::INFINITY,
                    );
                }
                synthetic_unknowns += 1;
                retire_slots += 1;
                continue;
            };
            let Some(record) = self.catalog.get(key) else {
                return GadgetCost::new(
                    instruction_count as u32,
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::INFINITY,
                );
            };
            retire_slots += record.perf.retire_slots.max(1);
            add_port_usage(&mut port_usage, &record.perf.ports);
        }

        let issue = compute_issue_limit(retire_slots, self.issue_width);
        let ports = compute_aggregated_port_usage_limit(&port_usage);
        let score = if synthetic_unknowns == instruction_count {
            synthetic_unknowns as f64
        } else {
            issue.max(ports) + synthetic_unknowns as f64
        };
        GadgetCost::new(instruction_count as u32, score, score, score)
    }

    fn score_assigned_path(
        &self,
        assigned_path: &AssignedPath,
        table: &TransitionTable,
    ) -> PathCost {
        if let Some(metadata) = &self.solution_metadata {
            let stream = lower_assigned_paths(
                metadata,
                table,
                std::slice::from_ref(assigned_path),
                LoweringOptions {
                    include_comments: false,
                },
            );
            if let Some(block) = stream.blocks.first() {
                return self.rough_score_block(block);
            }
        }

        let instruction_count = assigned_path
            .path()
            .iter()
            .zip(assigned_path.gadgets())
            .map(|((stage, transition), gadget_index)| {
                let transition_ref = TransitionRef {
                    stage: stage
                        .try_into()
                        .expect("stage index should fit in transition ref"),
                    transition,
                };
                self.score_gadget(table.transition(transition_ref).gadget(*gadget_index))
                    .instruction_count()
            })
            .sum();
        let score = assigned_path
            .path()
            .iter()
            .zip(assigned_path.gadgets())
            .map(|((stage, transition), gadget_index)| {
                let transition_ref = TransitionRef {
                    stage: stage
                        .try_into()
                        .expect("stage index should fit in transition ref"),
                    transition,
                };
                self.score_gadget(table.transition(transition_ref).gadget(*gadget_index))
                    .score()
            })
            .sum();
        PathCost::new(instruction_count, score, score)
    }

    fn score_assigned_path_snapshot(
        &self,
        assigned_path: &AssignedPath,
        snapshot: &PathScoringSnapshot,
    ) -> PathCost {
        if let Some(metadata) = &self.solution_metadata {
            let block = lower_assigned_path_snapshot(
                metadata,
                snapshot,
                assigned_path,
                LoweringOptions {
                    include_comments: false,
                },
            );
            return self.rough_score_block(&block);
        }

        let instruction_count = snapshot
            .steps()
            .iter()
            .zip(assigned_path.gadgets())
            .map(|(step, gadget_index)| {
                self.score_gadget(step.gadget(*gadget_index))
                    .instruction_count()
            })
            .sum();
        let score = snapshot
            .steps()
            .iter()
            .zip(assigned_path.gadgets())
            .map(|(step, gadget_index)| self.score_gadget(step.gadget(*gadget_index)).score())
            .sum();
        PathCost::new(instruction_count, score, score)
    }
}

fn add_port_usage(port_usage: &mut Vec<(u128, i32)>, port_data: &BTreeMap<String, i32>) {
    for (ports, n_uops) in port_data {
        let port_set = normalize_port_mask(ports);
        if port_set == 0 {
            continue;
        }

        if let Some((_, total)) = port_usage.iter_mut().find(|(set, _)| *set == port_set) {
            *total += *n_uops;
        } else {
            port_usage.push((port_set, *n_uops));
        }
    }
}

fn compute_aggregated_port_usage_limit(port_usage: &[(u128, i32)]) -> f64 {
    if port_usage.is_empty() {
        return 0.0;
    }

    let mut limit: f64 = 0.0;
    for (left_set, _) in port_usage {
        for (right_set, _) in port_usage {
            let candidate = left_set | right_set;
            if candidate == 0 {
                continue;
            }

            let total_uops: i32 = port_usage
                .iter()
                .filter(|(set, _)| (*set & !candidate) == 0)
                .map(|(_, n_uops)| *n_uops)
                .sum();
            limit = limit.max(total_uops as f64 / candidate.count_ones() as f64);
        }
    }

    limit
}

fn normalize_port_mask(ports: &str) -> u128 {
    ports
        .bytes()
        .filter(|byte| byte.is_ascii() && !byte.is_ascii_whitespace())
        .fold(0_u128, |mask, byte| mask | (1_u128 << u32::from(byte)))
}

pub fn uops_key_for_instruction(instruction: &ModeledInstruction) -> Option<String> {
    let mnemonic = normalized_mnemonic(&instruction.mnemonic);
    let classes = operand_classes(instruction)?;
    Some(format!("{mnemonic} ({})", classes.join(", ")))
}

fn operand_classes(instruction: &ModeledInstruction) -> Option<Vec<String>> {
    let memory_class = if instruction.operands.iter().any(|operand| {
        matches!(
            operand,
            Operand::Register(Register::Zmm(_)) | Operand::MaskedRegister(Register::Zmm(_), _)
        )
    }) {
        "M512"
    } else {
        "M256"
    };

    let mut classes = Vec::new();
    for operand in &instruction.operands {
        match operand {
            Operand::Register(Register::Ymm(_)) => classes.push("YMM".to_owned()),
            Operand::Register(Register::Zmm(_)) => classes.push("ZMM".to_owned()),
            Operand::KMask(_) => classes.push("K".to_owned()),
            Operand::MaskedRegister(Register::Ymm(_), _) => {
                classes.push("YMM".to_owned());
                classes.push("K".to_owned());
            }
            Operand::MaskedRegister(Register::Zmm(_), _) => {
                classes.push("ZMM".to_owned());
                classes.push("K".to_owned());
            }
            Operand::Gpr("eax") => classes.push("R32".to_owned()),
            Operand::Gpr("rax") => classes.push("R64".to_owned()),
            Operand::Gpr(_) => classes.push("R64".to_owned()),
            Operand::Immediate(_) => classes.push("I8".to_owned()),
            Operand::ConstantRef(_) => classes.push(memory_class.to_owned()),
            Operand::Unsupported(_) => return None,
        }
    }
    if normalized_mnemonic(&instruction.mnemonic) == "KMOVW"
        && matches!(classes.as_slice(), [first, second] if first == "K" && second == "I8")
    {
        classes[1] = "R32".to_owned();
    }
    Some(classes)
}

fn normalized_mnemonic(mnemonic: &str) -> String {
    let upper = mnemonic.to_ascii_uppercase();
    match upper.as_str() {
        "VMOVDQA32" | "VMOVDQA64" => "VMOVDQA".to_owned(),
        other => other.to_owned(),
    }
}

fn uops_key_for_intrinsic(name: &str) -> Option<&'static str> {
    Some(match name {
        "_mm256_permute_ps" => "VPERMILPS (YMM, YMM, I8)",
        "_mm256_permute_pd" => "VPERMILPD (YMM, YMM, I8)",
        "_mm256_permute4x64_epi64" => "VPERMQ (YMM, YMM, I8)",
        "_mm256_permutexvar_epi32" => "VPERMD (YMM, YMM, YMM)",
        "_mm256_permutexvar_epi64" => "VPERMQ (YMM, YMM, YMM)",
        "_mm256_permutevar_ps" => "VPERMILPS (YMM, YMM, YMM)",
        "_mm256_permutevar_pd" => "VPERMILPD (YMM, YMM, YMM)",
        "_mm256_shuffle_ps" => "VSHUFPS (YMM, YMM, YMM, I8)",
        "_mm256_shuffle_pd" => "VSHUFPD (YMM, YMM, YMM, I8)",
        "_mm256_unpacklo_epi32" => "VPUNPCKLDQ (YMM, YMM, YMM)",
        "_mm256_unpackhi_epi32" => "VPUNPCKHDQ (YMM, YMM, YMM)",
        "_mm256_unpacklo_epi64" => "VPUNPCKLQDQ (YMM, YMM, YMM)",
        "_mm256_unpackhi_epi64" => "VPUNPCKHQDQ (YMM, YMM, YMM)",
        "_mm256_permute2x128_si256" => "VPERM2I128 (YMM, YMM, YMM, I8)",
        "_mm256_blend_ps" => "VBLENDPS (YMM, YMM, YMM, I8)",
        "_mm256_blend_pd" => "VBLENDPD (YMM, YMM, YMM, I8)",
        "_mm256_alignr_epi8" => "VPALIGNR (YMM, YMM, YMM, I8)",
        "_mm512_permutexvar_epi32" => "VPERMD (ZMM, ZMM, ZMM)",
        "_mm512_permutexvar_epi64" => "VPERMQ (ZMM, ZMM, ZMM)",
        "_mm512_permutex2var_epi32" => "VPERMI2D (ZMM, ZMM, ZMM)",
        "_mm512_permutex2var_epi64" => "VPERMI2Q (ZMM, ZMM, ZMM)",
        "_mm512_permute_ps" => "VPERMILPS (ZMM, ZMM, I8)",
        "_mm512_permute_pd" => "VPERMILPD (ZMM, ZMM, I8)",
        "_mm512_shuffle_ps" => "VSHUFPS (ZMM, ZMM, ZMM, I8)",
        "_mm512_shuffle_pd" => "VSHUFPD (ZMM, ZMM, ZMM, I8)",
        "_mm512_unpacklo_epi32" => "VPUNPCKLDQ (ZMM, ZMM, ZMM)",
        "_mm512_unpackhi_epi32" => "VPUNPCKHDQ (ZMM, ZMM, ZMM)",
        "_mm512_unpacklo_epi64" => "VPUNPCKLQDQ (ZMM, ZMM, ZMM)",
        "_mm512_unpackhi_epi64" => "VPUNPCKHQDQ (ZMM, ZMM, ZMM)",
        "_mm512_shuffle_i32x4" => "VSHUFI32X4 (ZMM, ZMM, ZMM, I8)",
        "_mm512_alignr_epi32" => "VALIGND (ZMM, ZMM, ZMM, I8)",
        "_mm512_alignr_epi64" => "VALIGNQ (ZMM, ZMM, ZMM, I8)",
        "_mm512_mask_permutex2var_epi32" => "VPERMI2D (ZMM, K, ZMM, ZMM)",
        "_mm512_mask_permutex2var_epi64" => "VPERMI2Q (ZMM, K, ZMM, ZMM)",
        "_mm512_mask_shuffle_ps" => "VSHUFPS (ZMM, K, ZMM, ZMM, I8)",
        "_mm512_mask_shuffle_pd" => "VSHUFPD (ZMM, K, ZMM, ZMM, I8)",
        "_mm512_mask_unpacklo_epi32" => "VPUNPCKLDQ (ZMM, K, ZMM, ZMM)",
        "_mm512_mask_unpackhi_epi32" => "VPUNPCKHDQ (ZMM, K, ZMM, ZMM)",
        "_mm512_mask_unpacklo_epi64" => "VPUNPCKLQDQ (ZMM, K, ZMM, ZMM)",
        "_mm512_mask_unpackhi_epi64" => "VPUNPCKHQDQ (ZMM, K, ZMM, ZMM)",
        "_mm512_mask_shuffle_i32x4" => "VSHUFI32X4 (ZMM, K, ZMM, ZMM, I8)",
        "_mm512_mask_alignr_epi32" => "VALIGND (ZMM, K, ZMM, ZMM, I8)",
        "_mm512_mask_alignr_epi64" => "VALIGNQ (ZMM, K, ZMM, ZMM, I8)",
        _ => return None,
    })
}

fn explicit_register_operands(instruction: &ModeledInstruction) -> Vec<String> {
    instruction
        .operands
        .iter()
        .filter_map(register_name)
        .collect()
}

fn input_registers(instruction: &ModeledInstruction) -> Vec<String> {
    instruction
        .operands
        .iter()
        .skip(1)
        .filter_map(register_name)
        .collect()
}

fn output_registers(instruction: &ModeledInstruction) -> Vec<String> {
    instruction
        .operands
        .first()
        .and_then(register_name)
        .into_iter()
        .collect()
}

fn register_name(operand: &Operand) -> Option<String> {
    match operand {
        Operand::Register(register) => Some(register.name().to_ascii_uppercase()),
        Operand::MaskedRegister(register, mask) => {
            Some(format!("{},K{mask}", register.name().to_ascii_uppercase()))
        }
        Operand::KMask(index) => Some(format!("K{index}")),
        Operand::Gpr(register) => Some(register.to_ascii_uppercase()),
        _ => None,
    }
}

fn immediate_value(instruction: &ModeledInstruction) -> Option<i64> {
    instruction
        .operands
        .iter()
        .find_map(|operand| match operand {
            Operand::Immediate(value) => Some(*value as i64),
            _ => None,
        })
}

fn max_op_size_bytes(instruction: &ModeledInstruction) -> u8 {
    if instruction.operands.iter().any(|operand| {
        matches!(
            operand,
            Operand::Register(Register::Zmm(_)) | Operand::MaskedRegister(Register::Zmm(_), _)
        )
    }) {
        64
    } else if instruction.operands.iter().any(|operand| {
        matches!(
            operand,
            Operand::Register(Register::Ymm(_)) | Operand::MaskedRegister(Register::Ymm(_), _)
        )
    }) {
        32
    } else {
        4
    }
}

fn approximate_instruction_len(instruction: &ModeledInstruction) -> u32 {
    if instruction.mnemonic.starts_with('v') || instruction.mnemonic.starts_with('V') {
        6
    } else {
        3
    }
}
