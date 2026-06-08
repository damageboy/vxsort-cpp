use std::collections::{BTreeMap, BTreeSet, HashMap};

use z3::ast::{Ast, BV, Bool};
use z3::{SatResult, Solver};
use z3_avx::lanes::extract_element;
use z3_avx::registers::{ymm_reg, zmm_reg};

use crate::intrinsics::{dispatch_intrinsic, dual_input_nodes, single_input_nodes};
use crate::types::{
    Arch, DType, GadgetGraph, GadgetNode, InputRef, InstructionArg, InstructionSpec, IntrinsicNode,
    LaneLabel, MuxNode, MuxPruning, OperandBinding, OperandKey, PermutationGadget, Symbolic,
    SynthesisError, SynthesisOptions, TargetPair, VectorState,
};

const MAX_SUPPORTED_GADGET_DEPTH: u8 = 2;

struct OutputEnumeration<'a> {
    solver: &'a Solver,
    graph: &'a GadgetGraph,
    target_pairs: &'a [TargetPair],
    options: SynthesisOptions,
    top_output: &'a BV,
    bottom_output: &'a BV,
    symbolic_vars: &'a HashMap<String, BV>,
    symbolic_order: &'a [String],
    select_vars: &'a BTreeSet<String>,
}

struct EvaluationContext<'a> {
    registers: &'a HashMap<&'static str, BV>,
    solver: &'a Solver,
    symbolic_vars: &'a mut HashMap<String, BV>,
    symbolic_order: &'a mut Vec<String>,
    select_vars: &'a mut BTreeSet<String>,
    eval_cache: &'a mut HashMap<String, BV>,
}

pub struct GadgetSynthesizer {
    arch: Arch,
    dtype: DType,
}

impl GadgetSynthesizer {
    pub fn new(arch: Arch, dtype: DType) -> Self {
        Self { arch, dtype }
    }

    pub fn candidate_graph_templates(
        &self,
        gadget_depth: u8,
    ) -> Result<Vec<GadgetGraph>, SynthesisError> {
        reject_unsupported_gadget_depth(gadget_depth)?;
        self.precompute_all_candidates(gadget_depth, true)
    }

    pub fn candidate_graph_templates_excluding_shared_prefix(
        &self,
        gadget_depth: u8,
    ) -> Result<Vec<GadgetGraph>, SynthesisError> {
        reject_unsupported_gadget_depth(gadget_depth)?;
        self.precompute_all_candidates(gadget_depth, false)
    }

    pub fn available_intrinsic_names(&self) -> BTreeSet<&'static str> {
        let mut names = BTreeSet::new();
        for node in single_input_nodes(self.arch, self.dtype, "top") {
            names.insert(node.name);
        }
        for node in dual_input_nodes(self.arch, self.dtype, "top", "bottom") {
            names.insert(node.name);
        }
        names
    }

    pub fn precompute_candidates_stratified(
        &self,
        gadget_depth: u8,
    ) -> Result<(Vec<GadgetGraph>, Vec<GadgetGraph>), SynthesisError> {
        let candidates = self.candidate_graph_templates(gadget_depth)?;
        let mut shallow = Vec::new();
        let mut deep = Vec::new();
        for graph in candidates {
            if graph_max_depth(&graph) <= 1 {
                shallow.push(graph);
            } else {
                deep.push(graph);
            }
        }
        Ok((shallow, deep))
    }

    pub fn synthesize_graph(
        &self,
        graph: &GadgetGraph,
        input_state: &VectorState,
        target_pairs: &[TargetPair],
        options: SynthesisOptions,
    ) -> Result<Vec<(PermutationGadget, VectorState)>, SynthesisError> {
        let lanes = self.elements_per_vector();
        if input_state.top().len() != lanes || input_state.bottom().len() != lanes {
            return Err(SynthesisError::InvalidInputState {
                expected_lanes: lanes,
                top: input_state.top().len(),
                bottom: input_state.bottom().len(),
            });
        }

        let solver = Solver::new();
        let top_reg = self.input_register("top_input");
        let bottom_reg = self.input_register("bottom_input");
        self.constrain_input_register(&solver, &top_reg, input_state.top());
        self.constrain_input_register(&solver, &bottom_reg, input_state.bottom());

        let registers = HashMap::from([("top", top_reg.clone()), ("bottom", bottom_reg.clone())]);
        let mut symbolic_vars = HashMap::new();
        let mut symbolic_order = Vec::new();
        let mut select_vars = BTreeSet::new();
        let mut eval_cache = HashMap::new();

        let mut eval_context = EvaluationContext {
            registers: &registers,
            solver: &solver,
            symbolic_vars: &mut symbolic_vars,
            symbolic_order: &mut symbolic_order,
            select_vars: &mut select_vars,
            eval_cache: &mut eval_cache,
        };

        let top_output = match &graph.top {
            Some(node) => self.evaluate_intrinsic_node(node, &mut eval_context)?,
            None => top_reg,
        };
        let bottom_output = match &graph.bottom {
            Some(node) => self.evaluate_intrinsic_node(node, &mut eval_context)?,
            None => bottom_reg,
        };

        self.constrain_outputs_to_target_pairs(
            &solver,
            &top_output,
            &bottom_output,
            target_pairs,
            options.allow_any_lane_order,
        );

        self.enumerate_outputs(OutputEnumeration {
            solver: &solver,
            graph,
            target_pairs,
            options,
            top_output: &top_output,
            bottom_output: &bottom_output,
            symbolic_vars: &symbolic_vars,
            symbolic_order: &symbolic_order,
            select_vars: &select_vars,
        })
    }

    fn enumerate_outputs(
        &self,
        enumeration: OutputEnumeration<'_>,
    ) -> Result<Vec<(PermutationGadget, VectorState)>, SynthesisError> {
        let OutputEnumeration {
            solver,
            graph,
            target_pairs,
            options,
            top_output,
            bottom_output,
            symbolic_vars,
            symbolic_order,
            select_vars,
        } = enumeration;
        let assertions = solver.get_assertions();
        let mut results = Vec::new();

        if symbolic_vars.is_empty() {
            if solver.check() != SatResult::Sat {
                return Ok(results);
            }
            let model = solver.get_model().expect("sat solver should have a model");
            let output_state = self.output_state_from_model_or_target_pairs(
                &model,
                top_output,
                bottom_output,
                target_pairs,
                options.allow_any_lane_order,
            )?;
            let gadget = PermutationGadget::new(
                self.concretize_root(graph.top.as_ref(), &model, symbolic_vars)?,
                self.concretize_root(graph.bottom.as_ref(), &model, symbolic_vars)?,
            );
            results.push((gadget, output_state));
            return Ok(results);
        }

        let select_terms = symbolic_order
            .iter()
            .filter(|name| select_vars.contains(*name))
            .map(|name| {
                (
                    name,
                    symbolic_vars
                        .get(name)
                        .expect("symbolic order should reference an existing variable"),
                )
            })
            .collect::<Vec<_>>();
        let mut wiring_blocks = Vec::new();
        loop {
            let mut wiring_constraints = Vec::new();
            let mut wiring_block_terms = Vec::new();

            if !select_terms.is_empty() {
                let wiring_solver = Solver::new();
                for assertion in &assertions {
                    wiring_solver.assert(assertion);
                }
                for block in &wiring_blocks {
                    wiring_solver.assert(block);
                }

                if wiring_solver.check() != SatResult::Sat {
                    break;
                }

                let wiring_model = wiring_solver
                    .get_model()
                    .expect("sat solver should have a model");
                for (name, select_var) in select_terms.iter().copied() {
                    let value = wiring_model
                        .eval(select_var, true)
                        .ok_or_else(|| SynthesisError::ModelMissingValue(name.clone()))?;
                    wiring_constraints.push(select_var.eq(&value));
                    wiring_block_terms.push(select_var.eq(&value).not());
                }
            }

            let mut output_blocks = Vec::new();
            for _ in 0..options.max_unique_outputs {
                let output_solver = Solver::new();
                for assertion in &assertions {
                    output_solver.assert(assertion);
                }
                for constraint in &wiring_constraints {
                    output_solver.assert(constraint);
                }
                for block in &output_blocks {
                    output_solver.assert(block);
                }

                if output_solver.check() != SatResult::Sat {
                    break;
                }

                let model = output_solver
                    .get_model()
                    .expect("sat solver should have a model");
                let top_value = model
                    .eval(top_output, true)
                    .ok_or_else(|| SynthesisError::ModelMissingValue("top_output".to_owned()))?;
                let bottom_value = model
                    .eval(bottom_output, true)
                    .ok_or_else(|| SynthesisError::ModelMissingValue("bottom_output".to_owned()))?;
                let output_state = self.output_state_from_model_or_target_pairs(
                    &model,
                    top_output,
                    bottom_output,
                    target_pairs,
                    options.allow_any_lane_order,
                )?;
                let gadget = PermutationGadget::new(
                    self.concretize_root(graph.top.as_ref(), &model, symbolic_vars)?,
                    self.concretize_root(graph.bottom.as_ref(), &model, symbolic_vars)?,
                );
                results.push((gadget, output_state));

                output_blocks.push(Bool::or(&[
                    top_output.eq(&top_value).not(),
                    bottom_output.eq(&bottom_value).not(),
                ]));
            }

            if select_terms.is_empty() {
                break;
            }
            wiring_blocks.push(Bool::or(&wiring_block_terms));
        }

        results.sort_by_key(|(_, state)| state.as_tuple());
        Ok(results)
    }

    fn precompute_all_candidates(
        &self,
        gadget_depth: u8,
        include_shared_prefix: bool,
    ) -> Result<Vec<GadgetGraph>, SynthesisError> {
        let mut candidates = Vec::new();
        for top_depth in 0..=gadget_depth {
            for bottom_depth in 0..=gadget_depth {
                candidates
                    .extend(self.generate_candidate_graphs_at_depth(top_depth, bottom_depth)?);
            }
        }
        if include_shared_prefix && gadget_depth >= 2 {
            candidates.extend(self.build_shared_prefix_graphs());
        }
        Ok(candidates)
    }

    fn generate_candidate_graphs_at_depth(
        &self,
        top_depth: u8,
        bottom_depth: u8,
    ) -> Result<Vec<GadgetGraph>, SynthesisError> {
        if top_depth == 0 && bottom_depth == 0 {
            return Ok(Vec::new());
        }

        let top_graphs = self.build_gadget_graphs(top_depth, "top")?;
        let bottom_graphs = self.build_gadget_graphs(bottom_depth, "bottom")?;
        Ok(top_graphs
            .iter()
            .flat_map(|top| {
                bottom_graphs.iter().map(|bottom| GadgetGraph {
                    top: top.clone(),
                    bottom: bottom.clone(),
                })
            })
            .collect())
    }

    fn build_gadget_graphs(
        &self,
        depth: u8,
        input_name: &'static str,
    ) -> Result<Vec<Option<IntrinsicNode>>, SynthesisError> {
        if depth == 0 {
            return Ok(vec![None]);
        }

        let single_intrinsics = single_input_nodes(self.arch, self.dtype, input_name);
        let dual_intrinsics = dual_input_nodes(self.arch, self.dtype, "top", "bottom");

        if depth == 1 {
            let mut result: Vec<Option<IntrinsicNode>> =
                single_intrinsics.into_iter().map(Some).collect();

            for dual in dual_intrinsics {
                result.push(Some(dual.clone()));
                if !dual.isomorphic_order && input_ref_operand_count(&dual) == 2 {
                    result.push(Some(swap_register_operands(&dual)));
                }
            }

            return Ok(result);
        }

        if depth == 2 {
            let mut result = Vec::new();

            for inst0 in &single_intrinsics {
                for inst1 in &single_intrinsics {
                    result.push(Some(wire_hardwired(inst0, inst1)));
                }
            }

            for inst0 in &dual_intrinsics {
                let mut inst0_variants = vec![inst0.clone()];
                if !inst0.isomorphic_order && input_ref_operand_count(inst0) == 2 {
                    inst0_variants.push(swap_register_operands(inst0));
                }
                for inst0_variant in &inst0_variants {
                    for inst1 in &single_intrinsics {
                        result.push(Some(wire_hardwired(inst0_variant, inst1)));
                    }
                }
            }

            for inst0 in &single_intrinsics {
                for inst1 in &dual_intrinsics {
                    result.push(Some(wire_with_muxes(inst0, inst1, input_name)));
                }
            }

            return Ok(result);
        }

        Err(SynthesisError::UnsupportedGadgetDepth {
            depth,
            max_supported: MAX_SUPPORTED_GADGET_DEPTH,
        })
    }

    fn build_shared_prefix_graphs(&self) -> Vec<GadgetGraph> {
        let single_intrinsics_top = single_input_nodes(self.arch, self.dtype, "top");
        let single_intrinsics_bottom = single_input_nodes(self.arch, self.dtype, "bottom");
        let dual_intrinsics = dual_input_nodes(self.arch, self.dtype, "top", "bottom");
        let mut graphs = Vec::new();

        for prefix_top in &single_intrinsics_top {
            for prefix_bot in &single_intrinsics_bottom {
                for dual in &dual_intrinsics {
                    if register_operand_keys(dual).len() != 2 {
                        continue;
                    }

                    let has_symbolic = has_direct_symbolic_operand(dual);

                    if dual.isomorphic_order {
                        if !has_symbolic {
                            continue;
                        }
                        let tail_top =
                            build_shared_prefix_tail(dual, prefix_top, prefix_bot, "top");
                        let tail_bot =
                            build_shared_prefix_tail(dual, prefix_top, prefix_bot, "bot");
                        graphs.push(GadgetGraph::new(Some(tail_top), Some(tail_bot)));
                    } else if has_symbolic {
                        for (reg_a, reg_b) in [(prefix_top, prefix_bot), (prefix_bot, prefix_top)] {
                            let tail_top = build_shared_prefix_tail(dual, reg_a, reg_b, "top");
                            let tail_bot = build_shared_prefix_tail(dual, reg_a, reg_b, "bot");
                            graphs.push(GadgetGraph::new(Some(tail_top), Some(tail_bot)));
                        }
                    } else {
                        let tail_top =
                            build_shared_prefix_tail(dual, prefix_top, prefix_bot, "top");
                        let tail_bot =
                            build_shared_prefix_tail(dual, prefix_bot, prefix_top, "bot");
                        graphs.push(GadgetGraph::new(Some(tail_top), Some(tail_bot)));
                    }
                }
            }
        }

        graphs
    }

    fn register_bits(&self) -> u32 {
        match self.arch {
            Arch::Avx2 => 256,
            Arch::Avx512 => 512,
        }
    }

    fn lane_bits(&self) -> u32 {
        match self.dtype {
            DType::I32 => 32,
            DType::I64 => 64,
        }
    }

    fn elements_per_vector(&self) -> usize {
        (self.register_bits() / self.lane_bits()) as usize
    }

    fn input_register(&self, name: &str) -> BV {
        match self.arch {
            Arch::Avx2 => ymm_reg(name),
            Arch::Avx512 => zmm_reg(name),
        }
    }

    fn constrain_input_register(&self, solver: &Solver, reg: &BV, values: &[LaneLabel]) {
        let lane_bits = self.lane_bits();
        for (lane_idx, value) in values.iter().enumerate() {
            solver.assert(
                extract_element(reg, lane_bits, lane_idx)
                    .eq(BV::from_u64(u64::from(*value), lane_bits)),
            );
        }
    }

    fn evaluate_intrinsic_node(
        &self,
        node: &IntrinsicNode,
        context: &mut EvaluationContext<'_>,
    ) -> Result<BV, SynthesisError> {
        let cache_key = intrinsic_node_key(node);
        if let Some(cached) = context.eval_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let mut args = BTreeMap::new();
        for binding in node.operands.iter() {
            args.insert(
                binding.key.as_str(),
                self.evaluate_graph_node(&binding.node, node.name, context)?,
            );
        }
        let result = dispatch_intrinsic(node.name, &args, context.solver)?;
        apply_mux_pruning(node, context.symbolic_vars, context.solver)?;
        context.eval_cache.insert(cache_key, result.clone());
        Ok(result)
    }

    fn evaluate_graph_node(
        &self,
        node: &GadgetNode,
        parent_intrinsic: &'static str,
        context: &mut EvaluationContext<'_>,
    ) -> Result<BV, SynthesisError> {
        match node {
            GadgetNode::Input(input) => {
                context.registers.get(input.name).cloned().ok_or_else(|| {
                    SynthesisError::MissingOperand {
                        intrinsic: parent_intrinsic.to_owned(),
                        operand: input.name.to_owned(),
                    }
                })
            }
            GadgetNode::Symbolic(symbolic) => Ok(symbolic_bv(
                symbolic,
                context.symbolic_vars,
                context.symbolic_order,
            )),
            GadgetNode::Mux(mux) => self.evaluate_mux_node(mux, parent_intrinsic, context),
            GadgetNode::Intrinsic(child) => self.evaluate_intrinsic_node(child, context),
        }
    }

    fn evaluate_mux_node(
        &self,
        mux: &MuxNode,
        parent_intrinsic: &'static str,
        context: &mut EvaluationContext<'_>,
    ) -> Result<BV, SynthesisError> {
        let select_name = mux_select_solver_name(&mux.select);
        context.select_vars.insert(select_name.clone());
        let select = symbolic_bv_named(
            &mux.select,
            select_name,
            context.symbolic_vars,
            context.symbolic_order,
        );
        let Some(last_source_idx) = mux.sources.len().checked_sub(1) else {
            return Err(SynthesisError::ModelMissingValue(format!(
                "mux '{}' has no sources",
                mux.select.name
            )));
        };
        context
            .solver
            .assert(select.bvule(BV::from_u64(last_source_idx as u64, select.get_size())));

        let mut sources = Vec::with_capacity(mux.sources.len());
        for source in &mux.sources {
            sources.push(self.evaluate_graph_node(source, parent_intrinsic, context)?);
        }

        let mut result = sources[last_source_idx].clone();
        for source_idx in (0..last_source_idx).rev() {
            result = select
                .eq(BV::from_u64(source_idx as u64, select.get_size()))
                .ite(&sources[source_idx], &result);
        }
        Ok(result)
    }

    fn constrain_outputs_to_target_pairs(
        &self,
        solver: &Solver,
        top_output: &BV,
        bottom_output: &BV,
        target_pairs: &[TargetPair],
        allow_any_lane_order: bool,
    ) {
        let lane_bits = self.lane_bits();
        let lanes = self.elements_per_vector();
        let mut all_output_lanes = Vec::new();

        for lane_idx in 0..lanes {
            let top_lane = extract_element(top_output, lane_bits, lane_idx);
            let bottom_lane = extract_element(bottom_output, lane_bits, lane_idx);
            all_output_lanes.push(top_lane.clone());
            all_output_lanes.push(bottom_lane.clone());

            if allow_any_lane_order {
                let pair_options: Vec<Bool> = target_pairs
                    .iter()
                    .map(|(elem_a, elem_b)| {
                        let val_a = BV::from_u64(u64::from(*elem_a), lane_bits);
                        let val_b = BV::from_u64(u64::from(*elem_b), lane_bits);
                        let ordered = Bool::and(&[top_lane.eq(&val_a), bottom_lane.eq(&val_b)]);
                        let swapped = Bool::and(&[top_lane.eq(&val_b), bottom_lane.eq(&val_a)]);
                        Bool::or(&[ordered, swapped])
                    })
                    .collect();
                solver.assert(Bool::or(&pair_options));
            } else {
                let (elem_a, elem_b) = target_pairs[lane_idx];
                solver.assert(top_lane.eq(BV::from_u64(u64::from(elem_a), lane_bits)));
                solver.assert(bottom_lane.eq(BV::from_u64(u64::from(elem_b), lane_bits)));
            }
        }

        solver.assert(BV::distinct(&all_output_lanes));
    }

    fn extract_canonical_output_state(
        &self,
        model: &z3::Model,
        top_output: &BV,
        bottom_output: &BV,
    ) -> Result<VectorState, SynthesisError> {
        let lane_bits = self.lane_bits();
        let lanes = self.elements_per_vector();
        let mut top = Vec::with_capacity(lanes);
        let mut bottom = Vec::with_capacity(lanes);

        for lane_idx in 0..lanes {
            let top_value = model_bv_as_lane_label(
                model,
                &extract_element(top_output, lane_bits, lane_idx),
                format!("top_output[{lane_idx}]"),
            )?;
            let bottom_value = model_bv_as_lane_label(
                model,
                &extract_element(bottom_output, lane_bits, lane_idx),
                format!("bottom_output[{lane_idx}]"),
            )?;
            top.push(top_value.min(bottom_value));
            bottom.push(top_value.max(bottom_value));
        }

        Ok(VectorState::new(top, bottom))
    }

    fn output_state_from_model_or_target_pairs(
        &self,
        model: &z3::Model,
        top_output: &BV,
        bottom_output: &BV,
        target_pairs: &[TargetPair],
        allow_any_lane_order: bool,
    ) -> Result<VectorState, SynthesisError> {
        if allow_any_lane_order {
            self.extract_canonical_output_state(model, top_output, bottom_output)
        } else {
            Ok(VectorState::new(
                target_pairs.iter().map(|(top, _)| *top).collect(),
                target_pairs.iter().map(|(_, bottom)| *bottom).collect(),
            ))
        }
    }

    fn concretize_root(
        &self,
        root: Option<&IntrinsicNode>,
        model: &z3::Model,
        symbolic_vars: &HashMap<String, BV>,
    ) -> Result<Vec<InstructionSpec>, SynthesisError> {
        let Some(root) = root else {
            return Ok(Vec::new());
        };

        let order = intrinsic_topological_post_order(root);
        let node_to_idx = order
            .iter()
            .enumerate()
            .map(|(idx, node)| (intrinsic_node_key(node), idx))
            .collect::<HashMap<_, _>>();

        let mut instructions = Vec::with_capacity(order.len());
        for (inst_idx, node) in order.iter().enumerate() {
            let mut args = BTreeMap::new();
            for binding in node.operands.iter() {
                args.insert(
                    binding.key.as_str(),
                    self.concretize_operand(
                        &binding.node,
                        inst_idx,
                        model,
                        symbolic_vars,
                        &node_to_idx,
                    )?,
                );
            }
            instructions.push(InstructionSpec::new(node.name, args));
        }

        Ok(instructions)
    }

    fn concretize_operand(
        &self,
        operand: &GadgetNode,
        inst_idx: usize,
        model: &z3::Model,
        symbolic_vars: &HashMap<String, BV>,
        node_to_idx: &HashMap<String, usize>,
    ) -> Result<InstructionArg, SynthesisError> {
        match operand {
            GadgetNode::Input(input) => Ok(InstructionArg::Input(input.name.to_owned())),
            GadgetNode::Symbolic(symbolic) => {
                let bv = symbolic_vars
                    .get(&symbolic.name)
                    .ok_or_else(|| SynthesisError::ModelMissingValue(symbolic.name.clone()))?;
                if symbolic.bit_width > 64 {
                    Ok(InstructionArg::BitVec {
                        bits: symbolic.bit_width,
                        hex: model_bv_as_hex(model, bv, symbolic.name.clone())?,
                    })
                } else {
                    Ok(InstructionArg::U64(model_bv_as_u64(
                        model,
                        bv,
                        symbolic.name.clone(),
                    )?))
                }
            }
            GadgetNode::Mux(mux) => {
                let select_name = mux_select_solver_name(&mux.select);
                let select = symbolic_vars
                    .get(&select_name)
                    .ok_or_else(|| SynthesisError::ModelMissingValue(select_name.clone()))?;
                let select_value = model_bv_as_u64(model, select, select_name)?;
                Ok(InstructionArg::Input(select_to_source(
                    select_value,
                    inst_idx,
                )))
            }
            GadgetNode::Intrinsic(child) => {
                let ref_idx = node_to_idx.get(&intrinsic_node_key(child)).ok_or_else(|| {
                    SynthesisError::ModelMissingValue(format!(
                        "instruction reference for {}",
                        child.name
                    ))
                })?;
                let source = if *ref_idx + 1 == inst_idx {
                    "prev".to_owned()
                } else {
                    format!("result_{ref_idx}")
                };
                Ok(InstructionArg::Input(source))
            }
        }
    }
}

fn reject_unsupported_gadget_depth(gadget_depth: u8) -> Result<(), SynthesisError> {
    if gadget_depth > MAX_SUPPORTED_GADGET_DEPTH {
        return Err(SynthesisError::UnsupportedGadgetDepth {
            depth: gadget_depth,
            max_supported: MAX_SUPPORTED_GADGET_DEPTH,
        });
    }
    Ok(())
}

fn model_bv_as_u64(model: &z3::Model, bv: &BV, name: String) -> Result<u64, SynthesisError> {
    model
        .eval(bv, true)
        .and_then(|value| value.as_u64())
        .ok_or(SynthesisError::ModelMissingValue(name))
}

fn model_bv_as_lane_label(
    model: &z3::Model,
    bv: &BV,
    name: String,
) -> Result<LaneLabel, SynthesisError> {
    let value = model_bv_as_u64(model, bv, name.clone())?;
    LaneLabel::try_from(value)
        .map_err(|_| SynthesisError::ModelMissingValue(format!("{name}={value} exceeds u8")))
}

fn model_bv_as_hex(model: &z3::Model, bv: &BV, name: String) -> Result<String, SynthesisError> {
    let bits = bv.get_size();
    let chunk_count = bits.div_ceil(64);
    let mut chunks = Vec::with_capacity(chunk_count as usize);

    for chunk_idx in 0..chunk_count {
        let low = chunk_idx * 64;
        let high = (low + 63).min(bits - 1);
        let chunk_bits = high - low + 1;
        let chunk = bv.extract(high, low);
        let value = model_bv_as_u64(model, &chunk, format!("{name}[{high}:{low}]"))?;
        chunks.push((value, chunk_bits));
    }

    let mut hex = String::new();
    for (value, chunk_bits) in chunks.into_iter().rev() {
        let hex_digits = chunk_bits.div_ceil(4) as usize;
        hex.push_str(&format!("{value:0hex_digits$x}"));
    }
    Ok(hex)
}

fn symbolic_bv(
    symbolic: &Symbolic,
    symbolic_vars: &mut HashMap<String, BV>,
    symbolic_order: &mut Vec<String>,
) -> BV {
    symbolic_bv_named(
        symbolic,
        symbolic.name.clone(),
        symbolic_vars,
        symbolic_order,
    )
}

fn symbolic_bv_named(
    symbolic: &Symbolic,
    solver_name: String,
    symbolic_vars: &mut HashMap<String, BV>,
    symbolic_order: &mut Vec<String>,
) -> BV {
    symbolic_vars
        .entry(solver_name.clone())
        .or_insert_with(|| {
            symbolic_order.push(solver_name.clone());
            BV::new_const(solver_name, symbolic.bit_width)
        })
        .clone()
}

fn mux_select_solver_name(symbolic: &Symbolic) -> String {
    for prefix in ["sel_top_", "sel_bottom_"] {
        if let Some(suffix) = symbolic.name.strip_prefix(prefix) {
            return format!("sel_{suffix}");
        }
    }
    symbolic.name.clone()
}

fn apply_mux_pruning(
    node: &IntrinsicNode,
    symbolic_vars: &HashMap<String, BV>,
    solver: &Solver,
) -> Result<(), SynthesisError> {
    let mut mux_selects = Vec::new();
    for binding in node.operands.iter() {
        if let GadgetNode::Mux(mux) = &binding.node
            && let Some(pruning) = &mux.pruning
        {
            let select_name = mux_select_solver_name(&mux.select);
            let select = symbolic_vars
                .get(&select_name)
                .ok_or_else(|| SynthesisError::ModelMissingValue(select_name.clone()))?;
            mux_selects.push((select.clone(), mux.sources.len() - 1, pruning.clone()));
        }
    }

    if mux_selects.is_empty() {
        return Ok(());
    }

    let mut at_least_one_last = Vec::new();
    for (select, last_source_idx, pruning) in mux_selects {
        match pruning {
            MuxPruning::ForceLast => {
                solver.assert(select.eq(BV::from_u64(last_source_idx as u64, select.get_size())))
            }
            MuxPruning::AtLeastOneLast => at_least_one_last.push((select, last_source_idx)),
        }
    }

    if at_least_one_last.len() == 1 {
        let (select, last_source_idx) = &at_least_one_last[0];
        solver.assert(select.eq(BV::from_u64(*last_source_idx as u64, select.get_size())));
    } else if at_least_one_last.len() >= 2 {
        let constraints = at_least_one_last
            .iter()
            .map(|(select, last_source_idx)| {
                BV::from_u64(*last_source_idx as u64, select.get_size()).bvule(select)
            })
            .collect::<Vec<_>>();
        solver.assert(Bool::or(&constraints));
    }

    Ok(())
}

fn select_to_source(select_value: u64, inst_idx: usize) -> String {
    match select_value {
        0 => "top".to_owned(),
        1 => "bottom".to_owned(),
        _ => {
            let result_idx = (select_value - 2) as usize;
            if inst_idx > 0 && result_idx == inst_idx - 1 {
                "prev".to_owned()
            } else {
                format!("result_{result_idx}")
            }
        }
    }
}

fn intrinsic_node_key(node: &IntrinsicNode) -> String {
    let mut parts = vec![
        "intrinsic".to_owned(),
        node.name.to_owned(),
        node.isomorphic_order.to_string(),
    ];
    for binding in node.operands.iter() {
        parts.push(binding.key.as_str().to_owned());
        parts.push(operand_node_key(&binding.node));
    }
    parts.join("|")
}

fn operand_node_key(operand: &GadgetNode) -> String {
    match operand {
        GadgetNode::Input(input) => format!("input:{}", input.name),
        GadgetNode::Symbolic(symbolic) => {
            format!("symbolic:{}:{}", symbolic.name, symbolic.bit_width)
        }
        GadgetNode::Mux(mux) => {
            let pruning = mux
                .pruning
                .as_ref()
                .map(MuxPruning::as_str)
                .unwrap_or("none");
            let sources = mux
                .sources
                .iter()
                .map(operand_node_key)
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "mux:{}:{}:{}:[{}]",
                mux.select.name, mux.select.bit_width, pruning, sources
            )
        }
        GadgetNode::Intrinsic(child) => intrinsic_node_key(child),
    }
}

fn intrinsic_topological_post_order(root: &IntrinsicNode) -> Vec<&IntrinsicNode> {
    let mut visited = BTreeSet::new();
    let mut order = Vec::new();
    visit_intrinsic_post_order(root, &mut visited, &mut order);
    order
}

fn visit_intrinsic_post_order<'a>(
    node: &'a IntrinsicNode,
    visited: &mut BTreeSet<String>,
    order: &mut Vec<&'a IntrinsicNode>,
) {
    if !visited.insert(intrinsic_node_key(node)) {
        return;
    }
    for binding in node.operands.iter() {
        visit_operand_intrinsics(&binding.node, visited, order);
    }
    order.push(node);
}

fn visit_operand_intrinsics<'a>(
    operand: &'a GadgetNode,
    visited: &mut BTreeSet<String>,
    order: &mut Vec<&'a IntrinsicNode>,
) {
    match operand {
        GadgetNode::Intrinsic(child) => visit_intrinsic_post_order(child, visited, order),
        GadgetNode::Mux(mux) => {
            for source in &mux.sources {
                visit_operand_intrinsics(source, visited, order);
            }
        }
        GadgetNode::Input(_) | GadgetNode::Symbolic(_) => {}
    }
}

fn graph_max_depth(graph: &GadgetGraph) -> u8 {
    intrinsic_depth(graph.top.as_ref()).max(intrinsic_depth(graph.bottom.as_ref()))
}

fn intrinsic_depth(node: Option<&IntrinsicNode>) -> u8 {
    let Some(node) = node else {
        return 0;
    };
    let max_child_depth = node
        .operands
        .iter()
        .map(|binding| operand_intrinsic_depth(&binding.node))
        .max()
        .unwrap_or(0);
    1 + max_child_depth
}

fn operand_intrinsic_depth(operand: &GadgetNode) -> u8 {
    match operand {
        GadgetNode::Intrinsic(child) => intrinsic_depth(Some(child)),
        GadgetNode::Mux(mux) => mux
            .sources
            .iter()
            .map(operand_intrinsic_depth)
            .max()
            .unwrap_or(0),
        GadgetNode::Input(_) | GadgetNode::Symbolic(_) => 0,
    }
}

fn wire_hardwired(inst0: &IntrinsicNode, inst1: &IntrinsicNode) -> IntrinsicNode {
    IntrinsicNode {
        name: inst1.name,
        operands: inst1
            .operands
            .iter()
            .map(|binding| {
                let wired_operand = if matches!(&binding.node, GadgetNode::Input(_)) {
                    GadgetNode::Intrinsic(Box::new(inst0.clone()))
                } else {
                    binding.node.clone()
                };
                OperandBinding::new(binding.key, wired_operand)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        isomorphic_order: true,
    }
}

fn wire_with_muxes(
    inst0: &IntrinsicNode,
    inst1: &IntrinsicNode,
    input_name: &'static str,
) -> IntrinsicNode {
    let top = GadgetNode::Input(InputRef { name: "top" });
    let bottom = GadgetNode::Input(InputRef { name: "bottom" });
    let inst0_output = GadgetNode::Intrinsic(Box::new(inst0.clone()));

    IntrinsicNode {
        name: inst1.name,
        operands: inst1
            .operands
            .iter()
            .map(|binding| {
                let wired_operand = if matches!(
                    &binding.node,
                    GadgetNode::Input(_) | GadgetNode::Intrinsic(_)
                ) {
                    GadgetNode::Mux(MuxNode {
                        select: Symbolic {
                            name: format!(
                                "sel_{input_name}_{}_{}",
                                binding.key.as_str(),
                                inst1.name
                            ),
                            bit_width: 2,
                        },
                        sources: vec![top.clone(), bottom.clone(), inst0_output.clone()],
                        pruning: Some(MuxPruning::AtLeastOneLast),
                    })
                } else {
                    binding.node.clone()
                };
                OperandBinding::new(binding.key, wired_operand)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        isomorphic_order: true,
    }
}

fn has_direct_symbolic_operand(node: &IntrinsicNode) -> bool {
    node.operands
        .iter()
        .any(|binding| matches!(&binding.node, GadgetNode::Symbolic(_)))
}

fn register_operand_keys(node: &IntrinsicNode) -> Vec<OperandKey> {
    node.operands
        .iter()
        .filter_map(|binding| matches!(&binding.node, GadgetNode::Input(_)).then_some(binding.key))
        .collect()
}

fn build_shared_prefix_tail(
    dual: &IntrinsicNode,
    reg_a: &IntrinsicNode,
    reg_b: &IntrinsicNode,
    suffix: &str,
) -> IntrinsicNode {
    let reg_keys = register_operand_keys(dual);
    assert_eq!(reg_keys.len(), 2);
    let first_reg_key = reg_keys[0];
    let second_reg_key = reg_keys[1];

    IntrinsicNode {
        name: dual.name,
        operands: dual
            .operands
            .iter()
            .map(|binding| {
                let wired_operand = if binding.key == first_reg_key {
                    GadgetNode::Intrinsic(Box::new(reg_a.clone()))
                } else if binding.key == second_reg_key {
                    GadgetNode::Intrinsic(Box::new(reg_b.clone()))
                } else if let GadgetNode::Symbolic(symbolic) = &binding.node {
                    GadgetNode::Symbolic(Symbolic {
                        name: format!("{}_shared_{suffix}", symbolic.name),
                        bit_width: symbolic.bit_width,
                    })
                } else {
                    binding.node.clone()
                };
                OperandBinding::new(binding.key, wired_operand)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        isomorphic_order: dual.isomorphic_order,
    }
}

fn input_ref_operand_count(node: &IntrinsicNode) -> usize {
    register_operand_keys(node).len()
}

fn swap_register_operands(node: &IntrinsicNode) -> IntrinsicNode {
    let mut swapped = node.clone();
    let input_indices: Vec<usize> = swapped
        .operands
        .iter()
        .enumerate()
        .filter_map(|(idx, binding)| matches!(&binding.node, GadgetNode::Input(_)).then_some(idx))
        .collect();
    assert_eq!(input_indices.len(), 2);
    let first_idx = input_indices[0];
    let second_idx = input_indices[1];
    let first_operand = swapped.operands[first_idx].node.clone();
    swapped.operands[first_idx].node = swapped.operands[second_idx].node.clone();
    swapped.operands[second_idx].node = first_operand;
    swapped
}
