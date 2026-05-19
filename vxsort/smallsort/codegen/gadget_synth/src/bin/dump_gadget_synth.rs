use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::PathBuf;

use gadget_synth::{
    Arch, DType, GadgetGraph, GadgetNode, GadgetSynthesizer, InstructionArg, InstructionSpec,
    IntrinsicNode, Symbolic, SynthesisError, SynthesisOptions, VectorState,
};
use serde_json::{Map, Value, json};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
enum Command {
    Templates {
        arch: Arch,
        dtype: DType,
        gadget_depth: u8,
        intrinsic_filter: Option<BTreeSet<String>>,
        exclude_shared_prefix: bool,
        output: PathBuf,
    },
    Synthesized {
        arch: Arch,
        dtype: DType,
        gadget_depth: u8,
        fixture: String,
        max_unique_outputs: usize,
        intrinsic_filter: Option<BTreeSet<String>>,
        exclude_shared_prefix: bool,
        output: PathBuf,
    },
}

#[derive(Default)]
struct NormalizationContext {
    symbolic_ids: BTreeMap<String, String>,
    symbolic_roles: BTreeMap<String, String>,
}

impl NormalizationContext {
    fn normalize_symbolic(
        &mut self,
        symbolic: &Symbolic,
        operand_key: Option<&str>,
        mux_select: bool,
    ) -> Value {
        if !self.symbolic_ids.contains_key(&symbolic.name) {
            let var_id = format!("v{}", self.symbolic_ids.len());
            let role = symbolic_role(symbolic, operand_key, mux_select);
            self.symbolic_ids.insert(symbolic.name.clone(), var_id);
            self.symbolic_roles.insert(symbolic.name.clone(), role);
        }

        object([
            ("bits", json!(symbolic.bit_width)),
            ("kind", json!("symbolic")),
            (
                "role",
                json!(
                    self.symbolic_roles
                        .get(&symbolic.name)
                        .expect("symbolic role should be assigned")
                ),
            ),
            (
                "var_id",
                json!(
                    self.symbolic_ids
                        .get(&symbolic.name)
                        .expect("symbolic id should be assigned")
                ),
            ),
        ])
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<()> {
    match parse_args(env::args().skip(1))? {
        Command::Templates {
            arch,
            dtype,
            gadget_depth,
            intrinsic_filter,
            exclude_shared_prefix,
            output,
        } => dump_templates(
            arch,
            dtype,
            gadget_depth,
            intrinsic_filter,
            exclude_shared_prefix,
            output,
        )?,
        Command::Synthesized {
            arch,
            dtype,
            gadget_depth,
            fixture,
            max_unique_outputs,
            intrinsic_filter,
            exclude_shared_prefix,
            output,
        } => dump_synthesized(
            arch,
            dtype,
            gadget_depth,
            &fixture,
            max_unique_outputs,
            intrinsic_filter,
            exclude_shared_prefix,
            output,
        )?,
    }
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Command> {
    let mut args = args.into_iter();
    let mode = args.next().ok_or_else(usage)?;
    match mode.as_str() {
        "templates" => parse_templates_args(args),
        "synthesized" => parse_synthesized_args(args),
        _ => Err(format!("unknown mode '{mode}'\n{}", usage()).into()),
    }
}

fn parse_templates_args(args: impl Iterator<Item = String>) -> Result<Command> {
    let mut arch = None;
    let mut dtype = None;
    let mut gadget_depth = None;
    let mut output = None;
    let mut intrinsic_filter = None;
    let mut exclude_shared_prefix = false;

    let mut args = args.peekable();
    while let Some(flag) = args.next() {
        if flag == "--exclude-shared-prefix" {
            exclude_shared_prefix = true;
            continue;
        }

        let value = match flag.as_str() {
            "--arch" | "--vector-machine" | "--dtype" | "--datatype" | "--gadget-depth"
            | "--intrinsic-filter" | "--output" => args
                .next()
                .ok_or_else(|| format!("missing value for {flag}\n{}", usage()))?,
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        };

        match flag.as_str() {
            "--arch" | "--vector-machine" => arch = Some(parse_arch(&value)?),
            "--dtype" | "--datatype" => dtype = Some(parse_dtype(&value)?),
            "--gadget-depth" => gadget_depth = Some(value.parse::<u8>()?),
            "--intrinsic-filter" => intrinsic_filter = parse_intrinsic_filter(&value),
            "--output" => output = Some(PathBuf::from(value)),
            _ => unreachable!("flag was already matched"),
        }
    }

    Ok(Command::Templates {
        arch: arch.ok_or_else(|| format!("missing --arch\n{}", usage()))?,
        dtype: dtype.ok_or_else(|| format!("missing --dtype\n{}", usage()))?,
        gadget_depth: gadget_depth.ok_or_else(|| format!("missing --gadget-depth\n{}", usage()))?,
        intrinsic_filter,
        exclude_shared_prefix,
        output: output.ok_or_else(|| format!("missing --output\n{}", usage()))?,
    })
}

fn parse_synthesized_args(args: impl Iterator<Item = String>) -> Result<Command> {
    let mut arch = None;
    let mut dtype = None;
    let mut gadget_depth = None;
    let mut fixture = None;
    let mut max_unique_outputs = 3;
    let mut output = None;
    let mut intrinsic_filter = None;
    let mut exclude_shared_prefix = false;

    let mut args = args.peekable();
    while let Some(flag) = args.next() {
        if flag == "--exclude-shared-prefix" {
            exclude_shared_prefix = true;
            continue;
        }

        let value = match flag.as_str() {
            "--arch"
            | "--vector-machine"
            | "--dtype"
            | "--datatype"
            | "--gadget-depth"
            | "--fixture"
            | "--max-unique-outputs"
            | "--intrinsic-filter"
            | "--output" => args
                .next()
                .ok_or_else(|| format!("missing value for {flag}\n{}", usage()))?,
            _ => return Err(format!("unknown option '{flag}'\n{}", usage()).into()),
        };

        match flag.as_str() {
            "--arch" | "--vector-machine" => arch = Some(parse_arch(&value)?),
            "--dtype" | "--datatype" => dtype = Some(parse_dtype(&value)?),
            "--gadget-depth" => gadget_depth = Some(value.parse::<u8>()?),
            "--fixture" => fixture = Some(value),
            "--max-unique-outputs" => max_unique_outputs = value.parse::<usize>()?,
            "--intrinsic-filter" => intrinsic_filter = parse_intrinsic_filter(&value),
            "--output" => output = Some(PathBuf::from(value)),
            _ => unreachable!("flag was already matched"),
        }
    }

    Ok(Command::Synthesized {
        arch: arch.ok_or_else(|| format!("missing --arch\n{}", usage()))?,
        dtype: dtype.ok_or_else(|| format!("missing --dtype\n{}", usage()))?,
        gadget_depth: gadget_depth.ok_or_else(|| format!("missing --gadget-depth\n{}", usage()))?,
        fixture: fixture.ok_or_else(|| format!("missing --fixture\n{}", usage()))?,
        max_unique_outputs,
        intrinsic_filter,
        exclude_shared_prefix,
        output: output.ok_or_else(|| format!("missing --output\n{}", usage()))?,
    })
}

fn usage() -> String {
    "usage: dump_gadget_synth <templates|synthesized> --arch avx2 --dtype i64 --gadget-depth 1 [--intrinsic-filter NAME[,NAME...]] --output PATH\n\
     templates options: [--exclude-shared-prefix]\n\
     synthesized options: --fixture identity_pairs [--max-unique-outputs N] [--exclude-shared-prefix]"
        .to_owned()
}

fn parse_arch(value: &str) -> Result<Arch> {
    match value.to_ascii_lowercase().as_str() {
        "avx2" => Ok(Arch::Avx2),
        "avx512" => Ok(Arch::Avx512),
        _ => Err(format!("unsupported arch '{value}'").into()),
    }
}

fn parse_dtype(value: &str) -> Result<DType> {
    match value.to_ascii_lowercase().as_str() {
        "i32" => Ok(DType::I32),
        "i64" => Ok(DType::I64),
        _ => Err(format!("unsupported dtype '{value}'").into()),
    }
}

fn parse_intrinsic_filter(value: &str) -> Option<BTreeSet<String>> {
    let names = value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    (!names.is_empty()).then_some(names)
}

fn dump_templates(
    arch: Arch,
    dtype: DType,
    gadget_depth: u8,
    intrinsic_filter: Option<BTreeSet<String>>,
    exclude_shared_prefix: bool,
    output: PathBuf,
) -> Result<()> {
    let synth = GadgetSynthesizer::new(arch, dtype);
    let mut graphs = if exclude_shared_prefix {
        synth.candidate_graph_templates_excluding_shared_prefix(gadget_depth)?
    } else {
        synth.candidate_graph_templates(gadget_depth)?
    };
    if let Some(filter) = intrinsic_filter.as_ref() {
        validate_intrinsic_filter(filter, &synth, arch, dtype)?;
        graphs.retain(|graph| graph_uses_only_intrinsics(graph, filter));
    }

    let mut records = graphs
        .iter()
        .map(|graph| {
            serde_json::to_string(&normalize_template_record(graph, arch, dtype, gadget_depth))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    records.sort_unstable();

    write_jsonl(&records, output)?;
    Ok(())
}

fn dump_synthesized(
    arch: Arch,
    dtype: DType,
    gadget_depth: u8,
    fixture: &str,
    max_unique_outputs: usize,
    intrinsic_filter: Option<BTreeSet<String>>,
    exclude_shared_prefix: bool,
    output: PathBuf,
) -> Result<()> {
    let synth = GadgetSynthesizer::new(arch, dtype);
    let mut graphs = if exclude_shared_prefix {
        synth.candidate_graph_templates_excluding_shared_prefix(gadget_depth)?
    } else {
        synth.candidate_graph_templates(gadget_depth)?
    };
    if let Some(filter) = intrinsic_filter.as_ref() {
        validate_intrinsic_filter(filter, &synth, arch, dtype)?;
        graphs.retain(|graph| graph_uses_only_intrinsics(graph, filter));
    }

    let (input_state, target_pairs) = fixture_inputs(fixture, arch, dtype)?;
    let options = SynthesisOptions {
        max_unique_outputs,
        allow_any_lane_order: true,
    };

    let mut records = Vec::new();
    for graph in &graphs {
        let results = synth
            .synthesize_graph(graph, &input_state, &target_pairs, options.clone())
            .map_err(|error| synthesize_dump_error(error, intrinsic_filter.is_some()))?;
        for (gadget, output_state) in results {
            records.push(serde_json::to_string(&normalize_synthesized_record(
                &gadget,
                &output_state,
                graph,
                arch,
                dtype,
                gadget_depth,
                fixture,
                &input_state,
                &target_pairs,
            ))?);
        }
    }
    records.sort_unstable();

    write_jsonl(&records, output)?;
    Ok(())
}

fn synthesize_dump_error(
    error: SynthesisError,
    filter_was_provided: bool,
) -> Box<dyn std::error::Error> {
    match error {
        SynthesisError::UnsupportedIntrinsic(name) => {
            let hint = if filter_was_provided {
                "the selected --intrinsic-filter includes an intrinsic without Rust synthesis dispatch yet"
            } else {
                "pass --intrinsic-filter to limit synthesized dumps to intrinsics with Rust synthesis dispatch"
            };
            format!("unsupported intrinsic '{name}' in synthesized mode; {hint}").into()
        }
        other => Box::new(other),
    }
}

fn validate_intrinsic_filter(
    filter: &BTreeSet<String>,
    synth: &GadgetSynthesizer,
    arch: Arch,
    dtype: DType,
) -> Result<()> {
    let available = synth.available_intrinsic_names();
    let unknown = filter
        .iter()
        .filter(|name| !available.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(format!(
            "intrinsics not available for {}/{}: {:?}",
            arch_name(arch),
            dtype_name(dtype),
            unknown
        )
        .into());
    }
    Ok(())
}

fn graph_uses_only_intrinsics(graph: &GadgetGraph, allowed: &BTreeSet<String>) -> bool {
    root_uses_only_intrinsics(graph.top.as_ref(), allowed)
        && root_uses_only_intrinsics(graph.bottom.as_ref(), allowed)
}

fn root_uses_only_intrinsics(node: Option<&IntrinsicNode>, allowed: &BTreeSet<String>) -> bool {
    match node {
        Some(node) => intrinsic_uses_only_intrinsics(node, allowed),
        None => true,
    }
}

fn intrinsic_uses_only_intrinsics(node: &IntrinsicNode, allowed: &BTreeSet<String>) -> bool {
    allowed.contains(node.name)
        && node
            .operands
            .iter()
            .all(|(_, operand)| operand_uses_only_intrinsics(operand, allowed))
}

fn operand_uses_only_intrinsics(node: &GadgetNode, allowed: &BTreeSet<String>) -> bool {
    match node {
        GadgetNode::Input(_) | GadgetNode::Symbolic(_) => true,
        GadgetNode::Intrinsic(child) => intrinsic_uses_only_intrinsics(child, allowed),
        GadgetNode::Mux(mux) => mux
            .sources
            .iter()
            .all(|source| operand_uses_only_intrinsics(source, allowed)),
    }
}

fn fixture_inputs(
    fixture: &str,
    arch: Arch,
    dtype: DType,
) -> Result<(VectorState, Vec<(u64, u64)>)> {
    match fixture {
        "identity_pairs" => Ok(identity_pairs_fixture(arch, dtype)),
        _ => Err(format!("unknown synthesized dump fixture '{fixture}'").into()),
    }
}

fn identity_pairs_fixture(arch: Arch, dtype: DType) -> (VectorState, Vec<(u64, u64)>) {
    let lanes = elements_per_vector(arch, dtype) as u64;
    let top = (0..lanes).collect::<Vec<_>>();
    let bottom = (lanes..lanes * 2).collect::<Vec<_>>();
    let target_pairs = top
        .iter()
        .zip(bottom.iter())
        .map(|(top, bottom)| (*top, *bottom))
        .collect();
    (VectorState::new(top, bottom), target_pairs)
}

fn elements_per_vector(arch: Arch, dtype: DType) -> usize {
    let register_bits = match arch {
        Arch::Avx2 => 256,
        Arch::Avx512 => 512,
    };
    let lane_bits = match dtype {
        DType::I32 => 32,
        DType::I64 => 64,
    };
    register_bits / lane_bits
}

fn write_jsonl(records: &[String], output: PathBuf) -> io::Result<()> {
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(output)?;
    for record in records {
        writeln!(file, "{record}")?;
    }
    Ok(())
}

fn normalize_template_record(
    graph: &GadgetGraph,
    arch: Arch,
    dtype: DType,
    gadget_depth: u8,
) -> Value {
    object([
        ("arch", json!(arch_name(arch))),
        ("dtype", json!(dtype_name(dtype))),
        ("gadget_depth", json!(gadget_depth)),
        ("graph", normalize_graph(graph)),
        ("kind", json!("template")),
        ("tier", json!(graph_tier(graph))),
    ])
}

#[allow(clippy::too_many_arguments)]
fn normalize_synthesized_record(
    gadget: &gadget_synth::PermutationGadget,
    output_state: &VectorState,
    graph: &GadgetGraph,
    arch: Arch,
    dtype: DType,
    gadget_depth: u8,
    fixture: &str,
    input_state: &VectorState,
    target_pairs: &[(u64, u64)],
) -> Value {
    object([
        ("arch", json!(arch_name(arch))),
        (
            "bottom_instructions",
            normalize_instructions(gadget.bottom_instructions(), graph.bottom.as_ref()),
        ),
        ("dtype", json!(dtype_name(dtype))),
        ("fixture", json!(fixture)),
        ("gadget_depth", json!(gadget_depth)),
        ("input_state", normalize_vector_state(input_state)),
        ("kind", json!("synthesized")),
        ("output_state", normalize_vector_state(output_state)),
        ("target_pairs", normalize_target_pairs(target_pairs)),
        (
            "top_instructions",
            normalize_instructions(gadget.top_instructions(), graph.top.as_ref()),
        ),
    ])
}

pub(crate) fn normalize_graph(graph: &GadgetGraph) -> Value {
    let mut context = NormalizationContext::default();
    let top = normalize_root_node(graph.top.as_ref(), &mut context);
    let bottom = normalize_root_node(graph.bottom.as_ref(), &mut context);
    object([("top", top), ("bottom", bottom)])
}

fn normalize_root_node(node: Option<&IntrinsicNode>, context: &mut NormalizationContext) -> Value {
    match node {
        Some(node) => normalize_intrinsic_node(node, context),
        None => Value::Null,
    }
}

fn normalize_intrinsic_node(node: &IntrinsicNode, context: &mut NormalizationContext) -> Value {
    let mut operands = Map::new();
    let mut sorted_operands = node.operands.iter().collect::<Vec<_>>();
    sorted_operands.sort_by_key(|(key, _)| *key);
    for (key, operand) in sorted_operands {
        operands.insert(
            (*key).to_owned(),
            normalize_operand(operand, context, Some(key)),
        );
    }

    object([
        ("isomorphic_order", json!(node.isomorphic_order)),
        ("kind", json!("intrinsic")),
        ("name", json!(node.name)),
        ("operands", Value::Object(operands)),
    ])
}

fn normalize_operand(
    operand: &GadgetNode,
    context: &mut NormalizationContext,
    operand_key: Option<&str>,
) -> Value {
    match operand {
        GadgetNode::Input(input) => object([("kind", json!("input")), ("name", json!(input.name))]),
        GadgetNode::Symbolic(symbolic) => context.normalize_symbolic(symbolic, operand_key, false),
        GadgetNode::Mux(mux) => object([
            ("kind", json!("mux")),
            (
                "pruning",
                mux.pruning
                    .as_ref()
                    .map(|pruning| json!(pruning.as_str()))
                    .unwrap_or(Value::Null),
            ),
            (
                "select",
                context.normalize_symbolic(&mux.select, None, true),
            ),
            (
                "sources",
                Value::Array(
                    mux.sources
                        .iter()
                        .map(|source| normalize_operand(source, context, None))
                        .collect(),
                ),
            ),
        ]),
        GadgetNode::Intrinsic(child) => normalize_intrinsic_node(child, context),
    }
}

fn normalize_vector_state(state: &VectorState) -> Value {
    object([
        ("bottom", json!(state.bottom())),
        ("top", json!(state.top())),
    ])
}

fn normalize_target_pairs(target_pairs: &[(u64, u64)]) -> Value {
    Value::Array(
        target_pairs
            .iter()
            .map(|(left, right)| json!([left, right]))
            .collect(),
    )
}

fn normalize_instructions(instructions: &[InstructionSpec], root: Option<&IntrinsicNode>) -> Value {
    let arg_widths = instruction_arg_bit_widths(root);
    assert_eq!(
        instructions.len(),
        arg_widths.len(),
        "concrete instruction count should match source graph topology"
    );

    Value::Array(
        instructions
            .iter()
            .zip(arg_widths)
            .map(|(instruction, widths)| normalize_instruction(instruction, &widths))
            .collect(),
    )
}

fn instruction_arg_bit_widths(root: Option<&IntrinsicNode>) -> Vec<BTreeMap<&'static str, u32>> {
    let Some(root) = root else {
        return Vec::new();
    };

    intrinsic_topological_post_order(root)
        .into_iter()
        .map(|node| {
            let mut widths = BTreeMap::new();
            for (key, operand) in &node.operands {
                if let GadgetNode::Symbolic(symbolic) = operand {
                    widths.insert(*key, symbolic.bit_width);
                }
            }
            widths
        })
        .collect()
}

fn normalize_instruction(
    instruction: &InstructionSpec,
    arg_bit_widths: &BTreeMap<&'static str, u32>,
) -> Value {
    let mut args = Map::new();
    for (key, value) in instruction.args() {
        args.insert(
            (*key).to_owned(),
            normalize_instruction_arg(value, arg_bit_widths.get(key).copied()),
        );
    }
    object([
        ("args", Value::Object(args)),
        ("name", json!(instruction.intrinsic_name())),
    ])
}

fn normalize_instruction_arg(arg: &InstructionArg, bit_width: Option<u32>) -> Value {
    match arg {
        InstructionArg::Input(name) => json!(name),
        InstructionArg::U64(value) if bit_width.is_some_and(|bits| bits > 64) => {
            normalize_bitvec_hex(
                bit_width.expect("checked above"),
                &format!(
                    "{value:0width$x}",
                    width = bit_width.expect("checked above").div_ceil(4) as usize
                ),
            )
        }
        InstructionArg::U64(value) => json!(value),
        InstructionArg::BitVec { bits, hex } => normalize_bitvec_hex(*bits, hex),
    }
}

fn normalize_bitvec_hex(bits: u32, hex: &str) -> Value {
    object([
        ("bits", json!(bits)),
        ("hex", json!(format!("0x{hex}"))),
        ("kind", json!("bitvec")),
    ])
}

fn symbolic_role(symbolic: &Symbolic, operand_key: Option<&str>, mux_select: bool) -> String {
    if mux_select {
        "mux_select".to_owned()
    } else if operand_key == Some("imm8") || symbolic.name.starts_with("imm8") {
        "imm8".to_owned()
    } else if matches!(operand_key, Some("k" | "mask")) || symbolic.name.starts_with("k_mask") {
        "mask".to_owned()
    } else if operand_key == Some("op_idx") {
        "op_idx".to_owned()
    } else if symbolic.name.starts_with("ctrl") {
        "control_vector".to_owned()
    } else if let Some(operand_key) = operand_key {
        operand_key.to_owned()
    } else {
        "symbolic".to_owned()
    }
}

fn graph_tier(graph: &GadgetGraph) -> &'static str {
    if graph_max_depth(graph) <= 1 {
        "shallow"
    } else {
        "deep"
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
        .map(|(_, operand)| operand_intrinsic_depth(operand))
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

fn intrinsic_node_key(node: &IntrinsicNode) -> String {
    let mut parts = vec![
        "intrinsic".to_owned(),
        node.name.to_owned(),
        node.isomorphic_order.to_string(),
    ];
    for (key, operand) in &node.operands {
        parts.push((*key).to_owned());
        parts.push(operand_node_key(operand));
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
                .map(|pruning| pruning.as_str())
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
    for (_, operand) in &node.operands {
        visit_operand_intrinsics(operand, visited, order);
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

fn arch_name(arch: Arch) -> &'static str {
    match arch {
        Arch::Avx2 => "avx2",
        Arch::Avx512 => "avx512",
    }
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::I32 => "i32",
        DType::I64 => "i64",
    }
}

fn object<const N: usize>(entries: [(&str, Value); N]) -> Value {
    let mut map = Map::new();
    for (key, value) in entries {
        map.insert(key.to_owned(), value);
    }
    Value::Object(map)
}
