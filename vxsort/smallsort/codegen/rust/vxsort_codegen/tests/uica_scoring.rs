use std::collections::BTreeMap;

use gadget_synth::{InstructionSpec, PermutationGadget, VectorState};
use uica_core::analytical::{LatencyGraph, LatencyGraphEdge, compute_maximum_latency_for_graph};
use uica_data::{
    DATAPACK_SCHEMA_VERSION, DataPack, InstructionRecord, MappedUiPackRuntime, PerfRecord,
    encode_uipack,
};
use vxsort_codegen::instruction_stream::{InstructionBlock, ModeledInstruction, Operand, Register};
use vxsort_codegen::scoring::Scorer;
use vxsort_codegen::transition_table::TransitionTable;
use vxsort_codegen::uica_scoring::{UiPackScorer, uops_key_for_instruction};

fn record(iform: &str, string: &str, ports: &[(&str, i32)]) -> InstructionRecord {
    InstructionRecord {
        arch: "SKL".to_owned(),
        iform: iform.to_owned(),
        string: string.to_owned(),
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
        locked: false,
        xml_attrs: BTreeMap::new(),
        imm_zero: false,
        perf: PerfRecord {
            operands: vec![],
            latencies: vec![],
            uops: 1,
            retire_slots: 1,
            uops_mite: 1,
            uops_ms: 0,
            tp: Some(1.0),
            ports: ports
                .iter()
                .map(|(port, count)| ((*port).to_owned(), *count))
                .collect(),
            variants: BTreeMap::new(),
            div_cycles: 0,
            may_be_eliminated: false,
            complex_decoder: false,
            n_available_simple_decoders: 0,
            lcp_stall: false,
            implicit_rsp_change: 0,
            can_be_used_by_lsd: true,
            cannot_be_in_dsb_due_to_jcc_erratum: false,
            no_micro_fusion: false,
            no_macro_fusion: false,
            macro_fusible_with: vec![],
        },
    }
}

fn scorer_with_records(records: Vec<InstructionRecord>) -> UiPackScorer {
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
        instructions: records,
    };
    let runtime = MappedUiPackRuntime::from_bytes(encode_uipack(&pack, "SKL").unwrap()).unwrap();
    UiPackScorer::from_runtime("SKL", runtime).expect("fixture scorer should load")
}

fn block(instructions: Vec<ModeledInstruction>) -> InstructionBlock {
    InstructionBlock {
        label: Some("candidate".to_owned()),
        instructions,
    }
}

fn vpermq_instruction() -> ModeledInstruction {
    ModeledInstruction {
        mnemonic: "vpermq".to_owned(),
        operands: vec![
            Operand::Register(Register::Ymm(0)),
            Operand::Register(Register::Ymm(1)),
            Operand::Immediate(78),
        ],
        comment: None,
    }
}

#[test]
fn uops_key_uses_register_width_kmask_immediate_and_memory_classes() {
    assert_eq!(
        uops_key_for_instruction(&vpermq_instruction()).as_deref(),
        Some("VPERMQ (YMM, YMM, I8)")
    );

    let masked = ModeledInstruction {
        mnemonic: "vpermi2q".to_owned(),
        operands: vec![
            Operand::Register(Register::Zmm(0)),
            Operand::KMask(1),
            Operand::Register(Register::Zmm(2)),
            Operand::Register(Register::Zmm(3)),
        ],
        comment: None,
    };
    assert_eq!(
        uops_key_for_instruction(&masked).as_deref(),
        Some("VPERMI2Q (ZMM, K, ZMM, ZMM)")
    );

    let load = ModeledInstruction {
        mnemonic: "vmovdqa64".to_owned(),
        operands: vec![
            Operand::Register(Register::Ymm(4)),
            Operand::ConstantRef("LC0".to_owned()),
        ],
        comment: None,
    };
    assert_eq!(
        uops_key_for_instruction(&load).as_deref(),
        Some("VMOVDQA (YMM, M256)")
    );
}

#[test]
fn uops_key_covers_compare_swap_instruction_forms() {
    let cmpgtq = ModeledInstruction {
        mnemonic: "vpcmpgtq".to_owned(),
        operands: vec![
            Operand::Register(Register::Ymm(4)),
            Operand::Register(Register::Ymm(0)),
            Operand::Register(Register::Ymm(1)),
        ],
        comment: None,
    };
    assert_eq!(
        uops_key_for_instruction(&cmpgtq).as_deref(),
        Some("VPCMPGTQ (YMM, YMM, YMM)")
    );

    let blendvpd = ModeledInstruction {
        mnemonic: "vblendvpd".to_owned(),
        operands: vec![
            Operand::Register(Register::Ymm(2)),
            Operand::Register(Register::Ymm(1)),
            Operand::Register(Register::Ymm(0)),
            Operand::Register(Register::Ymm(4)),
        ],
        comment: None,
    };
    assert_eq!(
        uops_key_for_instruction(&blendvpd).as_deref(),
        Some("VBLENDVPD (YMM, YMM, YMM, YMM)")
    );

    let minsw = ModeledInstruction {
        mnemonic: "vpminsw".to_owned(),
        operands: vec![
            Operand::Register(Register::Zmm(2)),
            Operand::Register(Register::Zmm(0)),
            Operand::Register(Register::Zmm(1)),
        ],
        comment: None,
    };
    assert_eq!(
        uops_key_for_instruction(&minsw).as_deref(),
        Some("VPMINSW (ZMM, ZMM, ZMM)")
    );

    let maxsq = ModeledInstruction {
        mnemonic: "vpmaxsq".to_owned(),
        operands: vec![
            Operand::Register(Register::Zmm(3)),
            Operand::Register(Register::Zmm(0)),
            Operand::Register(Register::Zmm(1)),
        ],
        comment: None,
    };
    assert_eq!(
        uops_key_for_instruction(&maxsq).as_deref(),
        Some("VPMAXSQ (ZMM, ZMM, ZMM)")
    );
}

#[test]
fn rough_score_uses_uipack_issue_and_port_pressure_and_ignores_ret() {
    let scorer = scorer_with_records(vec![record(
        "VPERMQ_YMMqq_YMMqq_IMMb",
        "VPERMQ (YMM, YMM, I8)",
        &[("5", 1)],
    )]);

    let cost = scorer.rough_score_block(&block(vec![
        vpermq_instruction(),
        ModeledInstruction {
            mnemonic: "ret".to_owned(),
            operands: vec![],
            comment: None,
        },
    ]));

    assert_eq!(cost.instruction_count(), 1);
    assert_eq!(cost.estimated_cycles(), 1.0);
    assert_eq!(cost.score(), 1.0);
}

#[test]
fn k_best_assignment_returns_lowest_concrete_gadget_combinations() {
    let scorer = scorer_with_records(vec![
        record("A", "A (YMM)", &[("0", 1)]),
        record("B", "B (YMM)", &[("0", 3)]),
    ]);
    let mut table = TransitionTable::new(2);
    let s0 = VectorState::new(vec![1], vec![2]);
    let s1 = VectorState::new(vec![3], vec![4]);
    let s2 = VectorState::new(vec![5], vec![6]);
    let cheap =
        PermutationGadget::new(vec![InstructionSpec::new("a", BTreeMap::new())], Vec::new());
    let expensive = PermutationGadget::new(
        vec![
            InstructionSpec::new("a", BTreeMap::new()),
            InstructionSpec::new("a", BTreeMap::new()),
        ],
        Vec::new(),
    );

    table.add_transition(0, &s0, &s1, expensive.clone());
    table.add_transition(0, &s0, &s1, cheap.clone());
    table.add_transition(1, &s1, &s2, expensive);
    table.add_transition(1, &s1, &s2, cheap);
    let path = vec![
        (0, s0.as_tuple(), s1.as_tuple()),
        (1, s1.as_tuple(), s2.as_tuple()),
    ];

    let assigned = scorer.assign_path_gadgets_k_best(&path, &table, 3);
    let selections: Vec<Vec<usize>> = assigned
        .iter()
        .map(|path| {
            path.steps()
                .iter()
                .map(|step| step.gadget_index())
                .collect()
        })
        .collect();

    assert_eq!(selections, vec![vec![1, 1], vec![0, 1], vec![1, 0]]);
}

#[test]
fn full_score_synthetic_decoded_ir_returns_finite_uica_throughput() {
    let scorer = scorer_with_records(vec![record(
        "VPERMQ_YMMqq_YMMqq_IMMb",
        "VPERMQ (YMM, YMM, I8)",
        &[("5", 1)],
    )]);

    let cost = scorer
        .full_score_block(&block(vec![vpermq_instruction()]))
        .expect("uiCA should score synthetic decoded IR");

    assert!(
        cost.score().is_finite(),
        "expected finite cost, got {cost:?}"
    );
    assert!(cost.score() > 0.0);
}

#[test]
fn simulate_block_with_reports_renders_trace_html() {
    let scorer = scorer_with_records(vec![record(
        "VPERMQ_YMMqq_YMMqq_IMMb",
        "VPERMQ (YMM, YMM, I8)",
        &[("5", 1)],
    )]);

    let simulation = scorer
        .simulate_block(&block(vec![vpermq_instruction()]), true)
        .expect("uiCA report simulation should succeed");

    assert!(simulation.throughput_cycles_per_iteration.is_finite());
    assert!(simulation.reports.is_some());
    let html = uica_core::report::render_trace_html(&simulation.reports.unwrap().trace)
        .expect("trace html should render");
    assert!(html.contains("Execution Trace"));
}

#[test]
fn uica_dependency_summary_handles_dense_dependency_graph() {
    let nodes = (0..11).map(|idx| format!("n{idx}")).collect::<Vec<_>>();
    let mut graph = LatencyGraph {
        nodes_for_instr: vec![nodes.clone()],
        edges_for_node: BTreeMap::new(),
    };

    for source in &nodes {
        for target in &nodes {
            if source == target {
                continue;
            }
            graph
                .edges_for_node
                .entry(source.clone())
                .or_default()
                .push(LatencyGraphEdge {
                    source: source.clone(),
                    target: target.clone(),
                    cost: 2,
                    time: 1,
                });
        }
    }

    assert_eq!(
        compute_maximum_latency_for_graph(&graph).max_cycle_ratio,
        2.0
    );
}
