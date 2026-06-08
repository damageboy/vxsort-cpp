use gadget_synth::{
    GadgetNode, InputRef, IntrinsicNode, LaneLabel, OperandBinding, OperandKey, TargetPair,
    VectorState,
};

fn input(name: &'static str) -> GadgetNode {
    GadgetNode::Input(InputRef { name })
}

#[test]
fn intrinsic_node_stores_typed_boxed_operand_bindings() {
    let node = IntrinsicNode::new("_mm256_unpacklo_epi64").with_operands([
        OperandBinding::new(OperandKey::A, input("top")),
        OperandBinding::new(OperandKey::B, input("bottom")),
    ]);

    let operands: &[OperandBinding] = node.operands.as_ref();

    assert_eq!(operands.len(), 2);
    assert_eq!(operands[0].key, OperandKey::A);
    assert_eq!(operands[0].key.as_str(), "a");
    assert_eq!(operands[1].key, OperandKey::B);
    assert_eq!(operands[1].key.as_str(), "b");
}

#[test]
fn vector_state_stores_u8_lane_labels() {
    let state = VectorState::new(vec![0_u8, 1], vec![2, 3]);

    let top: &[LaneLabel] = state.top();
    let tuple: (Vec<LaneLabel>, Vec<LaneLabel>) = state.as_tuple();
    let target_pairs: Vec<TargetPair> = vec![(0_u8, 2), (1, 3)];

    assert_eq!(top, &[0, 1]);
    assert_eq!(tuple, (vec![0, 1], vec![2, 3]));
    assert_eq!(target_pairs, vec![(0, 2), (1, 3)]);
}
