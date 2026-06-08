use std::collections::BTreeMap;

use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget};

fn input(name: &str) -> InstructionArg {
    InstructionArg::Input(name.to_owned())
}

fn inst(name: &'static str, args: &[(&'static str, InstructionArg)]) -> InstructionSpec {
    InstructionSpec::new(name, args.iter().cloned().collect::<BTreeMap<_, _>>())
}

#[test]
fn unified_instructions_deduplicates_structurally_identical_top_and_bottom_chains() {
    let top = inst(
        "shuffle",
        &[("a", input("top")), ("imm8", InstructionArg::U64(5))],
    );
    let bottom = inst(
        "shuffle",
        &[("a", input("bottom")), ("imm8", InstructionArg::U64(5))],
    );
    let gadget = PermutationGadget::new(vec![top.clone()], vec![bottom.clone()]);

    let (unified, top_output, bottom_output) = gadget.unified_instructions();

    assert_eq!(unified, vec![top, bottom]);
    assert_eq!(top_output, 0);
    assert_eq!(bottom_output, 1);
    assert_eq!(gadget.instruction_count(), 2);
}

#[test]
fn unified_instructions_resolves_prev_references_when_deduplicating() {
    let first = inst(
        "shuffle",
        &[("a", input("top")), ("imm8", InstructionArg::U64(5))],
    );
    let top_second = inst(
        "permute",
        &[("a", input("prev")), ("imm8", InstructionArg::U64(1))],
    );
    let bottom_second = inst(
        "permute",
        &[("a", input("prev")), ("imm8", InstructionArg::U64(1))],
    );
    let gadget = PermutationGadget::new(
        vec![first.clone(), top_second.clone()],
        vec![first.clone(), bottom_second],
    );

    let (unified, top_output, bottom_output) = gadget.unified_instructions();

    assert_eq!(unified, vec![first, top_second]);
    assert_eq!(top_output, 1);
    assert_eq!(bottom_output, 1);
    assert_eq!(gadget.instruction_count(), 2);
}

#[test]
fn gadget_sort_key_preserves_instruction_order() {
    let first = inst("a", &[]);
    let second = inst("b", &[]);
    let forward = PermutationGadget::new(vec![first.clone(), second.clone()], Vec::new());
    let reversed = PermutationGadget::new(vec![second, first], Vec::new());

    assert_ne!(forward.sort_key(), reversed.sort_key());
}
