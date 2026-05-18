use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::control::{Imm8, mm_shuffle};
use z3_avx::permute::{mm256_permute_ps, mm512_mask_permute_ps, mm512_permute_ps};
use z3_avx::registers::{
    construct_zmm_reg_from_elements, ymm_reg, ymm_reg_with_unique_values, zmm_reg,
    zmm_reg_with_unique_values,
};

#[test]
fn mm256_permute_ps_literal_identity() {
    let solver = Solver::new();
    let input = ymm_reg("ymm0");
    let output = mm256_permute_ps(&input, Imm8::Literal(mm_shuffle(3, 2, 1, 0)));

    solver.assert(input.eq(output).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn mm512_permute_ps_literal_identity() {
    let solver = Solver::new();
    let input = zmm_reg("zmm0");
    let output = mm512_permute_ps(&input, Imm8::Literal(mm_shuffle(3, 2, 1, 0)));

    solver.assert(input.eq(output).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn mm256_permute_ps_symbolic_identity_finds_identity_imm8() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm_unique", &solver, 32);
    let imm8 = BV::new_const("imm8_256", 8);
    let output = mm256_permute_ps(&input, Imm8::Expr(imm8.clone()));

    solver.assert(input.eq(output));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm, mm_shuffle(3, 2, 1, 0) as u64);
}

#[test]
fn mm512_permute_ps_symbolic_identity_finds_identity_imm8() {
    let solver = Solver::new();
    let input = zmm_reg_with_unique_values("zmm_unique", &solver, 32);
    let imm8 = BV::new_const("imm8_512", 8);
    let output = mm512_permute_ps(&input, Imm8::Expr(imm8.clone()));

    solver.assert(input.eq(output));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm, mm_shuffle(3, 2, 1, 0) as u64);
}

#[test]
fn mm512_mask_permute_ps_all_zero_mask_preserves_src() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_zero_mask", &solver, 32);
    let a = zmm_reg_with_unique_values("a_zero_mask", &solver, 32);
    let mask = BV::from_u64(0, 16);

    let output = mm512_mask_permute_ps(
        &src,
        &mask,
        &a,
        Imm8::Literal(mm_shuffle(3, 2, 1, 0)),
        &solver,
    );

    solver.assert(output.eq(src).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn mm512_mask_permute_ps_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_wide_mask", &solver, 32);
    let a = zmm_reg_with_unique_values("a_wide_mask", &solver, 32);
    let mask = BV::new_const("k_wide", 64);

    let _output = mm512_mask_permute_ps(
        &src,
        &mask,
        &a,
        Imm8::Literal(mm_shuffle(3, 2, 1, 0)),
        &solver,
    );

    solver.assert(mask.extract(63, 16).eq(BV::from_u64(0, 48)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn mm512_mask_permute_ps_all_one_mask_matches_unmasked() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_one_mask", &solver, 32);
    let a = zmm_reg_with_unique_values("a_one_mask", &solver, 32);
    let mask = BV::from_u64(0xFFFF, 16);
    let imm = mm_shuffle(3, 2, 1, 0);

    let masked = mm512_mask_permute_ps(&src, &mask, &a, Imm8::Literal(imm), &solver);
    let unmasked = mm512_permute_ps(&a, Imm8::Literal(imm));

    solver.assert(masked.eq(unmasked).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn mm512_mask_permute_ps_alternating_mask_selects_by_element() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_alt_mask", &solver, 32);
    let a = zmm_reg_with_unique_values("a_alt_mask", &solver, 32);
    let mask = BV::from_u64(0x5555, 16);
    let imm = mm_shuffle(0, 1, 2, 3);

    let output = mm512_mask_permute_ps(&src, &mask, &a, Imm8::Literal(imm), &solver);
    let unmasked = mm512_permute_ps(&a, Imm8::Literal(imm));

    let expected_specs: Vec<(&BV, usize)> = (0..16)
        .map(|i| {
            if i % 2 == 0 {
                (&unmasked, i)
            } else {
                (&src, i)
            }
        })
        .collect();
    let expected = construct_zmm_reg_from_elements(32, &expected_specs);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}
