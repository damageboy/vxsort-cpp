use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::control::{Imm8, mm_shuffle};
use z3_avx::registers::{
    construct_ymm_reg_from_elements, construct_zmm_reg_from_elements, ymm_reg,
    ymm_reg_with_32b_values, ymm_reg_with_unique_values, zmm_reg, zmm_reg_with_32b_values,
    zmm_reg_with_unique_values,
};
use z3_avx::shuffle_ps::{mm256_shuffle_ps, mm512_mask_shuffle_ps, mm512_shuffle_ps};

const NULL_SHUFFLE_PS_IMM8: u8 = 0xE4;
const NULL_SHUFFLE_PS_2VEC_IMM8: u8 = 0x44;

fn ymm_reg_pair_with_unique_values(prefix: &str, solver: &Solver) -> (BV, BV) {
    let a = ymm_reg_with_32b_values(&format!("{prefix}_a"), solver, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = ymm_reg_with_32b_values(
        &format!("{prefix}_b"),
        solver,
        &[101, 102, 103, 104, 105, 106, 107, 108],
    );
    (a, b)
}

fn zmm_reg_pair_with_unique_values(prefix: &str, solver: &Solver) -> (BV, BV) {
    let a = zmm_reg_with_32b_values(
        &format!("{prefix}_a"),
        solver,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = zmm_reg_with_32b_values(
        &format!("{prefix}_b"),
        solver,
        &[
            101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
        ],
    );
    (a, b)
}

#[test]
fn test_mm256_shuffle_ps_operand_swap_cannot_reproduce_exact_order() {
    // This is the property needed by `isomorphic_order=True`: for a fixed
    // output of inst(a, b, imm), there must be some imm' such that
    // inst(b, a, imm') produces the exact same vector lane order.
    //
    // VSHUFPS does not have that property. With imm=0x44, one 128-bit lane is:
    //
    //   shuffle(a, b, 0x44)  = [a0, a1, b0, b1]
    //   shuffle(b, a, imm')  = [some b, some b, some a, some a]
    //
    // The first output lane of the swapped form must come from b, so it cannot
    // equal a0 for arbitrary/disjoint inputs. Z3 proves no imm' exists.
    let solver = Solver::new();
    let (a, b) = ymm_reg_pair_with_unique_values("shuffle_ps_not_order_isomorphic", &solver);
    let swapped_imm8 = BV::new_const("shuffle_ps_swapped_imm8", 8);

    let output = mm256_shuffle_ps(&a, &b, Imm8::Literal(NULL_SHUFFLE_PS_2VEC_IMM8));
    let swapped_output = mm256_shuffle_ps(&b, &a, Imm8::Expr(swapped_imm8));

    solver.assert(output.eq(swapped_output));
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_shuffle_ps_null_permute_works() {
    let solver = Solver::new();
    let input = ymm_reg("ymm0");
    let output = mm256_shuffle_ps(&input, &input, Imm8::Literal(NULL_SHUFFLE_PS_IMM8));

    solver.assert(output.eq(input).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_shuffle_ps_null_permute_found() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 32);
    let imm8 = BV::new_const("imm8", 8);
    let output = mm256_shuffle_ps(&input, &input, Imm8::Expr(imm8.clone()));

    solver.assert(output.eq(input));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm8 = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8, NULL_SHUFFLE_PS_IMM8 as u64);
}

#[test]
fn test_mm256_shuffle_ps_null_permute_2vec_works() {
    let solver = Solver::new();
    let (op1, op2) = ymm_reg_pair_with_unique_values("op_2vec_works", &solver);
    let output = mm256_shuffle_ps(&op1, &op2, Imm8::Literal(NULL_SHUFFLE_PS_2VEC_IMM8));

    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&op1, 0),
            (&op1, 1),
            (&op2, 0),
            (&op2, 1),
            (&op1, 4),
            (&op1, 5),
            (&op2, 4),
            (&op2, 5),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_shuffle_ps_null_permute_2vec_found() {
    let solver = Solver::new();
    let (op1, op2) = ymm_reg_pair_with_unique_values("op_2vec_found", &solver);
    let imm8 = BV::new_const("imm8_2vec", 8);
    let output = mm256_shuffle_ps(&op1, &op2, Imm8::Expr(imm8.clone()));

    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&op1, 0),
            (&op1, 1),
            (&op2, 0),
            (&op2, 1),
            (&op1, 4),
            (&op1, 5),
            (&op2, 4),
            (&op2, 5),
        ],
    );

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm8 = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8, NULL_SHUFFLE_PS_2VEC_IMM8 as u64);
}

#[test]
fn test_mm512_shuffle_ps_null_permute_works() {
    let solver = Solver::new();
    let input = zmm_reg("zmm0");
    let output = mm512_shuffle_ps(&input, &input, Imm8::Literal(NULL_SHUFFLE_PS_2VEC_IMM8));

    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&input, 0),
            (&input, 1),
            (&input, 0),
            (&input, 1),
            (&input, 4),
            (&input, 5),
            (&input, 4),
            (&input, 5),
            (&input, 8),
            (&input, 9),
            (&input, 8),
            (&input, 9),
            (&input, 12),
            (&input, 13),
            (&input, 12),
            (&input, 13),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_shuffle_ps_null_permute_found() {
    let solver = Solver::new();
    let input = zmm_reg_with_unique_values("zmm0", &solver, 32);
    let imm8 = BV::new_const("imm8_512", 8);
    let output = mm512_shuffle_ps(&input, &input, Imm8::Expr(imm8.clone()));

    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&input, 0),
            (&input, 1),
            (&input, 0),
            (&input, 1),
            (&input, 4),
            (&input, 5),
            (&input, 4),
            (&input, 5),
            (&input, 8),
            (&input, 9),
            (&input, 8),
            (&input, 9),
            (&input, 12),
            (&input, 13),
            (&input, 12),
            (&input, 13),
        ],
    );

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm8 = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8, NULL_SHUFFLE_PS_2VEC_IMM8 as u64);
}

#[test]
fn test_mm512_shuffle_ps_null_permute_2vec_works() {
    let solver = Solver::new();
    let (op1, op2) = zmm_reg_pair_with_unique_values("op_512_2vec_works", &solver);
    let output = mm512_shuffle_ps(&op1, &op2, Imm8::Literal(NULL_SHUFFLE_PS_2VEC_IMM8));

    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&op1, 0),
            (&op1, 1),
            (&op2, 0),
            (&op2, 1),
            (&op1, 4),
            (&op1, 5),
            (&op2, 4),
            (&op2, 5),
            (&op1, 8),
            (&op1, 9),
            (&op2, 8),
            (&op2, 9),
            (&op1, 12),
            (&op1, 13),
            (&op2, 12),
            (&op2, 13),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_shuffle_ps_null_permute_2vec_found() {
    let solver = Solver::new();
    let (op1, op2) = zmm_reg_pair_with_unique_values("op_512_2vec_found", &solver);
    let imm8 = BV::new_const("imm8_512_2vec", 8);
    let output = mm512_shuffle_ps(&op1, &op2, Imm8::Expr(imm8.clone()));

    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&op1, 0),
            (&op1, 1),
            (&op2, 0),
            (&op2, 1),
            (&op1, 4),
            (&op1, 5),
            (&op2, 4),
            (&op2, 5),
            (&op1, 8),
            (&op1, 9),
            (&op2, 8),
            (&op2, 9),
            (&op1, 12),
            (&op1, 13),
            (&op2, 12),
            (&op2, 13),
        ],
    );

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm8 = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8, NULL_SHUFFLE_PS_2VEC_IMM8 as u64);
}

#[test]
fn test_mm256_shuffle_ps_bitonic_stage_masks() {
    let solver = Solver::new();
    let (op1, op2) = ymm_reg_pair_with_unique_values("bitonic_first", &solver);
    let imm8_1 = BV::new_const("imm8_1", 8);
    let out1 = mm256_shuffle_ps(&op1, &op2, Imm8::Expr(imm8_1.clone()));
    let exp1 = construct_ymm_reg_from_elements(
        32,
        &[
            (&op1, 0),
            (&op1, 2),
            (&op2, 0),
            (&op2, 2),
            (&op1, 4),
            (&op1, 6),
            (&op2, 4),
            (&op2, 6),
        ],
    );

    solver.assert(out1.eq(exp1));
    assert_eq!(solver.check(), SatResult::Sat);
    let model = solver.get_model().expect("sat model");
    let model_imm8_1 = model
        .eval(&imm8_1, true)
        .expect("imm8_1 in model")
        .as_u64()
        .expect("imm8_1 numeral");
    assert_eq!(model_imm8_1, mm_shuffle(2, 0, 2, 0) as u64);

    let solver = Solver::new();
    let (op1, op2) = ymm_reg_pair_with_unique_values("bitonic_second", &solver);
    let imm8_2 = BV::new_const("imm8_2", 8);
    let out2 = mm256_shuffle_ps(&op1, &op2, Imm8::Expr(imm8_2.clone()));
    let exp2 = construct_ymm_reg_from_elements(
        32,
        &[
            (&op1, 1),
            (&op1, 3),
            (&op2, 1),
            (&op2, 3),
            (&op1, 5),
            (&op1, 7),
            (&op2, 5),
            (&op2, 7),
        ],
    );

    solver.assert(out2.eq(exp2));
    assert_eq!(solver.check(), SatResult::Sat);
    let model = solver.get_model().expect("sat model");
    let model_imm8_2 = model
        .eval(&imm8_2, true)
        .expect("imm8_2 in model")
        .as_u64()
        .expect("imm8_2 numeral");
    assert_eq!(model_imm8_2, mm_shuffle(3, 1, 3, 1) as u64);
}

#[test]
fn test_mm256_shuffle_ps_bitonic_stage_masks_literal() {
    let solver = Solver::new();
    let op1 = ymm_reg_with_32b_values("bitonic_op1_a", &solver, &[1, 2, 5, 6, 9, 10, 13, 14]);
    let op2 = ymm_reg_with_32b_values("bitonic_op2_a", &solver, &[4, 3, 8, 7, 12, 11, 16, 15]);
    let imm8_1 = BV::new_const("literal_imm8_1", 8);
    let out1 = mm256_shuffle_ps(&op1, &op2, Imm8::Expr(imm8_1.clone()));
    let exp1 = ymm_reg_with_32b_values("bitonic_exp1", &solver, &[1, 5, 4, 8, 9, 13, 12, 16]);

    solver.assert(out1.eq(exp1));
    assert_eq!(solver.check(), SatResult::Sat);
    let model = solver.get_model().expect("sat model");
    let model_imm8_1 = model
        .eval(&imm8_1, true)
        .expect("imm8_1 in model")
        .as_u64()
        .expect("imm8_1 numeral");
    assert_eq!(model_imm8_1, mm_shuffle(2, 0, 2, 0) as u64);

    let solver = Solver::new();
    let op1 = ymm_reg_with_32b_values("bitonic_op1_b", &solver, &[1, 2, 5, 6, 9, 10, 13, 14]);
    let op2 = ymm_reg_with_32b_values("bitonic_op2_b", &solver, &[4, 3, 8, 7, 12, 11, 16, 15]);
    let imm8_2 = BV::new_const("literal_imm8_2", 8);
    let out2 = mm256_shuffle_ps(&op1, &op2, Imm8::Expr(imm8_2.clone()));
    let exp2 = ymm_reg_with_32b_values("bitonic_exp2", &solver, &[2, 6, 3, 7, 10, 14, 11, 15]);

    solver.assert(out2.eq(exp2));
    assert_eq!(solver.check(), SatResult::Sat);
    let model = solver.get_model().expect("sat model");
    let model_imm8_2 = model
        .eval(&imm8_2, true)
        .expect("imm8_2 in model")
        .as_u64()
        .expect("imm8_2 numeral");
    assert_eq!(model_imm8_2, mm_shuffle(3, 1, 3, 1) as u64);
}

#[test]
fn test_mm512_mask_shuffle_ps_mask_all_zeros() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_zero_mask", &solver, 32);
    let (a, b) = zmm_reg_pair_with_unique_values("input_zero_mask", &solver);
    let mask = BV::from_u64(0, 16);

    let output = mm512_mask_shuffle_ps(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_PS_2VEC_IMM8),
        &solver,
    );

    solver.assert(output.eq(src).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_shuffle_ps_mask_all_ones() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_one_mask", &solver, 32);
    let (a, b) = zmm_reg_pair_with_unique_values("input_one_mask", &solver);
    let mask = BV::from_u64(0xFFFF, 16);

    let masked = mm512_mask_shuffle_ps(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_PS_2VEC_IMM8),
        &solver,
    );
    let unmasked = mm512_shuffle_ps(&a, &b, Imm8::Literal(NULL_SHUFFLE_PS_2VEC_IMM8));

    solver.assert(masked.eq(unmasked).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_shuffle_ps_partial_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_partial_mask", &solver, 32);
    let (a, b) = zmm_reg_pair_with_unique_values("input_partial_mask", &solver);
    let mask = BV::from_u64(0x00FF, 16);

    let output = mm512_mask_shuffle_ps(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_PS_2VEC_IMM8),
        &solver,
    );
    let unmasked = mm512_shuffle_ps(&a, &b, Imm8::Literal(NULL_SHUFFLE_PS_2VEC_IMM8));
    let expected_specs: Vec<(&BV, usize)> = (0..16)
        .map(|i| if i < 8 { (&unmasked, i) } else { (&src, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(32, &expected_specs);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_shuffle_ps_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_wide_mask", &solver, 32);
    let (a, b) = zmm_reg_pair_with_unique_values("input_wide_mask", &solver);
    let mask = BV::new_const("k_wide", 64);

    let _output = mm512_mask_shuffle_ps(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_PS_2VEC_IMM8),
        &solver,
    );

    solver.assert(mask.extract(63, 16).eq(BV::from_u64(0, 48)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}
