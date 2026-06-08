use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::permutexvar::{
    mm256_permutexvar_epi32, mm512_mask_permutexvar_epi32, mm512_permutexvar_epi32,
};
use z3_avx::registers::{
    construct_zmm_reg_from_elements, ymm_reg, ymm_reg_reversed, ymm_reg_with_32b_values,
    ymm_reg_with_unique_values, zmm_reg, zmm_reg_reversed, zmm_reg_with_32b_values,
    zmm_reg_with_unique_values,
};

const NULL_PERMUTE_VECTOR_EPI32_AVX2: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
const NULL_PERMUTE_VECTOR_EPI32_AVX512: [u64; 16] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const REVERSE_PERMUTE_VECTOR_EPI32_AVX2: [u64; 8] = [7, 6, 5, 4, 3, 2, 1, 0];
const REVERSE_PERMUTE_VECTOR_EPI32_AVX512: [u64; 16] =
    [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];

fn assert_model_index_lanes(solver: &Solver, indices: &BV, expected: &[u64], element_bits: u32) {
    let model = solver.get_model().expect("sat model");
    for (lane, expected_lane) in expected.iter().enumerate() {
        let low = lane as u32 * element_bits;
        let value = model
            .eval(&indices.extract(low + element_bits - 1, low), true)
            .expect("index lane in model")
            .as_u64()
            .expect("index lane numeral");
        assert_eq!(value, *expected_lane, "unexpected index lane {lane}");
    }
}

#[test]
fn test_mm256_permutexvar_epi32_null_permute_works() {
    let solver = Solver::new();
    let input = ymm_reg("ymm0");
    let indices = ymm_reg_with_32b_values("indices", &solver, &NULL_PERMUTE_VECTOR_EPI32_AVX2);
    let output = mm256_permutexvar_epi32(&input, &indices, &solver);

    solver.assert(input.eq(output).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permutexvar_epi32_null_permute_found() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 32);
    let indices = ymm_reg("indices_null_256");
    let output = mm256_permutexvar_epi32(&input, &indices, &solver);

    solver.assert(input.eq(output));
    assert_eq!(solver.check(), SatResult::Sat);
    assert_model_index_lanes(&solver, &indices, &NULL_PERMUTE_VECTOR_EPI32_AVX2, 32);
}

#[test]
fn test_mm256_permutexvar_epi32_reverse_permute_found() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 32);
    let indices = ymm_reg("indices_reverse_256");
    let output = mm256_permutexvar_epi32(&input, &indices, &solver);
    let reversed_input = ymm_reg_reversed("ymm_reversed", &solver, &input, 32);

    solver.assert(output.eq(reversed_input));
    assert_eq!(solver.check(), SatResult::Sat);
    assert_model_index_lanes(&solver, &indices, &REVERSE_PERMUTE_VECTOR_EPI32_AVX2, 32);
}

#[test]
fn test_mm512_permutexvar_epi32_null_permute_works() {
    let solver = Solver::new();
    let input = zmm_reg("zmm0");
    let indices = zmm_reg_with_32b_values("indices", &solver, &NULL_PERMUTE_VECTOR_EPI32_AVX512);
    let output = mm512_permutexvar_epi32(&input, &indices, &solver);

    solver.assert(input.eq(output).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permutexvar_epi32_null_permute_found() {
    let solver = Solver::new();
    let input = zmm_reg_with_unique_values("zmm0", &solver, 32);
    let indices = zmm_reg("indices_null_512");
    let output = mm512_permutexvar_epi32(&input, &indices, &solver);

    solver.assert(input.eq(output));
    assert_eq!(solver.check(), SatResult::Sat);
    assert_model_index_lanes(&solver, &indices, &NULL_PERMUTE_VECTOR_EPI32_AVX512, 32);
}

#[test]
fn test_mm512_permutexvar_epi32_reverse_permute_found() {
    let solver = Solver::new();
    let input = zmm_reg_with_unique_values("zmm0", &solver, 32);
    let indices = zmm_reg("indices_reverse_512");
    let output = mm512_permutexvar_epi32(&input, &indices, &solver);
    let reversed_input = zmm_reg_reversed("zmm_reversed", &solver, &input, 32);

    solver.assert(output.eq(reversed_input));
    assert_eq!(solver.check(), SatResult::Sat);
    assert_model_index_lanes(&solver, &indices, &REVERSE_PERMUTE_VECTOR_EPI32_AVX512, 32);
}

#[test]
fn test_mm512_mask_permutexvar_epi32_mask_all_zeros() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let a = zmm_reg_with_unique_values("a", &solver, 32);
    let indices = zmm_reg_with_32b_values("indices", &solver, &NULL_PERMUTE_VECTOR_EPI32_AVX512);
    let mask = BV::from_u64(0, 16);

    let output = mm512_mask_permutexvar_epi32(&src, &mask, &indices, &a, &solver);

    solver.assert(output.eq(src).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutexvar_epi32_mask_all_ones() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let a = zmm_reg_with_unique_values("a", &solver, 32);
    let indices = zmm_reg_with_32b_values("indices", &solver, &NULL_PERMUTE_VECTOR_EPI32_AVX512);
    let mask = BV::from_u64(0xFFFF, 16);

    let masked_output = mm512_mask_permutexvar_epi32(&src, &mask, &indices, &a, &solver);
    let unmasked_output = mm512_permutexvar_epi32(&a, &indices, &solver);

    solver.assert(masked_output.eq(unmasked_output).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutexvar_epi32_alternating_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let a = zmm_reg_with_unique_values("a", &solver, 32);
    let indices = zmm_reg_with_32b_values("indices", &solver, &REVERSE_PERMUTE_VECTOR_EPI32_AVX512);
    let mask = BV::from_u64(0x5555, 16);

    let output = mm512_mask_permutexvar_epi32(&src, &mask, &indices, &a, &solver);
    let unmasked = mm512_permutexvar_epi32(&a, &indices, &solver);
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

#[test]
fn test_mm512_mask_permutexvar_epi32_single_bit_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let a = zmm_reg_with_unique_values("a", &solver, 32);
    let indices = zmm_reg_with_32b_values("indices", &solver, &REVERSE_PERMUTE_VECTOR_EPI32_AVX512);
    let mask = BV::from_u64(1 << 7, 16);

    let output = mm512_mask_permutexvar_epi32(&src, &mask, &indices, &a, &solver);
    let unmasked = mm512_permutexvar_epi32(&a, &indices, &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..16)
        .map(|i| if i == 7 { (&unmasked, i) } else { (&src, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(32, &expected_specs);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutexvar_epi32_partial_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let a = zmm_reg_with_unique_values("a", &solver, 32);
    let indices = zmm_reg_with_32b_values("indices", &solver, &REVERSE_PERMUTE_VECTOR_EPI32_AVX512);
    let mask = BV::from_u64(0x00FF, 16);

    let output = mm512_mask_permutexvar_epi32(&src, &mask, &indices, &a, &solver);
    let reversed_a = zmm_reg_reversed("a_reversed", &solver, &a, 32);
    let expected_specs: Vec<(&BV, usize)> = (0..16)
        .map(|i| if i < 8 { (&reversed_a, i) } else { (&src, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(32, &expected_specs);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutexvar_epi32_find_mask_for_identity() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let a = zmm_reg_with_unique_values("a", &solver, 32);
    let indices = zmm_reg_with_32b_values("indices", &solver, &REVERSE_PERMUTE_VECTOR_EPI32_AVX512);
    let mask = BV::new_const("mask_identity", 16);

    let output = mm512_mask_permutexvar_epi32(&src, &mask, &indices, &a, &solver);

    solver.assert(output.eq(src));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_mask = solver
        .get_model()
        .expect("sat model")
        .eval(&mask, true)
        .expect("mask in model")
        .as_u64()
        .expect("mask numeral");
    assert_eq!(model_mask, 0);
}

#[test]
fn test_mm512_mask_permutexvar_epi32_find_mask_for_full_permute() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let a = zmm_reg_with_unique_values("a", &solver, 32);
    let indices = zmm_reg_with_32b_values("indices", &solver, &NULL_PERMUTE_VECTOR_EPI32_AVX512);
    let mask = BV::new_const("mask_full", 16);

    let output = mm512_mask_permutexvar_epi32(&src, &mask, &indices, &a, &solver);

    solver.assert(output.eq(a));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_mask = solver
        .get_model()
        .expect("sat model")
        .eval(&mask, true)
        .expect("mask in model")
        .as_u64()
        .expect("mask numeral");
    assert_eq!(model_mask, 0xFFFF);
}

#[test]
fn test_mm256_permutexvar_epi32_canonicalizes_dead_index_bits() {
    let solver = Solver::new();
    let a = ymm_reg("a_canonical_256");
    let indices = ymm_reg("indices_canonical_256");

    let _output = mm256_permutexvar_epi32(&a, &indices, &solver);

    solver.assert(indices.extract(31, 3).eq(BV::from_u64(0, 29)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permutexvar_epi32_canonicalizes_dead_index_bits() {
    let solver = Solver::new();
    let a = zmm_reg("a_canonical_512");
    let indices = zmm_reg("indices_canonical_512");

    let _output = mm512_permutexvar_epi32(&a, &indices, &solver);

    solver.assert(indices.extract(31, 4).eq(BV::from_u64(0, 28)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutexvar_epi32_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let src = zmm_reg("src_wide_mask");
    let a = zmm_reg("a_wide_mask");
    let indices = zmm_reg_with_32b_values("indices", &solver, &NULL_PERMUTE_VECTOR_EPI32_AVX512);
    let mask = BV::new_const("wide_kmask", 64);

    let _output = mm512_mask_permutexvar_epi32(&src, &mask, &indices, &a, &solver);

    solver.assert(mask.extract(63, 16).eq(BV::from_u64(0, 48)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}
