use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::permutex2var::{mm512_mask_permutex2var_epi64, mm512_permutex2var_epi64};
use z3_avx::registers::{
    construct_zmm_reg_from_elements, zmm_reg, zmm_reg_reversed, zmm_reg_with_64b_values,
    zmm_reg_with_unique_values,
};

const NULL_PERMUTEX2VAR_VECTOR_EPI64_AVX512: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

fn zmm_reg_pair_with_unique_values(prefix: &str, solver: &Solver, bits: u32) -> (BV, BV) {
    let a = zmm_reg_with_unique_values(&format!("{prefix}_a"), solver, bits);
    let b = zmm_reg_with_unique_values(&format!("{prefix}_b"), solver, bits);
    let lanes = 512 / bits;

    for a_lane in 0..lanes {
        let a_low = a_lane * bits;
        let a_elem = a.extract(a_low + bits - 1, a_low);
        for b_lane in 0..lanes {
            let b_low = b_lane * bits;
            let b_elem = b.extract(b_low + bits - 1, b_low);
            solver.assert(a_elem.eq(&b_elem).not());
        }
    }

    (a, b)
}

fn assert_unsat_eq(output: &BV, expected: &BV, solver: &Solver) {
    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permutex2var_epi64_null_permute_works() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let indices =
        zmm_reg_with_64b_values("indices", &solver, &NULL_PERMUTEX2VAR_VECTOR_EPI64_AVX512);
    let output = mm512_permutex2var_epi64(&a, &indices, &b, &solver);

    assert_unsat_eq(&output, &a, &solver);
}

#[test]
fn test_mm512_permutex2var_epi64_null_permute_found() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let indices = zmm_reg("indices");
    let output = mm512_permutex2var_epi64(&a, &indices, &b, &solver);

    solver.assert(output.eq(&a));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    for lane in 0..8 {
        let low = lane * 64;
        let lane_index = model
            .eval(&indices.extract(low + 3, low), true)
            .expect("lane index in model")
            .as_u64()
            .expect("lane index numeral");
        assert_eq!(lane_index, lane as u64);
    }
}

#[test]
fn test_mm512_permutex2var_epi64_select_from_b() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let select_b_indices: Vec<u64> = (0..8).map(|i| (1 << 3) | i).collect();
    let indices = zmm_reg_with_64b_values("indices", &solver, &select_b_indices);
    let output = mm512_permutex2var_epi64(&a, &indices, &b, &solver);

    assert_unsat_eq(&output, &b, &solver);
}

#[test]
fn test_mm512_permutex2var_epi64_reverse_permute_from_a() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let reverse_a_indices: Vec<u64> = (0..8).map(|i| 7 - i).collect();
    let indices = zmm_reg_with_64b_values("indices", &solver, &reverse_a_indices);
    let output = mm512_permutex2var_epi64(&a, &indices, &b, &solver);
    let reversed_a = zmm_reg_reversed("a_reversed", &solver, &a, 64);

    assert_unsat_eq(&output, &reversed_a, &solver);
}

#[test]
fn test_mm512_permutex2var_epi64_mixed_sources() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let mixed_indices: Vec<u64> = (0..8)
        .map(|i| if i % 2 == 0 { i } else { (1 << 3) | i })
        .collect();
    let indices = zmm_reg_with_64b_values("indices", &solver, &mixed_indices);
    let output = mm512_permutex2var_epi64(&a, &indices, &b, &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..8)
        .map(|i| if i % 2 == 0 { (&a, i) } else { (&b, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(64, &expected_specs);

    assert_unsat_eq(&output, &expected, &solver);
}

#[test]
fn test_mm512_permutex2var_epi64_canonicalizes_dead_index_bits() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let indices = zmm_reg("indices");
    let _output = mm512_permutex2var_epi64(&a, &indices, &b, &solver);

    for lane in 0..8 {
        let low = lane * 64 + 4;
        let high = lane * 64 + 63;
        solver.assert(indices.extract(high, low).eq(BV::from_u64(0, 60)).not());
    }
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutex2var_epi64_mask_all_zeros() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let indices =
        zmm_reg_with_64b_values("indices", &solver, &NULL_PERMUTEX2VAR_VECTOR_EPI64_AVX512);
    let mask = BV::from_u64(0, 8);
    let output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);

    assert_unsat_eq(&output, &a, &solver);
}

#[test]
fn test_mm512_mask_permutex2var_epi64_mask_all_ones() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let indices =
        zmm_reg_with_64b_values("indices", &solver, &NULL_PERMUTEX2VAR_VECTOR_EPI64_AVX512);
    let mask = BV::from_u64(0xFF, 8);
    let masked_output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);
    let unmasked_output = mm512_permutex2var_epi64(&a, &indices, &b, &solver);

    assert_unsat_eq(&masked_output, &unmasked_output, &solver);
}

#[test]
fn test_mm512_mask_permutex2var_epi64_alternating_mask() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let select_b_indices: Vec<u64> = (0..8).map(|i| (1 << 3) | i).collect();
    let indices = zmm_reg_with_64b_values("indices", &solver, &select_b_indices);
    let mask = BV::from_u64(0x55, 8);
    let output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);
    let unmasked = mm512_permutex2var_epi64(&a, &indices, &b, &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..8)
        .map(|i| if i % 2 == 0 { (&unmasked, i) } else { (&a, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(64, &expected_specs);

    assert_unsat_eq(&output, &expected, &solver);
}

#[test]
fn test_mm512_mask_permutex2var_epi64_single_bit_mask() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let repeated_b5 = [(1 << 3) | 5; 8];
    let indices = zmm_reg_with_64b_values("indices", &solver, &repeated_b5);
    let mask = BV::from_u64(1 << 3, 8);
    let output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..8)
        .map(|i| if i == 3 { (&b, 5) } else { (&a, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(64, &expected_specs);

    assert_unsat_eq(&output, &expected, &solver);
}

#[test]
fn test_mm512_mask_permutex2var_epi64_partial_mask() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let reverse_a_indices: Vec<u64> = (0..8).map(|i| 7 - i).collect();
    let indices = zmm_reg_with_64b_values("indices", &solver, &reverse_a_indices);
    let mask = BV::from_u64(0x0F, 8);
    let output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);
    let reversed_a = zmm_reg_reversed("a_reversed", &solver, &a, 64);
    let expected_specs: Vec<(&BV, usize)> = (0..8)
        .map(|i| if i < 4 { (&reversed_a, i) } else { (&a, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(64, &expected_specs);

    assert_unsat_eq(&output, &expected, &solver);
}

#[test]
fn test_mm512_mask_permutex2var_epi64_mixed_sources_with_mask() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let mixed_indices: Vec<u64> = (0..8)
        .map(|i| if i % 2 == 0 { i } else { (1 << 3) | i })
        .collect();
    let indices = zmm_reg_with_64b_values("indices", &solver, &mixed_indices);
    let mask = BV::from_u64(0x55, 8);
    let output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..8).map(|i| (&a, i)).collect();
    let expected = construct_zmm_reg_from_elements(64, &expected_specs);

    assert_unsat_eq(&output, &expected, &solver);
}

#[test]
fn test_mm512_mask_permutex2var_epi64_find_identity_mask() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let repeated_b7 = [(1 << 3) | 7; 8];
    let indices = zmm_reg_with_64b_values("indices", &solver, &repeated_b7);
    let mask = BV::new_const("mask", 8);
    let output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);

    solver.assert(output.eq(&a));
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
fn test_mm512_mask_permutex2var_epi64_find_full_permute_mask() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let select_b_indices: Vec<u64> = (0..8).map(|i| (1 << 3) | i).collect();
    let indices = zmm_reg_with_64b_values("indices", &solver, &select_b_indices);
    let mask = BV::new_const("mask", 8);
    let output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);

    solver.assert(output.eq(&b));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_mask = solver
        .get_model()
        .expect("sat model")
        .eval(&mask, true)
        .expect("mask in model")
        .as_u64()
        .expect("mask numeral");
    assert_eq!(model_mask, 0xFF);
}

#[test]
fn test_mm512_mask_permutex2var_epi64_find_partial_mask() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let select_b_indices: Vec<u64> = (0..8).map(|i| (1 << 3) | i).collect();
    let indices = zmm_reg_with_64b_values("indices", &solver, &select_b_indices);
    let mask = BV::new_const("mask", 8);
    let output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..8)
        .map(|i| if i < 3 { (&b, i) } else { (&a, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(64, &expected_specs);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_mask = solver
        .get_model()
        .expect("sat model")
        .eval(&mask, true)
        .expect("mask in model")
        .as_u64()
        .expect("mask numeral");
    assert_eq!(model_mask, 0x07);
}

#[test]
fn test_mm512_mask_permutex2var_epi64_find_indices_with_mask() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let mask = BV::from_u64(0x55, 8);
    let indices = zmm_reg("indices");
    let output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..8)
        .map(|i| if i % 2 == 0 { (&b, 0) } else { (&a, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(64, &expected_specs);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
    let model = solver.get_model().expect("sat model");
    let pos0_index = model
        .eval(&indices.extract(3, 0), true)
        .expect("position 0 index in model")
        .as_u64()
        .expect("position 0 index numeral");
    assert_eq!(pos0_index, 8);
}

#[test]
fn test_mm512_mask_permutex2var_epi64_cross_source_reverse() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let cross_reverse_indices: Vec<u64> = (0..8)
        .map(|i| {
            let offset = 7 - i;
            let source = if i % 2 == 0 { 1 } else { 0 };
            (source << 3) | offset
        })
        .collect();
    let indices = zmm_reg_with_64b_values("indices", &solver, &cross_reverse_indices);
    let mask = BV::from_u64(0xFF, 8);
    let output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..8)
        .map(|i| {
            let offset = 7 - i;
            if i % 2 == 0 {
                (&b, offset)
            } else {
                (&a, offset)
            }
        })
        .collect();
    let expected = construct_zmm_reg_from_elements(64, &expected_specs);

    assert_unsat_eq(&output, &expected, &solver);
}

#[test]
fn test_mm512_mask_permutex2var_epi64_canonicalizes_dead_index_and_wide_mask_bits() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver, 64);
    let indices = zmm_reg("indices");
    let mask = BV::new_const("wide_mask", 64);
    let _output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);

    solver.assert(mask.extract(63, 8).eq(BV::from_u64(0, 56)).not());
    for lane in 0..8 {
        let low = lane * 64 + 4;
        let high = lane * 64 + 63;
        solver.assert(indices.extract(high, low).eq(BV::from_u64(0, 60)).not());
    }
    assert_eq!(solver.check(), SatResult::Unsat);
}
