use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::registers::{
    construct_ymm_reg_from_elements, construct_zmm_reg_from_elements, ymm_reg,
    ymm_reg_with_32b_values, ymm_reg_with_unique_values, zmm_reg_with_32b_values,
    zmm_reg_with_unique_values,
};
use z3_avx::unpack::{
    mm256_unpackhi_epi32, mm256_unpacklo_epi32, mm512_mask_unpackhi_epi32,
    mm512_mask_unpacklo_epi32, mm512_unpackhi_epi32, mm512_unpacklo_epi32,
};

fn zmm_reg_pair_with_unique_values(prefix: &str, solver: &Solver) -> (BV, BV) {
    (
        zmm_reg_with_unique_values(&format!("{prefix}_a"), solver, 32),
        zmm_reg_with_unique_values(&format!("{prefix}_b"), solver, 32),
    )
}

fn ymm_reg_pair_with_unique_values(prefix: &str, solver: &Solver) -> (BV, BV) {
    (
        ymm_reg_with_unique_values(&format!("{prefix}_a"), solver, 32),
        ymm_reg_with_unique_values(&format!("{prefix}_b"), solver, 32),
    )
}

#[test]
fn test_mm256_unpacklo_epi32_basic() {
    let solver = Solver::new();
    let a = ymm_reg_with_32b_values(
        "a",
        &solver,
        &[0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7],
    );
    let b = ymm_reg_with_32b_values(
        "b",
        &solver,
        &[0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7],
    );
    let output = mm256_unpacklo_epi32(&a, &b);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&a, 0),
            (&b, 0),
            (&a, 1),
            (&b, 1),
            (&a, 4),
            (&b, 4),
            (&a, 5),
            (&b, 5),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_unpackhi_epi32_basic() {
    let solver = Solver::new();
    let a = ymm_reg_with_32b_values(
        "a",
        &solver,
        &[0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7],
    );
    let b = ymm_reg_with_32b_values(
        "b",
        &solver,
        &[0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7],
    );
    let output = mm256_unpackhi_epi32(&a, &b);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&a, 2),
            (&b, 2),
            (&a, 3),
            (&b, 3),
            (&a, 6),
            (&b, 6),
            (&a, 7),
            (&b, 7),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_unpacklo_epi32_basic() {
    let solver = Solver::new();
    let a = zmm_reg_with_32b_values(
        "a",
        &solver,
        &[
            0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD,
            0xAE, 0xAF,
        ],
    );
    let b = zmm_reg_with_32b_values(
        "b",
        &solver,
        &[
            0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD,
            0xBE, 0xBF,
        ],
    );
    let output = mm512_unpacklo_epi32(&a, &b);
    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&a, 0),
            (&b, 0),
            (&a, 1),
            (&b, 1),
            (&a, 4),
            (&b, 4),
            (&a, 5),
            (&b, 5),
            (&a, 8),
            (&b, 8),
            (&a, 9),
            (&b, 9),
            (&a, 12),
            (&b, 12),
            (&a, 13),
            (&b, 13),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_unpackhi_epi32_basic() {
    let solver = Solver::new();
    let a = zmm_reg_with_32b_values(
        "a",
        &solver,
        &[
            0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD,
            0xAE, 0xAF,
        ],
    );
    let b = zmm_reg_with_32b_values(
        "b",
        &solver,
        &[
            0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD,
            0xBE, 0xBF,
        ],
    );
    let output = mm512_unpackhi_epi32(&a, &b);
    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&a, 2),
            (&b, 2),
            (&a, 3),
            (&b, 3),
            (&a, 6),
            (&b, 6),
            (&a, 7),
            (&b, 7),
            (&a, 10),
            (&b, 10),
            (&a, 11),
            (&b, 11),
            (&a, 14),
            (&b, 14),
            (&a, 15),
            (&b, 15),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_unpacklo_epi32_identity_check() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("input", &solver, 32);
    let output = mm256_unpacklo_epi32(&input, &input);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&input, 0),
            (&input, 0),
            (&input, 1),
            (&input, 1),
            (&input, 4),
            (&input, 4),
            (&input, 5),
            (&input, 5),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_unpackhi_epi32_identity_check() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("input", &solver, 32);
    let output = mm256_unpackhi_epi32(&input, &input);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&input, 2),
            (&input, 2),
            (&input, 3),
            (&input, 3),
            (&input, 6),
            (&input, 6),
            (&input, 7),
            (&input, 7),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_unpacklo_epi32_mask_all_zeros() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver);
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let mask = BV::from_u64(0, 16);
    let output = mm512_mask_unpacklo_epi32(&src, &mask, &a, &b, &solver);

    solver.assert(output.eq(src).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_unpacklo_epi32_mask_all_ones() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver);
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let mask = BV::from_u64(0xFFFF, 16);
    let masked = mm512_mask_unpacklo_epi32(&src, &mask, &a, &b, &solver);
    let unmasked = mm512_unpacklo_epi32(&a, &b);

    solver.assert(masked.eq(unmasked).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_unpackhi_epi32_alternating_mask() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver);
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let mask = BV::from_u64(0x5555, 16);
    let output = mm512_mask_unpackhi_epi32(&src, &mask, &a, &b, &solver);
    let unpack = mm512_unpackhi_epi32(&a, &b);
    let specs: Vec<(&BV, usize)> = (0..16)
        .map(|i| if i % 2 == 0 { (&unpack, i) } else { (&src, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(32, &specs);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_unpacklo_epi32_single_bit_mask() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver);
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let mask = BV::from_u64(1 << 3, 16);
    let output = mm512_mask_unpacklo_epi32(&src, &mask, &a, &b, &solver);
    let unpack = mm512_unpacklo_epi32(&a, &b);
    let specs: Vec<(&BV, usize)> = (0..16)
        .map(|i| if i == 3 { (&unpack, i) } else { (&src, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(32, &specs);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_unpacklo_epi32_reconstruct_pattern() {
    let solver = Solver::new();
    let a = ymm_reg("a");
    let b = ymm_reg("b");
    let output = mm256_unpacklo_epi32(&a, &b);
    let target = BV::from_u64(0x12345678, 32);

    for i in 0..8 {
        solver.assert(output.extract(i * 32 + 31, i * 32).eq(&target));
    }

    assert_eq!(solver.check(), SatResult::Sat);
    let model = solver.get_model().expect("sat model");
    for (reg, lane) in [(&a, 0), (&a, 1), (&b, 0), (&b, 1)] {
        let lane_value = model
            .eval(&reg.extract(lane * 32 + 31, lane * 32), true)
            .expect("lane model")
            .as_u64()
            .expect("lane numeral");
        assert_eq!(lane_value, 0x12345678);
    }
}

#[test]
fn test_mm512_mask_unpackhi_epi32_find_mask() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver);
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let mask = BV::new_const("mask", 16);
    let output = mm512_mask_unpackhi_epi32(&src, &mask, &a, &b, &solver);
    let unpack = mm512_unpackhi_epi32(&a, &b);
    let specs: Vec<(&BV, usize)> = (0..16)
        .map(|i| if i < 4 { (&unpack, i) } else { (&src, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(32, &specs);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_mask = solver
        .get_model()
        .expect("sat model")
        .eval(&mask, true)
        .expect("mask model")
        .as_u64()
        .expect("mask numeral");
    assert_eq!(model_mask, 0x000F);
}

#[test]
fn test_mm256_unpack_combo_lo_hi() {
    let solver = Solver::new();
    let (a, b) = ymm_reg_pair_with_unique_values("input", &solver);
    let lo = mm256_unpacklo_epi32(&a, &b);
    let hi = mm256_unpackhi_epi32(&a, &b);
    solver.assert(lo.eq(hi));
    assert_ne!(solver.check(), SatResult::Unknown);
}

#[test]
fn test_mm512_unpack_lane_independence() {
    let solver = Solver::new();
    let a = zmm_reg_with_32b_values(
        "a",
        &solver,
        &[
            0x10, 0x11, 0x12, 0x13, 0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33, 0x40, 0x41,
            0x42, 0x43,
        ],
    );
    let b = zmm_reg_with_32b_values(
        "b",
        &solver,
        &[
            0x50, 0x51, 0x52, 0x53, 0x60, 0x61, 0x62, 0x63, 0x70, 0x71, 0x72, 0x73, 0x80, 0x81,
            0x82, 0x83,
        ],
    );
    let output = mm512_unpacklo_epi32(&a, &b);
    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&a, 0),
            (&b, 0),
            (&a, 1),
            (&b, 1),
            (&a, 4),
            (&b, 4),
            (&a, 5),
            (&b, 5),
            (&a, 8),
            (&b, 8),
            (&a, 9),
            (&b, 9),
            (&a, 12),
            (&b, 12),
            (&a, 13),
            (&b, 13),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_unpacklo_epi32_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver);
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let mask = BV::new_const("wide_mask_epi32", 64);
    let output = mm512_mask_unpacklo_epi32(&src, &mask, &a, &b, &solver);
    solver.assert(output.eq(src));
    assert_eq!(solver.check(), SatResult::Sat);

    let model_mask = solver
        .get_model()
        .expect("sat model")
        .eval(&mask, true)
        .expect("mask model")
        .as_u64()
        .expect("mask numeral");
    assert_eq!(model_mask >> 16, 0);
}
