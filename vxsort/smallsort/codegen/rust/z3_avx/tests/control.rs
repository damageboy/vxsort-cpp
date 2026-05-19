use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::canonical::{
    constrain_bits_outside_ranges_zero, constrain_high_bits_zero, constrain_kmask_high_bits_zero,
    constrain_lane_high_bits_zero,
};
use z3_avx::control::{
    Imm8, decode_shuffle_mask, decode_shuffle2_mask, mm_shuffle, mm_shuffle_str, mm_shuffle2,
    mm_shuffle2_str,
};

#[test]
fn test_decode_shuffle_mask_round_trip_all_values() {
    for imm8 in 0..=255 {
        let (z, y, x, w) = decode_shuffle_mask(imm8);
        assert!(w <= 3, "w={w} out of range for imm8=0x{imm8:02x}");
        assert!(x <= 3, "x={x} out of range for imm8=0x{imm8:02x}");
        assert!(y <= 3, "y={y} out of range for imm8=0x{imm8:02x}");
        assert!(z <= 3, "z={z} out of range for imm8=0x{imm8:02x}");

        assert_eq!(
            mm_shuffle(z, y, x, w),
            imm8,
            "Round-trip failed for 0x{imm8:02x}: decoded to ({z}, {y}, {x}, {w}), re-encoded to 0x{:02x}",
            mm_shuffle(z, y, x, w)
        );
    }
}

#[test]
fn test_mm_shuffle_str_identity() {
    assert_eq!(mm_shuffle_str(0xE4), "_MM_SHUFFLE(3, 2, 1, 0)");
}

#[test]
fn test_mm_shuffle_str_0x88() {
    assert_eq!(mm_shuffle_str(0x88), "_MM_SHUFFLE(2, 0, 2, 0)");
}

#[test]
fn test_mm_shuffle_str_0xdd() {
    assert_eq!(mm_shuffle_str(0xDD), "_MM_SHUFFLE(3, 1, 3, 1)");
}

#[test]
fn test_mm_shuffle_str_all_zeros() {
    assert_eq!(mm_shuffle_str(0x00), "_MM_SHUFFLE(0, 0, 0, 0)");
}

#[test]
fn test_mm_shuffle_str_all_ones() {
    assert_eq!(mm_shuffle_str(0xFF), "_MM_SHUFFLE(3, 3, 3, 3)");
}

#[test]
fn test_mm_shuffle_str_reverse() {
    assert_eq!(mm_shuffle_str(0x1B), "_MM_SHUFFLE(0, 1, 2, 3)");
}

#[test]
fn test_mm_shuffle_str_format() {
    for imm8 in [0x00, 0x88, 0xDD, 0xE4, 0xFF, 0x1B, 0x4E, 0xB1] {
        let result = mm_shuffle_str(imm8);
        assert!(
            result.starts_with("_MM_SHUFFLE("),
            "Bad format for 0x{imm8:02x}: {result}"
        );
        assert!(
            result.ends_with(')'),
            "Bad format for 0x{imm8:02x}: {result}"
        );
        assert_eq!(
            result.matches(',').count(),
            3,
            "Bad format for 0x{imm8:02x}: {result}"
        );
    }
}

#[test]
fn test_decode_shuffle2_mask_all_zeros() {
    assert_eq!(decode_shuffle2_mask(0x0), (0, 0));
}

#[test]
fn test_decode_shuffle2_mask_identity() {
    assert_eq!(decode_shuffle2_mask(0x5), (1, 1));
}

#[test]
fn test_decode_shuffle2_mask_swap() {
    assert_eq!(decode_shuffle2_mask(0xA), (2, 2));
}

#[test]
fn test_decode_shuffle2_mask_all_ones() {
    assert_eq!(decode_shuffle2_mask(0xF), (3, 3));
}

#[test]
fn test_decode_shuffle2_mask_mixed() {
    assert_eq!(decode_shuffle2_mask(0x1), (0, 1));
    assert_eq!(decode_shuffle2_mask(0x2), (0, 2));
    assert_eq!(decode_shuffle2_mask(0x3), (0, 3));
    assert_eq!(decode_shuffle2_mask(0x4), (1, 0));
    assert_eq!(decode_shuffle2_mask(0x9), (2, 1));
}

#[test]
fn test_mm_shuffle2_str_all_zeros() {
    assert_eq!(mm_shuffle2_str(0x0), "_MM_SHUFFLE2(0, 0)");
}

#[test]
fn test_mm_shuffle2_str_identity() {
    assert_eq!(mm_shuffle2_str(0x5), "_MM_SHUFFLE2(1, 1)");
}

#[test]
fn test_mm_shuffle2_str_swap() {
    assert_eq!(mm_shuffle2_str(0xA), "_MM_SHUFFLE2(2, 2)");
}

#[test]
fn test_mm_shuffle2_str_all_ones() {
    assert_eq!(mm_shuffle2_str(0xF), "_MM_SHUFFLE2(3, 3)");
}

#[test]
fn test_mm_shuffle2_str_format() {
    for imm8 in [0x00, 0x05, 0x0A, 0x0F, 0x01, 0x09] {
        let result = mm_shuffle2_str(imm8);
        assert!(
            result.starts_with("_MM_SHUFFLE2("),
            "Bad format for 0x{imm8:02x}: {result}"
        );
        assert!(
            result.ends_with(')'),
            "Bad format for 0x{imm8:02x}: {result}"
        );
        assert_eq!(
            result.matches(',').count(),
            1,
            "Bad format for 0x{imm8:02x}: {result}"
        );
    }
}

#[test]
fn imm8_converts_literal_or_expr_to_bv() {
    assert_eq!(Imm8::Literal(0xE4).to_bv().get_size(), 8);

    let symbolic = BV::new_const("imm8", 8);
    assert_eq!(Imm8::Expr(symbolic).to_bv().get_size(), 8);
}

#[test]
fn mm_shuffle2_encodes_two_controls() {
    assert_eq!(mm_shuffle2(1, 0), 0b10);
}

#[test]
fn canonical_scalar_high_bits_constraints_keep_only_low_live_bits() {
    let solver = Solver::new();
    let imm = BV::new_const("scalar_imm", 8);

    constrain_high_bits_zero(&imm, 3, &solver);
    solver.assert(imm.extract(2, 0).eq(BV::from_u64(0b101, 3)));
    assert_eq!(solver.check(), SatResult::Sat);

    solver.push();
    solver.assert(imm.extract(7, 3).eq(BV::from_u64(1, 5)));
    assert_eq!(solver.check(), SatResult::Unsat);
    solver.pop(1);
}

#[test]
fn canonical_lane_high_bits_constraints_keep_only_low_live_bits_per_lane() {
    let solver = Solver::new();
    let controls = BV::new_const("lane_controls", 128);

    constrain_lane_high_bits_zero(&controls, 32, 2, &solver);
    solver.assert(controls.extract(1, 0).eq(BV::from_u64(0b11, 2)));
    solver.assert(controls.extract(33, 32).eq(BV::from_u64(0b10, 2)));
    assert_eq!(solver.check(), SatResult::Sat);

    solver.push();
    solver.assert(controls.extract(31, 2).eq(BV::from_u64(1, 30)));
    assert_eq!(solver.check(), SatResult::Unsat);
    solver.pop(1);
}

#[test]
fn canonical_range_constraints_keep_only_requested_live_ranges() {
    let solver = Solver::new();
    let control = BV::new_const("range_control", 8);

    constrain_bits_outside_ranges_zero(&control, &[(0, 1), (3, 3), (5, 6)], &solver);
    solver.assert(control.extract(1, 0).eq(BV::from_u64(0b11, 2)));
    solver.assert(control.extract(3, 3).eq(BV::from_u64(1, 1)));
    solver.assert(control.extract(6, 5).eq(BV::from_u64(0b10, 2)));
    assert_eq!(solver.check(), SatResult::Sat);

    solver.push();
    solver.assert(control.extract(2, 2).eq(BV::from_u64(1, 1)));
    assert_eq!(solver.check(), SatResult::Unsat);
    solver.pop(1);

    solver.push();
    solver.assert(control.extract(7, 7).eq(BV::from_u64(1, 1)));
    assert_eq!(solver.check(), SatResult::Unsat);
    solver.pop(1);
}

#[test]
fn canonical_kmask_constraints_keep_only_live_low_mask_bits() {
    let solver = Solver::new();
    let mask = BV::new_const("wide_kmask", 64);

    constrain_kmask_high_bits_zero(&mask, 16, &solver);
    solver.assert(mask.extract(15, 0).eq(BV::from_u64(0xA5A5, 16)));
    assert_eq!(solver.check(), SatResult::Sat);

    solver.push();
    solver.assert(mask.extract(63, 16).eq(BV::from_u64(1, 48)));
    assert_eq!(solver.check(), SatResult::Unsat);
    solver.pop(1);
}
