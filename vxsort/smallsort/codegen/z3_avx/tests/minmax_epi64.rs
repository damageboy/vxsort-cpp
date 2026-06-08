use z3::ast::{BV, Bool};
use z3::{SatResult, Solver};
use z3_avx::lanes::concat_msb_first;
use z3_avx::minmax::{mm256_max_epi64, mm256_min_epi64, mm512_max_epi64, mm512_min_epi64};
use z3_avx::registers::{ymm_reg, ymm_reg_with_64b_values, zmm_reg_with_64b_values};

fn packed_64(values: &[u64], total_bits: u32) -> BV {
    assert_eq!(values.len(), (total_bits / 64) as usize);
    let elements: Vec<BV> = values
        .iter()
        .map(|value| BV::from_u64(*value, 64))
        .collect();
    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    concat_msb_first(&reversed)
}

#[test]
fn test_mm256_min_epi64_concrete() {
    let solver = Solver::new();
    let a = ymm_reg_with_64b_values("a", &solver, &[5, 3, 7, 1]);
    let b = ymm_reg_with_64b_values("b", &solver, &[2, 8, 4, 6]);
    let expected = packed_64(&[2, 3, 4, 1], 256);

    let result = mm256_min_epi64(&a, &b);

    solver.assert(result.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_max_epi64_concrete() {
    let solver = Solver::new();
    let a = ymm_reg_with_64b_values("a", &solver, &[5, 3, 7, 1]);
    let b = ymm_reg_with_64b_values("b", &solver, &[2, 8, 4, 6]);
    let expected = packed_64(&[5, 8, 7, 6], 256);

    let result = mm256_max_epi64(&a, &b);

    solver.assert(result.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_min_epi64_signed() {
    let solver = Solver::new();
    let neg_one = u64::MAX;
    let a = ymm_reg_with_64b_values("a", &solver, &[neg_one; 4]);
    let b = ymm_reg_with_64b_values("b", &solver, &[1; 4]);
    let expected = packed_64(&[neg_one; 4], 256);

    let result = mm256_min_epi64(&a, &b);

    solver.assert(result.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_minmax_epi64_preserves_elements() {
    let solver = Solver::new();
    let a = ymm_reg("a");
    let b = ymm_reg("b");
    let lo = mm256_min_epi64(&a, &b);
    let hi = mm256_max_epi64(&a, &b);

    for lane in 0..4 {
        let low = lane * 64;
        let high = low + 63;
        let a_elem = a.extract(high, low);
        let b_elem = b.extract(high, low);
        let lo_elem = lo.extract(high, low);
        let hi_elem = hi.extract(high, low);
        let keeps_order = Bool::and(&[lo_elem.eq(&a_elem), hi_elem.eq(&b_elem)]);
        let swaps_order = Bool::and(&[lo_elem.eq(&b_elem), hi_elem.eq(&a_elem)]);

        solver.assert(Bool::or(&[keeps_order, swaps_order]).not());
    }

    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_min_epi64_concrete() {
    let solver = Solver::new();
    let a = zmm_reg_with_64b_values("a", &solver, &[5, 3, 7, 1, 8, 2, 6, 4]);
    let b = zmm_reg_with_64b_values("b", &solver, &[2, 8, 4, 6, 1, 5, 3, 7]);
    let expected = packed_64(&[2, 3, 4, 1, 1, 2, 3, 4], 512);

    let result = mm512_min_epi64(&a, &b);

    solver.assert(result.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_max_epi64_concrete() {
    let solver = Solver::new();
    let a = zmm_reg_with_64b_values("a", &solver, &[5, 3, 7, 1, 8, 2, 6, 4]);
    let b = zmm_reg_with_64b_values("b", &solver, &[2, 8, 4, 6, 1, 5, 3, 7]);
    let expected = packed_64(&[5, 8, 7, 6, 8, 5, 6, 7], 512);

    let result = mm512_max_epi64(&a, &b);

    solver.assert(result.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}
