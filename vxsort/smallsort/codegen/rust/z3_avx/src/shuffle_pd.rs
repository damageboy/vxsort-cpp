use z3::Solver;
use z3::ast::{Ast, BV};

use crate::canonical::{constrain_high_bits_zero, constrain_kmask_high_bits_zero};
use crate::control::Imm8;
use crate::lanes::{concat_msb_first, create_if_tree, extract_128b_lane};
use crate::mask::apply_64b_writemask;

// Selects a 64-bit element from a 128-bit vector based on a 1-bit control.
fn select2_pd(src_128: &BV, select: &BV) -> BV {
    let elements = [src_128.extract(63, 0), src_128.extract(127, 64)];
    create_if_tree(select, &elements)
}

fn vshufpd_lane(lane_idx: usize, a: &BV, b: &BV, imm: &BV) -> Vec<BV> {
    let a_lane = extract_128b_lane(a, lane_idx);
    let b_lane = extract_128b_lane(b, lane_idx);
    let ctrl0 = imm.extract((2 * lane_idx) as u32, (2 * lane_idx) as u32);
    let ctrl1 = imm.extract((2 * lane_idx + 1) as u32, (2 * lane_idx + 1) as u32);

    vec![select2_pd(&a_lane, &ctrl0), select2_pd(&b_lane, &ctrl1)]
}

// Generic shuffle_pd implementation for any number of 128-bit lanes.
// Shuffles 64-bit elements within 128-bit lanes using control in imm8.
//
// If k (mask) and src are provided, applies masking: elements are copied from src when mask bit is not set.
//
// Operation:
// ```text
// FOR lane := 0 to num_lanes-1
//     dst[lane*128+63:lane*128] := (imm8[2*lane] == 0) ? a[lane*128+63:lane*128] : a[lane*128+127:lane*128+64]
//     dst[lane*128+127:lane*128+64] := (imm8[2*lane+1] == 0) ? b[lane*128+63:lane*128] : b[lane*128+127:lane*128+64]
// ENDFOR
// ```text
fn shuffle_pd_generic(
    a: &BV,
    b: &BV,
    imm8: Imm8,
    num_lanes: usize,
    solver: &Solver,
    mask_and_src: Option<(&BV, &BV)>,
) -> BV {
    let imm = imm8.to_bv();
    let chunks: Vec<BV> = (0..num_lanes)
        .flat_map(|lane_idx| vshufpd_lane(lane_idx, a, b, &imm))
        .collect();
    let reversed: Vec<BV> = chunks.into_iter().rev().collect();
    let result = concat_msb_first(&reversed).simplify();

    if let Imm8::Expr(expr) = imm8 {
        let live_bits = (num_lanes * 2) as u32;
        if live_bits < 8 {
            constrain_high_bits_zero(&expr, live_bits, solver);
        }
    }

    let num_elements = num_lanes * 2;
    if let Some((mask, src)) = mask_and_src {
        constrain_kmask_high_bits_zero(mask, num_elements as u32, solver);
        apply_64b_writemask(&result, src, mask, num_elements)
    } else {
        result
    }
}

// Shuffles 64-bit elements within 128-bit lanes. Operates on YMM registers (2 lanes).
pub fn mm256_shuffle_pd(a: &BV, b: &BV, imm8: Imm8, solver: &Solver) -> BV {
    shuffle_pd_generic(a, b, imm8, 2, solver, None)
}

pub fn mm512_shuffle_pd(a: &BV, b: &BV, imm8: Imm8, solver: &Solver) -> BV {
    shuffle_pd_generic(a, b, imm8, 4, solver, None)
}

// Shuffle double-precision (64-bit) floating-point elements within 128-bit lanes using the control in imm8,
// and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
// Implements __m512d _mm512_mask_shuffle_pd (__m512d src, __mmask8 k, __m512d a, __m512d b, const int imm8)
pub fn mm512_mask_shuffle_pd(src: &BV, k: &BV, a: &BV, b: &BV, imm8: Imm8, solver: &Solver) -> BV {
    shuffle_pd_generic(a, b, imm8, 4, solver, Some((k, src)))
}
