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

fn extract_ctl2(imm: &BV) -> (BV, BV) {
    (imm.extract(0, 0), imm.extract(1, 1))
}

fn vpermilpd_lane(lane_idx: usize, a: &BV, ctrl0: &BV, ctrl1: &BV) -> Vec<BV> {
    let src_lane = extract_128b_lane(a, lane_idx);
    vec![select2_pd(&src_lane, ctrl0), select2_pd(&src_lane, ctrl1)]
}

// Generic permute_pd implementation for any number of 128-bit lanes.
// Permutes 64-bit elements within each 128-bit lane using control bits in imm8.
//
// If k (mask) and src are provided, applies masking: elements are copied from src when mask bit is not set.
//
// Operation:
// ```text
// DEFINE SELECT2(src, control) {
//     CASE(control[0]) OF
//     0:      tmp[63:0] := src[63:0]
//     1:      tmp[63:0] := src[127:64]
//     ESAC
//     RETURN tmp[63:0]
// }
// FOR lane := 0 to num_lanes-1
//     dst[lane*128+63:lane*128] := SELECT2(a[lane*128+127:lane*128], imm8[0])
//     dst[lane*128+127:lane*128+64] := SELECT2(a[lane*128+127:lane*128], imm8[1])
// ENDFOR
// ```text
fn permute_pd_generic(
    a: &BV,
    imm8: Imm8,
    num_lanes: usize,
    mask_and_src: Option<(&BV, &BV)>,
    solver: &Solver,
) -> BV {
    if let Imm8::Expr(ref imm) = imm8 {
        constrain_high_bits_zero(imm, 2, solver);
    }

    let imm = imm8.to_bv();
    let (ctrl0, ctrl1) = extract_ctl2(&imm);
    let chunks: Vec<BV> = (0..num_lanes)
        .flat_map(|lane_idx| vpermilpd_lane(lane_idx, a, &ctrl0, &ctrl1))
        .collect();
    let reversed: Vec<BV> = chunks.into_iter().rev().collect();
    let result = concat_msb_first(&reversed).simplify();

    if let Some((mask, src)) = mask_and_src {
        let num_elements = num_lanes * 2;
        constrain_kmask_high_bits_zero(mask, num_elements as u32, solver);
        apply_64b_writemask(&result, src, mask, num_elements)
    } else {
        result
    }
}

// Permutes 64-bit elements within 128-bit lanes. Operates on YMM registers (2 lanes).
pub fn mm256_permute_pd(a: &BV, imm8: Imm8, solver: &Solver) -> BV {
    permute_pd_generic(a, imm8, 2, None, solver)
}

pub fn mm512_permute_pd(a: &BV, imm8: Imm8, solver: &Solver) -> BV {
    permute_pd_generic(a, imm8, 4, None, solver)
}

// Shuffle double-precision (64-bit) floating-point elements in a within 128-bit lanes using the control in imm8,
// and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
// Implements __m512d _mm512_mask_permute_pd (__m512d src, __mmask8 k, __m512d a, const int imm8)
pub fn mm512_mask_permute_pd(src: &BV, k: &BV, a: &BV, imm8: Imm8, solver: &Solver) -> BV {
    permute_pd_generic(a, imm8, 4, Some((k, src)), solver)
}
