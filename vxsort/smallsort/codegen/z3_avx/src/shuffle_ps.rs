use z3::Solver;
use z3::ast::{Ast, BV};

use crate::canonical::constrain_kmask_high_bits_zero;
use crate::control::Imm8;
use crate::lanes::{concat_msb_first, create_if_tree, extract_128b_lane};
use crate::mask::apply_32b_writemask;

// Selects a 32-bit element from a 128-bit vector based on a 2-bit control.
fn select4_ps(src_128: &BV, select: &BV) -> BV {
    let elements = [
        src_128.extract(31, 0),
        src_128.extract(63, 32),
        src_128.extract(95, 64),
        src_128.extract(127, 96),
    ];
    create_if_tree(select, &elements)
}

fn extract_ctl4(imm: &BV) -> (BV, BV, BV, BV) {
    (
        imm.extract(1, 0),
        imm.extract(3, 2),
        imm.extract(5, 4),
        imm.extract(7, 6),
    )
}

fn vshufps_lane(
    lane_idx: usize,
    a: &BV,
    b: &BV,
    ctrl01: &BV,
    ctrl23: &BV,
    ctrl45: &BV,
    ctrl67: &BV,
) -> Vec<BV> {
    let a_lane = extract_128b_lane(a, lane_idx);
    let b_lane = extract_128b_lane(b, lane_idx);
    vec![
        select4_ps(&a_lane, ctrl01),
        select4_ps(&a_lane, ctrl23),
        select4_ps(&b_lane, ctrl45),
        select4_ps(&b_lane, ctrl67),
    ]
}

// Generic shuffle_ps implementation for any number of 128-bit lanes.
// Shuffles 32-bit elements within 128-bit lanes using control in imm8.
//
// If k (mask) and src are provided, applies masking: elements are copied from src when mask bit is not set.
//
// Operation:
// ```text
// DEFINE SELECT4(src, control) {
//     CASE(control[1:0]) OF
//     0:      tmp[31:0] := src[31:0]
//     1:      tmp[31:0] := src[63:32]
//     2:      tmp[31:0] := src[95:64]
//     3:      tmp[31:0] := src[127:96]
//     ESAC
//     RETURN tmp[31:0]
// }
// FOR lane := 0 to num_lanes-1
//     dst[lane*128+31:lane*128] := SELECT4(a[lane*128+127:lane*128], imm8[1:0])
//     dst[lane*128+63:lane*128+32] := SELECT4(a[lane*128+127:lane*128], imm8[3:2])
//     dst[lane*128+95:lane*128+64] := SELECT4(b[lane*128+127:lane*128], imm8[5:4])
//     dst[lane*128+127:lane*128+96] := SELECT4(b[lane*128+127:lane*128], imm8[7:6])
// ENDFOR
// ```text
fn shuffle_ps_generic(
    a: &BV,
    b: &BV,
    imm8: Imm8,
    num_lanes: usize,
    mask_and_src: Option<(&BV, &BV, &Solver)>,
) -> BV {
    let imm = imm8.to_bv();
    let (ctrl01, ctrl23, ctrl45, ctrl67) = extract_ctl4(&imm);
    let chunks: Vec<BV> = (0..num_lanes)
        .flat_map(|lane_idx| vshufps_lane(lane_idx, a, b, &ctrl01, &ctrl23, &ctrl45, &ctrl67))
        .collect();
    let reversed: Vec<BV> = chunks.into_iter().rev().collect();
    let result = concat_msb_first(&reversed).simplify();

    let num_elements = num_lanes * 4;
    if let Some((mask, src, solver)) = mask_and_src {
        constrain_kmask_high_bits_zero(mask, num_elements as u32, solver);
        apply_32b_writemask(&result, src, mask, num_elements)
    } else {
        result
    }
}

// Shuffles 32-bit elements within 128-bit lanes. Operates on YMM registers (2 lanes).
pub fn mm256_shuffle_ps(a: &BV, b: &BV, imm8: Imm8) -> BV {
    shuffle_ps_generic(a, b, imm8, 2, None)
}

pub fn mm512_shuffle_ps(a: &BV, b: &BV, imm8: Imm8) -> BV {
    shuffle_ps_generic(a, b, imm8, 4, None)
}

// Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in imm8,
// and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
// Implements __m512 _mm512_mask_shuffle_ps (__m512 src, __mmask16 k, __m512 a, __m512 b, const int imm8)
pub fn mm512_mask_shuffle_ps(src: &BV, k: &BV, a: &BV, b: &BV, imm8: Imm8, solver: &Solver) -> BV {
    shuffle_ps_generic(a, b, imm8, 4, Some((k, src, solver)))
}
