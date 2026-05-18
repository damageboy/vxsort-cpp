use z3::Solver;
use z3::ast::{Ast, BV};

use crate::canonical::constrain_kmask_high_bits_zero;
use crate::control::Imm8;
use crate::lanes::{concat_msb_first, create_if_tree, extract_128b_lane};
use crate::mask::apply_32b_writemask;

// Selects a 128-bit lane from a 512-bit source based on 2-bit control according to vshufi32x4 semantics.
//
// DEFINE SELECT4(src, control) {
//     CASE(control[1:0]) OF
//     0:      tmp[127:0] := src[127:0]
//     1:      tmp[127:0] := src[255:128]
//     2:      tmp[127:0] := src[383:256]
//     3:      tmp[127:0] := src[511:384]
//     ESAC
//     RETURN tmp[127:0]
// }
fn select4_128b_from_zmm(src: &BV, control: &BV) -> BV {
    let lanes = [
        extract_128b_lane(src, 0),
        extract_128b_lane(src, 1),
        extract_128b_lane(src, 2),
        extract_128b_lane(src, 3),
    ];
    create_if_tree(control, &lanes)
}

fn shuffle_i32x4_generic(
    a: &BV,
    b: &BV,
    imm8: Imm8,
    mask_and_src: Option<(&BV, &BV, &Solver)>,
) -> BV {
    let imm = imm8.to_bv();
    let lanes: Vec<BV> = (0..4)
        .map(|lane_idx| {
            let source = if lane_idx < 2 { a } else { b };
            let low = lane_idx as u32 * 2;
            let control = imm.extract(low + 1, low);
            select4_128b_from_zmm(source, &control)
        })
        .collect();
    let reversed: Vec<BV> = lanes.into_iter().rev().collect();
    let result = concat_msb_first(&reversed).simplify();

    if let Some((mask, src, solver)) = mask_and_src {
        constrain_kmask_high_bits_zero(mask, 16, solver);
        apply_32b_writemask(&result, src, mask, 16)
    } else {
        result
    }
}

// Shuffle 128-bits (composed of 4 32-bit integers) selected by imm8 from a and b, and store the results in dst.
//
// Implements __m512i _mm512_shuffle_i32x4 (__m512i a, __m512i b, const int imm8)
// according to the Intel spec.
//
// Operation:
// ```text
// DEFINE SELECT4(src, control) {
//     CASE(control[1:0]) OF
//     0:      tmp[127:0] := src[127:0]
//     1:      tmp[127:0] := src[255:128]
//     2:      tmp[127:0] := src[383:256]
//     3:      tmp[127:0] := src[511:384]
//     ESAC
//     RETURN tmp[127:0]
// }
// dst[127:0] := SELECT4(a[511:0], imm8[1:0])
// dst[255:128] := SELECT4(a[511:0], imm8[3:2])
// dst[383:256] := SELECT4(b[511:0], imm8[5:4])
// dst[511:384] := SELECT4(b[511:0], imm8[7:6])
// dst[MAX:512] := 0
// ```text
pub fn mm512_shuffle_i32x4(a: &BV, b: &BV, imm8: Imm8) -> BV {
    shuffle_i32x4_generic(a, b, imm8, None)
}

// Shuffle 128-bit lanes from a and b using imm8, with merge masking.
//
// Implements __m512i _mm512_mask_shuffle_i32x4(__m512i src, __mmask16 k,
// __m512i a, __m512i b, const int imm8)
//
// Elements are copied from src when the corresponding mask bit is not set.
pub fn mm512_mask_shuffle_i32x4(
    src: &BV,
    k: &BV,
    a: &BV,
    b: &BV,
    imm8: Imm8,
    solver: &Solver,
) -> BV {
    shuffle_i32x4_generic(a, b, imm8, Some((k, src, solver)))
}
