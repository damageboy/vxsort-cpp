use z3::Solver;
use z3::ast::{Ast, BV};

use crate::control::Imm8;
use crate::lanes::{concat_msb_first, create_if_tree, extract_128b_lane};

// Selects a 128-bit lane based on 4-bit control according to vperm2i128 semantics.
//
// DEFINE SELECT4(src1, src2, control) {
//     CASE(control[1:0]) OF
//     0:      tmp[127:0] := src1[127:0]
//     1:      tmp[127:0] := src1[255:128]
//     2:      tmp[127:0] := src2[127:0]
//     3:      tmp[127:0] := src2[255:128]
//     ESAC
//     IF control[3]
//         tmp[127:0] := 0
//     FI
//     RETURN tmp[127:0]
// }
fn select4_128b(a: &BV, b: &BV, control: &BV) -> BV {
    let select_bits = control.extract(1, 0);
    let zero_flag = control.extract(3, 3);
    let lanes = [
        extract_128b_lane(a, 0),
        extract_128b_lane(a, 1),
        extract_128b_lane(b, 0),
        extract_128b_lane(b, 1),
    ];
    let selected = create_if_tree(&select_bits, &lanes);
    zero_flag
        .eq(BV::from_u64(1, 1))
        .ite(&BV::from_u64(0, 128), &selected)
        .simplify()
}

fn constrain_dead_control_bits(imm8: &Imm8, solver: &Solver) {
    if let Imm8::Expr(imm) = imm8 {
        solver.assert(imm.extract(2, 2).eq(BV::from_u64(0, 1)));
        solver.assert(imm.extract(6, 6).eq(BV::from_u64(0, 1)));
    }
}

// Shuffle 128-bits (composed of integer data) selected by imm8 from a and b, and store the results in dst.
//
// Implements __m256i _mm256_permute2x128_si256 (__m256i a, __m256i b, const int imm8)
// according to the Intel spec.
//
// Operation:
// ```text
// DEFINE SELECT4(src1, src2, control) {
//     CASE(control[1:0]) OF
//     0:      tmp[127:0] := src1[127:0]
//     1:      tmp[127:0] := src1[255:128]
//     2:      tmp[127:0] := src2[127:0]
//     3:      tmp[127:0] := src2[255:128]
//     ESAC
//     IF control[3]
//         tmp[127:0] := 0
//     FI
//     RETURN tmp[127:0]
// }
// dst[127:0] := SELECT4(a[255:0], b[255:0], imm8[3:0])
// dst[255:128] := SELECT4(a[255:0], b[255:0], imm8[7:4])
// dst[MAX:256] := 0
// ```text
pub fn mm256_permute2x128_si256(a: &BV, b: &BV, imm8: Imm8, solver: &Solver) -> BV {
    constrain_dead_control_bits(&imm8, solver);

    let imm = imm8.to_bv();
    let low_control = imm.extract(3, 0);
    let high_control = imm.extract(7, 4);
    let lanes = [
        select4_128b(a, b, &low_control),
        select4_128b(a, b, &high_control),
    ];
    let reversed: Vec<BV> = lanes.into_iter().rev().collect();
    concat_msb_first(&reversed).simplify()
}
