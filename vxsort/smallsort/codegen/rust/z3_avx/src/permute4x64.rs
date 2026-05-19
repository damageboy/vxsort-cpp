use z3::ast::{Ast, BV};

use crate::control::Imm8;
use crate::lanes::{concat_msb_first, create_if_tree, extract_element};

// Selects a 64-bit element from a 256-bit vector based on a 2-bit control.
fn select4_epi64(src_256: &BV, select: &BV) -> BV {
    let elements: Vec<BV> = (0..4)
        .map(|element_idx| extract_element(src_256, 64, element_idx))
        .collect();
    create_if_tree(select, &elements)
}

// Shuffle 64-bit integers in "a" across lanes using the control in "imm8", and store the results in "dst".
//
// Implements __m256i _mm256_permute4x64_epi64 (__m256i a, const int imm8)
//
// Operation:
// ```text
// DEFINE SELECT4(src, control) {
//     CASE(control[1:0]) OF
//     0:      tmp[63:0] := src[63:0]
//     1:      tmp[63:0] := src[127:64]
//     2:      tmp[63:0] := src[191:128]
//     3:      tmp[63:0] := src[255:192]
//     ESAC
//     RETURN tmp[63:0]
// }
// dst[63:0] := SELECT4(a[255:0], imm8[1:0])
// dst[127:64] := SELECT4(a[255:0], imm8[3:2])
// dst[191:128] := SELECT4(a[255:0], imm8[5:4])
// dst[255:192] := SELECT4(a[255:0], imm8[7:6])
// dst[MAX:256] := 0
// ```text
//
// Args:
//     a: Source vector (256-bit)
//     imm8: Immediate 8-bit control mask (uses all 8 bits for 4 elements, 2 bits each)
//
// Returns:
//     Permuted 256-bit vector
pub fn mm256_permute4x64_epi64(a: &BV, imm8: Imm8) -> BV {
    let imm = imm8.to_bv();
    let elements: Vec<BV> = (0..4)
        .map(|element_idx| {
            let low = element_idx * 2;
            let select = imm.extract(low + 1, low);
            select4_epi64(a, &select)
        })
        .collect();
    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    concat_msb_first(&reversed).simplify()
}
