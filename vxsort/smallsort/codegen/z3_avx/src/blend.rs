use z3::Solver;
use z3::ast::{Ast, BV};

use crate::canonical::constrain_high_bits_zero;
use crate::control::Imm8;
use crate::lanes::{concat_msb_first, extract_element};

// Generic implementation for immediate blend instructions that select elements from two source vectors.
//
// These instructions use an immediate mask where each bit controls the selection for one element.
// If the mask bit is 1, the element is selected from b; otherwise from a.
//
// Args:
//     a: First source vector
//     b: Second source vector
//     imm8: Immediate 8-bit control mask
//     total_width: Total bit width of the vectors (256)
//     element_width: Width of each element in bits (32 or 64)
//
// Returns:
//     Blended vector
//
// Generic Operation (where N = total_width / element_width):
// ```text
//     FOR j := 0 to N-1
//         i := j * element_width
//         IF imm8[j]
//             dst[i + element_width - 1 : i] := b[i + element_width - 1 : i]
//         ELSE
//             dst[i + element_width - 1 : i] := a[i + element_width - 1 : i]
//         FI
//     ENDFOR
//     dst[MAX:total_width] := 0
// ```text
//
// Examples:
//     - _mm256_blend_pd: total_width=256, element_width=64 → 4 elements
//     - _mm256_blend_ps: total_width=256, element_width=32 → 8 elements
fn generic_blend(a: &BV, b: &BV, imm8: Imm8, element_width: u32, solver: &Solver) -> BV {
    let total_width = 256;
    let num_elements = total_width / element_width;

    if let Imm8::Expr(ref imm) = imm8
        && num_elements < 8
    {
        constrain_high_bits_zero(imm, num_elements, solver);
    }

    let imm = imm8.to_bv();
    let elements: Vec<BV> = (0..num_elements)
        .map(|element_idx| {
            let mask_bit = imm.extract(element_idx, element_idx);
            let a_elem = extract_element(a, element_width, element_idx as usize);
            let b_elem = extract_element(b, element_width, element_idx as usize);
            mask_bit
                .eq(BV::from_u64(1, 1))
                .ite(&b_elem, &a_elem)
                .simplify()
        })
        .collect();
    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    concat_msb_first(&reversed).simplify()
}

// Generic implementation for variable blend instructions that select elements from two source vectors.
//
// These instructions use a mask vector where the sign bit (MSB) of each element controls the selection.
// If the sign bit is 1, the element is selected from b; otherwise from a.
//
// Args:
//     a: First source vector
//     b: Second source vector
//     mask: Variable mask vector (uses sign bit of each element)
//     total_width: Total bit width of the vectors (256)
//     element_width: Width of each element in bits (32 or 64)
//
// Returns:
//     Blended vector
//
// Generic Operation (where N = total_width / element_width):
// ```text
//     FOR j := 0 to N-1
//         i := j * element_width
//         IF mask[i + element_width - 1]  // sign bit (MSB)
//             dst[i + element_width - 1 : i] := b[i + element_width - 1 : i]
//         ELSE
//             dst[i + element_width - 1 : i] := a[i + element_width - 1 : i]
//         FI
//     ENDFOR
//     dst[MAX:total_width] := 0
// ```text
//
// Examples:
//     - _mm256_blendv_pd: total_width=256, element_width=64 → 4 elements
//     - _mm256_blendv_ps: total_width=256, element_width=32 → 8 elements
fn generic_blendv(a: &BV, b: &BV, mask: &BV, element_width: u32, solver: &Solver) -> BV {
    let total_width = 256;
    let num_elements = total_width / element_width;

    let elements: Vec<BV> = (0..num_elements)
        .map(|element_idx| {
            let low = element_idx * element_width;
            let high = low + element_width - 1;
            let sign_bit = mask.extract(high, high);
            let a_elem = extract_element(a, element_width, element_idx as usize);
            let b_elem = extract_element(b, element_width, element_idx as usize);
            if element_width > 1 {
                solver.assert(
                    mask.extract(high - 1, low)
                        .eq(BV::from_u64(0, element_width - 1)),
                );
            }
            sign_bit
                .eq(BV::from_u64(1, 1))
                .ite(&b_elem, &a_elem)
                .simplify()
        })
        .collect();
    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    concat_msb_first(&reversed).simplify()
}

// Blend packed double-precision (64-bit) floating-point elements from "a" and "b" using control mask "imm8",
// and store the results in "dst".
// Implements __m256d _mm256_blend_pd (__m256d a, __m256d b, const int imm8)
pub fn mm256_blend_pd(a: &BV, b: &BV, imm8: Imm8, solver: &Solver) -> BV {
    generic_blend(a, b, imm8, 64, solver)
}

pub fn mm256_blend_ps(a: &BV, b: &BV, imm8: Imm8, solver: &Solver) -> BV {
    generic_blend(a, b, imm8, 32, solver)
}

// Blend packed double-precision (64-bit) floating-point elements from "a" and "b" using "mask",
// and store the results in "dst".
// Implements __m256d _mm256_blendv_pd (__m256d a, __m256d b, __m256d mask)
pub fn mm256_blendv_pd(a: &BV, b: &BV, mask: &BV, solver: &Solver) -> BV {
    generic_blendv(a, b, mask, 64, solver)
}

pub fn mm256_blendv_ps(a: &BV, b: &BV, mask: &BV, solver: &Solver) -> BV {
    generic_blendv(a, b, mask, 32, solver)
}
