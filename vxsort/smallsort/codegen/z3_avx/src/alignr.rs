use z3::Solver;
use z3::ast::{Ast, BV};

use crate::canonical::{constrain_high_bits_zero, constrain_kmask_high_bits_zero};
use crate::control::Imm8;
use crate::lanes::{concat_msb_first, create_if_tree, extract_128b_lane, extract_element};
use crate::mask::apply_writemask;

fn shift_bits_needed(num_elements: usize) -> u32 {
    assert!(
        num_elements.is_power_of_two(),
        "alignr element count must be a power of two"
    );
    num_elements.ilog2()
}

// Generic implementation for alignr instructions that concatenate two vectors and shift right.
//
// These instructions concatenate vector a (high part) and vector b (low part) into a
// double-width temporary, shift the result right by imm8 elements, and store the
// low half in the destination. Optional masking is supported for AVX512 variants.
//
// Args:
//     a: First source vector (becomes high part of concatenation)
//     b: Second source vector (becomes low part of concatenation)
//     imm8: Immediate value specifying shift amount in elements
//     total_width: Total bit width of each vector (256 or 512)
//     element_width: Width of each element in bits (32 or 64)
//     src: Optional source vector for masked operations (values used when mask bit is 0)
//     k: Optional predicate mask (if provided, src must also be provided)
//
// Returns:
//     Aligned/shifted vector (optionally masked)
//
// Generic Operation (where N = total_width / element_width):
//     Without mask:
// ```text
//     temp[2*total_width-1:total_width] := a[total_width-1:0]
//     temp[total_width-1:0] := b[total_width-1:0]
//     temp[2*total_width-1:0] := temp[2*total_width-1:0] >> (element_width * imm8)
//     dst[total_width-1:0] := temp[total_width-1:0]
//     dst[MAX:total_width] := 0
// ```text
//
//     With mask:
// ```text
//     temp[2*total_width-1:total_width] := a[total_width-1:0]
//     temp[total_width-1:0] := b[total_width-1:0]
//     temp[2*total_width-1:0] := temp[2*total_width-1:0] >> (element_width * imm8)
//     FOR j := 0 to N-1
//         i := j * element_width
//         IF k[j]
//             dst[i + element_width - 1 : i] := temp[i + element_width - 1 : i]
//         ELSE
//             dst[i + element_width - 1 : i] := src[i + element_width - 1 : i]
//         FI
//     ENDFOR
//     dst[MAX:total_width] := 0
// ```text
//
// Examples:
//     - _mm256_alignr_epi8: total_width=256, element_width=8 → 32 elements, shift by 0-31
//     - _mm256_alignr_epi32: total_width=256, element_width=32 → 8 elements, shift by 0-7
//     - _mm512_alignr_epi32: total_width=512, element_width=32 → 16 elements, shift by 0-15
//     - _mm256_alignr_epi64: total_width=256, element_width=64 → 4 elements, shift by 0-3
//     - _mm512_alignr_epi64: total_width=512, element_width=64 → 8 elements, shift by 0-7
//     - _mm512_mask_alignr_epi32: total_width=512, element_width=32, with src and k
//     - _mm512_mask_alignr_epi64: total_width=512, element_width=64, with src and k
fn generic_alignr(
    a: &BV,
    b: &BV,
    imm8: Imm8,
    total_width: u32,
    element_width: u32,
    solver: &Solver,
    mask_and_src: Option<(&BV, &BV)>,
) -> BV {
    assert_eq!(a.get_size(), total_width);
    assert_eq!(b.get_size(), total_width);
    assert_eq!(total_width % element_width, 0);

    let num_elements = (total_width / element_width) as usize;
    let live_shift_bits = shift_bits_needed(num_elements);
    let imm = imm8.to_bv();
    let shift_amount = imm.extract(live_shift_bits - 1, 0);

    if let Imm8::Expr(expr) = imm8 {
        constrain_high_bits_zero(&expr, live_shift_bits, solver);
    }

    let sources: Vec<BV> = (0..num_elements)
        .map(|element_idx| extract_element(b, element_width, element_idx))
        .chain((0..num_elements).map(|element_idx| extract_element(a, element_width, element_idx)))
        .collect();

    let elements: Vec<BV> = (0..num_elements)
        .map(|dst_idx| {
            let candidates: Vec<BV> = (0..num_elements)
                .map(|shift| sources[shift + dst_idx].clone())
                .collect();
            create_if_tree(&shift_amount, &candidates)
        })
        .collect();

    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    let result = concat_msb_first(&reversed).simplify();

    if let Some((mask, src)) = mask_and_src {
        assert_eq!(src.get_size(), total_width);
        constrain_kmask_high_bits_zero(mask, num_elements as u32, solver);
        apply_writemask(&result, src, mask, element_width, num_elements)
    } else {
        result
    }
}

// Align packed bytes from b and a within each 128-bit lane.
//
// FOR j := 0 to 1
//     i := j*128
//     tmp[255:0] := ((a[i+127:i] << 128)[255:0] OR b[i+127:i]) >> (imm8*8)
//     dst[i+127:i] := tmp[127:0]
// ENDFOR
//
// Implements __m256i _mm256_alignr_epi8(__m256i a, __m256i b, const int imm8)
// (VPALIGNR YMM form; lane-local semantics).
pub fn mm256_alignr_epi8(a: &BV, b: &BV, imm8: Imm8, _solver: &Solver) -> BV {
    let imm = imm8.to_bv();
    let shift_bits = imm.zero_ext(248) * BV::from_u64(8, 256);

    let lanes: Vec<BV> = (0..2)
        .map(|lane_idx| {
            let a_lane = extract_128b_lane(a, lane_idx);
            let b_lane = extract_128b_lane(b, lane_idx);
            let lane_concat = a_lane.concat(&b_lane);
            lane_concat.bvlshr(&shift_bits).extract(127, 0)
        })
        .collect();

    concat_msb_first(&[lanes[1].clone(), lanes[0].clone()]).simplify()
}

// Concatenate a and b into a 64-byte result, shift right by imm8 32-bit elements,
// and store the low 32 bytes (8 elements) in dst.
// Implements __m256i _mm256_alignr_epi32(__m256i a, __m256i b, const int imm8)
// See _generic_alignr for operation details.
pub fn mm256_alignr_epi32(a: &BV, b: &BV, imm8: Imm8, solver: &Solver) -> BV {
    generic_alignr(a, b, imm8, 256, 32, solver, None)
}

// Concatenate a and b into a 128-byte result, shift right by imm8 32-bit elements,
// and store the low 64 bytes (16 elements) in dst.
// Implements __m512i _mm512_alignr_epi32(__m512i a, __m512i b, const int imm8)
// See _generic_alignr for operation details.
pub fn mm512_alignr_epi32(a: &BV, b: &BV, imm8: Imm8, solver: &Solver) -> BV {
    generic_alignr(a, b, imm8, 512, 32, solver, None)
}

// Concatenate a and b into a 128-byte result, shift right by imm8 32-bit elements,
// and store the low 64 bytes (16 elements) in dst using writemask k.
// Elements are copied from src when the corresponding mask bit is not set.
// Implements __m512i _mm512_mask_alignr_epi32(__m512i src, __mmask16 k, __m512i a, __m512i b, const int imm8)
// See _generic_alignr for operation details.
pub fn mm512_mask_alignr_epi32(
    src: &BV,
    k: &BV,
    a: &BV,
    b: &BV,
    imm8: Imm8,
    solver: &Solver,
) -> BV {
    generic_alignr(a, b, imm8, 512, 32, solver, Some((k, src)))
}

// Concatenate a and b into a 64-byte result, shift right by imm8 64-bit elements,
// and store the low 32 bytes (4 elements) in dst.
// Implements __m256i _mm256_alignr_epi64(__m256i a, __m256i b, const int imm8)
// See _generic_alignr for operation details.
pub fn mm256_alignr_epi64(a: &BV, b: &BV, imm8: Imm8, solver: &Solver) -> BV {
    generic_alignr(a, b, imm8, 256, 64, solver, None)
}

// Concatenate a and b into a 128-byte result, shift right by imm8 64-bit elements,
// and store the low 64 bytes (8 elements) in dst.
// Implements __m512i _mm512_alignr_epi64(__m512i a, __m512i b, const int imm8)
// See _generic_alignr for operation details.
pub fn mm512_alignr_epi64(a: &BV, b: &BV, imm8: Imm8, solver: &Solver) -> BV {
    generic_alignr(a, b, imm8, 512, 64, solver, None)
}

// Concatenate a and b into a 128-byte result, shift right by imm8 64-bit elements,
// and store the low 64 bytes (8 elements) in dst using writemask k.
// Elements are copied from src when the corresponding mask bit is not set.
// Implements __m512i _mm512_mask_alignr_epi64(__m512i src, __mmask8 k, __m512i a, __m512i b, const int imm8)
// See _generic_alignr for operation details.
pub fn mm512_mask_alignr_epi64(
    src: &BV,
    k: &BV,
    a: &BV,
    b: &BV,
    imm8: Imm8,
    solver: &Solver,
) -> BV {
    generic_alignr(a, b, imm8, 512, 64, solver, Some((k, src)))
}
