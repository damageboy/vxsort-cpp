use z3::Solver;
use z3::ast::{Ast, BV};

use crate::canonical::constrain_kmask_high_bits_zero;
use crate::lanes::{concat_msb_first, extract_128b_lane};
use crate::mask::{apply_32b_writemask, apply_64b_writemask};

// Generic unpack implementation for 32-bit integers with optional masking.
//
// Args:
//     a: First source register
//     b: Second source register
//     high: True for unpackhi (elements 2,3), False for unpacklo (elements 0,1)
//     total_bits: Register size (256 or 512)
//     src: Source register for masked operations (None for unmasked)
//     k: Write mask (None for unmasked operations)
//
// Returns:
//     BitVecRef representing the unpacked result
//
// Pseudocode:
// For each 128-bit lane in the input:
//   If high is False (unpacklo), interleave elements 0 and 1 of a and b within each lane:
//     dst[0] = a[0], dst[1] = b[0], dst[2] = a[1], dst[3] = b[1]
//   If high is True (unpackhi), interleave elements 2 and 3 of a and b within each lane:
//     dst[0] = a[2], dst[1] = b[2], dst[2] = a[3], dst[3] = b[3]
//
// For total_bits=256, process 2 lanes; for 512, process 4 lanes.
// If masking is requested (src and k are not None), for each 32-bit element, choose the result from
// the unpacked value if the corresponding mask bit is set, otherwise use the value from src.
fn unpack_epi32_generic(
    a: &BV,
    b: &BV,
    high: bool,
    total_bits: u32,
    mask_and_src: Option<(&BV, &BV, &Solver)>,
) -> BV {
    assert!(
        total_bits == 256 || total_bits == 512,
        "total_bits must be 256 or 512"
    );

    let lane_count = (total_bits / 128) as usize;
    let mut elements = Vec::with_capacity((total_bits / 32) as usize);

    for lane_idx in 0..lane_count {
        let a_lane = extract_128b_lane(a, lane_idx);
        let b_lane = extract_128b_lane(b, lane_idx);
        let base = if high { 2 } else { 0 };

        let a0 = a_lane.extract((base + 1) * 32 - 1, base * 32);
        let b0 = b_lane.extract((base + 1) * 32 - 1, base * 32);
        let a1 = a_lane.extract((base + 2) * 32 - 1, (base + 1) * 32);
        let b1 = b_lane.extract((base + 2) * 32 - 1, (base + 1) * 32);

        elements.extend([a0, b0, a1, b1]);
    }

    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    let result = concat_msb_first(&reversed).simplify();

    if let Some((mask, src, solver)) = mask_and_src {
        let num_elements = (total_bits / 32) as usize;
        constrain_kmask_high_bits_zero(mask, num_elements as u32, solver);
        apply_32b_writemask(&result, src, mask, num_elements)
    } else {
        result
    }
}

// Generic unpack implementation for 64-bit integers with optional masking.
//
// Args:
//     a: First source register
//     b: Second source register
//     high: True for unpackhi (element 1), False for unpacklo (element 0)
//     total_bits: Register size (256 or 512)
//     src: Source register for masked operations (None for unmasked)
//     k: Write mask (None for unmasked operations)
//
// Returns:
//     BitVecRef representing the unpacked result
//
// Pseudocode:
// For each 128-bit lane in the input:
//   If high is False (unpacklo), interleave element 0 of a and b within each lane:
//     dst[0] = a[0], dst[1] = b[0]
//   If high is True (unpackhi), interleave element 1 of a and b within each lane:
//     dst[0] = a[1], dst[1] = b[1]
//
// For total_bits=256, process 2 lanes (4 elements total); for 512, process 4 lanes (8 elements total).
// If masking is requested (src and k are not None), for each 64-bit element, choose the result from
// the unpacked value if the corresponding mask bit is set, otherwise use the value from src.
fn unpack_epi64_generic(
    a: &BV,
    b: &BV,
    high: bool,
    total_bits: u32,
    mask_and_src: Option<(&BV, &BV, &Solver)>,
) -> BV {
    assert!(
        total_bits == 256 || total_bits == 512,
        "total_bits must be 256 or 512"
    );

    let lane_count = (total_bits / 128) as usize;
    let mut elements = Vec::with_capacity((total_bits / 64) as usize);

    for lane_idx in 0..lane_count {
        let a_lane = extract_128b_lane(a, lane_idx);
        let b_lane = extract_128b_lane(b, lane_idx);
        let low = if high { 64 } else { 0 };
        let high_bit = low + 63;

        elements.push(a_lane.extract(high_bit, low));
        elements.push(b_lane.extract(high_bit, low));
    }

    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    let result = concat_msb_first(&reversed).simplify();

    if let Some((mask, src, solver)) = mask_and_src {
        let num_elements = (total_bits / 64) as usize;
        constrain_kmask_high_bits_zero(mask, num_elements as u32, solver);
        apply_64b_writemask(&result, src, mask, num_elements)
    } else {
        result
    }
}

// Unpack and interleave 32-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst".
// Implements __m256i _mm256_unpacklo_epi32(__m256i a, __m256i b)
pub fn mm256_unpacklo_epi32(a: &BV, b: &BV) -> BV {
    unpack_epi32_generic(a, b, false, 256, None)
}

pub fn mm256_unpackhi_epi32(a: &BV, b: &BV) -> BV {
    unpack_epi32_generic(a, b, true, 256, None)
}

// Unpack and interleave 32-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst".
// Implements __m512i _mm512_unpacklo_epi32(__m512i a, __m512i b)
pub fn mm512_unpacklo_epi32(a: &BV, b: &BV) -> BV {
    unpack_epi32_generic(a, b, false, 512, None)
}

pub fn mm512_unpackhi_epi32(a: &BV, b: &BV) -> BV {
    unpack_epi32_generic(a, b, true, 512, None)
}

// Unpack and interleave 32-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst"
// using writemask "k" (elements are copied from "src" when the corresponding mask bit is not set).
// Implements __m512i _mm512_mask_unpacklo_epi32(__m512i src, __mmask16 k, __m512i a, __m512i b)
pub fn mm512_mask_unpacklo_epi32(src: &BV, k: &BV, a: &BV, b: &BV, solver: &Solver) -> BV {
    unpack_epi32_generic(a, b, false, 512, Some((k, src, solver)))
}

pub fn mm512_mask_unpackhi_epi32(src: &BV, k: &BV, a: &BV, b: &BV, solver: &Solver) -> BV {
    unpack_epi32_generic(a, b, true, 512, Some((k, src, solver)))
}

// Unpack and interleave 64-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst".
// Implements __m256i _mm256_unpacklo_epi64(__m256i a, __m256i b)
pub fn mm256_unpacklo_epi64(a: &BV, b: &BV) -> BV {
    unpack_epi64_generic(a, b, false, 256, None)
}

pub fn mm256_unpackhi_epi64(a: &BV, b: &BV) -> BV {
    unpack_epi64_generic(a, b, true, 256, None)
}

// Unpack and interleave 64-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst".
// Implements __m512i _mm512_unpacklo_epi64(__m512i a, __m512i b)
pub fn mm512_unpacklo_epi64(a: &BV, b: &BV) -> BV {
    unpack_epi64_generic(a, b, false, 512, None)
}

pub fn mm512_unpackhi_epi64(a: &BV, b: &BV) -> BV {
    unpack_epi64_generic(a, b, true, 512, None)
}

// Unpack and interleave 64-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst"
// using writemask "k" (elements are copied from "src" when the corresponding mask bit is not set).
// Implements __m512i _mm512_mask_unpacklo_epi64(__m512i src, __mmask8 k, __m512i a, __m512i b)
pub fn mm512_mask_unpacklo_epi64(src: &BV, k: &BV, a: &BV, b: &BV, solver: &Solver) -> BV {
    unpack_epi64_generic(a, b, false, 512, Some((k, src, solver)))
}

pub fn mm512_mask_unpackhi_epi64(src: &BV, k: &BV, a: &BV, b: &BV, solver: &Solver) -> BV {
    unpack_epi64_generic(a, b, true, 512, Some((k, src, solver)))
}
