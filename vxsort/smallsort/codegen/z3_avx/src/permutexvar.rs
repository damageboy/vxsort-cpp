use z3::Solver;
use z3::ast::{Ast, BV};

use crate::canonical::{constrain_kmask_high_bits_zero, constrain_lane_high_bits_zero};
use crate::lanes::{concat_msb_first, create_if_tree, extract_element};
use crate::mask::apply_writemask;

fn index_bits_needed(num_elements: usize) -> u32 {
    assert!(
        num_elements.is_power_of_two(),
        "permutexvar element count must be a power of two"
    );
    num_elements.ilog2()
}

fn select_element(source_reg: &BV, idx_bits: &BV, num_elements: usize, element_bits: u32) -> BV {
    let elements: Vec<BV> = (0..num_elements)
        .map(|element_idx| extract_element(source_reg, element_bits, element_idx))
        .collect();
    create_if_tree(idx_bits, &elements)
}

// Generic implementation for permutexvar instructions that shuffle elements across lanes.
//
// These instructions use a variable index vector to permute elements from a single source vector.
// Each element in the output is selected from the source vector based on the corresponding
// index value in the index vector. Optional masking is supported for AVX512 variants.
//
// Args:
//     a: Source vector to permute
//     op_idx: Index vector containing the indices for each destination element
//     total_width: Total bit width of the vectors (256 or 512)
//     element_width: Width of each element in bits (32 or 64)
//     src: Optional source vector for masked operations (values used when mask bit is 0)
//     mask: Optional predicate mask (if provided, src must also be provided)
//
// Returns:
//     Permuted vector (optionally masked)
//
// Generic Operation (where N = total_width / element_width, IDX_BITS = log2(N)):
//     Without mask:
// ```text
//     FOR j := 0 to N-1
//         i := j * element_width
//         index := op_idx[i + IDX_BITS - 1 : i]
//         dst[i + element_width - 1 : i] := a[index * element_width + element_width - 1 : index * element_width]
//     ENDFOR
//     dst[MAX:total_width] := 0
// ```text
//
//     With mask:
// ```text
//     FOR j := 0 to N-1
//         i := j * element_width
//         index := op_idx[i + IDX_BITS - 1 : i]
//         IF mask[j]
//             dst[i + element_width - 1 : i] := a[index * element_width + element_width - 1 : index * element_width]
//         ELSE
//             dst[i + element_width - 1 : i] := src[i + element_width - 1 : i]
//         FI
//     ENDFOR
//     dst[MAX:total_width] := 0
// ```text
//
// Examples:
//     - _mm256_permutexvar_epi32: total_width=256, element_width=32 → 8 elements, 3 index bits
//     - _mm512_permutexvar_epi32: total_width=512, element_width=32 → 16 elements, 4 index bits
//     - _mm256_permutexvar_epi64: total_width=256, element_width=64 → 4 elements, 2 index bits
//     - _mm512_permutexvar_epi64: total_width=512, element_width=64 → 8 elements, 3 index bits
//     - _mm512_mask_permutexvar_epi32: total_width=512, element_width=32, with src and mask
//     - _mm512_mask_permutexvar_epi64: total_width=512, element_width=64, with src and mask
fn permutexvar_generic(
    a: &BV,
    op_idx: &BV,
    total_width: u32,
    element_width: u32,
    solver: &Solver,
    mask_and_src: Option<(&BV, &BV)>,
) -> BV {
    assert_eq!(a.get_size(), total_width);
    assert_eq!(op_idx.get_size(), total_width);
    assert_eq!(total_width % element_width, 0);

    let num_elements = (total_width / element_width) as usize;
    let live_index_bits = index_bits_needed(num_elements);

    constrain_lane_high_bits_zero(op_idx, element_width, live_index_bits, solver);

    let elements: Vec<BV> = (0..num_elements)
        .map(|dst_idx| {
            let low = dst_idx as u32 * element_width;
            let idx_bits = op_idx.extract(low + live_index_bits - 1, low);
            select_element(a, &idx_bits, num_elements, element_width)
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

// Shuffle 32-bit integers across lanes in a 256-bit vector.
// Implements __m256i _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx)
// See _generic_permutexvar for operation details.
pub fn mm256_permutexvar_epi32(a: &BV, op_idx: &BV, solver: &Solver) -> BV {
    permutexvar_generic(a, op_idx, 256, 32, solver, None)
}

pub fn mm512_permutexvar_epi32(a: &BV, op_idx: &BV, solver: &Solver) -> BV {
    permutexvar_generic(a, op_idx, 512, 32, solver, None)
}

// Shuffle 32-bit integers across lanes in a 512-bit vector using writemask.
// Implements __m512i _mm512_mask_permutexvar_epi32 (__m512i src, __mmask16 k, __m512i idx, __m512i a)
// Elements are copied from src when the corresponding mask bit is not set.
// See _generic_permutexvar for operation details.
pub fn mm512_mask_permutexvar_epi32(src: &BV, k: &BV, op_idx: &BV, a: &BV, solver: &Solver) -> BV {
    permutexvar_generic(a, op_idx, 512, 32, solver, Some((k, src)))
}

// Shuffle 64-bit integers across lanes in a 256-bit vector.
// Implements __m256i _mm256_permutexvar_epi64 (__m256i idx, __m256i a)
// See _generic_permutexvar for operation details.
pub fn mm256_permutexvar_epi64(a: &BV, op_idx: &BV, solver: &Solver) -> BV {
    permutexvar_generic(a, op_idx, 256, 64, solver, None)
}

pub fn mm512_permutexvar_epi64(a: &BV, op_idx: &BV, solver: &Solver) -> BV {
    permutexvar_generic(a, op_idx, 512, 64, solver, None)
}

// Shuffle 64-bit integers across lanes in a 512-bit vector using writemask.
// Implements __m512i _mm512_mask_permutexvar_epi64 (__m512i src, __mmask8 k, __m512i idx, __m512i a)
// Elements are copied from src when the corresponding mask bit is not set.
// See _generic_permutexvar for operation details.
pub fn mm512_mask_permutexvar_epi64(src: &BV, k: &BV, op_idx: &BV, a: &BV, solver: &Solver) -> BV {
    permutexvar_generic(a, op_idx, 512, 64, solver, Some((k, src)))
}
