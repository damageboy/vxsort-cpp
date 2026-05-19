use z3::Solver;
use z3::ast::{Ast, BV};

use crate::canonical::constrain_kmask_high_bits_zero;
use crate::lanes::{concat_msb_first, create_if_tree, extract_element};

// Create element selector for two-source permutation (permutex2var).
//
// Args:
//     source_a: First source register
//     source_b: Second source register
//     offset_bits: Bits specifying which element to select from the chosen source
//     source_selector: Bit specifying which source to choose from (0=a, 1=b)
//     num_elements: Number of elements in each source register
//     element_bits: Number of bits per element
//
// Returns:
//     A Z3 expression that selects the appropriate element
fn create_two_source_element_selector(
    a: &BV,
    b: &BV,
    offset_bits: &BV,
    source_selector: &BV,
    num_elements: usize,
    element_bits: u32,
) -> BV {
    let a_elements: Vec<BV> = (0..num_elements)
        .map(|element_idx| extract_element(a, element_bits, element_idx))
        .collect();
    let b_elements: Vec<BV> = (0..num_elements)
        .map(|element_idx| extract_element(b, element_bits, element_idx))
        .collect();

    let selected_a = create_if_tree(offset_bits, &a_elements);
    let selected_b = create_if_tree(offset_bits, &b_elements);

    source_selector
        .eq(BV::from_u64(0, 1))
        .ite(&selected_a, &selected_b)
        .simplify()
}

fn constrain_dead_index_bits(
    op_idx: &BV,
    element_width: u32,
    num_elements: usize,
    source_selector_bit: u32,
    solver: &Solver,
) {
    let live_bits = source_selector_bit + 1;
    if live_bits >= element_width {
        return;
    }

    let dead_width = element_width - live_bits;
    for element_idx in 0..num_elements {
        let low = element_idx as u32 * element_width + live_bits;
        let high = (element_idx as u32 + 1) * element_width - 1;
        solver.assert(op_idx.extract(high, low).eq(BV::from_u64(0, dead_width)));
    }
}

// Generic implementation for permutex2var instructions that shuffle elements from two source vectors.
//
// These instructions use an index vector where each element contains:
// - Offset bits: select which element from the chosen source
// - Source selector bit: choose between source a (0) or source b (1)
// - Optional: masking is supported for AVX512 variants.
//
// Args:
//     a: First source vector
//     op_idx: Index vector containing offsets and source selectors for each destination element
//     b: Second source vector
//     element_width: Width of each element in bits (32 or 64)
//     src: Optional source vector for masked operations (when mask bit is 0, copy from this)
//     mask: Optional predicate mask (if provided, src must also be provided)
//
// Returns:
//     Permuted vector (optionally masked)
//
// Generic Operation (for 512-bit registers, N elements, OFFSET_BITS bits, SRC_BIT position):
//     Without mask:
// ```text
//     FOR j := 0 to N-1
//         i := j * element_width
//         offset := idx[i + OFFSET_BITS - 1 : i]
//         source_sel := idx[i + SRC_BIT]
//         selected_vec := source_sel ? b : a
//         dst[i + element_width - 1 : i] := selected_vec[offset * element_width + element_width - 1 : offset * element_width]
//     ENDFOR
//     dst[MAX:512] := 0
// ```text
//
//     With mask:
// ```text
//     FOR j := 0 to N-1
//         i := j * element_width
//         offset := idx[i + OFFSET_BITS - 1 : i]
//         source_sel := idx[i + SRC_BIT]
//         IF mask[j]
//             selected_vec := source_sel ? b : a
//             dst[i + element_width - 1 : i] := selected_vec[offset * element_width + element_width - 1 : offset * element_width]
//         ELSE
//             dst[i + element_width - 1 : i] := src[i + element_width - 1 : i]
//         FI
//     ENDFOR
//     dst[MAX:512] := 0
// ```text
//
// Examples:
//     - _mm512_permutex2var_epi32: element_width=32 → 16 elements, 4 offset bits, bit 4 is source selector
//     - _mm512_permutex2var_epi64: element_width=64 → 8 elements, 3 offset bits, bit 3 is source selector
//     - _mm512_mask_permutex2var_ps: element_width=32 -> 16 elements, 4 offset bits, bit 4 is source selector, with src and mask
//     - _mm512_mask_permutex2var_pd: element_width=64 -> 8 elements, 3 offset bits, bit 3 is source selector, with src and mask
fn generic_permutex2var(
    a: &BV,
    op_idx: &BV,
    b: &BV,
    element_width: u32,
    mask_and_src: Option<(&BV, &BV)>,
    solver: &Solver,
) -> BV {
    let total_width = 512;
    let num_elements = total_width / element_width;
    assert!(
        element_width == 32 || element_width == 64,
        "permutex2var only supports 32-bit and 64-bit elements"
    );

    let offset_bits_count = (num_elements - 1).ilog2() + 1;
    let source_selector_bit = offset_bits_count;

    constrain_dead_index_bits(
        op_idx,
        element_width,
        num_elements as usize,
        source_selector_bit,
        solver,
    );
    if let Some((mask, _src)) = mask_and_src {
        constrain_kmask_high_bits_zero(mask, num_elements, solver);
    }

    let elements: Vec<BV> = (0..num_elements)
        .map(|element_idx| {
            let low = element_idx * element_width;
            let offset_bits = op_idx.extract(low + offset_bits_count - 1, low);
            let source_selector =
                op_idx.extract(low + source_selector_bit, low + source_selector_bit);
            let permuted_elem = create_two_source_element_selector(
                a,
                b,
                &offset_bits,
                &source_selector,
                num_elements as usize,
                element_width,
            );

            if let Some((mask, src)) = mask_and_src {
                let mask_bit = mask.extract(element_idx, element_idx);
                let src_elem = src.extract(low + element_width - 1, low);
                mask_bit
                    .eq(BV::from_u64(1, 1))
                    .ite(&permuted_elem, &src_elem)
                    .simplify()
            } else {
                permuted_elem
            }
        })
        .collect();

    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    concat_msb_first(&reversed).simplify()
}

// Shuffle 32-bit integers in a and b across lanes using two-source permutation.
// Implements __m512i _mm512_permutex2var_epi32 (__m512i a, __m512i idx, __m512i b)
// See _generic_permutex2var for operation details.
pub fn mm512_permutex2var_epi32(a: &BV, op_idx: &BV, b: &BV, solver: &Solver) -> BV {
    generic_permutex2var(a, op_idx, b, 32, None, solver)
}

pub fn mm512_permutex2var_epi64(a: &BV, op_idx: &BV, b: &BV, solver: &Solver) -> BV {
    generic_permutex2var(a, op_idx, b, 64, None, solver)
}

// Shuffle 32-bit integer elements in a and b across lanes using writemask.
// Implements __m512i _mm512_mask_permutex2var_epi32 (__m512i a, __mmask16 k, __m512i idx, __m512i b)
// Elements are copied from a when the corresponding mask bit is not set.
// See _generic_permutex2var for operation details.
pub fn mm512_mask_permutex2var_epi32(a: &BV, k: &BV, op_idx: &BV, b: &BV, solver: &Solver) -> BV {
    generic_permutex2var(a, op_idx, b, 32, Some((k, a)), solver)
}

// Shuffle 64-bit integer elements in a and b across lanes using writemask.
// Implements __m512i _mm512_mask_permutex2var_epi64 (__m512i a, __mmask8 k, __m512i idx, __m512i b)
// Elements are copied from a when the corresponding mask bit is not set.
// See _generic_permutex2var for operation details.
pub fn mm512_mask_permutex2var_epi64(a: &BV, k: &BV, op_idx: &BV, b: &BV, solver: &Solver) -> BV {
    generic_permutex2var(a, op_idx, b, 64, Some((k, a)), solver)
}
