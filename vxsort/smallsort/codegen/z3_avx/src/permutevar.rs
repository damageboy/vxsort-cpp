use z3::Solver;
use z3::ast::{Ast, BV};

use crate::canonical::constrain_kmask_high_bits_zero;
use crate::lanes::{concat_msb_first, create_if_tree, extract_128b_lane};
use crate::mask::{apply_32b_writemask, apply_64b_writemask};

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

// Selects a 64-bit element from a 128-bit vector based on a 1-bit control.
fn select2_pd(src_128: &BV, select: &BV) -> BV {
    let elements = [src_128.extract(63, 0), src_128.extract(127, 64)];
    create_if_tree(select, &elements)
}

fn constrain_zero(solver: &Solver, value: &BV) {
    solver.assert(value.eq(BV::from_u64(0, value.get_size())));
}

fn constrain_permutevar_controls_zero_dead_bits(
    control: &BV,
    element_bits: u32,
    num_elements: usize,
    solver: &Solver,
) {
    for element_idx in 0..num_elements {
        let base = element_idx as u32 * element_bits;
        match element_bits {
            32 => constrain_zero(solver, &control.extract(base + 31, base + 2)),
            64 => {
                constrain_zero(solver, &control.extract(base, base));
                constrain_zero(solver, &control.extract(base + 63, base + 2));
            }
            _ => panic!("unsupported permutevar element width: {element_bits}"),
        }
    }
}

// Generic implementation for permutevar instructions that shuffle elements within 128-bit lanes.
//
// These instructions use a variable index vector to permute elements within each 128-bit lane.
// Each element in the output is selected from the corresponding 128-bit lane based on control bits
// in the index vector. Optional masking is supported for AVX512 variants.
//
// Args:
//     a: Source vector to permute
//     b: Control/index vector containing the control bits for each destination element
//     total_width: Total bit width of the vectors (256 or 512)
//     element_width: Width of each element in bits (32 for ps, 64 for pd)
//     k: Optional predicate mask (if provided, src must also be provided)
//     src: Optional source vector for masked operations (values used when mask bit is 0)
//
// Returns:
//     Permuted vector (optionally masked)
//
// Generic Operation (where N = total_width / element_width, LANE_ELEMENTS = 128 / element_width):
//
//     For element_width=32 (ps - single precision):
//         - 4 elements per 128-bit lane
//         - Uses 2 control bits per element: b[i+1:i] where i = element_index * 32
//
//     For element_width=64 (pd - double precision):
//         - 2 elements per 128-bit lane
//         - Uses 1 control bit per element at specific positions:
//           b[1], b[65], b[129], b[193] for 256-bit (4 elements)
//           b[1], b[65], b[129], b[193], b[257], b[321], b[385], b[449] for 512-bit (8 elements)
//
//     Without mask:
// ```text
//     FOR j := 0 to N-1
//         lane_idx := j / LANE_ELEMENTS
//         lane := a[lane_idx*128+127 : lane_idx*128]
//         control_bits := extract_control_bits(b, j, element_width)
//         dst[j*element_width+element_width-1 : j*element_width] := SELECT(lane, control_bits)
//     ENDFOR
//     dst[MAX:total_width] := 0
// ```text
//
//     With mask:
// ```text
//     FOR j := 0 to N-1
//         lane_idx := j / LANE_ELEMENTS
//         lane := a[lane_idx*128+127 : lane_idx*128]
//         control_bits := extract_control_bits(b, j, element_width)
//         tmp_elem := SELECT(lane, control_bits)
//         IF k[j]
//             dst[j*element_width+element_width-1 : j*element_width] := tmp_elem
//         ELSE
//             dst[j*element_width+element_width-1 : j*element_width] := src[j*element_width+element_width-1 : j*element_width]
//         FI
//     ENDFOR
//     dst[MAX:total_width] := 0
// ```text
//
// Examples:
//     - _mm256_permutevar_ps: total_width=256, element_width=32 → 8 elements, 2 lanes
//     - _mm512_permutevar_ps: total_width=512, element_width=32 → 16 elements, 4 lanes
//     - _mm256_permutevar_pd: total_width=256, element_width=64 → 4 elements, 2 lanes
//     - _mm512_permutevar_pd: total_width=512, element_width=64 → 8 elements, 4 lanes
//     - _mm512_mask_permutevar_ps: total_width=512, element_width=32, with src and mask
//     - _mm512_mask_permutevar_pd: total_width=512, element_width=64, with src and mask
fn permutevar_generic(
    a: &BV,
    control: &BV,
    total_bits: u32,
    element_bits: u32,
    mask_and_src: Option<(&BV, &BV)>,
    solver: &Solver,
) -> BV {
    let num_elements = (total_bits / element_bits) as usize;
    let elements_per_lane = 128 / element_bits;

    constrain_permutevar_controls_zero_dead_bits(control, element_bits, num_elements, solver);

    let chunks: Vec<BV> = (0..num_elements)
        .map(|element_idx| {
            let bit_start = element_idx as u32 * element_bits;
            let lane_idx = element_idx as u32 / elements_per_lane;
            let lane = extract_128b_lane(a, lane_idx as usize);

            match element_bits {
                32 => {
                    let selector = control.extract(bit_start + 1, bit_start);
                    select4_ps(&lane, &selector)
                }
                64 => {
                    let selector = control.extract(bit_start + 1, bit_start + 1);
                    select2_pd(&lane, &selector)
                }
                _ => panic!("unsupported permutevar element width: {element_bits}"),
            }
        })
        .collect();
    let reversed: Vec<BV> = chunks.into_iter().rev().collect();
    let result = concat_msb_first(&reversed).simplify();

    if let Some((mask, src)) = mask_and_src {
        constrain_kmask_high_bits_zero(mask, num_elements as u32, solver);
        match element_bits {
            32 => apply_32b_writemask(&result, src, mask, num_elements),
            64 => apply_64b_writemask(&result, src, mask, num_elements),
            _ => panic!("unsupported permutevar element width: {element_bits}"),
        }
    } else {
        result
    }
}

// Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in b.
// Implements __m256 _mm256_permutevar_ps (__m256 a, __m256i b)
pub fn mm256_permutevar_ps(a: &BV, control: &BV, solver: &Solver) -> BV {
    permutevar_generic(a, control, 256, 32, None, solver)
}

pub fn mm512_permutevar_ps(a: &BV, control: &BV, solver: &Solver) -> BV {
    permutevar_generic(a, control, 512, 32, None, solver)
}

// Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in b,
// and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
// Implements __m512 _mm512_mask_permutevar_ps (__m512 src, __mmask16 k, __m512 a, __m512i b)
pub fn mm512_mask_permutevar_ps(src: &BV, k: &BV, a: &BV, control: &BV, solver: &Solver) -> BV {
    permutevar_generic(a, control, 512, 32, Some((k, src)), solver)
}

pub fn mm256_permutevar_pd(a: &BV, control: &BV, solver: &Solver) -> BV {
    permutevar_generic(a, control, 256, 64, None, solver)
}

// Shuffle double-precision (64-bit) floating-point elements in a within 128-bit lanes using the control in b.
// Implements __m512d _mm512_permutevar_pd (__m512d a, __m512i b)
pub fn mm512_permutevar_pd(a: &BV, control: &BV, solver: &Solver) -> BV {
    permutevar_generic(a, control, 512, 64, None, solver)
}

pub fn mm512_mask_permutevar_pd(src: &BV, k: &BV, a: &BV, control: &BV, solver: &Solver) -> BV {
    permutevar_generic(a, control, 512, 64, Some((k, src)), solver)
}
