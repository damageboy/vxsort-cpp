use z3::ast::{Ast, BV};

use crate::lanes::{concat_msb_first, extract_element};

// Element-wise signed min or max.
//
// Z3 BitVec ``<`` is ``bvslt`` (signed), matching the hardware instructions
// (vpminsd/vpmaxsd for i32, vpminsq/vpmaxsq for i64).
//
// Args:
//     a, b: Z3 BitVecRef operands of *total_width* bits.
//     total_width: Register width in bits (256 or 512).
//     element_width: Element width in bits (32 or 64).
//     take_min: If True return element-wise min, else max.
fn generic_minmax(a: &BV, b: &BV, total_width: u32, element_width: u32, take_min: bool) -> BV {
    assert_eq!(a.get_size(), total_width);
    assert_eq!(b.get_size(), total_width);
    assert!(matches!(total_width, 256 | 512));
    assert!(matches!(element_width, 32 | 64));

    let elements: Vec<BV> = (0..total_width / element_width)
        .map(|element_idx| {
            let a_elem = extract_element(a, element_width, element_idx as usize);
            let b_elem = extract_element(b, element_width, element_idx as usize);
            let a_lt_b = a_elem.bvslt(&b_elem);

            if take_min {
                a_lt_b.ite(&a_elem, &b_elem)
            } else {
                a_lt_b.ite(&b_elem, &a_elem)
            }
        })
        .collect();

    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    concat_msb_first(&reversed).simplify()
}

// Element-wise signed minimum.
pub fn generic_min(a: &BV, b: &BV, total_width: u32, element_width: u32) -> BV {
    generic_minmax(a, b, total_width, element_width, true)
}

pub fn generic_max(a: &BV, b: &BV, total_width: u32, element_width: u32) -> BV {
    generic_minmax(a, b, total_width, element_width, false)
}

// Element-wise signed 32-bit minimum across a 256-bit register.
pub fn mm256_min_epi32(a: &BV, b: &BV) -> BV {
    generic_min(a, b, 256, 32)
}

pub fn mm256_max_epi32(a: &BV, b: &BV) -> BV {
    generic_max(a, b, 256, 32)
}

// Element-wise signed 32-bit minimum across a 512-bit register.
pub fn mm512_min_epi32(a: &BV, b: &BV) -> BV {
    generic_min(a, b, 512, 32)
}

pub fn mm512_max_epi32(a: &BV, b: &BV) -> BV {
    generic_max(a, b, 512, 32)
}

// Element-wise signed 64-bit minimum across a 256-bit register.
pub fn mm256_min_epi64(a: &BV, b: &BV) -> BV {
    generic_min(a, b, 256, 64)
}

pub fn mm256_max_epi64(a: &BV, b: &BV) -> BV {
    generic_max(a, b, 256, 64)
}

// Element-wise signed 64-bit minimum across a 512-bit register.
pub fn mm512_min_epi64(a: &BV, b: &BV) -> BV {
    generic_min(a, b, 512, 64)
}

pub fn mm512_max_epi64(a: &BV, b: &BV) -> BV {
    generic_max(a, b, 512, 64)
}
