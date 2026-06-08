use z3::ast::{Ast, BV};

use crate::lanes::concat_msb_first;

pub fn apply_writemask(
    result: &BV,
    src: &BV,
    mask: &BV,
    element_bits: u32,
    num_elements: usize,
) -> BV {
    let elements: Vec<BV> = (0..num_elements)
        .map(|element_idx| {
            let low = element_idx as u32 * element_bits;
            let mask_bit = mask.extract(element_idx as u32, element_idx as u32);
            let active_elem = result.extract(low + element_bits - 1, low);
            let src_elem = src.extract(low + element_bits - 1, low);
            mask_bit
                .eq(BV::from_u64(1, 1))
                .ite(&active_elem, &src_elem)
                .simplify()
        })
        .collect();
    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    concat_msb_first(&reversed).simplify()
}

pub fn apply_32b_writemask(result: &BV, src: &BV, mask: &BV, num_elements: usize) -> BV {
    apply_writemask(result, src, mask, 32, num_elements)
}

pub fn apply_64b_writemask(result: &BV, src: &BV, mask: &BV, num_elements: usize) -> BV {
    apply_writemask(result, src, mask, 64, num_elements)
}
