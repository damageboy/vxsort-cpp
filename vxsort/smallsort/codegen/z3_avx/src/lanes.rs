use z3::ast::{Ast, BV};

pub fn concat_msb_first(elements: &[BV]) -> BV {
    assert!(!elements.is_empty(), "cannot concatenate zero elements");

    let mut iter = elements.iter();
    let first = iter.next().expect("checked non-empty").clone();
    iter.fold(first, |acc, element| acc.concat(element))
        .simplify()
}

pub fn extract_element(reg: &BV, element_bits: u32, element_idx: usize) -> BV {
    let start_bit = element_idx as u32 * element_bits;
    let end_bit = start_bit + element_bits - 1;
    reg.extract(end_bit, start_bit)
}

pub fn extract_128b_lane(reg: &BV, lane_idx: usize) -> BV {
    let start_bit = lane_idx as u32 * 128;
    let end_bit = start_bit + 127;
    reg.extract(end_bit, start_bit)
}

// Create nested If statements for element selection.
// Create a balanced tree of If statements for element selection.
//
// Args:
//     source_reg: The source register to select elements from
//     idx_bits: The index bits extracted from the index register
//     num_elements: Number of elements to choose from (2, 4, 8, 16)
//     element_bits: Number of bits per element (32 or 64)
//
// Returns:
//     A Z3 expression that selects the appropriate element based on idx_bits
pub fn create_if_tree(idx_bits: &BV, elements: &[BV]) -> BV {
    assert!(!elements.is_empty(), "cannot select from zero elements");

    let mut result = elements.last().expect("checked non-empty").clone();
    for i in (0..elements.len() - 1).rev() {
        let condition = idx_bits.eq(BV::from_u64(i as u64, idx_bits.get_size()));
        result = condition.ite(&elements[i], &result);
    }

    result.simplify()
}
