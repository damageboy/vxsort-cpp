use z3::Solver;
use z3::ast::{Ast, BV};

use crate::lanes::{concat_msb_first, extract_element};

pub fn ymm_reg(name: &str) -> BV {
    BV::new_const(name, 256)
}

pub fn zmm_reg(name: &str) -> BV {
    BV::new_const(name, 512)
}

/// Build a register from fresh symbolic lane variables and constrain those
/// lanes to concrete values in `solver`.
///
/// This mirrors the Python helper deliberately: using constrained named lanes,
/// rather than a single packed constant, keeps lane-level symbols available in
/// models and preserves the symbolic style used by the synthesis tests.
pub fn reg_with_values(
    name: &str,
    solver: &Solver,
    raw_values: &[u64],
    element_bits: u32,
    total_bits: u32,
) -> BV {
    let lanes = total_bits / element_bits;
    assert_eq!(
        raw_values.len(),
        lanes as usize,
        "expected {lanes} values for {element_bits}-bit elements in {total_bits}-bit register, got {}",
        raw_values.len()
    );

    let elements: Vec<BV> = raw_values
        .iter()
        .enumerate()
        .map(|(lane, raw_value)| {
            let element = BV::new_const(format!("{name}_l_{lane:02}"), element_bits);
            solver.assert(element.eq(BV::from_u64(*raw_value, element_bits)));
            element
        })
        .collect();

    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    concat_msb_first(&reversed).simplify()
}

pub fn ymm_reg_with_32b_values(name: &str, solver: &Solver, raw_values: &[u64]) -> BV {
    reg_with_values(name, solver, raw_values, 32, 256)
}

pub fn zmm_reg_with_32b_values(name: &str, solver: &Solver, raw_values: &[u64]) -> BV {
    reg_with_values(name, solver, raw_values, 32, 512)
}

pub fn ymm_reg_with_64b_values(name: &str, solver: &Solver, raw_values: &[u64]) -> BV {
    reg_with_values(name, solver, raw_values, 64, 256)
}

pub fn zmm_reg_with_64b_values(name: &str, solver: &Solver, raw_values: &[u64]) -> BV {
    reg_with_values(name, solver, raw_values, 64, 512)
}

pub fn construct_reg_from_elements(
    bits: u32,
    element_specs: &[(&BV, usize)],
    total_bits: u32,
) -> BV {
    let lanes = total_bits / bits;
    assert_eq!(
        element_specs.len(),
        lanes as usize,
        "expected {lanes} element specs for {bits}-bit elements in {total_bits}-bit register, got {}",
        element_specs.len()
    );

    let elements: Vec<BV> = element_specs
        .iter()
        .map(|(reg, element_idx)| {
            assert!(
                *element_idx < lanes as usize,
                "element index {element_idx} out of range for {bits}-bit elements"
            );
            extract_element(reg, bits, *element_idx)
        })
        .collect();
    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    concat_msb_first(&reversed).simplify()
}

pub fn construct_ymm_reg_from_elements(bits: u32, element_specs: &[(&BV, usize)]) -> BV {
    construct_reg_from_elements(bits, element_specs, 256)
}

pub fn construct_zmm_reg_from_elements(bits: u32, element_specs: &[(&BV, usize)]) -> BV {
    construct_reg_from_elements(bits, element_specs, 512)
}

// Create a register with given number of lanes and element width, ensuring each lane is unique.
pub fn reg_with_unique_values(name: &str, solver: &Solver, lanes: u32, bits: u32) -> BV {
    let total_bits = lanes * bits;
    assert!(
        total_bits == 256 || total_bits == 512,
        "total register size can only be 256 or 512 bits"
    );

    let reg = if total_bits == 256 {
        ymm_reg(name)
    } else {
        zmm_reg(name)
    };

    let elems: Vec<BV> = (0..lanes)
        .map(|lane| extract_element(&reg, bits, lane as usize))
        .collect();

    for i in 0..lanes as usize {
        for j in (i + 1)..lanes as usize {
            solver.assert(elems[i].eq(&elems[j]).not());
        }
    }

    reg
}

pub fn ymm_reg_with_unique_values(name: &str, solver: &Solver, bits: u32) -> BV {
    reg_with_unique_values(name, solver, 256 / bits, bits)
}

pub fn zmm_reg_with_unique_values(name: &str, solver: &Solver, bits: u32) -> BV {
    reg_with_unique_values(name, solver, 512 / bits, bits)
}

pub fn reg_reversed(name: &str, solver: &Solver, original_reg: &BV, lanes: u32, bits: u32) -> BV {
    let total_bits = lanes * bits;
    assert!(
        total_bits == 256 || total_bits == 512,
        "total register size can only be 256 or 512 bits"
    );

    let reversed_reg = if total_bits == 256 {
        ymm_reg(name)
    } else {
        zmm_reg(name)
    };

    for lane in 0..lanes {
        let rev_elem = extract_element(&reversed_reg, bits, lane as usize);
        let orig_elem = extract_element(original_reg, bits, (lanes - 1 - lane) as usize);
        solver.assert(rev_elem.eq(orig_elem));
    }

    reversed_reg
}

// Create a YMM register that is the reverse of the original register through constraints.
pub fn ymm_reg_reversed(name: &str, solver: &Solver, original_reg: &BV, bits: u32) -> BV {
    reg_reversed(name, solver, original_reg, 256 / bits, bits)
}

pub fn zmm_reg_reversed(name: &str, solver: &Solver, original_reg: &BV, bits: u32) -> BV {
    reg_reversed(name, solver, original_reg, 512 / bits, bits)
}
