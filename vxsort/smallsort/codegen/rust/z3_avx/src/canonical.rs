use z3::Solver;
use z3::ast::BV;

pub fn constrain_high_bits_zero(value: &BV, live_bits: u32, solver: &Solver) {
    let width = value.get_size();
    assert!(
        live_bits <= width,
        "live bit count {live_bits} exceeds {width}-bit value"
    );

    if live_bits == width {
        return;
    }

    if live_bits == 0 {
        solver.assert(value.eq(BV::from_u64(0, width)));
    } else {
        let dead_width = width - live_bits;
        solver.assert(
            value
                .extract(width - 1, live_bits)
                .eq(BV::from_u64(0, dead_width)),
        );
    }
}

pub fn constrain_lane_high_bits_zero(
    value: &BV,
    lane_bits: u32,
    live_low_bits: u32,
    solver: &Solver,
) {
    let width = value.get_size();
    assert!(lane_bits > 0, "lane width must be non-zero");
    assert!(
        width.is_multiple_of(lane_bits),
        "{width}-bit value is not divisible into {lane_bits}-bit lanes"
    );
    assert!(
        live_low_bits <= lane_bits,
        "live bit count {live_low_bits} exceeds {lane_bits}-bit lane"
    );

    if live_low_bits == lane_bits {
        return;
    }

    for lane in 0..(width / lane_bits) {
        let low = lane * lane_bits;
        let high = low + lane_bits - 1;
        if live_low_bits == 0 {
            solver.assert(value.extract(high, low).eq(BV::from_u64(0, lane_bits)));
        } else {
            let dead_low = low + live_low_bits;
            let dead_width = lane_bits - live_low_bits;
            solver.assert(
                value
                    .extract(high, dead_low)
                    .eq(BV::from_u64(0, dead_width)),
            );
        }
    }
}

pub fn constrain_bits_outside_ranges_zero(value: &BV, live_ranges: &[(u32, u32)], solver: &Solver) {
    let width = value.get_size();
    let mut live = vec![false; width as usize];

    for &(low, high) in live_ranges {
        assert!(
            low <= high,
            "live range low bit {low} exceeds high bit {high}"
        );
        assert!(
            high < width,
            "live range high bit {high} exceeds {width}-bit value"
        );

        for bit in low..=high {
            live[bit as usize] = true;
        }
    }

    for bit in 0..width {
        if !live[bit as usize] {
            solver.assert(value.extract(bit, bit).eq(BV::from_u64(0, 1)));
        }
    }
}

pub fn constrain_kmask_high_bits_zero(mask: &BV, live_bits: u32, solver: &Solver) {
    constrain_high_bits_zero(mask, live_bits, solver);
}
