use comfy_table::presets::ASCII_FULL_CONDENSED;
use comfy_table::{Cell, Table};
use z3::ast::BV;
use z3::{SatResult, Solver};

use crate::alignr::{
    mm256_alignr_epi8, mm256_alignr_epi32, mm256_alignr_epi64, mm512_mask_alignr_epi64,
};
use crate::blend::{mm256_blend_pd, mm256_blend_ps, mm256_blendv_pd, mm256_blendv_ps};
use crate::control::Imm8;
use crate::lanes::concat_msb_first;
use crate::minmax::{mm256_min_epi32, mm256_min_epi64};
use crate::permute::{mm256_permute_ps, mm512_permute_ps};
use crate::permute_pd::{mm256_permute_pd, mm512_permute_pd};
use crate::permute2x128::mm256_permute2x128_si256;
use crate::permute4x64::mm256_permute4x64_epi64;
use crate::permutevar::{mm256_permutevar_pd, mm256_permutevar_ps};
use crate::permutex2var::{mm512_mask_permutex2var_epi64, mm512_permutex2var_epi32};
use crate::permutexvar::{mm256_permutexvar_epi32, mm256_permutexvar_epi64};
use crate::registers::{
    construct_ymm_reg_from_elements, construct_zmm_reg_from_elements, reg_with_unique_values,
    ymm_reg, ymm_reg_reversed, ymm_reg_with_32b_values, ymm_reg_with_64b_values,
    ymm_reg_with_unique_values, zmm_reg, zmm_reg_with_64b_values, zmm_reg_with_unique_values,
};
use crate::shuffle_i32x4::mm512_shuffle_i32x4;
use crate::shuffle_pd::mm256_shuffle_pd;
use crate::shuffle_ps::{mm256_shuffle_ps, mm512_shuffle_ps};
use crate::unpack::{mm256_unpacklo_epi32, mm512_mask_unpackhi_epi32};

pub struct SolveBenchmarkResult {
    pub sat: SatResult,
}

pub struct SolveBenchmarkCase {
    pub name: &'static str,
    pub group: SolveBenchmarkGroup,
    pub family: &'static str,
    pub solve: fn() -> SolveBenchmarkResult,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolveBenchmarkGroup {
    Ymm256,
    Zmm512,
}

impl SolveBenchmarkGroup {
    pub fn id(self) -> &'static str {
        match self {
            SolveBenchmarkGroup::Ymm256 => "solve_intrinsics_256bit",
            SolveBenchmarkGroup::Zmm512 => "solve_intrinsics_512bit",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SolveBenchmarkGroup::Ymm256 => "256-bit",
            SolveBenchmarkGroup::Zmm512 => "512-bit",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SummaryFormat {
    Markdown,
    Ascii,
}

#[derive(Clone, Debug)]
pub struct SolveBenchmarkSummaryRow {
    pub name: String,
    pub group: String,
    pub family: String,
    pub avg_ms: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SolveBenchmarkSummaryArgs {
    pub summary_format: SummaryFormat,
    pub summary_iterations: u32,
    pub has_summary_args: bool,
    pub criterion_args: Vec<String>,
}

pub fn solve_benchmark_cases() -> &'static [SolveBenchmarkCase] {
    &SOLVE_BENCHMARK_CASES
}

pub fn solve_benchmark_groups() -> &'static [SolveBenchmarkGroup] {
    &SOLVE_BENCHMARK_GROUPS
}

pub fn parse_solve_benchmark_summary_args<I>(args: I) -> Result<SolveBenchmarkSummaryArgs, String>
where
    I: IntoIterator<Item = String>,
{
    let mut summary_format = SummaryFormat::Markdown;
    let mut summary_iterations = 3;
    let mut has_summary_args = false;
    let mut criterion_args = Vec::new();
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        if arg == "--summary-format" {
            let value = args
                .next()
                .ok_or_else(|| "--summary-format requires a value".to_owned())?;
            summary_format = parse_summary_format(&value)?;
            has_summary_args = true;
        } else if let Some(value) = arg.strip_prefix("--summary-format=") {
            summary_format = parse_summary_format(value)?;
            has_summary_args = true;
        } else if arg == "--summary-iterations" {
            let value = args
                .next()
                .ok_or_else(|| "--summary-iterations requires a value".to_owned())?;
            summary_iterations = parse_summary_iterations(&value)?;
            has_summary_args = true;
        } else if let Some(value) = arg.strip_prefix("--summary-iterations=") {
            summary_iterations = parse_summary_iterations(value)?;
            has_summary_args = true;
        } else {
            criterion_args.push(arg);
        }
    }

    Ok(SolveBenchmarkSummaryArgs {
        summary_format,
        summary_iterations,
        has_summary_args,
        criterion_args,
    })
}

fn parse_summary_format(value: &str) -> Result<SummaryFormat, String> {
    if value.eq_ignore_ascii_case("ascii") || value.eq_ignore_ascii_case("pretty") {
        Ok(SummaryFormat::Ascii)
    } else if value.eq_ignore_ascii_case("markdown") || value.eq_ignore_ascii_case("md") {
        Ok(SummaryFormat::Markdown)
    } else {
        Err(format!(
            "unknown summary format '{value}', expected markdown/md or ascii/pretty"
        ))
    }
}

fn parse_summary_iterations(value: &str) -> Result<u32, String> {
    let iterations = value
        .parse::<u32>()
        .map_err(|_| format!("invalid summary iteration count '{value}'"))?;
    if iterations == 0 {
        return Err("--summary-iterations must be at least 1".to_owned());
    }
    Ok(iterations)
}

pub fn format_solve_benchmark_summary(
    iterations: u32,
    rows: &[SolveBenchmarkSummaryRow],
    format: SummaryFormat,
) -> String {
    match format {
        SummaryFormat::Markdown => format_markdown_summary(iterations, rows),
        SummaryFormat::Ascii => format_ascii_summary(iterations, rows),
    }
}

fn summary_title(iterations: u32) -> String {
    format!("z3_avx solve benchmark summary (build + check, {iterations} iteration average)")
}

fn format_markdown_summary(iterations: u32, rows: &[SolveBenchmarkSummaryRow]) -> String {
    let mut output = String::new();
    output.push_str(&summary_title(iterations));
    output.push('\n');
    output.push_str("| intrinsic solve case | group | family | avg ms |\n");
    output.push_str("|---|---:|---:|---:|\n");
    for row in rows {
        output.push_str(&format!(
            "| `{}` | `{}` | `{}` | {:.3} |\n",
            row.name, row.group, row.family, row.avg_ms
        ));
    }
    output
}

fn format_ascii_summary(iterations: u32, rows: &[SolveBenchmarkSummaryRow]) -> String {
    let mut table = Table::new();
    table.load_preset(ASCII_FULL_CONDENSED);
    table.set_header(vec!["intrinsic solve case", "group", "family", "avg ms"]);
    for row in rows {
        table.add_row(vec![
            Cell::new(&row.name),
            Cell::new(&row.group),
            Cell::new(&row.family),
            Cell::new(format!("{:.3}", row.avg_ms)),
        ]);
    }

    format!("{}\n{}", summary_title(iterations), table)
}

const SOLVE_BENCHMARK_GROUPS: [SolveBenchmarkGroup; 2] =
    [SolveBenchmarkGroup::Ymm256, SolveBenchmarkGroup::Zmm512];

const SOLVE_BENCHMARK_CASES: [SolveBenchmarkCase; 28] = [
    SolveBenchmarkCase {
        name: "mm256_permute_ps/imm8_identity",
        group: SolveBenchmarkGroup::Ymm256,
        family: "permute_ps",
        solve: solve_mm256_permute_ps_imm8_identity,
    },
    SolveBenchmarkCase {
        name: "mm512_permute_ps/imm8_identity",
        group: SolveBenchmarkGroup::Zmm512,
        family: "permute_ps",
        solve: solve_mm512_permute_ps_imm8_identity,
    },
    SolveBenchmarkCase {
        name: "mm256_permute_pd/imm8_identity",
        group: SolveBenchmarkGroup::Ymm256,
        family: "permute_pd",
        solve: solve_mm256_permute_pd_imm8_identity,
    },
    SolveBenchmarkCase {
        name: "mm512_permute_pd/imm8_identity",
        group: SolveBenchmarkGroup::Zmm512,
        family: "permute_pd",
        solve: solve_mm512_permute_pd_imm8_identity,
    },
    SolveBenchmarkCase {
        name: "mm256_shuffle_ps/imm8_two_vec",
        group: SolveBenchmarkGroup::Ymm256,
        family: "shuffle_ps",
        solve: solve_mm256_shuffle_ps_imm8_two_vec,
    },
    SolveBenchmarkCase {
        name: "mm512_shuffle_ps/imm8_two_vec",
        group: SolveBenchmarkGroup::Zmm512,
        family: "shuffle_ps",
        solve: solve_mm512_shuffle_ps_imm8_two_vec,
    },
    SolveBenchmarkCase {
        name: "mm256_shuffle_pd/imm8_two_vec",
        group: SolveBenchmarkGroup::Ymm256,
        family: "shuffle_pd",
        solve: solve_mm256_shuffle_pd_imm8_two_vec,
    },
    SolveBenchmarkCase {
        name: "mm256_permute2x128/imm8_identity",
        group: SolveBenchmarkGroup::Ymm256,
        family: "permute2x128",
        solve: solve_mm256_permute2x128_imm8_identity,
    },
    SolveBenchmarkCase {
        name: "mm512_shuffle_i32x4/imm8_identity",
        group: SolveBenchmarkGroup::Zmm512,
        family: "shuffle_i32x4",
        solve: solve_mm512_shuffle_i32x4_imm8_identity,
    },
    SolveBenchmarkCase {
        name: "mm256_permute4x64_epi64/imm8_reverse",
        group: SolveBenchmarkGroup::Ymm256,
        family: "permute4x64",
        solve: solve_mm256_permute4x64_imm8_reverse,
    },
    SolveBenchmarkCase {
        name: "mm256_blend_ps/imm8_first_half",
        group: SolveBenchmarkGroup::Ymm256,
        family: "blend",
        solve: solve_mm256_blend_ps_imm8_first_half,
    },
    SolveBenchmarkCase {
        name: "mm256_blend_pd/imm8_alternating",
        group: SolveBenchmarkGroup::Ymm256,
        family: "blend",
        solve: solve_mm256_blend_pd_imm8_alternating,
    },
    SolveBenchmarkCase {
        name: "mm256_blendv_ps/mask_signs",
        group: SolveBenchmarkGroup::Ymm256,
        family: "blendv",
        solve: solve_mm256_blendv_ps_mask_signs,
    },
    SolveBenchmarkCase {
        name: "mm256_blendv_pd/mask_signs",
        group: SolveBenchmarkGroup::Ymm256,
        family: "blendv",
        solve: solve_mm256_blendv_pd_mask_signs,
    },
    SolveBenchmarkCase {
        name: "mm256_alignr_epi8/imm8_shift",
        group: SolveBenchmarkGroup::Ymm256,
        family: "alignr",
        solve: solve_mm256_alignr_epi8_imm8_shift,
    },
    SolveBenchmarkCase {
        name: "mm256_alignr_epi32/imm8_shift",
        group: SolveBenchmarkGroup::Ymm256,
        family: "alignr",
        solve: solve_mm256_alignr_epi32_imm8_shift,
    },
    SolveBenchmarkCase {
        name: "mm256_alignr_epi64/imm8_shift",
        group: SolveBenchmarkGroup::Ymm256,
        family: "alignr",
        solve: solve_mm256_alignr_epi64_imm8_shift,
    },
    SolveBenchmarkCase {
        name: "mm512_mask_alignr_epi64/imm8_and_mask",
        group: SolveBenchmarkGroup::Zmm512,
        family: "alignr",
        solve: solve_mm512_mask_alignr_epi64_imm8_and_mask,
    },
    SolveBenchmarkCase {
        name: "mm256_permutexvar_epi32/index_vector",
        group: SolveBenchmarkGroup::Ymm256,
        family: "permutexvar",
        solve: solve_mm256_permutexvar_epi32_indices,
    },
    SolveBenchmarkCase {
        name: "mm256_permutexvar_epi64/index_vector",
        group: SolveBenchmarkGroup::Ymm256,
        family: "permutexvar",
        solve: solve_mm256_permutexvar_epi64_indices,
    },
    SolveBenchmarkCase {
        name: "mm512_permutex2var_epi32/index_vector",
        group: SolveBenchmarkGroup::Zmm512,
        family: "permutex2var",
        solve: solve_mm512_permutex2var_epi32_indices,
    },
    SolveBenchmarkCase {
        name: "mm512_mask_permutex2var_epi64/index_vector_and_mask",
        group: SolveBenchmarkGroup::Zmm512,
        family: "permutex2var",
        solve: solve_mm512_mask_permutex2var_epi64_indices_and_mask,
    },
    SolveBenchmarkCase {
        name: "mm256_permutevar_ps/control_vector",
        group: SolveBenchmarkGroup::Ymm256,
        family: "permutevar",
        solve: solve_mm256_permutevar_ps_controls,
    },
    SolveBenchmarkCase {
        name: "mm256_permutevar_pd/control_vector",
        group: SolveBenchmarkGroup::Ymm256,
        family: "permutevar",
        solve: solve_mm256_permutevar_pd_controls,
    },
    SolveBenchmarkCase {
        name: "mm512_mask_unpackhi_epi32/mask",
        group: SolveBenchmarkGroup::Zmm512,
        family: "unpack",
        solve: solve_mm512_mask_unpackhi_epi32_mask,
    },
    SolveBenchmarkCase {
        name: "mm256_unpacklo_epi32/input_values",
        group: SolveBenchmarkGroup::Ymm256,
        family: "unpack",
        solve: solve_mm256_unpacklo_epi32_inputs,
    },
    SolveBenchmarkCase {
        name: "mm256_min_epi32/input_values",
        group: SolveBenchmarkGroup::Ymm256,
        family: "minmax",
        solve: solve_mm256_min_epi32_inputs,
    },
    SolveBenchmarkCase {
        name: "mm256_min_epi64/input_values",
        group: SolveBenchmarkGroup::Ymm256,
        family: "minmax",
        solve: solve_mm256_min_epi64_inputs,
    },
];

fn result(sat: SatResult) -> SolveBenchmarkResult {
    SolveBenchmarkResult { sat }
}

fn unique_zmm_pair(prefix: &str, solver: &Solver, bits: u32) -> (BV, BV) {
    let a = zmm_reg_with_unique_values(&format!("{prefix}_a"), solver, bits);
    let b = zmm_reg_with_unique_values(&format!("{prefix}_b"), solver, bits);
    let lanes = 512 / bits;

    for a_lane in 0..lanes {
        let a_low = a_lane * bits;
        let a_elem = a.extract(a_low + bits - 1, a_low);
        for b_lane in 0..lanes {
            let b_low = b_lane * bits;
            let b_elem = b.extract(b_low + bits - 1, b_low);
            solver.assert(a_elem.eq(&b_elem).not());
        }
    }

    (a, b)
}

fn packed_32(values: &[u64], total_bits: u32) -> BV {
    let elements: Vec<BV> = values
        .iter()
        .map(|value| BV::from_u64(*value, 32))
        .collect();
    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    assert_eq!(values.len(), (total_bits / 32) as usize);
    concat_msb_first(&reversed)
}

fn packed_64(values: &[u64], total_bits: u32) -> BV {
    let elements: Vec<BV> = values
        .iter()
        .map(|value| BV::from_u64(*value, 64))
        .collect();
    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    assert_eq!(values.len(), (total_bits / 64) as usize);
    concat_msb_first(&reversed)
}

fn solve_mm256_permute_ps_imm8_identity() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("bench_permute_ps_ymm", &solver, 32);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm256_permute_ps(&input, Imm8::Expr(imm8));
    solver.assert(output.eq(input));
    result(solver.check())
}

fn solve_mm512_permute_ps_imm8_identity() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let input = zmm_reg_with_unique_values("bench_permute_ps_zmm", &solver, 32);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm512_permute_ps(&input, Imm8::Expr(imm8));
    solver.assert(output.eq(input));
    result(solver.check())
}

fn solve_mm256_permute_pd_imm8_identity() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("bench_permute_pd_ymm", &solver, 64);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm256_permute_pd(&input, Imm8::Expr(imm8), &solver);
    solver.assert(output.eq(input));
    result(solver.check())
}

fn solve_mm512_permute_pd_imm8_identity() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let input = zmm_reg_with_unique_values("bench_permute_pd_zmm", &solver, 64);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm512_permute_pd(&input, Imm8::Expr(imm8), &solver);
    solver.assert(output.eq(input));
    result(solver.check())
}

fn solve_mm256_shuffle_ps_imm8_two_vec() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("bench_shuffle_ps_a", &solver, 32);
    let b = ymm_reg_with_unique_values("bench_shuffle_ps_b", &solver, 32);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm256_shuffle_ps(&a, &b, Imm8::Expr(imm8));
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&a, 0),
            (&a, 1),
            (&b, 2),
            (&b, 3),
            (&a, 4),
            (&a, 5),
            (&b, 6),
            (&b, 7),
        ],
    );
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm512_shuffle_ps_imm8_two_vec() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("bench_shuffle_ps_a", &solver, 32);
    let b = zmm_reg_with_unique_values("bench_shuffle_ps_b", &solver, 32);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm512_shuffle_ps(&a, &b, Imm8::Expr(imm8));
    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&a, 0),
            (&a, 1),
            (&b, 2),
            (&b, 3),
            (&a, 4),
            (&a, 5),
            (&b, 6),
            (&b, 7),
            (&a, 8),
            (&a, 9),
            (&b, 10),
            (&b, 11),
            (&a, 12),
            (&a, 13),
            (&b, 14),
            (&b, 15),
        ],
    );
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_shuffle_pd_imm8_two_vec() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("bench_shuffle_pd_a", &solver, 64);
    let b = ymm_reg_with_unique_values("bench_shuffle_pd_b", &solver, 64);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm256_shuffle_pd(&a, &b, Imm8::Expr(imm8), &solver);
    let expected = construct_ymm_reg_from_elements(64, &[(&a, 0), (&b, 1), (&a, 2), (&b, 3)]);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_permute2x128_imm8_identity() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("bench_perm2x128", &solver, 128);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm256_permute2x128_si256(&input, &input, Imm8::Expr(imm8), &solver);
    solver.assert(output.eq(input));
    result(solver.check())
}

fn solve_mm512_shuffle_i32x4_imm8_identity() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let input = zmm_reg_with_unique_values("bench_shuffle_i32x4", &solver, 128);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm512_shuffle_i32x4(&input, &input, Imm8::Expr(imm8));
    solver.assert(output.eq(input));
    result(solver.check())
}

fn solve_mm256_permute4x64_imm8_reverse() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("bench_perm4x64", &solver, 64);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm256_permute4x64_epi64(&input, Imm8::Expr(imm8));
    let expected = ymm_reg_reversed("bench_reversed", &solver, &input, 64);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_blend_ps_imm8_first_half() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("bench_blend_ps_a", &solver, 32);
    let b = ymm_reg_with_unique_values("bench_blend_ps_b", &solver, 32);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm256_blend_ps(&a, &b, Imm8::Expr(imm8), &solver);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&b, 0),
            (&b, 1),
            (&b, 2),
            (&b, 3),
            (&a, 4),
            (&a, 5),
            (&a, 6),
            (&a, 7),
        ],
    );
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_blend_pd_imm8_alternating() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("bench_blend_pd_a", &solver, 64);
    let b = ymm_reg_with_unique_values("bench_blend_pd_b", &solver, 64);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm256_blend_pd(&a, &b, Imm8::Expr(imm8), &solver);
    let expected = construct_ymm_reg_from_elements(64, &[(&a, 0), (&b, 1), (&a, 2), (&b, 3)]);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_blendv_ps_mask_signs() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("bench_blendv_ps_a", &solver, 32);
    let b = ymm_reg_with_unique_values("bench_blendv_ps_b", &solver, 32);
    let mask = ymm_reg("bench_mask");
    let output = mm256_blendv_ps(&a, &b, &mask, &solver);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&b, 0),
            (&a, 1),
            (&b, 2),
            (&a, 3),
            (&b, 4),
            (&a, 5),
            (&b, 6),
            (&a, 7),
        ],
    );
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_blendv_pd_mask_signs() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("bench_blendv_pd_a", &solver, 64);
    let b = ymm_reg_with_unique_values("bench_blendv_pd_b", &solver, 64);
    let mask = ymm_reg("bench_mask");
    let output = mm256_blendv_pd(&a, &b, &mask, &solver);
    let expected = construct_ymm_reg_from_elements(64, &[(&b, 0), (&a, 1), (&b, 2), (&a, 3)]);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_alignr_epi8_imm8_shift() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = reg_with_unique_values("bench_alignr_epi8_a", &solver, 32, 8);
    let b = reg_with_unique_values("bench_alignr_epi8_b", &solver, 32, 8);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm256_alignr_epi8(&a, &b, Imm8::Expr(imm8), &solver);
    let expected = construct_ymm_reg_from_elements(
        8,
        &[
            (&b, 2),
            (&b, 3),
            (&b, 4),
            (&b, 5),
            (&b, 6),
            (&b, 7),
            (&b, 8),
            (&b, 9),
            (&b, 10),
            (&b, 11),
            (&b, 12),
            (&b, 13),
            (&b, 14),
            (&b, 15),
            (&a, 0),
            (&a, 1),
            (&b, 18),
            (&b, 19),
            (&b, 20),
            (&b, 21),
            (&b, 22),
            (&b, 23),
            (&b, 24),
            (&b, 25),
            (&b, 26),
            (&b, 27),
            (&b, 28),
            (&b, 29),
            (&b, 30),
            (&b, 31),
            (&a, 16),
            (&a, 17),
        ],
    );
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_alignr_epi32_imm8_shift() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = ymm_reg_with_32b_values("bench_alignr_a", &solver, &(10..18).collect::<Vec<_>>());
    let b = ymm_reg_with_32b_values("bench_alignr_b", &solver, &(0..8).collect::<Vec<_>>());
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm256_alignr_epi32(&a, &b, Imm8::Expr(imm8), &solver);
    let expected = ymm_reg_with_32b_values("bench_expected", &solver, &[2, 3, 4, 5, 6, 7, 10, 11]);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_alignr_epi64_imm8_shift() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = ymm_reg_with_64b_values("bench_alignr_a", &solver, &(100..104).collect::<Vec<_>>());
    let b = ymm_reg_with_64b_values("bench_alignr_b", &solver, &(0..4).collect::<Vec<_>>());
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm256_alignr_epi64(&a, &b, Imm8::Expr(imm8), &solver);
    let expected = ymm_reg_with_64b_values("bench_expected", &solver, &[1, 2, 3, 100]);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm512_mask_alignr_epi64_imm8_and_mask() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let src = zmm_reg_with_64b_values("bench_src", &solver, &(100..108).collect::<Vec<_>>());
    let a = zmm_reg_with_64b_values("bench_a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("bench_b", &solver, &(0..8).collect::<Vec<_>>());
    let mask = BV::new_const("bench_k", 8);
    let imm8 = BV::new_const("bench_imm8", 8);
    let output = mm512_mask_alignr_epi64(&src, &mask, &a, &b, Imm8::Expr(imm8), &solver);
    let expected = zmm_reg_with_64b_values(
        "bench_expected",
        &solver,
        &[5, 6, 7, 103, 104, 105, 106, 107],
    );
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_permutexvar_epi32_indices() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("bench_permutexvar_input", &solver, 32);
    let indices = ymm_reg("bench_indices");
    let output = mm256_permutexvar_epi32(&input, &indices, &solver);
    let expected = ymm_reg_reversed("bench_reversed", &solver, &input, 32);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_permutexvar_epi64_indices() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("bench_permutexvar_input", &solver, 64);
    let indices = ymm_reg("bench_indices");
    let output = mm256_permutexvar_epi64(&input, &indices, &solver);
    let expected = ymm_reg_reversed("bench_reversed", &solver, &input, 64);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm512_permutex2var_epi32_indices() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let (a, b) = unique_zmm_pair("bench_permutex2var", &solver, 32);
    let indices = zmm_reg("bench_indices");
    let output = mm512_permutex2var_epi32(&a, &indices, &b, &solver);
    solver.assert(output.eq(&b));
    result(solver.check())
}

fn solve_mm512_mask_permutex2var_epi64_indices_and_mask() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let (a, b) = unique_zmm_pair("bench_permutex2var", &solver, 64);
    let indices = zmm_reg("bench_indices");
    let mask = BV::new_const("bench_k", 8);
    let output = mm512_mask_permutex2var_epi64(&a, &mask, &indices, &b, &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..8)
        .map(|idx| if idx < 4 { (&b, idx) } else { (&a, idx) })
        .collect();
    let expected = construct_zmm_reg_from_elements(64, &expected_specs);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_permutevar_ps_controls() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("bench_permutevar_input", &solver, 32);
    let controls = ymm_reg("bench_controls");
    let output = mm256_permutevar_ps(&input, &controls, &solver);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&input, 3),
            (&input, 2),
            (&input, 1),
            (&input, 0),
            (&input, 7),
            (&input, 6),
            (&input, 5),
            (&input, 4),
        ],
    );
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_permutevar_pd_controls() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("bench_permutevar_input", &solver, 64);
    let controls = ymm_reg("bench_controls");
    let output = mm256_permutevar_pd(&input, &controls, &solver);
    let expected =
        construct_ymm_reg_from_elements(64, &[(&input, 1), (&input, 0), (&input, 3), (&input, 2)]);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm512_mask_unpackhi_epi32_mask() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("bench_src", &solver, 32);
    let a = zmm_reg_with_unique_values("bench_unpack_a", &solver, 32);
    let b = zmm_reg_with_unique_values("bench_unpack_b", &solver, 32);
    let mask = BV::new_const("bench_k", 16);
    let output = mm512_mask_unpackhi_epi32(&src, &mask, &a, &b, &solver);
    let unmasked = mm512_mask_unpackhi_epi32(&src, &BV::from_u64(0x000F, 16), &a, &b, &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..16)
        .map(|idx| {
            if idx < 4 {
                (&unmasked, idx)
            } else {
                (&src, idx)
            }
        })
        .collect();
    let expected = construct_zmm_reg_from_elements(32, &expected_specs);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_unpacklo_epi32_inputs() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = ymm_reg("bench_unpack_a");
    let b = ymm_reg("bench_unpack_b");
    let output = mm256_unpacklo_epi32(&a, &b);
    let expected = packed_32(&[0x1234_5678; 8], 256);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_min_epi32_inputs() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = ymm_reg("bench_min_a");
    let b = ymm_reg("bench_min_b");
    let output = mm256_min_epi32(&a, &b);
    let expected = packed_32(&[1, 2, 3, 4, 5, 6, 7, 8], 256);
    solver.assert(output.eq(expected));
    result(solver.check())
}

fn solve_mm256_min_epi64_inputs() -> SolveBenchmarkResult {
    let solver = Solver::new();
    let a = ymm_reg("bench_min_a");
    let b = ymm_reg("bench_min_b");
    let output = mm256_min_epi64(&a, &b);
    let expected = packed_64(&[1, 2, 3, 4], 256);
    solver.assert(output.eq(expected));
    result(solver.check())
}
