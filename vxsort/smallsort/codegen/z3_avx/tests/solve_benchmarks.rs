use std::collections::HashSet;

use z3::SatResult;
use z3_avx::solve_benchmarks::{
    SolveBenchmarkGroup, SolveBenchmarkSummaryRow, SummaryFormat, format_solve_benchmark_summary,
    parse_solve_benchmark_summary_args, solve_benchmark_cases, solve_benchmark_groups,
};

#[test]
fn solve_benchmark_registry_has_unique_cases_that_solve() {
    let cases = solve_benchmark_cases();
    assert!(!cases.is_empty(), "expected at least one solve benchmark");

    let mut names = HashSet::new();
    for case in cases {
        assert!(
            names.insert(case.name),
            "duplicate solve benchmark case name: {}",
            case.name
        );
        let result = (case.solve)();
        assert_eq!(
            result.sat,
            SatResult::Sat,
            "case {} did not solve",
            case.name
        );
    }
}

#[test]
fn solve_benchmark_registry_has_explicit_width_groups() {
    let groups: Vec<_> = solve_benchmark_groups()
        .iter()
        .map(|group| group.id())
        .collect();
    assert_eq!(
        groups,
        [
            SolveBenchmarkGroup::Ymm256.id(),
            SolveBenchmarkGroup::Zmm512.id()
        ]
    );

    for case in solve_benchmark_cases() {
        if case.name.starts_with("mm256_") {
            assert_eq!(case.group, SolveBenchmarkGroup::Ymm256, "{}", case.name);
        } else if case.name.starts_with("mm512_") {
            assert_eq!(case.group, SolveBenchmarkGroup::Zmm512, "{}", case.name);
        } else {
            panic!("unexpected benchmark case prefix: {}", case.name);
        }
    }
}

#[test]
fn solve_benchmark_summary_can_render_markdown() {
    let rows = [SolveBenchmarkSummaryRow {
        name: "mm256_permute_ps/imm8_identity".to_owned(),
        group: "256-bit".to_owned(),
        family: "permute_ps".to_owned(),
        avg_ms: 12.3456,
    }];

    let summary = format_solve_benchmark_summary(3, &rows, SummaryFormat::Markdown);

    assert!(summary.contains("| intrinsic solve case | group | family | avg ms |"));
    assert!(
        summary
            .contains("| `mm256_permute_ps/imm8_identity` | `256-bit` | `permute_ps` | 12.346 |")
    );
}

#[test]
fn solve_benchmark_summary_can_render_pretty_ascii() {
    let rows = [SolveBenchmarkSummaryRow {
        name: "mm256_permute_ps/imm8_identity".to_owned(),
        group: "256-bit".to_owned(),
        family: "permute_ps".to_owned(),
        avg_ms: 12.3456,
    }];

    let summary = format_solve_benchmark_summary(3, &rows, SummaryFormat::Ascii);

    assert!(summary.contains("z3_avx solve benchmark summary"));
    assert!(summary.contains("+"));
    assert!(summary.contains("mm256_permute_ps/imm8_identity"));
    assert!(summary.contains("256-bit"));
    assert!(summary.contains("12.346"));
}

#[test]
fn solve_benchmark_summary_args_accept_format_flag() {
    let options = parse_solve_benchmark_summary_args([
        "--summary-format".to_owned(),
        "ascii".to_owned(),
        "--sample-size".to_owned(),
        "10".to_owned(),
    ])
    .expect("summary args should parse");

    assert_eq!(options.summary_format, SummaryFormat::Ascii);
    assert_eq!(options.summary_iterations, 3);
    assert!(options.has_summary_args);
    assert_eq!(options.criterion_args, ["--sample-size", "10"]);
}

#[test]
fn solve_benchmark_summary_args_accept_equals_form_and_iterations() {
    let options = parse_solve_benchmark_summary_args([
        "--summary-format=md".to_owned(),
        "--summary-iterations=5".to_owned(),
    ])
    .expect("summary args should parse");

    assert_eq!(options.summary_format, SummaryFormat::Markdown);
    assert_eq!(options.summary_iterations, 5);
    assert!(options.criterion_args.is_empty());
}

#[test]
fn solve_benchmark_summary_args_reject_unknown_format() {
    let error = parse_solve_benchmark_summary_args(["--summary-format=html".to_owned()])
        .expect_err("unknown summary format should fail");

    assert!(error.contains("unknown summary format"));
}
