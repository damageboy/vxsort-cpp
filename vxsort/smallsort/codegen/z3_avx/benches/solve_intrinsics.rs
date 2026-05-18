use std::time::{Duration, Instant};

use criterion::{BenchmarkFilter, Criterion, black_box};
use z3::SatResult;
use z3_avx::solve_benchmarks::{
    SolveBenchmarkSummaryArgs, SolveBenchmarkSummaryRow, format_solve_benchmark_summary,
    parse_solve_benchmark_summary_args, solve_benchmark_cases, solve_benchmark_groups,
};

fn print_summary_table(options: &SolveBenchmarkSummaryArgs) {
    let mut rows = Vec::with_capacity(solve_benchmark_cases().len());

    for case in solve_benchmark_cases() {
        let mut total = Duration::ZERO;
        for _ in 0..options.summary_iterations {
            let start = Instant::now();
            let result = (case.solve)();
            total += start.elapsed();
            assert_eq!(
                result.sat,
                SatResult::Sat,
                "case {} did not solve",
                case.name
            );
        }
        let avg_ms = total.as_secs_f64() * 1000.0 / f64::from(options.summary_iterations);
        rows.push(SolveBenchmarkSummaryRow {
            name: case.name.to_owned(),
            group: case.group.label().to_owned(),
            family: case.family.to_owned(),
            avg_ms,
        });
    }
    println!();
    println!(
        "{}",
        format_solve_benchmark_summary(options.summary_iterations, &rows, options.summary_format)
    );
    println!();
}

fn parse_value_arg(
    args: &[String],
    index: &mut usize,
    option: &str,
    value_prefix: Option<&str>,
) -> Result<String, String> {
    if let Some(value) = value_prefix {
        return Ok(value.to_owned());
    }

    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn parse_seconds(value: &str, option: &str) -> Result<Duration, String> {
    let seconds = value
        .parse::<f64>()
        .map_err(|_| format!("{option} requires a numeric value"))?;
    if seconds <= 0.0 {
        return Err(format!("{option} must be greater than zero"));
    }
    Ok(Duration::from_secs_f64(seconds))
}

fn parse_summary_mode_criterion(args: &[String]) -> Result<Criterion, String> {
    let mut criterion = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(250))
        .measurement_time(Duration::from_secs(2));
    let mut filter = None;
    let mut exact = false;
    let mut index = 0;

    while index < args.len() {
        let arg = &args[index];
        if arg == "--bench" || arg == "--nocapture" || arg == "--show-output" {
            index += 1;
            continue;
        }

        if arg == "--exact" {
            exact = true;
        } else if arg == "--noplot" {
            criterion = criterion.without_plots();
        } else if arg == "--sample-size" || arg.starts_with("--sample-size=") {
            let value = parse_value_arg(
                args,
                &mut index,
                "--sample-size",
                arg.strip_prefix("--sample-size="),
            )?;
            let sample_size = value
                .parse::<usize>()
                .map_err(|_| "--sample-size requires an integer value".to_owned())?;
            criterion = criterion.sample_size(sample_size);
        } else if arg == "--warm-up-time" || arg.starts_with("--warm-up-time=") {
            let value = parse_value_arg(
                args,
                &mut index,
                "--warm-up-time",
                arg.strip_prefix("--warm-up-time="),
            )?;
            criterion = criterion.warm_up_time(parse_seconds(&value, "--warm-up-time")?);
        } else if arg == "--measurement-time" || arg.starts_with("--measurement-time=") {
            let value = parse_value_arg(
                args,
                &mut index,
                "--measurement-time",
                arg.strip_prefix("--measurement-time="),
            )?;
            criterion = criterion.measurement_time(parse_seconds(&value, "--measurement-time")?);
        } else if arg == "--color" || arg.starts_with("--color=") {
            let value = parse_value_arg(args, &mut index, "--color", arg.strip_prefix("--color="))?;
            match value.as_str() {
                "always" => criterion = criterion.with_output_color(true),
                "never" => criterion = criterion.with_output_color(false),
                "auto" => {}
                _ => return Err("--color expects auto, always, or never".to_owned()),
            }
        } else if arg.starts_with('-') {
            return Err(format!(
                "unsupported Criterion argument '{arg}' when using --summary-format; supported options are --sample-size, --warm-up-time, --measurement-time, --noplot, --color, --exact, and a positional filter"
            ));
        } else if filter.is_none() {
            filter = Some(arg.clone());
        } else {
            return Err(format!("unexpected extra positional argument '{arg}'"));
        }

        index += 1;
    }

    if let Some(filter) = filter {
        criterion = if exact {
            criterion.with_benchmark_filter(BenchmarkFilter::Exact(filter))
        } else {
            criterion.with_filter(filter)
        };
    }

    Ok(criterion)
}

fn configure_criterion(options: &SolveBenchmarkSummaryArgs) -> Result<Criterion, String> {
    let default = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(250))
        .measurement_time(Duration::from_secs(2));

    if options.has_summary_args {
        parse_summary_mode_criterion(&options.criterion_args)
    } else {
        Ok(default.configure_from_args())
    }
}

fn main() {
    let options =
        parse_solve_benchmark_summary_args(std::env::args().skip(1)).unwrap_or_else(|error| {
            eprintln!("error: {error}");
            std::process::exit(2);
        });

    print_summary_table(&options);

    let mut criterion = configure_criterion(&options).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(2);
    });
    for benchmark_group in solve_benchmark_groups() {
        let mut group = criterion.benchmark_group(benchmark_group.id());
        for case in solve_benchmark_cases()
            .iter()
            .filter(|case| case.group == *benchmark_group)
        {
            group.bench_function(case.name, |bench| {
                bench.iter(|| {
                    let result = (case.solve)();
                    assert_eq!(
                        result.sat,
                        SatResult::Sat,
                        "case {} did not solve",
                        case.name
                    );
                    black_box(result);
                });
            });
        }
        group.finish();
    }
    criterion.final_summary();
}
