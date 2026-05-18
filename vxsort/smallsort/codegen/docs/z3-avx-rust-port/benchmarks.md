# z3_avx Solve Benchmarks

The Rust port includes a Criterion benchmark suite for Z3 solve scenarios where an intrinsic parameter is symbolic and constrained by an expected output.

Run the suite:

```bash
cargo bench -p z3_avx --bench solve_intrinsics
```

The benchmark prints a compact Markdown table before Criterion starts:

```text
z3_avx solve benchmark summary (build + check, 3 iteration average)
| intrinsic solve case | group | family | avg ms |
|---|---:|---:|---:|
...
```

To render a prettier terminal table instead:

```bash
cargo bench -p z3_avx --bench solve_intrinsics -- --summary-format ascii
```

Accepted summary formats are `markdown`/`md` and `ascii`/`pretty`. Markdown is the default so redirected output remains easy to paste into reports.

Criterion writes its normal detailed reports under:

```text
target/criterion/report/index.html
```

The Criterion measurements are split into width groups:

```text
solve_intrinsics_256bit
solve_intrinsics_512bit
```

Those group names can be used as filters and show up as separate sections in the HTML output.

To make the quick summary cheaper or more stable, set:

```bash
cargo bench -p z3_avx --bench solve_intrinsics -- --summary-iterations 1
```

The summary flags can be combined with common Criterion arguments:

```bash
cargo bench -p z3_avx --bench solve_intrinsics -- \
  --summary-format ascii --summary-iterations 1 --sample-size 10 mm256_permute_ps
```

To benchmark just the 512-bit group:

```bash
cargo bench -p z3_avx --bench solve_intrinsics -- \
  --summary-format ascii --summary-iterations 1 solve_intrinsics_512bit
```

The timed unit is complete benchmark construction plus `solver.check()`, matching the test-style usage where the symbolic parameter, intrinsic expression, expected output constraint, and solver call are built together.
