# z3_avx Rust Port Batch Plan

This is the working plan for completing the Rust port of `src/z3_avx.py` after the first `permute_ps` wave.

The rule for this port is strict parity: every test in `tests/test_z3_avx.py` and `tests/test_z3_avx_canonicalization.py` must have a Rust counterpart. Reports under `docs/z3-avx-rust-port/intrinsics/` are the source of truth for old Python test names, planned or actual Rust test names, paths, canonicalization requirements, and k-mask requirements.

## Cross-cutting requirements

- Keep immediate operands idiomatic as an enum like the current `Imm8`: literal `u8` for direct tests and symbolic `BV` for synthesis/search tests.
- Any function that canonicalizes dead control bits must accept a `Solver` at the instruction construction boundary.
- Apply dead-bit constraints locally in the generic implementation when the generic has the live-bit context.
- For every AVX512 masked intrinsic, constrain non-live high k-mask bits to zero if the Rust mask operand is wider than the architectural mask width.
- Use exact or near-exact Rust test names from the reports. New ports should prefer the Python function name verbatim as the Rust test name unless a test already exists with a different name.
- Each worker owns disjoint source, test, and report files. Workers are not alone in the codebase and must not revert or rewrite unrelated edits.

## Shared Rust infrastructure

- Create `rust/z3_avx/src/canonical.rs` for reusable dead-bit constraints:
  - scalar low-bit immediates: constrain `[width-1:live_bits] == 0`
  - per-lane low-bit controls: constrain unused high bits in each element
  - arbitrary live bit ranges for cases like `permutevar_pd` and `permute2x128`
  - k-mask high-bit canonicalization
- Create or extend `rust/z3_avx/src/mask.rs` for shared AVX512 writemask merge helpers for 32-bit and 64-bit element widths.
- Keep instruction semantics split by family rather than adding a single large module.
- Add Rust canonicalization tests before implementing each family.

## Batches

### Batch 0: Infrastructure and already-started coverage

Files:
- Modify: `rust/z3_avx/src/lib.rs`
- Create: `rust/z3_avx/src/canonical.rs`
- Create: `rust/z3_avx/src/mask.rs`
- Modify: `rust/z3_avx/src/control.rs`
- Modify: `rust/z3_avx/src/permute.rs`
- Modify: `rust/z3_avx/tests/control.rs`
- Modify: `rust/z3_avx/tests/permute_ps.rs`

Work:
- Finish exact test parity for `_MM_SHUFFLE`, `_MM_SHUFFLE2`, and decode/string helpers.
- Keep `_mm256_permute_ps`, `_mm512_permute_ps`, and `_mm512_mask_permute_ps` as the reference shape for future families.
- Add the widened-mask regression pattern for `_mm512_mask_permute_ps`.

Verification:
- `cargo test -p z3_avx --test control`
- `cargo test -p z3_avx --test permute_ps`
- `cargo test -p z3_avx`

### Batch 1: Static immediate permutes and shuffles

Families:
- `_mm256_permute_pd`, `_mm512_permute_pd`, `_mm512_mask_permute_pd`
- `_mm256_shuffle_ps`, `_mm512_shuffle_ps`, `_mm512_mask_shuffle_ps`
- `_mm256_shuffle_pd`, `_mm512_shuffle_pd`, `_mm512_mask_shuffle_pd`
- `_mm256_permute4x64_epi64`

Files:
- Create: `rust/z3_avx/src/permute_pd.rs`
- Create: `rust/z3_avx/src/shuffle_ps.rs`
- Create: `rust/z3_avx/src/shuffle_pd.rs`
- Create: `rust/z3_avx/src/permute4x64.rs`
- Create/modify matching test files listed in the reports.

Canonicalization:
- `permute_pd`: only imm bits `[1:0]` live; constrain `[7:2]`.
- `shuffle_pd` YMM: bits `[3:0]` live; constrain `[7:4]`.
- `permute4x64_epi64` and `shuffle_ps`: all imm8 bits are live.
- Masked 512-bit variants must constrain high k-mask bits (`16` live for ps, `8` live for pd).

### Batch 2: Lane-level immediate operations

Families:
- `_mm256_permute2x128_si256`
- `_mm512_shuffle_i32x4`, `_mm512_mask_shuffle_i32x4`

Files:
- Create: `rust/z3_avx/src/permute2x128.rs`
- Create: `rust/z3_avx/src/shuffle_i32x4.rs`
- Create/modify matching test files listed in the reports.

Canonicalization:
- `permute2x128`: in each control nibble, bits `[1:0]` and zero bit `[3]` are live; bits `2` and `6` are dead.
- `shuffle_i32x4`: all imm8 bits are live.
- `_mm512_mask_shuffle_i32x4` must constrain k-mask bits above bit 15.

### Batch 3: Cross-lane variable permutes

Families:
- `_mm256_permutexvar_epi32`, `_mm512_permutexvar_epi32`, `_mm512_mask_permutexvar_epi32`
- `_mm256_permutexvar_epi64`, `_mm512_permutexvar_epi64`, `_mm512_mask_permutexvar_epi64`
- `_mm512_permutex2var_epi32`, `_mm512_mask_permutex2var_epi32`
- `_mm512_permutex2var_epi64`, `_mm512_mask_permutex2var_epi64`

Files:
- Create: `rust/z3_avx/src/permutexvar.rs`
- Create: `rust/z3_avx/src/permutex2var.rs`
- Create/modify matching test files listed in the reports.

Canonicalization:
- `permutexvar_epi32`: YMM lane controls keep `[2:0]`; ZMM keep `[3:0]`.
- `permutexvar_epi64`: YMM lane controls keep `[1:0]`; ZMM keep `[2:0]`.
- `permutex2var_epi32`: keep offset bits `[3:0]` plus source bit `[4]`.
- `permutex2var_epi64`: keep offset bits `[2:0]` plus source bit `[3]`.
- Masked epi32 variants must constrain k-mask bits above bit 15; masked epi64 variants above bit 7.

### Batch 4: Within-lane variable permutes

Families:
- `_mm256_permutevar_ps`, `_mm512_permutevar_ps`, `_mm512_mask_permutevar_ps`
- `_mm256_permutevar_pd`, `_mm512_permutevar_pd`, `_mm512_mask_permutevar_pd`

Files:
- Create: `rust/z3_avx/src/permutevar.rs`
- Create/modify matching test files listed in the reports.

Canonicalization:
- `permutevar_ps`: each 32-bit control element keeps bits `[1:0]`.
- `permutevar_pd`: each 64-bit control element keeps bit `[1]`; bit `[0]` and bits `[63:2]` are dead.
- Masked ps variants must constrain k-mask bits above bit 15; masked pd variants above bit 7.

### Batch 5: Unpack and blend

Families:
- `_mm256_unpacklo_epi32`, `_mm256_unpackhi_epi32`, `_mm512_unpacklo_epi32`, `_mm512_unpackhi_epi32`, masked epi32 forms
- `_mm256_unpacklo_epi64`, `_mm256_unpackhi_epi64`, `_mm512_unpacklo_epi64`, `_mm512_unpackhi_epi64`, masked epi64 forms
- `_mm256_blend_pd`, `_mm256_blend_ps`
- `_mm256_blendv_pd`, `_mm256_blendv_ps`

Files:
- Create: `rust/z3_avx/src/unpack.rs`
- Create: `rust/z3_avx/src/blend.rs`
- Create/modify matching test files listed in the reports.

Canonicalization:
- `blend_pd`: imm bits `[3:0]` live; constrain `[7:4]`.
- `blend_ps`: all imm8 bits live.
- `blendv_ps` and `blendv_pd`: only each element sign bit is live; all other mask-vector bits are dead.
- Masked unpack epi32 variants must constrain k-mask bits above bit 15; masked epi64 variants above bit 7.

### Batch 6: Align and signed min/max

Families:
- `_mm256_alignr_epi8`
- `_mm256_alignr_epi32`, `_mm512_alignr_epi32`, `_mm512_mask_alignr_epi32`
- `_mm256_alignr_epi64`, `_mm512_alignr_epi64`, `_mm512_mask_alignr_epi64`
- `_mm256_min_epi32`, `_mm256_max_epi32`, `_mm512_min_epi32`, `_mm512_max_epi32`
- `_mm256_min_epi64`, `_mm256_max_epi64`, `_mm512_min_epi64`, `_mm512_max_epi64`

Files:
- Create: `rust/z3_avx/src/alignr.rs`
- Create: `rust/z3_avx/src/minmax.rs`
- Create/modify matching test files listed in the reports.

Canonicalization:
- `alignr_epi32`: YMM keeps shift bits `[2:0]`; ZMM keeps `[3:0]`.
- `alignr_epi64`: YMM keeps `[1:0]`; ZMM keeps `[2:0]`.
- Masked alignr epi32 variants must constrain k-mask bits above bit 15; masked epi64 variants above bit 7.
- min/max has no control canonicalization.

## Subagent execution model

For each batch, dispatch one worker per independent family. Example for Batch 3:

- Worker A owns `rust/z3_avx/src/permutexvar.rs`, `rust/z3_avx/tests/permutexvar_epi32.rs`, `rust/z3_avx/tests/permutexvar_epi64.rs`, and `docs/z3-avx-rust-port/intrinsics/permutexvar.md`.
- Worker B owns `rust/z3_avx/src/permutex2var.rs`, `rust/z3_avx/tests/permutex2var_epi32.rs`, `rust/z3_avx/tests/permutex2var_epi64.rs`, and `docs/z3-avx-rust-port/intrinsics/permutex2var.md`.

Each worker must:
- port tests first,
- implement the smallest family-local generic helper,
- add canonicalization tests for dead control bits and widened k-mask bits when masked,
- run the family-specific Rust tests,
- update the family report status from `Planned` to `Ported`.

After every batch, the coordinator must run:

```bash
cargo fmt --all
cargo test -p z3_avx
uv run pytest
uv run ruff check .
uv run vulture
```

`uv run vulture` may return existing findings; inspect and record whether the Rust port introduced any actionable Python dead-code findings.
