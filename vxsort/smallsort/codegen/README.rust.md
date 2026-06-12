# bitonic codegen

This directory contains the Rust super-optimizer used to generate bitonic sorter
building blocks for vxsort. It searches for efficient AVX permutation/shuffle
gadgets for each bitonic comparison stage, proves candidate gadgets with Z3,
scores complete paths with uiCA, and exports solution JSON or assembly.

Run commands from `vxsort/smallsort/codegen`.

## Crates

- `z3_avx`: symbolic AVX intrinsic semantics.
- `gadget_synth`: gadget/minigraph candidate generation and Z3 solving.
- `gadget_viz`: template JSONL visualization as Mermaid, Markdown, or HTML.
- `bitonic_codegen`: solver CLI, wave engine, JSON/ASM export, verifier, runtime tracing, and uiCA scoring.

## Search Flow

1. `BitonicSorter` generates the comparison stages for the requested element count.
2. `gadget_synth` enumerates candidate AVX instruction graphs for each stage.
3. `z3_avx` gives those instructions symbolic semantics and Z3 validates the gadget output state.
4. `WaveEngine` records valid stage transitions and discovers complete root-to-terminal paths.
5. Scoring ranks paths first by rough instruction cost, then by uiCA when requested.
6. Exporters write solution JSON and annotated x86-64 assembly from the same lowered instruction stream.

Core data structures:

- `VectorState`: element labels in the top/bottom vectors.
- `InstructionSpec`: one AVX instruction plus operands.
- `PermutationGadget`: concrete top/bottom instruction sequences for one transition.
- `TransitionTable`: stage-indexed transitions and complete paths.
- `AssignedPath`: a complete path plus selected gadget index per transition.

## Supported Targets

The CLI accepts `AVX2` and `AVX512`. Solver synthesis currently supports `i32`
and `i64`; other primitive types are accepted for JSON/ASM import/export paths
where applicable.

Useful AVX instruction families include:

- 32-bit elements: `VSHUFPS`, `VUNPCK*PS`, `VPUNPCK*DQ`, `VPSHUFD`, `VPERMILPS`, `VPERMD`, `VPERMPS`, `VPERM2F128`/`VPERM2I128`, and blends.
- 64-bit elements: `VUNPCK*PD`, `VPUNPCK*QDQ`, `VSHUFPD`, `VPERMILPD`, `VPERMQ`, `VPERMPD`, `VPERM2F128`/`VPERM2I128`, and blends.

## Fresh Checkout

The workspace uses the in-tree `uica/` submodule for Rust uiCA crates.

```bash
cargo test --release -q -p bitonic_codegen

cargo run -q -p bitonic_codegen -- \
  fetch-uica-data \
  --target-cpu SKL
```

`fetch-uica-data` downloads `manifest.json` and the requested `.uipack` files
from `https://uica.houmus.org/data` into `uica-data/`. The `uica-data/`
directory is local cache data and is intentionally ignored by git.

## Build and Test

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features --release -- -D warnings
cargo test --release -q
```

The Rust workspace is rooted at `Cargo.toml`; crates live directly under this
directory.

## Solve

Run a small scored solver pass:

```bash
cargo run -q -p bitonic_codegen -- \
  solve \
  --vector-machine AVX2 \
  --datatype i64 \
  --target-cpu SKL \
  --top-k 5 \
  --gadget-depth 1 \
  --max-waves 1 \
  --wave-attempts 100 \
  --wave-outputs 5 \
  --runtime-ui none \
  --output-path /tmp/vxsort-solutions.json \
  --asm-output-path /tmp/vxsort-solutions.asm
```

For scored solving, `--target-cpu` and `--top-k` are required. `--target-cpu`
must name one or more uiCA manifest architecture keys, matched
case-insensitively. Missing local `.uipack` data is an error; run
`fetch-uica-data` for the same CPU first.

By default, final ranking is bounded: the solver keeps the best `10 * top_k`
rough candidates and re-ranks them with the Rust uiCA simulator.

Depth-2 search can be scoped with `--deep-search-mode`:

- `baseline`: default. Normal waves try every candidate allowed by `--gadget-depth`.
- `shared-prefix`: normal waves try depth-1 candidates plus depth-2 shared-prefix candidates only. A shared-prefix candidate is a depth-2 top/bottom expression where common setup work is reused by both outputs. Internally this can look like two instructions per side, but lowering emits the identical prefix once, so the common case is up to four emitted instructions total: `top_prefix`, `bottom_prefix`, `top_tail`, `bottom_tail`.
- `adaptive`: normal waves use the `shared-prefix` set, then spend extra full depth-2 attempts on prioritized `(stage, input_state)` targets. Use this with `--gadget-depth 2`; adaptive decisions are visible in `--runtime-trace` events named `adaptive_deep_started`, `adaptive_deep_candidate_selected`, and `adaptive_deep_finished`.

Multiple targets are comma-separated:

```bash
cargo run -q -p bitonic_codegen -- \
  fetch-uica-data \
  --target-cpu SKL,TGL

cargo run -q -p bitonic_codegen -- \
  solve \
  --vector-machine AVX2 \
  --datatype i64 \
  --target-cpu SKL,TGL \
  --top-k 5 \
  --gadget-depth 1 \
  --max-waves 1 \
  --wave-attempts 100 \
  --wave-outputs 5 \
  --runtime-ui none \
  --output-path /tmp/vxsort-solutions.json \
  --asm-output-path /tmp/vxsort-solutions.asm
```

With multiple CPUs, outputs are suffixed by target CPU, for example
`/tmp/vxsort-solutions.SKL.json`, `/tmp/vxsort-solutions.TGL.json`, and matching
`.asm` files.

## Runtime Tracing

Use `--runtime-ui textual` for line-oriented progress, or `--runtime-ui none`
for quiet batch runs. Use `--runtime-trace PATH` to write JSONL trace events.

```bash
cargo run -q -p bitonic_codegen -- \
  solve \
  --vector-machine AVX2 \
  --datatype i64 \
  --target-cpu SKL \
  --top-k 5 \
  --gadget-depth 1 \
  --max-waves 1 \
  --runtime-ui textual \
  --runtime-trace /tmp/vxsort-runtime.jsonl
```

## JSON, ASM, Verify, and Score

Convert an exported solution JSON to assembly:

```bash
cargo run -q -p bitonic_codegen -- \
  json-to-asm \
  --input /tmp/vxsort-solutions.json \
  --output /tmp/vxsort-solutions.asm
```

`json-to-asm` accepts the Rust solution JSON schema written by `--output-path`.
It reconstructs the transition table and root-to-leaf solution paths, then uses
the same assembly lowering/emission path as `--asm-output-path`.

Verify solution JSON with Z3:

```bash
cargo run -q -p bitonic_codegen -- \
  verify \
  --input /tmp/vxsort-solutions.json \
  --top-k 1 \
  --workers 1
```

Estimate an existing solution JSON with uiCA:

```bash
cargo run -q -p bitonic_codegen -- \
  estimate \
  --input /tmp/vxsort-solutions.json \
  --target-cpu SKL \
  --uica-data-dir uica-data \
  --top-k 5 \
  --output-dir /tmp/vxsort-uica-estimate
```

Inspect the lowered instructions and uiCA score for selected paths:

```bash
cargo run -q -p bitonic_codegen -- \
  score-json \
  --input /tmp/vxsort-solutions.json \
  --target-cpu SKL \
  --uica-data-dir uica-data \
  --path-index 0
```

Generated artifacts:

- `*.json`: solution graph, roots, nodes, transitions, and optional concrete path assignments.
- `*.asm`: annotated x86-64 assembly emitted through the canonical assembly exporter.
- `uica-data/`: local uiCA manifest and architecture pack cache.

## Dump Gadget Templates

Template dumps export candidate minigraph shapes before Z3 solving: intrinsic
nodes, input references, symbolic immediates/control vectors/masks, mux nodes,
and top/bottom outputs.

Synthesized concrete gadgets are intentionally not exported here. The template
dump is the inspection artifact for candidate generation.

```bash
cargo run -q -p gadget_synth --bin dump_gadget_synth -- \
  templates \
  --arch avx2 \
  --dtype i64 \
  --gadget-depth 2 \
  --exclude-shared-prefix \
  --output /tmp/avx2-i64-depth2-templates.jsonl
```

Useful dump options:

- `--gadget-depth 1|2`: candidate graph depth. Depths above 2 are not supported yet.
- `--exclude-shared-prefix`: omit depth-2 shared-prefix templates.
- `--intrinsic-filter NAME[,NAME...]`: restrict the candidate set to exact intrinsic names.
- `--output PATH`: JSONL output path.

Example filtered dump:

```bash
cargo run -q -p gadget_synth --bin dump_gadget_synth -- \
  templates \
  --arch avx512 \
  --dtype i64 \
  --gadget-depth 1 \
  --intrinsic-filter _mm512_permute_pd,_mm512_shuffle_pd \
  --output /tmp/avx512-i64-filtered-templates.jsonl
```

## Template JSONL Schema

Each line is one normalized template record:

```json
{
  "kind": "template",
  "arch": "avx2",
  "dtype": "i64",
  "gadget_depth": 2,
  "tier": "deep",
  "graph": {
    "top": { "kind": "intrinsic", "name": "...", "operands": {} },
    "bottom": null
  }
}
```

Graph nodes may be:

- `input`: `{ "kind": "input", "name": "top" }` or `bottom`.
- `symbolic`: normalized symbolic values such as `imm8`, `mask`, `control_vector`, or `mux_select`.
- `mux`: a symbolic selector plus source alternatives and optional pruning.
- `intrinsic`: an intrinsic name plus normalized operands.

Symbolic IDs (`v0`, `v1`, ...) are stable within a record and preserve aliasing
without leaking object identities.

## Visualize Templates

`gadget_viz` reads template JSONL and renders selected records or the full file.

Render one record to HTML:

```bash
cargo run -q -p gadget_viz -- \
  --input /tmp/avx2-i64-depth2-templates.jsonl \
  --index 0 \
  --format html \
  --output /tmp/gadget-template-0.html
```

Render all records to an HTML gallery:

```bash
cargo run -q -p gadget_viz -- \
  --input /tmp/avx2-i64-depth2-templates.jsonl \
  --format html \
  --output /tmp/gadget-templates.html
```

Emit raw Mermaid:

```bash
cargo run -q -p gadget_viz -- \
  --input /tmp/avx2-i64-depth2-templates.jsonl \
  --index 0 \
  --format mermaid \
  --output /tmp/gadget-template-0.mmd
```

Supported formats:

- `mermaid`: raw Mermaid flowchart text.
- `markdown`: metadata plus fenced Mermaid diagrams.
- `html`: standalone browser-viewable page with pre-rendered inline SVG diagrams.

If `--index N` is omitted, all JSONL records are rendered. `--index` is
zero-based. Full depth-2 dumps can contain thousands of records, so prefer
`--index` while inspecting individual templates.

The Mermaid graph represents template dataflow only:

```text
top input / bottom input
  -> intrinsic, mux, and symbolic operand nodes
  -> top output / bottom output
```

Record metadata such as `arch`, `dtype`, `gadget_depth`, and `tier` appears
outside the Mermaid graph in Markdown/HTML output.
