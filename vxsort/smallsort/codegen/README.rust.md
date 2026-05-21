# Rust gadget tooling

This directory contains the Rust ports used by the bitonic super-optimizer work:

- `rust/z3_avx`: symbolic AVX intrinsic semantics.
- `rust/gadget_synth`: gadget/minigraph candidate generation and parity dump tooling.
- `rust/gadget_viz`: template JSONL visualization as Mermaid, Markdown, or HTML.

Run commands from `vxsort/smallsort/codegen`.

## Build and test

```bash
cargo fmt --all --check
cargo test -q
```

The Rust workspace is rooted at `Cargo.toml`; the crates live under `rust/`.

## Dump gadget templates

The dumpers export **template records only**. A template is a candidate minigraph shape before Z3 solving: intrinsic nodes, input references, symbolic immediates/control vectors/masks, mux nodes, and top/bottom outputs.

Synthesized concrete gadgets are intentionally not exported here. The template dump is the inspection/parity artifact for candidate generation.

### Rust dumper

```bash
cargo run -q -p gadget_synth --bin dump_gadget_synth -- \
  templates \
  --arch avx2 \
  --dtype i64 \
  --gadget-depth 2 \
  --exclude-shared-prefix \
  --output /tmp/avx2-i64-depth2-templates.jsonl
```

### Python reference dumper

```bash
uv run python tools/dump_gadget_synth.py \
  templates \
  --vector-machine AVX2 \
  --datatype i64 \
  --gadget-depth 2 \
  --exclude-shared-prefix \
  --output /tmp/python-avx2-i64-depth2-templates.jsonl
```

### Useful dump options

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

## Compare Python and Rust dumps

Use the comparer to verify candidate-generation parity:

```bash
uv run python tools/compare_gadget_synth_dumps.py \
  /tmp/python-avx2-i64-depth2-templates.jsonl \
  /tmp/avx2-i64-depth2-templates.jsonl \
  --max-diffs 10
```

Expected success output:

```text
Dumps match
```

## Template JSONL schema

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

Symbolic IDs (`v0`, `v1`, ...) are stable within a record and preserve aliasing without leaking Python/Rust object identities.

## Visualize templates

`gadget_viz` reads template JSONL and renders selected records or the full file.

### Render one record to HTML

```bash
cargo run -q -p gadget_viz -- \
  --input /tmp/avx2-i64-depth2-templates.jsonl \
  --index 0 \
  --format html \
  --output /tmp/gadget-template-0.html
```

Open `/tmp/gadget-template-0.html` in a browser. HTML output uses the `mermaid-rs-renderer` library in-process, embeds rendered SVGs as base64 data URIs, and does not load Mermaid in the browser.

### Render all records to an HTML gallery

```bash
cargo run -q -p gadget_viz -- \
  --input /tmp/avx2-i64-depth2-templates.jsonl \
  --format html \
  --output /tmp/gadget-templates.html
```

### Emit raw Mermaid

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

If `--index N` is omitted, all JSONL records are rendered. `--index` is zero-based. Full depth-2 dumps can contain thousands of records, so prefer `--index` while inspecting individual templates.

## What the diagrams show

The Mermaid graph represents the **template dataflow only**:

```text
top input / bottom input
  -> intrinsic, mux, and symbolic operand nodes
  -> top output / bottom output
```

Record metadata such as `arch`, `dtype`, `gadget_depth`, and `tier` appears outside the Mermaid graph in Markdown/HTML output. It is not drawn as gadget dataflow.

`gadget_viz` rejects non-template records, including old synthesized dumps, because this tooling is now template-only.

## Quick end-to-end example

```bash
PY=/tmp/python-avx2-i64-depth2-templates.jsonl
RS=/tmp/rust-avx2-i64-depth2-templates.jsonl
HTML=/tmp/rust-avx2-i64-depth2-templates.html

uv run python tools/dump_gadget_synth.py templates \
  --vector-machine AVX2 --datatype i64 --gadget-depth 2 \
  --exclude-shared-prefix --output "$PY"

cargo run -q -p gadget_synth --bin dump_gadget_synth -- templates \
  --arch avx2 --dtype i64 --gadget-depth 2 \
  --exclude-shared-prefix --output "$RS"

uv run python tools/compare_gadget_synth_dumps.py "$PY" "$RS"

cargo run -q -p gadget_viz -- \
  --input "$RS" --format html --output "$HTML"
```
