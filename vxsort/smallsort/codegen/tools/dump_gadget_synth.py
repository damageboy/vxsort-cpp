#!/usr/bin/env python3
"""Dump normalized gadget synthesizer artifacts as JSONL.

The JSON schema is kept independent from Python object identity and Z3 object
representations so it can be compared against a future Rust dumper.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bitonic_types import (  # type: ignore[import-not-found]
    GadgetGraph,
    InputRef,
    IntrinsicNode,
    Mux,
    Symbolic,
)
from gadget_synthesizer import (  # type: ignore[import-not-found]
    GadgetSynthesizer,
    graph_max_depth,
)
from util.enums import primitive_type, vector_machine  # type: ignore[import-not-found]


@dataclass
class _NormalizationContext:
    """Per-graph state for assigning deterministic symbolic ids."""

    symbolic_ids: dict[int, str] = field(default_factory=dict)
    symbolic_roles: dict[int, str] = field(default_factory=dict)

    def normalize_symbolic(
        self,
        symbolic: Symbolic,
        *,
        operand_key: str | None = None,
        mux_select: bool = False,
    ) -> dict[str, Any]:
        symbol_id = id(symbolic)
        if symbol_id not in self.symbolic_ids:
            self.symbolic_ids[symbol_id] = f"v{len(self.symbolic_ids)}"
            self.symbolic_roles[symbol_id] = _symbolic_role(
                symbolic,
                operand_key=operand_key,
                mux_select=mux_select,
            )

        return {
            "kind": "symbolic",
            "role": self.symbolic_roles[symbol_id],
            "bits": symbolic.bit_width,
            "var_id": self.symbolic_ids[symbol_id],
        }


def _symbolic_role(
    symbolic: Symbolic,
    *,
    operand_key: str | None,
    mux_select: bool,
) -> str:
    """Return a stable role for a symbolic variable without emitting raw names."""
    if mux_select:
        return "mux_select"

    if operand_key == "imm8" or symbolic.name.startswith("imm8"):
        return "imm8"
    if operand_key in {"k", "mask"} or symbolic.name.startswith("k_mask"):
        return "mask"
    if operand_key == "op_idx":
        return "op_idx"
    if symbolic.name.startswith("ctrl"):
        return "control_vector"
    if operand_key is not None:
        return operand_key
    return "symbolic"


def _normalize_node(
    node: InputRef | Symbolic | Mux | IntrinsicNode | None,
    context: _NormalizationContext,
    *,
    operand_key: str | None = None,
) -> Any:
    if node is None:
        return None
    if isinstance(node, InputRef):
        return {"kind": "input", "name": node.name}
    if isinstance(node, Symbolic):
        return context.normalize_symbolic(node, operand_key=operand_key)
    if isinstance(node, Mux):
        return {
            "kind": "mux",
            "select": context.normalize_symbolic(node.select, mux_select=True),
            "sources": [_normalize_node(source, context) for source in node.sources],
            "pruning": node.pruning,
        }
    if isinstance(node, IntrinsicNode):
        operands = {
            key: _normalize_node(value, context, operand_key=key)
            for key, value in sorted(node.operands.items())
        }
        return {
            "kind": "intrinsic",
            "name": node.name,
            "operands": operands,
            "isomorphic_order": node.isomorphic_order,
        }

    raise TypeError(f"Unsupported gadget graph node: {type(node).__name__}")


def normalize_graph(graph: GadgetGraph) -> dict[str, Any]:
    """Normalize a ``GadgetGraph`` template for stable JSON serialization."""
    context = _NormalizationContext()
    return {
        "top": _normalize_node(graph.top, context),
        "bottom": _normalize_node(graph.bottom, context),
    }


def normalize_template_record(
    graph: GadgetGraph,
    *,
    vm: vector_machine,
    prim_type: primitive_type,
    gadget_depth: int,
) -> dict[str, Any]:
    """Build one canonical template dump record."""
    return {
        "kind": "template",
        "arch": vm.name.lower(),
        "dtype": prim_type.name.lower(),
        "gadget_depth": gadget_depth,
        "tier": "shallow" if graph_max_depth(graph) <= 1 else "deep",
        "graph": normalize_graph(graph),
    }


def dump_template_records(
    *,
    vm: vector_machine,
    prim_type: primitive_type,
    gadget_depth: int,
    intrinsic_filter: set[str] | None = None,
    exclude_shared_prefix: bool = False,
) -> list[dict[str, Any]]:
    """Construct and normalize all template records for the requested target."""
    synth = GadgetSynthesizer(vm, prim_type, intrinsic_filter=intrinsic_filter)
    graphs = _template_candidate_graphs(
        synth,
        gadget_depth,
        exclude_shared_prefix=exclude_shared_prefix,
    )
    records = [
        normalize_template_record(
            graph,
            vm=vm,
            prim_type=prim_type,
            gadget_depth=gadget_depth,
        )
        for graph in graphs
    ]
    return sorted(records, key=_record_sort_key)


def _template_candidate_graphs(
    synth: GadgetSynthesizer,
    gadget_depth: int,
    *,
    exclude_shared_prefix: bool,
) -> list[GadgetGraph]:
    if not exclude_shared_prefix:
        return synth.precompute_all_candidates(gadget_depth)

    graphs: list[GadgetGraph] = []
    for top_depth in range(gadget_depth + 1):
        for bottom_depth in range(gadget_depth + 1):
            graphs.extend(
                synth._generate_candidate_graphs_at_depth(top_depth, bottom_depth)
            )
    return graphs


def _record_sort_key(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _write_jsonl(records: Iterable[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _parse_vector_machine(value: str) -> vector_machine:
    try:
        return vector_machine[value.upper()]
    except KeyError as exc:
        choices = ", ".join(sorted(member.name for member in vector_machine))
        raise argparse.ArgumentTypeError(f"expected one of: {choices}") from exc


def _parse_primitive_type(value: str) -> primitive_type:
    try:
        return primitive_type[value.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(member.name for member in primitive_type))
        raise argparse.ArgumentTypeError(f"expected one of: {choices}") from exc


def _parse_intrinsic_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump normalized gadget synthesizer artifacts as JSONL.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    templates = subparsers.add_parser(
        "templates",
        help="dump normalized candidate graph templates",
    )
    templates.add_argument(
        "--vector-machine", required=True, type=_parse_vector_machine
    )
    templates.add_argument("--datatype", required=True, type=_parse_primitive_type)
    templates.add_argument("--gadget-depth", required=True, type=int)
    templates.add_argument("--output", required=True, type=Path)
    templates.add_argument(
        "--intrinsic-filter",
        help="comma-separated exact intrinsic allowlist",
    )
    templates.add_argument(
        "--exclude-shared-prefix",
        action="store_true",
        help="exclude depth-2 shared-prefix template graphs",
    )
    templates.set_defaults(handler=_run_templates)

    return parser


def _run_templates(args: argparse.Namespace) -> None:
    records = dump_template_records(
        vm=args.vector_machine,
        prim_type=args.datatype,
        gadget_depth=args.gadget_depth,
        intrinsic_filter=_parse_intrinsic_filter(args.intrinsic_filter),
        exclude_shared_prefix=args.exclude_shared_prefix,
    )
    _write_jsonl(records, args.output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
