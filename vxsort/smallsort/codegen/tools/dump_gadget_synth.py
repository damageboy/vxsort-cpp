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
    InstructionSpec,
    IntrinsicNode,
    Mux,
    PermutationGadget,
    Symbolic,
    VectorState,
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


def _normalize_vector_state(state: VectorState) -> dict[str, list[int]]:
    return {"top": list(state.top), "bottom": list(state.bottom)}


def _identity_pairs_fixture(
    synth: GadgetSynthesizer,
) -> tuple[VectorState, list[tuple[int, int]]]:
    lanes = synth.elements_per_vector
    top = list(range(lanes))
    bottom = list(range(lanes, lanes * 2))
    return VectorState(top=top, bottom=bottom), list(zip(top, bottom))


def _fixture_inputs(
    fixture: str,
    synth: GadgetSynthesizer,
) -> tuple[VectorState, list[tuple[int, int]]]:
    if fixture == "identity_pairs":
        return _identity_pairs_fixture(synth)
    raise ValueError(f"Unknown synthesized dump fixture: {fixture}")


def _instruction_arg_bit_widths(root: IntrinsicNode | None) -> list[dict[str, int]]:
    """Return symbolic operand widths in instruction topological order."""
    if root is None:
        return []

    order: list[IntrinsicNode] = []
    visited: set[int] = set()

    def topo_visit(node: InputRef | Symbolic | Mux | IntrinsicNode | None) -> None:
        if node is None:
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, IntrinsicNode):
            for operand in node.operands.values():
                topo_visit(operand)
            order.append(node)
        elif isinstance(node, Mux):
            for source in node.sources:
                topo_visit(source)

    topo_visit(root)
    return [
        {
            key: operand.bit_width
            for key, operand in node.operands.items()
            if isinstance(operand, Symbolic)
        }
        for node in order
    ]


def _normalize_bitvec(value: int, bits: int) -> dict[str, Any]:
    if value < 0 or value >= (1 << bits):
        raise ValueError(f"Concrete bit-vector value does not fit in {bits} bits")
    hex_digits = bits // 4
    return {"kind": "bitvec", "bits": bits, "hex": f"0x{value:0{hex_digits}x}"}


def _normalize_concrete_arg(value: Any, *, bit_width: int | None = None) -> Any:
    """Normalize a concrete instruction arg without leaking Z3 representations."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid concrete gadget arguments")
    if hasattr(value, "as_long"):
        value = value.as_long()
    if isinstance(value, int):
        if bit_width is not None and bit_width > 64:
            return _normalize_bitvec(value, bit_width)
        return value
    raise TypeError(f"Unsupported concrete gadget argument: {type(value).__name__}")


def _normalize_instruction(
    instruction: InstructionSpec,
    arg_bit_widths: dict[str, int],
) -> dict[str, Any]:
    return {
        "name": instruction.intrinsic_name,
        "args": {
            key: _normalize_concrete_arg(value, bit_width=arg_bit_widths.get(key))
            for key, value in sorted(instruction.args.items())
        },
    }


def _normalize_instructions(
    instructions: list[InstructionSpec],
    root: IntrinsicNode | None,
) -> list[dict[str, Any]]:
    arg_widths = _instruction_arg_bit_widths(root)
    if len(instructions) != len(arg_widths):
        raise ValueError(
            "Concrete instruction count does not match source graph topology"
        )
    return [
        _normalize_instruction(instruction, widths)
        for instruction, widths in zip(instructions, arg_widths, strict=True)
    ]


def normalize_synthesized_record(
    gadget: PermutationGadget,
    output_state: VectorState,
    graph: GadgetGraph,
    *,
    vm: vector_machine,
    prim_type: primitive_type,
    gadget_depth: int,
    fixture: str,
    input_state: VectorState,
    target_pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    """Build one canonical synthesized gadget dump record."""
    return {
        "kind": "synthesized",
        "arch": vm.name.lower(),
        "dtype": prim_type.name.lower(),
        "gadget_depth": gadget_depth,
        "fixture": fixture,
        "input_state": _normalize_vector_state(input_state),
        "target_pairs": [[left, right] for left, right in target_pairs],
        "output_state": _normalize_vector_state(output_state),
        "top_instructions": _normalize_instructions(
            gadget.top_instructions,
            graph.top,
        ),
        "bottom_instructions": _normalize_instructions(
            gadget.bottom_instructions,
            graph.bottom,
        ),
    }


def dump_synthesized_records(
    *,
    vm: vector_machine,
    prim_type: primitive_type,
    gadget_depth: int,
    fixture: str,
    max_unique_outputs: int,
    intrinsic_filter: set[str] | None = None,
    exclude_shared_prefix: bool = False,
) -> list[dict[str, Any]]:
    """Synthesize and normalize concrete gadget records for one fixture."""
    synth = GadgetSynthesizer(vm, prim_type, intrinsic_filter=intrinsic_filter)
    input_state, target_pairs = _fixture_inputs(fixture, synth)
    records: list[dict[str, Any]] = []

    for graph in _template_candidate_graphs(
        synth,
        gadget_depth,
        exclude_shared_prefix=exclude_shared_prefix,
    ):
        results, _, _ = synth.synthesize_gadget_with_symbolic(
            graph,
            input_state,
            target_pairs,
            max_solutions=1,
            max_unique_outputs=max_unique_outputs,
        )
        records.extend(
            normalize_synthesized_record(
                gadget,
                output_state,
                graph,
                vm=vm,
                prim_type=prim_type,
                gadget_depth=gadget_depth,
                fixture=fixture,
                input_state=input_state,
                target_pairs=target_pairs,
            )
            for gadget, output_state in results
        )

    return sorted(records, key=_record_sort_key)


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

    synthesized = subparsers.add_parser(
        "synthesized",
        help="dump normalized synthesized gadget records",
    )
    synthesized.add_argument(
        "--vector-machine", required=True, type=_parse_vector_machine
    )
    synthesized.add_argument("--datatype", required=True, type=_parse_primitive_type)
    synthesized.add_argument("--gadget-depth", required=True, type=int)
    synthesized.add_argument("--fixture", required=True, choices=["identity_pairs"])
    synthesized.add_argument("--max-unique-outputs", type=int, default=3)
    synthesized.add_argument("--output", required=True, type=Path)
    synthesized.add_argument(
        "--intrinsic-filter",
        help="comma-separated exact intrinsic allowlist",
    )
    synthesized.add_argument(
        "--exclude-shared-prefix",
        action="store_true",
        help="exclude depth-2 shared-prefix template graphs",
    )
    synthesized.set_defaults(handler=_run_synthesized)

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


def _run_synthesized(args: argparse.Namespace) -> None:
    records = dump_synthesized_records(
        vm=args.vector_machine,
        prim_type=args.datatype,
        gadget_depth=args.gadget_depth,
        fixture=args.fixture,
        max_unique_outputs=args.max_unique_outputs,
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
