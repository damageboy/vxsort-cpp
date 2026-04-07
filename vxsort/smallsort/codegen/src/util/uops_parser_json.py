"""JSON parser for compact uops instruction database representation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import zstandard

try:
    from .cost_model import InstructionCost
except ImportError:
    from cost_model import InstructionCost

from .xml_json_converter import SCHEMA_VERSION

# Keep filtering behavior aligned with XML parser.
_MEMORY_MARKERS = (
    "M8,",
    "M8)",
    "M16,",
    "M16)",
    "M32,",
    "M32)",
    "M32_",
    "M64,",
    "M64)",
    "M64_",
    "M128",
    "M256",
    "M512",
)

_PORT_RE = re.compile(r"\d+\*(\w+)")
_NONE_ID = -1


def parse_uops_json(
    json_path: str | Path, target_arch: str
) -> dict[str, InstructionCost]:
    """Parse compact instructions JSON and return costs for *target_arch*."""
    strings, root = _load_compact_doc(json_path)

    results: dict[str, InstructionCost] = {}
    for instruction_node in _iter_instruction_nodes(root, strings):
        instr_attrs = _attrs_dict(instruction_node[1], strings)
        string_attr = instr_attrs.get("string", "")
        if not string_attr:
            continue

        if "_Z " in string_attr:
            continue

        if any(marker in string_attr for marker in _MEMORY_MARKERS):
            continue

        cost = _extract_measurement(instruction_node, target_arch, strings)
        if cost is not None:
            results[string_attr] = cost

    return results


def list_available_architectures(json_path: str | Path) -> list[str]:
    """Return sorted architecture names present in compact JSON input."""
    strings, root = _load_compact_doc(json_path)

    arch_names: set[str] = set()
    for instruction_node in _iter_instruction_nodes(root, strings):
        for child in instruction_node[4]:
            if _node_tag(child, strings) != "architecture":
                continue
            arch_attrs = _attrs_dict(child[1], strings)
            name = arch_attrs.get("name")
            if name:
                arch_names.add(name)

    return sorted(arch_names)


def _extract_measurement(
    instruction_node: list[object],
    target_arch: str,
    strings: list[str],
) -> InstructionCost | None:
    for arch_node in instruction_node[4]:
        if _node_tag(arch_node, strings) != "architecture":
            continue

        arch_attrs = _attrs_dict(arch_node[1], strings)
        if arch_attrs.get("name") != target_arch:
            continue

        measurement_node = None
        for child in arch_node[4]:
            if _node_tag(child, strings) == "measurement":
                measurement_node = child
                break

        if measurement_node is None:
            return None

        measurement_attrs = _attrs_dict(measurement_node[1], strings)
        latencies = []
        for latency_node in measurement_node[4]:
            if _node_tag(latency_node, strings) != "latency":
                continue
            latency_attrs = _attrs_dict(latency_node[1], strings)
            cycles = latency_attrs.get("cycles")
            if cycles is not None:
                latencies.append(int(cycles))

        latency = float(max(latencies) if latencies else 1)
        throughput = float(measurement_attrs.get("TP_loop", "1.0"))
        ports = _PORT_RE.findall(measurement_attrs.get("ports", ""))

        return InstructionCost(latency=latency, throughput=throughput, ports=ports)

    return None


def _iter_instruction_nodes(
    root: list[object], strings: list[str]
) -> list[list[object]]:
    nodes: list[list[object]] = []
    for extension in root[4]:
        if _node_tag(extension, strings) != "extension":
            continue
        for child in extension[4]:
            if _node_tag(child, strings) == "instruction":
                nodes.append(child)
    return nodes


def _load_compact_doc(path: str | Path) -> tuple[list[str], list[object]]:
    payload = _read_json_payload(path)
    doc = json.loads(payload)

    schema = doc.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema: {schema!r}")

    strings = doc.get("strings")
    root = doc.get("root")

    if not isinstance(strings, list) or not isinstance(root, list):
        raise ValueError("Invalid compact JSON payload")

    return strings, root


def _read_json_payload(path: str | Path) -> str:
    p = Path(path)
    if p.suffix == ".zst":
        dctx = zstandard.ZstdDecompressor()
        data = dctx.decompress(p.read_bytes(), max_output_size=300_000_000)
        return data.decode("utf-8")
    return p.read_text()


def _node_tag(node: list[object], strings: list[str]) -> str:
    tag_id = node[0]
    if not isinstance(tag_id, int):
        raise ValueError("Invalid tag id")
    return strings[tag_id]


def _attrs_dict(attrs: object, strings: list[str]) -> dict[str, str]:
    if not isinstance(attrs, list):
        raise ValueError("Invalid attrs payload")

    result: dict[str, str] = {}
    for pair in attrs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Invalid attr pair")
        key_id, value_id = pair
        if not isinstance(key_id, int) or not isinstance(value_id, int):
            raise ValueError("Invalid attr token")
        result[strings[key_id]] = strings[value_id]
    return result
