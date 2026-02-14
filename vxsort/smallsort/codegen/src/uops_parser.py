"""Parser for the uops.info XML instruction database.

Parses the zstd-compressed XML file from https://uops.info/ and extracts
instruction cost data (latency, throughput, execution ports) for a given
CPU microarchitecture.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree.ElementTree import Element, fromstring

import zstandard

try:
    from .cost_model import InstructionCost
except ImportError:
    from cost_model import InstructionCost

# Memory operand markers that indicate non-register forms.
# Covers plain memory (M32, M64, M128, M256, M512) and broadcast (M32_1toN, M64_1toN).
# These never collide with register names (XMM, YMM, ZMM) since those have letter prefixes.
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

# Regex for port specifications like "1*p5" or "2*p015".
_PORT_RE = re.compile(r"\d+\*(\w+)")


def load_xml_root(xml_path: str | Path) -> Element:
    """Decompress and parse the uops.info XML database.

    Args:
        xml_path: Path to the zstd-compressed ``instructions.xml.zst`` file.

    Returns:
        Root :class:`Element` of the parsed XML tree.
    """
    xml_path = Path(xml_path)
    dctx = zstandard.ZstdDecompressor()
    with open(xml_path, "rb") as f:
        xml_bytes = dctx.decompress(f.read(), max_output_size=200_000_000)
    return fromstring(xml_bytes)


def parse_uops_xml(
    xml_path: str | Path, target_arch: str
) -> dict[str, InstructionCost]:
    """Parse the uops.info XML database and return costs for *target_arch*.

    Args:
        xml_path: Path to the zstd-compressed ``instructions.xml.zst`` file.
        target_arch: Architecture name (e.g. ``"HSW"``, ``"TGL"``, ``"SKX"``).

    Returns:
        Dictionary mapping the XML ``string`` attribute (e.g.
        ``"VPERMD (YMM, YMM, YMM)"``) to an :class:`InstructionCost`.
        Only register-register forms are included (no memory operands,
        no zeroing-masking ``_Z`` forms).
    """
    root = load_xml_root(xml_path)
    index = _build_xml_index(root)

    results: dict[str, InstructionCost] = {}
    for key, elem in index.items():
        cost = _extract_measurement(elem, target_arch)
        if cost is not None:
            results[key] = cost

    return results


def _build_xml_index(root: Element) -> dict[str, Element]:
    """Build an index from XML ``string`` attribute to instruction Element.

    Skips entries that:
    - contain memory operand markers (M128, M256, M512, M32_, M64_), or
    - start with a ``_Z `` prefix (zeroing-masking forms).
    """
    index: dict[str, Element] = {}

    for extension in root.iter("extension"):
        for instruction in extension.iter("instruction"):
            string_attr = instruction.get("string", "")
            if not string_attr:
                continue

            # Skip zeroing-masking forms.
            if "_Z " in string_attr:
                continue

            # Skip memory operand forms.
            if any(marker in string_attr for marker in _MEMORY_MARKERS):
                continue

            index[string_attr] = instruction

    return index


def _extract_measurement(elem: Element, arch_name: str) -> InstructionCost | None:
    """Extract measurement data for *arch_name* from an instruction element.

    Returns ``None`` if the architecture or its measurement is not found.
    """
    for arch_elem in elem.findall("architecture"):
        if arch_elem.get("name") == arch_name:
            measurement = arch_elem.find("measurement")
            if measurement is None:
                return None

            # Latency: worst-case across all dependency chains.
            # Some latency elements use cycles_addr/cycles_mem instead of
            # cycles (memory-operand measurements); skip those.
            latencies = [
                int(lat.attrib["cycles"])
                for lat in measurement.findall("latency")
                if "cycles" in lat.attrib
            ]
            latency = max(latencies) if latencies else 1

            # Throughput: reciprocal throughput from the measurement loop.
            throughput = float(measurement.attrib.get("TP_loop", "1.0"))

            # Ports: parse strings like "1*p5" or "1*p23+1*p5".
            ports_str = measurement.attrib.get("ports", "")
            ports = _PORT_RE.findall(ports_str)

            return InstructionCost(
                latency=float(latency),
                throughput=throughput,
                ports=ports,
            )

    return None


def list_available_architectures(xml_path: str | Path) -> list[str]:
    """Return sorted list of unique architecture names in the XML database.

    Args:
        xml_path: Path to the zstd-compressed ``instructions.xml.zst`` file.

    Returns:
        Sorted list of architecture name strings (e.g.
        ``["ADL-E", "ADL-P", "BDW", "CFL", ...]``).
    """
    root = load_xml_root(xml_path)

    arch_names: set[str] = set()
    for arch_elem in root.iter("architecture"):
        name = arch_elem.get("name")
        if name:
            arch_names.add(name)

    return sorted(arch_names)
