"""Compact, lossless XML↔JSON conversion for the uops instruction database."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element

import zstandard

SCHEMA_VERSION = "uops-xml-compact-v1"
_NONE_ID = -1


@dataclass
class _StringInterner:
    """Deduplicate strings into a shared table."""

    strings: list[str] = field(default_factory=list)
    _index: dict[str, int] = field(default_factory=dict)

    def intern(self, value: str) -> int:
        existing = self._index.get(value)
        if existing is not None:
            return existing

        idx = len(self.strings)
        self.strings.append(value)
        self._index[value] = idx
        return idx


def load_xml_root(xml_path: str | Path) -> Element:
    """Load XML root from .xml or .zst-compressed XML files."""
    xml_path = Path(xml_path)

    if xml_path.suffix == ".zst":
        dctx = zstandard.ZstdDecompressor()
        xml_bytes = dctx.decompress(xml_path.read_bytes(), max_output_size=300_000_000)
        return ET.fromstring(xml_bytes)

    return ET.parse(xml_path).getroot()


def xml_root_to_compact_json(root: Element) -> dict[str, object]:
    """Encode an XML root element using the compact tokenized schema."""
    interner = _StringInterner()
    encoded_root = _encode_element(root, interner)

    return {
        "schema": SCHEMA_VERSION,
        "strings": interner.strings,
        "root": encoded_root,
    }


def compact_json_to_xml_root(doc: dict[str, object]) -> Element:
    """Decode a compact JSON document back to an XML root element."""
    schema = doc.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema: {schema!r}")

    strings = doc.get("strings")
    root = doc.get("root")
    if not isinstance(strings, list) or root is None:
        raise ValueError("Invalid compact JSON: missing strings/root")

    return _decode_element(root, strings)


def read_compact_json(path: str | Path) -> dict[str, object]:
    """Read a compact JSON document from disk."""
    return json.loads(Path(path).read_text())


def write_compact_json(doc: dict[str, object], path: str | Path) -> None:
    """Write compact JSON with minimal separators for reduced size."""
    Path(path).write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))


def write_xml_root(root: Element, path: str | Path) -> None:
    """Write an XML tree to disk."""
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def convert_xml_to_json(xml_input: str | Path, json_output: str | Path) -> None:
    """Convert XML(.zst) to compact JSON."""
    root = load_xml_root(xml_input)
    doc = xml_root_to_compact_json(root)
    write_compact_json(doc, json_output)


def convert_json_to_xml(json_input: str | Path, xml_output: str | Path) -> None:
    """Convert compact JSON back to XML."""
    doc = read_compact_json(json_input)
    root = compact_json_to_xml_root(doc)
    write_xml_root(root, xml_output)


def semantic_compare(left: Element, right: Element) -> tuple[bool, str]:
    """Strict semantic XML comparison.

    Preserves:
      - element hierarchy and child order
      - tag names
      - all attribute key/value pairs (attribute order ignored)
      - non-formatting text and tail values

    Ignores:
      - attribute order
      - formatting-only whitespace (text/tail where ``strip() == ""``)
    """
    return _compare_elements(left, right, path=f"/{left.tag}")


def _encode_element(elem: Element, interner: _StringInterner) -> list[object]:
    tag_id = interner.intern(elem.tag)
    attrs = [
        [interner.intern(key), interner.intern(value)]
        for key, value in sorted(elem.attrib.items())
    ]
    text_id = _encode_optional_text(elem.text, interner)
    tail_id = _encode_optional_text(elem.tail, interner)
    children = [_encode_element(child, interner) for child in list(elem)]

    # [tag_id, attrs, text_id, tail_id, children]
    return [tag_id, attrs, text_id, tail_id, children]


def _decode_element(node: object, strings: list[str]) -> Element:
    if not isinstance(node, list) or len(node) != 5:
        raise ValueError("Invalid encoded node")

    tag_id, attrs, text_id, tail_id, children = node
    if not isinstance(tag_id, int):
        raise ValueError("Invalid tag id")

    elem = ET.Element(strings[tag_id])

    if not isinstance(attrs, list):
        raise ValueError("Invalid attrs payload")
    for pair in attrs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Invalid attribute pair")
        key_id, value_id = pair
        if not isinstance(key_id, int) or not isinstance(value_id, int):
            raise ValueError("Invalid attribute token ids")
        elem.set(strings[key_id], strings[value_id])

    elem.text = _decode_optional_text(text_id, strings)
    elem.tail = _decode_optional_text(tail_id, strings)

    if not isinstance(children, list):
        raise ValueError("Invalid children payload")
    for child in children:
        elem.append(_decode_element(child, strings))

    return elem


def _encode_optional_text(text: str | None, interner: _StringInterner) -> int:
    if text is None:
        return _NONE_ID
    return interner.intern(text)


def _decode_optional_text(token: object, strings: list[str]) -> str | None:
    if not isinstance(token, int):
        raise ValueError("Invalid text token")
    if token == _NONE_ID:
        return None
    return strings[token]


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    if value.strip() == "":
        return None
    return value


def _compare_elements(left: Element, right: Element, path: str) -> tuple[bool, str]:
    if left.tag != right.tag:
        return False, f"{path}: tag mismatch ({left.tag!r} != {right.tag!r})"

    if sorted(left.attrib.items()) != sorted(right.attrib.items()):
        return (
            False,
            f"{path}: attributes mismatch ({left.attrib!r} != {right.attrib!r})",
        )

    left_text = _normalize_text(left.text)
    right_text = _normalize_text(right.text)
    if left_text != right_text:
        return False, f"{path}: text mismatch ({left_text!r} != {right_text!r})"

    left_tail = _normalize_text(left.tail)
    right_tail = _normalize_text(right.tail)
    if left_tail != right_tail:
        return False, f"{path}: tail mismatch ({left_tail!r} != {right_tail!r})"

    left_children = list(left)
    right_children = list(right)
    if len(left_children) != len(right_children):
        return (
            False,
            f"{path}: child-count mismatch ({len(left_children)} != {len(right_children)})",
        )

    for idx, (left_child, right_child) in enumerate(zip(left_children, right_children)):
        ok, detail = _compare_elements(
            left_child,
            right_child,
            path=f"{path}/{left_child.tag}[{idx}]",
        )
        if not ok:
            return False, detail

    return True, ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Convert XML(.zst) to compact JSON")
    convert.add_argument("--input", required=True, help="Path to input XML or XML.zst")
    convert.add_argument("--output", required=True, help="Path to output JSON")

    to_xml = sub.add_parser("to-xml", help="Convert compact JSON back to XML")
    to_xml.add_argument("--input", required=True, help="Path to input JSON")
    to_xml.add_argument("--output", required=True, help="Path to output XML")

    verify = sub.add_parser(
        "roundtrip-verify",
        help="Run XML->JSON->XML conversion and semantic comparison",
    )
    verify.add_argument("--input", required=True, help="Path to input XML or XML.zst")
    verify.add_argument(
        "--json-output",
        help="Optional path to save intermediate compact JSON",
    )
    verify.add_argument(
        "--xml-output",
        help="Optional path to save reconstructed XML",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "convert":
            convert_xml_to_json(args.input, args.output)
            return 0

        if args.command == "to-xml":
            convert_json_to_xml(args.input, args.output)
            return 0

        if args.command == "roundtrip-verify":
            original = load_xml_root(args.input)
            doc = xml_root_to_compact_json(original)
            reconstructed = compact_json_to_xml_root(doc)

            if args.json_output:
                write_compact_json(doc, args.json_output)
            if args.xml_output:
                write_xml_root(reconstructed, args.xml_output)

            ok, detail = semantic_compare(original, reconstructed)
            if ok:
                print("Roundtrip semantic comparison: OK")
                return 0

            print(f"Roundtrip semantic comparison: FAIL\n{detail}", file=sys.stderr)
            return 1

        parser.print_help()
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
