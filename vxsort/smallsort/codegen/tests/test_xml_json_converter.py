"""Tests for compact XML↔JSON conversion and semantic roundtrip checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from xml.etree.ElementTree import fromstring

import zstandard

# Ensure src/ is on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from util.xml_json_converter import (  # type: ignore[import-not-found]
    compact_json_to_xml_root,
    load_xml_root,
    main,
    semantic_compare,
    xml_root_to_compact_json,
)


_FIXTURE_XML = """<?xml version=\"1.0\"?>
<root date=\"2026-03-29\">
  <extension name=\"TEST\">
    <instruction asm=\"FOO\" string=\"FOO (MM, MM)\">payload<operand idx=\"1\" type=\"reg\">MM0</operand><operand idx=\"2\" type=\"reg\">MM1</operand>tail</instruction>
  </extension>
</root>
"""


def test_compact_json_roundtrip_is_semantically_equal():
    root = fromstring(_FIXTURE_XML)

    doc = xml_root_to_compact_json(root)
    assert doc["schema"] == "uops-xml-compact-v1"
    assert isinstance(doc["strings"], list)

    roundtrip = compact_json_to_xml_root(doc)
    equal, detail = semantic_compare(root, roundtrip)
    assert equal, detail


def test_semantic_compare_ignores_formatting_whitespace_only():
    left = fromstring("""<root>\n  <a>\n    <b x=\"1\"/>\n  </a>\n</root>""")
    right = fromstring('<root><a><b x="1"/></a></root>')

    equal, detail = semantic_compare(left, right)
    assert equal, detail


def test_semantic_compare_detects_real_semantic_mismatch():
    left = fromstring('<root><a x="1"/></root>')
    right = fromstring('<root><a x="2"/></root>')

    equal, detail = semantic_compare(left, right)
    assert not equal
    assert "attributes" in detail


def test_load_xml_root_supports_zstd_xml(tmp_path: Path):
    xml_path = tmp_path / "mini.xml"
    zst_path = tmp_path / "mini.xml.zst"

    xml_path.write_text(_FIXTURE_XML)

    compressor = zstandard.ZstdCompressor(level=3)
    zst_path.write_bytes(compressor.compress(xml_path.read_bytes()))

    root = load_xml_root(zst_path)
    assert root.tag == "root"
    assert root.attrib["date"] == "2026-03-29"


def test_cli_roundtrip_verify_success_exit_code(tmp_path: Path):
    xml_path = tmp_path / "mini.xml"
    xml_path.write_text(_FIXTURE_XML)

    exit_code = main(["roundtrip-verify", "--input", str(xml_path)])
    assert exit_code == 0
