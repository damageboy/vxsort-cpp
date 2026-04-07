"""Tests for intrinsic registry, JSON uops parser, and cost-model overlay."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from xml.etree.ElementTree import fromstring

import pytest
import zstandard

# Ensure src/ is on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cost_model import CostModel, InstructionCost, resolve_arch_name, _ARCH_ALIASES
from intrinsic_registry import get_intrinsic_registry, xml_string_key
from util.uops_parser import (
    find_uops_database_path,
    list_available_architectures,
    parse_uops_database,
    parse_uops_json,
)
from util.xml_json_converter import write_compact_json, xml_root_to_compact_json

_DB_PATH = find_uops_database_path(
    base_dir=Path(__file__).resolve().parents[1],
    prefer="json",
)
_HAS_DB = _DB_PATH is not None


# ---------------------------------------------------------------------------
# Registry coverage tests
# ---------------------------------------------------------------------------


def test_registry_covers_all_cost_model_intrinsics():
    """Every intrinsic in _init_generic_costs() has a registry entry."""
    cm = CostModel("generic")
    registry = get_intrinsic_registry()
    missing = set(cm.instruction_costs.keys()) - set(registry.keys())
    assert not missing, f"Intrinsics in cost_model but not in registry: {missing}"


def test_registry_covers_all_asm_exporter_intrinsics():
    """Every intrinsic formerly in _intrinsic_to_asm_mnemonic() has a registry entry."""
    old_mapping_intrinsics = {
        "_mm256_permute4x64_epi64",
        "_mm256_permute_ps",
        "_mm256_permute_pd",
        "_mm256_permutexvar_epi32",
        "_mm256_permutexvar_epi64",
        "_mm256_permutevar_ps",
        "_mm256_permutevar_pd",
        "_mm512_permutexvar_epi64",
        "_mm512_permutexvar_epi32",
        "_mm512_mask_permutexvar_epi32",
        "_mm512_mask_permutexvar_epi64",
        "_mm512_permutex2var_epi32",
        "_mm512_permutex2var_epi64",
        "_mm512_mask_permutex2var_epi32",
        "_mm512_mask_permutex2var_epi64",
        "_mm512_permute_pd",
        "_mm512_mask_permute_ps",
        "_mm512_mask_permute_pd",
        "_mm512_permutevar_ps",
        "_mm512_permutevar_pd",
        "_mm512_mask_permutevar_ps",
        "_mm512_mask_permutevar_pd",
        "_mm256_shuffle_ps",
        "_mm256_shuffle_pd",
        "_mm256_shuffle_epi32",
        "_mm512_shuffle_ps",
        "_mm512_shuffle_pd",
        "_mm512_mask_shuffle_ps",
        "_mm512_mask_shuffle_pd",
        "_mm512_shuffle_epi32",
        "_mm256_unpacklo_epi32",
        "_mm256_unpackhi_epi32",
        "_mm256_unpacklo_epi64",
        "_mm256_unpackhi_epi64",
        "_mm256_unpacklo_ps",
        "_mm256_unpackhi_ps",
        "_mm512_unpacklo_epi32",
        "_mm512_unpackhi_epi32",
        "_mm512_unpacklo_epi64",
        "_mm512_unpackhi_epi64",
        "_mm512_mask_unpacklo_epi32",
        "_mm512_mask_unpackhi_epi32",
        "_mm512_mask_unpacklo_epi64",
        "_mm512_mask_unpackhi_epi64",
        "_mm256_permute2x128_si256",
        "_mm256_permute2f128_ps",
        "_mm256_blend_ps",
        "_mm256_blend_pd",
        "_mm256_blendv_ps",
        "_mm256_blendv_pd",
        "_mm256_blend_epi32",
        "_mm512_mask_blend_ps",
        "_mm512_mask_blend_epi32",
        "_mm256_alignr_epi32",
        "_mm512_alignr_epi32",
        "_mm256_alignr_epi64",
        "_mm512_alignr_epi64",
        "_mm512_mask_alignr_epi32",
        "_mm512_mask_alignr_epi64",
        "_mm512_shuffle_i32x4",
        "_mm512_mask_shuffle_i32x4",
        "_mm256_min_ps",
        "_mm256_max_ps",
        "_mm512_min_ps",
        "_mm512_max_ps",
    }
    registry = get_intrinsic_registry()
    missing = old_mapping_intrinsics - set(registry.keys())
    assert not missing, f"Intrinsics in old asm_exporter but not in registry: {missing}"


@pytest.mark.skipif(not _HAS_DB, reason="instructions.json(.zst) not found")
def test_xml_string_keys_exist_in_json_database():
    """Every registry xml_string_key exists in the compact JSON database parse."""
    costs = parse_uops_database(_DB_PATH, "TGL")
    registry = get_intrinsic_registry()

    missing = []
    for name, info in registry.items():
        key = xml_string_key(info)
        evex_key = key.replace(info.asm_mnemonic, f"{info.asm_mnemonic}_EVEX", 1)
        if key not in costs and evex_key not in costs:
            missing.append((name, key))

    if missing:
        msg = "\n".join(f"  {n} -> {k}" for n, k in missing)
        pytest.fail(f"Registry entries not found in JSON-parsed key set:\n{msg}")


# ---------------------------------------------------------------------------
# uops_parser tests
# ---------------------------------------------------------------------------


def test_parse_uops_json_from_fixture(tmp_path: Path):
    """Parse compact JSON fixture and verify known instruction costs."""
    xml_text = """
<root>
  <extension name="AVX2">
    <instruction string="VPERMILPS (YMM, YMM, I8)">
      <architecture name="TGL">
        <measurement TP_loop="0.5" ports="1*p5+1*p23">
          <latency cycles="1"/>
          <latency cycles="2"/>
        </measurement>
      </architecture>
    </instruction>
    <instruction string="VPERMILPS_Z (YMM, YMM, I8)">
      <architecture name="TGL">
        <measurement TP_loop="1.0" ports="1*p0">
          <latency cycles="3"/>
        </measurement>
      </architecture>
    </instruction>
  </extension>
</root>
"""
    json_path = tmp_path / "mini.json"
    write_compact_json(xml_root_to_compact_json(fromstring(xml_text)), json_path)

    costs = parse_uops_json(json_path, "TGL")
    assert "VPERMILPS (YMM, YMM, I8)" in costs
    assert costs["VPERMILPS (YMM, YMM, I8)"].latency == 2.0
    assert costs["VPERMILPS (YMM, YMM, I8)"].throughput == 0.5
    assert costs["VPERMILPS (YMM, YMM, I8)"].ports == ["p5", "p23"]
    assert "VPERMILPS_Z (YMM, YMM, I8)" not in costs


def test_parse_uops_database_dispatches_json_zst(tmp_path: Path):
    """Dispatch recognizes .json.zst as JSON database input."""
    xml_text = """
<root>
  <extension name="AVX2">
    <instruction string="VPERMD (YMM, YMM, YMM)">
      <architecture name="TGL">
        <measurement TP_loop="1.0" ports="1*p5">
          <latency cycles="3"/>
        </measurement>
      </architecture>
    </instruction>
  </extension>
</root>
"""
    json_path = tmp_path / "mini.json"
    json_zst_path = tmp_path / "mini.json.zst"

    write_compact_json(xml_root_to_compact_json(fromstring(xml_text)), json_path)
    compressor = zstandard.ZstdCompressor(level=3)
    json_zst_path.write_bytes(compressor.compress(json_path.read_bytes()))

    costs = parse_uops_database(json_zst_path, "TGL")
    assert "VPERMD (YMM, YMM, YMM)" in costs
    assert costs["VPERMD (YMM, YMM, YMM)"].latency == 3.0


def test_parse_uops_database_rejects_xml_path(tmp_path: Path):
    """XML database paths are intentionally unsupported at runtime."""
    xml_path = tmp_path / "instructions.xml.zst"
    xml_path.write_bytes(b"not used")

    with pytest.raises(ValueError, match="Only .json/.json.zst"):
        parse_uops_database(xml_path, "TGL")


def test_list_available_architectures_from_json(tmp_path: Path):
    """Architecture listing works for compact JSON input."""
    xml_text = """
<root>
  <extension name="AVX2">
    <instruction string="VPERMD (YMM, YMM, YMM)">
      <architecture name="TGL"><measurement TP_loop="1.0" ports="1*p5"/></architecture>
      <architecture name="ZEN4"><measurement TP_loop="1.0" ports="1*p5"/></architecture>
    </instruction>
  </extension>
</root>
"""
    json_path = tmp_path / "mini.json"
    write_compact_json(xml_root_to_compact_json(fromstring(xml_text)), json_path)

    archs = list_available_architectures(json_path)
    assert archs == ["TGL", "ZEN4"]


def test_find_uops_database_path_prefers_json_zst_when_present(tmp_path: Path):
    """Auto mode prefers compressed JSON when both JSON variants exist."""
    json_zst_path = tmp_path / "instructions.json.zst"
    json_path = tmp_path / "instructions.json"
    json_zst_path.write_text("placeholder")
    json_path.write_text("{}")

    selected = find_uops_database_path(base_dir=tmp_path, prefer="auto")
    assert selected == str(json_zst_path)


def test_find_uops_database_path_env_override(monkeypatch, tmp_path: Path):
    """Explicit environment path override takes precedence."""
    env_choice = tmp_path / "custom.json"
    env_choice.write_text("{}")

    monkeypatch.setenv("VXSORT_UOPS_DB_PATH", str(env_choice))
    selected = find_uops_database_path(base_dir=tmp_path, prefer="json")
    assert selected == str(env_choice)


# ---------------------------------------------------------------------------
# CostModel tests
# ---------------------------------------------------------------------------


def test_cost_model_generic_unchanged():
    """CostModel('generic') keeps the expected baseline costs."""
    cm = CostModel("generic")
    assert cm.instruction_costs["_mm256_permute_ps"] == InstructionCost(
        1.0, 1.0, ["p5"]
    )
    assert cm.instruction_costs["_mm256_blend_ps"] == InstructionCost(
        1.0, 0.33, ["p015"]
    )
    assert cm.instruction_costs["_mm256_blendv_ps"] == InstructionCost(
        2.0, 0.67, ["p015"]
    )
    assert cm.instruction_costs["_mm512_permutexvar_epi32"] == InstructionCost(
        3.0, 1.0, ["p5"]
    )


@pytest.mark.skipif(not _HAS_DB, reason="instructions.json(.zst) not found")
def test_cost_model_with_arch_overlay():
    """CostModel overlays architecture-specific costs from JSON DB."""
    cm = CostModel("TGL")
    assert len(cm.instruction_costs) > 0
    cost = cm.instruction_costs["_mm256_permute_ps"]
    assert cost.latency == 1.0
    assert cost.throughput > 0


# ---------------------------------------------------------------------------
# Arch alias resolution
# ---------------------------------------------------------------------------


def test_resolve_arch_aliases():
    assert resolve_arch_name("tigerlake") == "TGL"
    assert resolve_arch_name("zen4") == "ZEN4"
    assert resolve_arch_name("haswell") == "HSW"
    assert resolve_arch_name("skylake-x") == "SKX"
    assert resolve_arch_name("generic") is None
    assert resolve_arch_name("TGL") == "TGL"


def test_arch_aliases_complete():
    for friendly, xml_name in _ARCH_ALIASES.items():
        assert isinstance(xml_name, str)
        assert len(xml_name) > 0
        assert resolve_arch_name(friendly) == xml_name


# ---------------------------------------------------------------------------
# asm_exporter regression
# ---------------------------------------------------------------------------


def test_asm_exporter_mnemonics_unchanged():
    from asm_exporter import _intrinsic_to_asm_mnemonic

    expected = {
        "_mm256_permute4x64_epi64": "vpermq",
        "_mm256_permute_ps": "vpermilps",
        "_mm256_shuffle_ps": "vshufps",
        "_mm256_unpacklo_epi32": "vpunpckldq",
        "_mm256_blend_ps": "vblendps",
        "_mm256_alignr_epi8": "vpalignr",
        "_mm256_alignr_epi32": "valignd",
        "_mm512_permutexvar_epi32": "vpermd",
        "_mm512_mask_shuffle_ps": "vshufps",
        "_mm512_shuffle_i32x4": "vshufi32x4",
        "_mm512_mask_blend_ps": "vblendmps",
        "_mm512_permutex2var_epi32": "vpermi2d",
        "_mm256_permute2x128_si256": "vperm2i128",
    }
    for intrinsic, mnemonic in expected.items():
        assert _intrinsic_to_asm_mnemonic(intrinsic) == mnemonic

    assert _intrinsic_to_asm_mnemonic("_mm_unknown_op") == "_mm_unknown_op"


def test_asm_exporter_metadata_unchanged():
    from asm_exporter import _get_instruction_metadata

    for name in ["_mm256_shuffle_pd", "_mm512_shuffle_pd", "_mm256_permute_pd"]:
        assert _get_instruction_metadata(name)["imm_type"] == "shuffle2"

    for name in ["_mm256_shuffle_ps", "_mm512_shuffle_ps", "_mm512_shuffle_i32x4"]:
        assert _get_instruction_metadata(name)["imm_type"] == "shuffle4"

    assert (
        _get_instruction_metadata("_mm256_permutexvar_epi32")["control_vector_width"]
        == 32
    )
    assert (
        _get_instruction_metadata("_mm256_permutexvar_epi64")["control_vector_width"]
        == 64
    )
    assert (
        _get_instruction_metadata("_mm512_permutex2var_epi32")["control_vector_width"]
        == 32
    )

    for name in ["_mm256_blend_ps", "_mm256_blend_pd", "_mm256_blend_epi32"]:
        assert _get_instruction_metadata(name)["imm_type"] == "binary"

    meta = _get_instruction_metadata("_mm_unknown_op")
    assert meta["imm_type"] is None
    assert meta["control_vector_width"] is None
