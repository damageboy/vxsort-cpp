"""Tests for intrinsic_registry, uops_parser, and cost_model arch overlay."""

from __future__ import annotations

import os
import sys

import pytest

# Ensure src/ is on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cost_model import CostModel, InstructionCost, resolve_arch_name, _ARCH_ALIASES
from intrinsic_registry import get_intrinsic_registry, xml_string_key
from uops_parser import (
    parse_uops_xml,
    list_available_architectures,
    load_xml_root,
    _build_xml_index,
)

# Path to the XML database.
_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "instructions.xml.zst")
_HAS_XML = os.path.exists(_XML_PATH)

# Module-level cache: decompress and parse the XML once, reuse across all tests.
_cached_xml_root = None


def _get_xml_root():
    """Return the cached XML root, loading it on first call."""
    global _cached_xml_root
    if _cached_xml_root is None:
        _cached_xml_root = load_xml_root(_XML_PATH)
    return _cached_xml_root


@pytest.fixture(autouse=True)
def _cache_xml_root(monkeypatch):
    """Monkeypatch load_xml_root so every caller in uops_parser reuses the cached root."""
    if not _HAS_XML:
        return
    import uops_parser as _uops_mod

    root = _get_xml_root()
    monkeypatch.setattr(_uops_mod, "load_xml_root", lambda _path: root)


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
    # These are all the intrinsics that were in the old asm_exporter mapping.
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


@pytest.mark.skipif(not _HAS_XML, reason="instructions.xml.zst not found")
def test_xml_string_keys_exist_in_xml():
    """Every registry entry's xml_string_key() exists in the XML index."""
    root = _get_xml_root()
    index = _build_xml_index(root)

    registry = get_intrinsic_registry()
    missing = []
    for name, info in registry.items():
        key = xml_string_key(info)
        if key not in index:
            missing.append((name, key))

    # Some AVX-512VL instructions at 256-bit width only exist as EVEX-encoded
    # forms in the XML (e.g., VPERMQ_EVEX (YMM, YMM, YMM) instead of
    # VPERMQ (YMM, YMM, YMM)). Accept those as present.
    actual_missing = []
    for name, key in missing:
        info = registry[name]
        evex_key = key.replace(info.asm_mnemonic, f"{info.asm_mnemonic}_EVEX", 1)
        if evex_key not in index:
            actual_missing.append((name, key))

    if actual_missing:
        msg = "\n".join(f"  {n} -> {k}" for n, k in actual_missing)
        pytest.fail(f"Registry entries not found in XML index:\n{msg}")


# ---------------------------------------------------------------------------
# uops_parser tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_XML, reason="instructions.xml.zst not found")
def test_parse_uops_xml_tgl():
    """Parse TGL arch and verify known instruction costs."""
    costs = parse_uops_xml(_XML_PATH, "TGL")
    assert len(costs) > 0, "Expected non-empty cost dict for TGL"

    # VPERMILPS (YMM, YMM, I8) should have latency=1
    key = "VPERMILPS (YMM, YMM, I8)"
    assert key in costs, f"Expected {key} in TGL costs"
    assert costs[key].latency == 1.0

    # VPERMD (YMM, YMM, YMM) should have latency=3
    key = "VPERMD (YMM, YMM, YMM)"
    assert key in costs, f"Expected {key} in TGL costs"
    assert costs[key].latency == 3.0


@pytest.mark.skipif(not _HAS_XML, reason="instructions.xml.zst not found")
def test_parse_uops_xml_zen4():
    """Parse ZEN4 arch and verify it returns data."""
    costs = parse_uops_xml(_XML_PATH, "ZEN4")
    assert len(costs) > 0, "Expected non-empty cost dict for ZEN4"

    # At minimum VPERMILPS should be present
    key = "VPERMILPS (YMM, YMM, I8)"
    assert key in costs


@pytest.mark.skipif(not _HAS_XML, reason="instructions.xml.zst not found")
def test_parse_uops_xml_missing_arch():
    """Bogus architecture returns empty dict."""
    costs = parse_uops_xml(_XML_PATH, "NONEXISTENT_ARCH_42")
    assert costs == {}


@pytest.mark.skipif(not _HAS_XML, reason="instructions.xml.zst not found")
def test_list_available_architectures():
    """list_available_architectures returns known arch names."""
    archs = list_available_architectures(_XML_PATH)
    assert "TGL" in archs
    assert "ZEN4" in archs
    assert "HSW" in archs
    assert len(archs) >= 20  # We know there are 28


# ---------------------------------------------------------------------------
# CostModel tests
# ---------------------------------------------------------------------------


def test_cost_model_generic_unchanged():
    """CostModel("generic") produces same costs as before the refactor."""
    cm = CostModel("generic")
    # Spot-check a few known values from the original _init_generic_costs
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
    assert cm.instruction_costs["_mm512_shuffle_i32x4"] == InstructionCost(
        3.0, 1.0, ["p5"]
    )


@pytest.mark.skipif(not _HAS_XML, reason="instructions.xml.zst not found")
def test_cost_model_with_arch_overlay():
    """CostModel("TGL") overlays arch-specific costs on top of generic."""
    cm = CostModel("TGL")
    # Should have costs loaded
    assert len(cm.instruction_costs) > 0
    # The TGL cost for _mm256_permute_ps should come from XML (latency=1)
    cost = cm.instruction_costs["_mm256_permute_ps"]
    assert cost.latency == 1.0
    # Throughput may differ from generic
    assert cost.throughput > 0


# ---------------------------------------------------------------------------
# Arch alias resolution
# ---------------------------------------------------------------------------


def test_resolve_arch_aliases():
    """Friendly CPU names resolve to XML names."""
    assert resolve_arch_name("tigerlake") == "TGL"
    assert resolve_arch_name("zen4") == "ZEN4"
    assert resolve_arch_name("haswell") == "HSW"
    assert resolve_arch_name("skylake-x") == "SKX"
    assert resolve_arch_name("generic") is None
    # Exact XML names pass through
    assert resolve_arch_name("TGL") == "TGL"
    assert resolve_arch_name("ZEN4") == "ZEN4"


def test_arch_aliases_complete():
    """All aliases resolve to valid strings."""
    for friendly, xml_name in _ARCH_ALIASES.items():
        assert isinstance(xml_name, str)
        assert len(xml_name) > 0
        assert resolve_arch_name(friendly) == xml_name


# ---------------------------------------------------------------------------
# asm_exporter mnemonic regression
# ---------------------------------------------------------------------------


def test_asm_exporter_mnemonics_unchanged():
    """Regression: _intrinsic_to_asm_mnemonic returns same values as before."""
    from asm_exporter import _intrinsic_to_asm_mnemonic

    # Spot-check the old mapping values (all lowercase)
    expected = {
        "_mm256_permute4x64_epi64": "vpermq",
        "_mm256_permute_ps": "vpermilps",
        "_mm256_shuffle_ps": "vshufps",
        "_mm256_unpacklo_epi32": "vpunpckldq",
        "_mm256_blend_ps": "vblendps",
        "_mm256_alignr_epi32": "valignd",
        "_mm512_permutexvar_epi32": "vpermd",
        "_mm512_mask_shuffle_ps": "vshufps",
        "_mm512_shuffle_i32x4": "vshufi32x4",
        "_mm512_mask_blend_ps": "vblendmps",
        "_mm512_permutex2var_epi32": "vpermi2d",
        "_mm256_permute2x128_si256": "vperm2i128",
    }
    for intrinsic, mnemonic in expected.items():
        assert (
            _intrinsic_to_asm_mnemonic(intrinsic) == mnemonic
        ), f"{intrinsic}: expected {mnemonic}, got {_intrinsic_to_asm_mnemonic(intrinsic)}"

    # Unknown intrinsics should return the name unchanged
    assert _intrinsic_to_asm_mnemonic("_mm_unknown_op") == "_mm_unknown_op"


def test_asm_exporter_metadata_unchanged():
    """Regression: _get_instruction_metadata returns same values as before."""
    from asm_exporter import _get_instruction_metadata

    # shuffle2 instructions
    for name in ["_mm256_shuffle_pd", "_mm512_shuffle_pd", "_mm256_permute_pd"]:
        meta = _get_instruction_metadata(name)
        assert meta["imm_type"] == "shuffle2", f"{name} should be shuffle2"

    # shuffle4 instructions
    for name in ["_mm256_shuffle_ps", "_mm512_shuffle_ps", "_mm512_shuffle_i32x4"]:
        meta = _get_instruction_metadata(name)
        assert meta["imm_type"] == "shuffle4", f"{name} should be shuffle4"

    # control vector instructions
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

    # Binary (blend) instructions
    for name in ["_mm256_blend_ps", "_mm256_blend_pd", "_mm256_blend_epi32"]:
        meta = _get_instruction_metadata(name)
        assert meta["imm_type"] == "binary", f"{name} should be binary"

    # Unknown instruction
    meta = _get_instruction_metadata("_mm_unknown_op")
    assert meta["imm_type"] is None
    assert meta["control_vector_width"] is None
