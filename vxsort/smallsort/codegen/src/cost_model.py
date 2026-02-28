"""Cost model for AVX instructions based on CPU microarchitecture."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class InstructionCost:
    """Cost characteristics for a single instruction."""

    latency: float  # Latency in cycles
    throughput: float  # Reciprocal throughput (CPI)
    ports: list[str]  # Execution ports (e.g., ["p0", "p1", "p5"])

    def __repr__(self):
        return f"Cost(lat={self.latency}, tput={self.throughput}, ports={self.ports})"


@dataclass(frozen=True)
class ArchInfo:
    """Single source of truth for a target CPU architecture."""

    canonical: str  # XML arch name, e.g. "SKX"
    aliases: tuple[str, ...]  # User-friendly names, e.g. ("skylake-x",)
    osaca_code: str | None  # OSACA model name, or None
    vendor: str  # "Intel" or "AMD"


# Authoritative registry of all supported architectures.
_ARCH_REGISTRY: list[ArchInfo] = [
    # Intel
    ArchInfo("SNB", ("sandybridge",), "SNB", "Intel"),
    ArchInfo("IVB", ("ivybridge",), "IVB", "Intel"),
    ArchInfo("HSW", ("haswell",), "HSW", "Intel"),
    ArchInfo("BDW", ("broadwell",), "BDW", "Intel"),
    ArchInfo("SKL", ("skylake",), "SKX", "Intel"),
    ArchInfo("KBL", ("kabylake",), "SKX", "Intel"),
    ArchInfo("CFL", ("coffeelake",), "SKX", "Intel"),
    ArchInfo("CNL", ("cannonlake",), "ICL", "Intel"),
    ArchInfo("ICL", ("icelake",), "ICL", "Intel"),
    ArchInfo("TGL", ("tigerlake",), "ICL", "Intel"),
    ArchInfo("RKL", ("rocketlake",), "ICL", "Intel"),
    ArchInfo("SKX", ("skylake-x",), "SKX", "Intel"),
    ArchInfo("CLX", ("cascadelake",), "CSX", "Intel"),
    ArchInfo("ADL-P", ("alderlake-p",), "SPR", "Intel"),
    ArchInfo("ADL-E", ("alderlake-e",), "SPR", "Intel"),
    # AMD
    ArchInfo("ZEN+", ("zen+",), "ZEN1", "AMD"),
    ArchInfo("ZEN1", ("zen1",), "ZEN1", "AMD"),
    ArchInfo("ZEN2", ("zen2",), "ZEN2", "AMD"),
    ArchInfo("ZEN3", ("zen3",), "ZEN3", "AMD"),
    ArchInfo("ZEN4", ("zen4",), "ZEN4", "AMD"),
    ArchInfo("ZEN5", ("zen5",), "ZEN5", "AMD"),
]

# Derived lookup: friendly name (lowercase) -> canonical XML name.
_ARCH_ALIASES: dict[str, str] = {
    alias: info.canonical for info in _ARCH_REGISTRY for alias in info.aliases
}

# Derived lookup: canonical XML name -> ArchInfo.
_ARCH_BY_CANONICAL: dict[str, ArchInfo] = {
    info.canonical: info for info in _ARCH_REGISTRY
}


def _find_xml_path() -> str | None:
    """Find instructions.xml.zst relative to this module's directory."""
    module_dir = os.path.dirname(__file__)
    candidate = os.path.join(module_dir, "..", "instructions.xml.zst")
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return None


def resolve_arch_name(target_cpu: str) -> str | None:
    """Resolve a target CPU name to the XML architecture name.

    Accepts either a friendly name (e.g. "tigerlake") or an exact XML name
    (e.g. "TGL"). Returns None for "generic".
    """
    if target_cpu == "generic":
        return None
    lower = target_cpu.lower()
    if lower in _ARCH_ALIASES:
        return _ARCH_ALIASES[lower]
    # Accept exact XML names (case-sensitive)
    return target_cpu


def resolve_osaca_arch(target_cpu: str) -> str | None:
    """Resolve a --target-cpu value to an OSACA architecture code.

    Returns None if the architecture is not supported by OSACA.
    """
    canonical = resolve_arch_name(target_cpu)
    if canonical is None:
        return None
    info = _ARCH_BY_CANONICAL.get(canonical)
    if info is None:
        return None
    return info.osaca_code


def get_supported_cpus() -> list[ArchInfo]:
    """Return all supported CPU architectures."""
    return list(_ARCH_REGISTRY)


class CostModel:
    """Cost model for evaluating instruction sequences."""

    def __init__(self, target_cpu: str = "generic"):
        self.target_cpu = target_cpu
        self.instruction_costs: dict[str, InstructionCost] = {}
        self._initialize_costs()

    def _initialize_costs(self):
        """Initialize instruction costs: generic defaults + optional arch overlay."""
        self._init_generic_costs()

        arch_name = resolve_arch_name(self.target_cpu)
        if arch_name is None:
            return

        xml_path = _find_xml_path()
        if xml_path is None:
            print("Warning: instructions.xml.zst not found, using generic costs")
            return

        self._overlay_arch_costs(xml_path, arch_name)

    def _overlay_arch_costs(self, xml_path: str, arch_name: str):
        """Overlay architecture-specific costs from the uops.info XML."""
        try:
            from intrinsic_registry import get_intrinsic_registry, xml_string_key
            from uops_parser import parse_uops_xml
        except ImportError:
            from .intrinsic_registry import get_intrinsic_registry, xml_string_key
            from .uops_parser import parse_uops_xml

        xml_costs = parse_uops_xml(xml_path, arch_name)
        if not xml_costs:
            print(
                f"Warning: No data for architecture '{arch_name}', using generic costs"
            )
            return

        registry = get_intrinsic_registry()
        overlaid = 0
        for intrinsic_name, info in registry.items():
            key = xml_string_key(info)
            if key in xml_costs:
                self.instruction_costs[intrinsic_name] = xml_costs[key]
                overlaid += 1
            else:
                # Some AVX-512VL instructions at 256-bit width only exist as
                # EVEX-encoded forms in the XML (e.g., VPERMQ_EVEX (YMM, ...)).
                evex_key = key.replace(
                    info.asm_mnemonic, f"{info.asm_mnemonic}_EVEX", 1
                )
                if evex_key in xml_costs:
                    self.instruction_costs[intrinsic_name] = xml_costs[evex_key]
                    overlaid += 1

        print(f"Loaded {overlaid} instruction costs from uops.info for {arch_name}")

    def _init_generic_costs(self):
        """Generic/simple cost model - just instruction count."""
        # AVX2 instructions (approximations)
        self.instruction_costs.update(
            {
                "_mm256_permutexvar_epi32": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm256_permute4x64_epi64": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm256_permute_ps": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm256_shuffle_ps": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm256_unpacklo_epi32": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm256_unpackhi_epi32": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm256_permute2x128_si256": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm256_blend_ps": InstructionCost(1.0, 0.33, ["p015"]),
                "_mm256_blendv_ps": InstructionCost(2.0, 0.67, ["p015"]),
                "_mm256_alignr_epi32": InstructionCost(1.0, 1.0, ["p5"]),
                # AVX2 i64 instructions
                "_mm256_permutexvar_epi64": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm256_permute_pd": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm256_permutevar_pd": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm256_shuffle_pd": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm256_unpacklo_epi64": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm256_unpackhi_epi64": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm256_blend_pd": InstructionCost(1.0, 0.33, ["p015"]),
                "_mm256_blendv_pd": InstructionCost(2.0, 0.67, ["p015"]),
                "_mm256_alignr_epi64": InstructionCost(1.0, 1.0, ["p5"]),
            }
        )

        # AVX512 instructions (approximations)
        self.instruction_costs.update(
            {
                "_mm512_permutexvar_epi32": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm512_permutexvar_epi64": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm512_permutex2var_epi32": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm512_permutex2var_epi64": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm512_permute_ps": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_shuffle_ps": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_unpacklo_epi32": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_unpackhi_epi32": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_shuffle_i32x4": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm512_mask_permutexvar_epi32": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm512_mask_permutexvar_epi64": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm512_mask_permutex2var_epi32": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm512_mask_permutex2var_epi64": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm512_permute_pd": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_permute_ps": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_permute_pd": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_permutevar_ps": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_permutevar_pd": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_permutevar_ps": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_permutevar_pd": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_shuffle_pd": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_shuffle_ps": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_shuffle_pd": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_unpacklo_epi64": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_unpackhi_epi64": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_unpacklo_epi32": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_unpackhi_epi32": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_unpacklo_epi64": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_unpackhi_epi64": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_shuffle_i32x4": InstructionCost(3.0, 1.0, ["p5"]),
                "_mm512_alignr_epi32": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_alignr_epi64": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_alignr_epi32": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_alignr_epi64": InstructionCost(1.0, 1.0, ["p5"]),
            }
        )

    def get_instruction_cost(self, intrinsic_name: str) -> InstructionCost:
        """Get cost for a specific intrinsic."""
        return self.instruction_costs.get(
            intrinsic_name,
            InstructionCost(latency=1.0, throughput=1.0, ports=["unknown"]),
        )

    def calculate_gadget_cost(self, gadget) -> float:
        """Calculate cost for a permutation gadget (sum of latencies).

        Uses the deduplicated unified instruction list so that shared
        instructions between top and bottom sides are counted only once.
        """
        unified, _, _ = gadget.unified_instructions()
        return sum(
            self.get_instruction_cost(inst.intrinsic_name).latency for inst in unified
        )
