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
    vendor: str  # "Intel" or "AMD"
    llvm_mca_cpu: str | None  # llvm-mca -mcpu= value


# Authoritative registry of all supported architectures.
_ARCH_REGISTRY: list[ArchInfo] = [
    # Intel
    ArchInfo("SNB", ("sandybridge",), "Intel", "sandybridge"),
    ArchInfo("IVB", ("ivybridge",), "Intel", "ivybridge"),
    ArchInfo("HSW", ("haswell",), "Intel", "haswell"),
    ArchInfo("BDW", ("broadwell",), "Intel", "broadwell"),
    ArchInfo("SKL", ("skylake",), "Intel", "skylake"),
    ArchInfo("KBL", ("kabylake",), "Intel", "skylake"),
    ArchInfo("CFL", ("coffeelake",), "Intel", "skylake"),
    ArchInfo("CNL", ("cannonlake",), "Intel", "cannonlake"),
    ArchInfo("ICL", ("icelake",), "Intel", "icelake-server"),
    ArchInfo("TGL", ("tigerlake",), "Intel", "icelake-server"),
    ArchInfo("RKL", ("rocketlake",), "Intel", "icelake-server"),
    ArchInfo("SKX", ("skylake-x",), "Intel", "skylake-avx512"),
    ArchInfo("CLX", ("cascadelake",), "Intel", "cascadelake"),
    ArchInfo("ADL-P", ("alderlake-p",), "Intel", "alderlake"),
    ArchInfo("ADL-E", ("alderlake-e",), "Intel", "alderlake"),
    # AMD
    ArchInfo("ZEN+", ("zen+",), "AMD", "znver1"),
    ArchInfo("ZEN1", ("zen1",), "AMD", "znver1"),
    ArchInfo("ZEN2", ("zen2",), "AMD", "znver2"),
    ArchInfo("ZEN3", ("zen3",), "AMD", "znver3"),
    ArchInfo("ZEN4", ("zen4",), "AMD", "znver4"),
    ArchInfo("ZEN5", ("zen5",), "AMD", "znver5"),
]

# Derived lookup: friendly name (lowercase) -> canonical XML name.
_ARCH_ALIASES: dict[str, str] = {
    alias: info.canonical for info in _ARCH_REGISTRY for alias in info.aliases
}

# Derived lookup: canonical XML name -> ArchInfo.
_ARCH_BY_CANONICAL: dict[str, ArchInfo] = {
    info.canonical: info for info in _ARCH_REGISTRY
}


def _find_uops_database_path() -> str | None:
    """Find compact JSON uops database path."""
    try:
        from util.uops_parser import find_uops_database_path
    except ImportError:
        from .util.uops_parser import find_uops_database_path

    return find_uops_database_path(
        base_dir=os.path.join(os.path.dirname(__file__), "..")
    )


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


def resolve_llvm_mca_cpu(target_cpu: str) -> str | None:
    """Resolve a --target-cpu value to an llvm-mca -mcpu= value.

    Returns None if target_cpu is "generic".
    """
    canonical = resolve_arch_name(target_cpu)
    if canonical is None:
        return None
    info = _ARCH_BY_CANONICAL.get(canonical)
    if info is None:
        return None
    return info.llvm_mca_cpu


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

        database_path = _find_uops_database_path()
        if database_path is None:
            print("Warning: no uops database found, using generic costs")
            return

        self._overlay_arch_costs(database_path, arch_name)

    def _overlay_arch_costs(self, database_path: str, arch_name: str):
        """Overlay architecture-specific costs from compact JSON uops data."""
        try:
            from intrinsic_registry import get_intrinsic_registry, xml_string_key
            from util.uops_parser import parse_uops_database
        except ImportError:
            from .intrinsic_registry import get_intrinsic_registry, xml_string_key
            from .util.uops_parser import parse_uops_database

        xml_costs = parse_uops_database(database_path, arch_name)
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
                "_mm256_alignr_epi8": InstructionCost(1.0, 1.0, ["p5"]),
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
