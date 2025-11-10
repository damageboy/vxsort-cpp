#!/usr/bin/env python3
"""Cost model for AVX instructions based on CPU microarchitecture."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class InstructionCost:
    """Cost characteristics for a single instruction."""

    latency: float  # Latency in cycles
    throughput: float  # Reciprocal throughput (CPI)
    ports: list[str]  # Execution ports (e.g., ["p0", "p1", "p5"])

    def __repr__(self):
        return f"Cost(lat={self.latency}, tput={self.throughput}, ports={self.ports})"


class CostModel:
    """Cost model for evaluating instruction sequences."""

    def __init__(self, target_cpu: str = "generic"):
        self.target_cpu = target_cpu
        self.instruction_costs: Dict[str, InstructionCost] = {}
        self._initialize_costs()

    def _initialize_costs(self):
        """Initialize instruction costs based on target CPU."""
        if self.target_cpu == "generic":
            self._init_generic_costs()
        elif self.target_cpu == "zen5":
            self._init_zen5_costs()
        elif self.target_cpu == "icelake":
            self._init_icelake_costs()
        else:
            # Default to generic
            self._init_generic_costs()

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
                "_mm512_mask_unpacklo_epi32": InstructionCost(1.0, 1.0, ["p5"]),
                "_mm512_mask_unpackhi_epi32": InstructionCost(1.0, 1.0, ["p5"]),
            }
        )

    def _init_zen5_costs(self):
        """AMD Zen 5 cost model - loads from uops.info data if available."""
        # Start with generic
        self._init_generic_costs()

        # Try to load Zen 5 specific costs from uops.info data
        zen5_costs = load_costs_from_uops_info("zen5")
        if zen5_costs:
            self.instruction_costs.update(zen5_costs)

    def _init_icelake_costs(self):
        """Intel Ice Lake cost model - loads from uops.info data if available."""
        # Start with generic
        self._init_generic_costs()

        # Try to load Ice Lake specific costs from uops.info data
        icelake_costs = load_costs_from_uops_info("icelake")
        if icelake_costs:
            self.instruction_costs.update(icelake_costs)

    def get_instruction_cost(self, intrinsic_name: str) -> InstructionCost:
        """Get cost for a specific intrinsic."""
        if intrinsic_name in self.instruction_costs:
            return self.instruction_costs[intrinsic_name]
        else:
            # Unknown instruction, use default cost
            return InstructionCost(latency=1.0, throughput=1.0, ports=["unknown"])

    def calculate_gadget_cost(self, gadget) -> float:
        """
        Calculate cost for a permutation gadget.
        For now, use simple latency sum. More sophisticated models
        could account for instruction-level parallelism.
        """
        total_cost = 0.0

        # Sum costs for top instructions
        for inst in gadget.top_instructions:
            cost = self.get_instruction_cost(inst.intrinsic_name)
            total_cost += cost.latency

        # Sum costs for bottom instructions
        for inst in gadget.bottom_instructions:
            cost = self.get_instruction_cost(inst.intrinsic_name)
            total_cost += cost.latency

        return total_cost

    def calculate_path_cost(self, solution_path: list) -> float:
        """Calculate total cost for a complete solution path."""
        return sum(self.calculate_gadget_cost(node.gadget) for node in solution_path)


def load_costs_from_uops_info(cpu_model: str) -> Dict[str, InstructionCost]:
    """
    Load instruction costs from uops.info data.

    Reads pre-computed instruction costs from JSON files.
    Data can be generated by scraping https://uops.info/ or manually entered.

    Expected JSON format:
    {
      "cpu_model": "zen5",
      "instructions": {
        "VPERMD": {
          "latency": 3.0,
          "throughput": 1.0,
          "ports": ["p5"]
        },
        ...
      }
    }
    """
    import json
    import os

    # Look for cost data file
    script_dir = os.path.dirname(__file__)
    cost_file = os.path.join(script_dir, f"uops_data_{cpu_model}.json")

    if not os.path.exists(cost_file):
        print(f"Note: Cost data file {cost_file} not found, using generic costs")
        return {}

    try:
        with open(cost_file, "r") as f:
            data = json.load(f)

        costs = {}
        for instruction_name, cost_data in data.get("instructions", {}).items():
            costs[instruction_name] = InstructionCost(
                latency=cost_data.get("latency", 1.0),
                throughput=cost_data.get("throughput", 1.0),
                ports=cost_data.get("ports", ["unknown"]),
            )

        print(f"Loaded {len(costs)} instruction costs from {cost_file}")
        return costs

    except Exception as e:
        print(f"Warning: Failed to load cost data from {cost_file}: {e}")
        return {}
