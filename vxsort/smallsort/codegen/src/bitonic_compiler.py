#!/usr/bin/env python3
from __future__ import annotations
import argparse

# Handle both relative and absolute imports
try:
    from .cost_model import CostModel
    from .bitonic_super_optimizer import BitonicSuperVectorizer
    from .utils import vector_machine, primitive_type, width_dict

except ImportError:
    from cost_model import CostModel
    from bitonic_super_optimizer import BitonicSuperVectorizer
    from utils import vector_machine, primitive_type, width_dict


def generate_bitonic_sorter(num_vecs: int, type: primitive_type, vm: vector_machine):
    """
    Generate bitonic sorter with super-optimized permutation sequences.

    Args:
        num_vecs: Number of SIMD vectors to sort
        type: Primitive type (i32, f32, i64, f64)
        vm: Vector machine (AVX2, AVX512)

    Returns:
        List of SolutionNode trees representing different optimized solutions
    """
    total_elements = int(num_vecs * (width_dict[vm] / int(type.value[0])))

    print(
        f"Building {vm.name} sorter for {total_elements} elements ({num_vecs} vectors)"
    )

    # Create super-vectorizer
    super_opt = BitonicSuperVectorizer(num_vecs, type, vm)

    # Synthesize all stages to build solution tree
    print("Synthesizing permutation gadgets...")
    solutions = super_opt.synthesize_all_stages()

    print(f"Found {len(solutions)} root solutions")

    # Compute costs
    print("Computing costs...")
    cost_model = CostModel("generic")
    super_opt.compute_costs(solutions, cost_model)

    # Export solutions to JSON
    output_path = f"bitonic_solutions_{num_vecs}x{vm.name}_{type.name}.json"
    super_opt.export_solutions(solutions, output_path)

    return solutions


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bitonic sorter with optimized permutations.")
    parser.add_argument(
        "--num-vecs", type=int, default=2, help="Number of SIMD vectors to sort (default: 2)"
    )
    parser.add_argument(
        "--vector-machine",
        type=str,
        required=True,
        choices=list(vector_machine.__members__.keys()),
        help="Vector architecture (AVX2, AVX512)",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        required=True,
        choices=list(primitive_type.__members__.keys()),
        help="Primitive data type (i16, u16, i32, u32, i64, u64, f32, f64)",
    )

    args = parser.parse_args()

    # Convert string arguments to Enum members
    vm = vector_machine[args.vector_machine]
    dtype = primitive_type[args.datatype]

    generate_bitonic_sorter(args.num_vecs, dtype, vm)
