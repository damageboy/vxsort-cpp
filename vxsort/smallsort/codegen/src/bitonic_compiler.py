#!/usr/bin/env python3
from __future__ import annotations

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
    # Start with 2 vectors, i32, AVX2 as per plan
    generate_bitonic_sorter(2, primitive_type.i32, vector_machine.AVX2)
