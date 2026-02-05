#!/usr/bin/env python3
from __future__ import annotations
import argparse

# Handle both relative and absolute imports
try:
    from .cost_model import CostModel
    from .bitonic_super_optimizer import BitonicSuperVectorizer
    from .utils import vector_machine, primitive_type, width_dict
    from .asm_exporter import export_solutions_as_assembly

except ImportError:
    from cost_model import CostModel
    from bitonic_super_optimizer import BitonicSuperVectorizer
    from utils import vector_machine, primitive_type, width_dict
    from asm_exporter import export_solutions_as_assembly


def _get_min_leaf_cost(node) -> float:
    """Get the minimum cost among all leaf nodes in the tree."""
    if not node.children:
        # This is a leaf node
        return node.cost
    # Return minimum cost among all children
    return min(_get_min_leaf_cost(child) for child in node.children)


def generate_bitonic_sorter(
    num_vecs: int,
    type: primitive_type,
    vm: vector_machine,
    depth_limit: int | None = None,
    top_k: int | None = None,
    output_format: str = "json",
    gadget_depth: int = 3,
):
    """
    Generate bitonic sorter with super-optimized permutation sequences.

    Args:
        num_vecs: Number of SIMD vectors to sort
        type: Primitive type (i32, f32, i64, f64)
        vm: Vector machine (AVX2, AVX512)
        depth_limit: Maximum stage depth to explore (inclusive). If None, all stages are explored.
        top_k: Number of best solutions to keep. If None, all solutions are kept.
        output_format: Output format ("json" or "asm")
        gadget_depth: Maximum instruction depth per gadget (1-3, default 3)

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
    solutions = super_opt.synthesize_all_stages(
        depth_limit=depth_limit, gadget_depth=gadget_depth
    )

    print(f"Found {len(solutions)} root solutions")

    # Compute costs
    print("Computing costs...")
    cost_model = CostModel("generic")
    super_opt.compute_costs(solutions, cost_model)

    # Filter to top K solutions if requested
    if top_k is not None and len(solutions) > top_k:
        print(f"Filtering to top {top_k} solutions (out of {len(solutions)})...")
        # Sort by minimum leaf cost (best complete path)
        solutions = sorted(solutions, key=_get_min_leaf_cost)[:top_k]
        print(f"Kept {len(solutions)} best solutions")

    # Export solutions in the requested format
    if output_format == "asm":
        output_path = f"bitonic_solutions_{num_vecs}x{vm.name}_{type.name}.asm"
        export_solutions_as_assembly(solutions, num_vecs, vm, output_path)
    else:  # json
        output_path = f"bitonic_solutions_{num_vecs}x{vm.name}_{type.name}.json"
        super_opt.export_solutions_to_json(solutions, output_path)

    return solutions


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bitonic sorter with optimized permutations."
    )
    parser.add_argument(
        "--num-vecs",
        type=int,
        default=2,
        help="Number of SIMD vectors to sort (default: 2)",
    )
    parser.add_argument(
        "--vector-machine",
        type=str,
        required=True,
        choices=list(vector_machine.__members__.keys()),
        help=f"Vector architecture ({', '.join(vector_machine.__members__.keys())}",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        required=True,
        choices=list(primitive_type.__members__.keys()),
        help=f"Primitive data type ({', '.join(primitive_type.__members__.keys())})",
    )
    parser.add_argument(
        "--depth-limit",
        type=int,
        default=None,
        help="Maximum stage depth to explore (inclusive). If not specified, all stages are explored.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of best solutions to keep in output. If not specified, all solutions are kept.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "asm"],
        help="Output format: 'json' for structured JSON output, 'asm' for readable assembly code (default: json)",
    )
    parser.add_argument(
        "--gadget-depth",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Maximum instruction depth per gadget (1-3, default: 3)",
    )

    args = parser.parse_args()

    # Convert string arguments to Enum members
    vm = vector_machine[args.vector_machine]
    dtype = primitive_type[args.datatype]

    generate_bitonic_sorter(
        args.num_vecs,
        dtype,
        vm,
        depth_limit=args.depth_limit,
        top_k=args.top_k,
        output_format=args.output_format,
        gadget_depth=args.gadget_depth + 1,  # +1 because range is exclusive
    )
