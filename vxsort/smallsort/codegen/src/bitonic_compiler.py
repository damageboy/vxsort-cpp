#!/usr/bin/env python3
from __future__ import annotations
import argparse
import tempfile

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


def _collect_leaf_paths(node, current_path=None):
    """Yield all root-to-leaf paths as (leaf_cost, [node, ...]) tuples."""
    if current_path is None:
        current_path = []
    current_path = current_path + [node]

    if not node.children:
        yield (node.cost, current_path)
    else:
        for child in node.children:
            yield from _collect_leaf_paths(child, current_path)


def _count_control_vectors_in_path(path):
    """Count instructions that use YMM/ZMM control vectors instead of immediates.

    Control vector instructions (less preferred):
    - Use 'op_idx' key (e.g., _mm256_permutexvar_epi32)
    - Use 'mask' key for variable masks (e.g., _mm256_blendv_ps)
    - Use 'b' as control in permutevar instructions

    Immediate-based instructions (preferred):
    - Use 'imm8' key (e.g., _mm256_permute_ps, _mm256_shuffle_ps)
    """
    count = 0
    for node in path:
        gadget = node.best_gadget()
        for inst in gadget.top_instructions + gadget.bottom_instructions:
            # Check if instruction uses a control vector
            if "op_idx" in inst.args:
                # permutexvar family - uses control vector
                count += 1
            elif "mask" in inst.args and inst.intrinsic_name.endswith("v_ps"):
                # blendv_ps - uses variable mask (256-bit control)
                count += 1
            elif "b" in inst.args and "permutevar" in inst.intrinsic_name:
                # permutevar_ps - 'b' is control vector
                count += 1
    return count


def _prune_to_top_k_paths(solutions, top_k):
    """Keep only the top K cheapest root-to-leaf paths, preferring immediate-based instructions.

    Sorts by (cost, control_vector_count) where control_vector_count is the number
    of instructions requiring YMM/ZMM control inputs. At equal cost, paths with
    fewer control vector instructions are preferred (immediates are more efficient).

    Modifies the tree in place by removing children not on any selected path.
    Returns the filtered list of roots.
    """
    all_paths = []
    for root in solutions:
        all_paths.extend(_collect_leaf_paths(root))

    # Sort by (cost, control_vector_count) - prefer fewer control vectors at same cost
    all_paths.sort(key=lambda x: (x[0], _count_control_vectors_in_path(x[1])))
    selected_paths = all_paths[:top_k]

    # Build set of kept edges (parent_id, child_id) and kept root ids
    kept_edges = set()
    kept_roots = set()
    for _, path in selected_paths:
        kept_roots.add(id(path[0]))
        for i in range(len(path) - 1):
            kept_edges.add((id(path[i]), id(path[i + 1])))

    def prune_node(node):
        node.children = [
            child for child in node.children if (id(node), id(child)) in kept_edges
        ]
        for child in node.children:
            prune_node(child)

    solutions = [root for root in solutions if id(root) in kept_roots]
    for root in solutions:
        prune_node(root)

    return solutions


def generate_bitonic_sorter(
    num_vecs: int,
    type: primitive_type,
    vm: vector_machine,
    depth_limit: int | None = None,
    top_k: int | None = None,
    output_format: str = "json",
    gadget_depth: int = 3,
    smt2_dump_dir: str | None = None,
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
        smt2_dump_dir: Directory to dump SMT2 files if requested

    Returns:
        List of SolutionNode trees representing different optimized solutions
    """
    total_elements = int(num_vecs * (width_dict[vm] / int(type.value[0])))

    print(
        f"Building {vm.name} sorter for {total_elements} elements ({num_vecs} vectors)"
    )

    # Create super-vectorizer
    super_opt = BitonicSuperVectorizer(num_vecs, type, vm, smt2_dump_dir=smt2_dump_dir)

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

    # Filter to top K cheapest root-to-leaf paths if requested
    if top_k is not None:
        total_paths = sum(1 for root in solutions for _ in _collect_leaf_paths(root))
        if total_paths > top_k:
            print(
                f"Filtering to top {top_k} cheapest paths (out of {total_paths} total)..."
            )
            solutions = _prune_to_top_k_paths(solutions, top_k)
            print(f"Kept {len(solutions)} roots after pruning")

    # Export solutions in the requested format
    if output_format == "asm":
        output_path = f"bitonic_solutions_{num_vecs}x{vm.name}_{type.name}.asm"
        export_solutions_as_assembly(solutions, num_vecs, type, vm, output_path)
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
    parser.add_argument(
        "--dump-smt2",
        action="store_true",
        help="Dump SMT2 files from Z3 into compressed tar files in /tmp",
    )

    args = parser.parse_args()

    # Convert string arguments to Enum members
    vm = vector_machine[args.vector_machine]
    dtype = primitive_type[args.datatype]

    smt2_dump_dir = None
    if args.dump_smt2:
        smt2_dump_dir = tempfile.mkdtemp(prefix="vxsort_smt2_", dir="/tmp")
        print(f"SMT2 dump directory: {smt2_dump_dir}")

    generate_bitonic_sorter(
        args.num_vecs,
        dtype,
        vm,
        depth_limit=args.depth_limit,
        top_k=args.top_k,
        output_format=args.output_format,
        gadget_depth=args.gadget_depth + 1,  # +1 because range is exclusive
        smt2_dump_dir=smt2_dump_dir,
    )
