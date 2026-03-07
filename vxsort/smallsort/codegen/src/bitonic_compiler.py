#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import tempfile

from multiprocessing import Pool

from json_exporter import export_solutions_to_json

# Handle both relative and absolute imports
try:
    from .cost_model import CostModel, get_supported_cpus
    from .bitonic_super_optimizer import BitonicSuperVectorizer
    from .utils import vector_machine, primitive_type, width_dict
    from .asm_exporter import export_solutions_to_asm
    from .path_selector import PathSelector
    from .bitonic_verifier import (
        load_solutions_from_json,
        extract_verify_steps,
        _verify_path_worker,
    )
    from .success_progress import SuccessProgress
    from .checkpoint import (
        CheckpointConfig,
        load_checkpoint,
        validate_checkpoint_config,
    )

except ImportError:
    from cost_model import CostModel, get_supported_cpus
    from bitonic_super_optimizer import BitonicSuperVectorizer
    from utils import vector_machine, primitive_type, width_dict
    from asm_exporter import export_solutions_to_asm
    from path_selector import PathSelector
    from bitonic_verifier import (  # type: ignore
        load_solutions_from_json,
        extract_verify_steps,
        _verify_path_worker,
    )
    from success_progress import SuccessProgress
    from checkpoint import (  # type: ignore
        CheckpointConfig,
        load_checkpoint,
        validate_checkpoint_config,
    )


def _run_verification(
    paths_to_verify: list,
    vm: vector_machine,
    prim_type: primitive_type,
    natural_order: bool,
    max_workers: int | None = None,
    max_tasks_per_child: int | None = 1000,
):
    """Run Z3 end-to-end verification on a list of paths using multiprocessing.

    Raises SystemExit(1) if any path fails.
    """
    total = len(paths_to_verify)

    # Build lightweight picklable jobs (no SolutionNode tree references)
    jobs = [
        (i, extract_verify_steps(path), vm, prim_type, natural_order)
        for i, path in enumerate(paths_to_verify)
    ]

    progress = SuccessProgress.create(
        description_column="[orange1]{task.description}",
        width=60,
        success_style="green",
        attempt_style="red",
        success_label="Verified",
    )
    progress.enable_memory_monitor()
    progress.start()
    task_id = progress.add_task("Verifying paths", total=total, successes=0)

    failures = []
    try:
        with Pool(processes=max_workers, maxtasksperchild=max_tasks_per_child) as pool:
            for path_index, result in pool.imap_unordered(_verify_path_worker, jobs):
                progress.update(task_id, advance=1, success=1 if result.verified else 0)
                if not result.verified:
                    failures.append((path_index, result))
    finally:
        progress.stop()

    if failures:
        failures.sort(key=lambda x: x[0])
        print(f"\nVERIFICATION FAILED: {len(failures)}/{total} paths failed")
        for path_index, result in failures:
            print(f"  Path {path_index + 1}: counterexample={result.counterexample}")
        raise SystemExit(1)

    print(f"\nAll {total} paths verified correct.")


def verify_only_from_json(
    json_path: str,
    top_k: int | None = None,
    target_cpu: str = "generic",
    estimate: bool = False,
    nasm_path: str | None = None,
    max_workers: int | None = None,
    max_tasks_per_child: int | None = 1000,
):
    """Load solutions from a JSON file and verify them without re-running synthesis.

    The JSON file must contain vector_machine and primitive_type metadata
    (written by json_exporter since Feb 2026).

    Args:
        json_path: Path to a JSON solutions file.
        top_k: Number of best paths to verify. If None, verify all paths.
        target_cpu: Target CPU for cost model during path selection.
        estimate: If True, run OSACA performance estimation on selected paths.
        nasm_path: Path to nasm binary for assembly verification (used with estimate).
    """
    bundle = load_solutions_from_json(json_path)

    if bundle.vm_name is None or bundle.prim_type_name is None:
        raise SystemExit(
            f"Error: {json_path} has no vector_machine/primitive_type metadata.\n"
            "Re-export the solutions with a current version of the tool."
        )

    vm = vector_machine[bundle.vm_name]
    prim_type = primitive_type[bundle.prim_type_name]

    print(
        f"Loaded {len(bundle.roots)} roots from {json_path} "
        f"({vm.name} {prim_type.name}, "
        f"natural_order={bundle.natural_order})"
    )

    # Select paths
    cost_model = CostModel(target_cpu)
    path_selector = PathSelector(cost_model)
    paths = path_selector.select_top_k_paths(bundle.roots, top_k or 10_000)

    _run_verification(
        paths,
        vm,
        prim_type,
        bundle.natural_order,
        max_workers=max_workers,
        max_tasks_per_child=max_tasks_per_child,
    )

    if estimate:
        if bundle.num_vecs is None:
            raise SystemExit(
                f"Error: {json_path} has no num_vecs metadata. "
                "Re-export the solutions with a current version of the tool."
            )
        from osaca_estimator import estimate_solutions, print_estimation_table

        results = estimate_solutions(
            paths,
            vm,
            prim_type,
            bundle.num_vecs,
            bundle.natural_order,
            target_cpu,
            nasm_path,
        )
        print_estimation_table(results, paths)


def estimate_only_from_json(
    json_path: str,
    top_k: int | None = None,
    target_cpu: str = "generic",
    nasm_path: str | None = None,
):
    """Load solutions from a JSON file and run OSACA estimation without verification.

    Args:
        json_path: Path to a JSON solutions file.
        top_k: Number of best paths to estimate. If None, estimate all paths.
        target_cpu: Target CPU for cost model during path selection.
        nasm_path: Path to nasm binary for assembly verification.
    """
    bundle = load_solutions_from_json(json_path)

    if bundle.vm_name is None or bundle.prim_type_name is None:
        raise SystemExit(
            f"Error: {json_path} has no vector_machine/primitive_type metadata.\n"
            "Re-export the solutions with a current version of the tool."
        )

    if bundle.num_vecs is None:
        raise SystemExit(
            f"Error: {json_path} has no num_vecs metadata. "
            "Re-export the solutions with a current version of the tool."
        )

    vm = vector_machine[bundle.vm_name]
    prim_type = primitive_type[bundle.prim_type_name]

    print(
        f"Loaded {len(bundle.roots)} roots from {json_path} "
        f"({vm.name} {prim_type.name}, "
        f"natural_order={bundle.natural_order})"
    )

    # Select paths
    cost_model = CostModel(target_cpu)
    path_selector = PathSelector(cost_model)
    paths = path_selector.select_top_k_paths(bundle.roots, top_k or 10_000)

    from osaca_estimator import estimate_solutions, print_estimation_table

    results = estimate_solutions(
        paths,
        vm,
        prim_type,
        bundle.num_vecs,
        bundle.natural_order,
        target_cpu,
        nasm_path,
    )
    print_estimation_table(results, paths)


def _count_dag_paths(roots):
    """Count total root-to-leaf paths through the DAG using memoization.

    O(unique nodes) time and space, regardless of the (potentially
    exponential) number of paths.
    """
    cache: dict[int, int] = {}

    def _count(node):
        nid = id(node)
        if nid in cache:
            return cache[nid]
        if not node.children:
            cache[nid] = 1
            return 1
        total = sum(_count(child) for child in node.children)
        cache[nid] = total
        return total

    return sum(_count(root) for root in roots)


def generate_bitonic_sorter(
    num_vecs: int,
    prim_type: primitive_type,
    vm: vector_machine,
    depth_limit: int | None = None,
    top_k: int | None = None,
    output_formats: list[str] = None,
    gadget_depth: int = 1,
    smt2_dump_dir: str | None = None,
    natural_order: bool = False,
    max_gadget_solutions: int = 3,
    target_cpu: str = "generic",
    verify: bool = False,
    checkpoint_dir: str | None = None,
    resume_file: str | None = None,
    estimate: bool = False,
    nasm_path: str | None = None,
    max_workers: int | None = None,
    max_tasks_per_child: int | None = 1000,
):
    """
    Generate bitonic sorter with super-optimized permutation sequences.

    Args:
        num_vecs: Number of SIMD vectors to sort
        prim_type: Primitive type (i32, f32, i64, f64)
        vm: Vector machine (AVX2, AVX512)
        depth_limit: Maximum stage depth to explore (inclusive). If None, all stages are explored.
        top_k: Number of best solutions to keep. If None, all solutions are kept.
        output_formats: List of output formats (e.g., ["json", "asm"]). Default is ["json"].
        gadget_depth: Maximum instruction depth per gadget (1-3, default 1)
        smt2_dump_dir: Directory to dump SMT2 files if requested
        natural_order: If True, add final stage restoring natural element order for memory writeback
        max_gadget_solutions: Number of smallest unique output states to enumerate per
            gadget template. Higher values increase search diversity at the cost of speed. Default: 3.
        target_cpu: Target CPU for cost model (e.g., "generic", "TGL", "ZEN4", "tigerlake").
            Default: "generic".
        verify: If True, verify selected paths with Z3 end-to-end proof of sorting
            correctness. Default: False.
        checkpoint_dir: Directory to save checkpoints after each stage completes.
            If None, no checkpoints are saved.
        resume_file: Path to a checkpoint file to resume from. If provided, completed
            stages are skipped and synthesis continues from where it left off.
        estimate: If True, run OSACA performance estimation on selected paths.
        nasm_path: Path to nasm binary for assembly verification (used with --estimate).
        max_tasks_per_child: Maximum tasks per worker process before recycling.
            Limits memory growth in long runs. None disables recycling.

    Returns:
        List of SolutionNode trees representing different optimized solutions
    """
    if output_formats is None:
        output_formats = ["json"]
    total_elements = int(num_vecs * (width_dict[vm] / int(prim_type.value[0])))

    print(
        f"Building {vm.name} sorter for {total_elements} elements ({num_vecs} vectors)"
    )

    # Handle resume from checkpoint
    resume_data = None
    if resume_file is not None:
        print(f"Loading checkpoint from {resume_file}...")
        saved_config, per_stage_data, stages_completed, last_completed_stage = (
            load_checkpoint(resume_file)
        )

        # Build current config for validation
        current_config = CheckpointConfig(
            num_vecs=num_vecs,
            vm=vm.name,
            prim_type=prim_type.name,
            gadget_depth=gadget_depth,
            natural_order=natural_order,
            max_unique_outputs=max_gadget_solutions,
            depth_limit=depth_limit,
        )

        ok, msg = validate_checkpoint_config(saved_config, current_config)
        if not ok:
            raise SystemExit(f"Checkpoint incompatible: {msg}")

        print(
            f"Checkpoint valid: {len(stages_completed)} stages completed "
            f"(last: {last_completed_stage})"
        )
        resume_data = {
            "per_stage_data": per_stage_data,
            "last_completed_stage": last_completed_stage,
        }

    # Create checkpoint directory if needed
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Create super-vectorizer
    super_opt = BitonicSuperVectorizer(
        num_vecs, prim_type, vm, smt2_dump_dir=smt2_dump_dir
    )

    # Synthesize all stages to build solution tree
    print("Synthesizing permutation gadgets...")
    solutions, all_stages_complete = super_opt.synthesize_all_stages(
        depth_limit=depth_limit,
        gadget_depth=gadget_depth,
        natural_order=natural_order,
        max_unique_outputs=max_gadget_solutions,
        checkpoint_dir=checkpoint_dir,
        resume_data=resume_data,
        max_workers=max_workers,
        max_tasks_per_child=max_tasks_per_child,
    )

    print(f"Found {len(solutions)} root solutions")

    if not all_stages_complete:
        print(
            "\nWARNING: The super-optimizer did not find solutions for all stages.\n"
            "The output will be incomplete. Consider increasing --gadget-depth.\n"
        )

    # Filter to top K cheapest root-to-leaf paths if requested
    selected_paths = None
    if top_k is not None:
        total_paths = _count_dag_paths(solutions)
        if total_paths > top_k:
            print(
                f"Filtering to top {top_k} cheapest paths (out of {total_paths} total)..."
            )
            # Use PathSelector for unified scoring
            cost_model = CostModel(target_cpu)
            path_selector = PathSelector(cost_model)
            solutions, selected_paths = path_selector.prune_to_top_k_paths(
                solutions, top_k
            )
            print(f"Kept {len(solutions)} roots after pruning")

    # Export solutions in the requested format(s)
    order_suffix = "_natural" if natural_order else ""
    for output_format in output_formats:
        if output_format == "asm":
            output_path = f"bitonic_solutions_{num_vecs}x{vm.name}_{prim_type.name}{order_suffix}.asm"
            export_solutions_to_asm(
                solutions,
                num_vecs,
                prim_type,
                vm,
                output_path,
                selected_paths=selected_paths,
                natural_order=natural_order,
            )
        elif output_format == "json":
            output_path = f"bitonic_solutions_{num_vecs}x{vm.name}_{prim_type.name}{order_suffix}.json"
            export_solutions_to_json(
                solutions,
                output_path,
                natural_order=natural_order,
                vm_name=vm.name,
                prim_type_name=prim_type.name,
                num_vecs=num_vecs,
            )
        else:
            print(f"Warning: Unknown output format '{output_format}', skipping")

    # End-to-end verification
    if verify:
        # Build paths if we don't already have them
        paths_to_verify = selected_paths
        if paths_to_verify is None:
            cost_model = CostModel(target_cpu)
            path_selector = PathSelector(cost_model)
            paths_to_verify = path_selector.select_top_k_paths(
                solutions, top_k or 10_000
            )

        _run_verification(
            paths_to_verify,
            vm,
            prim_type,
            natural_order,
            max_workers=max_workers,
            max_tasks_per_child=max_tasks_per_child,
        )

    # OSACA performance estimation
    if estimate:
        from osaca_estimator import estimate_solutions, print_estimation_table

        paths_to_estimate = selected_paths
        if paths_to_estimate is None:
            cost_model = CostModel(target_cpu)
            path_selector = PathSelector(cost_model)
            paths_to_estimate = path_selector.select_top_k_paths(
                solutions, top_k or 10_000
            )

        results = estimate_solutions(
            paths_to_estimate,
            vm,
            prim_type,
            num_vecs,
            natural_order,
            target_cpu,
            nasm_path,
        )
        print_estimation_table(results, paths_to_estimate)

    return solutions


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
        default=None,
        choices=list(vector_machine.__members__.keys()),
        help=f"Vector architecture ({', '.join(vector_machine.__members__.keys())})",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default=None,
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
        action="append",
        choices=["json", "asm"],
        help="Output format: 'json' for structured JSON output, 'asm' for readable assembly code. Can be specified multiple times (default: json)",
    )
    parser.add_argument(
        "--gadget-depth",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Maximum instruction depth per gadget (1-3, default: 1)",
    )
    parser.add_argument(
        "--dump-smt2",
        action="store_true",
        help="Dump SMT2 files from Z3 into compressed tar files in /tmp",
    )
    parser.add_argument(
        "--natural-order",
        action="store_true",
        default=False,
        help="Add final permutation stage to restore natural element order for memory writeback",
    )
    parser.add_argument(
        "--max-gadget-solutions",
        type=int,
        default=3,
        help="Number of unique output states to enumerate per gadget template. "
        "Higher values increase search diversity at the cost of speed (default: 3)",
    )
    parser.add_argument(
        "--target-cpu",
        type=str,
        default="generic",
        help="Target CPU for cost model (e.g. generic, TGL, SKX, ZEN4, "
        "tigerlake, skylake-x, zen4). Use --list-cpus to see all options (default: generic)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Verify selected paths with Z3 end-to-end proof of sorting correctness",
    )
    parser.add_argument(
        "--verify-only",
        type=str,
        default=None,
        metavar="JSON_FILE",
        help="Skip synthesis and verify solutions from a JSON file. "
        "The file must contain vector_machine/primitive_type metadata.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save checkpoints after each stage completes. "
        "Enables resuming long-running compilations.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_FILE",
        help="Resume synthesis from a checkpoint file (.json.zst). "
        "Completed stages are skipped.",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        default=False,
        help="Run OSACA performance estimation on selected paths",
    )
    parser.add_argument(
        "--nasm-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to nasm binary for assembly verification (used with --estimate)",
    )
    parser.add_argument(
        "--estimate-only",
        type=str,
        default=None,
        metavar="JSON_FILE",
        help="Skip synthesis and verification; load solutions from a JSON file "
        "and run OSACA performance estimation only.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of worker processes for parallel synthesis and verification. "
        "Defaults to os.cpu_count(). Use this to limit memory usage on large machines.",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=1000,
        metavar="N",
        help="Maximum tasks per worker process before recycling (default: 1000). "
        "Limits memory growth in long runs. Set to 0 to disable recycling.",
    )
    parser.add_argument(
        "--list-cpus",
        action="store_true",
        default=False,
        help="List all supported CPU architectures with uops.info and OSACA availability, then exit",
    )

    args = parser.parse_args()

    # Convert 0 → None (Pool interprets None as "no limit")
    max_tasks_per_child = args.max_tasks_per_child or None

    # --list-cpus mode: print supported architectures and exit
    if args.list_cpus:
        from uops_parser import list_available_architectures

        xml_path = os.path.join(os.path.dirname(__file__), "..", "instructions.xml.zst")
        has_xml = os.path.exists(xml_path)
        xml_archs = list_available_architectures(xml_path) if has_xml else set()

        rows = []
        for info in get_supported_cpus():
            aliases = ", ".join(info.aliases)
            uops_col = "yes" if info.canonical in xml_archs else "-"
            osaca_col = info.osaca_code if info.osaca_code else "-"
            rows.append([info.canonical, aliases, info.vendor, uops_col, osaca_col])

        from tabulate import tabulate

        headers = ["Arch", "Alias", "Vendor", "uops.info", "OSACA model"]
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        raise SystemExit(0)

    # --verify-only mode: load JSON and verify without synthesis
    if args.verify_only is not None:
        verify_only_from_json(
            args.verify_only,
            top_k=args.top_k,
            target_cpu=args.target_cpu,
            estimate=args.estimate,
            nasm_path=args.nasm_path,
            max_workers=args.max_workers,
            max_tasks_per_child=max_tasks_per_child,
        )
        raise SystemExit(0)

    # --estimate-only mode: load JSON and run OSACA estimation without synthesis or verification
    if args.estimate_only is not None:
        estimate_only_from_json(
            args.estimate_only,
            top_k=args.top_k,
            target_cpu=args.target_cpu,
            nasm_path=args.nasm_path,
        )
        raise SystemExit(0)

    # Normal synthesis mode requires --vector-machine and --datatype
    if not args.vector_machine or not args.datatype:
        parser.error("--vector-machine and --datatype are required")

    # Convert string arguments to Enum members
    vm = vector_machine[args.vector_machine]
    prim_type = primitive_type[args.datatype]

    smt2_dump_dir = None
    if args.dump_smt2:
        smt2_dump_dir = tempfile.mkdtemp(prefix="vxsort_smt2_", dir="/tmp")
        print(f"SMT2 dump directory: {smt2_dump_dir}")

    generate_bitonic_sorter(
        args.num_vecs,
        prim_type,
        vm,
        depth_limit=args.depth_limit,
        top_k=args.top_k,
        output_formats=args.output_format,  # Will be None if not specified, handled by function default
        gadget_depth=args.gadget_depth,
        smt2_dump_dir=smt2_dump_dir,
        natural_order=args.natural_order,
        max_gadget_solutions=args.max_gadget_solutions,
        target_cpu=args.target_cpu,
        verify=args.verify,
        checkpoint_dir=args.checkpoint_dir,
        resume_file=args.resume,
        estimate=args.estimate,
        nasm_path=args.nasm_path,
        max_workers=args.max_workers,
        max_tasks_per_child=max_tasks_per_child,
    )
