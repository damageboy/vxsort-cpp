#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import tempfile
import time
from multiprocessing import Pool, Queue
from queue import Empty

from json_exporter import export_solutions_to_json

# Handle both relative and absolute imports
try:
    from .cost_model import CostModel, get_supported_cpus
    from .bitonic_super_optimizer import BitonicSuperVectorizer, apply_retroactive_input
    from .utils import vector_machine, primitive_type, width_dict
    from .asm_exporter import export_solutions_to_asm
    from .path_selector import PathSelector
    from .bitonic_verifier import (
        load_solutions_from_json,
        extract_verify_steps,
        _verify_path_worker,
        _init_verify_worker,
        num_verify_chunks,
    )
    from .success_progress import SuccessProgress
    from .checkpoint import (
        CheckpointConfig,
        load_checkpoint,
        validate_checkpoint_config,
    )

except ImportError:
    from cost_model import CostModel, get_supported_cpus
    from bitonic_super_optimizer import BitonicSuperVectorizer, apply_retroactive_input
    from utils import vector_machine, primitive_type, width_dict
    from asm_exporter import export_solutions_to_asm
    from path_selector import PathSelector
    from bitonic_verifier import (  # type: ignore
        load_solutions_from_json,
        extract_verify_steps,
        _verify_path_worker,
        _init_verify_worker,
        num_verify_chunks,
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

    Reports progress at chunk granularity (each path is split into multiple
    chunks by the chunked verifier), so the user sees advancement within
    long-running paths.

    Raises SystemExit(1) if any path fails.
    """
    total_paths = len(paths_to_verify)
    total_elements = 2 * (width_dict[vm] // int(prim_type.value[0]))
    chunks_per_path = num_verify_chunks(total_elements)
    total_chunks = total_paths * chunks_per_path

    # Build lightweight picklable jobs (no SolutionNode tree references)
    jobs = [
        (i, extract_verify_steps(path), vm, prim_type, natural_order)
        for i, path in enumerate(paths_to_verify)
    ]

    # Shared queue for per-chunk progress from workers
    chunk_queue: Queue = Queue()

    progress = SuccessProgress.create(
        description_column="[orange1]{task.description}",
        width=60,
        success_style="green",
        attempt_style="green",
        success_label="Verified",
    )
    progress.enable_memory_monitor()
    progress.start()
    task_id = progress.add_task(
        "Verifying paths", total=total_chunks, successes=0, unique=0
    )

    failures = []
    paths_done = 0
    try:
        with Pool(
            processes=max_workers,
            maxtasksperchild=max_tasks_per_child,
            initializer=_init_verify_worker,
            initargs=(chunk_queue,),
        ) as pool:
            # Submit all jobs asynchronously
            async_results = [
                pool.apply_async(_verify_path_worker, (job,)) for job in jobs
            ]

            while paths_done < total_paths:
                # Drain chunk progress queue — each entry = one chunk verified
                try:
                    while True:
                        chunk_queue.get_nowait()
                        progress.update(task_id, advance=1, success=1)
                except Empty:
                    pass

                # Check for completed paths
                for ar in async_results:
                    if ar.ready():
                        try:
                            path_index, result = ar.get()
                        except Exception as exc:
                            raise SystemExit(
                                f"Verification worker crashed: {exc}"
                            ) from exc
                        paths_done += 1
                        if not result.verified:
                            failures.append((path_index, result))

                # Remove completed results to avoid re-processing
                async_results = [ar for ar in async_results if not ar.ready()]

                if paths_done < total_paths:
                    time.sleep(0.1)
    finally:
        progress.stop()

    if failures:
        failures.sort(key=lambda x: x[0])
        print(f"\nVERIFICATION FAILED: {len(failures)}/{total_paths} paths failed")
        for path_index, result in failures:
            print(f"  Path {path_index + 1}: counterexample={result.counterexample}")
        raise SystemExit(1)

    print(f"\nAll {total_paths} paths verified correct.")


def _load_and_select_paths(
    json_path: str, top_k: int | None, target_cpu: str, require_num_vecs: bool = False
):
    """Helper to load solutions, validate metadata, and select top K paths."""
    bundle = load_solutions_from_json(json_path)

    if bundle.vm_name is None or bundle.prim_type_name is None:
        raise SystemExit(
            f"Error: {json_path} has no vector_machine/primitive_type metadata.\n"
            "Re-export the solutions with a current version of the tool."
        )

    if require_num_vecs and bundle.num_vecs is None:
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

    cost_model = CostModel(target_cpu)
    path_selector = PathSelector(cost_model)
    paths = path_selector.select_top_k_paths(bundle.roots, top_k or 10_000)

    return bundle, vm, prim_type, paths


def _load_solutions_from_checkpoint(
    checkpoint_dir: str,
    top_k: int | None,
    target_cpu: str,
):
    """Load solutions from a wave checkpoint directory and select top K paths.

    Returns (solutions, vm, prim_type, natural_order, num_vecs, paths).
    """
    from wave_engine import WaveEngine, WaveConfig
    from wave_checkpoint import WaveCheckpoint

    ckpt = WaveCheckpoint(checkpoint_dir)
    master = ckpt.load_master()
    vm = vector_machine[master.vm]
    pt = primitive_type[master.prim_type]

    # Create a minimal WaveEngine to load checkpoint and export
    config = WaveConfig(
        num_vecs=master.num_vecs,
        vm=vm,
        prim_type=pt,
        gadget_depth=master.gadget_depth,
        natural_order=master.natural_order,
        retroactive_input=master.retroactive_input,
        wave_attempts=1,
        wave_outputs=1,
        max_paths_per_wave=1,
        target_cpus=[],
        max_workers=1,
        max_tasks_per_child=None,
        depth2_threshold=0.1,
    )
    engine = WaveEngine(config)
    engine.resume(checkpoint_dir)
    solutions = engine.export_to_solution_nodes()

    if master.retroactive_input:
        solutions = apply_retroactive_input(solutions)

    print(
        f"Loaded {len(solutions)} roots from checkpoint {checkpoint_dir} "
        f"({vm.name} {pt.name}, natural_order={master.natural_order})"
    )

    cost_model = CostModel(target_cpu)
    path_selector = PathSelector(cost_model)
    paths = path_selector.select_top_k_paths(solutions, top_k or 10_000)

    return solutions, vm, pt, master.natural_order, master.num_vecs, paths


def verify_only_from_json(
    json_path: str,
    top_k: int | None = None,
    target_cpu: str = "generic",
    estimate: bool = False,
    nasm_path: str | None = None,
    llvm_mca_path: str | None = None,
    max_workers: int | None = None,
    max_tasks_per_child: int | None = 1000,
):
    """Load solutions from a JSON file or checkpoint dir and verify without re-running synthesis.

    The JSON file must contain vector_machine and primitive_type metadata
    (written by json_exporter since Feb 2026).

    Args:
        json_path: Path to a JSON solutions file or a wave checkpoint directory.
        top_k: Number of best paths to verify. If None, verify all paths.
        target_cpu: Target CPU for cost model during path selection.
        estimate: If True, run LLVM-MCA performance estimation on selected paths.
        nasm_path: Path to nasm binary for assembly verification (used with estimate).
        llvm_mca_path: Explicit path to llvm-mca binary. If None, auto-detected.
    """
    if os.path.isdir(json_path):
        _solutions, vm, prim_type, natural_order, num_vecs, paths = (
            _load_solutions_from_checkpoint(json_path, top_k, target_cpu)
        )
    else:
        bundle, vm, prim_type, paths = _load_and_select_paths(
            json_path, top_k, target_cpu
        )
        natural_order = bundle.natural_order
        num_vecs = bundle.num_vecs

    _run_verification(
        paths,
        vm,
        prim_type,
        natural_order,
        max_workers=max_workers,
        max_tasks_per_child=max_tasks_per_child,
    )

    if estimate:
        if num_vecs is None:
            raise SystemExit(
                f"Error: {json_path} has no num_vecs metadata. "
                "Re-export the solutions with a current version of the tool."
            )
        from perf_estimator import estimate_solutions, print_estimation_table

        results = estimate_solutions(
            paths,
            vm,
            prim_type,
            num_vecs,
            natural_order,
            target_cpu,
            nasm_path,
            llvm_mca_path,
        )
        print_estimation_table(results, paths)


def estimate_only_from_json(
    json_path: str,
    top_k: int | None = None,
    target_cpu: str = "generic",
    nasm_path: str | None = None,
    llvm_mca_path: str | None = None,
):
    """Load solutions from a JSON file or checkpoint dir and run LLVM-MCA estimation.

    Args:
        json_path: Path to a JSON solutions file or a wave checkpoint directory.
        top_k: Number of best paths to estimate. If None, estimate all paths.
        target_cpu: Target CPU for cost model during path selection.
        nasm_path: Path to nasm binary for assembly verification.
    """
    if os.path.isdir(json_path):
        _solutions, vm, prim_type, natural_order, num_vecs, paths = (
            _load_solutions_from_checkpoint(json_path, top_k, target_cpu)
        )
    else:
        bundle, vm, prim_type, paths = _load_and_select_paths(
            json_path, top_k, target_cpu, require_num_vecs=True
        )
        natural_order = bundle.natural_order
        num_vecs = bundle.num_vecs

    if num_vecs is None:
        raise SystemExit(
            f"Error: {json_path} has no num_vecs metadata. "
            "Re-export the solutions with a current version of the tool."
        )

    from perf_estimator import estimate_solutions, print_estimation_table

    results = estimate_solutions(
        paths,
        vm,
        prim_type,
        num_vecs,
        natural_order,
        target_cpu,
        nasm_path,
        llvm_mca_path,
    )
    print_estimation_table(results, paths)


def export_only_from_json(
    json_path: str,
    export_formats: list[str],
    top_k: int | None = None,
    target_cpu: str = "generic",
    nasm_path: str | None = None,
    output_path: str | None = None,
):
    """Load solutions from a JSON file or checkpoint dir and export to other formats.

    Args:
        json_path: Path to a JSON solutions file or a wave checkpoint directory.
        export_formats: List of output formats (e.g., ["asm"]).
        top_k: Number of best paths to export. If None, export all paths.
        target_cpu: Target CPU for cost model during path selection.
        nasm_path: Path to nasm binary for assembly verification.
        output_path: Explicit output path for the exported file.
    """
    if not export_formats:
        export_formats = ["asm"]

    if os.path.isdir(json_path):
        solutions, vm, prim_type, natural_order, num_vecs, paths = (
            _load_solutions_from_checkpoint(json_path, top_k, target_cpu)
        )
        roots = solutions
    else:
        bundle, vm, prim_type, paths = _load_and_select_paths(
            json_path, top_k, target_cpu, require_num_vecs=True
        )
        roots = bundle.roots
        natural_order = bundle.natural_order
        num_vecs = bundle.num_vecs

    if num_vecs is None:
        raise SystemExit(
            f"Error: {json_path} has no num_vecs metadata. "
            "Re-export the solutions with a current version of the tool."
        )

    for export_format in export_formats:
        if export_format == "asm":
            if output_path is not None:
                asm_output_path = output_path
            else:
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                asm_output_path = f"{base_name}.asm"
            export_solutions_to_asm(
                roots,
                num_vecs,
                prim_type,
                vm,
                asm_output_path,
                selected_paths=paths,
                natural_order=natural_order,
                nasm_path=nasm_path,
            )
        else:
            print(f"Warning: Unknown export format '{export_format}', skipping")


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
    llvm_mca_path: str | None = None,
    max_workers: int | None = None,
    max_tasks_per_child: int | None = 1000,
    retroactive_input: bool = False,
    depth2_threshold: float = 0.1,
):
    """
    Generate bitonic sorter with super-optimized permutation sequences.

    Args:
        num_vecs: Number of SIMD vectors to sort
        prim_type: Primitive type (i32, f32, i64, f64)
        vm: Vector machine (AVX2, AVX512)
        depth_limit: Maximum stage depth to explore (inclusive). If None, all stages are explored.
        top_k: Number of best solutions to keep. If None, all solutions are kept.
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
        estimate: If True, run LLVM-MCA performance estimation on selected paths.
        nasm_path: Path to nasm binary for assembly verification (used with --estimate).
        llvm_mca_path: Explicit path to llvm-mca binary. If None, auto-detected.
        max_tasks_per_child: Maximum tasks per worker process before recycling.
            Limits memory growth in long runs. None disables recycling.
        depth2_threshold: Coverage threshold for depth-2 escalation (default: 0.1).

    Returns:
        List of SolutionNode trees representing different optimized solutions
    """
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
        depth2_threshold=depth2_threshold,
    )

    print(f"Found {len(solutions)} root solutions")

    if retroactive_input:
        before = len(solutions)
        solutions = apply_retroactive_input(solutions)
        print(
            f"Retroactive input: {before} roots → {len(solutions)} "
            f"(deduped {before - len(solutions)})"
        )

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
    output_path = (
        f"bitonic_solutions_{num_vecs}x{vm.name}_{prim_type.name}{order_suffix}.json"
    )
    export_solutions_to_json(
        solutions,
        output_path,
        natural_order=natural_order,
        vm_name=vm.name,
        prim_type_name=prim_type.name,
        num_vecs=num_vecs,
    )

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

    # LLVM-MCA performance estimation
    if estimate:
        from perf_estimator import estimate_solutions, print_estimation_table

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
            llvm_mca_path,
        )
        print_estimation_table(results, paths_to_estimate)

    return solutions


def main():
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
        "--export-only",
        type=str,
        default=None,
        metavar="JSON_FILE",
        help="Skip synthesis and verify, load solutions from a JSON file "
        "and export them to other formats.",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        action="append",
        choices=["asm"],
        help="Export format: 'asm' for readable assembly code. Can be specified multiple times (default: asm)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit output path for exported files (used with --export-only). If not provided, derives from the input JSON filename.",
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
        "--retroactive-input",
        action="store_true",
        default=False,
        help="Post-process Stage 0: retroactively redefine initial state to match "
        "each root's output, replacing Stage 0 gadgets with free identity. "
        "Reduces total cost by eliminating Stage 0 permutation instructions.",
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
        help="Target CPU(s) for cost model and LLVM-MCA scoring. "
        "Comma-separated for multiple (e.g. ZEN4,ADL-P). "
        "Use --list-cpus to see all options (default: generic)",
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
        metavar="CHECKPOINT_DIR",
        help="Resume synthesis from a wave checkpoint directory. "
        "Restores the TransitionTable and wave count from the checkpoint.",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        default=False,
        help="Run LLVM-MCA performance estimation on selected paths",
    )
    parser.add_argument(
        "--nasm-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to nasm binary for assembly verification (used with --estimate)",
    )
    parser.add_argument(
        "--llvm-mca-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit path to the llvm-mca binary. "
        "If not specified, searches Homebrew locations and PATH.",
    )
    parser.add_argument(
        "--estimate-only",
        type=str,
        default=None,
        metavar="JSON_FILE",
        help="Skip synthesis and verification; load solutions from a JSON file "
        "and run LLVM-MCA performance estimation only.",
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
        "--depth2-threshold",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Coverage threshold for depth-2 escalation (default: 0.1). "
        "After depth-1 completes, if coverage >= threshold, depth-2 is skipped. "
        "Set to 0 to always escalate. Values > 1.0 are allowed as multipliers.",
    )
    parser.add_argument(
        "--list-cpus",
        action="store_true",
        default=False,
        help="List all supported CPU architectures with uops.info and LLVM-MCA availability, then exit",
    )
    parser.add_argument(
        "--max-waves",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of waves before termination. If not specified, runs until "
        "Ctrl-C or exhaustion.",
    )
    parser.add_argument(
        "--wave-attempts",
        type=int,
        default=10_000,
        metavar="N",
        help="Budget per stage per wave in attempts (default: 10000).",
    )
    parser.add_argument(
        "--wave-outputs",
        type=int,
        default=100,
        metavar="N",
        help="Budget per stage per wave in distinct outputs (default: 100).",
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
            mca_col = info.llvm_mca_cpu if info.llvm_mca_cpu else "-"
            rows.append([info.canonical, aliases, info.vendor, uops_col, mca_col])

        from tabulate import tabulate

        headers = ["Arch", "Alias", "Vendor", "uops.info", "llvm-mca CPU"]
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
            llvm_mca_path=args.llvm_mca_path,
            max_workers=args.max_workers,
            max_tasks_per_child=max_tasks_per_child,
        )
        raise SystemExit(0)

    # --estimate-only mode: load JSON and run LLVM-MCA estimation without synthesis or verification
    if args.estimate_only is not None:
        estimate_only_from_json(
            args.estimate_only,
            top_k=args.top_k,
            target_cpu=args.target_cpu,
            nasm_path=args.nasm_path,
            llvm_mca_path=args.llvm_mca_path,
        )
        raise SystemExit(0)

    # --export-only mode: load JSON and export solutions
    if args.export_only is not None:
        export_only_from_json(
            args.export_only,
            args.export_format,
            top_k=args.top_k,
            target_cpu=args.target_cpu,
            nasm_path=args.nasm_path,
            output_path=args.output_path,
        )
        raise SystemExit(0)

    # Normal synthesis mode requires --vector-machine and --datatype
    if not args.vector_machine or not args.datatype:
        parser.error("--vector-machine and --datatype are required")

    # Convert string arguments to Enum members
    vm = vector_machine[args.vector_machine]
    prim_type = primitive_type[args.datatype]

    # Parse target CPUs: first one used for CostModel, all used for LLVM-MCA
    primary_target_cpu = args.target_cpu.split(",")[0].strip()

    smt2_dump_dir = None
    if args.dump_smt2:
        smt2_dump_dir = tempfile.mkdtemp(prefix="vxsort_smt2_", dir="/tmp")
        print(f"SMT2 dump directory: {smt2_dump_dir}")

    from wave_engine import WaveEngine, WaveConfig

    wave_config = WaveConfig(
        num_vecs=args.num_vecs,
        vm=vm,
        prim_type=prim_type,
        gadget_depth=args.gadget_depth,
        natural_order=args.natural_order,
        retroactive_input=args.retroactive_input,
        wave_attempts=args.wave_attempts,
        wave_outputs=args.wave_outputs,
        max_paths_per_wave=50,
        target_cpus=[
            c.strip() for c in args.target_cpu.split(",") if c.strip() != "generic"
        ],
        max_workers=args.max_workers,
        max_tasks_per_child=max_tasks_per_child,
        max_unique_outputs=args.max_gadget_solutions,
        depth2_threshold=args.depth2_threshold,
        smt2_dump_dir=smt2_dump_dir,
        checkpoint_dir=args.checkpoint_dir,
        llvm_mca_path=getattr(args, "llvm_mca_path", None),
        max_waves=args.max_waves,
        top_k=args.top_k or 10,
    )

    engine = WaveEngine(wave_config)

    if args.resume:
        engine.resume(args.resume)

    engine.run()

    # Export to legacy SolutionNode format
    solutions = engine.export_to_solution_nodes()

    print(f"Found {len(solutions)} root solutions")

    if args.retroactive_input:
        before = len(solutions)
        solutions = apply_retroactive_input(solutions)
        print(
            f"Retroactive input: {before} roots -> {len(solutions)} "
            f"(deduped {before - len(solutions)})"
        )

    # Filter to top K cheapest root-to-leaf paths if requested
    selected_paths = None
    if args.top_k is not None:
        total_paths = _count_dag_paths(solutions)
        if total_paths > args.top_k:
            print(
                f"Filtering to top {args.top_k} cheapest paths "
                f"(out of {total_paths} total)..."
            )
            cost_model = CostModel(primary_target_cpu)
            path_selector = PathSelector(cost_model)
            solutions, selected_paths = path_selector.prune_to_top_k_paths(
                solutions, args.top_k
            )
            print(f"Kept {len(solutions)} roots after pruning")

    # Export solutions to JSON
    order_suffix = "_natural" if args.natural_order else ""
    output_path = (
        f"bitonic_solutions_{args.num_vecs}x{vm.name}"
        f"_{prim_type.name}{order_suffix}.json"
    )
    export_solutions_to_json(
        solutions,
        output_path,
        natural_order=args.natural_order,
        vm_name=vm.name,
        prim_type_name=prim_type.name,
        num_vecs=args.num_vecs,
    )

    # End-to-end verification
    if args.verify:
        paths_to_verify = selected_paths
        if paths_to_verify is None:
            cost_model = CostModel(primary_target_cpu)
            path_selector = PathSelector(cost_model)
            paths_to_verify = path_selector.select_top_k_paths(
                solutions, args.top_k or 10_000
            )

        _run_verification(
            paths_to_verify,
            vm,
            prim_type,
            args.natural_order,
            max_workers=args.max_workers,
            max_tasks_per_child=max_tasks_per_child,
        )

    # LLVM-MCA performance estimation
    if args.estimate:
        from perf_estimator import estimate_solutions, print_estimation_table

        paths_to_estimate = selected_paths
        if paths_to_estimate is None:
            cost_model = CostModel(primary_target_cpu)
            path_selector = PathSelector(cost_model)
            paths_to_estimate = path_selector.select_top_k_paths(
                solutions, args.top_k or 10_000
            )

        results = estimate_solutions(
            paths_to_estimate,
            vm,
            prim_type,
            args.num_vecs,
            args.natural_order,
            primary_target_cpu,
            args.nasm_path,
            args.llvm_mca_path,
        )
        print_estimation_table(results, paths_to_estimate)


def estimate_main():
    """Entry point for 'uv run estimate'. Prepends --estimate-only to argv."""
    import sys

    sys.argv.insert(1, "--estimate-only")
    main()


def verify_main():
    """Entry point for 'uv run verify'. Prepends --verify-only to argv."""
    import sys

    sys.argv.insert(1, "--verify-only")
    main()


def export_main():
    """Entry point for 'uv run export'. Prepends --export-only to argv."""
    import sys

    sys.argv.insert(1, "--export-only")
    main()


if __name__ == "__main__":
    main()
