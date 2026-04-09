"""LLVM-MCA based performance estimation for bitonic sort solutions."""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from runtime_logging import log_event


@dataclass
class McaResult:
    """Result from running llvm-mca on a single solution."""

    solution_index: int
    asm_path: str
    throughput: float  # BlockRThroughput (cycles/iteration, resource-bound)
    simulated_cycles: float  # TotalCycles/Iterations (simulated, incl. dependencies)
    analysis_path: str = ""  # Path to full llvm-mca text analysis
    warnings: list[str] = field(default_factory=list)


def find_llvm_mca(explicit_cmd: str | None) -> str | None:
    """Locate the llvm-mca binary or command.

    If *explicit_cmd* is provided, returns it directly (may be a full
    command like ``"docker run -i silkeh/clang /usr/bin/llvm-mca"``).
    Otherwise searches common locations and PATH.
    """
    if explicit_cmd:
        return explicit_cmd

    # Common Homebrew locations on macOS (Apple Silicon and Intel)
    for homebrew_path in [
        "/opt/homebrew/opt/llvm/bin/llvm-mca",  # Apple Silicon
        "/usr/local/opt/llvm/bin/llvm-mca",  # Intel Mac
    ]:
        if Path(homebrew_path).is_file():
            return homebrew_path

    return shutil.which("llvm-mca")


def _parse_mca_json(
    mca_json: dict,
    solution_index: int,
    asm_path: str,
) -> McaResult:
    """Parse llvm-mca JSON output into an McaResult."""
    warnings: list[str] = []

    regions = mca_json.get("CodeRegions", [])
    if not regions:
        warnings.append("No code regions found in llvm-mca output")
        return McaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            throughput=-1.0,
            simulated_cycles=-1.0,
            warnings=warnings,
        )

    # Use first region (we emit one region per solution)
    summary = regions[0].get("SummaryView", {})
    block_rthroughput = summary.get("BlockRThroughput", -1.0)
    total_cycles = summary.get("TotalCycles", 0)
    iterations = summary.get("Iterations", 1)
    simulated = total_cycles / iterations if iterations > 0 else -1.0

    return McaResult(
        solution_index=solution_index,
        asm_path=asm_path,
        throughput=block_rthroughput,
        simulated_cycles=simulated,
        warnings=warnings,
    )


def run_llvm_mca(
    asm_text: str,
    mcpu: str,
    llvm_mca_cmd: str,
    solution_index: int,
    output_dir: str | None = None,
    include_timeline: bool = True,
) -> McaResult:
    """Run llvm-mca on Intel-syntax assembly and return parsed results.

    Args:
        asm_text: Intel-syntax assembly text (with LLVM-MCA markers).
        mcpu: CPU model for -mcpu= flag.
        llvm_mca_cmd: Command to run llvm-mca.  Can be a plain path or a
            full command string like
            ``"docker run -i silkeh/clang /usr/bin/llvm-mca"``.
            Split via :func:`shlex.split`.
        solution_index: 1-based solution index for reporting.
        output_dir: Optional directory for saving .s input files.
        include_timeline: When True (default), run a second llvm-mca pass
            with ``-timeline`` and write full text analysis output.
            Disable for lower-overhead scoring when only JSON metrics
            are needed.

    Returns:
        McaResult with throughput and simulated cycle estimates.
    """
    logger = logging.getLogger("vxsort.runtime.llvm_mca")
    started_at = time.perf_counter()

    base_parts = shlex.split(llvm_mca_cmd)
    log_event(
        logger,
        "DEBUG",
        "mca_run_started",
        solution_index=solution_index,
        mcpu=mcpu,
        include_timeline=include_timeline,
    )

    # Save asm to file for debugging if output_dir specified
    asm_path = ""
    if output_dir:
        asm_path = str(Path(output_dir) / f"solution_{solution_index:03d}.s")
        Path(asm_path).write_text(asm_text)

    base_cmd = [
        *base_parts,
        "-march=x86-64",
        f"-mcpu={mcpu}",
        "-x86-asm-syntax=intel",
    ]

    # Pipe via stdin (works with docker, local binary, etc.)
    json_cmd = [*base_cmd, "--json"]

    try:
        result = subprocess.run(
            json_cmd,
            input=asm_text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            stderr_tail = result.stderr.strip()[-400:]
            log_event(
                logger,
                "ERROR",
                "mca_run_failed",
                solution_index=solution_index,
                mcpu=mcpu,
                returncode=result.returncode,
                elapsed_ms=round(elapsed_ms, 3),
                stderr_tail=stderr_tail,
            )
            return McaResult(
                solution_index=solution_index,
                asm_path=asm_path,
                throughput=-1.0,
                simulated_cycles=-1.0,
                warnings=[
                    f"llvm-mca failed (rc={result.returncode}): "
                    f"{result.stderr.strip()}"
                ],
            )

        mca_json = json.loads(result.stdout)
        mca_result = _parse_mca_json(mca_json, solution_index, asm_path)

        # Optional second pass for full text timeline analysis.
        if include_timeline:
            mca_result.analysis_path = _run_full_analysis(
                base_cmd,
                asm_text,
                output_dir,
                solution_index,
                mca_result.warnings,
            )

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        log_event(
            logger,
            "DEBUG",
            "mca_run_completed",
            solution_index=solution_index,
            mcpu=mcpu,
            elapsed_ms=round(elapsed_ms, 3),
            throughput=mca_result.throughput,
            simulated_cycles=mca_result.simulated_cycles,
        )

        return mca_result

    except FileNotFoundError:
        log_event(
            logger,
            "ERROR",
            "mca_run_missing_binary",
            solution_index=solution_index,
            command=base_parts[0] if base_parts else llvm_mca_cmd,
        )
        return McaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            throughput=-1.0,
            simulated_cycles=-1.0,
            warnings=[f"llvm-mca command not found: '{base_parts[0]}'"],
        )
    except json.JSONDecodeError as e:
        log_event(
            logger,
            "ERROR",
            "mca_run_json_parse_failed",
            solution_index=solution_index,
            error=str(e),
        )
        return McaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            throughput=-1.0,
            simulated_cycles=-1.0,
            warnings=[f"Failed to parse llvm-mca JSON output: {e}"],
        )
    except subprocess.TimeoutExpired:
        log_event(
            logger,
            "ERROR",
            "mca_run_timeout",
            solution_index=solution_index,
            timeout_seconds=30,
        )
        return McaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            throughput=-1.0,
            simulated_cycles=-1.0,
            warnings=["llvm-mca timed out after 30s"],
        )


def _run_full_analysis(
    base_cmd: list[str],
    asm_text: str,
    output_dir: str | None,
    solution_index: int,
    warnings: list[str],
) -> str:
    """Run llvm-mca without --json and save the full text analysis."""
    text_cmd = [*base_cmd, "-timeline"]
    try:
        result = subprocess.run(
            text_cmd,
            input=asm_text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            warnings.append(
                f"llvm-mca text analysis failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )
            return ""

        analysis_text = result.stdout
        if output_dir:
            analysis_path = str(
                Path(output_dir) / f"solution_{solution_index:03d}_analysis.txt"
            )
            Path(analysis_path).write_text(analysis_text)
        else:
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".analysis.txt",
                prefix=f"vxsort_mca_{solution_index:03d}_",
                delete=False,
            )
            tmp.write(analysis_text)
            tmp.close()
            analysis_path = tmp.name
        return analysis_path

    except (subprocess.TimeoutExpired, OSError) as e:
        warnings.append(f"llvm-mca text analysis error: {e}")
        return ""
