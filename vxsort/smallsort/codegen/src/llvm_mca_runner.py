"""LLVM-MCA based performance estimation for bitonic sort solutions."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class McaResult:
    """Result from running llvm-mca on a single solution."""

    solution_index: int
    asm_path: str
    throughput: float  # BlockRThroughput (cycles/iteration, resource-bound)
    simulated_cycles: float  # TotalCycles/Iterations (simulated, incl. dependencies)
    analysis_path: str = ""  # Path to full llvm-mca text analysis
    warnings: list[str] = field(default_factory=list)


def find_llvm_mca(explicit_path: str | None) -> str | None:
    """Locate the llvm-mca binary.

    Checks explicit_path first, then common Homebrew locations, then PATH.
    """
    if explicit_path:
        return explicit_path

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
    llvm_mca_path: str,
    solution_index: int,
    output_dir: str | None = None,
) -> McaResult:
    """Run llvm-mca on Intel-syntax assembly and return parsed results.

    Args:
        asm_text: Intel-syntax assembly text (with LLVM-MCA markers).
        mcpu: CPU model for -mcpu= flag.
        llvm_mca_path: Path to the llvm-mca binary.
        solution_index: 1-based solution index for reporting.
        output_dir: Optional directory for saving .s input files.

    Returns:
        McaResult with throughput and simulated cycle estimates.
    """
    # Write asm to temp file
    if output_dir:
        asm_path = str(Path(output_dir) / f"solution_{solution_index:03d}.s")
        Path(asm_path).write_text(asm_text)
    else:
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".s",
            prefix=f"vxsort_mca_{solution_index:03d}_",
            delete=False,
        )
        tmp.write(asm_text)
        tmp.close()
        asm_path = tmp.name

    base_cmd = [
        llvm_mca_path,
        "-march=x86-64",
        f"-mcpu={mcpu}",
        "-x86-asm-syntax=intel",
    ]
    json_cmd = [*base_cmd, "--json", asm_path]

    try:
        result = subprocess.run(
            json_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return McaResult(
                solution_index=solution_index,
                asm_path=asm_path,
                throughput=-1.0,
                simulated_cycles=-1.0,
                warnings=[
                    f"llvm-mca failed (rc={result.returncode}): {result.stderr.strip()}"
                ],
            )

        mca_json = json.loads(result.stdout)
        mca_result = _parse_mca_json(mca_json, solution_index, asm_path)

        # Run again without --json for full text analysis
        mca_result.analysis_path = _run_full_analysis(
            base_cmd,
            asm_path,
            output_dir,
            solution_index,
            mca_result.warnings,
        )

        return mca_result

    except FileNotFoundError:
        return McaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            throughput=-1.0,
            simulated_cycles=-1.0,
            warnings=[f"llvm-mca not found at '{llvm_mca_path}'"],
        )
    except json.JSONDecodeError as e:
        return McaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            throughput=-1.0,
            simulated_cycles=-1.0,
            warnings=[f"Failed to parse llvm-mca JSON output: {e}"],
        )
    except subprocess.TimeoutExpired:
        return McaResult(
            solution_index=solution_index,
            asm_path=asm_path,
            throughput=-1.0,
            simulated_cycles=-1.0,
            warnings=["llvm-mca timed out after 30s"],
        )


def _run_full_analysis(
    base_cmd: list[str],
    asm_path: str,
    output_dir: str | None,
    solution_index: int,
    warnings: list[str],
) -> str:
    """Run llvm-mca without --json and save the full text analysis."""
    text_cmd = [*base_cmd, "-timeline", asm_path]
    try:
        result = subprocess.run(
            text_cmd,
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
        else:
            analysis_path = str(Path(asm_path).with_suffix(".analysis.txt"))
        Path(analysis_path).write_text(analysis_text)
        return analysis_path

    except (subprocess.TimeoutExpired, OSError) as e:
        warnings.append(f"llvm-mca text analysis error: {e}")
        return ""
