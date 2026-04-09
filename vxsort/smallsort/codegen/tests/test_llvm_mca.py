"""Tests for LLVM-MCA integration."""

import subprocess
from unittest.mock import patch

import pytest
from cost_model import resolve_llvm_mca_cpu, get_supported_cpus
from perf_estimator import sanitize_asm_for_llvm_mca
from util.llvm_mca_runner import (
    McaResult,
    find_llvm_mca,
    _parse_mca_json,
    run_llvm_mca,
)


class TestLlvmMcaCpuResolution:
    """Tests for resolving target-cpu to llvm-mca -mcpu values."""

    def test_resolve_known_intel_cpu(self):
        assert resolve_llvm_mca_cpu("SKX") == "skylake-avx512"

    def test_resolve_known_amd_cpu(self):
        assert resolve_llvm_mca_cpu("ZEN4") == "znver4"

    def test_resolve_by_alias(self):
        assert resolve_llvm_mca_cpu("tigerlake") == "icelake-server"

    def test_resolve_generic_returns_none(self):
        assert resolve_llvm_mca_cpu("generic") is None

    def test_all_archs_have_llvm_mca_cpu(self):
        """Every architecture in the registry must have an llvm_mca_cpu value."""
        for info in get_supported_cpus():
            assert (
                info.llvm_mca_cpu is not None
            ), f"{info.canonical} has no llvm_mca_cpu mapping"


class TestSanitizeAsmForLlvmMca:
    """Tests for NASM → LLVM-Intel sanitization."""

    def test_rel_memory_operand(self):
        result = sanitize_asm_for_llvm_mca(
            "    vmovdqa32            ymm5, [rel cv_s0_t0]  ; load CV\n"
        )
        assert "[rip + cv_s0_t0]" in result
        assert "[rel" not in result

    def test_semicolon_comment_to_hash(self):
        result = sanitize_asm_for_llvm_mca(
            "    vmovdqa32            ymm2, ymm0  ; save input\n"
        )
        assert "# save input" in result
        assert ";" not in result

    def test_comment_only_line(self):
        result = sanitize_asm_for_llvm_mca("    ; Stage 0\n")
        assert result.strip() == "# Stage 0"

    def test_osaca_begin_to_mca_begin(self):
        result = sanitize_asm_for_llvm_mca("    ; OSACA-BEGIN\n")
        assert result.strip() == "# LLVM-MCA-BEGIN"

    def test_osaca_end_to_mca_end(self):
        result = sanitize_asm_for_llvm_mca("    ; OSACA-END\n")
        assert result.strip() == "# LLVM-MCA-END"

    def test_strip_bits_64(self):
        assert sanitize_asm_for_llvm_mca("bits 64\n").strip() == ""

    def test_strip_default_rel(self):
        assert sanitize_asm_for_llvm_mca("default rel\n").strip() == ""

    def test_strip_section_directives(self):
        assert sanitize_asm_for_llvm_mca("section .text\n").strip() == ""
        assert sanitize_asm_for_llvm_mca("section .rodata\n").strip() == ""

    def test_strip_global(self):
        assert sanitize_asm_for_llvm_mca("global _osaca_kernel\n").strip() == ""

    def test_strip_labels(self):
        assert sanitize_asm_for_llvm_mca("_osaca_kernel:\n").strip() == ""
        assert sanitize_asm_for_llvm_mca("cv_s0_t0:\n").strip() == ""

    def test_strip_data_directives(self):
        assert sanitize_asm_for_llvm_mca("    dd 7, 6, 5, 4\n").strip() == ""
        assert sanitize_asm_for_llvm_mca("    dq 1, 0\n").strip() == ""

    def test_strip_align(self):
        assert sanitize_asm_for_llvm_mca("align 32\n").strip() == ""

    def test_strip_ret(self):
        assert sanitize_asm_for_llvm_mca("    ret\n").strip() == ""

    def test_instructions_pass_through_unchanged(self):
        """Instructions should not be modified — no operand reordering."""
        result = sanitize_asm_for_llvm_mca(
            "    vpermq               ymm3, ymm2, 0xd8\n"
        )
        assert result.strip() == "vpermq               ymm3, ymm2, 0xd8"

    def test_full_block(self):
        intel_asm = (
            "bits 64\n"
            "default rel\n"
            "\n"
            "section .rodata\n"
            "align 32\n"
            "cv_s0_t0:\n"
            "    dd 3, 2, 1, 0\n"
            "\n"
            "section .text\n"
            "global _osaca_kernel\n"
            "_osaca_kernel:\n"
            "    ; OSACA-BEGIN\n"
            "    ; Stage 0\n"
            "    vmovdqa32            ymm5, [rel cv_s0_t0]  ; [3, 2, 1, 0]\n"
            "    vpermq               ymm2, ymm0, 0xd8\n"
            "    vpcmpgtq             ymm3, ymm0, ymm2\n"
            "    ; OSACA-END\n"
            "    ret\n"
        )
        result = sanitize_asm_for_llvm_mca(intel_asm)
        lines = [line for line in result.strip().split("\n") if line.strip()]
        assert lines[0] == "# LLVM-MCA-BEGIN"
        assert lines[1] == "# Stage 0"
        assert "[rip + cv_s0_t0]" in lines[2]
        assert "ymm5" in lines[2]
        # Instructions stay in Intel syntax (dest first)
        assert "vpermq               ymm2, ymm0, 0xd8" in lines[3]
        assert "vpcmpgtq             ymm3, ymm0, ymm2" in lines[4]
        assert lines[-1] == "# LLVM-MCA-END"


class TestFindLlvmMca:
    """Tests for locating the llvm-mca binary."""

    def test_find_with_explicit_path(self):
        result = find_llvm_mca("/usr/bin/llvm-mca")
        assert result == "/usr/bin/llvm-mca"

    @patch("util.llvm_mca_runner.Path.is_file", return_value=False)
    @patch("util.llvm_mca_runner.shutil.which", return_value="/usr/bin/llvm-mca")
    def test_find_on_path(self, mock_which, mock_is_file):
        result = find_llvm_mca(None)
        assert result == "/usr/bin/llvm-mca"

    @patch("util.llvm_mca_runner.Path.is_file", return_value=False)
    @patch("util.llvm_mca_runner.shutil.which", return_value=None)
    def test_find_returns_none_when_missing(self, mock_which, mock_is_file):
        result = find_llvm_mca(None)
        assert result is None


class TestParseMcaJson:
    """Tests for parsing llvm-mca JSON output."""

    def test_parse_summary(self):
        mca_json = {
            "CodeRegions": [
                {
                    "SummaryView": {
                        "BlockRThroughput": 3.5,
                        "TotalCycles": 350,
                        "Iterations": 100,
                        "Instructions": 500,
                        "TotaluOps": 600,
                        "IPC": 1.43,
                        "uOpsPerCycle": 1.71,
                        "DispatchWidth": 6,
                    },
                }
            ],
        }
        result = _parse_mca_json(mca_json, solution_index=1, asm_path="/tmp/test.s")
        assert result.throughput == pytest.approx(3.5)
        assert result.simulated_cycles == pytest.approx(3.5)  # 350/100
        assert result.solution_index == 1

    def test_parse_empty_regions(self):
        mca_json = {"CodeRegions": []}
        result = _parse_mca_json(mca_json, solution_index=1, asm_path="/tmp/test.s")
        assert result.throughput == -1.0
        assert len(result.warnings) > 0


class TestMcaResult:
    """Tests for the McaResult data class."""

    def test_result_fields(self):
        result = McaResult(
            solution_index=1,
            asm_path="/tmp/test.s",
            throughput=3.5,
            simulated_cycles=4.2,
            warnings=[],
        )
        assert result.solution_index == 1
        assert result.throughput == 3.5
        assert result.simulated_cycles == 4.2


class TestRunLlvmMcaModes:
    """Mode-specific behavior for llvm-mca invocations."""

    @patch(
        "util.llvm_mca_runner.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["llvm-mca"],
            returncode=0,
            stdout=(
                '{"CodeRegions":[{"SummaryView":'
                '{"BlockRThroughput":3.5,"TotalCycles":350,"Iterations":100}}]}'
            ),
            stderr="",
        ),
    )
    def test_skip_timeline_analysis_when_disabled(self, mock_run, caplog):
        with caplog.at_level("DEBUG", logger="vxsort.runtime.llvm_mca"):
            result = run_llvm_mca(
                "vmovdqa ymm1, ymm0\n",
                "znver5",
                "llvm-mca",
                solution_index=1,
                include_timeline=False,
            )

        assert result.throughput == pytest.approx(3.5)
        assert result.simulated_cycles == pytest.approx(3.5)
        assert result.analysis_path == ""
        assert mock_run.call_count == 1
        assert any(rec.getMessage() == "mca_run_started" for rec in caplog.records)
        assert any(rec.getMessage() == "mca_run_completed" for rec in caplog.records)


@pytest.mark.skipif(
    find_llvm_mca(None) is None,
    reason="llvm-mca not found on system",
)
class TestLlvmMcaIntegration:
    """Integration tests that run the real llvm-mca binary."""

    def test_simple_avx2_snippet(self):
        intel_asm = (
            "# LLVM-MCA-BEGIN test\n"
            "vmovdqa ymm1, ymm0\n"
            "vpermq ymm2, ymm1, 0xd8\n"
            "# LLVM-MCA-END test\n"
        )
        mca_path = find_llvm_mca(None)
        result = run_llvm_mca(intel_asm, "znver4", mca_path, solution_index=1)
        assert result.throughput > 0
        assert result.simulated_cycles > 0
        assert len(result.warnings) == 0
