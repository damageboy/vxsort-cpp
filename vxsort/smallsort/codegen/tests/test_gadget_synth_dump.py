"""Tests for the standalone gadget synthesizer dump tool."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest  # type: ignore[import-not-found]

from bitonic_types import (  # type: ignore[import-not-found]
    GadgetGraph,
    InputRef,
    IntrinsicNode,
    Mux,
    Symbolic,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DUMPER = _REPO_ROOT / "tools" / "dump_gadget_synth.py"
_COMPARER = _REPO_ROOT / "tools" / "compare_gadget_synth_dumps.py"
_OBJECT_ID_RE = re.compile(r"0x[0-9a-fA-F]+|<[^>]+ object at")
_OBJECT_AT_RE = re.compile(r"<[^>]+ object at")
_REGISTER_ARG_RE = re.compile(r"^(?:top|bottom|prev|result_\d+)$")
_AVX512_UNMASKED_INTRINSICS = {
    "i32": (
        "_mm512_permutexvar_epi32",
        "_mm512_permute_ps",
        "_mm512_permutevar_ps",
        "_mm512_permutex2var_epi32",
        "_mm512_shuffle_ps",
        "_mm512_unpacklo_epi32",
        "_mm512_unpackhi_epi32",
        "_mm512_shuffle_i32x4",
        "_mm512_alignr_epi32",
    ),
    "i64": (
        "_mm512_permutexvar_epi64",
        "_mm512_permute_pd",
        "_mm512_permutevar_pd",
        "_mm512_permutex2var_epi64",
        "_mm512_shuffle_pd",
        "_mm512_unpacklo_epi64",
        "_mm512_unpackhi_epi64",
        "_mm512_shuffle_i32x4",
        "_mm512_alignr_epi64",
    ),
}
_AVX512_UNMASKED_FILTERS = {
    dtype: ",".join(intrinsics)
    for dtype, intrinsics in _AVX512_UNMASKED_INTRINSICS.items()
}
_AVX512_MASKED_INTRINSICS = {
    "i32": (
        "_mm512_mask_permutexvar_epi32",
        "_mm512_mask_permute_ps",
        "_mm512_mask_permutevar_ps",
        "_mm512_mask_permutex2var_epi32",
        "_mm512_mask_shuffle_ps",
        "_mm512_mask_unpacklo_epi32",
        "_mm512_mask_unpackhi_epi32",
        "_mm512_mask_shuffle_i32x4",
        "_mm512_mask_alignr_epi32",
    ),
    "i64": (
        "_mm512_mask_permutexvar_epi64",
        "_mm512_mask_permute_pd",
        "_mm512_mask_permutevar_pd",
        "_mm512_mask_permutex2var_epi64",
        "_mm512_mask_shuffle_pd",
        "_mm512_mask_unpacklo_epi64",
        "_mm512_mask_unpackhi_epi64",
        "_mm512_mask_shuffle_i32x4",
        "_mm512_mask_alignr_epi64",
    ),
}
_AVX512_MASKED_FILTERS = {
    dtype: ",".join(intrinsics)
    for dtype, intrinsics in _AVX512_MASKED_INTRINSICS.items()
}


def _walk_json(value: Any):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records))


def _assert_saw_single_to_dual_mux_record(
    records: list[dict[str, Any]],
    *,
    single_intrinsic: str,
    dual_intrinsic: str,
) -> None:
    for record in records:
        for instruction_list_name in ["top_instructions", "bottom_instructions"]:
            instructions = record[instruction_list_name]
            if (
                len(instructions) == 2
                and instructions[0]["name"] == single_intrinsic
                and instructions[1]["name"] == dual_intrinsic
                and any(value == "prev" for value in instructions[1]["args"].values())
            ):
                return

    raise AssertionError(
        f"did not find synthesized single→dual mux record for "
        f"{single_intrinsic}→{dual_intrinsic}"
    )


def _assert_saw_shared_prefix_record(
    records: list[dict[str, Any]],
    *,
    prefix_intrinsic: str,
    tail_intrinsic: str,
) -> None:
    for record in records:
        top_instructions = record["top_instructions"]
        bottom_instructions = record["bottom_instructions"]
        if len(top_instructions) != 3 or len(bottom_instructions) != 3:
            continue
        if (
            top_instructions[0]["name"] == prefix_intrinsic
            and top_instructions[1]["name"] == prefix_intrinsic
            and top_instructions[2]["name"] == tail_intrinsic
            and bottom_instructions[0]["name"] == prefix_intrinsic
            and bottom_instructions[1]["name"] == prefix_intrinsic
            and bottom_instructions[2]["name"] == tail_intrinsic
        ):
            return

    raise AssertionError(
        f"did not find synthesized shared-prefix record for "
        f"{prefix_intrinsic}→{tail_intrinsic}"
    )


def _run_comparer(
    expected: Path,
    actual: Path,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_COMPARER),
            str(expected),
            str(actual),
            *extra_args,
        ],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _dump_python(mode: str, output: Path, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(_DUMPER), mode, *args, "--output", str(output)],
        cwd=_REPO_ROOT,
        check=True,
    )


def _dump_rust(mode: str, output: Path, *args: str) -> None:
    subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "gadget_synth",
            "--bin",
            "dump_gadget_synth",
            "--",
            mode,
            *args,
            "--output",
            str(output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def _collect_symbolics(value: Any) -> list[dict[str, Any]]:
    return [
        item
        for item in _walk_json(value)
        if isinstance(item, dict) and item.get("kind") == "symbolic"
    ]


def _assert_allowed_concrete_arg(value: Any) -> bool:
    """Assert a synthesized instruction arg is a concrete portable value."""
    if isinstance(value, str):
        assert _REGISTER_ARG_RE.fullmatch(value)
        return False
    if isinstance(value, bool):
        raise AssertionError("booleans are not valid concrete gadget args")
    if isinstance(value, int):
        return False
    if isinstance(value, dict):
        assert set(value) == {"kind", "bits", "hex"}
        assert value["kind"] == "bitvec"
        assert value["bits"] in {256, 512}
        assert isinstance(value["hex"], str)
        assert value["hex"].startswith("0x")
        assert len(value["hex"]) == 2 + value["bits"] // 4
        assert int(value["hex"], 16) < (1 << value["bits"])
        return True
    raise AssertionError(f"unexpected concrete gadget arg type: {type(value)}")


def _assert_records_exclude_masked_intrinsics(records: list[dict[str, Any]]) -> None:
    for record in records:
        for item in _walk_json(record):
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                assert "_mask_" not in item["name"]


def _is_intrinsic_name_record(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and (item.get("kind") == "intrinsic" or "args" in item)
    )


def _assert_records_use_only_intrinsics(
    records: list[dict[str, Any]], allowed_intrinsics: set[str]
) -> None:
    seen = set()
    for record in records:
        for item in _walk_json(record):
            if _is_intrinsic_name_record(item):
                seen.add(item["name"])
                assert item["name"] in allowed_intrinsics
    assert seen


def _assert_records_include_masked_intrinsics(records: list[dict[str, Any]]) -> None:
    for record in records:
        for item in _walk_json(record):
            if _is_intrinsic_name_record(item) and "_mask_" in item["name"]:
                return
    raise AssertionError("did not find any masked intrinsic record")


def _assert_saw_synthesized_instruction(
    records: list[dict[str, Any]], intrinsic_name: str
) -> None:
    for record in records:
        for instruction_list_name in ["top_instructions", "bottom_instructions"]:
            if any(
                instruction["name"] == intrinsic_name
                for instruction in record[instruction_list_name]
            ):
                return

    raise AssertionError(f"did not find synthesized {intrinsic_name} record")


def test_templates_cli_dumps_stable_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "templates.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(_DUMPER),
            "templates",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--gadget-depth",
            "1",
            "--output",
            str(output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )

    records = _read_jsonl(output)
    assert records
    assert records == sorted(
        records, key=lambda record: json.dumps(record, sort_keys=True)
    )

    for record in records:
        assert record["kind"] == "template"
        assert record["arch"] == "avx2"
        assert record["dtype"] == "i64"
        assert record["gadget_depth"] == 1
        assert record["tier"] == "shallow"
        assert "graph" in record
        assert not _OBJECT_ID_RE.search(json.dumps(record, sort_keys=True))

        for symbolic in _collect_symbolics(record):
            assert set(symbolic) == {"kind", "role", "bits", "var_id"}
            assert symbolic["var_id"].startswith("v")
            assert isinstance(symbolic["bits"], int)


def test_synthesized_cli_dumps_identity_pairs_schema(tmp_path: Path) -> None:
    output = tmp_path / "synthesized.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(_DUMPER),
            "synthesized",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--gadget-depth",
            "1",
            "--fixture",
            "identity_pairs",
            "--max-unique-outputs",
            "3",
            "--output",
            str(output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )

    records = _read_jsonl(output)
    assert records
    assert records == sorted(
        records, key=lambda record: json.dumps(record, sort_keys=True)
    )

    required_fields = {
        "kind",
        "arch",
        "dtype",
        "gadget_depth",
        "fixture",
        "input_state",
        "target_pairs",
        "output_state",
        "top_instructions",
        "bottom_instructions",
    }
    expected_input_state = {"top": [0, 1, 2, 3], "bottom": [4, 5, 6, 7]}
    expected_target_pairs = [[0, 4], [1, 5], [2, 6], [3, 7]]
    saw_bitvec = False

    for record in records:
        assert required_fields <= set(record)
        assert record["kind"] == "synthesized"
        assert record["arch"] == "avx2"
        assert record["dtype"] == "i64"
        assert record["gadget_depth"] == 1
        assert record["fixture"] == "identity_pairs"
        assert record["input_state"] == expected_input_state
        assert record["target_pairs"] == expected_target_pairs
        assert set(record["output_state"]) == {"top", "bottom"}
        assert len(record["output_state"]["top"]) == 4
        assert len(record["output_state"]["bottom"]) == 4
        assert not _OBJECT_AT_RE.search(json.dumps(record, sort_keys=True))

        for instruction_list_name in ["top_instructions", "bottom_instructions"]:
            instruction_list = record[instruction_list_name]
            assert isinstance(instruction_list, list)
            for instruction in instruction_list:
                assert set(instruction) == {"name", "args"}
                assert isinstance(instruction["name"], str)
                assert isinstance(instruction["args"], dict)
                for value in instruction["args"].values():
                    saw_bitvec = _assert_allowed_concrete_arg(value) or saw_bitvec

    assert saw_bitvec


def test_python_and_rust_template_dumps_match_depth1(tmp_path: Path) -> None:
    cases = [
        ("AVX2", "i64", "avx2", "i64"),
        ("AVX2", "i32", "avx2", "i32"),
        ("AVX512", "i64", "avx512", "i64"),
        ("AVX512", "i32", "avx512", "i32"),
    ]

    for vector_machine, datatype, rust_arch, rust_dtype in cases:
        case_dir = tmp_path / f"{rust_arch}-{rust_dtype}"
        case_dir.mkdir()
        python_output = case_dir / "python-templates.jsonl"
        rust_output = case_dir / "rust-templates.jsonl"

        subprocess.run(
            [
                sys.executable,
                str(_DUMPER),
                "templates",
                "--vector-machine",
                vector_machine,
                "--datatype",
                datatype,
                "--gadget-depth",
                "1",
                "--output",
                str(python_output),
            ],
            cwd=_REPO_ROOT,
            check=True,
        )
        subprocess.run(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "gadget_synth",
                "--bin",
                "dump_gadget_synth",
                "--",
                "templates",
                "--arch",
                rust_arch,
                "--dtype",
                rust_dtype,
                "--gadget-depth",
                "1",
                "--output",
                str(rust_output),
            ],
            cwd=_REPO_ROOT,
            check=True,
        )

        result = _run_comparer(python_output, rust_output, "--max-diffs", "5")
        assert result.returncode == 0, (
            f"template dumps differ for {rust_arch}/{rust_dtype}\n"
            f"{_combined_output(result)}"
        )


@pytest.mark.parametrize(
    ("vector_machine", "datatype", "rust_arch", "rust_dtype"),
    [
        ("AVX2", "i64", "avx2", "i64"),
        ("AVX2", "i32", "avx2", "i32"),
    ],
)
def test_python_and_rust_template_dumps_match_avx2_depth2_non_shared(
    tmp_path: Path,
    vector_machine: str,
    datatype: str,
    rust_arch: str,
    rust_dtype: str,
) -> None:
    python_output = tmp_path / "python-templates.jsonl"
    rust_output = tmp_path / "rust-templates.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(_DUMPER),
            "templates",
            "--vector-machine",
            vector_machine,
            "--datatype",
            datatype,
            "--gadget-depth",
            "2",
            "--exclude-shared-prefix",
            "--output",
            str(python_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "gadget_synth",
            "--bin",
            "dump_gadget_synth",
            "--",
            "templates",
            "--arch",
            rust_arch,
            "--dtype",
            rust_dtype,
            "--gadget-depth",
            "2",
            "--exclude-shared-prefix",
            "--output",
            str(rust_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )

    result = _run_comparer(python_output, rust_output, "--max-diffs", "10")
    assert result.returncode == 0, (
        f"template dumps differ for {rust_arch}/{rust_dtype} "
        "depth=2 excluding shared-prefix\n"
        f"{_combined_output(result)}"
    )


def test_python_and_rust_template_dumps_match_avx2_i64_depth2_shared_prefix(
    tmp_path: Path,
) -> None:
    python_output = tmp_path / "python-templates.jsonl"
    rust_output = tmp_path / "rust-templates.jsonl"
    rust_excluding_shared_output = tmp_path / "rust-templates-non-shared.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(_DUMPER),
            "templates",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--gadget-depth",
            "2",
            "--output",
            str(python_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "gadget_synth",
            "--bin",
            "dump_gadget_synth",
            "--",
            "templates",
            "--arch",
            "avx2",
            "--dtype",
            "i64",
            "--gadget-depth",
            "2",
            "--output",
            str(rust_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "gadget_synth",
            "--bin",
            "dump_gadget_synth",
            "--",
            "templates",
            "--arch",
            "avx2",
            "--dtype",
            "i64",
            "--gadget-depth",
            "2",
            "--exclude-shared-prefix",
            "--output",
            str(rust_excluding_shared_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )

    result = _run_comparer(python_output, rust_output, "--max-diffs", "10")
    assert result.returncode == 0, (
        "template dumps differ for avx2/i64 depth=2 with shared-prefix\n"
        f"{_combined_output(result)}"
    )
    assert len(_read_jsonl(rust_output)) > len(
        _read_jsonl(rust_excluding_shared_output)
    )


@pytest.mark.parametrize("datatype", ["i64", "i32"])
@pytest.mark.parametrize("gadget_depth", [1, 2])
def test_python_and_rust_template_dumps_match_avx512_unmasked(
    tmp_path: Path,
    datatype: str,
    gadget_depth: int,
) -> None:
    python_output = tmp_path / "python-templates.jsonl"
    rust_output = tmp_path / "rust-templates.jsonl"
    intrinsic_filter = _AVX512_UNMASKED_FILTERS[datatype]
    python_args = [
        "--vector-machine",
        "AVX512",
        "--datatype",
        datatype,
        "--gadget-depth",
        str(gadget_depth),
        "--intrinsic-filter",
        intrinsic_filter,
    ]
    rust_args = [
        "--arch",
        "avx512",
        "--dtype",
        datatype,
        "--gadget-depth",
        str(gadget_depth),
        "--intrinsic-filter",
        intrinsic_filter,
    ]
    if gadget_depth == 2:
        python_args.append("--exclude-shared-prefix")
        rust_args.append("--exclude-shared-prefix")

    _dump_python("templates", python_output, *python_args)
    _dump_rust("templates", rust_output, *rust_args)

    python_records = _read_jsonl(python_output)
    rust_records = _read_jsonl(rust_output)
    assert python_records
    assert rust_records
    _assert_records_exclude_masked_intrinsics(python_records)
    _assert_records_exclude_masked_intrinsics(rust_records)

    result = _run_comparer(python_output, rust_output, "--max-diffs", "10")
    assert result.returncode == 0, (
        f"template dumps differ for avx512/{datatype} depth={gadget_depth} "
        f"with unmasked --intrinsic-filter={intrinsic_filter}\n"
        f"{_combined_output(result)}"
    )


@pytest.mark.parametrize("datatype", ["i64", "i32"])
def test_python_and_rust_template_dumps_match_avx512_masked_depth1(
    tmp_path: Path,
    datatype: str,
) -> None:
    python_output = tmp_path / "python-templates.jsonl"
    rust_output = tmp_path / "rust-templates.jsonl"
    intrinsic_filter = _AVX512_MASKED_FILTERS[datatype]
    common_depth_args = [
        "--gadget-depth",
        "1",
        "--intrinsic-filter",
        intrinsic_filter,
    ]

    _dump_python(
        "templates",
        python_output,
        "--vector-machine",
        "AVX512",
        "--datatype",
        datatype,
        *common_depth_args,
    )
    _dump_rust(
        "templates",
        rust_output,
        "--arch",
        "avx512",
        "--dtype",
        datatype,
        *common_depth_args,
    )

    allowed = set(_AVX512_MASKED_INTRINSICS[datatype])
    python_records = _read_jsonl(python_output)
    rust_records = _read_jsonl(rust_output)
    assert python_records
    assert rust_records
    _assert_records_use_only_intrinsics(python_records, allowed)
    _assert_records_use_only_intrinsics(rust_records, allowed)
    _assert_records_include_masked_intrinsics(python_records)
    _assert_records_include_masked_intrinsics(rust_records)

    result = _run_comparer(python_output, rust_output, "--max-diffs", "10")
    assert result.returncode == 0, (
        f"template dumps differ for avx512/{datatype} depth=1 "
        f"with masked --intrinsic-filter={intrinsic_filter}\n"
        f"{_combined_output(result)}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("datatype", "intrinsic_filter", "expected_intrinsics"),
    [
        pytest.param(
            "i64",
            "_mm512_permute_pd,_mm512_permutevar_pd,"
            "_mm512_permutexvar_epi64,_mm512_shuffle_pd",
            (
                "_mm512_permute_pd",
                "_mm512_permutevar_pd",
                "_mm512_permutexvar_epi64",
                "_mm512_shuffle_pd",
            ),
            id="i64-control-and-shuffle",
        ),
        pytest.param(
            "i64",
            "_mm512_permutex2var_epi64,_mm512_alignr_epi64,"
            "_mm512_unpacklo_epi64,_mm512_unpackhi_epi64",
            ("_mm512_permutex2var_epi64", "_mm512_alignr_epi64"),
            id="i64-two-source-avx512",
        ),
        pytest.param(
            "i32",
            "_mm512_permutexvar_epi32,_mm512_shuffle_i32x4",
            ("_mm512_permutexvar_epi32", "_mm512_shuffle_i32x4"),
            id="i32-avx512-specific",
        ),
    ],
)
def test_python_and_rust_synthesized_dumps_match_avx512_depth1_unmasked_filtered(
    tmp_path: Path,
    datatype: str,
    intrinsic_filter: str,
    expected_intrinsics: tuple[str, ...],
) -> None:
    """Representative AVX512 unmasked synthesized parity.

    Full AVX512 i32 unmasked depth-1 parity is deferred because the Python
    reference dump exceeded 120s locally. These filters explicitly exclude
    masked intrinsics on both Python and Rust dumpers while exercising i64 and
    i32 AVX512-specific dispatch.
    """
    python_output = tmp_path / "python-synthesized.jsonl"
    rust_output = tmp_path / "rust-synthesized.jsonl"
    common_args = [
        "--gadget-depth",
        "1",
        "--fixture",
        "identity_pairs",
        "--max-unique-outputs",
        "1",
        "--intrinsic-filter",
        intrinsic_filter,
    ]

    _dump_python(
        "synthesized",
        python_output,
        "--vector-machine",
        "AVX512",
        "--datatype",
        datatype,
        *common_args,
    )
    _dump_rust(
        "synthesized",
        rust_output,
        "--arch",
        "avx512",
        "--dtype",
        datatype,
        *common_args,
    )

    python_records = _read_jsonl(python_output)
    rust_records = _read_jsonl(rust_output)
    assert python_records
    assert rust_records
    _assert_records_exclude_masked_intrinsics(python_records)
    _assert_records_exclude_masked_intrinsics(rust_records)
    for intrinsic_name in expected_intrinsics:
        _assert_saw_synthesized_instruction(python_records, intrinsic_name)

    result = _run_comparer(python_output, rust_output, "--max-diffs", "10")
    assert result.returncode == 0, (
        f"synthesized dumps differ for avx512/{datatype} depth=1 "
        f"with unmasked --intrinsic-filter={intrinsic_filter}\n"
        f"{_combined_output(result)}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("datatype", "intrinsic_filter", "expected_intrinsics"),
    [
        pytest.param(
            "i64",
            "_mm512_mask_permute_pd,_mm512_mask_permutevar_pd,"
            "_mm512_mask_permutexvar_epi64,_mm512_mask_shuffle_pd",
            (
                "_mm512_mask_permute_pd",
                "_mm512_mask_permutevar_pd",
                "_mm512_mask_permutexvar_epi64",
                "_mm512_mask_shuffle_pd",
            ),
            id="i64-masked-control-and-shuffle",
        ),
        pytest.param(
            "i64",
            "_mm512_mask_permutex2var_epi64,_mm512_mask_alignr_epi64,"
            "_mm512_mask_unpacklo_epi64,_mm512_mask_unpackhi_epi64,"
            "_mm512_mask_shuffle_i32x4",
            (
                "_mm512_mask_permutex2var_epi64",
                "_mm512_mask_alignr_epi64",
                "_mm512_mask_unpacklo_epi64",
                "_mm512_mask_unpackhi_epi64",
                "_mm512_mask_shuffle_i32x4",
            ),
            id="i64-masked-two-source-avx512",
        ),
        pytest.param(
            "i32",
            "_mm512_mask_permute_ps,_mm512_mask_permutevar_ps,"
            "_mm512_mask_permutexvar_epi32,_mm512_mask_shuffle_ps",
            (
                "_mm512_mask_permute_ps",
                "_mm512_mask_permutevar_ps",
                "_mm512_mask_permutexvar_epi32",
                "_mm512_mask_shuffle_ps",
            ),
            id="i32-masked-control-and-shuffle",
        ),
        pytest.param(
            "i32",
            "_mm512_mask_permutex2var_epi32,_mm512_mask_alignr_epi32,"
            "_mm512_mask_unpacklo_epi32,_mm512_mask_unpackhi_epi32,"
            "_mm512_mask_shuffle_i32x4",
            (
                "_mm512_mask_permutex2var_epi32",
                "_mm512_mask_alignr_epi32",
                "_mm512_mask_unpacklo_epi32",
                "_mm512_mask_unpackhi_epi32",
                "_mm512_mask_shuffle_i32x4",
            ),
            id="i32-masked-two-source-avx512",
        ),
    ],
)
def test_python_and_rust_synthesized_dumps_match_avx512_depth1_masked_filtered(
    tmp_path: Path,
    datatype: str,
    intrinsic_filter: str,
    expected_intrinsics: tuple[str, ...],
) -> None:
    """Representative AVX512 masked synthesized parity.

    Full unfiltered AVX512 masked depth-1 parity is deferred because the
    candidate menu is expensive, especially for i32. These filters are exact
    masked-intrinsic allowlists passed to both dumpers, so failures localize to
    masked dispatch/semantics and canonicalized concrete arguments.
    """
    python_output = tmp_path / "python-synthesized-masked.jsonl"
    rust_output = tmp_path / "rust-synthesized-masked.jsonl"
    common_args = [
        "--gadget-depth",
        "1",
        "--fixture",
        "identity_pairs",
        "--max-unique-outputs",
        "1",
        "--intrinsic-filter",
        intrinsic_filter,
    ]

    _dump_python(
        "synthesized",
        python_output,
        "--vector-machine",
        "AVX512",
        "--datatype",
        datatype,
        *common_args,
    )
    _dump_rust(
        "synthesized",
        rust_output,
        "--arch",
        "avx512",
        "--dtype",
        datatype,
        *common_args,
    )

    allowed = set(intrinsic_filter.split(","))
    python_records = _read_jsonl(python_output)
    rust_records = _read_jsonl(rust_output)
    assert python_records
    assert rust_records
    _assert_records_use_only_intrinsics(python_records, allowed)
    _assert_records_use_only_intrinsics(rust_records, allowed)
    _assert_records_include_masked_intrinsics(python_records)
    _assert_records_include_masked_intrinsics(rust_records)
    for intrinsic_name in expected_intrinsics:
        _assert_saw_synthesized_instruction(python_records, intrinsic_name)

    result = _run_comparer(python_output, rust_output, "--max-diffs", "10")
    assert result.returncode == 0, (
        f"synthesized dumps differ for avx512/{datatype} depth=1 "
        f"with masked --intrinsic-filter={intrinsic_filter}\n"
        f"{_combined_output(result)}"
    )


@pytest.mark.slow
def test_python_and_rust_synthesized_dumps_match_avx512_i64_depth2_non_shared_filtered(
    tmp_path: Path,
) -> None:
    """Representative AVX512 depth-2 parity for single→dual mux paths.

    Full AVX512 unmasked depth-2 synthesized parity is deferred because it is
    too expensive for routine tests. This filtered case explicitly excludes
    masked intrinsics and keeps a runnable depth-2 ratchet.
    """
    intrinsic_filter = "_mm512_permute_pd,_mm512_shuffle_i32x4"
    python_output = tmp_path / "python-synthesized-depth2.jsonl"
    rust_output = tmp_path / "rust-synthesized-depth2.jsonl"
    common_args = [
        "--gadget-depth",
        "2",
        "--exclude-shared-prefix",
        "--fixture",
        "identity_pairs",
        "--max-unique-outputs",
        "1",
        "--intrinsic-filter",
        intrinsic_filter,
    ]

    _dump_python(
        "synthesized",
        python_output,
        "--vector-machine",
        "AVX512",
        "--datatype",
        "i64",
        *common_args,
    )
    _dump_rust(
        "synthesized",
        rust_output,
        "--arch",
        "avx512",
        "--dtype",
        "i64",
        *common_args,
    )

    python_records = _read_jsonl(python_output)
    rust_records = _read_jsonl(rust_output)
    assert python_records
    assert rust_records
    _assert_records_exclude_masked_intrinsics(python_records)
    _assert_records_exclude_masked_intrinsics(rust_records)
    _assert_saw_single_to_dual_mux_record(
        python_records,
        single_intrinsic="_mm512_permute_pd",
        dual_intrinsic="_mm512_shuffle_i32x4",
    )

    result = _run_comparer(python_output, rust_output, "--max-diffs", "10")
    assert result.returncode == 0, (
        "synthesized dumps differ for avx512/i64 depth=2 identity_pairs "
        "excluding shared-prefix with unmasked "
        f"--intrinsic-filter={intrinsic_filter}\n{_combined_output(result)}"
    )


def test_python_and_rust_synthesized_dumps_match_avx2_i64_depth1(
    tmp_path: Path,
) -> None:
    python_output = tmp_path / "python-synthesized.jsonl"
    rust_output = tmp_path / "rust-synthesized.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(_DUMPER),
            "synthesized",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--gadget-depth",
            "1",
            "--fixture",
            "identity_pairs",
            "--max-unique-outputs",
            "3",
            "--output",
            str(python_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "gadget_synth",
            "--bin",
            "dump_gadget_synth",
            "--",
            "synthesized",
            "--arch",
            "avx2",
            "--dtype",
            "i64",
            "--gadget-depth",
            "1",
            "--fixture",
            "identity_pairs",
            "--max-unique-outputs",
            "3",
            "--output",
            str(rust_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )

    result = _run_comparer(python_output, rust_output, "--max-diffs", "10")
    assert result.returncode == 0, (
        "synthesized dumps differ for avx2/i64 depth=1 identity_pairs\n"
        f"{_combined_output(result)}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("intrinsic_filter", "single_intrinsic"),
    [
        pytest.param(
            "_mm256_permute_pd,_mm256_shuffle_pd",
            "_mm256_permute_pd",
            id="permute_pd-shuffle_pd",
        ),
        pytest.param(
            "_mm256_permutexvar_epi64,_mm256_shuffle_pd",
            "_mm256_permutexvar_epi64",
            id="permutexvar_epi64-shuffle_pd",
        ),
    ],
)
def test_python_and_rust_synthesized_dumps_match_avx2_i64_depth2_non_shared_filtered_mux(
    tmp_path: Path,
    intrinsic_filter: str,
    single_intrinsic: str,
) -> None:
    """Representative Task 11 depth-2 parity for mux/K-per-wiring paths.

    Full unfiltered AVX2 i64 depth-2 synthesized parity is deferred because the
    Python reference dump currently times out. This filtered ratchet keeps the
    non-shared depth-2 single→dual mux path runnable.

    Run explicitly with:

    uv run pytest tests/test_gadget_synth_dump.py -q -m slow \
      -k synthesized_dumps_match_avx2_i64_depth2_non_shared_filtered_mux
    """
    python_output = tmp_path / "python-synthesized-depth2.jsonl"
    rust_output = tmp_path / "rust-synthesized-depth2.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(_DUMPER),
            "synthesized",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--gadget-depth",
            "2",
            "--exclude-shared-prefix",
            "--fixture",
            "identity_pairs",
            "--max-unique-outputs",
            "3",
            "--intrinsic-filter",
            intrinsic_filter,
            "--output",
            str(python_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "gadget_synth",
            "--bin",
            "dump_gadget_synth",
            "--",
            "synthesized",
            "--arch",
            "avx2",
            "--dtype",
            "i64",
            "--gadget-depth",
            "2",
            "--exclude-shared-prefix",
            "--fixture",
            "identity_pairs",
            "--max-unique-outputs",
            "3",
            "--intrinsic-filter",
            intrinsic_filter,
            "--output",
            str(rust_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )

    _assert_saw_single_to_dual_mux_record(
        _read_jsonl(python_output),
        single_intrinsic=single_intrinsic,
        dual_intrinsic="_mm256_shuffle_pd",
    )

    result = _run_comparer(python_output, rust_output, "--max-diffs", "10")
    assert result.returncode == 0, (
        "synthesized dumps differ for avx2/i64 depth=2 identity_pairs "
        "excluding shared-prefix with "
        f"--intrinsic-filter={intrinsic_filter}\n{_combined_output(result)}"
    )


@pytest.mark.slow
def test_python_and_rust_synthesized_dumps_match_avx2_i64_depth2_shared_prefix_filtered(
    tmp_path: Path,
) -> None:
    """Representative shared-prefix depth-2 parity.

    Full unfiltered AVX2 i64 depth-2 synthesized parity remains deferred because
    the Python reference dump is too slow. This filtered case keeps a runnable
    ratchet for shared-prefix synthesis.

    Run explicitly with:

    uv run pytest tests/test_gadget_synth_dump.py -q -m slow \
      -k synthesized_dumps_match_avx2_i64_depth2_shared_prefix_filtered
    """
    intrinsic_filter = "_mm256_permute_pd,_mm256_permute2x128_si256"
    python_output = tmp_path / "python-synthesized-depth2-shared.jsonl"
    rust_output = tmp_path / "rust-synthesized-depth2-shared.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(_DUMPER),
            "synthesized",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i64",
            "--gadget-depth",
            "2",
            "--fixture",
            "identity_pairs",
            "--max-unique-outputs",
            "1",
            "--intrinsic-filter",
            intrinsic_filter,
            "--output",
            str(python_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "gadget_synth",
            "--bin",
            "dump_gadget_synth",
            "--",
            "synthesized",
            "--arch",
            "avx2",
            "--dtype",
            "i64",
            "--gadget-depth",
            "2",
            "--fixture",
            "identity_pairs",
            "--max-unique-outputs",
            "1",
            "--intrinsic-filter",
            intrinsic_filter,
            "--output",
            str(rust_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )

    _assert_saw_shared_prefix_record(
        _read_jsonl(python_output),
        prefix_intrinsic="_mm256_permute_pd",
        tail_intrinsic="_mm256_permute2x128_si256",
    )

    result = _run_comparer(python_output, rust_output, "--max-diffs", "10")
    assert result.returncode == 0, (
        "synthesized dumps differ for avx2/i64 depth=2 identity_pairs "
        f"with shared-prefix and --intrinsic-filter={intrinsic_filter}\n"
        f"{_combined_output(result)}"
    )


@pytest.mark.slow
def test_python_and_rust_synthesized_dumps_match_avx2_i32_depth1(
    tmp_path: Path,
) -> None:
    python_output = tmp_path / "python-synthesized.jsonl"
    rust_output = tmp_path / "rust-synthesized.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(_DUMPER),
            "synthesized",
            "--vector-machine",
            "AVX2",
            "--datatype",
            "i32",
            "--gadget-depth",
            "1",
            "--fixture",
            "identity_pairs",
            "--max-unique-outputs",
            "3",
            "--output",
            str(python_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "gadget_synth",
            "--bin",
            "dump_gadget_synth",
            "--",
            "synthesized",
            "--arch",
            "avx2",
            "--dtype",
            "i32",
            "--gadget-depth",
            "1",
            "--fixture",
            "identity_pairs",
            "--max-unique-outputs",
            "3",
            "--output",
            str(rust_output),
        ],
        cwd=_REPO_ROOT,
        check=True,
    )

    result = _run_comparer(python_output, rust_output, "--max-diffs", "10")
    assert result.returncode == 0, (
        "synthesized dumps differ for avx2/i32 depth=1 identity_pairs\n"
        f"{_combined_output(result)}"
    )


def test_template_normalization_preserves_symbolic_aliasing_and_mux_pruning() -> None:
    from tools.dump_gadget_synth import normalize_graph

    shared = Symbolic("imm8_shared_raw_name", 8)
    independent = Symbolic("imm8_independent_raw_name", 8)
    mux_select = Symbolic("sel_a_intrinsic_123456", 2)
    graph = GadgetGraph(
        top=IntrinsicNode(
            "demo_top",
            {
                "z": shared,
                "a": Mux(
                    select=mux_select,
                    sources=(InputRef("top"), InputRef("bottom")),
                    pruning="force_last",
                ),
                "b": shared,
                "c": independent,
            },
            isomorphic_order=False,
        ),
        bottom=None,
    )

    normalized = normalize_graph(graph)
    top = normalized["top"]
    operands = top["operands"]

    assert list(operands) == ["a", "b", "c", "z"]
    assert top["isomorphic_order"] is False
    assert normalized["bottom"] is None

    assert operands["b"] == operands["z"]
    assert operands["b"]["role"] == "imm8"
    assert operands["b"]["bits"] == 8
    assert operands["c"]["var_id"] != operands["b"]["var_id"]

    mux = operands["a"]
    assert mux["kind"] == "mux"
    assert mux["pruning"] == "force_last"
    assert mux["select"]["role"] == "mux_select"
    assert mux["select"]["bits"] == 2
    assert mux["sources"] == [
        {"kind": "input", "name": "top"},
        {"kind": "input", "name": "bottom"},
    ]


def _template_record(template_id: str, intrinsic_name: str) -> dict[str, Any]:
    return {
        "kind": "template",
        "arch": "avx2",
        "dtype": "i64",
        "gadget_depth": 1,
        "template_id": template_id,
        "graph": {
            "top": {
                "kind": "intrinsic",
                "name": intrinsic_name,
                "operands": {},
                "isomorphic_order": True,
            },
            "bottom": None,
        },
    }


def test_compare_dumps_equal_files_return_exit_zero(tmp_path: Path) -> None:
    expected = tmp_path / "expected.jsonl"
    actual = tmp_path / "actual.jsonl"
    expected.write_text('{"b": 2, "a": 1}\n')
    actual.write_text('{"a": 1, "b": 2}\n')

    result = _run_comparer(expected, actual)

    assert result.returncode == 0, _combined_output(result)


def test_compare_dumps_missing_records_return_nonzero(tmp_path: Path) -> None:
    expected = tmp_path / "expected.jsonl"
    actual = tmp_path / "actual.jsonl"
    _write_jsonl(
        expected,
        [
            _template_record("first", "demo_a"),
            _template_record("second", "demo_b"),
            _template_record("third", "demo_c"),
        ],
    )
    _write_jsonl(actual, [_template_record("first", "demo_a")])

    result = _run_comparer(expected, actual, "--max-diffs", "1")
    output = _combined_output(result)

    assert result.returncode != 0
    assert "Missing records" in output
    assert "template_id=second" in output
    assert "template_id=third" not in output
    assert "1 more" in output


def test_compare_dumps_extra_records_return_nonzero(tmp_path: Path) -> None:
    expected = tmp_path / "expected.jsonl"
    actual = tmp_path / "actual.jsonl"
    _write_jsonl(expected, [_template_record("first", "demo_a")])
    _write_jsonl(
        actual,
        [
            _template_record("first", "demo_a"),
            _template_record("second", "demo_b"),
        ],
    )

    result = _run_comparer(expected, actual)
    output = _combined_output(result)

    assert result.returncode != 0
    assert "Extra records" in output
    assert "template_id=second" in output


def test_compare_dumps_changed_records_return_nonzero_with_key(tmp_path: Path) -> None:
    expected = tmp_path / "expected.jsonl"
    actual = tmp_path / "actual.jsonl"
    _write_jsonl(expected, [_template_record("shared", "demo_a")])
    _write_jsonl(actual, [_template_record("shared", "demo_b")])

    result = _run_comparer(expected, actual)
    output = _combined_output(result)

    assert result.returncode != 0
    assert "Different records" in output
    assert "template_id=shared" in output
    assert "demo_a" in output
    assert "demo_b" in output


def test_compare_dumps_changed_records_show_long_prefix_difference(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected.jsonl"
    actual = tmp_path / "actual.jsonl"
    expected_record = _template_record("shared-long", "demo")
    actual_record = _template_record("shared-long", "demo")
    expected_record["payload"] = f"{'x' * 320}expected-tail"
    actual_record["payload"] = f"{'x' * 320}actual-tail"
    _write_jsonl(expected, [expected_record])
    _write_jsonl(actual, [actual_record])

    result = _run_comparer(expected, actual)
    output = _combined_output(result)

    assert result.returncode != 0
    assert "Different records" in output
    assert "template_id=shared-long" in output
    assert "first_diff_index=" in output
    assert "expected-tail" in output
    assert "actual-tail" in output


def test_compare_dumps_duplicate_canonical_records_reported(tmp_path: Path) -> None:
    expected = tmp_path / "expected.jsonl"
    actual = tmp_path / "actual.jsonl"
    record = _template_record("duplicate", "demo")
    _write_jsonl(expected, [record, record])
    _write_jsonl(actual, [record, record])

    result = _run_comparer(expected, actual)
    output = _combined_output(result)

    assert result.returncode == 0, output
    assert "Duplicate canonical records" in output
    assert "expected count=2" in output
    assert "actual count=2" in output
