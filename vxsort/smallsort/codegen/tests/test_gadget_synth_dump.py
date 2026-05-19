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


def test_python_dumper_rejects_synthesized_mode(tmp_path: Path) -> None:
    output = tmp_path / "synthesized.jsonl"

    result = subprocess.run(
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
            "--output",
            str(output),
        ],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "templates" in result.stderr
    assert not output.exists()


def test_rust_dumper_rejects_synthesized_mode(tmp_path: Path) -> None:
    output = tmp_path / "synthesized.jsonl"

    result = subprocess.run(
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
            "--output",
            str(output),
        ],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "templates" in result.stderr
    assert not output.exists()


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
