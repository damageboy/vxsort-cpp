#!/usr/bin/env python3
"""Compare normalized gadget synthesizer JSONL dumps.

The comparer treats each JSONL line as one canonical record and compares the two
inputs as multisets.  It reports stable, human-oriented keys for actionable
parity diffs while using sorted JSON serialization for exact equality.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParsedRecord:
    """One parsed JSONL record and its comparison metadata."""

    path: Path
    line_number: int
    record: Any
    canonical: str
    display_key: str


@dataclass
class DiffEntry:
    """A grouped diff entry for one canonical record."""

    display_key: str
    canonical: str
    count: int


@dataclass(frozen=True)
class ChangedEntry:
    """A grouped diff entry for records sharing a display key but differing."""

    display_key: str
    expected_canonical: str
    actual_canonical: str
    count: int


class JsonlInputError(Exception):
    """Raised when an input file is not well-formed JSONL."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _parse_jsonl(path: Path) -> list[ParsedRecord]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise JsonlInputError(f"{path}: unable to read file: {exc}") from exc

    records: list[ParsedRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise JsonlInputError(
                f"{path}:{line_number}: blank lines are not valid JSONL records"
            )
        try:
            record = json.loads(line, parse_constant=_reject_json_constant)
            canonical = _canonical_json(record)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise JsonlInputError(
                f"{path}:{line_number}: malformed JSON: {exc}"
            ) from exc
        records.append(
            ParsedRecord(
                path=path,
                line_number=line_number,
                record=record,
                canonical=canonical,
                display_key=display_key(record),
            )
        )
    return records


def display_key(record: Any) -> str:
    """Return a stable, concise display key for one dump record."""
    if not isinstance(record, dict):
        return f"record_sha={_short_hash(_canonical_json(record))}"

    parts: list[str] = []
    for field in ("kind", "arch", "dtype", "gadget_depth", "fixture"):
        if field in record:
            parts.append(f"{field}={_format_key_value(record[field])}")

    if "template_id" in record:
        parts.append(f"template_id={_format_key_value(record['template_id'])}")
    elif "graph_key" in record:
        parts.append(f"graph_key={_format_key_value(record['graph_key'])}")
    elif record.get("kind") == "template" and "graph" in record:
        graph = record["graph"]
        graph_names = _graph_intrinsic_names(graph)
        if graph_names:
            parts.append(f"graph_instrs={','.join(graph_names)}")
        parts.append(f"graph_sha={_short_hash(_canonical_json(graph))}")

    if record.get("kind") == "synthesized":
        if "output_state" in record:
            parts.append(f"output_state={_compact_json(record['output_state'])}")
        instruction_key = _instruction_key(record)
        if instruction_key:
            parts.append(f"instructions={instruction_key}")

    if not parts:
        parts.append(f"record_sha={_short_hash(_canonical_json(record))}")
    return " ".join(parts)


def _format_key_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return str(value)
    return _compact_json(value)


def _compact_json(value: Any) -> str:
    return _canonical_json(value)


def _short_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:12]


def _graph_intrinsic_names(value: Any) -> list[str]:
    names: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("kind") == "intrinsic" and "name" in node:
                names.append(str(node["name"]))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return sorted(set(names))


def _instruction_key(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in ("top_instructions", "bottom_instructions", "instructions"):
        value = record.get(field)
        if not isinstance(value, list) or not value:
            continue
        names = []
        for instruction in value:
            if isinstance(instruction, dict):
                names.append(str(instruction.get("name", "<unnamed>")))
            else:
                names.append(type(instruction).__name__)
        parts.append(f"{field}=[{','.join(names)}]")
    return ";".join(parts)


def _count_records(records: Iterable[ParsedRecord]) -> Counter[str]:
    return Counter(record.canonical for record in records)


def _records_by_canonical(records: Iterable[ParsedRecord]) -> dict[str, ParsedRecord]:
    by_canonical: dict[str, ParsedRecord] = {}
    for record in records:
        by_canonical.setdefault(record.canonical, record)
    return by_canonical


def _diff_entries(
    counts: Counter[str],
    examples: dict[str, ParsedRecord],
) -> list[DiffEntry]:
    return sorted(
        (
            DiffEntry(
                display_key=examples[canonical].display_key,
                canonical=canonical,
                count=count,
            )
            for canonical, count in counts.items()
            if count > 0
        ),
        key=lambda entry: (entry.display_key, entry.canonical),
    )


def _separate_changed_entries(
    missing: list[DiffEntry],
    extra: list[DiffEntry],
) -> tuple[list[ChangedEntry], list[DiffEntry], list[DiffEntry]]:
    """Pair records with the same display key as changed entries."""
    missing_by_key = _group_by_display_key(missing)
    extra_by_key = _group_by_display_key(extra)
    changed: list[ChangedEntry] = []

    for display_key in sorted(set(missing_by_key) & set(extra_by_key)):
        missing_entries = missing_by_key[display_key]
        extra_entries = extra_by_key[display_key]
        missing_index = 0
        extra_index = 0
        while missing_index < len(missing_entries) and extra_index < len(extra_entries):
            missing_entry = missing_entries[missing_index]
            extra_entry = extra_entries[extra_index]
            count = min(missing_entry.count, extra_entry.count)
            changed.append(
                ChangedEntry(
                    display_key=display_key,
                    expected_canonical=missing_entry.canonical,
                    actual_canonical=extra_entry.canonical,
                    count=count,
                )
            )
            missing_entry.count -= count
            extra_entry.count -= count
            if missing_entry.count == 0:
                missing_index += 1
            if extra_entry.count == 0:
                extra_index += 1

    remaining_missing = [entry for entry in missing if entry.count > 0]
    remaining_extra = [entry for entry in extra if entry.count > 0]
    return changed, remaining_missing, remaining_extra


def _group_by_display_key(entries: Iterable[DiffEntry]) -> dict[str, list[DiffEntry]]:
    groups: dict[str, list[DiffEntry]] = defaultdict(list)
    for entry in entries:
        groups[entry.display_key].append(entry)
    for grouped_entries in groups.values():
        grouped_entries.sort(key=lambda entry: entry.canonical)
    return groups


def _duplicate_entries(records: list[ParsedRecord]) -> list[DiffEntry]:
    counts = _count_records(records)
    examples = _records_by_canonical(records)
    duplicates = Counter(
        {canonical: count for canonical, count in counts.items() if count > 1}
    )
    return _diff_entries(duplicates, examples)


def compare_dumps(
    expected_records: list[ParsedRecord],
    actual_records: list[ParsedRecord],
) -> tuple[list[ChangedEntry], list[DiffEntry], list[DiffEntry]]:
    """Compare two parsed dumps as multisets."""
    expected_counts = _count_records(expected_records)
    actual_counts = _count_records(actual_records)
    expected_examples = _records_by_canonical(expected_records)
    actual_examples = _records_by_canonical(actual_records)

    missing = _diff_entries(expected_counts - actual_counts, expected_examples)
    extra = _diff_entries(actual_counts - expected_counts, actual_examples)
    return _separate_changed_entries(missing, extra)


def _record_total(entries: Iterable[DiffEntry | ChangedEntry]) -> int:
    return sum(entry.count for entry in entries)


def _print_duplicate_report(
    expected_duplicates: list[DiffEntry],
    actual_duplicates: list[DiffEntry],
    *,
    max_diffs: int,
) -> None:
    if not expected_duplicates and not actual_duplicates:
        return

    print("Duplicate canonical records:")
    _print_duplicate_side("expected", expected_duplicates, max_diffs=max_diffs)
    _print_duplicate_side("actual", actual_duplicates, max_diffs=max_diffs)


def _print_duplicate_side(
    label: str,
    entries: list[DiffEntry],
    *,
    max_diffs: int,
) -> None:
    if not entries:
        print(f"  {label}: none")
        return
    for entry in entries[:max_diffs]:
        print(f"  {label} count={entry.count} key={entry.display_key}")
    _print_suppressed(entries, max_diffs, label=f"duplicate {label} records")


def _print_diff_report(
    changed: list[ChangedEntry],
    missing: list[DiffEntry],
    extra: list[DiffEntry],
    *,
    max_diffs: int,
) -> None:
    if changed:
        print(f"Different records: {_record_total(changed)}")
        for entry in changed[:max_diffs]:
            count = _count_suffix(entry.count)
            diff_index = _first_difference_index(
                entry.expected_canonical,
                entry.actual_canonical,
            )
            print(f"  ~ key={entry.display_key}{count}")
            print(f"    first_diff_index={diff_index}")
            print(
                "    expected="
                f"{_first_difference_context(entry.expected_canonical, diff_index)}"
            )
            print(
                "    actual  ="
                f"{_first_difference_context(entry.actual_canonical, diff_index)}"
            )
        _print_suppressed(changed, max_diffs, label="different records")

    if missing:
        print(f"Missing records: {_record_total(missing)}")
        for entry in missing[:max_diffs]:
            count = _count_suffix(entry.count)
            print(f"  - key={entry.display_key}{count}")
            print(f"    record={_preview(entry.canonical)}")
        _print_suppressed(missing, max_diffs, label="missing records")

    if extra:
        print(f"Extra records: {_record_total(extra)}")
        for entry in extra[:max_diffs]:
            count = _count_suffix(entry.count)
            print(f"  + key={entry.display_key}{count}")
            print(f"    record={_preview(entry.canonical)}")
        _print_suppressed(extra, max_diffs, label="extra records")


def _print_suppressed(
    entries: Sequence[DiffEntry | ChangedEntry],
    max_diffs: int,
    *,
    label: str,
) -> None:
    suppressed = _record_total(entries[max_diffs:])
    if suppressed:
        print(f"  ... {suppressed} more {label} not shown (--max-diffs={max_diffs})")


def _count_suffix(count: int) -> str:
    if count == 1:
        return ""
    return f" count={count}"


def _first_difference_index(expected: str, actual: str) -> int:
    for index, (expected_char, actual_char) in enumerate(zip(expected, actual)):
        if expected_char != actual_char:
            return index
    return min(len(expected), len(actual))


def _first_difference_context(
    canonical: str, diff_index: int, radius: int = 120
) -> str:
    start = max(diff_index - radius, 0)
    end = min(diff_index + radius, len(canonical))
    context = canonical[start:end]
    if diff_index >= len(canonical):
        context = f"{context}<END>"
    if start > 0:
        context = f"...{context}"
    if end < len(canonical):
        context = f"{context}..."
    return context


def _preview(canonical: str, limit: int = 240) -> str:
    if len(canonical) <= limit:
        return canonical
    return f"{canonical[: limit - 3]}..."


def _parse_max_diffs(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two gadget synthesizer JSONL dumps as multisets.",
    )
    parser.add_argument("expected", type=Path, help="reference JSONL dump")
    parser.add_argument("actual", type=Path, help="candidate JSONL dump")
    parser.add_argument(
        "--max-diffs",
        type=_parse_max_diffs,
        default=20,
        help="maximum entries to print per diff group (default: 20)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        expected_records = _parse_jsonl(args.expected)
        actual_records = _parse_jsonl(args.actual)
    except JsonlInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    expected_duplicates = _duplicate_entries(expected_records)
    actual_duplicates = _duplicate_entries(actual_records)
    changed, missing, extra = compare_dumps(expected_records, actual_records)

    _print_duplicate_report(
        expected_duplicates,
        actual_duplicates,
        max_diffs=args.max_diffs,
    )

    if changed or missing or extra:
        if expected_duplicates or actual_duplicates:
            print()
        print(f"Dumps differ: expected={args.expected} actual={args.actual}")
        _print_diff_report(changed, missing, extra, max_diffs=args.max_diffs)
        return 1

    print("Dumps match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
