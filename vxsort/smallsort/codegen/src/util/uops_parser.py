"""uops.info parser facade for compact JSON databases only."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from .uops_parser_json import (
        list_available_architectures as _list_available_architectures_json,
    )
    from .uops_parser_json import parse_uops_json as _parse_uops_json
except ImportError:
    from util.uops_parser_json import (  # type: ignore[no-redef]
        list_available_architectures as _list_available_architectures_json,
    )
    from util.uops_parser_json import parse_uops_json as _parse_uops_json  # type: ignore[no-redef]

if TYPE_CHECKING:
    from cost_model import InstructionCost

_UOPS_DB_PATH_ENV = "VXSORT_UOPS_DB_PATH"
_UOPS_DB_FORMAT_ENV = "VXSORT_UOPS_DB_FORMAT"


def find_uops_database_path(
    base_dir: str | Path | None = None,
    *,
    prefer: str = "auto",
) -> str | None:
    """Find compact JSON uops database path.

    Resolution order:
    1. ``VXSORT_UOPS_DB_PATH`` if set and file exists.
    2. ``prefer`` (or ``VXSORT_UOPS_DB_FORMAT`` override):
       - ``json``/``auto``: instructions.json.zst, then instructions.json
    """
    env_path = os.getenv(_UOPS_DB_PATH_ENV)
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return str(candidate)

    env_prefer = os.getenv(_UOPS_DB_FORMAT_ENV)
    selected = (env_prefer or prefer).strip().lower()
    if selected not in {"auto", "json"}:
        selected = "auto"

    if base_dir is None:
        # src/util/uops_parser.py -> project root at parents[2]
        base_path = Path(__file__).resolve().parents[2]
    else:
        base_path = Path(base_dir)

    candidates = [base_path / "instructions.json.zst", base_path / "instructions.json"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def _is_json_database_path(path: Path) -> bool:
    return path.suffix == ".json" or path.name.endswith(".json.zst")


def parse_uops_json(
    json_path: str | Path,
    target_arch: str,
) -> dict[str, InstructionCost]:
    """Parse compact JSON uops database for a target architecture."""
    return _parse_uops_json(json_path, target_arch)


def parse_uops_database(
    path: str | Path,
    target_arch: str,
) -> dict[str, InstructionCost]:
    """Parse compact JSON(.zst) uops database."""
    path = Path(path)
    if not _is_json_database_path(path):
        raise ValueError(
            f"Unsupported uops database format: {path}. "
            "Only .json/.json.zst are supported."
        )
    return parse_uops_json(path, target_arch)


def list_available_architectures(path: str | Path) -> list[str]:
    """List architecture names from compact JSON(.zst) uops database."""
    path = Path(path)
    if not _is_json_database_path(path):
        raise ValueError(
            f"Unsupported uops database format: {path}. "
            "Only .json/.json.zst are supported."
        )
    return _list_available_architectures_json(path)
