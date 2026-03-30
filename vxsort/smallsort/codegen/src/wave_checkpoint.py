"""Wave checkpoint format: per-stage zstd files + master.json directory layout.

Directory layout::

    checkpoint_dir/
        master.json              # Small, always rewritten
        stage_00.json.zst        # Per-stage synthesis data
        stage_01.json.zst
        ...
        scored_paths.json.zst    # LLVM-MCA results
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass

import zstandard as zstd

try:
    from .bitonic_types import InstructionSpec, PermutationGadget, VectorState
    from .transition_table import TransitionTable
except ImportError:
    from bitonic_types import InstructionSpec, PermutationGadget, VectorState  # type: ignore[no-redef]
    from transition_table import TransitionTable  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ZSTD_LEVEL = 3


# ---------------------------------------------------------------------------
# WaveMasterConfig
# ---------------------------------------------------------------------------


@dataclass
class WaveMasterConfig:
    """Master configuration for a wave-based synthesis run."""

    num_vecs: int
    vm: str
    prim_type: str
    gadget_depth: int
    natural_order: bool
    retroactive_input: bool
    num_stages: int
    wave_count: int
    best_scores: dict  # target_cpu -> {throughput, cycles, path_index}
    stage_stats: dict  # stage_idx (int) -> {attempts, distinct_outputs, success_rate}
    exhausted_stages: list[int]
    target_cpus: list[str]


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _atomic_write(path: str, data: bytes) -> None:
    """Write *data* to *path* atomically via temp file + os.replace()."""
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    closed = False
    try:
        os.write(fd, data)
        os.close(fd)
        closed = True
        os.replace(tmp_path, path)
    except BaseException:
        if not closed:
            os.close(fd)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_instruction(inst: InstructionSpec) -> dict:
    return {"intrinsic_name": inst.intrinsic_name, "args": inst.args}


def _deserialize_instruction(d: dict) -> InstructionSpec:
    return InstructionSpec(intrinsic_name=d["intrinsic_name"], args=d["args"])


def _serialize_gadget(gadget: PermutationGadget) -> dict:
    return {
        "top": [_serialize_instruction(i) for i in gadget.top_instructions],
        "bottom": [_serialize_instruction(i) for i in gadget.bottom_instructions],
    }


def _deserialize_gadget(d: dict) -> PermutationGadget:
    return PermutationGadget(
        top_instructions=[_deserialize_instruction(i) for i in d["top"]],
        bottom_instructions=[_deserialize_instruction(i) for i in d["bottom"]],
        validated=True,
    )


def _encode_transition_key(
    inp: tuple[tuple[int, ...], tuple[int, ...]],
    out: tuple[tuple[int, ...], tuple[int, ...]],
) -> str:
    """Encode (input_tuple, output_tuple) as a JSON-safe string key."""
    return json.dumps([list(inp[0]), list(inp[1]), list(out[0]), list(out[1])])


def _decode_transition_key(
    key: str,
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]
]:
    """Decode a JSON string key back to (input_tuple, output_tuple)."""
    parts = json.loads(key)
    inp = (tuple(parts[0]), tuple(parts[1]))
    out = (tuple(parts[2]), tuple(parts[3]))
    return inp, out


def _serialize_attempted_pairs(
    pairs: set[tuple[tuple[tuple[int, ...], tuple[int, ...]], int]],
) -> list:
    """Serialize attempted_pairs as [[list(inp[0]), list(inp[1]), candidate_idx], ...]."""
    result = []
    for state_tuple, candidate_idx in pairs:
        result.append([list(state_tuple[0]), list(state_tuple[1]), candidate_idx])
    return result


def _deserialize_attempted_pairs(
    data: list,
) -> set[tuple[tuple[tuple[int, ...], tuple[int, ...]], int]]:
    """Deserialize attempted_pairs from list form."""
    result = set()
    for item in data:
        state_tuple = (tuple(item[0]), tuple(item[1]))
        candidate_idx = item[2]
        result.add((state_tuple, candidate_idx))
    return result


def _serialize_forwarded_outputs(
    outputs: set[tuple[tuple[int, ...], tuple[int, ...]]],
) -> list:
    """Serialize forwarded_outputs as [[list(top), list(bottom)], ...]."""
    return [[list(t[0]), list(t[1])] for t in outputs]


def _deserialize_forwarded_outputs(
    data: list,
) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Deserialize forwarded_outputs from list form."""
    return {(tuple(item[0]), tuple(item[1])) for item in data}


# ---------------------------------------------------------------------------
# WaveCheckpoint
# ---------------------------------------------------------------------------


class WaveCheckpoint:
    """Manages a wave checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : str
        Path to the checkpoint directory.  Created if it does not exist.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    # -- paths -------------------------------------------------------------

    def _master_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "master.json")

    def _stage_path(self, stage_idx: int) -> str:
        return os.path.join(self.checkpoint_dir, f"stage_{stage_idx:02d}.json.zst")

    def _scored_paths_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "scored_paths.json.zst")

    # -- exists ------------------------------------------------------------

    def exists(self) -> bool:
        """Return True if a master.json exists in the checkpoint directory."""
        return os.path.exists(self._master_path())

    # -- master config -----------------------------------------------------

    def save_master(self, config: WaveMasterConfig) -> None:
        """Save master config as plain JSON (not compressed)."""
        d = asdict(config)
        # Ensure stage_stats keys are strings for JSON
        d["stage_stats"] = {str(k): v for k, v in d["stage_stats"].items()}
        data = json.dumps(d, indent=2).encode("utf-8")
        _atomic_write(self._master_path(), data)

    def load_master(self) -> WaveMasterConfig:
        """Load master config from master.json.

        Raises FileNotFoundError if master.json does not exist.
        """
        path = self._master_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Master config not found: {path}")
        with open(path, "rb") as f:
            d = json.loads(f.read().decode("utf-8"))
        # Convert stage_stats keys back to int
        d["stage_stats"] = {int(k): v for k, v in d["stage_stats"].items()}
        return WaveMasterConfig(**d)

    # -- stage data --------------------------------------------------------

    def save_stage(self, stage_idx: int, tt: TransitionTable) -> None:
        """Save a single stage's data from the TransitionTable."""
        sd = tt.stages[stage_idx]

        # Serialize transitions
        transitions_data = {}
        for (inp_t, out_t), gadgets in sd.transitions.items():
            key = _encode_transition_key(inp_t, out_t)
            transitions_data[key] = [_serialize_gadget(g) for g in gadgets]

        stage_data = {
            "transitions": transitions_data,
            "attempts": sd.attempts,
            "attempted_pairs": _serialize_attempted_pairs(sd.attempted_pairs),
            "forwarded_outputs": _serialize_forwarded_outputs(sd.forwarded_outputs),
            "consecutive_zero_budgets": sd.consecutive_zero_budgets,
            "unproductive_waves": sd.unproductive_waves,
        }

        raw_json = json.dumps(stage_data).encode("utf-8")
        cctx = zstd.ZstdCompressor(level=_ZSTD_LEVEL)
        compressed = cctx.compress(raw_json)
        _atomic_write(self._stage_path(stage_idx), compressed)

    def load_stage(self, stage_idx: int, tt: TransitionTable) -> None:
        """Load a stage's data into the TransitionTable.

        Populates ``tt.stages[stage_idx]`` with transitions and bookkeeping.
        Raises FileNotFoundError if the stage file does not exist.
        """
        path = self._stage_path(stage_idx)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stage file not found: {path}")

        with open(path, "rb") as f:
            raw = f.read()
        dctx = zstd.ZstdDecompressor()
        stage_data = json.loads(dctx.decompress(raw))

        sd = tt.stages[stage_idx]

        # Restore transitions via add_transition for proper bookkeeping
        for key_str, gadget_dicts in stage_data["transitions"].items():
            inp_t, out_t = _decode_transition_key(key_str)
            input_state = VectorState(top=list(inp_t[0]), bottom=list(inp_t[1]))
            output_state = VectorState(top=list(out_t[0]), bottom=list(out_t[1]))
            for gd in gadget_dicts:
                gadget = _deserialize_gadget(gd)
                tt.add_transition(stage_idx, input_state, output_state, gadget)

        # Restore bookkeeping fields
        sd.attempts = stage_data["attempts"]
        sd.attempted_pairs = _deserialize_attempted_pairs(stage_data["attempted_pairs"])
        sd.forwarded_outputs = _deserialize_forwarded_outputs(
            stage_data["forwarded_outputs"]
        )
        sd.consecutive_zero_budgets = stage_data["consecutive_zero_budgets"]
        sd.unproductive_waves = stage_data.get("unproductive_waves", 0)

    # -- scored paths ------------------------------------------------------

    def save_scored_paths(self, scored_paths: list[dict]) -> None:
        """Save scored paths as compressed JSON."""
        raw_json = json.dumps(scored_paths).encode("utf-8")
        cctx = zstd.ZstdCompressor(level=_ZSTD_LEVEL)
        compressed = cctx.compress(raw_json)
        _atomic_write(self._scored_paths_path(), compressed)

    def load_scored_paths(self) -> list[dict]:
        """Load scored paths from compressed JSON.

        Returns an empty list if the file does not exist.
        """
        path = self._scored_paths_path()
        if not os.path.exists(path):
            return []
        with open(path, "rb") as f:
            raw = f.read()
        dctx = zstd.ZstdDecompressor()
        return json.loads(dctx.decompress(raw))
