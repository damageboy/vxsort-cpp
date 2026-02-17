"""Stage-level checkpoint save/load for the bitonic compiler.

Checkpoints capture per-stage solution data so that long-running
compilations can be resumed after interruption.  The on-disk format
is JSON compressed with zstandard.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass

import zstandard as zstd

try:
    from .bitonic_super_optimizer import (
        InstructionSpec,
        PermutationGadget,
        SolutionNode,
        VectorState,
    )
except ImportError:
    from bitonic_super_optimizer import (  # type: ignore
        InstructionSpec,
        PermutationGadget,
        SolutionNode,
        VectorState,
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_VERSION = 1


@dataclass
class CheckpointConfig:
    """Captures the parameters of a compilation run."""

    num_vecs: int
    vm: str  # vector_machine name, e.g. "AVX2"
    prim_type: str  # primitive_type name, e.g. "i32"
    gadget_depth: int
    natural_order: bool
    max_unique_outputs: int
    depth_limit: int | None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_parent_path(parent_path: tuple) -> str:
    """Convert a parent-path tuple to a JSON-safe string key.

    ``()``  (root)  becomes ``"root"``.
    ``("canonical", ((1,2,3,4), (5,6,7,8)))`` becomes a JSON-encoded
    list ``["canonical", [[1,2,3,4], [5,6,7,8]]]``.
    """
    if parent_path == ():
        return "root"
    # parent_path is ("canonical", ((ints...), (ints...)))
    tag, state_tuple = parent_path
    return json.dumps([tag, [list(state_tuple[0]), list(state_tuple[1])]])


def _deserialize_parent_path(key: str) -> tuple:
    """Inverse of :func:`_serialize_parent_path`."""
    if key == "root":
        return ()
    data = json.loads(key)
    tag = data[0]
    state_tuple = (tuple(data[1][0]), tuple(data[1][1]))
    return (tag, state_tuple)


def _serialize_instruction(inst: InstructionSpec) -> dict:
    """Serialize an InstructionSpec to a plain dict."""
    return {"name": inst.intrinsic_name, "args": inst.args}


def _deserialize_instruction(d: dict) -> InstructionSpec:
    """Reconstruct an InstructionSpec from a plain dict."""
    return InstructionSpec(intrinsic_name=d["name"], args=d["args"])


def _serialize_gadget(gadget: PermutationGadget) -> dict:
    """Serialize a PermutationGadget to a plain dict."""
    return {
        "top_instructions": [
            _serialize_instruction(i) for i in gadget.top_instructions
        ],
        "bottom_instructions": [
            _serialize_instruction(i) for i in gadget.bottom_instructions
        ],
        "validated": gadget.validated,
    }


def _deserialize_gadget(d: dict) -> PermutationGadget:
    """Reconstruct a PermutationGadget from a plain dict."""
    return PermutationGadget(
        top_instructions=[_deserialize_instruction(i) for i in d["top_instructions"]],
        bottom_instructions=[
            _deserialize_instruction(i) for i in d["bottom_instructions"]
        ],
        validated=d.get("validated", True),
    )


def _serialize_vector_state(state: VectorState) -> dict:
    """Serialize a VectorState to a plain dict."""
    return {"top": state.top, "bottom": state.bottom}


def _deserialize_vector_state(d: dict) -> VectorState:
    """Reconstruct a VectorState from a plain dict."""
    return VectorState(top=d["top"], bottom=d["bottom"])


def _serialize_node(node: SolutionNode) -> dict:
    """Serialize a SolutionNode to a plain dict (without children)."""
    return {
        "stage": node.stage,
        "input_state": _serialize_vector_state(node.input_state),
        "output_state": _serialize_vector_state(node.output_state),
        "gadgets": [_serialize_gadget(g) for g in node.gadgets],
    }


def _deserialize_node(d: dict) -> SolutionNode:
    """Reconstruct a SolutionNode from a plain dict (children empty)."""
    return SolutionNode(
        stage=d["stage"],
        input_state=_deserialize_vector_state(d["input_state"]),
        output_state=_deserialize_vector_state(d["output_state"]),
        gadgets=[_deserialize_gadget(g) for g in d["gadgets"]],
        children=[],
    )


def _serialize_output_tuple(state_tuple: tuple) -> str:
    """Serialize an output-state tuple to a JSON string key."""
    return json.dumps([list(state_tuple[0]), list(state_tuple[1])])


def _deserialize_output_tuple(key: str) -> tuple:
    """Inverse of :func:`_serialize_output_tuple`."""
    data = json.loads(key)
    return (tuple(data[0]), tuple(data[1]))


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------


def checkpoint_filename(
    num_vecs: int,
    vm: str,
    prim_type: str,
    natural_order: bool,
    stage_idx: int,
) -> str:
    """Return the canonical checkpoint file name for a given stage."""
    order_suffix = "_natural" if natural_order else ""
    return f"checkpoint_{num_vecs}x{vm}_{prim_type}{order_suffix}_stage{stage_idx}.json.zst"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    config: CheckpointConfig,
    per_stage_data: dict[
        int, tuple[dict[tuple, list[SolutionNode]], dict[tuple, VectorState]]
    ],
    stages_completed: list[int],
    last_completed_stage: int,
) -> None:
    """Write a checkpoint to *path* using JSON + zstandard compression.

    The write is atomic: data is written to a temporary file in the same
    directory and then renamed into place so a crash mid-write cannot
    corrupt an existing checkpoint.

    Args:
        path: Destination file path.
        config: Compilation parameters.
        per_stage_data: ``{stage_idx: (nodes_by_parent, unique_outputs)}``
            where *nodes_by_parent* maps parent-path tuples to lists of
            ``SolutionNode`` and *unique_outputs* maps output-state tuples
            to ``VectorState`` objects.
        stages_completed: Sorted list of stage indices that completed.
        last_completed_stage: Index of the most recently completed stage.
    """
    # Build serializable stages dict
    stages_dict: dict[str, dict] = {}
    for stage_idx, (nodes_by_parent, unique_outputs) in per_stage_data.items():
        # Serialize nodes_by_parent
        serialized_nodes: dict[str, list[dict]] = {}
        for parent_path, nodes in nodes_by_parent.items():
            key = _serialize_parent_path(parent_path)
            serialized_nodes[key] = [_serialize_node(n) for n in nodes]

        # Serialize unique_outputs
        serialized_outputs: dict[str, dict] = {}
        for output_tuple, state in unique_outputs.items():
            key = _serialize_output_tuple(output_tuple)
            serialized_outputs[key] = _serialize_vector_state(state)

        stages_dict[str(stage_idx)] = {
            "nodes": serialized_nodes,
            "unique_outputs": serialized_outputs,
        }

    payload = {
        "version": CHECKPOINT_VERSION,
        "config": asdict(config),
        "last_completed_stage": last_completed_stage,
        "stages_completed": sorted(stages_completed),
        "stages": stages_dict,
    }

    json_bytes = json.dumps(payload, indent=2).encode("utf-8")

    cctx = zstd.ZstdCompressor(level=3)
    compressed = cctx.compress(json_bytes)

    # Atomic write: temp file + rename
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    closed = False
    try:
        os.write(fd, compressed)
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
# Load
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: str,
) -> tuple[
    CheckpointConfig,
    dict[int, tuple[dict[tuple, list[SolutionNode]], dict[tuple, VectorState]]],
    list[int],
    int,
]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Returns:
        ``(config, per_stage_data, stages_completed, last_completed_stage)``
    """
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        decompressed = dctx.decompress(f.read())

    payload = json.loads(decompressed.decode("utf-8"))

    version = payload["version"]
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version} "
            f"(expected {CHECKPOINT_VERSION})"
        )

    config = CheckpointConfig(**payload["config"])
    last_completed_stage = payload["last_completed_stage"]
    stages_completed = payload["stages_completed"]

    per_stage_data: dict[
        int, tuple[dict[tuple, list[SolutionNode]], dict[tuple, VectorState]]
    ] = {}

    for stage_key, stage_data in payload["stages"].items():
        stage_idx = int(stage_key)

        # Deserialize nodes_by_parent
        nodes_by_parent: dict[tuple, list[SolutionNode]] = {}
        for parent_key, node_list in stage_data["nodes"].items():
            parent_path = _deserialize_parent_path(parent_key)
            nodes_by_parent[parent_path] = [_deserialize_node(n) for n in node_list]

        # Deserialize unique_outputs
        unique_outputs: dict[tuple, VectorState] = {}
        for output_key, state_dict in stage_data["unique_outputs"].items():
            output_tuple = _deserialize_output_tuple(output_key)
            unique_outputs[output_tuple] = _deserialize_vector_state(state_dict)

        per_stage_data[stage_idx] = (nodes_by_parent, unique_outputs)

    return config, per_stage_data, stages_completed, last_completed_stage


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_checkpoint_config(
    saved: CheckpointConfig,
    current: CheckpointConfig,
) -> tuple[bool, str]:
    """Check whether a saved checkpoint is compatible with the current run.

    All fields must match **except** ``depth_limit``: resuming with a
    *higher* (or unlimited) depth limit is permitted because it simply
    means exploring additional stages beyond what was already checkpointed.

    Args:
        saved: Configuration read from the checkpoint file.
        current: Configuration for the current compilation run.

    Returns:
        ``(ok, message)`` where *ok* is ``True`` when the configs are
        compatible and *message* describes the first mismatch when they
        are not.
    """
    for field in (
        "num_vecs",
        "vm",
        "prim_type",
        "gadget_depth",
        "natural_order",
        "max_unique_outputs",
    ):
        saved_val = getattr(saved, field)
        current_val = getattr(current, field)
        if saved_val != current_val:
            return False, (
                f"Config mismatch on '{field}': "
                f"checkpoint has {saved_val!r}, current run has {current_val!r}"
            )

    # depth_limit: current must be >= saved (or unlimited)
    if current.depth_limit is not None and saved.depth_limit is not None:
        if current.depth_limit < saved.depth_limit:
            return False, (
                f"Cannot resume with a lower depth_limit: "
                f"checkpoint has {saved.depth_limit}, "
                f"current run has {current.depth_limit}"
            )

    return True, "OK"
