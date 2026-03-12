"""Shared data types for the bitonic sort super-optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field

from tabulate import tabulate


@dataclass
class VectorState:
    """Tracks which elements are in which lanes of top/bottom vectors."""

    top: list[int]  # element indices in top vector
    bottom: list[int]  # element indices in bottom vector

    def __repr__(self):
        # Create transposed table: Top and Bottom as rows, lanes as columns
        max_len = max(len(self.top), len(self.bottom))

        # Pad vectors if needed
        top_vals = self.top + [""] * (max_len - len(self.top))
        bottom_vals = self.bottom + [""] * (max_len - len(self.bottom))

        # Create table data: each row is [label, val0, val1, val2, ...]
        table_data = [["Top"] + top_vals, ["Bottom"] + bottom_vals]

        # Headers are lane indices
        headers = [""] + list(range(max_len))
        return tabulate(table_data, headers=headers, tablefmt="rounded_outline")

    def copy(self):
        return VectorState(top=self.top.copy(), bottom=self.bottom.copy())

    def as_tuple(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return a hashable tuple representation for grouping/deduplication."""
        return (tuple(self.top), tuple(self.bottom))


# ---------------------------------------------------------------------------
# Gadget data-flow graph nodes
#
# These types describe gadget instruction templates as explicit graphs.
# Nodes are trivially picklable (no Z3 objects) and are built once during
# candidate enumeration then sent to worker subprocesses for synthesis.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputRef:
    """Reference to a named input register ("top" or "bottom")."""

    name: str  # "top" or "bottom"


@dataclass(frozen=True)
class Symbolic:
    """A symbolic variable resolved to a Z3 BitVec during evaluation."""

    name: str
    bit_width: int


@dataclass
class Mux:
    """Multiplexer: selects among *sources* via a symbolic select variable.

    Pruning modes (applied by the evaluator):
      - ``None``: unconstrained.
      - ``"force_last"``: select variable forced to the last source index.
      - ``"at_least_one_last"``: when sibling muxes share an ``IntrinsicNode``
        parent, at least one must pick a source >= the last index.
    """

    select: Symbolic
    sources: tuple  # of GadgetNode
    pruning: str | None = None


@dataclass
class IntrinsicNode:
    """An AVX intrinsic applied to operands drawn from the graph."""

    name: str
    operands: dict  # str → GadgetNode
    isomorphic_order: bool = field(default=True, compare=False, hash=False)


# Union of all node types (for type annotations).
GadgetNode = InputRef | Symbolic | Mux | IntrinsicNode


@dataclass
class GadgetGraph:
    """A complete gadget template: one data-flow graph per vector side.

    ``None`` outputs represent identity (depth-0).
    """

    top: IntrinsicNode | None
    bottom: IntrinsicNode | None


@dataclass
class InstructionSpec:
    """Represents a single AVX instruction with its arguments."""

    intrinsic_name: str
    args: dict  # operands, immediates, masks, etc.

    def sort_key(self) -> tuple:
        """Return a deterministic sort key for canonical ordering."""
        return (
            self.intrinsic_name,
            tuple(sorted((k, str(v)) for k, v in self.args.items())),
        )

    def __repr__(self):
        return f"{self.intrinsic_name}({self.args})"


@dataclass
class PermutationGadget:
    """Encapsulates instruction sequence for top/bottom vectors."""

    top_instructions: list[InstructionSpec]  # 0-3 instructions
    bottom_instructions: list[InstructionSpec]  # 0-3 instructions
    validated: bool = False

    _unified_cache: tuple[list[InstructionSpec], int, int] | None = field(
        default=None, repr=False, compare=False
    )
    _unified_map_cache: tuple[list[int], list[int]] | None = field(
        default=None, repr=False, compare=False
    )

    def instruction_count(self) -> int:
        """Total number of deduplicated instructions in this gadget."""
        unified, _, _ = self.unified_instructions()
        return len(unified)

    def sort_key(self) -> tuple:
        """Return a deterministic sort key for canonical ordering."""
        return (
            tuple(i.sort_key() for i in self.top_instructions),
            tuple(i.sort_key() for i in self.bottom_instructions),
        )

    def unified_instructions(self) -> tuple[list[InstructionSpec], int, int]:
        """Merge top/bottom into a unified deduplicated instruction list.

        Two instructions are considered duplicates when they have the same
        intrinsic name and the same structural arguments (recursively
        resolving "prev"/"result_N" references to their dependency
        signatures).

        Returns:
            (instructions, top_output_idx, bottom_output_idx) where
            the indices point into the unified list.
        """
        if self._unified_cache is not None:
            return self._unified_cache

        unified: list[InstructionSpec] = []
        sig_to_idx: dict[tuple, int] = {}

        def _compute_sig(inst: InstructionSpec, prior_sigs: list[tuple]) -> tuple:
            normalized = {}
            for key, value in inst.args.items():
                if isinstance(value, str):
                    if value == "prev":
                        normalized[key] = prior_sigs[-1] if prior_sigs else value
                    elif value.startswith("result_"):
                        idx = int(value[len("result_") :])
                        normalized[key] = (
                            prior_sigs[idx] if idx < len(prior_sigs) else value
                        )
                    else:
                        normalized[key] = value
                else:
                    normalized[key] = value
            return (inst.intrinsic_name, tuple(sorted(normalized.items())))

        # Process top instructions
        top_sigs: list[tuple] = []
        top_unified_indices: list[int] = []
        for inst in self.top_instructions:
            sig = _compute_sig(inst, top_sigs)
            top_sigs.append(sig)
            if sig not in sig_to_idx:
                sig_to_idx[sig] = len(unified)
                unified.append(inst)
            top_unified_indices.append(sig_to_idx[sig])

        # Process bottom instructions, deduplicating against top
        bottom_sigs: list[tuple] = []
        bottom_unified_indices: list[int] = []
        for inst in self.bottom_instructions:
            sig = _compute_sig(inst, bottom_sigs)
            bottom_sigs.append(sig)
            if sig not in sig_to_idx:
                sig_to_idx[sig] = len(unified)
                unified.append(inst)
            bottom_unified_indices.append(sig_to_idx[sig])

        top_out = top_unified_indices[-1] if top_unified_indices else -1
        bottom_out = bottom_unified_indices[-1] if bottom_unified_indices else -1

        self._unified_cache = (unified, top_out, bottom_out)
        self._unified_map_cache = (top_unified_indices, bottom_unified_indices)
        return self._unified_cache

    def unified_instruction_map(
        self,
    ) -> tuple[list[InstructionSpec], list[int], list[int], int, int]:
        """Like unified_instructions() but returns full per-chain index mappings.

        Returns:
            (unified, top_indices, bottom_indices, top_output_idx, bottom_output_idx)
            where top_indices[i] is the unified list index for top_instructions[i],
            and similarly for bottom_indices.
        """
        unified, top_out, bottom_out = self.unified_instructions()
        assert self._unified_map_cache is not None
        top_indices, bottom_indices = self._unified_map_cache
        return (unified, top_indices, bottom_indices, top_out, bottom_out)

    def __repr__(self):
        return f"Gadget(top={len(self.top_instructions)}, bottom={len(self.bottom_instructions)}, validated={self.validated})"


@dataclass
class SolutionNode:
    """Tree node for one stage's solutions.

    Each node represents a unique (input_state, output_state) transition.
    Multiple gadgets that achieve the same transition are stored together,
    allowing pruning of semantically equivalent paths.
    """

    stage: int
    input_state: VectorState
    output_state: VectorState
    gadgets: list[PermutationGadget]  # All gadgets that produce this transition
    children: list["SolutionNode"]

    def __repr__(self):
        return f"SolutionNode(stage={self.stage}, gadgets={len(self.gadgets)}, children={len(self.children)})"
