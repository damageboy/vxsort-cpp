from __future__ import annotations
import time
import copy
import os
import tarfile
import io
import zstandard as zstd
import queue as queue_mod
from dataclasses import dataclass, field
from tabulate import tabulate
from multiprocessing import Pool
from z3 import (
    Solver,
    Optimize,
    Context,
    main_ctx,
    Extract,
    BitVecVal,
    sat,
    BitVec,
    Distinct,
    Or,
    And,
    If,
    ULE,
    simplify,
)

try:
    from .success_progress import SuccessProgress
    from . import z3_avx
    from .bitonic_sorter import BitonicSorter
    from .utils import vector_machine, primitive_type, width_dict
except ImportError:
    from success_progress import SuccessProgress
    import z3_avx  # type: ignore
    from bitonic_sorter import BitonicSorter
    from utils import vector_machine, primitive_type, width_dict


def _all_smt(s, terms, max_results=None):
    """Enumerate all satisfying models over the given terms.

    Uses the corrected all_smt algorithm from 'Programming Z3'
    (Z3 issue #5765). Implemented iteratively with an explicit
    work-stack to avoid hitting Python's recursion limit when a
    variable has many valid assignments.

    Args:
        s: Z3 Solver instance.
        terms: Sequence of Z3 terms to enumerate over.
        max_results: Optional cap on the number of models returned.
            When ``None`` (the default) all models are enumerated.
    """
    terms = list(terms)
    count = 0

    # Each stack frame mirrors one call to the recursive algorithm:
    #   all_smt_rec(terms) would check sat, yield model, then iterate
    #   over terms[i:] pushing/blocking/fixing.
    # We represent the work as (terms_slice, model, next_i) where:
    #   terms_slice: the terms list for this "call"
    #   model: the model found at this level (None if not yet checked)
    #   next_i: the next index in the for-loop to process
    # A sentinel value of model=None means "need to check sat first".
    stack = [(terms, None, 0)]

    while stack:
        if max_results is not None and count >= max_results:
            # Unwind all pushed solver scopes
            for _ in range(len(stack) - 1):
                s.pop()
            break

        cur_terms, model, next_i = stack[-1]

        # First visit to this frame: check satisfiability
        if model is None:
            if sat == s.check():
                model = s.model()
                count += 1
                stack[-1] = (cur_terms, model, 0)
                yield model
                continue
            else:
                # Unsatisfiable — pop this frame
                stack.pop()
                if stack:
                    s.pop()
                continue

        # Advance the for-loop over terms
        if next_i >= len(cur_terms):
            # Done with all sub-partitions at this level
            stack.pop()
            if stack:
                s.pop()
            continue

        # Process partition i = next_i
        i = next_i
        stack[-1] = (cur_terms, model, next_i + 1)

        # Push solver scope and add blocking/fixing constraints
        s.push()
        s.add(cur_terms[i] != model.eval(cur_terms[i], model_completion=True))
        for j in range(i):
            s.add(cur_terms[j] == model.eval(cur_terms[j], model_completion=True))

        # "Recurse" into all_smt_rec(cur_terms[i:])
        stack.append((cur_terms[i:], None, 0))


@dataclass(frozen=True)
class _IntrinsicDispatchRule:
    """One ordered intrinsic-dispatch rule.

    A rule matches when the argument key-set exactly equals ``arg_keys``.
    The intrinsic is then called using ``call_order`` to avoid argument-order
    ambiguity.
    """

    name: str
    arg_keys: frozenset[str]
    call_order: tuple[str, ...]


def _rule(
    name: str,
    arg_keys: tuple[str, ...],
    call_order: tuple[str, ...],
) -> _IntrinsicDispatchRule:
    """Convenience constructor for readable dispatch table entries."""
    return _IntrinsicDispatchRule(
        name=name,
        arg_keys=frozenset(arg_keys),
        call_order=call_order,
    )


_INTRINSIC_DISPATCH_RULES: tuple[_IntrinsicDispatchRule, ...] = (
    _rule(
        name="masked_single_with_control",
        arg_keys=("k", "src", "op_idx", "a"),
        call_order=("src", "k", "op_idx", "a"),
    ),
    _rule(
        name="masked_dual_with_immediate",
        arg_keys=("k", "src", "a", "b", "imm8"),
        call_order=("src", "k", "a", "b", "imm8"),
    ),
    _rule(
        name="masked_single_with_immediate",
        arg_keys=("k", "src", "a", "imm8"),
        call_order=("src", "k", "a", "imm8"),
    ),
    _rule(
        name="masked_dual_without_immediate",
        arg_keys=("k", "src", "a", "b"),
        call_order=("src", "k", "a", "b"),
    ),
    _rule(
        name="masked_permutex2var",
        arg_keys=("k", "a", "op_idx", "b"),
        call_order=("a", "k", "op_idx", "b"),
    ),
    _rule(
        name="unmasked_permutex2var",
        arg_keys=("a", "op_idx", "b"),
        call_order=("a", "op_idx", "b"),
    ),
    _rule(
        name="single_source_with_control_index",
        arg_keys=("a", "op_idx"),
        call_order=("a", "op_idx"),
    ),
    _rule(
        name="single_source_with_immediate",
        arg_keys=("a", "imm8"),
        call_order=("a", "imm8"),
    ),
    _rule(
        name="dual_source_with_immediate",
        arg_keys=("a", "b", "imm8"),
        call_order=("a", "b", "imm8"),
    ),
    _rule(
        name="dual_source_with_mask",
        arg_keys=("a", "b", "mask"),
        call_order=("a", "b", "mask"),
    ),
    _rule(
        name="dual_source_plain",
        arg_keys=("a", "b"),
        call_order=("a", "b"),
    ),
)


def _match_dispatch_rule(arg_keys: frozenset[str]) -> _IntrinsicDispatchRule | None:
    """Return the first matching dispatch rule or ``None``.

    Uses exact key-set matching for simpler, unambiguous dispatch.
    """
    for rule in _INTRINSIC_DISPATCH_RULES:
        if arg_keys == rule.arg_keys:
            return rule
    return None


def _dispatch_intrinsic_fallback(intrinsic, args: dict):
    """Fallback positional dispatch that preserves incoming argument order."""
    arg_order = tuple(args.keys())
    return intrinsic(*(args[key] for key in arg_order))


def _dispatch_intrinsic_by_signature(intrinsic, args: dict):
    """Call an intrinsic using the first matching dispatch rule.

    Falls back to legacy positional call order when no rule matches.
    """
    rule = _match_dispatch_rule(frozenset(args.keys()))
    if rule is None:
        return _dispatch_intrinsic_fallback(intrinsic, args)
    return intrinsic(*(args[key] for key in rule.call_order))


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


@dataclass
class SymbolicPlaceholder:
    """Represents a symbolic Z3 variable that can be safely pickled."""

    name: str
    size: int


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
        return self._unified_cache

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


def get_available_intrinsics(
    vm: vector_machine, prim_type: primitive_type
) -> dict[str, callable]:
    """Get available intrinsics for the given VM and primitive type."""
    intrinsics = {}

    # For AVX2 i32, we need YMM (256-bit) operations on 32-bit elements
    if vm == vector_machine.AVX2 and prim_type == primitive_type.i32:
        # Single input permutes (with immediates)
        intrinsics["_mm256_permute4x64_epi64"] = z3_avx._mm256_permute4x64_epi64
        intrinsics["_mm256_permute_ps"] = z3_avx._mm256_permute_ps

        # Single input permutes (with control vectors)
        intrinsics["_mm256_permutexvar_epi32"] = z3_avx._mm256_permutexvar_epi32
        intrinsics["_mm256_permutevar_ps"] = z3_avx._mm256_permutevar_ps

        # Two input permutes/shuffles
        intrinsics["_mm256_shuffle_ps"] = z3_avx._mm256_shuffle_ps
        intrinsics["_mm256_unpacklo_epi32"] = z3_avx._mm256_unpacklo_epi32
        intrinsics["_mm256_unpackhi_epi32"] = z3_avx._mm256_unpackhi_epi32
        intrinsics["_mm256_permute2x128_si256"] = z3_avx._mm256_permute2x128_si256

        # Blends
        intrinsics["_mm256_blend_ps"] = z3_avx._mm256_blend_ps
        intrinsics["_mm256_blendv_ps"] = z3_avx._mm256_blendv_ps

        # Align operations
        intrinsics["_mm256_alignr_epi32"] = z3_avx._mm256_alignr_epi32

    # For AVX2 i64, we need YMM (256-bit) operations on 64-bit elements
    if vm == vector_machine.AVX2 and prim_type == primitive_type.i64:
        # Single input permutes (with immediates)
        intrinsics["_mm256_permute4x64_epi64"] = z3_avx._mm256_permute4x64_epi64
        intrinsics["_mm256_permute_pd"] = z3_avx._mm256_permute_pd

        # Single input permutes (with control vectors)
        intrinsics["_mm256_permutexvar_epi64"] = z3_avx._mm256_permutexvar_epi64
        intrinsics["_mm256_permutevar_pd"] = z3_avx._mm256_permutevar_pd

        # Two input permutes/shuffles
        intrinsics["_mm256_shuffle_pd"] = z3_avx._mm256_shuffle_pd
        intrinsics["_mm256_unpacklo_epi64"] = z3_avx._mm256_unpacklo_epi64
        intrinsics["_mm256_unpackhi_epi64"] = z3_avx._mm256_unpackhi_epi64
        intrinsics["_mm256_permute2x128_si256"] = z3_avx._mm256_permute2x128_si256

        # Blends
        intrinsics["_mm256_blend_pd"] = z3_avx._mm256_blend_pd
        intrinsics["_mm256_blendv_pd"] = z3_avx._mm256_blendv_pd

        # Align operations
        intrinsics["_mm256_alignr_epi64"] = z3_avx._mm256_alignr_epi64

    # For AVX512 i64, we need ZMM (512-bit) operations on 64-bit elements
    if vm == vector_machine.AVX512 and prim_type == primitive_type.i64:
        # Unmasked single-input
        intrinsics["_mm512_permutexvar_epi64"] = z3_avx._mm512_permutexvar_epi64
        intrinsics["_mm512_permute_pd"] = z3_avx._mm512_permute_pd
        intrinsics["_mm512_permutevar_pd"] = z3_avx._mm512_permutevar_pd
        # Masked single-input
        intrinsics["_mm512_mask_permutexvar_epi64"] = (
            z3_avx._mm512_mask_permutexvar_epi64
        )
        intrinsics["_mm512_mask_permute_pd"] = z3_avx._mm512_mask_permute_pd
        intrinsics["_mm512_mask_permutevar_pd"] = z3_avx._mm512_mask_permutevar_pd
        # Unmasked dual-input
        intrinsics["_mm512_permutex2var_epi64"] = z3_avx._mm512_permutex2var_epi64
        intrinsics["_mm512_shuffle_pd"] = z3_avx._mm512_shuffle_pd
        intrinsics["_mm512_unpacklo_epi64"] = z3_avx._mm512_unpacklo_epi64
        intrinsics["_mm512_unpackhi_epi64"] = z3_avx._mm512_unpackhi_epi64
        intrinsics["_mm512_shuffle_i32x4"] = z3_avx._mm512_shuffle_i32x4
        intrinsics["_mm512_alignr_epi64"] = z3_avx._mm512_alignr_epi64
        # Masked dual-input
        intrinsics["_mm512_mask_permutex2var_epi64"] = (
            z3_avx._mm512_mask_permutex2var_epi64
        )
        intrinsics["_mm512_mask_shuffle_pd"] = z3_avx._mm512_mask_shuffle_pd
        intrinsics["_mm512_mask_unpacklo_epi64"] = z3_avx._mm512_mask_unpacklo_epi64
        intrinsics["_mm512_mask_unpackhi_epi64"] = z3_avx._mm512_mask_unpackhi_epi64
        intrinsics["_mm512_mask_shuffle_i32x4"] = z3_avx._mm512_mask_shuffle_i32x4
        intrinsics["_mm512_mask_alignr_epi64"] = z3_avx._mm512_mask_alignr_epi64

    # For AVX512 i32, we need ZMM (512-bit) operations on 32-bit elements
    if vm == vector_machine.AVX512 and prim_type == primitive_type.i32:
        # Unmasked single-input
        intrinsics["_mm512_permutexvar_epi32"] = z3_avx._mm512_permutexvar_epi32
        intrinsics["_mm512_permute_ps"] = z3_avx._mm512_permute_ps
        intrinsics["_mm512_permutevar_ps"] = z3_avx._mm512_permutevar_ps
        # Masked single-input
        intrinsics["_mm512_mask_permutexvar_epi32"] = (
            z3_avx._mm512_mask_permutexvar_epi32
        )
        intrinsics["_mm512_mask_permute_ps"] = z3_avx._mm512_mask_permute_ps
        intrinsics["_mm512_mask_permutevar_ps"] = z3_avx._mm512_mask_permutevar_ps
        # Unmasked dual-input
        intrinsics["_mm512_permutex2var_epi32"] = z3_avx._mm512_permutex2var_epi32
        intrinsics["_mm512_shuffle_ps"] = z3_avx._mm512_shuffle_ps
        intrinsics["_mm512_unpacklo_epi32"] = z3_avx._mm512_unpacklo_epi32
        intrinsics["_mm512_unpackhi_epi32"] = z3_avx._mm512_unpackhi_epi32
        intrinsics["_mm512_shuffle_i32x4"] = z3_avx._mm512_shuffle_i32x4
        intrinsics["_mm512_alignr_epi32"] = z3_avx._mm512_alignr_epi32
        # Masked dual-input
        intrinsics["_mm512_mask_permutex2var_epi32"] = (
            z3_avx._mm512_mask_permutex2var_epi32
        )
        intrinsics["_mm512_mask_shuffle_ps"] = z3_avx._mm512_mask_shuffle_ps
        intrinsics["_mm512_mask_unpacklo_epi32"] = z3_avx._mm512_mask_unpacklo_epi32
        intrinsics["_mm512_mask_unpackhi_epi32"] = z3_avx._mm512_mask_unpackhi_epi32
        intrinsics["_mm512_mask_shuffle_i32x4"] = z3_avx._mm512_mask_shuffle_i32x4
        intrinsics["_mm512_mask_alignr_epi32"] = z3_avx._mm512_mask_alignr_epi32

    return intrinsics


"""
mermaid
title Gadget Synthesis Pipeline
flowchart TD
    subgraph INIT["Initialization (once per GadgetSynthesizer)"]
        direction TB
        avx["get_available_intrinsics(vm, prim_type)"]
        avx --> single["_enumerate_single_input_instructions<br/>e.g. permute_ps(a, imm8)"]
        avx --> dual["_enumerate_dual_input_instructions<br/>e.g. shuffle_ps(a, b, imm8)"]
        single --> templates["Instruction templates<br/>(with SymbolicPlaceholder immediates)"]
        dual --> templates
    end

    subgraph CANDIDATES["Candidate Enumeration (precompute_all_candidates)"]
        direction TB
        templates2["templates"] --> build["_build_instruction_sequences(depth)"]
        build --> d0["depth 0: identity"]
        build --> d1["depth 1: single inst"]
        build --> d2["depth 2: inst pairs<br/>(single,single) (dual,single) (single,dual)"]
        d0 --> combine["_generate_candidate_gadgets_at_depth<br/>cartesian product: top_seq × bottom_seq"]
        d1 --> combine
        d2 --> combine
    end

    subgraph SYNTHESIS["Per-Candidate Synthesis (synthesize_gadget_with_symbolic)"]
        direction TB
        regs["_create_input_registers<br/>Z3 bitvecs with concrete element labels"]
        resolve["_resolve_symbolic_instruction_args<br/>SymbolicPlaceholder → Z3 BitVec"]
        regs --> apply
        resolve --> apply["_apply_instructions (top + bottom sides)"]

        subgraph APPLY["_apply_instructions"]
            direction TB
            inst0["inst0: resolve args via<br/>_substitute_register_names"]
            inst0 --> inst1["inst1+: register args get<br/>_create_operand_mux<br/>(Z3 If-chain over top/bottom/result_0)"]
            inst1 --> prune["_prune_depth2_mux<br/>(for 2-inst sequences only)"]
            prune --> prune_s["1 mux var: force to result_0"]
            prune --> prune_d["2 mux vars: Or(a≥result_0, b≥result_0)"]
        end

        apply --> coex["Add COEX constraints<br/>output = min/max of top_out, bottom_out"]
        coex --> distinct["Add Distinct constraint<br/>(no element duplication)"]
        distinct --> solve["Z3 Solver.check()"]
        solve -->|sat| extract["concretize_instructions<br/>model → concrete imm8/cv values<br/>mux select → 'prev'/'top'/'bottom'"]
        solve -->|unsat| discard["discard candidate"]
        extract --> gadget["PermutationGadget + VectorState"]
    end

    subgraph PARALLEL["Parallel Validation (_validate_gadgets)"]
        direction TB
        pool["multiprocessing.Pool"]
        pool --> workers["_validate_gadget_worker × N<br/>(each runs synthesize_gadget_with_symbolic)"]
        workers --> collect["collect validated gadgets"]
    end

    subgraph TREE["Solution Tree (build_solution_tree)"]
        direction TB
        stages["for each bitonic sort stage"]
        stages --> input_states["for each unique input VectorState"]
        input_states --> jobs["create jobs: candidate × input_state"]
        jobs --> validate["_validate_gadgets (parallel)"]
        validate --> nodes["build SolutionNode DAG<br/>(parent → children across stages)"]
    end

    INIT --> CANDIDATES
    CANDIDATES --> PARALLEL
    PARALLEL --> TREE
"""


class GadgetSynthesizer:
    """Synthesizes permutation gadgets using Z3."""

    def __init__(
        self,
        vm: vector_machine,
        prim_type: primitive_type,
        intrinsic_filter: set[str] | None = None,
    ):
        self.vm = vm
        self.prim_type = prim_type
        self.elements_per_vector = width_dict[vm] // int(prim_type.value[0])
        self.lane_width = int(prim_type.value[0]) * 8  # 16, 32, or 64 bits per element
        self.available_intrinsics = get_available_intrinsics(vm, prim_type)

        if intrinsic_filter is not None:
            unknown = intrinsic_filter - set(self.available_intrinsics)
            if unknown:
                raise ValueError(
                    f"Intrinsics not available for {vm}/{prim_type}: {sorted(unknown)}"
                )
            self.available_intrinsics = {
                k: v
                for k, v in self.available_intrinsics.items()
                if k in intrinsic_filter
            }

        # Memoize instruction templates - these are reused across all gadget generation
        allowed = set(self.available_intrinsics)
        self.single_input_insts_top = [
            i
            for i in self._enumerate_single_input_instructions("top")
            if i.intrinsic_name in allowed
        ]
        self.single_input_insts_bottom = [
            i
            for i in self._enumerate_single_input_instructions("bottom")
            if i.intrinsic_name in allowed
        ]
        self.dual_input_insts_top_bottom = [
            i
            for i in self._enumerate_dual_input_instructions("top", "bottom")
            if i.intrinsic_name in allowed
        ]

    def _create_input_registers(
        self,
        solver: Solver,
        ctx: Context,
        input_state: VectorState,
    ) -> tuple:
        """
        Create Z3 symbolic registers with actual element values.
        Returns (top_reg, bottom_reg).
        """
        # Create registers based on VM type
        if self.vm == vector_machine.AVX2:
            top_reg = z3_avx.ymm_reg("top_input", ctx=ctx)
            bottom_reg = z3_avx.ymm_reg("bottom_input", ctx=ctx)
        elif self.vm == vector_machine.AVX512:
            top_reg = z3_avx.zmm_reg("top_input", ctx=ctx)
            bottom_reg = z3_avx.zmm_reg("bottom_input", ctx=ctx)
        else:
            raise NotImplementedError(
                f"Register creation not implemented for VM: {self.vm}"
            )

        # Set up constraints: each lane contains the actual element value
        for lane_idx in range(self.elements_per_vector):
            top_elem = input_state.top[lane_idx]
            bottom_elem = input_state.bottom[lane_idx]

            lane_start = lane_idx * self.lane_width
            lane_end = lane_start + self.lane_width - 1

            top_lane = Extract(lane_end, lane_start, top_reg)
            bottom_lane = Extract(lane_end, lane_start, bottom_reg)

            solver.add(top_lane == BitVecVal(top_elem, self.lane_width, ctx=ctx))
            solver.add(bottom_lane == BitVecVal(bottom_elem, self.lane_width, ctx=ctx))

        return top_reg, bottom_reg

    def _resolve_symbolic_instruction_args(
        self,
        instructions: list[InstructionSpec],
        ctx: Context,
        symbolic_vars: dict,
    ) -> None:
        """Resolve placeholders to concrete Z3 terms and track symbolic args."""
        # SymbolicPlaceholder is used to communicate with multiprocessing.
        # In tests, Z3 expressions may be passed directly.
        for inst in instructions:
            for key, value in inst.args.items():
                if isinstance(value, SymbolicPlaceholder):
                    if value.size == 8:
                        actual_val = BitVec(value.name, 8, ctx=ctx)
                    elif value.size == 16:
                        actual_val = BitVec(value.name, 16, ctx=ctx)
                    elif value.size == 256:
                        actual_val = z3_avx.ymm_reg(value.name, ctx=ctx)
                    elif value.size == 512:
                        actual_val = z3_avx.zmm_reg(value.name, ctx=ctx)
                    else:
                        actual_val = BitVec(value.name, value.size, ctx=ctx)

                    inst.args[key] = actual_val
                    symbolic_vars[id(actual_val)] = actual_val
                elif hasattr(value, "decl") and callable(getattr(value, "decl", None)):
                    symbolic_vars[id(value)] = value

    def _constrain_output_lanes_to_target_pairs(
        self,
        solver: Solver,
        ctx: Context,
        top_output,
        bottom_output,
        target_pairs: list[tuple[int, int]],
        allow_any_lane_order: bool,
    ) -> list:
        """Constrain each lane to match target_pairs and return all output lanes."""
        all_output_lanes = []
        for lane_idx in range(self.elements_per_vector):
            lane_start = lane_idx * self.lane_width
            lane_end = lane_start + self.lane_width - 1

            top_lane = Extract(lane_end, lane_start, top_output)
            bottom_lane = Extract(lane_end, lane_start, bottom_output)

            all_output_lanes.append(top_lane)
            all_output_lanes.append(bottom_lane)

            if allow_any_lane_order:
                pair_options = []
                for elem_a, elem_b in target_pairs:
                    val_a = BitVecVal(elem_a, self.lane_width, ctx=ctx)
                    val_b = BitVecVal(elem_b, self.lane_width, ctx=ctx)
                    pair_options.append(
                        Or(
                            And(top_lane == val_a, bottom_lane == val_b),
                            And(top_lane == val_b, bottom_lane == val_a),
                        )
                    )
                solver.add(Or(*pair_options))
            else:
                elem_a, elem_b = target_pairs[lane_idx]
                solver.add(top_lane == BitVecVal(elem_a, self.lane_width, ctx=ctx))
                solver.add(bottom_lane == BitVecVal(elem_b, self.lane_width, ctx=ctx))

        return all_output_lanes

    def _fixed_output_state_from_target_pairs(
        self, target_pairs: list[tuple[int, int]], allow_any_lane_order: bool
    ) -> VectorState | None:
        """Build deterministic output state in strict lane-order mode."""
        if allow_any_lane_order:
            return None
        return VectorState(
            top=[pair[0] for pair in target_pairs],
            bottom=[pair[1] for pair in target_pairs],
        )

    def synthesize_gadget_with_symbolic(
        self,
        top_instructions_template: list[InstructionSpec],
        bottom_instructions_template: list[InstructionSpec],
        input_state: VectorState,
        target_pairs: list[tuple[int, int]],
        max_solutions: int = 1,
        max_unique_outputs: int = 3,
        solver_callback: callable | None = None,
        allow_any_lane_order: bool = True,
    ) -> tuple[list[tuple[PermutationGadget, VectorState]], float, float]:
        """
        Synthesize gadgets using symbolic immediates in Z3.

        Takes instruction templates with symbolic values (e.g., BitVec("imm8", 8))
        and lets Z3 find concrete immediate values that satisfy constraints.

        The symbolic values are represented using SymbolicPlaceholder for pickling.
        The pickling is required for multiprocessing.

        Returns (results, construction_time, solver_time) where results is
        list of (gadget, output_state) tuples. The output state is computed
        directly from the satisfying model, avoiding a redundant Z3 solve. The output
        state is in canonical form: for each pair, the lower element index goes to
        the top vector and the higher index goes to bottom, reflecting the
        compare-and-exchange operation.

        Args:
            max_solutions: Maximum number of solutions to return. Defaults to 1,
                which enumerates unique output states deterministically.
                Values > 1 use Z3 Solver with enumeration (non-deterministic ordering).
            max_unique_outputs: When max_solutions == 1, the number of smallest
                unique output states to enumerate per template. Higher values
                increase search diversity at the cost of speed. Default: 3.
            solver_callback: Optional callback receiving the Solver instance.
            allow_any_lane_order: When True (default), any target pair can land
                in any lane and the output state is canonicalized (min to top,
                max to bottom). When False, pair[i] is pinned to lane i with
                strict top/bottom assignment and no canonicalization.
        """
        start_construction = time.perf_counter()
        ctx = main_ctx()
        solver = Solver(ctx=ctx)

        # Create input registers with actual element values
        top_reg, bottom_reg = self._create_input_registers(solver, ctx, input_state)

        # Collect all symbolic variables from instruction templates and resolve them
        symbolic_vars = {}
        self._resolve_symbolic_instruction_args(
            top_instructions_template, ctx, symbolic_vars
        )
        self._resolve_symbolic_instruction_args(
            bottom_instructions_template, ctx, symbolic_vars
        )

        # Apply gadget instructions to get output registers
        top_output, top_mux_constraints = self._apply_instructions(
            top_reg,
            bottom_reg,
            top_instructions_template,
            is_top=True,
            solver=solver,
            symbolic_vars=symbolic_vars,
        )
        bottom_output, bottom_mux_constraints = self._apply_instructions(
            top_reg,
            bottom_reg,
            bottom_instructions_template,
            is_top=False,
            solver=solver,
            symbolic_vars=symbolic_vars,
        )

        # Add mux range constraints (select vars must be in {0, 1, 2})
        for c in top_mux_constraints:
            solver.add(c)
        for c in bottom_mux_constraints:
            solver.add(c)

        all_output_lanes = self._constrain_output_lanes_to_target_pairs(
            solver,
            ctx,
            top_output,
            bottom_output,
            target_pairs,
            allow_any_lane_order,
        )

        # All output elements must be distinct — prevents element duplication
        # where an instruction copies the same element to both top and bottom.
        solver.add(Distinct(*all_output_lanes))

        if solver_callback:
            solver_callback(solver)

        # When strict lane order, the output state is deterministic from pairs
        fixed_output_state = self._fixed_output_state_from_target_pairs(
            target_pairs, allow_any_lane_order
        )

        # Partition symbolic vars: select variables (mux wiring) vs
        # immediate/control vector variables
        select_vars = {
            k: v
            for k, v in symbolic_vars.items()
            if isinstance(k, str) and k.startswith("sel_")
        }
        imm_vars = {k: v for k, v in symbolic_vars.items() if k not in select_vars}
        imm_terms = list(imm_vars.values())

        construction_time = time.perf_counter() - start_construction
        solver_start = time.perf_counter()

        # If no symbolic variables at all, single check suffices
        if not imm_terms and not select_vars:
            result = solver.check()
            if result != sat:
                solver_time = time.perf_counter() - solver_start
                return [], construction_time, solver_time
            model = solver.model()
            gadget, output_state = self._extract_solution_from_model(
                model,
                top_output,
                bottom_output,
                symbolic_vars,
                top_instructions_template,
                bottom_instructions_template,
                fixed_output_state=fixed_output_state,
            )
            solver_time = time.perf_counter() - solver_start
            return [(gadget, output_state)], construction_time, solver_time

        if max_solutions == 1:
            # K-per-wiring algorithm: deterministic synthesis with diversity.
            #
            # Outer loop: discover unique operand wirings (which source each
            # inst2 operand reads from: top, bottom, or prev). The wiring
            # space is small (at most 3^num_select_vars) and many wirings
            # are UNSAT, so this terminates quickly.
            #
            # Inner loop: for each valid wiring, find the K smallest unique
            # output states using Optimize.minimize with output-blocking.
            # For each unique output, pin the output + wiring and minimize
            # immediates for a deterministic gadget.
            #
            # For depth-1 templates (no select vars), the outer loop runs
            # once and the inner loop behaves identically to the previous
            # K-smallest algorithm — fully backward compatible.
            original_assertions = list(solver.assertions())
            results = []
            wiring_blocks = []

            # Outer loop: discover unique wirings
            while True:
                opt_wiring = Optimize(ctx=ctx)
                for a in original_assertions:
                    opt_wiring.add(a)
                for block in wiring_blocks:
                    opt_wiring.add(block)
                # Minimize output to get a deterministic first hit
                opt_wiring.minimize(top_output)
                opt_wiring.minimize(bottom_output)

                if opt_wiring.check() != sat:
                    break  # No more valid wirings

                model = opt_wiring.model()

                # Extract the wiring from the model
                wiring = {}
                for name, var in select_vars.items():
                    wiring[name] = model.evaluate(var, model_completion=True).as_long()

                # Pin this wiring for the inner loop
                wiring_constraints = [
                    var == BitVecVal(wiring[name], var.size())
                    for name, var in select_vars.items()
                ]

                # Inner loop: K smallest unique outputs for this wiring
                output_blocks = []
                for _ in range(max_unique_outputs):
                    opt_k = Optimize(ctx=ctx)
                    for a in original_assertions:
                        opt_k.add(a)
                    for c in wiring_constraints:
                        opt_k.add(c)
                    for block in output_blocks:
                        opt_k.add(block)
                    opt_k.minimize(top_output)
                    opt_k.minimize(bottom_output)

                    if opt_k.check() != sat:
                        break

                    model_k = opt_k.model()
                    top_val = simplify(model_k.eval(top_output))
                    bot_val = simplify(model_k.eval(bottom_output))

                    # Pin output + wiring, minimize immediates only
                    opt_imm = Optimize(ctx=ctx)
                    for a in original_assertions:
                        opt_imm.add(a)
                    for c in wiring_constraints:
                        opt_imm.add(c)
                    opt_imm.add(top_output == top_val)
                    opt_imm.add(bottom_output == bot_val)
                    for term in imm_terms:
                        opt_imm.minimize(term)

                    if opt_imm.check() == sat:
                        gadget, output_state = self._extract_solution_from_model(
                            opt_imm.model(),
                            top_output,
                            bottom_output,
                            symbolic_vars,
                            top_instructions_template,
                            bottom_instructions_template,
                            fixed_output_state=fixed_output_state,
                        )
                        results.append((gadget, output_state))

                    # Block this output for the next inner iteration
                    output_blocks.append(
                        Or(top_output != top_val, bottom_output != bot_val)
                    )

                # Block this entire wiring for the next outer iteration
                if select_vars:
                    wiring_blocks.append(
                        Or(
                            *(
                                var != BitVecVal(wiring[name], var.size())
                                for name, var in select_vars.items()
                            )
                        )
                    )
                else:
                    # No select vars (depth-1): only run outer loop once
                    break

            # Sort by output state for deterministic ordering
            results.sort(key=lambda x: x[1].as_tuple())
            solver_time = time.perf_counter() - solver_start
            return results, construction_time, solver_time

        # Enumerate multiple solutions over symbolic variables
        all_terms = list(symbolic_vars.values())
        results = []
        for model in _all_smt(solver, all_terms, max_results=max_solutions):
            gadget, output_state = self._extract_solution_from_model(
                model,
                top_output,
                bottom_output,
                symbolic_vars,
                top_instructions_template,
                bottom_instructions_template,
                fixed_output_state=fixed_output_state,
            )
            results.append((gadget, output_state))
        solver_time = time.perf_counter() - solver_start
        return results, construction_time, solver_time

    def _extract_solution_from_model(
        self,
        model,
        top_output,
        bottom_output,
        symbolic_vars: dict,
        top_instructions_template: list[InstructionSpec],
        bottom_instructions_template: list[InstructionSpec],
        fixed_output_state: VectorState | None = None,
    ) -> tuple[PermutationGadget, VectorState]:
        """Extract a concrete gadget and output state from a Z3 model.

        Reads element values from the symbolic output registers, builds the
        canonical output state (min to top, max to bottom per lane), and
        concretizes all symbolic variables in the instruction templates
        using the model's assignments.

        When ``fixed_output_state`` is provided (strict lane order mode), the
        output state is used directly without reading from the model or
        applying min/max canonicalization.
        """
        if fixed_output_state is not None:
            output_state = fixed_output_state
        else:
            # Extract element values from output registers to build canonical output state
            output_top = []
            output_bottom = []
            for lane_idx in range(self.elements_per_vector):
                lane_start = lane_idx * self.lane_width
                lane_end = lane_start + self.lane_width - 1

                top_lane = Extract(lane_end, lane_start, top_output)
                bottom_lane = Extract(lane_end, lane_start, bottom_output)

                top_val = model.evaluate(top_lane, model_completion=True)
                bottom_val = model.evaluate(bottom_lane, model_completion=True)

                top_elem = top_val.as_long() if hasattr(top_val, "as_long") else top_val
                bottom_elem = (
                    bottom_val.as_long()
                    if hasattr(bottom_val, "as_long")
                    else bottom_val
                )

                # Canonical ordering: lower element index to top, higher to bottom
                output_top.append(min(top_elem, bottom_elem))
                output_bottom.append(max(top_elem, bottom_elem))

            output_state = VectorState(top=output_top, bottom=output_bottom)

        # Create concrete instructions by substituting symbolic values.
        # Select variables (from mux encoding) are resolved to source name
        # strings ("top", "bottom", "result_N") via _select_to_source().
        def concretize_instructions(
            instructions: list[InstructionSpec],
        ) -> list[InstructionSpec]:
            concrete_insts = []
            for inst_idx, inst in enumerate(instructions):
                concrete_args = {}
                for key, value in inst.args.items():
                    # Check if this arg has a corresponding select variable
                    select_name = f"sel_{key}_{inst.intrinsic_name}_{id(inst)}"
                    if select_name in symbolic_vars:
                        select_val = model.evaluate(
                            symbolic_vars[select_name], model_completion=True
                        ).as_long()
                        concrete_args[key] = self._select_to_source(
                            select_val, inst_idx
                        )
                    elif id(value) in symbolic_vars:
                        if hasattr(value, "size") and callable(
                            getattr(value, "size", None)
                        ):
                            bit_size = value.size()
                            if bit_size == 256:
                                concrete_bitvec = model.evaluate(
                                    value, model_completion=True
                                )
                                if hasattr(concrete_bitvec, "as_long"):
                                    concrete_value = concrete_bitvec.as_long()
                                else:
                                    concrete_value = concrete_bitvec
                                concrete_args[key] = concrete_value
                            elif bit_size == 8:
                                concrete_value = model.evaluate(
                                    value, model_completion=True
                                ).as_long()
                                concrete_args[key] = concrete_value
                            else:
                                concrete_value = model.evaluate(
                                    value, model_completion=True
                                ).as_long()
                                concrete_args[key] = concrete_value
                        else:
                            concrete_value = model.evaluate(
                                value, model_completion=True
                            ).as_long()
                            concrete_args[key] = concrete_value
                    else:
                        concrete_args[key] = value
                concrete_insts.append(
                    InstructionSpec(inst.intrinsic_name, concrete_args)
                )
            return concrete_insts

        concrete_top = concretize_instructions(top_instructions_template)
        concrete_bottom = concretize_instructions(bottom_instructions_template)

        gadget = PermutationGadget(
            top_instructions=concrete_top,
            bottom_instructions=concrete_bottom,
            validated=True,
        )

        return gadget, output_state

    def compute_output_state(
        self, input_state: VectorState, gadget: PermutationGadget
    ) -> VectorState:
        """
        Compute the output state after applying a gadget to an input state.

        Uses Z3 to symbolically execute the gadget and determine element positions.

        Args:
            input_state: Input element positions
            gadget: Gadget to apply

        Returns:
            Output state with new element positions
        """
        ctx = main_ctx()
        solver = Solver(ctx=ctx)

        # Create input registers where each lane contains the element index
        if self.vm == vector_machine.AVX2:
            top_reg = z3_avx.ymm_reg("top_input", ctx=ctx)
            bottom_reg = z3_avx.ymm_reg("bottom_input", ctx=ctx)
        elif self.vm == vector_machine.AVX512:
            top_reg = z3_avx.zmm_reg("top_input", ctx=ctx)
            bottom_reg = z3_avx.zmm_reg("bottom_input", ctx=ctx)
        else:
            raise NotImplementedError(
                f"Register creation not implemented for VM: {self.vm}"
            )

        # Constrain input registers to contain element indices
        for lane_idx in range(self.elements_per_vector):
            top_elem = input_state.top[lane_idx]
            bottom_elem = input_state.bottom[lane_idx]

            lane_start = lane_idx * self.lane_width
            lane_end = lane_start + self.lane_width - 1

            top_lane = Extract(lane_end, lane_start, top_reg)
            bottom_lane = Extract(lane_end, lane_start, bottom_reg)

            solver.add(top_lane == BitVecVal(top_elem, self.lane_width, ctx=ctx))
            solver.add(bottom_lane == BitVecVal(bottom_elem, self.lane_width, ctx=ctx))

        # Apply gadget instructions (mux constraints ignored here —
        # concrete gadgets have resolved args, no select variables)
        top_output, _ = self._apply_instructions(
            top_reg, bottom_reg, gadget.top_instructions, is_top=True
        )
        bottom_output, _ = self._apply_instructions(
            top_reg, bottom_reg, gadget.bottom_instructions, is_top=False
        )

        # Solve to get concrete output values
        if solver.check() != sat:
            # If unsatisfiable, return input state (shouldn't happen with valid gadgets)
            print("Warning: Could not compute output state, returning input")
            return input_state.copy()

        model = solver.model()

        # Extract output element positions from the model
        output_top = []
        output_bottom = []

        for lane_idx in range(self.elements_per_vector):
            lane_start = lane_idx * self.lane_width
            lane_end = lane_start + self.lane_width - 1

            top_lane = Extract(lane_end, lane_start, top_output)
            bottom_lane = Extract(lane_end, lane_start, bottom_output)

            # Evaluate the output lanes in the model
            top_val = model.evaluate(top_lane, model_completion=True)
            bottom_val = model.evaluate(bottom_lane, model_completion=True)

            # Convert Z3 bit-vector values to Python integers
            top_val_long = top_val.as_long() if hasattr(top_val, "as_long") else top_val
            bottom_val_long = (
                bottom_val.as_long() if hasattr(bottom_val, "as_long") else bottom_val
            )

            output_top.append(top_val_long)
            output_bottom.append(bottom_val_long)

        return VectorState(top=output_top, bottom=output_bottom)

    def _substitute_register_names(
        self, arg, top_reg, bottom_reg, current_reg, symbolic_vars=None, key=None
    ):
        """
        Substitute register name strings with actual Z3 register variables.

        Args:
            arg: Argument value (could be string register name, immediate, Z3 symbolic var, etc.)
            top_reg: Top Z3 register
            bottom_reg: Bottom Z3 register
            current_reg: Current Z3 register being computed
            symbolic_vars: Optional dict to track symbolic variables by name

        Returns:
            Actual register if arg is a register name, otherwise returns arg unchanged
        """
        # If it's a Z3 expression (BitVec), track it in symbolic_vars if it has a name
        if hasattr(arg, "decl") and callable(getattr(arg, "decl", None)):
            # This is a Z3 expression
            if symbolic_vars is not None:
                # Try to extract the variable name
                symbolic_vars[id(arg)] = arg
            return arg

        if isinstance(arg, int):
            # Check if this is a 256-bit (or 512-bit) control vector or an 8-bit immediate
            if key == "imm8":
                return BitVecVal(arg, 8)
            else:
                return BitVecVal(arg, width_dict[self.vm] * 8)

        if isinstance(arg, str):
            if arg == "top":
                return top_reg
            elif arg == "bottom":
                return bottom_reg
            elif arg == "prev":
                return current_reg
            elif arg.startswith("result_"):
                return current_reg  # best-effort for concrete gadgets
            elif arg in ["input", "a"]:  # Generic input register
                return current_reg
        return arg

    @staticmethod
    def _select_to_source(select_val: int, inst_idx: int) -> str:
        """Map a mux select value to a source name string.

        0 → "top", 1 → "bottom".
        For result references: the immediately preceding result (result_{inst_idx-1})
        is called "prev" for backward compatibility. Earlier results use "result_N".
        """
        if select_val == 0:
            return "top"
        if select_val == 1:
            return "bottom"
        result_idx = select_val - 2
        if result_idx == inst_idx - 1:
            return "prev"
        return f"result_{result_idx}"

    def _create_operand_mux(
        self,
        key: str,
        inst: InstructionSpec,
        inst_idx: int,
        top_reg,
        bottom_reg,
        results: list,
        symbolic_vars: dict | None,
        mux_constraints: list,
    ):
        """Create a Z3 If-else chain selecting between {top, bottom, result_0, ...}.

        For inst2+ in a multi-instruction chain, register operands ("a", "b")
        get a select variable that Z3 uses to pick the optimal source from
        the original top/bottom registers or any prior instruction's output.

        Args:
            key: Argument key ("a" or "b")
            inst: The instruction spec (used for unique naming)
            inst_idx: Index of this instruction in the sequence
            top_reg: Z3 register for original top input
            bottom_reg: Z3 register for original bottom input
            results: List of Z3 registers from prior instructions' outputs
            symbolic_vars: Dict to track the select variable for later extraction
            mux_constraints: List to append range constraints to

        Returns:
            Z3 If-expression selecting between the sources
        """
        # Number of options: top, bottom, plus one per prior result
        num_options = 2 + len(results)
        # Bit-width: 2 bits for up to 4 options, 3 bits for 5-8
        bit_width = 2 if num_options <= 4 else 3

        select_name = f"sel_{key}_{inst.intrinsic_name}_{id(inst)}"
        select_var = BitVec(select_name, bit_width)

        if symbolic_vars is not None:
            symbolic_vars[select_name] = select_var

        # Constrain to valid range
        mux_constraints.append(ULE(select_var, BitVecVal(num_options - 1, bit_width)))

        # Build If-chain: 0→top, 1→bottom, 2→result_0, 3→result_1, ...
        # Start from the last option (fallback) and build backwards
        mux = results[-1]  # last result is the default/fallback
        for i in range(num_options - 2, -1, -1):
            if i == 0:
                source = top_reg
            elif i == 1:
                source = bottom_reg
            else:
                source = results[i - 2]
            mux = If(select_var == i, source, mux)

        return mux

    def _apply_instructions(
        self,
        top_reg,
        bottom_reg,
        instructions: list[InstructionSpec],
        is_top: bool,
        solver=None,
        symbolic_vars=None,
    ):
        """
        Apply a sequence of instructions to compute output register.

        For multi-instruction sequences, inst1+ register operands ("a", "b"
        referencing "top"/"bottom") get a mux select variable so Z3 can pick
        from {top, bottom, result_0, ...}.

        For exactly-2-instruction sequences (auto-generated depth-2), the mux
        is then constrained to prevent degeneration to depth-1:
        - Single-input inst1: mux forced to result_0 (picking top/bottom
          would discard inst0, duplicating a depth-1 gadget).
        - Dual-input inst1: at least one of (a, b) must select result_0.

        For 3+ instruction sequences (user templates with arbitrary wiring),
        mux variables are left unconstrained.

        Returns:
            (output_register, mux_constraints) tuple.
        """
        current_reg = top_reg if is_top else bottom_reg
        results = []
        mux_constraints = []

        for inst_idx, inst in enumerate(instructions):
            intrinsic = self.available_intrinsics.get(inst.intrinsic_name)
            if intrinsic is None:
                raise ValueError(f"Unknown intrinsic: {inst.intrinsic_name}")

            args = {}
            for key, value in inst.args.items():
                if (
                    inst_idx > 0
                    and results
                    and key in ("a", "b")
                    and isinstance(value, str)
                    and value in ("top", "bottom")
                ):
                    args[key] = self._create_operand_mux(
                        key,
                        inst,
                        inst_idx,
                        top_reg,
                        bottom_reg,
                        results,
                        symbolic_vars,
                        mux_constraints,
                    )
                else:
                    args[key] = self._substitute_register_names(
                        value,
                        top_reg,
                        bottom_reg,
                        current_reg,
                        symbolic_vars,
                        key=key,
                    )

            current_reg = _dispatch_intrinsic_by_signature(intrinsic, args)
            results.append(current_reg)

        # For depth-2 sequences, constrain inst1's mux to prevent
        # degeneration to a depth-1 gadget.
        if len(instructions) == 2 and symbolic_vars is not None:
            self._prune_depth2_mux(instructions[1], mux_constraints, symbolic_vars)

        return current_reg, mux_constraints

    @staticmethod
    def _prune_depth2_mux(
        inst1: InstructionSpec,
        mux_constraints: list,
        symbolic_vars: dict,
    ):
        """Constrain inst1's mux in a depth-2 sequence to use result_0.

        Mux encoding: 0=top, 1=bottom, 2=result_0, ...
        Without constraints, Z3 can pick top/bottom for all operands,
        making inst0 dead code — equivalent to a depth-1 gadget.

        - 1 mux var (single-input inst1): force it to result_0.
        - 2 mux vars (dual-input inst1): at least one must be >= result_0.
        """
        FIRST_RESULT = 2

        sel_vars = []
        for key in ("a", "b"):
            name = f"sel_{key}_{inst1.intrinsic_name}_{id(inst1)}"
            if name in symbolic_vars:
                sel_vars.append(symbolic_vars[name])

        if len(sel_vars) == 1:
            mux_constraints.append(sel_vars[0] == FIRST_RESULT)
        elif len(sel_vars) >= 2:
            mux_constraints.append(
                Or(
                    ULE(BitVecVal(FIRST_RESULT, sel_vars[0].size()), sel_vars[0]),
                    ULE(BitVecVal(FIRST_RESULT, sel_vars[1].size()), sel_vars[1]),
                )
            )

    def _generate_candidate_gadgets_at_depth(
        self, top_depth: int, bottom_depth: int
    ) -> list[tuple[list[InstructionSpec], list[InstructionSpec]]]:
        """
        Generate candidate instruction sequences without validation.
        Returns list of (top_sequence, bottom_sequence) tuples.
        """
        top_sequences = self._build_instruction_sequences(
            top_depth, self.single_input_insts_top
        )
        bottom_sequences = self._build_instruction_sequences(
            bottom_depth, self.single_input_insts_bottom
        )

        # Generate all combinations
        candidates = []
        for top_seq in top_sequences:
            for bottom_seq in bottom_sequences:
                candidates.append((top_seq, bottom_seq))

        return candidates

    def _build_instruction_sequences(
        self, depth: int, single_insts: list[InstructionSpec]
    ) -> list[list[InstructionSpec]]:
        """
        mermaid
        title Permutation Gadget Data-Flow Shapes
        flowchart TD

            subgraph D0["DEPTH 0 — Identity"]
                direction TB
                d0_top["top reg"] --> d0_COEX["COEX<br/>min/max"]
                d0_bot["bottom reg"] --> d0_COEX
            end

            subgraph D1["1 inst per side"]
                subgraph A["Shape A: single-input (e.g. permute_ps)"]
                    direction TB
                    a_top["top reg"]
                    a_bot["bottom reg"]
                    a_top --> a_i0["inst0(a, imm8/cv)"]
                    a_i0 --> a_COEX["COEX"]
                    a_bot --> a_COEX
                end
                subgraph B["Shape B: dual-input (e.g. shuffle_ps)"]
                    direction TB
                    b_top["top reg"]
                    b_bot["bottom reg"]
                    b_top --> b_i0["inst0(a, b, imm8)"]
                    b_bot --> b_i0
                    b_i0 --> b_COEX["COEX"]
                end
            end

            subgraph D2["2 inst per side"]
                subgraph C["Shape C: single → single (hardwired chain)"]
                    direction TB
                    c_top["top reg"]
                    c_bot["bottom reg"]
                    c_top --> c_i0["inst0(a, imm8/cv)"]
                    c_i0 --> c_i1["inst1(result_0, imm8/cv)"]
                    c_i1 --> c_COEX["COEX"]
                end
                subgraph D["Shape D: dual → single (hardwired chain)"]
                    direction TB
                    d_top["top reg"]
                    d_bot["bottom reg"]
                    d_top --> d_i0["inst0(a, b, imm8)"]
                    d_bot --> d_i0
                    d_i0 --> d_i1["inst1(result_0, imm8/cv)"]
                    d_i1 --> d_COEX["COEX"]
                end
                subgraph E["Shape E: single → dual (mux, ≥1 must use result_0)"]
                    direction TB
                    e_top["top reg"]
                    e_bot["bottom reg"]
                    e_top --> e_i0["inst0(a, imm8/cv)"]
                    e_i0 --> e_r0((result_0))
                    e_r0 -.-> e_mux_a{{mux a}}
                    e_top -.-> e_mux_a
                    e_bot -.-> e_mux_a
                    e_r0 -.-> e_mux_b{{mux b}}
                    e_top -.-> e_mux_b
                    e_bot -.-> e_mux_b
                    e_mux_a --> e_i1["inst1(a, b, imm8)"]
                    e_mux_b --> e_i1
                    e_i1 --> e_COEX["COEX"]
                end
            end

            subgraph FG["Full Gadget: top side + bottom side + COEX"]
                direction TB
                fg_top["top reg"] --> fg_ts["top-side instructions<br/>(shape A-E)"]
                fg_bot["bottom reg"] --> fg_ts
                fg_top --> fg_bs["bottom-side instructions<br/>(shape A-E)"]
                fg_bot --> fg_bs
                fg_ts --> fg_nt["new top"]
                fg_bs --> fg_nb["new bottom"]
                fg_nt --> fg_COEX["COEX<br/>min(new_top, new_bot)<br/>max(new_top, new_bot)"]
                fg_nb --> fg_COEX
                fg_COEX --> fg_ot["output top<br/>(mins)"]
                fg_COEX --> fg_ob["output bottom<br/>(maxs)"]
            end
        """
        if depth <= 0:
            return [[]]

        sequences = []
        if depth == 1:
            sequences = [[inst] for inst in single_insts]
            sequences.extend([[inst] for inst in self.dual_input_insts_top_bottom])
        elif depth == 2:
            # Try pairs: (single, single), (dual, single), (single, dual)
            for inst1 in single_insts:
                for inst2 in single_insts:
                    sequences.append([inst1, inst2])
            for inst1 in self.dual_input_insts_top_bottom:
                for inst2 in single_insts:
                    sequences.append([inst1, inst2])
            for inst1 in single_insts:
                for inst2 in self.dual_input_insts_top_bottom:
                    sequences.append([inst1, inst2])
        elif depth > 2:
            raise ValueError(f"Instruction depth {depth} is not supported (max is 2)")
        return sequences

    def precompute_all_candidates(
        self, gadget_depth: int
    ) -> list[tuple[list[InstructionSpec], list[InstructionSpec]]]:
        """Pre-compute all candidate instruction sequences for all depth combinations.

        The candidates depend only on gadget_depth and the available intrinsics,
        not on input state or stage pairs. This allows computing them once and
        reusing across all stages and input states.
        """
        all_candidates = []
        for top_depth in range(gadget_depth + 1):
            for bottom_depth in range(gadget_depth + 1):
                candidates = self._generate_candidate_gadgets_at_depth(
                    top_depth, bottom_depth
                )
                all_candidates.extend(candidates)
        return all_candidates

    def _validate_gadgets(
        self,
        jobs: list[tuple],
        progress: SuccessProgress | None = None,
        task_id=None,
        max_workers: int | None = None,
        max_tasks_per_child: int | None = 1000,
    ) -> list[tuple[PermutationGadget, VectorState, VectorState, dict]]:
        """
        Validate candidate gadgets using synthesis in parallel.

        Returns list of (gadget, input_state, output_state, metadata) tuples.
        The output_state is computed directly during validation, avoiding
        a redundant Z3 solve.
        """
        validated_gadgets = []

        def _log(msg: str):
            if progress:
                progress.console.print(msg)
            else:
                print(msg)

        pool = Pool(processes=max_workers, maxtasksperchild=max_tasks_per_child)
        total_construct_time = 0.0
        total_solve_time = 0.0

        # Use imap_unordered for streaming results and progress updates
        for (
            gadget_results,
            job_input_state,
            job_metadata,
            construct_time,
            solver_time,
        ) in pool.imap_unordered(_validate_gadget_worker, jobs):
            total_construct_time += construct_time
            total_solve_time += solver_time

            success_inc = 0
            for gadget, output_state in gadget_results:
                validated_gadgets.append(
                    (gadget, job_input_state, output_state, job_metadata)
                )
                success_inc = 1

            if progress:
                progress.update(task_id, advance=1, success=success_inc)

        pool.close()
        pool.join()

        if jobs:
            _log(
                f"TOTAL construction time: {total_construct_time:.2f}s, TOTAL solver time: {total_solve_time:.2f}s"
            )

        # Phase 2.5: Compress tar files if smt2_dump_dir is set
        if jobs and "smt2_dump_dir" in jobs[0][6]:
            smt2_dump_dir = jobs[0][6]["smt2_dump_dir"]
            stage_idx = jobs[0][6]["stage_idx"]
            for filename in os.listdir(smt2_dump_dir):
                if filename.startswith(f"stage{stage_idx}_") and filename.endswith(
                    ".tar"
                ):
                    tar_path = os.path.join(smt2_dump_dir, filename)
                    zst_path = tar_path + ".zst"
                    cctx = zstd.ZstdCompressor()
                    with open(tar_path, "rb") as f_in:
                        with open(zst_path, "wb") as f_out:
                            cctx.copy_stream(f_in, f_out)
                    os.remove(tar_path)

        # Sort validated gadgets by a deterministic key to ensure consistent
        # dict insertion order downstream, regardless of imap_unordered return order.
        validated_gadgets.sort(
            key=lambda x: (
                x[3]["parent_path"],
                x[1].as_tuple(),
                x[2].as_tuple(),
            )
        )

        return validated_gadgets

    def _enumerate_single_input_instructions(
        self, reg_name: str = "input"
    ) -> list[InstructionSpec]:
        """
        Generate single-input instruction templates with symbolic immediates or control vectors.
        Z3 will solve for the concrete values.

        Single-input means: operates on ONE of our vectors (top OR bottom),
        even if it takes additional operands like control vectors.
        """
        if self.vm == vector_machine.AVX2 and self.prim_type == primitive_type.i32:
            input_reg = reg_name
            unique_id = id(input_reg)

            # Permute within 128-bit lanes using immediate
            permute_ps = InstructionSpec(
                "_mm256_permute_ps",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute_ps_{unique_id}", 8),
                },
            )

            # Permute 64-bit chunks across full register
            permute4x64 = InstructionSpec(
                "_mm256_permute4x64_epi64",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute4x64_{unique_id}", 8),
                },
            )

            # Variable permute across all lanes (most powerful)
            permutexvar = InstructionSpec(
                "_mm256_permutexvar_epi32",
                {
                    "a": input_reg,
                    "op_idx": SymbolicPlaceholder(f"ctrl_permutexvar_{unique_id}", 256),
                },
            )

            # Variable permute within 128-bit lanes
            permutevar_ps = InstructionSpec(
                "_mm256_permutevar_ps",
                {
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_permutevar_ps_{unique_id}", 256),
                },
            )

            return [permute_ps, permute4x64, permutexvar, permutevar_ps]

        if self.vm == vector_machine.AVX2 and self.prim_type == primitive_type.i64:
            input_reg = reg_name
            unique_id = id(input_reg)

            # Permute within 128-bit lanes using immediate (for pd/64-bit doubles)
            permute_pd = InstructionSpec(
                "_mm256_permute_pd",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute_pd_{unique_id}", 8),
                },
            )

            # Permute 64-bit elements across full register
            permute4x64 = InstructionSpec(
                "_mm256_permute4x64_epi64",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute4x64_{unique_id}", 8),
                },
            )

            # Variable permute across all lanes (most powerful for 64-bit)
            permutexvar = InstructionSpec(
                "_mm256_permutexvar_epi64",
                {
                    "a": input_reg,
                    "op_idx": SymbolicPlaceholder(f"ctrl_permutexvar_{unique_id}", 256),
                },
            )

            # Variable permute within 128-bit lanes (for pd)
            permutevar_pd = InstructionSpec(
                "_mm256_permutevar_pd",
                {
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_permutevar_pd_{unique_id}", 256),
                },
            )

            return [permute_pd, permute4x64, permutexvar, permutevar_pd]

        if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i64:
            input_reg = reg_name
            unique_id = id(input_reg)

            # --- Unmasked ---
            # Cross-lane variable permute (most powerful)
            permutexvar = InstructionSpec(
                "_mm512_permutexvar_epi64",
                {
                    "a": input_reg,
                    "op_idx": SymbolicPlaceholder(f"ctrl_permutexvar_{unique_id}", 512),
                },
            )
            # In-lane permute with immediate
            permute_pd = InstructionSpec(
                "_mm512_permute_pd",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute_pd_{unique_id}", 8),
                },
            )
            # In-lane variable permute
            permutevar_pd = InstructionSpec(
                "_mm512_permutevar_pd",
                {
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_permutevar_pd_{unique_id}", 512),
                },
            )

            # --- Masked ---
            mask_permutexvar = InstructionSpec(
                "_mm512_mask_permutexvar_epi64",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permutexvar_{unique_id}", 8),
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_m_permutexvar_{unique_id}", 512
                    ),
                    "a": input_reg,
                },
            )
            mask_permute_pd = InstructionSpec(
                "_mm512_mask_permute_pd",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permute_pd_{unique_id}", 8),
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_m_permute_pd_{unique_id}", 8),
                },
            )
            mask_permutevar_pd = InstructionSpec(
                "_mm512_mask_permutevar_pd",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permutevar_pd_{unique_id}", 8),
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_m_permutevar_pd_{unique_id}", 512),
                },
            )

            return [
                permutexvar,
                permute_pd,
                permutevar_pd,
                mask_permutexvar,
                mask_permute_pd,
                mask_permutevar_pd,
            ]

        if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i32:
            input_reg = reg_name
            unique_id = id(input_reg)

            # --- Unmasked ---
            permutexvar = InstructionSpec(
                "_mm512_permutexvar_epi32",
                {
                    "a": input_reg,
                    "op_idx": SymbolicPlaceholder(f"ctrl_permutexvar_{unique_id}", 512),
                },
            )
            permute_ps = InstructionSpec(
                "_mm512_permute_ps",
                {
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_permute_ps_{unique_id}", 8),
                },
            )
            permutevar_ps = InstructionSpec(
                "_mm512_permutevar_ps",
                {
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_permutevar_ps_{unique_id}", 512),
                },
            )

            # --- Masked (16-bit k-mask for 16 elements) ---
            mask_permutexvar = InstructionSpec(
                "_mm512_mask_permutexvar_epi32",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permutexvar_{unique_id}", 16),
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_m_permutexvar_{unique_id}", 512
                    ),
                    "a": input_reg,
                },
            )
            mask_permute_ps = InstructionSpec(
                "_mm512_mask_permute_ps",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permute_ps_{unique_id}", 16),
                    "a": input_reg,
                    "imm8": SymbolicPlaceholder(f"imm8_m_permute_ps_{unique_id}", 8),
                },
            )
            mask_permutevar_ps = InstructionSpec(
                "_mm512_mask_permutevar_ps",
                {
                    "src": input_reg,
                    "k": SymbolicPlaceholder(f"k_mask_permutevar_ps_{unique_id}", 16),
                    "a": input_reg,
                    "b": SymbolicPlaceholder(f"ctrl_m_permutevar_ps_{unique_id}", 512),
                },
            )

            return [
                permutexvar,
                permute_ps,
                permutevar_ps,
                mask_permutexvar,
                mask_permute_ps,
                mask_permutevar_ps,
            ]

        raise NotImplementedError(
            f"Single-input instructions not implemented for {self.vm} and {self.prim_type}"
        )

    def _enumerate_dual_input_instructions(
        self, reg1: str = "top", reg2: str = "bottom"
    ) -> list[InstructionSpec]:
        """
        Generate dual-input instruction templates with symbolic immediates.
        Z3 will solve for the concrete immediate values.
        """
        if self.vm == vector_machine.AVX2 and self.prim_type == primitive_type.i32:
            unique_id = f"{id(reg1)}_{id(reg2)}"

            # Shuffle: select elements from both inputs within 128-bit lanes
            shuffle_ps = InstructionSpec(
                "_mm256_shuffle_ps",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuffle_{unique_id}", 8),
                },
            )

            # Unpack low: interleave low elements from both inputs
            unpacklo = InstructionSpec(
                "_mm256_unpacklo_epi32",
                {"a": reg1, "b": reg2},
            )

            # Unpack high: interleave high elements from both inputs
            unpackhi = InstructionSpec(
                "_mm256_unpackhi_epi32",
                {"a": reg1, "b": reg2},
            )

            # Permute 128-bit lanes between two registers
            permute2x128 = InstructionSpec(
                "_mm256_permute2x128_si256",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_perm2x128_{unique_id}", 8),
                },
            )

            # Blend: select elements from either input based on mask
            blend_ps = InstructionSpec(
                "_mm256_blend_ps",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_blend_{unique_id}", 8),
                },
            )

            # Align right: concatenate and shift
            alignr = InstructionSpec(
                "_mm256_alignr_epi32",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_alignr_{unique_id}", 8),
                },
            )

            return [shuffle_ps, unpacklo, unpackhi, permute2x128, blend_ps, alignr]

        if self.vm == vector_machine.AVX2 and self.prim_type == primitive_type.i64:
            unique_id = f"{id(reg1)}_{id(reg2)}"

            # Shuffle: select 64-bit elements from both inputs within 128-bit lanes (pd variant)
            shuffle_pd = InstructionSpec(
                "_mm256_shuffle_pd",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuffle_{unique_id}", 8),
                },
            )

            # Unpack low: interleave low 64-bit elements from both inputs
            unpacklo = InstructionSpec(
                "_mm256_unpacklo_epi64",
                {"a": reg1, "b": reg2},
            )

            # Unpack high: interleave high 64-bit elements from both inputs
            unpackhi = InstructionSpec(
                "_mm256_unpackhi_epi64",
                {"a": reg1, "b": reg2},
            )

            # Permute 128-bit lanes between two registers (works for any element size)
            permute2x128 = InstructionSpec(
                "_mm256_permute2x128_si256",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_perm2x128_{unique_id}", 8),
                },
            )

            # Blend: select 64-bit elements from either input based on mask (pd variant)
            blend_pd = InstructionSpec(
                "_mm256_blend_pd",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_blend_{unique_id}", 8),
                },
            )

            # Align right: concatenate and shift by 64-bit elements
            alignr = InstructionSpec(
                "_mm256_alignr_epi64",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_alignr_{unique_id}", 8),
                },
            )

            return [shuffle_pd, unpacklo, unpackhi, permute2x128, blend_pd, alignr]

        if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i64:
            unique_id = f"{id(reg1)}_{id(reg2)}"

            # --- Unmasked ---
            # Two-source variable permute (most powerful AVX512 instruction)
            permutex2var = InstructionSpec(
                "_mm512_permutex2var_epi64",
                {
                    "a": reg1,
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_permutex2var_{unique_id}", 512
                    ),
                    "b": reg2,
                },
            )
            # Shuffle pd within 128-bit lanes
            shuffle_pd = InstructionSpec(
                "_mm512_shuffle_pd",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuffle_pd_{unique_id}", 8),
                },
            )
            # Unpack low/high
            unpacklo = InstructionSpec("_mm512_unpacklo_epi64", {"a": reg1, "b": reg2})
            unpackhi = InstructionSpec("_mm512_unpackhi_epi64", {"a": reg1, "b": reg2})
            # 128-bit lane shuffle
            shuffle_i32x4 = InstructionSpec(
                "_mm512_shuffle_i32x4",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuf_i32x4_{unique_id}", 8),
                },
            )
            # Align right
            alignr = InstructionSpec(
                "_mm512_alignr_epi64",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_alignr_{unique_id}", 8),
                },
            )

            # --- Masked ---
            mask_permutex2var = InstructionSpec(
                "_mm512_mask_permutex2var_epi64",
                {
                    "a": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_permutex2var_{unique_id}", 8),
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_m_permutex2var_{unique_id}", 512
                    ),
                    "b": reg2,
                },
            )
            mask_shuffle_pd = InstructionSpec(
                "_mm512_mask_shuffle_pd",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_shuffle_pd_{unique_id}", 8),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_shuffle_pd_{unique_id}", 8),
                },
            )
            mask_unpacklo = InstructionSpec(
                "_mm512_mask_unpacklo_epi64",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_unpacklo_{unique_id}", 8),
                    "a": reg1,
                    "b": reg2,
                },
            )
            mask_unpackhi = InstructionSpec(
                "_mm512_mask_unpackhi_epi64",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_unpackhi_{unique_id}", 8),
                    "a": reg1,
                    "b": reg2,
                },
            )
            mask_shuffle_i32x4 = InstructionSpec(
                "_mm512_mask_shuffle_i32x4",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_shuf_i32x4_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_shuf_i32x4_{unique_id}", 8),
                },
            )
            mask_alignr = InstructionSpec(
                "_mm512_mask_alignr_epi64",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_alignr_{unique_id}", 8),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_alignr_{unique_id}", 8),
                },
            )

            return [
                permutex2var,
                shuffle_pd,
                unpacklo,
                unpackhi,
                shuffle_i32x4,
                alignr,
                mask_permutex2var,
                mask_shuffle_pd,
                mask_unpacklo,
                mask_unpackhi,
                mask_shuffle_i32x4,
                mask_alignr,
            ]

        if self.vm == vector_machine.AVX512 and self.prim_type == primitive_type.i32:
            unique_id = f"{id(reg1)}_{id(reg2)}"

            # --- Unmasked ---
            permutex2var = InstructionSpec(
                "_mm512_permutex2var_epi32",
                {
                    "a": reg1,
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_permutex2var_{unique_id}", 512
                    ),
                    "b": reg2,
                },
            )
            shuffle_ps = InstructionSpec(
                "_mm512_shuffle_ps",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuffle_ps_{unique_id}", 8),
                },
            )
            unpacklo = InstructionSpec("_mm512_unpacklo_epi32", {"a": reg1, "b": reg2})
            unpackhi = InstructionSpec("_mm512_unpackhi_epi32", {"a": reg1, "b": reg2})
            shuffle_i32x4 = InstructionSpec(
                "_mm512_shuffle_i32x4",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_shuf_i32x4_{unique_id}", 8),
                },
            )
            alignr = InstructionSpec(
                "_mm512_alignr_epi32",
                {
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_alignr_{unique_id}", 8),
                },
            )

            # --- Masked (16-bit k-mask for 16 elements) ---
            mask_permutex2var = InstructionSpec(
                "_mm512_mask_permutex2var_epi32",
                {
                    "a": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_permutex2var_{unique_id}", 16),
                    "op_idx": SymbolicPlaceholder(
                        f"ctrl_m_permutex2var_{unique_id}", 512
                    ),
                    "b": reg2,
                },
            )
            mask_shuffle_ps = InstructionSpec(
                "_mm512_mask_shuffle_ps",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_shuffle_ps_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_shuffle_ps_{unique_id}", 8),
                },
            )
            mask_unpacklo = InstructionSpec(
                "_mm512_mask_unpacklo_epi32",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_unpacklo_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                },
            )
            mask_unpackhi = InstructionSpec(
                "_mm512_mask_unpackhi_epi32",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_unpackhi_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                },
            )
            mask_shuffle_i32x4 = InstructionSpec(
                "_mm512_mask_shuffle_i32x4",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_shuf_i32x4_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_shuf_i32x4_{unique_id}", 8),
                },
            )
            mask_alignr = InstructionSpec(
                "_mm512_mask_alignr_epi32",
                {
                    "src": reg1,
                    "k": SymbolicPlaceholder(f"k_mask_alignr_{unique_id}", 16),
                    "a": reg1,
                    "b": reg2,
                    "imm8": SymbolicPlaceholder(f"imm8_m_alignr_{unique_id}", 8),
                },
            )

            return [
                permutex2var,
                shuffle_ps,
                unpacklo,
                unpackhi,
                shuffle_i32x4,
                alignr,
                mask_permutex2var,
                mask_shuffle_ps,
                mask_unpacklo,
                mask_unpackhi,
                mask_shuffle_i32x4,
                mask_alignr,
            ]

        raise NotImplementedError(
            f"Dual-input instructions not implemented for {self.vm} and {self.prim_type}"
        )


# ---------------------------------------------------------------------------
# Pipeline support: geometric-series thresholds
# ---------------------------------------------------------------------------

_BATCH_THRESHOLDS = (0.5, 0.75, 0.875, 1.0)


@dataclass
class StageTracker:
    """Tracks progress and accumulated results for one pipelined stage."""

    stage_idx: int
    stage_pairs: list  # comparison pairs from bitonic_sorter

    # Job tracking
    total_jobs_submitted: int = 0
    completed_jobs: int = 0

    # Result accumulation (raw results awaiting batch processing)
    unprocessed_results: list = field(default_factory=list)

    # Incremental Phase 3/4 state
    gadgets_by_transition: dict = field(default_factory=dict)
    nodes_by_parent: dict = field(default_factory=dict)
    unique_outputs: dict = field(default_factory=dict)

    # Prevent duplicate forwarding to next stage
    forwarded_output_tuples: set = field(default_factory=set)

    # Threshold tracking
    next_threshold_index: int = 0

    # Lifecycle
    all_inputs_received: bool = False  # True when prior stage finalizes
    finalized: bool = False  # True when all results processed

    # Total expected jobs (may increase as prior stage discovers more outputs)
    total_jobs_expected: int | None = None

    # Progress
    progress_task_id: int | None = None

    @property
    def is_complete(self) -> bool:
        """True when all submitted jobs are done AND no more inputs coming."""
        return (
            self.all_inputs_received
            and self.total_jobs_expected is not None
            and self.completed_jobs >= self.total_jobs_expected
        )

    @property
    def completion_fraction(self) -> float:
        if self.total_jobs_submitted == 0:
            return 0.0
        return self.completed_jobs / self.total_jobs_submitted

    def should_process_batch(self) -> bool:
        """Check if we've crossed the next geometric threshold."""
        if self.next_threshold_index >= len(_BATCH_THRESHOLDS):
            return False
        return self.completion_fraction >= _BATCH_THRESHOLDS[self.next_threshold_index]


class BitonicSuperVectorizer:
    """Super-optimizer for bitonic sorting networks using Z3-based gadget synthesis."""

    def __init__(
        self,
        num_vecs: int,
        prim_type: primitive_type,
        vm: vector_machine,
        smt2_dump_dir: str | None = None,
    ):
        self.num_vecs = num_vecs
        self.prim_type = prim_type
        self.vm = vm
        self.smt2_dump_dir = smt2_dump_dir

        # Calculate total elements and elements per vector
        self.elements_per_vector = width_dict[vm] // int(prim_type.value[0])
        self.total_elements = num_vecs * self.elements_per_vector

        # Initialize bitonic sorter to get comparison pairs per stage
        self.bitonic_sorter = BitonicSorter(self.total_elements)

        # Initialize gadget synthesizer
        self.synthesizer = GadgetSynthesizer(vm, prim_type)
        # Single-use contract: one synthesis run per instance to avoid mutable
        # run-state carryover between calls.
        self._synthesis_started = False

        print(
            f"BitonicSuperVectorizer: {num_vecs} x {vm.name} vectors, {self.elements_per_vector} x {prim_type.name} elements per vector, {self.total_elements} total elements, {len(self.bitonic_sorter.stages)} stages"
        )

    def _create_initial_state(self) -> VectorState:
        """
        Create initial vector state based on first stage pairs.

        The initial state is constructed from the FIRST stage's comparison pairs.
        For each pair (a, b), element a goes to top vector and element b goes to bottom.
        This ensures the first stage requires no permutation (0-instruction gadget).
        """
        first_stage_pairs = self.bitonic_sorter.stages[0]

        # Initialize empty lists for top and bottom vectors
        top = []
        bottom = []

        # For each comparison pair in the first stage:
        # - First element goes to top vector
        # - Second element goes to bottom vector
        for pair in first_stage_pairs:
            top.append(pair[0])
            bottom.append(pair[1])

        # Verify we have the expected number of elements
        assert (
            len(top) == self.elements_per_vector
        ), f"Expected {self.elements_per_vector} elements in top, got {len(top)}"
        assert (
            len(bottom) == self.elements_per_vector
        ), f"Expected {self.elements_per_vector} elements in bottom, got {len(bottom)}"

        return VectorState(top=top, bottom=bottom)

    def build_solution_tree(
        self,
        depth_limit: int | None = None,
        gadget_depth: int = 1,
        natural_order: bool = False,
        max_unique_outputs: int = 3,
        checkpoint_dir: str | None = None,
        resume_data: dict | None = None,
        pipeline: bool = True,
        max_workers: int | None = None,
        max_tasks_per_child: int | None = 1000,
    ) -> tuple[list[SolutionNode], bool]:
        """
        Iteratively explore all stage transitions to build solution tree.
        Returns (root_nodes, all_stages_complete).

        Args:
            depth_limit: Maximum stage depth to explore (inclusive). If None, all stages are explored.
            gadget_depth: Maximum instruction depth per gadget (1-3, default 1).
            natural_order: If True, append a final stage that restores natural
                element order (1..N in top, N+1..2N in bottom) for memory writeback.
            max_unique_outputs: Number of smallest unique output states to enumerate
                per template for deterministic diversity. Default: 3.
            checkpoint_dir: Directory to save checkpoints after each stage completes.
                If None, no checkpoints are saved.
            resume_data: If provided, resume from a previous checkpoint. Expected
                keys: ``per_stage_data`` and ``last_completed_stage``.
            pipeline: If True (default), use pipelined stage processing where
                stage N+1 starts before stage N completes. If False, use the
                sequential path with a hard barrier between stages.
            max_workers: Maximum number of worker processes. If None, defaults
                to os.cpu_count().
            max_tasks_per_child: Maximum tasks per worker process before recycling.
                Limits memory growth in long runs. None disables recycling.

        Returns:
            Tuple of (root nodes, all_stages_complete) where all_stages_complete
            is True if every expected stage was solved successfully.

        Note:
            This method is single-use per ``BitonicSuperVectorizer`` instance.
            Create a new instance for each synthesis run.
        """
        if self._synthesis_started:
            raise RuntimeError(
                "BitonicSuperVectorizer instances are single-use. "
                "Create a new instance for each synthesis run."
            )
        self._synthesis_started = True
        self._max_workers = max_workers
        self._max_tasks_per_child = max_tasks_per_child

        # Inject natural-order stage if requested
        self._natural_order_stage = None
        if natural_order:
            natural_pairs = [
                (i + 1, self.elements_per_vector + i + 1)
                for i in range(self.elements_per_vector)
            ]
            new_stage = max(self.bitonic_sorter.stages.keys()) + 1
            self.bitonic_sorter.stages[new_stage] = natural_pairs
            self._natural_order_stage = new_stage

        # Track which stages produce at least one valid gadget
        self._stages_completed: set[int] = set()

        initial_state = self._create_initial_state()
        # Pre-compute all candidates once - they're independent of stage/input state
        all_candidates = self.synthesizer.precompute_all_candidates(gadget_depth)

        # Build checkpoint config if checkpoint_dir is provided
        checkpoint_config = None
        if checkpoint_dir is not None:
            try:
                from checkpoint import CheckpointConfig
            except ImportError:
                from .checkpoint import CheckpointConfig  # type: ignore[no-redef]
            checkpoint_config = CheckpointConfig(
                num_vecs=self.num_vecs,
                vm=self.vm.name,
                prim_type=self.prim_type.name,
                gadget_depth=gadget_depth,
                natural_order=natural_order,
                max_unique_outputs=max_unique_outputs,
                depth_limit=depth_limit,
            )

        # Start with empty parent path for the root
        input_states_with_context = [(initial_state, ())]

        # Dispatch to selected execution path
        build_fn = (
            self._build_tree_pipelined if pipeline else self._build_tree_sequential
        )
        nodes_by_path = build_fn(
            input_states_with_context,
            depth_limit,
            all_candidates,
            max_unique_outputs=max_unique_outputs,
            checkpoint_dir=checkpoint_dir,
            checkpoint_config=checkpoint_config,
            resume_data=resume_data,
        )

        # Determine expected stages
        total_stages = len(self.bitonic_sorter.stages)
        expected_stages = total_stages
        if depth_limit is not None:
            expected_stages = min(depth_limit, total_stages)

        all_stages_complete = self._stages_completed == set(range(expected_stages))

        # Return root nodes (those with empty parent path)
        return nodes_by_path.get((), []), all_stages_complete

    def _process_single_stage(
        self,
        input_states_with_context: list[tuple[VectorState, tuple]],
        stage_idx: int,
        all_candidates: list[tuple],
        max_unique_outputs: int = 3,
        progress: SuccessProgress | None = None,
        task_id=None,
        max_workers: int | None = None,
        max_tasks_per_child: int | None = 1000,
    ) -> tuple[dict[tuple, list[SolutionNode]], dict[tuple, VectorState]]:
        """Process one stage: validate gadgets, group, create nodes.

        Includes Phase 1 (collect candidates), Phase 2 (validate in parallel),
        Phase 3 (group by transition), and Phase 4 (create nodes, track unique
        outputs). Children are NOT wired by this method.

        Args:
            input_states_with_context: List of (input_state, parent_path) tuples
                where parent_path tracks the chain of previous gadgets leading
                to this state.
            stage_idx: Current stage index.
            all_candidates: Pre-computed list of (top_seq, bottom_seq) instruction
                templates, independent of stage/input state.
            max_unique_outputs: Number of smallest unique output states to
                enumerate per template for deterministic diversity.
            progress: Optional SuccessProgress instance for live display.
            task_id: Task ID within the progress instance for this stage.

        Returns:
            Tuple of (nodes_by_parent, unique_outputs) where nodes_by_parent maps
            parent_path to list of SolutionNodes and unique_outputs maps output
            state tuples to VectorState instances. Children on the nodes are empty.
        """

        def _log(msg: str):
            if progress:
                progress.console.print(msg)
            else:
                print(msg)

        stage_pairs = self.bitonic_sorter.stages[stage_idx]

        _log(
            f"Stage {stage_idx}: Collecting candidates for {len(input_states_with_context)} nodes from the previous stage"
        )

        # Phase 1: Collect all candidate gadgets for entire stage
        all_jobs = []

        for input_state, parent_path in input_states_with_context:
            metadata = {
                "input_state": input_state,
                "parent_path": parent_path,
                "stage_idx": stage_idx,
                "max_unique_outputs": max_unique_outputs,
            }
            if (
                self._natural_order_stage is not None
                and stage_idx == self._natural_order_stage
            ):
                metadata["allow_any_lane_order"] = False
            if self.smt2_dump_dir:
                metadata["smt2_dump_dir"] = self.smt2_dump_dir

            # Enrich pre-computed candidates with per-state metadata
            for top_seq, bottom_seq in all_candidates:
                all_jobs.append(
                    (
                        top_seq,
                        bottom_seq,
                        input_state,
                        stage_pairs,
                        self.vm,
                        self.prim_type,
                        metadata,
                    )
                )

        _log(f"Stage {stage_idx}: Generated {len(all_jobs)} candidates to validate")

        # Set task total now that we know the job count
        if progress and task_id is not None:
            progress.update(task_id, total=len(all_jobs))

        # Phase 2: Validate all candidates in parallel with progress reporting
        validated_gadgets = self.synthesizer._validate_gadgets(
            all_jobs,
            progress=progress,
            task_id=task_id,
            max_workers=max_workers,
            max_tasks_per_child=max_tasks_per_child,
        )

        _log(
            f"Stage {stage_idx}: Validated {len(validated_gadgets)}/{len(all_jobs)} gadgets"
        )

        # Phase 3: Group gadgets by (parent_path, input_state, output_state)
        # This prunes semantically equivalent gadgets - we only need one path per unique state transition
        # Key: (parent_path, input_state_tuple, output_state_tuple) -> list of gadgets
        gadgets_by_transition: dict[tuple, list[PermutationGadget]] = {}

        # Collect all validated gadgets
        for gadget, input_state, output_state, metadata in validated_gadgets:
            parent_path = metadata["parent_path"]
            key = (parent_path, input_state.as_tuple(), output_state.as_tuple())
            if key not in gadgets_by_transition:
                gadgets_by_transition[key] = []
            gadgets_by_transition[key].append(gadget)

        _log(
            f"Stage {stage_idx}: Pruned to {len(gadgets_by_transition)} unique state transitions"
        )

        # Post-validation statistics: analyze output_state sharing across input_states
        output_to_inputs: dict[tuple, set[tuple]] = {}
        for _parent_path, input_tuple, output_tuple in gadgets_by_transition.keys():
            if output_tuple not in output_to_inputs:
                output_to_inputs[output_tuple] = set()
            output_to_inputs[output_tuple].add(input_tuple)

        # Phase 4: Create nodes from grouped gadgets
        nodes_by_parent: dict[tuple, list[SolutionNode]] = {}
        # Deduplicate: track one representative per unique output_state
        unique_outputs: dict[tuple, VectorState] = {}

        # Sort transition keys for deterministic iteration order
        sorted_transitions = sorted(gadgets_by_transition.items(), key=lambda x: x[0])

        for (
            parent_path,
            input_tuple,
            output_tuple,
        ), gadgets in sorted_transitions:
            # Reconstruct VectorState from tuples
            input_state = VectorState(
                top=list(input_tuple[0]), bottom=list(input_tuple[1])
            )
            output_state = VectorState(
                top=list(output_tuple[0]), bottom=list(output_tuple[1])
            )

            # Sort gadgets within each group for deterministic ordering
            gadgets.sort(key=lambda g: g.sort_key())

            # Create node with all equivalent gadgets
            node = SolutionNode(
                stage=stage_idx,
                input_state=input_state,
                output_state=output_state,
                gadgets=gadgets,
                children=[],
            )

            # Add to nodes_by_parent
            if parent_path not in nodes_by_parent:
                nodes_by_parent[parent_path] = []
            nodes_by_parent[parent_path].append(node)

            # Track unique output_states for deduplication
            if output_tuple not in unique_outputs:
                unique_outputs[output_tuple] = output_state

        return nodes_by_parent, unique_outputs

    # ------------------------------------------------------------------
    # Pipelined helpers
    # ------------------------------------------------------------------

    def _make_jobs_for_inputs(
        self,
        input_states_with_context: list[tuple[VectorState, tuple]],
        stage_idx: int,
        stage_pairs: list,
        all_candidates: list[tuple],
        max_unique_outputs: int,
    ) -> list[tuple]:
        """Create validation jobs for a set of input states (Phase 1)."""
        jobs = []
        for input_state, parent_path in input_states_with_context:
            metadata = {
                "input_state": input_state,
                "parent_path": parent_path,
                "stage_idx": stage_idx,
                "max_unique_outputs": max_unique_outputs,
            }
            if (
                self._natural_order_stage is not None
                and stage_idx == self._natural_order_stage
            ):
                metadata["allow_any_lane_order"] = False
            if self.smt2_dump_dir:
                metadata["smt2_dump_dir"] = self.smt2_dump_dir

            for top_seq, bottom_seq in all_candidates:
                jobs.append(
                    (
                        top_seq,
                        bottom_seq,
                        input_state,
                        stage_pairs,
                        self.vm,
                        self.prim_type,
                        metadata,
                    )
                )
        return jobs

    @staticmethod
    def _submit_stage_jobs(
        pool: Pool,
        tracker: StageTracker,
        jobs: list[tuple],
        pending_count: list[int],
        completion_queue: queue_mod.Queue,
    ) -> None:
        """Submit jobs to the pool and register callbacks for completion."""
        for job in jobs:
            pool.apply_async(
                _validate_gadget_worker,
                (job,),
                callback=lambda result, s=tracker.stage_idx: completion_queue.put(
                    (s, result, None)
                ),
                error_callback=lambda exc, s=tracker.stage_idx: completion_queue.put(
                    (s, None, exc)
                ),
            )
        tracker.total_jobs_submitted += len(jobs)
        pending_count[0] += len(jobs)

    @staticmethod
    def _process_batch(tracker: StageTracker) -> list[tuple[VectorState, tuple]]:
        """Drain unprocessed results, group by transition, return NEW unique outputs.

        This is the incremental equivalent of _process_single_stage phases 3-4.
        Results are accumulated into the tracker's gadgets_by_transition,
        nodes_by_parent, and unique_outputs dicts.

        Returns list of (output_state, canonical_parent_path) for outputs
        not yet forwarded to the next stage.
        """
        results = tracker.unprocessed_results
        tracker.unprocessed_results = []

        # Phase 3 (incremental): group by transition
        for gadget_results, job_input_state, job_metadata, _ct, _st in results:
            parent_path = job_metadata["parent_path"]
            for gadget, output_state in gadget_results:
                key = (
                    parent_path,
                    job_input_state.as_tuple(),
                    output_state.as_tuple(),
                )
                if key not in tracker.gadgets_by_transition:
                    tracker.gadgets_by_transition[key] = []
                tracker.gadgets_by_transition[key].append(gadget)

                # Phase 4 (incremental): track unique outputs
                out_tuple = output_state.as_tuple()
                if out_tuple not in tracker.unique_outputs:
                    tracker.unique_outputs[out_tuple] = output_state

        # Identify NEW outputs not yet forwarded
        new_outputs = []
        for out_tuple, state in tracker.unique_outputs.items():
            if out_tuple not in tracker.forwarded_output_tuples:
                tracker.forwarded_output_tuples.add(out_tuple)
                new_outputs.append((state, ("canonical", out_tuple)))

        tracker.next_threshold_index += 1
        return new_outputs

    @staticmethod
    def _finalize_stage(tracker: StageTracker) -> None:
        """Sort transitions and gadgets for deterministic output.

        Must be called exactly once after all results are processed.
        Rebuilds nodes_by_parent from the accumulated gadgets_by_transition
        in sorted order, matching the sequential path's output.
        """
        tracker.nodes_by_parent = {}

        sorted_transitions = sorted(
            tracker.gadgets_by_transition.items(), key=lambda x: x[0]
        )

        for (parent_path, input_tuple, output_tuple), gadgets in sorted_transitions:
            input_state = VectorState(
                top=list(input_tuple[0]), bottom=list(input_tuple[1])
            )
            output_state = VectorState(
                top=list(output_tuple[0]), bottom=list(output_tuple[1])
            )

            gadgets.sort(key=lambda g: g.sort_key())

            node = SolutionNode(
                stage=tracker.stage_idx,
                input_state=input_state,
                output_state=output_state,
                gadgets=gadgets,
                children=[],
            )

            if parent_path not in tracker.nodes_by_parent:
                tracker.nodes_by_parent[parent_path] = []
            tracker.nodes_by_parent[parent_path].append(node)

        tracker.finalized = True

    @staticmethod
    def _cancel_downstream_stages(
        from_stage: int,
        effective_limit: int,
        cancelled_stages: set[int],
        pending_count: list[int],
        trackers: dict,
        progress: SuccessProgress,
        stage_task_ids: dict[int, int],
    ) -> None:
        """Mark downstream stages as cancelled and adjust pending count."""
        for s in range(from_stage + 1, effective_limit):
            if s in trackers:
                tracker = trackers[s]
                remaining = tracker.total_jobs_submitted - tracker.completed_jobs
                pending_count[0] -= remaining
                cancelled_stages.add(s)
            if s in stage_task_ids:
                progress.update(stage_task_ids[s], visible=False)

    def _build_tree_sequential(
        self,
        initial_inputs: list[tuple[VectorState, tuple]],
        depth_limit: int | None,
        all_candidates: list[tuple],
        max_unique_outputs: int = 3,
        checkpoint_dir: str | None = None,
        checkpoint_config=None,
        resume_data: dict | None = None,
    ) -> dict[tuple, list[SolutionNode]]:
        """Build the solution tree iteratively using a forward pass then backward wiring.

        The forward pass iterates through stages sequentially, processing each
        stage with ``_process_single_stage``. The backward pass wires children
        from later stages back into earlier stage nodes.

        Args:
            initial_inputs: List of (input_state, parent_path) tuples for stage 0.
            depth_limit: Maximum stage depth to explore. If None, all stages.
            all_candidates: Pre-computed instruction templates.
            max_unique_outputs: Diversity parameter per template.
            checkpoint_dir: Directory to save checkpoints after each stage.
            checkpoint_config: Configuration object for checkpoint saving.
            resume_data: If provided, resume from a previous checkpoint. Expected
                keys: ``per_stage_data`` and ``last_completed_stage``.

        Returns:
            Dictionary mapping parent_path to list of SolutionNodes for stage 0.
        """
        per_stage_data = {}  # stage_idx -> (nodes_by_parent, unique_outputs)

        # Determine effective limit
        effective_limit = len(self.bitonic_sorter.stages)
        if depth_limit is not None:
            effective_limit = min(depth_limit, effective_limit)

        # Handle resume: skip already-completed stages
        start_stage = 0
        current_inputs = initial_inputs
        if resume_data is not None:
            per_stage_data = resume_data["per_stage_data"]
            start_stage = resume_data["last_completed_stage"] + 1
            # Reconstruct current_inputs from last completed stage
            last_stage = resume_data["last_completed_stage"]
            _, unique_outputs = per_stage_data[last_stage]
            current_inputs = [
                (state, ("canonical", out_tuple))
                for out_tuple, state in unique_outputs.items()
            ]

        # Create persistent multi-stage progress display
        progress = SuccessProgress.create(
            width=60,
            success_style="green",
            attempt_style="yellow",
            success_label="Valid",
        )
        progress.enable_memory_monitor()

        # Pre-create one task per stage
        stage_task_ids = {}
        for s in range(effective_limit):
            if s < start_stage:
                # Resumed stages: show as completed
                tid = progress.add_task(
                    f"Stage {s} (resumed)",
                    total=1,
                    completed=1,
                    successes=1,
                )
                stage_task_ids[s] = tid
            else:
                # Pending stages: indeterminate until started
                tid = progress.add_task(
                    f"Stage {s}",
                    total=None,
                    start=False,
                    successes=0,
                )
                stage_task_ids[s] = tid

        # Forward pass: iterate through stages
        with progress:
            if resume_data is not None:
                progress.console.print(
                    f"Resuming from stage {start_stage} with {len(current_inputs)} inputs"
                )

            for stage_idx in range(start_stage, effective_limit):
                task_id = stage_task_ids[stage_idx]
                progress.start_task(task_id)

                nodes_by_parent, unique_outputs = self._process_single_stage(
                    current_inputs,
                    stage_idx,
                    all_candidates,
                    max_unique_outputs,
                    progress=progress,
                    task_id=task_id,
                    max_workers=self._max_workers,
                    max_tasks_per_child=self._max_tasks_per_child,
                )
                per_stage_data[stage_idx] = (nodes_by_parent, unique_outputs)

                if nodes_by_parent:
                    self._stages_completed.add(stage_idx)

                # Checkpoint save point
                if checkpoint_dir is not None and checkpoint_config is not None:
                    try:
                        from checkpoint import save_checkpoint, checkpoint_filename
                    except ImportError:
                        from .checkpoint import save_checkpoint, checkpoint_filename  # type: ignore[no-redef]
                    ckpt_path = os.path.join(
                        checkpoint_dir,
                        checkpoint_filename(
                            checkpoint_config.num_vecs,
                            checkpoint_config.vm,
                            checkpoint_config.prim_type,
                            checkpoint_config.natural_order,
                            stage_idx,
                        ),
                    )
                    save_checkpoint(
                        ckpt_path,
                        checkpoint_config,
                        per_stage_data,
                        sorted(self._stages_completed),
                        stage_idx,
                    )

                # Log deduplication and prepare inputs for next stage
                if unique_outputs:
                    total_next = sum(len(nodes) for nodes in nodes_by_parent.values())
                    progress.console.print(
                        f"Stage {stage_idx}: Deduplicated {total_next} -> {len(unique_outputs)} next-stage inputs"
                    )
                    current_inputs = [
                        (state, ("canonical", out_tuple))
                        for out_tuple, state in unique_outputs.items()
                    ]
                else:
                    # Hide remaining stage tasks on early exit
                    for s in range(stage_idx + 1, effective_limit):
                        progress.update(stage_task_ids[s], visible=False)
                    break  # No valid outputs, can't continue

        # Backward pass: wire children by iterating stages in reverse
        sorted_stages = sorted(per_stage_data.keys())
        for i in range(len(sorted_stages) - 1):
            stage_idx = sorted_stages[i]
            next_stage_idx = sorted_stages[i + 1]

            nodes_by_parent, _ = per_stage_data[stage_idx]
            next_nodes_by_parent, _ = per_stage_data[next_stage_idx]

            for _parent_path, nodes in nodes_by_parent.items():
                for node in nodes:
                    out_key = node.output_state.as_tuple()
                    canonical_path = ("canonical", out_key)
                    node.children = next_nodes_by_parent.get(canonical_path, [])

        # Collect root stage nodes
        all_nodes = {}
        if sorted_stages:
            all_nodes = per_stage_data[sorted_stages[0]][0]

        return all_nodes

    def _build_tree_pipelined(
        self,
        initial_inputs: list[tuple[VectorState, tuple]],
        depth_limit: int | None,
        all_candidates: list[tuple],
        max_unique_outputs: int = 3,
        checkpoint_dir: str | None = None,
        checkpoint_config=None,
        resume_data: dict | None = None,
    ) -> dict[tuple, list[SolutionNode]]:
        """Build the solution tree with pipelined stage processing.

        Replaces the sequential stage barrier with an event-driven pipeline:
        stage N+1 jobs are submitted as soon as unique outputs from stage N
        are discovered (at geometric-series thresholds), allowing overlap.

        The final output is deterministic: ``_finalize_stage`` sorts transitions
        and gadgets identically to the sequential path.

        Args/Returns: same as ``_build_tree_sequential``.
        """
        per_stage_data: dict[
            int, tuple[dict[tuple, list[SolutionNode]], dict[tuple, VectorState]]
        ] = {}

        effective_limit = len(self.bitonic_sorter.stages)
        if depth_limit is not None:
            effective_limit = min(depth_limit, effective_limit)

        # Handle resume
        start_stage = 0
        current_inputs = initial_inputs
        if resume_data is not None:
            per_stage_data = resume_data["per_stage_data"]
            start_stage = resume_data["last_completed_stage"] + 1
            last_stage = resume_data["last_completed_stage"]
            _, unique_outputs = per_stage_data[last_stage]
            current_inputs = [
                (state, ("canonical", out_tuple))
                for out_tuple, state in unique_outputs.items()
            ]

        # Create persistent multi-stage progress display
        progress = SuccessProgress.create(
            width=60,
            success_style="green",
            attempt_style="yellow",
            success_label="Valid",
        )
        progress.enable_memory_monitor()

        # Pre-create one task per stage
        stage_task_ids: dict[int, int] = {}
        for s in range(effective_limit):
            if s < start_stage:
                tid = progress.add_task(
                    f"Stage {s} (resumed)",
                    total=1,
                    completed=1,
                    successes=1,
                )
            else:
                tid = progress.add_task(
                    f"Stage {s}",
                    total=None,
                    start=False,
                    successes=0,
                )
            stage_task_ids[s] = tid

        trackers: dict[int, StageTracker] = {}
        completion_queue: queue_mod.Queue = queue_mod.Queue()
        pending_count = [0]
        cancelled_stages: set[int] = set()
        highest_checkpointed_stage = start_stage - 1

        def _maybe_checkpoint() -> None:
            """Save checkpoint if a contiguous prefix of stages is finalized."""
            nonlocal highest_checkpointed_stage
            if checkpoint_dir is None or checkpoint_config is None:
                return
            try:
                from checkpoint import save_checkpoint, checkpoint_filename
            except ImportError:
                from .checkpoint import save_checkpoint, checkpoint_filename  # type: ignore[no-redef]

            s = highest_checkpointed_stage + 1
            while s in trackers and trackers[s].finalized:
                t = trackers[s]
                per_stage_data[s] = (t.nodes_by_parent, t.unique_outputs)
                s += 1
            new_watermark = s - 1
            if new_watermark > highest_checkpointed_stage:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    checkpoint_filename(
                        checkpoint_config.num_vecs,
                        checkpoint_config.vm,
                        checkpoint_config.prim_type,
                        checkpoint_config.natural_order,
                        new_watermark,
                    ),
                )
                save_checkpoint(
                    ckpt_path,
                    checkpoint_config,
                    per_stage_data,
                    sorted(
                        s_idx
                        for s_idx in range(new_watermark + 1)
                        if s_idx in per_stage_data
                    ),
                    new_watermark,
                )
                highest_checkpointed_stage = new_watermark

        def _finalize_and_cascade(stage_idx: int) -> None:
            """Finalize a stage and cascade to downstream stages if ready."""
            tracker = trackers[stage_idx]

            # Process any remaining unprocessed results
            if tracker.unprocessed_results:
                new_outputs = self._process_batch(tracker)
                if new_outputs and stage_idx + 1 < effective_limit:
                    _forward_outputs(stage_idx + 1, new_outputs)

            self._finalize_stage(tracker)

            if tracker.nodes_by_parent:
                self._stages_completed.add(stage_idx)
                per_stage_data[stage_idx] = (
                    tracker.nodes_by_parent,
                    tracker.unique_outputs,
                )
            else:
                per_stage_data[stage_idx] = ({}, {})
                # Empty stage: cancel everything downstream
                self._cancel_downstream_stages(
                    stage_idx,
                    effective_limit,
                    cancelled_stages,
                    pending_count,
                    trackers,
                    progress,
                    stage_task_ids,
                )

            progress.console.print(
                f"Stage {stage_idx}: Finalized — "
                f"{len(tracker.gadgets_by_transition)} transitions, "
                f"{len(tracker.unique_outputs)} unique outputs"
            )

            # Cascade: mark next stage as having all inputs
            if stage_idx + 1 in trackers:
                nxt = trackers[stage_idx + 1]
                nxt.all_inputs_received = True
                nxt.total_jobs_expected = nxt.total_jobs_submitted
                if nxt.is_complete and not nxt.finalized:
                    _finalize_and_cascade(stage_idx + 1)

            _maybe_checkpoint()

        def _forward_outputs(
            next_stage_idx: int,
            new_outputs: list[tuple[VectorState, tuple]],
        ) -> None:
            """Submit jobs for newly-discovered outputs to the next stage."""
            if next_stage_idx >= effective_limit:
                return

            stage_pairs = self.bitonic_sorter.stages[next_stage_idx]

            if next_stage_idx not in trackers:
                # Launch new stage
                tracker = StageTracker(
                    stage_idx=next_stage_idx, stage_pairs=stage_pairs
                )
                tracker.progress_task_id = stage_task_ids.get(next_stage_idx)
                trackers[next_stage_idx] = tracker

                jobs = self._make_jobs_for_inputs(
                    new_outputs,
                    next_stage_idx,
                    stage_pairs,
                    all_candidates,
                    max_unique_outputs,
                )
                self._submit_stage_jobs(
                    pool, tracker, jobs, pending_count, completion_queue
                )

                if tracker.progress_task_id is not None:
                    progress.start_task(tracker.progress_task_id)
                    progress.update(
                        tracker.progress_task_id,
                        total=tracker.total_jobs_submitted,
                    )

                progress.console.print(
                    f"Stage {next_stage_idx}: Launched {tracker.total_jobs_submitted} jobs "
                    f"(pipelined from stage {next_stage_idx - 1})"
                )
            else:
                # Add more jobs to existing stage
                tracker = trackers[next_stage_idx]
                jobs = self._make_jobs_for_inputs(
                    new_outputs,
                    next_stage_idx,
                    stage_pairs,
                    all_candidates,
                    max_unique_outputs,
                )
                self._submit_stage_jobs(
                    pool, tracker, jobs, pending_count, completion_queue
                )

                if tracker.progress_task_id is not None:
                    progress.update(
                        tracker.progress_task_id,
                        total=tracker.total_jobs_submitted,
                    )

                progress.console.print(
                    f"Stage {next_stage_idx}: Added {len(jobs)} jobs "
                    f"(total: {tracker.total_jobs_submitted})"
                )

        # Main event loop
        with progress:
            if resume_data is not None:
                progress.console.print(
                    f"Resuming from stage {start_stage} with {len(current_inputs)} inputs"
                )

            pool = Pool(
                processes=self._max_workers,
                maxtasksperchild=self._max_tasks_per_child,
            )
            try:
                # Launch stage 0 (or first stage after resume)
                if start_stage < effective_limit:
                    tracker0 = StageTracker(
                        stage_idx=start_stage,
                        stage_pairs=self.bitonic_sorter.stages[start_stage],
                    )
                    tracker0.progress_task_id = stage_task_ids.get(start_stage)
                    tracker0.all_inputs_received = True  # first stage has all inputs
                    trackers[start_stage] = tracker0

                    jobs = self._make_jobs_for_inputs(
                        current_inputs,
                        start_stage,
                        tracker0.stage_pairs,
                        all_candidates,
                        max_unique_outputs,
                    )
                    self._submit_stage_jobs(
                        pool, tracker0, jobs, pending_count, completion_queue
                    )
                    tracker0.total_jobs_expected = tracker0.total_jobs_submitted

                    if tracker0.progress_task_id is not None:
                        progress.start_task(tracker0.progress_task_id)
                        progress.update(
                            tracker0.progress_task_id,
                            total=tracker0.total_jobs_submitted,
                        )

                    progress.console.print(
                        f"Stage {start_stage}: Launched {tracker0.total_jobs_submitted} jobs "
                        f"for {len(current_inputs)} inputs"
                    )

                # Event loop: process completed results via callback queue
                while pending_count[0] > 0:
                    stage_idx, result, error = completion_queue.get()
                    pending_count[0] -= 1

                    if stage_idx in cancelled_stages:
                        continue

                    if error is not None:
                        raise error

                    tracker = trackers[stage_idx]
                    tracker.unprocessed_results.append(result)
                    tracker.completed_jobs += 1

                    # Update progress
                    gadget_results = result[0]
                    success_inc = 1 if gadget_results else 0
                    if tracker.progress_task_id is not None:
                        progress.update(
                            tracker.progress_task_id,
                            advance=1,
                            success=success_inc,
                        )

                    # Check threshold
                    if tracker.should_process_batch():
                        threshold_pct = (
                            _BATCH_THRESHOLDS[tracker.next_threshold_index] * 100
                        )
                        new_outputs = self._process_batch(tracker)
                        if new_outputs and stage_idx + 1 < effective_limit:
                            progress.console.print(
                                f"Stage {stage_idx} → {stage_idx + 1}: "
                                f"forwarding {len(new_outputs)} new unique outputs "
                                f"at {threshold_pct:.0f}% completion "
                                f"({tracker.completed_jobs}/{tracker.total_jobs_submitted} jobs)"
                            )
                            _forward_outputs(stage_idx + 1, new_outputs)

                    # Check completion
                    if tracker.is_complete and not tracker.finalized:
                        _finalize_and_cascade(stage_idx)

                # Finalize any stages that haven't been finalized yet
                # (e.g., stages with 0 jobs due to no inputs)
                for s in range(start_stage, effective_limit):
                    if s in trackers and not trackers[s].finalized:
                        trackers[s].all_inputs_received = True
                        trackers[s].total_jobs_expected = trackers[
                            s
                        ].total_jobs_submitted
                        _finalize_and_cascade(s)
            finally:
                pool.terminate()
                pool.join()

            # Hide stages that were never launched
            for s in range(start_stage, effective_limit):
                if s not in trackers:
                    progress.update(stage_task_ids[s], visible=False)

        # Ensure per_stage_data is populated from all finalized trackers
        for s_idx, tracker in trackers.items():
            if tracker.finalized and s_idx not in per_stage_data:
                per_stage_data[s_idx] = (
                    tracker.nodes_by_parent,
                    tracker.unique_outputs,
                )

        # Backward pass: wire children (identical to sequential path)
        sorted_stages = sorted(per_stage_data.keys())
        for i in range(len(sorted_stages) - 1):
            stage_idx = sorted_stages[i]
            next_stage_idx = sorted_stages[i + 1]

            nodes_by_parent, _ = per_stage_data[stage_idx]
            next_nodes_by_parent, _ = per_stage_data[next_stage_idx]

            for _parent_path, nodes in nodes_by_parent.items():
                for node in nodes:
                    out_key = node.output_state.as_tuple()
                    canonical_path = ("canonical", out_key)
                    node.children = next_nodes_by_parent.get(canonical_path, [])

        # Collect root stage nodes
        all_nodes = {}
        if sorted_stages:
            all_nodes = per_stage_data[sorted_stages[0]][0]

        return all_nodes

    def synthesize_all_stages(
        self,
        depth_limit: int | None = None,
        gadget_depth: int = 1,
        natural_order: bool = False,
        max_unique_outputs: int = 3,
        checkpoint_dir: str | None = None,
        resume_data: dict | None = None,
        pipeline: bool = True,
        max_workers: int | None = None,
        max_tasks_per_child: int | None = 1000,
    ) -> tuple[list[SolutionNode], bool]:
        """Entry point: builds solution tree for all stages.

        Args:
            depth_limit: Maximum stage depth to explore (inclusive). If None, all stages are explored.
            gadget_depth: Maximum instruction depth per gadget (1-3, default 1).
            natural_order: If True, append a final stage restoring natural element order.
            max_unique_outputs: Number of smallest unique output states to enumerate
                per template for deterministic diversity. Default: 3.
            checkpoint_dir: Directory to save checkpoints after each stage completes.
                If None, no checkpoints are saved.
            resume_data: If provided, resume from a previous checkpoint. Expected
                keys: ``per_stage_data`` and ``last_completed_stage``.
            pipeline: If True (default), use pipelined stage processing.
            max_workers: Maximum number of worker processes. If None, defaults
                to os.cpu_count().
            max_tasks_per_child: Maximum tasks per worker process before recycling.
                Limits memory growth in long runs. None disables recycling.

        Returns:
            Tuple of (root nodes, all_stages_complete) where all_stages_complete
            is True if every expected stage was solved successfully.

        Note:
            This method is single-use per ``BitonicSuperVectorizer`` instance.
            Create a new instance for each synthesis run.
        """
        return self.build_solution_tree(
            depth_limit=depth_limit,
            gadget_depth=gadget_depth,
            natural_order=natural_order,
            max_unique_outputs=max_unique_outputs,
            checkpoint_dir=checkpoint_dir,
            resume_data=resume_data,
            pipeline=pipeline,
            max_workers=max_workers,
            max_tasks_per_child=max_tasks_per_child,
        )


_worker_tar = None
_worker_job_count = 0


def _validate_gadget_worker(job):
    """Worker function for parallel gadget validation.

    Returns (gadget_results, input_state, metadata, construction_time, solver_time)
    where gadget_results is a list of (gadget, output_state) tuples.
    """
    global _worker_tar, _worker_job_count
    top_seq, bottom_seq, input_state, target_pairs, vm, prim_type, metadata = job

    # Create clones of sequences to avoid modifying the ones in the main process
    top_seq_clone = copy.deepcopy(top_seq)
    bottom_seq_clone = copy.deepcopy(bottom_seq)

    # Create a local synthesizer and synthesis in its own Z3 context
    synthesizer = GadgetSynthesizer(vm, prim_type)

    smt2_dump_dir = metadata.get("smt2_dump_dir")
    stage_idx = metadata.get("stage_idx")

    def dump_smt2_to_tar(solver):
        global _worker_tar, _worker_job_count
        if smt2_dump_dir is None:
            return

        if _worker_tar is None:
            pid = os.getpid()
            # Write to an uncompressed tar file first; we'll compress it in the main process
            tar_filename = f"stage{stage_idx}_pid_{pid}.tar"
            tar_path = os.path.join(smt2_dump_dir, tar_filename)
            _worker_tar = tarfile.open(tar_path, mode="a")

        _worker_job_count += 1
        smt2_text = "(reset)\n" + solver.sexpr() + "\n(check-sat)\n"
        smt2_bytes = smt2_text.encode("utf-8")

        tar_info = tarfile.TarInfo(name=f"job_{_worker_job_count}.smt2")
        tar_info.size = len(smt2_bytes)
        _worker_tar.addfile(tar_info, io.BytesIO(smt2_bytes))
        # Ensure it's written to disk
        _worker_tar.fileobj.flush()

    allow_any_lane_order = metadata.get("allow_any_lane_order", True)
    max_unique_outputs = metadata.get("max_unique_outputs", 3)

    gadget_results, construction_time, solver_time = (
        synthesizer.synthesize_gadget_with_symbolic(
            top_seq_clone,
            bottom_seq_clone,
            input_state,
            target_pairs,
            max_solutions=1,
            max_unique_outputs=max_unique_outputs,
            solver_callback=dump_smt2_to_tar,
            allow_any_lane_order=allow_any_lane_order,
        )
    )

    metadata["worker_pid"] = os.getpid()
    return gadget_results, input_state, metadata, construction_time, solver_time
