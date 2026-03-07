"""Gadget synthesis engine for the bitonic sort super-optimizer.

Contains the GadgetSynthesizer class, intrinsic dispatch rules,
available intrinsics registry, and the parallel validation worker.
"""

from __future__ import annotations

import copy
import io
import os
import tarfile
import time
from dataclasses import dataclass

from z3 import (
    And,
    BitVec,
    BitVecVal,
    Context,
    Distinct,
    Extract,
    If,
    Optimize,
    Or,
    Solver,
    ULE,
    main_ctx,
    sat,
    simplify,
)

try:
    from . import z3_avx
    from .utils import vector_machine, primitive_type, width_dict
    from .bitonic_types import (
        VectorState,
        SymbolicPlaceholder,
        InstructionSpec,
        PermutationGadget,
    )
except ImportError:
    import z3_avx  # type: ignore
    from utils import vector_machine, primitive_type, width_dict
    from bitonic_types import (  # type: ignore
        VectorState,
        SymbolicPlaceholder,
        InstructionSpec,
        PermutationGadget,
    )


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

    subgraph SYNTH["synthesize_gadget_with_symbolic (per job)"]
        direction TB
        create_in["_create_input_registers<br/>concrete element values"]
        resolve["_resolve_symbolic_instruction_args<br/>SymbolicPlaceholder → Z3 BitVec"]
        create_in --> apply
        resolve --> apply["_apply_instructions (top + bottom sides)"]

        subgraph APPLY["_apply_instructions"]
            direction TB
            inst0["inst0: resolve args via<br/>_substitute_register_names"]
            inst0 --> inst1["inst1+: register args get<br/>_create_operand_mux<br/>(Z3 If-chain over top/bottom/result_0)"]
            inst1 --> prune["_prune_depth2_mux<br/>(for 2-inst sequences only)"]
            prune --> prune_s["1 mux var: force to result_0"]
            prune --> prune_d["2 mux vars: Or(a≥result_0, b≥result_0)"]
        end

        apply --> constrain["_constrain_output_lanes_to_target_pairs<br/>+ Distinct"]
        constrain --> kpw["K-per-wiring algorithm<br/>(Optimize with minimize)"]
        kpw -->|sat| extract["_extract_solution_from_model<br/>model → concrete imm8/cv values<br/>mux select → 'prev'/'top'/'bottom'"]
        kpw -->|unsat| discard["discard candidate"]
        extract --> gadget["PermutationGadget + VectorState"]
    end

    subgraph PARALLEL["Parallel Validation (pipelined via apply_async)"]
        direction TB
        pool["multiprocessing.Pool"]
        pool --> workers["_validate_gadget_worker × N<br/>(each calls synthesize_gadget_with_symbolic)"]
        workers --> collect["collect validated gadgets"]
    end

    subgraph TREE["Solution Tree (build_solution_tree)"]
        direction TB
        stages["for each bitonic sort stage"]
        stages --> input_states["for each unique input VectorState"]
        input_states --> jobs["create jobs: candidate × input_state"]
        jobs --> validate["apply_async → _validate_gadget_worker"]
        validate --> nodes["build SolutionNode DAG<br/>(parent → children across stages)"]
    end

    INIT --> CANDIDATES
    CANDIDATES --> SYNTH
    SYNTH --> PARALLEL
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

    def _build_instruction_sequences(
        self, depth: int, single_input_insts: list[InstructionSpec]
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
            sequences = [[inst] for inst in single_input_insts]
            sequences.extend([[inst] for inst in self.dual_input_insts_top_bottom])
        elif depth == 2:
            # Try pairs:
            # (single, single)
            for inst1 in single_input_insts:
                for inst2 in single_input_insts:
                    sequences.append([inst1, inst2])
            # (dual, single)
            for inst1 in self.dual_input_insts_top_bottom:
                for inst2 in single_input_insts:
                    sequences.append([inst1, inst2])
            # (single, dual)
            for inst1 in single_input_insts:
                for inst2 in self.dual_input_insts_top_bottom:
                    sequences.append([inst1, inst2])
        elif depth > 2:
            raise ValueError(f"Instruction depth {depth} is not supported (max is 2)")
        return sequences

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
