"""Gadget synthesis engine for the bitonic sort super-optimizer.

Contains the GadgetSynthesizer class, intrinsic dispatch rules,
available intrinsics registry, and the parallel validation worker.
"""

from __future__ import annotations

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
        InstructionSpec,
        PermutationGadget,
        InputRef,
        Symbolic,
        Mux,
        IntrinsicNode,
        GadgetGraph,
    )
    from .intrinsics import (
        get_z3_functions,
        get_single_input_nodes,
        get_dual_input_nodes,
    )
except ImportError:
    import z3_avx  # type: ignore
    from utils import vector_machine, primitive_type, width_dict
    from bitonic_types import (  # type: ignore
        VectorState,
        InstructionSpec,
        PermutationGadget,
        InputRef,
        Symbolic,
        Mux,
        IntrinsicNode,
        GadgetGraph,
    )
    from intrinsics import (
        get_z3_functions,
        get_single_input_nodes,
        get_dual_input_nodes,
    )  # type: ignore


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
) -> dict[str, object]:
    """Get available intrinsics for the given VM and primitive type."""
    return get_z3_functions(vm, prim_type)


"""
mermaid
title Gadget Synthesis Pipeline
flowchart TD
    subgraph INIT["Initialization (once per GadgetSynthesizer)"]
        direction TB
        avx["get_available_intrinsics(vm, prim_type)"]
        avx --> single["_enumerate_single_input_intrinsics<br/>→ list[IntrinsicNode]"]
        avx --> dual["_enumerate_dual_input_intrinsics<br/>→ list[IntrinsicNode]"]
    end

    subgraph CANDIDATES["Candidate Enumeration (precompute_all_candidates)"]
        direction TB
        intrinsics["IntrinsicNode templates"] --> build["_build_gadget_graphs(depth)"]
        build --> d0["depth 0: identity (None)"]
        build --> d1["depth 1: single IntrinsicNode"]
        build --> d2["depth 2: _wire_hardwired or<br/>_wire_with_muxes<br/>(explicit Mux nodes in graph)"]
        intrinsics --> shared["_build_shared_prefix_graphs<br/>shared-prefix pattern:<br/>both sides share prefix nodes,<br/>CSE → 4 unified instructions"]
        d0 --> combine["_generate_candidate_graphs_at_depth<br/>cartesian product → list[GadgetGraph]"]
        d1 --> combine
        d2 --> combine
        shared --> combine
    end

    subgraph TREE["Solution Tree (build_solution_tree)"]
        direction TB
        stages["for each bitonic sort stage"]
        stages --> input_states["for each unique input VectorState"]
        input_states --> jobs["create jobs: GadgetGraph × input_state"]
        jobs --> dispatch["apply_async → multiprocessing.Pool"]
        dispatch --> collect["collect validated gadgets<br/>via completion_queue"]
        collect --> nodes["build SolutionNode DAG<br/>(parent → children across stages)"]
    end

    subgraph WORKER["_validate_gadget_worker (per subprocess)"]
        direction TB
        synth_create["GadgetSynthesizer(vm, prim_type)"]

        subgraph SYNTH["synthesize_gadget_with_symbolic"]
            direction TB
            create_in["_create_input_registers<br/>concrete element values"]

            subgraph EVAL["_evaluate (recursive graph walk)"]
                direction TB
                input_ref["InputRef → registers dict"]
                symbolic["Symbolic → BitVec (created on first visit)"]
                mux_node["Mux → If-chain + range constraint<br/>pruning applied by parent IntrinsicNode"]
                intr_node["IntrinsicNode → dispatch intrinsic(args)<br/>+ _apply_mux_pruning for Mux children"]
            end

            create_in --> EVAL
            EVAL --> constrain["_constrain_output_lanes_to_target_pairs<br/>+ Distinct"]
            constrain --> kpw["K-per-wiring algorithm<br/>(Optimize with minimize)"]
            kpw -->|sat| extract["_extract_solution_from_graph<br/>_concretize_graph → list[InstructionSpec]<br/>mux select → 'prev'/'top'/'bottom'"]
            kpw -->|unsat| discard["discard candidate"]
            extract --> gadget["PermutationGadget + VectorState"]
        end

        synth_create --> SYNTH
    end

    INIT --> CANDIDATES
    CANDIDATES --> TREE
    TREE --> WORKER

    style INIT fill:#e8f0fe,stroke:#4285f4
    style CANDIDATES fill:#e8f0fe,stroke:#4285f4
    style TREE fill:#e8f0fe,stroke:#4285f4
    style WORKER fill:#fef7e0,stroke:#f9ab00
    style SYNTH fill:#fef7e0,stroke:#f9ab00
    style EVAL fill:#fef7e0,stroke:#f9ab00
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

        # Memoize graph-based instruction templates
        allowed = set(self.available_intrinsics)
        self._top_ref = InputRef("top")
        self._bottom_ref = InputRef("bottom")
        self.single_intrinsics_top = [
            n
            for n in self._enumerate_single_input_intrinsics(self._top_ref)
            if n.name in allowed
        ]
        self.single_intrinsics_bottom = [
            n
            for n in self._enumerate_single_input_intrinsics(self._bottom_ref)
            if n.name in allowed
        ]
        self.dual_intrinsics = [
            n
            for n in self._enumerate_dual_input_intrinsics(
                self._top_ref, self._bottom_ref
            )
            if n.name in allowed
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
        graph: GadgetGraph,
        input_state: VectorState,
        target_pairs: list[tuple[int, int]],
        max_solutions: int = 1,
        max_unique_outputs: int = 3,
        solver_callback: callable | None = None,
        allow_any_lane_order: bool = True,
    ) -> tuple[list[tuple[PermutationGadget, VectorState]], float, float]:
        """Synthesize gadgets from a data-flow graph using Z3.

        Evaluates the graph to build Z3 expressions, then solves for
        symbolic variable assignments that satisfy the target permutation.

        Returns (results, construction_time, solver_time) where results is
        list of (gadget, output_state) tuples.

        Args:
            graph: Data-flow graph with symbolic immediates and optional muxes.
            max_solutions: Maximum number of solutions to return. Defaults to 1,
                which enumerates unique output states deterministically.
            max_unique_outputs: When max_solutions == 1, the number of smallest
                unique output states to enumerate per template. Default: 3.
            solver_callback: Optional callback receiving the Solver instance.
            allow_any_lane_order: When True (default), any target pair can land
                in any lane and the output state is canonicalized.
        """
        start_construction = time.perf_counter()
        ctx = main_ctx()
        solver = Solver(ctx=ctx)

        # Create input registers with actual element values
        top_reg, bottom_reg = self._create_input_registers(solver, ctx, input_state)
        registers = {"top": top_reg, "bottom": bottom_reg}

        # Evaluate the graph to build Z3 expressions
        symbolic_vars = {}
        mux_constraints = []
        eval_cache = {}

        top_output = self._evaluate(
            graph.top, registers, ctx, symbolic_vars, mux_constraints, eval_cache
        )
        bottom_output = self._evaluate(
            graph.bottom, registers, ctx, symbolic_vars, mux_constraints, eval_cache
        )

        # Identity side: pass through the input register
        if top_output is None:
            top_output = top_reg
        if bottom_output is None:
            bottom_output = bottom_reg

        # Add mux constraints
        for c in mux_constraints:
            solver.add(c)

        all_output_lanes = self._constrain_output_lanes_to_target_pairs(
            solver,
            ctx,
            top_output,
            bottom_output,
            target_pairs,
            allow_any_lane_order,
        )

        solver.add(Distinct(*all_output_lanes))

        if solver_callback:
            solver_callback(solver)

        fixed_output_state = self._fixed_output_state_from_target_pairs(
            target_pairs, allow_any_lane_order
        )

        # Partition symbolic vars: select variables vs immediate variables
        select_vars = {k: v for k, v in symbolic_vars.items() if k.startswith("sel_")}
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
            gadget, output_state = self._extract_solution_from_graph(
                model,
                top_output,
                bottom_output,
                graph,
                symbolic_vars,
                fixed_output_state=fixed_output_state,
            )
            solver_time = time.perf_counter() - solver_start
            return [(gadget, output_state)], construction_time, solver_time

        if max_solutions == 1:
            # K-per-wiring algorithm: deterministic synthesis with diversity.
            original_assertions = list(solver.assertions())
            results = []
            wiring_blocks = []

            while True:
                opt_wiring = Optimize(ctx=ctx)
                for a in original_assertions:
                    opt_wiring.add(a)
                for block in wiring_blocks:
                    opt_wiring.add(block)
                opt_wiring.minimize(top_output)
                opt_wiring.minimize(bottom_output)

                if opt_wiring.check() != sat:
                    break

                model = opt_wiring.model()

                wiring = {}
                for name, var in select_vars.items():
                    wiring[name] = model.evaluate(var, model_completion=True).as_long()

                wiring_constraints = [
                    var == BitVecVal(wiring[name], var.size())
                    for name, var in select_vars.items()
                ]

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
                        gadget, output_state = self._extract_solution_from_graph(
                            opt_imm.model(),
                            top_output,
                            bottom_output,
                            graph,
                            symbolic_vars,
                            fixed_output_state=fixed_output_state,
                        )
                        results.append((gadget, output_state))

                    output_blocks.append(
                        Or(top_output != top_val, bottom_output != bot_val)
                    )

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
                    break

            results.sort(key=lambda x: x[1].as_tuple())
            solver_time = time.perf_counter() - solver_start
            return results, construction_time, solver_time

        # Enumerate multiple solutions over symbolic variables
        all_terms = list(symbolic_vars.values())
        results = []
        for model in _all_smt(solver, all_terms, max_results=max_solutions):
            gadget, output_state = self._extract_solution_from_graph(
                model,
                top_output,
                bottom_output,
                graph,
                symbolic_vars,
                fixed_output_state=fixed_output_state,
            )
            results.append((gadget, output_state))
        solver_time = time.perf_counter() - solver_start
        return results, construction_time, solver_time

    def _evaluate(
        self,
        node: InputRef | Symbolic | Mux | IntrinsicNode | None,
        registers: dict,
        ctx: Context,
        symbolic_vars: dict,
        mux_constraints: list,
        cache: dict,
    ):
        """Recursively evaluate a graph node to a Z3 expression.

        Memoizes by ``id(node)`` to avoid redundant evaluation when the same
        node is referenced by multiple parents (e.g. ``InputRef("top")``
        shared across operands).
        """
        if node is None:
            return None

        nid = id(node)
        if nid in cache:
            return cache[nid]

        if isinstance(node, InputRef):
            result = registers[node.name]

        elif isinstance(node, Symbolic):
            if node.name in symbolic_vars:
                result = symbolic_vars[node.name]
            else:
                if node.bit_width == 256:
                    result = z3_avx.ymm_reg(node.name, ctx=ctx)
                elif node.bit_width == 512:
                    result = z3_avx.zmm_reg(node.name, ctx=ctx)
                else:
                    result = BitVec(node.name, node.bit_width, ctx=ctx)
                symbolic_vars[node.name] = result

        elif isinstance(node, Mux):
            sel = self._evaluate(
                node.select, registers, ctx, symbolic_vars, mux_constraints, cache
            )
            sources = [
                self._evaluate(s, registers, ctx, symbolic_vars, mux_constraints, cache)
                for s in node.sources
            ]
            # Range constraint
            mux_constraints.append(ULE(sel, BitVecVal(len(sources) - 1, sel.size())))
            # Build If-chain backwards
            result = sources[-1]
            for i in range(len(sources) - 2, -1, -1):
                result = If(sel == i, sources[i], result)
            # Store select var for extraction
            symbolic_vars[node.select.name] = sel
            cache[nid] = result
            # Pruning applied by parent IntrinsicNode via _apply_mux_pruning
            return result

        elif isinstance(node, IntrinsicNode):
            intrinsic_fn = self.available_intrinsics.get(node.name)
            if intrinsic_fn is None:
                raise ValueError(f"Unknown intrinsic: {node.name}")
            args = {}
            for key, operand in node.operands.items():
                args[key] = self._evaluate(
                    operand, registers, ctx, symbolic_vars, mux_constraints, cache
                )
            result = _dispatch_intrinsic_by_signature(intrinsic_fn, args)
            # Apply combined mux pruning for any Mux children
            self._apply_mux_pruning(node, symbolic_vars, mux_constraints)
        else:
            raise TypeError(f"Unknown graph node type: {type(node)}")

        cache[nid] = result
        return result

    @staticmethod
    def _apply_mux_pruning(
        node: IntrinsicNode,
        symbolic_vars: dict,
        mux_constraints: list,
    ):
        """Apply pruning constraints for Mux children of an IntrinsicNode.

        - ``"force_last"``: single mux forced to pick the last source.
        - ``"at_least_one_last"``: with multiple sibling muxes, at least one
          must select a source index >= the last (i.e., a prior result).
        """
        mux_selects = []
        for operand in node.operands.values():
            if isinstance(operand, Mux) and operand.pruning is not None:
                sel_var = symbolic_vars[operand.select.name]
                last_idx = len(operand.sources) - 1
                mux_selects.append((sel_var, last_idx, operand.pruning))

        if not mux_selects:
            return

        # Collect all "at_least_one_last" muxes for joint constraint
        joint = [
            (sel, last) for sel, last, p in mux_selects if p == "at_least_one_last"
        ]

        for sel, last, pruning in mux_selects:
            if pruning == "force_last":
                mux_constraints.append(sel == last)

        if len(joint) == 1:
            sel, last = joint[0]
            mux_constraints.append(sel == last)
        elif len(joint) >= 2:
            mux_constraints.append(
                Or(*(ULE(BitVecVal(last, sel.size()), sel) for sel, last in joint))
            )

    def _extract_solution_from_graph(
        self,
        model,
        top_output,
        bottom_output,
        graph: GadgetGraph,
        symbolic_vars: dict,
        fixed_output_state: VectorState | None = None,
    ) -> tuple[PermutationGadget, VectorState]:
        """Extract a concrete gadget and output state from a Z3 model.

        Walks the graph in topological order, evaluating symbolic variables
        and mux selects against the model to produce concrete InstructionSpec
        lists for the PermutationGadget.
        """
        if fixed_output_state is not None:
            output_state = fixed_output_state
        else:
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

                output_top.append(min(top_elem, bottom_elem))
                output_bottom.append(max(top_elem, bottom_elem))

            output_state = VectorState(top=output_top, bottom=output_bottom)

        concrete_top = self._concretize_graph(graph.top, model, symbolic_vars)
        concrete_bottom = self._concretize_graph(graph.bottom, model, symbolic_vars)

        gadget = PermutationGadget(
            top_instructions=concrete_top,
            bottom_instructions=concrete_bottom,
            validated=True,
        )
        return gadget, output_state

    def _concretize_graph(
        self,
        root: IntrinsicNode | None,
        model,
        symbolic_vars: dict,
    ) -> list[InstructionSpec]:
        """Walk a graph and produce a flat list of concrete InstructionSpec.

        Collects IntrinsicNodes in topological (post-)order, then concretizes
        each node's operands against the model.
        """
        if root is None:
            return []

        # Collect IntrinsicNodes in topological order
        order: list[IntrinsicNode] = []
        visited: set[int] = set()

        def topo_visit(node):
            nid = id(node)
            if nid in visited:
                return
            visited.add(nid)
            if isinstance(node, IntrinsicNode):
                for operand in node.operands.values():
                    topo_visit(operand)
                order.append(node)
            elif isinstance(node, Mux):
                for source in node.sources:
                    topo_visit(source)

        topo_visit(root)

        # Map IntrinsicNode id → index in linearized order
        node_to_idx = {id(n): idx for idx, n in enumerate(order)}

        result = []
        for inst_idx, node in enumerate(order):
            concrete_args = {}
            for key, operand in node.operands.items():
                concrete_args[key] = self._concretize_operand(
                    operand, inst_idx, model, symbolic_vars, node_to_idx
                )
            result.append(InstructionSpec(node.name, concrete_args))
        return result

    def _concretize_operand(
        self,
        operand,
        inst_idx: int,
        model,
        symbolic_vars: dict,
        node_to_idx: dict,
    ):
        """Concretize a single graph operand against a Z3 model."""
        if isinstance(operand, InputRef):
            return operand.name

        if isinstance(operand, Symbolic):
            z3_var = symbolic_vars[operand.name]
            return model.evaluate(z3_var, model_completion=True).as_long()

        if isinstance(operand, Mux):
            sel_val = model.evaluate(
                symbolic_vars[operand.select.name], model_completion=True
            ).as_long()
            return self._select_to_source(sel_val, inst_idx)

        if isinstance(operand, IntrinsicNode):
            ref_idx = node_to_idx[id(operand)]
            if ref_idx == inst_idx - 1:
                return "prev"
            return f"result_{ref_idx}"

        raise TypeError(f"Unknown operand type: {type(operand)}")

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

        top_reg, bottom_reg = self._create_input_registers(solver, ctx, input_state)

        # Apply concrete gadget instructions
        top_output, _ = self._apply_concrete_instructions(
            top_reg, bottom_reg, gadget.top_instructions, is_top=True
        )
        bottom_output, _ = self._apply_concrete_instructions(
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

    def _apply_concrete_instructions(
        self,
        top_reg,
        bottom_reg,
        instructions: list[InstructionSpec],
        is_top: bool,
    ):
        """Apply concrete (fully-resolved) instructions to compute an output register.

        Used by ``compute_output_state`` where all operands are already concrete
        strings ("top", "bottom", "prev", "result_N") or integer immediates.

        Returns:
            (output_register, []) tuple (empty mux_constraints for API compat).
        """
        current_reg = top_reg if is_top else bottom_reg
        results = []

        for inst in instructions:
            intrinsic = self.available_intrinsics.get(inst.intrinsic_name)
            if intrinsic is None:
                raise ValueError(f"Unknown intrinsic: {inst.intrinsic_name}")

            args = {}
            for key, value in inst.args.items():
                if isinstance(value, str):
                    if value == "top":
                        args[key] = top_reg
                    elif value == "bottom":
                        args[key] = bottom_reg
                    elif value == "prev":
                        args[key] = results[-1] if results else current_reg
                    elif value.startswith("result_"):
                        idx = int(value[len("result_") :])
                        args[key] = results[idx] if idx < len(results) else current_reg
                    else:
                        args[key] = current_reg
                elif isinstance(value, int):
                    if key == "imm8":
                        args[key] = BitVecVal(value, 8)
                    else:
                        args[key] = BitVecVal(value, width_dict[self.vm] * 8)
                else:
                    args[key] = value

            current_reg = _dispatch_intrinsic_by_signature(intrinsic, args)
            results.append(current_reg)

        return current_reg, []

    # ------------------------------------------------------------------
    # Graph-based candidate enumeration
    # ------------------------------------------------------------------

    def _build_gadget_graphs(
        self, depth: int, single_intrinsics: list[IntrinsicNode]
    ) -> list[IntrinsicNode | None]:
        """Build data-flow graphs for all instruction sequences at a given depth.

        Depth 0: identity (None).
        Depth 1: one intrinsic node per template.
        Depth 2: pairs with explicit wiring and mux nodes where needed.

        Ordering variants driven by ``isomorphic_order`` tag:

        Shapes B & D (dual-input at depth 1 and depth 2 respectively):
          Symmetric duals (isomorphic_order=True) produce one operand ordering.
          Asymmetric duals (isomorphic_order=False) produce both orderings:
          inst(top, bot) and inst(bot, top).

        Shape F (shared-prefix → dual tails, built by _build_shared_prefix_graphs):
          isomorphic_order=True  + Symbolics   → 1 ordering; independent CVs per side
          isomorphic_order=False + Symbolics   → 2 same-ordering variants (A,B)+(B,A)
          isomorphic_order=False + no Symbolics → cross-ordering: top uses (A,B),
                                                  bottom uses (B,A) — the only way to
                                                  get distinct outputs without a CV
          isomorphic_order=True  + no Symbolics → skipped (top ≡ bottom always)

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
                subgraph B["Shape B: dual-input; asymmetric duals emit both orderings"]
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
                    c_top["top/bottom reg"]
                    c_top --> c_i0["inst0(a, imm8/cv)"]
                    c_i0 --> c_i1["inst1(result_0, imm8/cv)"]
                    c_i1 --> c_COEX["COEX"]
                end
                subgraph D["Shape D: dual → single; asymmetric duals emit both orderings"]
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
                    e_i0 -.-> e_mux_a{{mux a}}
                    e_top -.-> e_mux_a
                    e_bot -.-> e_mux_a
                    e_i0 -.-> e_mux_b{{mux b}}
                    e_top -.-> e_mux_b
                    e_bot -.-> e_mux_b
                    e_mux_a --> e_i1["inst1(a, b, imm8)"]
                    e_mux_b --> e_i1
                    e_i1 --> e_COEX["COEX"]
                end
            end

            subgraph D2S["2 inst per side (shared prefix)"]
                subgraph F["Shape F: shared prefix → dual tails (tag-aware, see above)"]
                    direction TB
                    f_top["top reg"]
                    f_bot["bottom reg"]
                    f_top --> f_pA["prefix_A(a, imm8/cv)"]
                    f_bot --> f_pB["prefix_B(a, imm8/cv)"]
                    f_pA --> f_tail_top["tail_top(a, b, imm8_top)"]
                    f_pB --> f_tail_top
                    f_pA --> f_tail_bot["tail_bot(a, b, imm8_bot)"]
                    f_pB --> f_tail_bot
                    f_tail_top --> f_COEX["COEX"]
                    f_tail_bot --> f_COEX
                end
            end
        """
        if depth <= 0:
            return [None]

        if depth == 1:
            result: list[IntrinsicNode] = list(single_intrinsics)
            for dual in self.dual_intrinsics:
                result.append(dual)
                if not dual.isomorphic_order:
                    result.append(self._swap_register_operands(dual))
            return result

        if depth == 2:
            top = self._top_ref
            bottom = self._bottom_ref
            graphs: list[IntrinsicNode] = []

            # single → single: inst1 hardwired to inst0
            for inst0 in single_intrinsics:
                for inst1 in single_intrinsics:
                    graphs.append(self._wire_hardwired(inst0, inst1))

            # dual → single: inst1 hardwired to inst0
            for inst0 in self.dual_intrinsics:
                inst0_variants = [inst0]
                if not inst0.isomorphic_order:
                    inst0_variants.append(self._swap_register_operands(inst0))
                for inst0_v in inst0_variants:
                    for inst1 in single_intrinsics:
                        graphs.append(self._wire_hardwired(inst0_v, inst1))

            # single → dual: inst1 gets muxes on both register operands
            for inst0 in single_intrinsics:
                for inst1 in self.dual_intrinsics:
                    graphs.append(self._wire_with_muxes(inst0, inst1, top, bottom))
            return graphs

        raise ValueError(f"Instruction depth {depth} is not supported (max is 2)")

    def _build_shared_prefix_graphs(self) -> list[GadgetGraph]:
        """Build shared-prefix gadget graphs (Shape F).

        Both sides share a prefix of two single-input permutations (one on
        each register), then each side applies a dual-input tail instruction.
        After CSE the shared prefix is deduplicated, yielding 4 unified
        instructions (2 shared + 2 tails).

        The tail enumeration is tag-aware (isomorphic_order):

          isomorphic_order=True  + Symbolics → one ordering, independent
                                               Symbolic names per side
          isomorphic_order=False + Symbolics → two same-ordering variants
                                               (A,B) and (B,A) for both sides
          isomorphic_order=False + no Symbolics → cross-ordering: top uses
                                                  (A,B), bottom uses (B,A);
                                                  the only way to get distinct
                                                  side outputs without a CV
          isomorphic_order=True  + no Symbolics → skip (top ≡ bottom always)
        """

        def _has_symbolic(node: IntrinsicNode) -> bool:
            return any(isinstance(v, Symbolic) for v in node.operands.values())

        def _register_keys(node: IntrinsicNode) -> list[str]:
            return [k for k, v in node.operands.items() if isinstance(v, InputRef)]

        def _build_tail(
            dual: IntrinsicNode,
            reg_a: IntrinsicNode,
            reg_b: IntrinsicNode,
            suffix: str,
        ) -> IntrinsicNode:
            """Build one tail node wiring reg_a→first-reg-key, reg_b→second-reg-key."""
            reg_keys = _register_keys(dual)
            ops = {}
            for key, operand in dual.operands.items():
                if key == reg_keys[0]:
                    ops[key] = reg_a
                elif key == reg_keys[1]:
                    ops[key] = reg_b
                elif isinstance(operand, Symbolic):
                    ops[key] = Symbolic(
                        f"{operand.name}_shared_{suffix}", operand.bit_width
                    )
                else:
                    ops[key] = operand
            return IntrinsicNode(dual.name, ops, isomorphic_order=dual.isomorphic_order)

        graphs: list[GadgetGraph] = []

        for prefix_top in self.single_intrinsics_top:
            for prefix_bot in self.single_intrinsics_bottom:
                for dual in self.dual_intrinsics:
                    reg_keys = _register_keys(dual)
                    if len(reg_keys) != 2:
                        continue

                    has_sym = _has_symbolic(dual)

                    if dual.isomorphic_order:
                        # Symmetric: one ordering only
                        if not has_sym:
                            continue  # top ≡ bottom always — useless
                        tail_top = _build_tail(dual, prefix_top, prefix_bot, "top")
                        tail_bot = _build_tail(dual, prefix_top, prefix_bot, "bot")
                        graphs.append(GadgetGraph(top=tail_top, bottom=tail_bot))
                    else:
                        # Asymmetric
                        if has_sym:
                            # Two same-ordering variants
                            for reg_a, reg_b in [
                                (prefix_top, prefix_bot),
                                (prefix_bot, prefix_top),
                            ]:
                                tail_top = _build_tail(dual, reg_a, reg_b, "top")
                                tail_bot = _build_tail(dual, reg_a, reg_b, "bot")
                                graphs.append(
                                    GadgetGraph(top=tail_top, bottom=tail_bot)
                                )
                        else:
                            # Cross-ordering: only way to get distinct outputs
                            tail_top = _build_tail(dual, prefix_top, prefix_bot, "top")
                            tail_bot = _build_tail(dual, prefix_bot, prefix_top, "bot")
                            graphs.append(GadgetGraph(top=tail_top, bottom=tail_bot))

        return graphs

    @staticmethod
    def _wire_hardwired(inst0: IntrinsicNode, inst1: IntrinsicNode) -> IntrinsicNode:
        """Wire all of inst1's register (InputRef) operands to inst0's output."""
        operands = {}
        for key, operand in inst1.operands.items():
            if isinstance(operand, InputRef):
                # Replace the register input with inst0's output
                operands[key] = inst0
            else:
                operands[key] = operand
        return IntrinsicNode(inst1.name, operands)

    @staticmethod
    def _swap_register_operands(node: IntrinsicNode) -> IntrinsicNode:
        """Return a copy of *node* with its two InputRef operands swapped."""
        reg_keys = [k for k, v in node.operands.items() if isinstance(v, InputRef)]
        assert len(reg_keys) == 2, f"Expected 2 InputRef operands, got {reg_keys}"
        new_ops = dict(node.operands)
        new_ops[reg_keys[0]], new_ops[reg_keys[1]] = (
            new_ops[reg_keys[1]],
            new_ops[reg_keys[0]],
        )
        return IntrinsicNode(node.name, new_ops, isomorphic_order=node.isomorphic_order)

    @staticmethod
    def _wire_with_muxes(
        inst0: IntrinsicNode,
        inst1: IntrinsicNode,
        top: InputRef,
        bottom: InputRef,
    ) -> IntrinsicNode:
        """Wire inst1's register operands through muxes over {top, bottom, inst0}."""
        operands = {}
        for key, operand in inst1.operands.items():
            if isinstance(operand, (InputRef, IntrinsicNode)):
                # Register operand gets a mux
                sel_name = f"sel_{key}_{inst1.name}_{id(inst1)}"
                operands[key] = Mux(
                    select=Symbolic(sel_name, 2),
                    sources=(top, bottom, inst0),
                    pruning="at_least_one_last",
                )
            else:
                operands[key] = operand
        return IntrinsicNode(inst1.name, operands)

    def _generate_candidate_graphs_at_depth(
        self, top_depth: int, bottom_depth: int
    ) -> list[GadgetGraph]:
        """Generate candidate GadgetGraphs for a given (top_depth, bottom_depth)."""
        top_graphs = self._build_gadget_graphs(top_depth, self.single_intrinsics_top)
        bottom_graphs = self._build_gadget_graphs(
            bottom_depth, self.single_intrinsics_bottom
        )
        return [GadgetGraph(top=t, bottom=b) for t in top_graphs for b in bottom_graphs]

    def precompute_all_candidates(self, gadget_depth: int) -> list[GadgetGraph]:
        """Pre-compute all candidate gadget graphs for all depth combinations.

        The candidates depend only on gadget_depth and the available intrinsics,
        not on input state or stage pairs.
        """
        all_candidates: list[GadgetGraph] = []
        for top_depth in range(gadget_depth + 1):
            for bottom_depth in range(gadget_depth + 1):
                all_candidates.extend(
                    self._generate_candidate_graphs_at_depth(top_depth, bottom_depth)
                )
        if gadget_depth >= 2:
            all_candidates.extend(self._build_shared_prefix_graphs())
        return all_candidates

    # ------------------------------------------------------------------
    # Intrinsic node enumeration (graph-based replacements)
    # ------------------------------------------------------------------

    def _enumerate_single_input_intrinsics(
        self, input_ref: InputRef
    ) -> list[IntrinsicNode]:
        """Generate single-input IntrinsicNode templates with Symbolic immediates."""
        return get_single_input_nodes(self.vm, self.prim_type, input_ref)

    def _enumerate_dual_input_intrinsics(
        self, ref1: InputRef, ref2: InputRef
    ) -> list[IntrinsicNode]:
        """Generate dual-input IntrinsicNode templates with Symbolic immediates."""
        return get_dual_input_nodes(self.vm, self.prim_type, ref1, ref2)


_worker_tar = None
_worker_job_count = 0


def _validate_gadget_worker(job):
    """Worker function for parallel gadget validation.

    Returns (gadget_results, input_state, metadata, construction_time, solver_time)
    where gadget_results is a list of (gadget, output_state) tuples.
    """
    global _worker_tar, _worker_job_count
    graph, input_state, target_pairs, vm, prim_type, metadata = job

    # Create a local synthesizer in its own Z3 context
    synthesizer = GadgetSynthesizer(vm, prim_type)

    smt2_dump_dir = metadata.get("smt2_dump_dir")
    stage_idx = metadata.get("stage_idx")

    def dump_smt2_to_tar(solver):
        global _worker_tar, _worker_job_count
        if smt2_dump_dir is None:
            return

        if _worker_tar is None:
            pid = os.getpid()
            tar_filename = f"stage{stage_idx}_pid_{pid}.tar"
            tar_path = os.path.join(smt2_dump_dir, tar_filename)
            _worker_tar = tarfile.open(tar_path, mode="a")

        _worker_job_count += 1
        smt2_text = "(reset)\n" + solver.sexpr() + "\n(check-sat)\n"
        smt2_bytes = smt2_text.encode("utf-8")

        tar_info = tarfile.TarInfo(name=f"job_{_worker_job_count}.smt2")
        tar_info.size = len(smt2_bytes)
        _worker_tar.addfile(tar_info, io.BytesIO(smt2_bytes))
        _worker_tar.fileobj.flush()

    allow_any_lane_order = metadata.get("allow_any_lane_order", True)
    max_unique_outputs = metadata.get("max_unique_outputs", 3)

    gadget_results, construction_time, solver_time = (
        synthesizer.synthesize_gadget_with_symbolic(
            graph,
            input_state,
            target_pairs,
            max_solutions=1,
            max_unique_outputs=max_unique_outputs,
            solver_callback=dump_smt2_to_tar,
            allow_any_lane_order=allow_any_lane_order,
        )
    )

    # Reject gadgets where CSE doesn't collapse to <= 4 unified instructions
    gadget_results = [
        (g, out) for g, out in gadget_results if g.instruction_count() <= 4
    ]

    metadata["worker_pid"] = os.getpid()
    return gadget_results, input_state, metadata, construction_time, solver_time
