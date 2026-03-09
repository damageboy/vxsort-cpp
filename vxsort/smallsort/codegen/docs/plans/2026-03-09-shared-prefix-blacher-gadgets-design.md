# Shared-Prefix (Blacher) Gadget Pattern

**Date**: 2026-03-09
**Status**: Design

## Problem

The current gadget graph builder (`_build_gadget_graphs`) constructs per-side
instruction graphs independently, then combines them via Cartesian product in
`_generate_candidate_graphs_at_depth`. This cannot express cross-side sharing
where both the top and bottom sides of a `GadgetGraph` reference the same
intermediate instruction nodes.

The Blacher sorting network uses a pattern where both sides share a permutation
prefix — two single-input permutes (one on each register) — feeding into a
dual-input tail instruction with independent immediates per side. By
construction the prefix is identical across sides, so CSE (`unified_instructions`)
collapses 6 raw instructions (3 per side) into 4 unified instructions (2 shared
prefix + 2 independent tails), making this effectively a 2-instruction-per-side
gadget.

## Data-Flow Shape

```
  InputRef("top")       InputRef("bottom")
       |                       |
  prefix_perm_A          prefix_perm_B         ← SHARED nodes (same Python objects)
       |                       |
       +-------+-------+------+
               |               |
        dual_tail_TOP    dual_tail_BOT         ← independent Symbolic immediates
```

Both `dual_tail_TOP` and `dual_tail_BOT` reference the exact same `prefix_perm_A`
and `prefix_perm_B` objects. The tail nodes differ only in their Symbolic
immediate names (`_blacher_top` vs `_blacher_bot`).

### Why the existing machinery handles this

- **`_evaluate()`**: Memoizes by `id(node)`. Shared prefix nodes are evaluated
  once; both sides see the same Z3 expressions and share the same Symbolic
  variables, so Z3 constrains them to identical concrete values.
- **`_concretize_graph()`**: Called independently per side (visited set is
  per-call), so both sides produce 3 `InstructionSpec` each. The shared prefix
  gets identical concrete args from the same Z3 model values.
- **`unified_instructions()`**: Signature-based CSE compares `(intrinsic_name,
  normalized_args)`. Shared prefix instructions have identical signatures across
  sides → deduplicated. Tail instructions differ in their immediate values → kept
  separate. Result: 4 unified instructions.

## Design

### New method: `_build_shared_prefix_graphs`

A new method on `GadgetSynthesizer` that returns `list[GadgetGraph]`. Called from
`precompute_all_candidates` when `gadget_depth >= 2`.

**Enumeration**: for each combination of:

1. `prefix_top` from `single_intrinsics_top` — permutes the top register
2. `prefix_bot` from `single_intrinsics_bottom` — permutes the bottom register
3. `tail_template` from `dual_intrinsics` — must have ≥1 `Symbolic` operand
   (otherwise both tails would be identical, producing useless duplicate output)

Construct:

- Two tail `IntrinsicNode` objects, each wiring both shared prefix nodes as
  register operands, with side-specific Symbolic names for any immediate/CV
  operands.
- Also enumerate the reverse wiring (swap which prefix result feeds operand `a`
  vs `b`), since non-commutative instructions treat operands differently and
  some prefix permutations have limited expressiveness (e.g. `permute_ps` is
  in-lane only).
- Emit `GadgetGraph(top=tail_top, bottom=tail_bot)`.

**Combinatorial size** (AVX2 i32): 4 × 4 × 4 × 2 = 128 graphs. Comparable to
existing depth-2 counts.

### Tail node construction

For each `dual_template` with Symbolic operands:

1. Identify register operands (`InputRef` or `IntrinsicNode` typed) and Symbolic
   operands in the template.
2. Replace register operands with the two shared prefix nodes (in both orderings).
3. Replace each `Symbolic` with a fresh `Symbolic` bearing a side-specific name
   suffix (`_blacher_top` / `_blacher_bot`) to give Z3 independent freedom per
   side.

### Post-synthesis filtering

In `_validate_gadget_worker`, after `synthesize_gadget_with_symbolic` returns,
reject any gadget where `gadget.instruction_count() > 4`. This is a no-op for
depth-0/1/2 patterns (always ≤ 4) and acts as the CSE validation gate for
shared-prefix patterns (6 raw → 4 expected after CSE).

### What does NOT change

- `bitonic_types.py` — `GadgetGraph`, `PermutationGadget`, `unified_instructions`
  all work correctly as-is.
- `_evaluate` / `_extract_solution_from_graph` / `_concretize_graph` — shared
  nodes handled by existing `id()`-based memoization.
- The super-optimizer pipeline (`_build_tree_pipelined`, `_process_batch`, etc.)
  — shared-prefix `GadgetGraph` objects are just more candidates in the list.

## Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add shared-prefix (Blacher) gadget graphs as a new depth-2 search pattern, with CSE-based post-synthesis filtering.

**Architecture:** New method `_build_shared_prefix_graphs` on `GadgetSynthesizer` generates `GadgetGraph` objects where both sides share prefix `IntrinsicNode` objects. These are appended to the existing depth-2 candidates. A post-synthesis filter rejects results where CSE doesn't collapse to ≤ 4 unified instructions.

**Tech Stack:** Python, Z3 (via z3-solver), pytest

---

### Task 1: Add `_build_shared_prefix_graphs` method

**Files:**
- Modify: `src/gadget_synthesizer.py:1171` (after `_build_gadget_graphs`, before `_wire_hardwired`)

**Step 1: Write the failing test**

Create a test that calls `_build_shared_prefix_graphs` and verifies the graph
structure: shared prefix nodes, independent tail Symbolics, correct count.

Add to `tests/test_blacher_sorting_network.py`:

```python
def test_build_shared_prefix_graphs_structure():
    """_build_shared_prefix_graphs produces graphs with shared prefix nodes."""
    synth = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)
    graphs = synth._build_shared_prefix_graphs()

    assert len(graphs) > 0, "Should produce at least one shared-prefix graph"

    # Each graph should have 3-deep top and bottom (2 prefix + 1 tail)
    for g in graphs:
        assert g.top is not None
        assert g.bottom is not None
        assert isinstance(g.top, IntrinsicNode)
        assert isinstance(g.bottom, IntrinsicNode)

        # Tail nodes should reference shared prefix nodes
        # Collect IntrinsicNode operands from top and bottom tails
        top_intrinsic_ops = [
            v for v in g.top.operands.values() if isinstance(v, IntrinsicNode)
        ]
        bot_intrinsic_ops = [
            v for v in g.bottom.operands.values() if isinstance(v, IntrinsicNode)
        ]

        # Both tails must have exactly 2 IntrinsicNode operands (the prefix)
        assert len(top_intrinsic_ops) == 2, (
            f"Tail should have 2 prefix refs, got {len(top_intrinsic_ops)}"
        )
        assert len(bot_intrinsic_ops) == 2, (
            f"Tail should have 2 prefix refs, got {len(bot_intrinsic_ops)}"
        )

        # Shared by identity: same Python objects
        for top_op, bot_op in zip(top_intrinsic_ops, bot_intrinsic_ops):
            assert top_op is bot_op, (
                "Prefix nodes must be the same Python object (shared by identity)"
            )


def test_shared_prefix_tail_symbolics_differ():
    """Top and bottom tails have independently-named Symbolic operands."""
    synth = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)
    graphs = synth._build_shared_prefix_graphs()

    for g in graphs:
        top_symbolics = {
            (k, v.name)
            for k, v in g.top.operands.items()
            if isinstance(v, Symbolic)
        }
        bot_symbolics = {
            (k, v.name)
            for k, v in g.bottom.operands.items()
            if isinstance(v, Symbolic)
        }
        # Same operand keys, but different Symbolic names
        top_keys = {k for k, _ in top_symbolics}
        bot_keys = {k for k, _ in bot_symbolics}
        assert top_keys == bot_keys, "Top and bottom tails should have same operand keys"

        top_names = {n for _, n in top_symbolics}
        bot_names = {n for _, n in bot_symbolics}
        assert top_names.isdisjoint(bot_names), (
            f"Top and bottom Symbolic names must differ: {top_names} vs {bot_names}"
        )
```

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run pytest tests/test_blacher_sorting_network.py::test_build_shared_prefix_graphs_structure tests/test_blacher_sorting_network.py::test_shared_prefix_tail_symbolics_differ -v`

Expected: FAIL — `GadgetSynthesizer` has no `_build_shared_prefix_graphs` method.

**Step 2: Implement `_build_shared_prefix_graphs`**

Add to `src/gadget_synthesizer.py` after `_build_gadget_graphs` (line ~1171),
before `_wire_hardwired`:

```python
def _build_shared_prefix_graphs(self) -> list[GadgetGraph]:
    """Build shared-prefix (Blacher) gadget graphs.

    Both sides share a prefix of two single-input permutations (one on
    each register), then each side applies a dual-input tail instruction
    with independent Symbolic immediates.  After CSE the shared prefix
    is deduplicated, yielding 4 unified instructions (2 shared + 2 tails).

    Only dual intrinsics with at least one Symbolic operand are used as
    the tail — parameterless duals would produce identical tails on both
    sides (useless).  Both operand orderings are enumerated since
    non-commutative instructions treat a/b differently.
    """
    # Filter duals to those with ≥1 Symbolic operand
    parametric_duals = [
        d for d in self.dual_intrinsics
        if any(isinstance(v, Symbolic) for v in d.operands.values())
    ]
    if not parametric_duals:
        return []

    # Identify register-typed operand keys in the dual template
    # (InputRef positions that will be replaced by prefix nodes)
    def _register_keys(node: IntrinsicNode) -> list[str]:
        return [k for k, v in node.operands.items() if isinstance(v, InputRef)]

    def _symbolic_keys(node: IntrinsicNode) -> list[str]:
        return [k for k, v in node.operands.items() if isinstance(v, Symbolic)]

    graphs: list[GadgetGraph] = []

    for prefix_top in self.single_intrinsics_top:
        for prefix_bot in self.single_intrinsics_bottom:
            for dual in parametric_duals:
                reg_keys = _register_keys(dual)
                sym_keys = _symbolic_keys(dual)

                if len(reg_keys) != 2:
                    continue  # need exactly 2 register operands

                # Both operand orderings
                orderings = [
                    (prefix_top, prefix_bot),
                    (prefix_bot, prefix_top),
                ]
                for reg_a, reg_b in orderings:
                    top_ops = {}
                    bot_ops = {}
                    for key, operand in dual.operands.items():
                        if key == reg_keys[0]:
                            top_ops[key] = reg_a
                            bot_ops[key] = reg_a
                        elif key == reg_keys[1]:
                            top_ops[key] = reg_b
                            bot_ops[key] = reg_b
                        elif isinstance(operand, Symbolic):
                            top_ops[key] = Symbolic(
                                f"{operand.name}_blacher_top", operand.bit_width
                            )
                            bot_ops[key] = Symbolic(
                                f"{operand.name}_blacher_bot", operand.bit_width
                            )
                        else:
                            top_ops[key] = operand
                            bot_ops[key] = operand

                    tail_top = IntrinsicNode(dual.name, top_ops)
                    tail_bot = IntrinsicNode(dual.name, bot_ops)
                    graphs.append(GadgetGraph(top=tail_top, bottom=tail_bot))

    return graphs
```

**Step 3: Run tests to verify they pass**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run pytest tests/test_blacher_sorting_network.py::test_build_shared_prefix_graphs_structure tests/test_blacher_sorting_network.py::test_shared_prefix_tail_symbolics_differ -v`

Expected: PASS

**Step 4: Commit**

---

### Task 2: Wire `_build_shared_prefix_graphs` into `precompute_all_candidates`

**Files:**
- Modify: `src/gadget_synthesizer.py:1217-1229` (`precompute_all_candidates`)

**Step 1: Write the failing test**

Add to `tests/test_blacher_sorting_network.py`:

```python
def test_precompute_includes_shared_prefix_graphs():
    """precompute_all_candidates includes shared-prefix graphs at depth >= 2."""
    synth = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)

    candidates_d1 = synth.precompute_all_candidates(gadget_depth=1)
    candidates_d2 = synth.precompute_all_candidates(gadget_depth=2)

    shared_prefix_count = len(synth._build_shared_prefix_graphs())
    assert shared_prefix_count > 0

    # depth=1 should NOT include shared-prefix graphs
    # depth=2 should include them as additional candidates
    assert len(candidates_d2) >= len(candidates_d1) + shared_prefix_count
```

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run pytest tests/test_blacher_sorting_network.py::test_precompute_includes_shared_prefix_graphs -v`

Expected: FAIL — shared-prefix graphs not yet included.

**Step 2: Modify `precompute_all_candidates`**

In `src/gadget_synthesizer.py`, modify `precompute_all_candidates` (line 1217):

```python
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
```

**Step 3: Run tests to verify they pass**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run pytest tests/test_blacher_sorting_network.py::test_precompute_includes_shared_prefix_graphs -v`

Expected: PASS

**Step 4: Commit**

---

### Task 3: Add post-synthesis instruction-count filter

**Files:**
- Modify: `src/gadget_synthesizer.py:1290-1300` (`_validate_gadget_worker`)

**Step 1: Write the failing test**

Add to `tests/test_blacher_sorting_network.py`:

```python
def test_worker_filters_high_instruction_count():
    """_validate_gadget_worker rejects gadgets with instruction_count > 4."""
    from gadget_synthesizer import _validate_gadget_worker

    synth = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)
    shared_graphs = synth._build_shared_prefix_graphs()
    assert len(shared_graphs) > 0

    # Use Blacher stage 9 input (known to have a solution)
    input_state = VectorState(
        top=[1, 9, 6, 14, 2, 10, 5, 13],
        bottom=[3, 11, 8, 16, 4, 12, 7, 15],
    )
    target_pairs = [
        (1, 2), (9, 10), (3, 4), (11, 12),
        (5, 6), (13, 14), (7, 8), (15, 16),
    ]

    # Find a graph that uses permutexvar + shuffle_ps (known to work)
    test_graph = None
    for g in shared_graphs:
        if (
            g.top.name == "_mm256_shuffle_ps"
            and any(
                isinstance(v, IntrinsicNode)
                and v.name == "_mm256_permutexvar_epi32"
                for v in g.top.operands.values()
            )
        ):
            test_graph = g
            break

    assert test_graph is not None, "Should find a permutexvar+shuffle graph"

    metadata = {
        "input_state": input_state,
        "parent_path": (),
        "stage_idx": 0,
    }
    job = (
        test_graph, input_state, target_pairs,
        vector_machine.AVX2, primitive_type.i32, metadata,
    )

    results, _, _, _, _ = _validate_gadget_worker(job)

    # All returned gadgets must have instruction_count <= 4
    for gadget, _ in results:
        assert gadget.instruction_count() <= 4, (
            f"Gadget with {gadget.instruction_count()} instructions should be filtered"
        )
```

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run pytest tests/test_blacher_sorting_network.py::test_worker_filters_high_instruction_count -v`

Expected: May PASS if CSE always works, or FAIL if any unfiltered gadget leaks through. Either way, we add the filter for safety.

**Step 2: Add filter to `_validate_gadget_worker`**

In `src/gadget_synthesizer.py`, after line 1300 (after `synthesize_gadget_with_symbolic` call):

```python
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

    # Reject gadgets where CSE doesn't collapse to ≤ 4 unified instructions
    gadget_results = [
        (g, os) for g, os in gadget_results if g.instruction_count() <= 4
    ]

    metadata["worker_pid"] = os.getpid()
```

**Step 3: Run tests to verify they pass**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run pytest tests/test_blacher_sorting_network.py::test_worker_filters_high_instruction_count -v`

Expected: PASS

**Step 4: Commit**

---

### Task 4: End-to-end synthesis test via generated graphs

**Files:**
- Modify: `tests/test_blacher_sorting_network.py`

**Step 1: Write the test**

This test verifies that `_build_shared_prefix_graphs` generates graphs that
can solve Blacher stage 9, matching the behavior of the hand-constructed
graph in `test_filtered_synthesis_blacher_stage9_cse`.

```python
def test_generated_shared_prefix_solves_blacher_stage9():
    """Generated shared-prefix graphs can solve Blacher stage 9 with CSE."""
    synth = GadgetSynthesizer(
        vector_machine.AVX2,
        primitive_type.i32,
        intrinsic_filter={"_mm256_permutexvar_epi32", "_mm256_shuffle_ps"},
    )

    input_state = VectorState(
        top=[1, 9, 6, 14, 2, 10, 5, 13],
        bottom=[3, 11, 8, 16, 4, 12, 7, 15],
    )
    target_pairs = [
        (1, 2), (9, 10), (3, 4), (11, 12),
        (5, 6), (13, 14), (7, 8), (15, 16),
    ]

    shared_graphs = synth._build_shared_prefix_graphs()
    assert len(shared_graphs) > 0

    # Try all generated shared-prefix graphs until one solves the stage
    found = False
    for graph in shared_graphs:
        results, _, _ = synth.synthesize_gadget_with_symbolic(
            graph, input_state, target_pairs,
            max_solutions=1, allow_any_lane_order=False,
        )
        if results:
            gadget, output_state = results[0]
            # Must CSE-collapse to 4 instructions
            assert gadget.instruction_count() == 4
            assert len(gadget.top_instructions) == 3
            assert len(gadget.bottom_instructions) == 3
            # Shared prefix: first 2 instructions identical across sides
            for i in range(2):
                assert (
                    gadget.top_instructions[i].args
                    == gadget.bottom_instructions[i].args
                )
            found = True
            break

    assert found, "No generated shared-prefix graph solved Blacher stage 9"
```

**Step 2: Run it**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run pytest tests/test_blacher_sorting_network.py::test_generated_shared_prefix_solves_blacher_stage9 -v`

Expected: PASS (if implementation is correct from Tasks 1-3)

**Step 3: Commit**

---

### Task 5: Full test suite + integration check

**Step 1: Run the full test suite**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run pytest -v`

Expected: All tests PASS, total time < 60 seconds.

**Step 2: Run ruff**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run ruff check .`

Expected: No errors.

**Step 3: Run vulture**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run vulture`

Expected: No new dead code introduced.

**Step 4: Quick integration test (AVX2 i64)**

Run: `cd /Users/dmg/projects/vxsort-cpp/vxsort/smallsort/codegen && uv run python src/bitonic_compiler.py --vector-machine AVX2 --datatype i64 --depth-limit 3 --gadget-depth 2 --top-k 5`

Expected: Completes without error. Shared-prefix candidates are part of the
search space; they may or may not find solutions for i64.

**Step 5: Commit**
