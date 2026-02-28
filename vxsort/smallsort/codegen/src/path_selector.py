"""Unified path selection and gadget scoring for bitonic sorting networks.

This module provides a consistent approach to:
1. Scoring gadgets using latency-based costs (not instruction count)
2. Applying control vector penalties uniformly
3. Representing paths as explicit (node, gadget) selections
4. Allowing A* search to explore different gadgets at each node
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bitonic_super_optimizer import SolutionNode, PermutationGadget
    from cost_model import CostModel


@dataclass
class GadgetScore:
    """Unified scoring for a gadget: latency + control vector penalty.

    Attributes:
        latency_cost: Base latency cost in cycles
        control_vector_count: Number of control vector instructions
        total_score: latency_cost + (control_vector_count * penalty_weight)
    """

    latency_cost: float
    control_vector_count: int
    total_score: float

    def __repr__(self):
        return f"Score(lat={self.latency_cost:.1f}, cv={self.control_vector_count}, total={self.total_score:.1f})"


@dataclass
class PathStep:
    """One step in a path with explicit gadget selection.

    Attributes:
        node: The SolutionNode at this stage
        gadget_index: Index into node.gadgets (which gadget was selected)
        score: The GadgetScore for the selected gadget
    """

    node: "SolutionNode"
    gadget_index: int
    score: GadgetScore

    @property
    def gadget(self) -> "PermutationGadget":
        """Get the selected gadget."""
        return self.node.gadgets[self.gadget_index]

    def __repr__(self):
        return f"Step(stage={self.node.stage}, gadget={self.gadget_index}/{len(self.node.gadgets)}, {self.score})"


@dataclass
class CompletePath:
    """Complete root-to-leaf path with specific gadget choices.

    Attributes:
        steps: List of PathStep from root to leaf
        total_latency: Sum of latency costs across all steps
        total_cv_count: Total control vector instructions in path
        total_score: total_latency + (total_cv_count * penalty_weight)
    """

    steps: list[PathStep]
    total_latency: float
    total_cv_count: int
    total_score: float

    def __repr__(self):
        return f"Path({len(self.steps)} steps, lat={self.total_latency:.1f}, cv={self.total_cv_count}, score={self.total_score:.1f})"


@dataclass
class PathSelectorConfig:
    """Configuration for path selection.

    Attributes:
        cv_penalty_weight: Penalty cycles per control vector instruction
    """

    cv_penalty_weight: float = 0.5  # Cycles per CV instruction


def _is_control_vector_instruction(inst) -> bool:
    """Detect if an instruction uses a control vector (YMM/ZMM register).

    Control vector instructions (less preferred):
    - Use 'op_idx' key (e.g., _mm256_permutexvar_epi32)
    - Use 'mask' key for variable masks (e.g., _mm256_blendv_ps)
    - Use 'b' as control in permutevar instructions

    Immediate-based instructions (preferred):
    - Use 'imm8' key (e.g., _mm256_permute_ps, _mm256_shuffle_ps)
    """
    # Check if instruction uses a control vector
    if "op_idx" in inst.args:
        # permutexvar family - uses control vector
        return True
    elif "mask" in inst.args and inst.intrinsic_name.endswith("v_ps"):
        # blendv_ps - uses variable mask (256-bit control)
        return True
    elif "mask" in inst.args and inst.intrinsic_name.endswith("v_pd"):
        # blendv_pd - uses variable mask (256-bit control)
        return True
    elif "b" in inst.args and "permutevar" in inst.intrinsic_name:
        # permutevar_ps/pd - 'b' is control vector
        return True
    return False


def _count_control_vectors(gadget: "PermutationGadget") -> int:
    """Count control vector instructions in a gadget (deduplicated)."""
    unified, _, _ = gadget.unified_instructions()
    count = 0
    for inst in unified:
        if _is_control_vector_instruction(inst):
            count += 1
    return count


class PathSelector:
    """Selects top-K paths through a solution DAG using unified scoring.

    Uses modified A* search that explores (node, gadget_index) combinations,
    allowing the search to choose different gadgets at the same node based
    on latency + control vector penalty.
    """

    def __init__(
        self, cost_model: "CostModel", config: PathSelectorConfig | None = None
    ):
        """Initialize path selector.

        Args:
            cost_model: CostModel for computing instruction latencies
            config: Configuration for scoring and limits (optional)
        """
        self.cost_model = cost_model
        self.config = config or PathSelectorConfig()

    @staticmethod
    def _find_max_stage(roots: list["SolutionNode"]) -> int:
        """Find the maximum stage number across all reachable nodes."""
        max_stage = 0
        visited: set[int] = set()

        def _traverse(node: "SolutionNode"):
            nonlocal max_stage
            nid = id(node)
            if nid in visited:
                return
            visited.add(nid)
            max_stage = max(max_stage, node.stage)
            for child in node.children:
                _traverse(child)

        for root in roots:
            _traverse(root)
        return max_stage

    def score_gadget(self, gadget: "PermutationGadget") -> GadgetScore:
        """Compute unified score for a gadget.

        Args:
            gadget: The gadget to score

        Returns:
            GadgetScore with latency, CV count, and total score
        """
        latency_cost = self.cost_model.calculate_gadget_cost(gadget)
        cv_count = _count_control_vectors(gadget)
        total_score = latency_cost + (cv_count * self.config.cv_penalty_weight)

        return GadgetScore(
            latency_cost=latency_cost,
            control_vector_count=cv_count,
            total_score=total_score,
        )

    def _compute_admissible_heuristic(
        self, roots: list["SolutionNode"], max_stage: int
    ) -> dict[int, float]:
        """Compute h(n) = minimum cost from node to any leaf.

        For each node, computes the minimum score across all possible
        gadget choices at children. This ensures the heuristic is admissible
        (never overestimates), maintaining A* optimality.

        Dead-end nodes (no children, but not at max_stage) get infinity cost
        so A* never selects paths through them.

        Args:
            roots: List of root nodes to process
            max_stage: The final stage number; only childless nodes at this
                stage are valid leaves

        Returns:
            Dictionary mapping node id -> minimum remaining cost
        """
        min_remaining: dict[int, float] = {}

        def _compute_h(node: "SolutionNode") -> float:
            nid = id(node)
            if nid in min_remaining:
                return min_remaining[nid]

            if not node.children:
                if node.stage == max_stage:
                    # True leaf node - no remaining cost
                    min_remaining[nid] = 0.0
                    return 0.0
                else:
                    # Dead-end at intermediate stage - unreachable
                    min_remaining[nid] = float("inf")
                    return float("inf")

            # For each child, consider all gadgets and find minimum path
            best_cost = float("inf")
            for child in node.children:
                # Find best gadget at child
                for gadget in child.gadgets:
                    score = self.score_gadget(gadget)
                    child_total = score.total_score + _compute_h(child)
                    best_cost = min(best_cost, child_total)

            min_remaining[nid] = best_cost
            return best_cost

        for root in roots:
            _compute_h(root)

        return min_remaining

    def select_top_k_paths(
        self, roots: list["SolutionNode"], top_k: int
    ) -> list[CompletePath]:
        """Select top K cheapest root-to-leaf paths using A* search.

        Modified A* that explores (node, gadget_index) combinations instead
        of just nodes. This allows the search to discover that a more expensive
        gadget at one node might lead to cheaper overall paths.

        Only paths that reach the final stage are considered complete.
        Dead-end nodes at intermediate stages are skipped.

        Args:
            roots: List of root SolutionNode objects
            top_k: Maximum number of paths to return

        Returns:
            List of CompletePath objects, sorted by total_score (best first)
        """
        # Phase 0: Determine the final stage
        max_stage = self._find_max_stage(roots)

        # Phase 1: Compute admissible heuristic (dead-end intermediates get inf)
        min_remaining = self._compute_admissible_heuristic(roots, max_stage)

        # Phase 2: A* priority queue
        # Heap entries: (estimated_total, tiebreaker, cumulative_score, cumulative_latency, cumulative_cv, path_steps)
        counter = 0
        heap: list[tuple[float, int, float, float, int, list[PathStep]]] = []

        for root in roots:
            # Skip roots that can't reach the final stage
            if min_remaining.get(id(root), float("inf")) == float("inf"):
                continue

            # Try all gadgets at root
            for gadget_idx, gadget in enumerate(root.gadgets):
                score = self.score_gadget(gadget)
                step = PathStep(node=root, gadget_index=gadget_idx, score=score)

                # Estimated total = current score + heuristic
                est = score.total_score + min_remaining.get(id(root), 0.0)

                heapq.heappush(
                    heap,
                    (
                        est,
                        counter,
                        score.total_score,
                        score.latency_cost,
                        score.control_vector_count,
                        [step],
                    ),
                )
                counter += 1

        selected: list[CompletePath] = []

        while heap and len(selected) < top_k:
            _est, _tie, cumulative_score, cumulative_lat, cumulative_cv, path_steps = (
                heapq.heappop(heap)
            )
            current_node = path_steps[-1].node

            if not current_node.children:
                if current_node.stage == max_stage:
                    # True complete path reaching the final stage
                    complete_path = CompletePath(
                        steps=path_steps,
                        total_latency=cumulative_lat,
                        total_cv_count=cumulative_cv,
                        total_score=cumulative_score,
                    )
                    selected.append(complete_path)
                # else: dead-end at intermediate stage, discard
            else:
                # Expand to children
                for child in current_node.children:
                    # Skip children that can't reach the final stage
                    if min_remaining.get(id(child), float("inf")) == float("inf"):
                        continue

                    # Try all gadgets at child
                    for gadget_idx, gadget in enumerate(child.gadgets):
                        score = self.score_gadget(gadget)
                        step = PathStep(
                            node=child, gadget_index=gadget_idx, score=score
                        )

                        new_score = cumulative_score + score.total_score
                        new_lat = cumulative_lat + score.latency_cost
                        new_cv = cumulative_cv + score.control_vector_count
                        new_est = new_score + min_remaining.get(id(child), 0.0)

                        heapq.heappush(
                            heap,
                            (
                                new_est,
                                counter,
                                new_score,
                                new_lat,
                                new_cv,
                                path_steps + [step],
                            ),
                        )
                        counter += 1

        return selected

    def prune_to_top_k_paths(
        self, roots: list["SolutionNode"], top_k: int
    ) -> tuple[list["SolutionNode"], list[CompletePath]]:
        """Keep only nodes/edges on top K paths, modifying tree in place.

        Args:
            roots: List of root SolutionNode objects
            top_k: Number of paths to keep

        Returns:
            Tuple of (filtered_roots, selected_paths)
        """
        # Select top K paths
        selected_paths = self.select_top_k_paths(roots, top_k)

        # Build set of kept edges (parent_id, child_id) and kept root ids
        kept_edges = set()
        kept_roots = set()

        for path in selected_paths:
            kept_roots.add(id(path.steps[0].node))
            for i in range(len(path.steps) - 1):
                parent_id = id(path.steps[i].node)
                child_id = id(path.steps[i + 1].node)
                kept_edges.add((parent_id, child_id))

        # Prune tree
        def prune_node(node: "SolutionNode", visited: set | None = None):
            if visited is None:
                visited = set()
            if id(node) in visited:
                return
            visited.add(id(node))

            # Keep only children that are on selected paths
            node.children = [
                child for child in node.children if (id(node), id(child)) in kept_edges
            ]

            for child in node.children:
                prune_node(child, visited)

        # Filter roots and prune
        filtered_roots = [root for root in roots if id(root) in kept_roots]
        for root in filtered_roots:
            prune_node(root)

        return filtered_roots, selected_paths
