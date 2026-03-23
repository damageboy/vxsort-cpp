"""TransitionTable: stage-indexed transition storage for wave-based synthesis.

Replaces SolutionNode during the working phase of synthesis.  Each stage
holds a flat dictionary of (input, output) -> list[PermutationGadget]
transitions, enabling efficient lookup, deduplication, and path
enumeration without recursive tree structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from .bitonic_types import PermutationGadget, VectorState
except ImportError:
    from bitonic_types import PermutationGadget, VectorState  # type: ignore[no-redef]

# Type aliases for readability
StateTuple = tuple[tuple[int, ...], tuple[int, ...]]
TransitionKey = tuple[StateTuple, StateTuple]  # (input_tuple, output_tuple)


@dataclass
class StageData:
    """Per-stage storage for transitions and bookkeeping."""

    # (input_tuple, output_tuple) -> list of gadgets achieving the transition
    transitions: dict[TransitionKey, list[PermutationGadget]] = field(
        default_factory=dict
    )

    # output_tuple -> VectorState (canonical instances)
    unique_outputs: dict[StateTuple, VectorState] = field(default_factory=dict)
    unique_inputs: set[StateTuple] = field(default_factory=set)
    # (input_state_tuple, candidate_index) pairs
    attempted_pairs: set[tuple[StateTuple, int]] = field(default_factory=set)
    forwarded_outputs: set[StateTuple] = field(default_factory=set)

    consecutive_zero_budgets: int = 0
    attempts: int = 0
    dirty: bool = False  # Set when data changes; cleared after checkpoint save

    # Internal: set of sort_keys already stored per transition, for fast dedup
    _gadget_keys: dict[TransitionKey, set[tuple]] = field(
        default_factory=dict, repr=False
    )


class TransitionTable:
    """Stage-indexed transition table for wave-based synthesis.

    Parameters
    ----------
    num_stages : int
        Number of bitonic sort stages (one StageData per stage).
    """

    def __init__(self, num_stages: int) -> None:
        self.stages: list[StageData] = [StageData() for _ in range(num_stages)]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_transition(
        self,
        stage: int,
        input_state: VectorState,
        output_state: VectorState,
        gadget: PermutationGadget,
    ) -> bool:
        """Add a transition to the given stage.

        Deduplicates by ``gadget.sort_key()``.  Returns True if the gadget
        was new (actually added), False if it was a duplicate.
        """
        sd = self.stages[stage]
        inp_key = input_state.as_tuple()
        out_key = output_state.as_tuple()
        trans_key: TransitionKey = (inp_key, out_key)

        gk = gadget.sort_key()

        # Fast dedup check
        if trans_key not in sd._gadget_keys:
            sd._gadget_keys[trans_key] = set()

        if gk in sd._gadget_keys[trans_key]:
            return False

        # New gadget -- store it
        sd._gadget_keys[trans_key].add(gk)
        sd.dirty = True

        if trans_key not in sd.transitions:
            sd.transitions[trans_key] = []
        sd.transitions[trans_key].append(gadget)

        sd.unique_inputs.add(inp_key)
        if out_key not in sd.unique_outputs:
            sd.unique_outputs[out_key] = output_state

        return True

    # ------------------------------------------------------------------
    # Attempt tracking
    # ------------------------------------------------------------------

    def record_attempt(self, stage: int, count: int = 1) -> None:
        """Increment the attempt counter for a stage."""
        sd = self.stages[stage]
        sd.attempts += count
        sd.dirty = True

    def record_attempted_pair(
        self, stage: int, input_tuple: StateTuple, candidate_index: int
    ) -> None:
        """Record that (input_tuple, candidate_index) has been attempted."""
        sd = self.stages[stage]
        sd.attempted_pairs.add((input_tuple, candidate_index))
        sd.dirty = True

    def was_attempted(
        self, stage: int, input_tuple: StateTuple, candidate_index: int
    ) -> bool:
        """Check whether (input_tuple, candidate_index) was already attempted."""
        return (input_tuple, candidate_index) in self.stages[stage].attempted_pairs

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_transitions(
        self, stage: int, input_tuple: StateTuple
    ) -> dict[StateTuple, list[PermutationGadget]]:
        """Return all output -> gadgets mappings for a given input at a stage.

        Returns an empty dict if no transitions exist for that input.
        """
        sd = self.stages[stage]
        result: dict[StateTuple, list[PermutationGadget]] = {}
        for (inp, out), gadgets in sd.transitions.items():
            if inp == input_tuple:
                result[out] = gadgets
        return result

    def get_all_transitions(
        self, stage: int
    ) -> dict[TransitionKey, list[PermutationGadget]]:
        """Return all transitions for a stage as a dict keyed by (input, output)."""
        return self.stages[stage].transitions

    def get_unique_outputs(self, stage: int) -> dict[StateTuple, VectorState]:
        """Return the unique output states for a stage (tuple -> VectorState)."""
        return self.stages[stage].unique_outputs

    def unique_output_count(self, stage: int) -> int:
        """Return the number of unique output states for a stage."""
        return len(self.stages[stage].unique_outputs)

    # ------------------------------------------------------------------
    # Forwarding
    # ------------------------------------------------------------------

    def get_unforwarded_outputs(
        self, stage: int
    ) -> list[tuple[StateTuple, VectorState]]:
        """Return outputs not yet in forwarded_outputs."""
        sd = self.stages[stage]
        result: list[tuple[StateTuple, VectorState]] = []
        for out_t, vs in sd.unique_outputs.items():
            if out_t not in sd.forwarded_outputs:
                result.append((out_t, vs))
        return result

    def mark_forwarded(self, stage: int, output_tuples: list[StateTuple]) -> None:
        """Add tuples to the stage's forwarded_outputs set."""
        self.stages[stage].forwarded_outputs.update(output_tuples)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stage_stats(self, stage: int) -> dict:
        """Return summary statistics for a stage.

        Keys: attempts, distinct_outputs, success_rate, transition_count,
        total_gadgets.
        """
        sd = self.stages[stage]
        transition_count = len(sd.transitions)
        total_gadgets = sum(len(gs) for gs in sd.transitions.values())
        distinct_outputs = len(sd.unique_outputs)
        attempts = sd.attempts
        success_rate = distinct_outputs / attempts if attempts > 0 else 0.0

        return {
            "attempts": attempts,
            "distinct_outputs": distinct_outputs,
            "success_rate": success_rate,
            "transition_count": transition_count,
            "total_gadgets": total_gadgets,
        }

    def weakest_stage(self, exclude_exhausted: set[int] | None = None) -> int | None:
        """Return the stage most in need of exploration.

        Selection criteria (in priority order):
        1. Fewest distinct outputs (the bottleneck stage)
        2. Lowest success rate (distinct_outputs / attempts) as tiebreaker
        3. Earliest stage index as final tiebreaker

        Parameters
        ----------
        exclude_exhausted :
            Stage indices to skip.  If all stages are excluded (or the
            table is empty), returns ``None``.
        """
        excluded = exclude_exhausted or set()
        best_idx: int | None = None
        best_key = (float("inf"), float("inf"))

        for i, sd in enumerate(self.stages):
            if i in excluded:
                continue

            n_outputs = len(sd.unique_outputs)
            rate = n_outputs / sd.attempts if sd.attempts > 0 else 0.0
            key = (n_outputs, rate)

            if key < best_key:
                best_key = key
                best_idx = i

        return best_idx

    # ------------------------------------------------------------------
    # Path enumeration
    # ------------------------------------------------------------------

    def enumerate_complete_paths(
        self,
        start: StateTuple | None = None,
        *,
        max_paths: int | None = None,
        exclude_paths: set[tuple] | None = None,
    ) -> list[list[tuple[int, StateTuple, StateTuple]]]:
        """Enumerate complete paths through every stage.

        A complete path has exactly ``len(self.stages)`` steps, one per
        transition (not per gadget).  Each step is a
        ``(stage_index, input_tuple, output_tuple)`` triple.

        Parameters
        ----------
        start :
            If given, only paths beginning with this stage-0 input are
            returned.  If ``None``, paths from every stage-0 input are
            enumerated.
        max_paths :
            If set, stop after collecting this many paths.
        exclude_paths :
            Set of path keys to skip.  A path key is
            ``tuple((stage, input_tuple, output_tuple) for each step)``.

        Returns
        -------
        list of paths, where each path is a list of
        ``(stage_index, input_tuple, output_tuple)`` triples.
        """
        if not self.stages:
            return []

        exclude = exclude_paths or set()
        results: list[list[tuple[int, StateTuple, StateTuple]]] = []

        def _dfs(
            stage: int,
            current_path: list[tuple[int, StateTuple, StateTuple]],
            required_input: StateTuple | None,
        ) -> None:
            if max_paths is not None and len(results) >= max_paths:
                return

            if stage >= len(self.stages):
                path_key = tuple(current_path)
                if path_key not in exclude:
                    results.append(list(current_path))
                return

            sd = self.stages[stage]
            for in_t, out_t in sd.transitions:
                if required_input is not None and in_t != required_input:
                    continue
                current_path.append((stage, in_t, out_t))
                _dfs(stage + 1, current_path, out_t)
                current_path.pop()
                if max_paths is not None and len(results) >= max_paths:
                    return

        _dfs(0, [], start)
        return results
