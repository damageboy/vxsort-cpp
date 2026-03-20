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

    unique_outputs: set[StateTuple] = field(default_factory=set)
    unique_inputs: set[StateTuple] = field(default_factory=set)
    attempted_pairs: set[TransitionKey] = field(default_factory=set)
    forwarded_outputs: set[StateTuple] = field(default_factory=set)

    consecutive_zero_budgets: int = 0
    attempts: int = 0

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

        if trans_key not in sd.transitions:
            sd.transitions[trans_key] = []
        sd.transitions[trans_key].append(gadget)

        sd.unique_inputs.add(inp_key)
        sd.unique_outputs.add(out_key)

        return True

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

    def weakest_stage(self) -> int:
        """Return the index of the stage with the lowest success rate.

        Ties are broken by earliest stage index.  Stages with zero attempts
        are treated as having a success rate of 0.0.
        """
        best_idx = 0
        best_rate = float("inf")

        for i, sd in enumerate(self.stages):
            if sd.attempts > 0:
                rate = len(sd.unique_outputs) / sd.attempts
            else:
                rate = 0.0

            if rate < best_rate:
                best_rate = rate
                best_idx = i

        return best_idx

    # ------------------------------------------------------------------
    # Path enumeration
    # ------------------------------------------------------------------

    def enumerate_complete_paths(
        self,
        start: StateTuple,
        *,
        max_paths: int | None = None,
        exclude_paths: set[tuple] | None = None,
    ) -> list[list[tuple[StateTuple, StateTuple, PermutationGadget]]]:
        """Enumerate all complete paths from *start* through every stage.

        A complete path has exactly ``len(self.stages)`` steps, one per stage.
        Each step is a ``(input_tuple, output_tuple, gadget)`` triple.

        Parameters
        ----------
        start :
            The input state tuple to begin from (stage 0 input).
        max_paths :
            If set, stop after collecting this many paths.
        exclude_paths :
            Set of path keys to skip.  A path key is
            ``tuple((input_tuple, output_tuple) for each step)``.

        Returns
        -------
        list of paths, where each path is a list of
        ``(input_tuple, output_tuple, gadget)`` triples.
        """
        if not self.stages:
            return []

        exclude = exclude_paths or set()
        results: list[list[tuple[StateTuple, StateTuple, PermutationGadget]]] = []

        # DFS with explicit stack: (stage, current_input, path_so_far)
        stack: list[
            tuple[
                int, StateTuple, list[tuple[StateTuple, StateTuple, PermutationGadget]]
            ]
        ] = [(0, start, [])]

        while stack:
            if max_paths is not None and len(results) >= max_paths:
                break

            stage, inp, path = stack.pop()

            if stage >= len(self.stages):
                # Completed all stages -- check exclusion
                path_key = tuple((step[0], step[1]) for step in path)
                if path_key not in exclude:
                    results.append(path)
                continue

            # Find all transitions from inp at this stage
            transitions = self.get_transitions(stage, inp)
            for out, gadgets in transitions.items():
                for gadget in gadgets:
                    new_path = path + [(inp, out, gadget)]
                    stack.append((stage + 1, out, new_path))

        return results
