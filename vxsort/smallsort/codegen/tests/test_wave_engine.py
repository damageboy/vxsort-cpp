"""Tests for WaveEngine: wave-based iterative synthesis orchestrator."""

import math
import queue

from utils import primitive_type, vector_machine
from wave_engine import WaveConfig, WaveEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fast_config(**overrides) -> WaveConfig:
    """Build a WaveConfig for fast integration testing (AVX2/i64, small budgets)."""
    defaults = dict(
        num_vecs=2,
        vm=vector_machine.AVX2,
        prim_type=primitive_type.i64,
        gadget_depth=1,
        natural_order=False,
        retroactive_input=False,
        wave_attempts=100,
        wave_outputs=3,
        max_paths_per_wave=50,
        target_cpus=[],
        max_workers=4,
        max_tasks_per_child=100,
        depth2_threshold=0.1,
        max_waves=1,
        top_k=5,
    )
    defaults.update(overrides)
    return WaveConfig(**defaults)


class _RecordingRuntimeSession:
    """Capture wave runtime adapter calls for integration-style testing."""

    def __init__(self, selected_stage: int) -> None:
        self._selected_stage = selected_stage
        self.selected_stage_writes: list[int] = []
        self.stage_metrics = []
        self.memory_lines: list[str] = []
        self.instruction_stats = []
        self.logs: list[str] = []
        self.statuses: list[str | None] = []

    @property
    def selected_stage(self) -> int:
        return self._selected_stage

    @selected_stage.setter
    def selected_stage(self, value: int) -> None:
        self.selected_stage_writes.append(value)
        self._selected_stage = value

    def update_stage_metrics(self, stage_metrics) -> None:
        self.stage_metrics.append(stage_metrics)

    def update_memory_line(self, memory_text: str) -> None:
        self.memory_lines.append(memory_text)

    def update_instruction_stats(self, snapshot_by_stage) -> None:
        self.instruction_stats.append(snapshot_by_stage)

    def log(self, line: str) -> None:
        self.logs.append(line)

    def set_status(self, text: str | None) -> None:
        self.statuses.append(text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWaveEngineInit:
    """WaveEngine construction and initial state."""

    def test_creates_initial_state(self):
        config = _fast_config()
        engine = WaveEngine(config)

        # AVX2 i64 -> 4 elements per vector, 2 vectors -> 8 total elements
        assert engine.elements_per_vector == 4
        assert engine.total_elements == 8

        # Initial state should have 4 elements per vector
        assert len(engine.initial_state.top) == 4
        assert len(engine.initial_state.bottom) == 4

    def test_stages_created(self):
        config = _fast_config()
        engine = WaveEngine(config)

        # 8-element bitonic sort has multiple stages
        assert len(engine.all_stages) > 0
        assert len(engine.tt.stages) == len(engine.all_stages)

    def test_natural_order_adds_stage(self):
        config_no = _fast_config(natural_order=False)
        config_yes = _fast_config(natural_order=True)
        engine_no = WaveEngine(config_no)
        engine_yes = WaveEngine(config_yes)

        assert len(engine_yes.all_stages) == len(engine_no.all_stages) + 1

    def test_candidates_precomputed(self):
        config = _fast_config()
        engine = WaveEngine(config)

        assert len(engine.shallow_candidates) > 0
        assert len(engine._candidates_with_index) > 0


def test_wave_engine_emits_stage_updates_without_mutating_selected_stage():
    config = _fast_config()
    engine = WaveEngine(config)

    from bitonic_types import InstructionSpec, PermutationGadget, VectorState

    input_state = engine.initial_state
    output_state = VectorState(
        top=[x + 10 for x in input_state.top],
        bottom=[x + 10 for x in input_state.bottom],
    )
    gadget = PermutationGadget(
        top_instructions=[InstructionSpec("test_add", {"src": "top"})],
        bottom_instructions=[],
        validated=True,
    )
    engine.tt.add_transition(0, input_state, output_state, gadget)
    engine.tt.record_attempt(0, count=4)
    engine._stage_attempts[0] = 4
    engine.instruction_stats.record_attempt(1, ["test_add"], "unique")

    session = _RecordingRuntimeSession(selected_stage=2)

    engine._sync_runtime_session(session)

    assert session.selected_stage == 2
    assert session.selected_stage_writes == []
    assert session.stage_metrics
    assert len(session.stage_metrics[-1]) == len(engine.all_stages)
    assert session.stage_metrics[-1][0].stage_idx == 0
    assert session.stage_metrics[-1][0].attempts == 4
    assert session.stage_metrics[-1][0].unique == 1
    assert session.memory_lines[-1].startswith("Memory:")
    assert session.instruction_stats[-1][1]["test_add"].attempts_unique == 1


class TestFirstWaveDiscoveries:
    """First wave should discover transitions at stage 0."""

    def test_first_wave_discovers_transitions(self):
        config = _fast_config(wave_attempts=500, wave_outputs=50, max_waves=1)
        engine = WaveEngine(config)
        engine.run()

        stage0_outputs = engine.tt.unique_output_count(0)
        assert (
            stage0_outputs > 0
        ), f"Expected stage 0 to have discoveries, got {stage0_outputs}"


class TestMultiWaveProgress:
    """Multiple waves should make progress."""

    def test_three_waves_make_progress(self):
        config = _fast_config(max_waves=3, wave_attempts=500, wave_outputs=50)
        engine = WaveEngine(config)

        summary = engine.run()

        assert summary["wave_count"] == 3
        assert not summary["interrupted"]

        # Stage 0 should have outputs after 3 waves
        stage0_outputs = engine.tt.unique_output_count(0)
        assert stage0_outputs > 0


class TestTargetingHeuristic:
    """Target selection picks the weakest stage."""

    def test_targets_weakest_stage(self):
        config = _fast_config()
        engine = WaveEngine(config)

        # Manually populate stages to control which is weakest.
        # We need to give ALL stages some attempts to avoid empty stages
        # having rate 0.0 and being chosen instead.
        from bitonic_types import InstructionSpec, PermutationGadget, VectorState

        dummy_gadget = PermutationGadget(
            top_instructions=[InstructionSpec("test", {"a": "top"})],
            bottom_instructions=[],
            validated=True,
        )

        inp = engine.initial_state
        num_stages = len(engine.all_stages)

        # Stage 0: 10 attempts, 5 outputs => rate 0.5
        for i in range(5):
            out = VectorState(
                top=[x + i * 10 for x in inp.top],
                bottom=[x + i * 10 for x in inp.bottom],
            )
            engine.tt.add_transition(0, inp, out, dummy_gadget)
        engine.tt.record_attempt(0, count=10)

        # Stage 1: 10 attempts, 1 output => rate 0.1 (weakest among populated)
        out1 = VectorState(top=[100, 200, 300, 400], bottom=[500, 600, 700, 800])
        engine.tt.add_transition(1, inp, out1, dummy_gadget)
        engine.tt.record_attempt(1, count=10)

        # All other stages: exhaust them so they are excluded
        for s in range(2, num_stages):
            engine.exhausted_stages.add(s)

        # Skip wave 0 (first wave always targets stage 0)
        engine.wave_count = 1

        target = engine._select_target_stage()
        # Stage 1 has worse success rate (0.1) than stage 0 (0.5)
        assert target == 1


class TestBubbleUp:
    """Bubble-up: stuck stages redirect upstream."""

    def test_bubble_up_when_stuck(self):
        config = _fast_config()
        engine = WaveEngine(config)

        # Stage 1 has consecutive_zero_budgets >= 2
        engine.tt.stages[1].consecutive_zero_budgets = 3
        # Make stage 1 the weakest by giving it low success rate
        engine.tt.record_attempt(1, count=10)
        # Stage 0 has some outputs (better rate)
        from bitonic_types import InstructionSpec, PermutationGadget, VectorState

        dummy_gadget = PermutationGadget(
            top_instructions=[InstructionSpec("test", {"a": "top"})],
            bottom_instructions=[],
            validated=True,
        )
        inp = engine.initial_state
        out = VectorState(top=[10, 20, 30, 40], bottom=[50, 60, 70, 80])
        engine.tt.add_transition(0, inp, out, dummy_gadget)
        engine.tt.record_attempt(0, count=10)

        engine.wave_count = 1  # skip first-wave override

        target = engine._select_target_stage()
        # Stage 1 is weakest (rate 0.0 < stage 0's 0.1)
        # but it's stuck, so bubble up to stage 0
        assert target == 0


class TestExportToSolutionNodes:
    """export_to_solution_nodes produces valid SolutionNode tree."""

    def test_export_empty(self):
        config = _fast_config()
        engine = WaveEngine(config)
        nodes = engine.export_to_solution_nodes()
        assert nodes == []

    def test_export_with_data(self):
        config = _fast_config()
        engine = WaveEngine(config)

        from bitonic_types import InstructionSpec, PermutationGadget, VectorState

        dummy_gadget = PermutationGadget(
            top_instructions=[InstructionSpec("test", {"a": "top"})],
            bottom_instructions=[],
            validated=True,
        )

        inp = engine.initial_state
        out0 = VectorState(top=[10, 20, 30, 40], bottom=[50, 60, 70, 80])
        out1 = VectorState(top=[100, 200, 300, 400], bottom=[500, 600, 700, 800])

        engine.tt.add_transition(0, inp, out0, dummy_gadget)
        engine.tt.add_transition(1, out0, out1, dummy_gadget)

        nodes = engine.export_to_solution_nodes()
        assert len(nodes) == 1  # one root node at stage 0
        assert nodes[0].stage == 0
        assert nodes[0].output_state.top == [10, 20, 30, 40]

        # Children should link to stage 1
        assert len(nodes[0].children) == 1
        assert nodes[0].children[0].stage == 1


class TestInstructionStatsDrain:
    """_drain_results should classify attempt outcomes for instruction stats."""

    def test_unique_output_attempt_is_recorded(self):
        config = _fast_config()
        engine = WaveEngine(config)

        from bitonic_types import InstructionSpec, PermutationGadget, VectorState

        gadget = PermutationGadget(
            top_instructions=[InstructionSpec("test_add", {"src": "top"})],
            bottom_instructions=[],
            validated=True,
        )
        input_state = engine.initial_state
        output_state = VectorState(
            top=[x + 1 for x in input_state.top],
            bottom=[x + 1 for x in input_state.bottom],
        )

        completion_queue: queue.Queue = queue.Queue()
        completion_queue.put(
            (
                "ok",
                0,
                (
                    [(gadget, output_state)],
                    input_state,
                    {"candidate_index": 7, "instruction_keys": ["test_add"]},
                    None,
                    None,
                ),
            )
        )
        engine._in_flight[0] = 1

        drained = engine._drain_results(completion_queue)

        assert drained == 1
        by_stage = engine.instruction_stats.snapshot_by_stage()
        assert by_stage[0]["test_add"].attempts_total == 1
        assert by_stage[0]["test_add"].attempts_no_valid == 0
        assert by_stage[0]["test_add"].attempts_valid == 0
        assert by_stage[0]["test_add"].attempts_unique == 1

    def test_valid_non_unique_attempt_is_recorded(self):
        config = _fast_config()
        engine = WaveEngine(config)

        from bitonic_types import InstructionSpec, PermutationGadget, VectorState

        gadget = PermutationGadget(
            top_instructions=[InstructionSpec("test_add", {"src": "top"})],
            bottom_instructions=[],
            validated=True,
        )
        input_state = engine.initial_state
        output_state = VectorState(
            top=[x + 2 for x in input_state.top],
            bottom=[x + 2 for x in input_state.bottom],
        )
        engine.tt.add_transition(0, input_state, output_state, gadget)

        completion_queue: queue.Queue = queue.Queue()
        completion_queue.put(
            (
                "ok",
                0,
                (
                    [(gadget, output_state)],
                    input_state,
                    {"candidate_index": 11, "instruction_keys": ["test_add"]},
                    None,
                    None,
                ),
            )
        )
        engine._in_flight[0] = 1

        drained = engine._drain_results(completion_queue)

        assert drained == 1
        by_stage = engine.instruction_stats.snapshot_by_stage()
        assert by_stage[0]["test_add"].attempts_total == 1
        assert by_stage[0]["test_add"].attempts_no_valid == 0
        assert by_stage[0]["test_add"].attempts_valid == 1
        assert by_stage[0]["test_add"].attempts_unique == 0

    def test_no_valid_attempt_is_recorded(self):
        config = _fast_config()
        engine = WaveEngine(config)
        input_state = engine.initial_state

        completion_queue: queue.Queue = queue.Queue()
        completion_queue.put(
            (
                "ok",
                0,
                (
                    [],
                    input_state,
                    {"candidate_index": 13, "instruction_keys": ["test_add"]},
                    None,
                    None,
                ),
            )
        )
        engine._in_flight[0] = 1

        drained = engine._drain_results(completion_queue)

        assert drained == 1
        by_stage = engine.instruction_stats.snapshot_by_stage()
        assert by_stage[0]["test_add"].attempts_total == 1
        assert by_stage[0]["test_add"].attempts_no_valid == 1
        assert by_stage[0]["test_add"].attempts_valid == 0
        assert by_stage[0]["test_add"].attempts_unique == 0


class TestRetroactiveInput:
    """Tests for --retroactive-input pre-population of stage 0."""

    def test_prepopulates_stage0_with_n_factorial_permutations(self):
        config = _fast_config(retroactive_input=True)
        engine = WaveEngine(config)

        # AVX2 i64: 4 pairs -> 4! = 24 permutations
        n_pairs = engine.elements_per_vector  # 4
        expected_count = math.factorial(n_pairs)  # 24

        stage0_outputs = engine.tt.get_unique_outputs(0)
        assert len(stage0_outputs) == expected_count

    def test_stage0_gadgets_are_identity(self):
        config = _fast_config(retroactive_input=True)
        engine = WaveEngine(config)

        all_trans = engine.tt.get_all_transitions(0)
        for (_inp_t, _out_t), gadgets in all_trans.items():
            assert len(gadgets) == 1
            g = gadgets[0]
            assert g.top_instructions == []
            assert g.bottom_instructions == []
            assert g.validated is True

    def test_stage0_input_equals_output(self):
        config = _fast_config(retroactive_input=True)
        engine = WaveEngine(config)

        all_trans = engine.tt.get_all_transitions(0)
        for (inp_t, out_t), _gadgets in all_trans.items():
            assert inp_t == out_t

    def test_stage0_marked_exhausted(self):
        config = _fast_config(retroactive_input=True)
        engine = WaveEngine(config)

        assert 0 in engine.exhausted_stages

    def test_no_prepopulation_without_flag(self):
        config = _fast_config(retroactive_input=False)
        engine = WaveEngine(config)

        stage0_outputs = engine.tt.get_unique_outputs(0)
        assert len(stage0_outputs) == 0
        assert 0 not in engine.exhausted_stages


class TestEndToEnd:
    """Full pipeline: synthesis -> checkpoint -> resume -> export."""

    def test_avx2_i64_full_pipeline(self, tmp_path):
        """Full pipeline: synthesis -> checkpoint -> resume -> export."""
        config = _fast_config(
            wave_attempts=1000,
            wave_outputs=20,
            max_paths_per_wave=50,
            max_waves=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )

        # Run 2 waves
        engine = WaveEngine(config)
        engine.run()
        assert engine.wave_count == 2

        # Verify checkpoint exists
        import os

        assert os.path.exists(str(tmp_path / "ckpt" / "master.json"))

        # Resume and run 1 more wave
        config2 = _fast_config(
            wave_attempts=1000,
            wave_outputs=20,
            max_paths_per_wave=50,
            max_waves=3,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        engine2 = WaveEngine(config2)
        engine2.resume(str(tmp_path / "ckpt"))
        engine2.run()
        assert engine2.wave_count == 3

        # Export to SolutionNode tree
        solutions = engine2.export_to_solution_nodes()
        assert len(solutions) > 0
