"""Tests for checkpoint save/load and iterative tree building."""

import os
import tempfile
from collections import Counter

import pytest
from bitonic_super_optimizer import (
    BitonicSuperVectorizer,
    InstructionSpec,
    PermutationGadget,
    SolutionNode,
    VectorState,
)
from checkpoint import (
    CheckpointConfig,
    _deserialize_node,
    _deserialize_output_tuple,
    _deserialize_parent_path,
    _serialize_node,
    _serialize_output_tuple,
    _serialize_parent_path,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_config,
)
from utils import primitive_type, vector_machine


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestParentPathRoundtrip:
    """Parent path serialization/deserialization."""

    def test_root_path(self):
        assert _serialize_parent_path(()) == "root"
        assert _deserialize_parent_path("root") == ()

    def test_canonical_path(self):
        path = ("canonical", ((1, 2, 3, 4), (5, 6, 7, 8)))
        serialized = _serialize_parent_path(path)
        assert _deserialize_parent_path(serialized) == path

    def test_canonical_path_single_element(self):
        path = ("canonical", ((42,), (99,)))
        serialized = _serialize_parent_path(path)
        assert _deserialize_parent_path(serialized) == path


class TestOutputTupleRoundtrip:
    """Output tuple serialization/deserialization."""

    def test_basic(self):
        t = ((1, 2, 3, 4), (5, 6, 7, 8))
        assert _deserialize_output_tuple(_serialize_output_tuple(t)) == t

    def test_single_element(self):
        t = ((10,), (20,))
        assert _deserialize_output_tuple(_serialize_output_tuple(t)) == t


class TestNodeRoundtrip:
    """SolutionNode serialization/deserialization (without children)."""

    def test_node_with_gadgets(self):
        gadget = PermutationGadget(
            top_instructions=[
                InstructionSpec("_mm256_permute_ps", {"a": "top", "imm8": 0xB1}),
            ],
            bottom_instructions=[
                InstructionSpec(
                    "_mm256_blend_ps", {"a": "top", "b": "bottom", "imm8": 0xAA}
                ),
            ],
            validated=True,
        )
        node = SolutionNode(
            stage=2,
            input_state=VectorState(top=[1, 2, 3, 4], bottom=[5, 6, 7, 8]),
            output_state=VectorState(top=[1, 3, 2, 4], bottom=[5, 7, 6, 8]),
            gadgets=[gadget],
            children=[],
        )
        serialized = _serialize_node(node)
        restored = _deserialize_node(serialized)

        assert restored.stage == node.stage
        assert restored.input_state.top == node.input_state.top
        assert restored.input_state.bottom == node.input_state.bottom
        assert restored.output_state.top == node.output_state.top
        assert restored.output_state.bottom == node.output_state.bottom
        assert len(restored.gadgets) == 1
        assert (
            restored.gadgets[0].top_instructions[0].intrinsic_name
            == "_mm256_permute_ps"
        )
        assert restored.gadgets[0].top_instructions[0].args["imm8"] == 0xB1
        assert (
            restored.gadgets[0].bottom_instructions[0].intrinsic_name
            == "_mm256_blend_ps"
        )
        assert restored.children == []

    def test_node_no_gadgets(self):
        node = SolutionNode(
            stage=0,
            input_state=VectorState(top=[1, 2], bottom=[3, 4]),
            output_state=VectorState(top=[1, 2], bottom=[3, 4]),
            gadgets=[],
            children=[],
        )
        restored = _deserialize_node(_serialize_node(node))
        assert restored.stage == 0
        assert restored.gadgets == []


# ---------------------------------------------------------------------------
# Checkpoint save/load round-trip
# ---------------------------------------------------------------------------


class TestCheckpointSaveLoad:
    """Full checkpoint save → load round-trip."""

    def test_roundtrip(self):
        config = CheckpointConfig(
            num_vecs=2,
            vm="AVX2",
            prim_type="i64",
            gadget_depth=1,
            natural_order=False,
            max_unique_outputs=3,
            depth_limit=None,
        )

        # Build fake per_stage_data
        node0 = SolutionNode(
            stage=0,
            input_state=VectorState(top=[1, 3], bottom=[2, 4]),
            output_state=VectorState(top=[1, 2], bottom=[3, 4]),
            gadgets=[
                PermutationGadget(
                    top_instructions=[],
                    bottom_instructions=[
                        InstructionSpec(
                            "_mm256_permute4x64_epi64", {"a": "bottom", "imm8": 78}
                        ),
                    ],
                    validated=True,
                ),
            ],
            children=[],
        )

        out_tuple = node0.output_state.as_tuple()
        per_stage_data = {
            0: (
                {(): [node0]},  # nodes_by_parent
                {out_tuple: node0.output_state},  # unique_outputs
            ),
        }
        stages_completed = [0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_checkpoint.json.zst")
            save_checkpoint(path, config, per_stage_data, stages_completed, 0)

            assert os.path.exists(path)

            loaded_config, loaded_data, loaded_stages, loaded_last = load_checkpoint(
                path
            )

            assert loaded_config == config
            assert loaded_stages == [0]
            assert loaded_last == 0
            assert 0 in loaded_data

            loaded_nodes_by_parent, loaded_unique_outputs = loaded_data[0]
            assert () in loaded_nodes_by_parent
            loaded_nodes = loaded_nodes_by_parent[()]
            assert len(loaded_nodes) == 1

            restored_node = loaded_nodes[0]
            assert restored_node.stage == 0
            assert restored_node.input_state.top == [1, 3]
            assert restored_node.output_state.top == [1, 2]
            assert len(restored_node.gadgets) == 1
            assert restored_node.gadgets[0].bottom_instructions[0].args["imm8"] == 78

            assert out_tuple in loaded_unique_outputs

    def test_multiple_stages(self):
        config = CheckpointConfig(
            num_vecs=2,
            vm="AVX2",
            prim_type="i64",
            gadget_depth=1,
            natural_order=False,
            max_unique_outputs=3,
            depth_limit=3,
        )

        node0 = SolutionNode(
            stage=0,
            input_state=VectorState(top=[1, 3], bottom=[2, 4]),
            output_state=VectorState(top=[1, 2], bottom=[3, 4]),
            gadgets=[],
            children=[],
        )
        node1 = SolutionNode(
            stage=1,
            input_state=VectorState(top=[1, 2], bottom=[3, 4]),
            output_state=VectorState(top=[1, 3], bottom=[2, 4]),
            gadgets=[],
            children=[],
        )

        out0 = node0.output_state.as_tuple()
        out1 = node1.output_state.as_tuple()
        per_stage_data = {
            0: ({(): [node0]}, {out0: node0.output_state}),
            1: (
                {("canonical", out0): [node1]},
                {out1: node1.output_state},
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json.zst")
            save_checkpoint(path, config, per_stage_data, [0, 1], 1)

            loaded_config, loaded_data, loaded_stages, loaded_last = load_checkpoint(
                path
            )
            assert loaded_stages == [0, 1]
            assert loaded_last == 1
            assert 0 in loaded_data
            assert 1 in loaded_data

            # Check stage 1 has canonical parent path
            s1_nodes, _ = loaded_data[1]
            assert ("canonical", out0) in s1_nodes


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Checkpoint config compatibility checks."""

    def _base_config(self, **overrides):
        defaults = dict(
            num_vecs=2,
            vm="AVX2",
            prim_type="i64",
            gadget_depth=1,
            natural_order=False,
            max_unique_outputs=3,
            depth_limit=None,
        )
        defaults.update(overrides)
        return CheckpointConfig(**defaults)

    def test_identical_configs(self):
        c = self._base_config()
        ok, msg = validate_checkpoint_config(c, c)
        assert ok
        assert msg == "OK"

    def test_higher_depth_limit_ok(self):
        saved = self._base_config(depth_limit=3)
        current = self._base_config(depth_limit=5)
        ok, _ = validate_checkpoint_config(saved, current)
        assert ok

    def test_unlimited_depth_limit_ok(self):
        saved = self._base_config(depth_limit=3)
        current = self._base_config(depth_limit=None)
        ok, _ = validate_checkpoint_config(saved, current)
        assert ok

    def test_lower_depth_limit_rejected(self):
        saved = self._base_config(depth_limit=5)
        current = self._base_config(depth_limit=3)
        ok, msg = validate_checkpoint_config(saved, current)
        assert not ok
        assert "depth_limit" in msg

    def test_vm_mismatch_rejected(self):
        saved = self._base_config(vm="AVX2")
        current = self._base_config(vm="AVX512")
        ok, msg = validate_checkpoint_config(saved, current)
        assert not ok
        assert "vm" in msg

    def test_prim_type_mismatch_rejected(self):
        saved = self._base_config(prim_type="i64")
        current = self._base_config(prim_type="i32")
        ok, msg = validate_checkpoint_config(saved, current)
        assert not ok
        assert "prim_type" in msg

    def test_num_vecs_mismatch_rejected(self):
        saved = self._base_config(num_vecs=2)
        current = self._base_config(num_vecs=4)
        ok, msg = validate_checkpoint_config(saved, current)
        assert not ok
        assert "num_vecs" in msg


# ---------------------------------------------------------------------------
# Integration: iterative produces same results as before
# ---------------------------------------------------------------------------


class TestIterativeMatchesOriginal:
    """Verify the iterative refactor produces correct results."""

    @pytest.mark.slow
    def test_avx2_i64_depth3(self):
        """Full synthesis with AVX2 i64 (4 elements, fast) matches expected structure."""
        sv = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX2)
        roots, _ = sv.build_solution_tree(
            depth_limit=3,
            gadget_depth=1,
            max_unique_outputs=3,
        )
        # Should find at least some solutions
        assert len(roots) > 0
        # Should complete at least some stages
        assert len(sv._stages_completed) > 0

    @pytest.mark.slow
    def test_avx2_i64_depth3_checkpoint_equivalence(self):
        """Property test: checkpointing must not change synthesized AVX2/i64 output."""
        params = {
            "depth_limit": 3,
            "gadget_depth": 1,
            "max_unique_outputs": 3,
        }

        # Baseline: no checkpointing
        sv_no_ckpt = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX2)
        roots_no_ckpt, _ = sv_no_ckpt.build_solution_tree(**params)
        outputs_no_ckpt = _collect_output_states(roots_no_ckpt)
        intrinsics_no_ckpt = _collect_intrinsic_types_by_stage(roots_no_ckpt)

        # Property run: same synthesis parameters, but checkpointing enabled
        with tempfile.TemporaryDirectory() as tmpdir:
            sv_ckpt = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX2)
            roots_ckpt, _ = sv_ckpt.build_solution_tree(
                checkpoint_dir=tmpdir,
                **params,
            )

        outputs_ckpt = _collect_output_states(roots_ckpt)
        intrinsics_ckpt = _collect_intrinsic_types_by_stage(roots_ckpt)

        assert outputs_no_ckpt == outputs_ckpt
        assert intrinsics_no_ckpt == intrinsics_ckpt

    @pytest.mark.slow
    def test_checkpoint_roundtrip_integration(self):
        """Run with checkpointing, then verify checkpoint can be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sv = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX2)
            sv.build_solution_tree(
                depth_limit=3,
                gadget_depth=1,
                max_unique_outputs=3,
                checkpoint_dir=tmpdir,
            )

            # Verify checkpoint files were created
            ckpt_files = [f for f in os.listdir(tmpdir) if f.endswith(".json.zst")]
            assert len(ckpt_files) > 0

            # Load the last checkpoint
            ckpt_files.sort()
            last_ckpt = os.path.join(tmpdir, ckpt_files[-1])
            config, per_stage_data, stages_completed, last_stage = load_checkpoint(
                last_ckpt
            )

            assert config.vm == "AVX2"
            assert config.prim_type == "i64"
            assert config.num_vecs == 2
            assert len(stages_completed) > 0

    @pytest.mark.slow
    def test_resume_produces_same_results(self):
        """Checkpoint at stage 1, resume, compare to full run."""
        # Full run
        sv_full = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX2)
        full_roots, _ = sv_full.build_solution_tree(
            depth_limit=3,
            gadget_depth=1,
            max_unique_outputs=3,
        )

        # Collect full run output states per stage for comparison
        full_outputs = _collect_output_states(full_roots)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run with checkpoint up to depth 1
            sv_ckpt = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX2)
            sv_ckpt.build_solution_tree(
                depth_limit=1,
                gadget_depth=1,
                max_unique_outputs=3,
                checkpoint_dir=tmpdir,
            )

            # Find checkpoint file
            ckpt_files = sorted(
                f for f in os.listdir(tmpdir) if f.endswith(".json.zst")
            )
            assert len(ckpt_files) > 0
            last_ckpt = os.path.join(tmpdir, ckpt_files[-1])

            # Resume from checkpoint
            config, per_stage_data, stages_completed, last_stage = load_checkpoint(
                last_ckpt
            )
            resume_data = {
                "per_stage_data": per_stage_data,
                "last_completed_stage": last_stage,
            }

            sv_resume = BitonicSuperVectorizer(
                2, primitive_type.i64, vector_machine.AVX2
            )
            resumed_roots, _ = sv_resume.build_solution_tree(
                depth_limit=3,
                gadget_depth=1,
                max_unique_outputs=3,
                resume_data=resume_data,
            )

            # Compare: same number of roots and same output structure
            resumed_outputs = _collect_output_states(resumed_roots)
            assert full_outputs == resumed_outputs


def _collect_output_states(roots):
    """Collect all unique output state tuples from a solution tree, keyed by stage."""
    result = {}
    visited = set()

    def _walk(node):
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)
        stage = node.stage
        if stage not in result:
            result[stage] = set()
        result[stage].add(node.output_state.as_tuple())
        for child in node.children:
            _walk(child)

    for root in roots:
        _walk(root)
    return result


def _collect_intrinsic_types_by_stage(roots) -> dict[int, Counter]:
    """Collect intrinsic usage counts per stage from a solution DAG."""
    result: dict[int, Counter] = {}
    visited = set()

    def _walk(node):
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)

        stage_counter = result.setdefault(node.stage, Counter())
        for gadget in node.gadgets:
            for inst in gadget.top_instructions:
                stage_counter[inst.intrinsic_name] += 1
            for inst in gadget.bottom_instructions:
                stage_counter[inst.intrinsic_name] += 1

        for child in node.children:
            _walk(child)

    for root in roots:
        _walk(root)
    return result


class TestNaturalOrderWithCheckpoint:
    """Natural order stage works through checkpoint cycle."""

    @pytest.mark.slow
    def test_natural_order_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sv = BitonicSuperVectorizer(2, primitive_type.i64, vector_machine.AVX2)
            roots, _ = sv.build_solution_tree(
                depth_limit=4,
                gadget_depth=1,
                natural_order=True,
                max_unique_outputs=3,
                checkpoint_dir=tmpdir,
            )

            ckpt_files = [f for f in os.listdir(tmpdir) if f.endswith(".json.zst")]
            assert len(ckpt_files) > 0

            # Verify natural_order is saved in config
            last_ckpt = os.path.join(tmpdir, sorted(ckpt_files)[-1])
            config, _, _, _ = load_checkpoint(last_ckpt)
            assert config.natural_order is True
