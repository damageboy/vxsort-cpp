"""Tests for end-to-end Z3 verification of bitonic sort paths."""

import os

import pytest
from bitonic_super_optimizer import (
    PermutationGadget,
    SolutionNode,
)
from bitonic_verifier import BitonicPathVerifier, load_solutions_from_json
from cost_model import CostModel
from path_selector import CompletePath, PathSelector, PathStep
from util.enums import primitive_type, vector_machine

FIXTURE_DIR = os.path.dirname(__file__)

# Full Z3 proofs take ~20s each; mark them so `pytest -m "not slow"` skips them.
slow = pytest.mark.slow


def _load_and_select(fixture_name, top_k=1):
    """Load solutions from a JSON fixture and select the top-k paths."""
    path = os.path.join(FIXTURE_DIR, fixture_name)
    bundle = load_solutions_from_json(path)
    cost_model = CostModel("generic")
    selector = PathSelector(cost_model)
    paths = selector.select_top_k_paths(bundle.roots, top_k)
    assert len(paths) >= 1, f"No paths found in {fixture_name}"
    return bundle, paths


class TestLoadSolutions:
    """Test JSON fixture loading (fast, no Z3 proofs)."""

    def test_load_i64(self):
        """Loading the i64 fixture produces a valid solution tree."""
        bundle = load_solutions_from_json(
            os.path.join(FIXTURE_DIR, "fixture_2xAVX2_i64.json")
        )
        assert len(bundle.roots) >= 1
        root = bundle.roots[0]
        assert root.stage == 0
        assert len(root.gadgets) >= 1
        assert len(root.children) >= 1

    def test_load_and_select_path(self):
        """Path selection on loaded tree produces valid paths."""
        _, paths = _load_and_select("fixture_2xAVX2_i64.json")
        path = paths[0]
        assert len(path.steps) >= 2, "Path should have multiple stages"
        assert path.steps[0].node.stage == 0

    def test_counterexample_on_bad_gadget(self):
        """Replace last stage with identity, verify it fails with counterexample."""
        _, paths = _load_and_select("fixture_2xAVX2_i64.json")
        path = paths[0]

        last_idx = len(path.steps) - 1
        identity_gadget = PermutationGadget(
            top_instructions=[], bottom_instructions=[], validated=True
        )
        broken_steps = []
        for i, step in enumerate(path.steps):
            if i == last_idx:
                broken_node = SolutionNode(
                    stage=step.node.stage,
                    input_state=step.node.input_state,
                    output_state=step.node.input_state,  # identity
                    gadgets=[identity_gadget],
                    children=[],
                )
                broken_steps.append(
                    PathStep(node=broken_node, gadget_index=0, score=step.score)
                )
            else:
                broken_steps.append(step)

        broken_path = CompletePath(
            steps=broken_steps,
            total_latency=path.total_latency,
            total_cv_count=path.total_cv_count,
            total_score=path.total_score,
        )

        verifier = BitonicPathVerifier(vector_machine.AVX2, primitive_type.i64)
        result = verifier.verify_path(broken_path)
        assert not result.verified, "Broken path should fail verification"
        assert result.counterexample is not None, "Should provide counterexample"


class TestVerifierProofs:
    """Full Z3 verification proofs (~20s each). Skipped by default."""

    @slow
    def test_known_good_path_i64(self):
        """Best i64 path verifies as a correct sorting network."""
        _, paths = _load_and_select("fixture_2xAVX2_i64.json")
        verifier = BitonicPathVerifier(vector_machine.AVX2, primitive_type.i64)
        result = verifier.verify_path(paths[0])
        assert result.verified, f"Path failed: {result.counterexample}"

    @slow
    def test_natural_order_i64(self):
        """Natural-order i64 path verifies correctly."""
        _, paths = _load_and_select("fixture_2xAVX2_i64_natural.json")
        verifier = BitonicPathVerifier(vector_machine.AVX2, primitive_type.i64)
        result = verifier.verify_path(paths[0], natural_order=True)
        assert result.verified, f"Natural-order failed: {result.counterexample}"
