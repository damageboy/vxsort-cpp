#!/usr/bin/env python3
"""Unit tests for PathSelector and unified gadget scoring."""
import pytest
from dataclasses import dataclass
from path_selector import (
    PathSelector,
    PathSelectorConfig,
    GadgetScore,
    PathStep,
    CompletePath,
    _is_control_vector_instruction,
    _count_control_vectors,
)
from bitonic_super_optimizer import (
    PermutationGadget,
    SolutionNode,
    VectorState,
    InstructionSpec,
)
from cost_model import CostModel
@dataclass
class MockInstruction:
    """Mock instruction for testing control vector detection."""
    intrinsic_name: str
    args: dict
def test_is_control_vector_instruction():
    """Test detection of control vector instructions."""
    # Control vector instructions
    assert _is_control_vector_instruction(
        MockInstruction("_mm256_permutexvar_epi32", {"op_idx": "idx"})
    )
    assert _is_control_vector_instruction(
        MockInstruction("_mm256_blendv_ps", {"mask": "m"})
    )
    assert _is_control_vector_instruction(
        MockInstruction("_mm256_blendv_pd", {"mask": "m"})
    )
    assert _is_control_vector_instruction(
        MockInstruction("_mm256_permutevar_ps", {"b": "ctrl"})
    )
    # Immediate-based instructions (not control vector)
    assert not _is_control_vector_instruction(
        MockInstruction("_mm256_permute_ps", {"imm8": 0xD8})
    )
    assert not _is_control_vector_instruction(
        MockInstruction("_mm256_shuffle_ps", {"imm8": 0x88})
    )
    assert not _is_control_vector_instruction(
        MockInstruction("_mm256_blend_ps", {"imm8": 0xF0})
    )
def test_count_control_vectors():
    """Test counting control vectors in a gadget."""
    # Create gadget with mix of control vector and immediate instructions
    top_insts = [
        InstructionSpec(
            intrinsic_name="_mm256_permutexvar_epi32",
            args={"a": "top", "op_idx": "idx"}
        ),
        InstructionSpec(
            intrinsic_name="_mm256_permute_ps",
            args={"a": "tmp1", "imm8": 0xD8}
        ),
    ]
    bottom_insts = [
        InstructionSpec(
            intrinsic_name="_mm256_blendv_ps",
            args={"a": "bottom", "b": "top", "mask": "m"}
        ),
    ]
    gadget = PermutationGadget(
        top_instructions=top_insts,
        bottom_instructions=bottom_insts,
        validated=True
    )
    # Should count 2 control vectors: permutexvar and blendv
    assert _count_control_vectors(gadget) == 2
def test_gadget_score_computation():
    """Test GadgetScore computation with latency and CV penalty."""
    cost_model = CostModel("generic")
    config = PathSelectorConfig(cv_penalty_weight=0.5)
    selector = PathSelector(cost_model, config)
    # Create gadget with 1 control vector instruction
    top_insts = [
        InstructionSpec(
            intrinsic_name="_mm256_permutexvar_epi32",  # latency 3.0, CV
            args={"a": "top", "op_idx": "idx"},
        ),
    ]
    bottom_insts = [
        InstructionSpec(
            intrinsic_name="_mm256_permute_ps",  # latency 1.0, not CV
            args={"a": "bottom", "imm8": 0xD8},
        ),
    ]
    gadget = PermutationGadget(
        top_instructions=top_insts,
        bottom_instructions=bottom_insts,
        validated=True
    )
    score = selector.score_gadget(gadget)
    # Expected: latency 3.0 + 1.0 = 4.0, CV count = 1
    # Total = 4.0 + (1 * 0.5) = 4.5
    assert score.latency_cost == 4.0
    assert score.control_vector_count == 1
    assert score.total_score == 4.5
def test_path_selector_single_path():
    """Test PathSelector with a simple linear tree (no branching)."""
    cost_model = CostModel("generic")
    selector = PathSelector(cost_model)
    # Create simple linear path: root -> child1 -> leaf
    # Each node has one gadget
    leaf_state = VectorState(top=[0, 1, 2, 3], bottom=[4, 5, 6, 7])
    leaf = SolutionNode(
        stage=2,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_permute_ps",
                        args={"a": "top", "imm8": 0xD8},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[]
    )
    child1_state = VectorState(top=[0, 1, 2, 3], bottom=[4, 5, 6, 7])
    child1 = SolutionNode(
        stage=1,
        input_state=child1_state,
        output_state=child1_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_shuffle_ps",
                        args={"a": "top", "b": "bottom", "imm8": 0x88},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[leaf]
    )
    root_state = VectorState(top=[0, 1, 2, 3], bottom=[4, 5, 6, 7])
    root = SolutionNode(
        stage=0,
        input_state=root_state,
        output_state=root_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_permute_ps",
                        args={"a": "top", "imm8": 0xD8},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[child1]
    )
    # Select top 1 path
    paths = selector.select_top_k_paths([root], top_k=1)
    assert len(paths) == 1
    path = paths[0]
    assert len(path.steps) == 3
    assert path.steps[0].node == root
    assert path.steps[1].node == child1
    assert path.steps[2].node == leaf
    # All are permute_ps (latency 1.0), no CVs
    assert path.total_latency == 3.0
    assert path.total_cv_count == 0
def test_path_selector_multiple_gadgets_per_node():
    """Test that A* explores different gadget choices at same node."""
    cost_model = CostModel("generic")
    config = PathSelectorConfig(cv_penalty_weight=0.5)
    selector = PathSelector(cost_model, config)
    # Create tree where root has 2 gadgets with different costs
    leaf_state = VectorState(top=[0, 1, 2, 3], bottom=[4, 5, 6, 7])
    leaf = SolutionNode(
        stage=1,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[]
    )
    root_state = VectorState(top=[0, 1, 2, 3], bottom=[4, 5, 6, 7])
    root = SolutionNode(
        stage=0,
        input_state=root_state,
        output_state=root_state,
        gadgets=[
            # Gadget 0: Low latency, no CV (cost = 1.0)
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_permute_ps",
                        args={"a": "top", "imm8": 0xD8},
                    )
                ],
                bottom_instructions=[],
                validated=True
            ),
            # Gadget 1: High latency with CV (cost = 3.0 + 0.5 = 3.5)
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_permutexvar_epi32",
                        args={"a": "top", "op_idx": "idx"},
                    )
                ],
                bottom_instructions=[],
                validated=True
            ),
        ],
        children=[leaf]
    )
    # Select top 2 paths
    paths = selector.select_top_k_paths([root], top_k=2)
    assert len(paths) == 2
    # First path should use gadget 0 (lower cost)
    assert paths[0].steps[0].gadget_index == 0
    assert paths[0].total_score == 1.0
    # Second path should use gadget 1 (higher cost)
    assert paths[1].steps[0].gadget_index == 1
    assert paths[1].total_score == 3.5
def test_path_selector_dag_structure():
    """Test PathSelector with DAG (shared children)."""
    cost_model = CostModel("generic")
    selector = PathSelector(cost_model)
    # Create DAG where two roots share a common child
    #     root1 --\
    #              --> shared_child --> leaf
    #     root2 --/
    leaf_state = VectorState(top=[0, 1, 2, 3], bottom=[4, 5, 6, 7])
    leaf = SolutionNode(
        stage=2,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_permute_ps",
                        args={"a": "top", "imm8": 0xD8},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[]
    )
    shared_child = SolutionNode(
        stage=1,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_shuffle_ps",
                        args={"a": "top", "b": "bottom", "imm8": 0x88},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[leaf]
    )
    root1 = SolutionNode(
        stage=0,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_permute_ps",
                        args={"a": "top", "imm8": 0xD8},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[shared_child]
    )
    root2 = SolutionNode(
        stage=0,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_blend_ps",
                        args={"a": "top", "b": "bottom", "imm8": 0xF0},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[shared_child]
    )
    # Select top 2 paths
    paths = selector.select_top_k_paths([root1, root2], top_k=2)
    assert len(paths) == 2
    # Both paths should go through shared_child and reach leaf
    for path in paths:
        assert len(path.steps) == 3
        assert path.steps[1].node == shared_child
        assert path.steps[2].node == leaf
def test_admissible_heuristic():
    """Test that heuristic never overestimates remaining cost."""
    cost_model = CostModel("generic")
    selector = PathSelector(cost_model)
    # Create tree with multiple paths
    leaf_state = VectorState(top=[0, 1, 2, 3], bottom=[4, 5, 6, 7])
    leaf1 = SolutionNode(
        stage=2,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_permute_ps",  # latency 1.0
                        args={"a": "top", "imm8": 0xD8},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[]
    )
    leaf2 = SolutionNode(
        stage=2,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_permutexvar_epi32",  # latency 3.0
                        args={"a": "top", "op_idx": "idx"},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[]
    )
    child = SolutionNode(
        stage=1,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_shuffle_ps",  # latency 1.0
                        args={"a": "top", "b": "bottom", "imm8": 0x88},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[leaf1, leaf2]
    )
    root = SolutionNode(
        stage=0,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[child]
    )
    # Compute heuristic
    min_remaining = selector._compute_admissible_heuristic([root], max_stage=2)
    # At child, heuristic should be min(1.0, 3.5) = 1.0 (leaf1 latency)
    # (leaf2 has CV penalty: 3.0 + 0.5 = 3.5)
    assert min_remaining[id(child)] == 1.0
    # At root, heuristic should be 1.0 (child) + 1.0 (leaf1) = 2.0
    assert min_remaining[id(root)] == 2.0
def test_prune_to_top_k_paths():
    """Test tree pruning keeps only nodes on selected paths."""
    cost_model = CostModel("generic")
    selector = PathSelector(cost_model)
    # Create tree with multiple paths
    #       root
    #      /    \
    #   cheap   expensive
    #     |        |
    #   leaf1    leaf2
    leaf_state = VectorState(top=[0, 1, 2, 3], bottom=[4, 5, 6, 7])
    leaf1 = SolutionNode(
        stage=2,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[]
    )
    leaf2 = SolutionNode(
        stage=2,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[]
    )
    cheap = SolutionNode(
        stage=1,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_permute_ps",  # latency 1.0
                        args={"a": "top", "imm8": 0xD8},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[leaf1]
    )
    expensive = SolutionNode(
        stage=1,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[
                    InstructionSpec(
                        intrinsic_name="_mm256_permutexvar_epi32",  # latency 3.0
                        args={"a": "top", "op_idx": "idx"},
                    )
                ],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[leaf2]
    )
    root = SolutionNode(
        stage=0,
        input_state=leaf_state,
        output_state=leaf_state,
        gadgets=[
            PermutationGadget(
                top_instructions=[],
                bottom_instructions=[],
                validated=True
            )
        ],
        children=[cheap, expensive]
    )
    # Prune to top 1 path (should keep only cheap path)
    filtered_roots, selected_paths = selector.prune_to_top_k_paths([root], top_k=1)
    assert len(filtered_roots) == 1
    assert len(selected_paths) == 1
    # Root should only have cheap child now
    assert len(filtered_roots[0].children) == 1
    assert filtered_roots[0].children[0] == cheap
    # Expensive branch should be pruned
    assert expensive not in filtered_roots[0].children
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
