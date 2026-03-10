"""Tests for the src/intrinsics/ registry package."""

from __future__ import annotations

import pytest

from utils import vector_machine, primitive_type
from bitonic_types import InputRef, IntrinsicNode, Symbolic
from intrinsics import get_z3_functions, get_single_input_nodes, get_dual_input_nodes

# All four supported targets
_TARGETS = [
    (vector_machine.AVX2, primitive_type.i32),
    (vector_machine.AVX2, primitive_type.i64),
    (vector_machine.AVX512, primitive_type.i32),
    (vector_machine.AVX512, primitive_type.i64),
]

_TARGET_IDS = ["AVX2_i32", "AVX2_i64", "AVX512_i32", "AVX512_i64"]


@pytest.mark.parametrize("vm, prim", _TARGETS, ids=_TARGET_IDS)
def test_z3_functions_returns_nonempty_dict_of_callables(vm, prim):
    funcs = get_z3_functions(vm, prim)
    assert isinstance(funcs, dict)
    assert len(funcs) > 0
    for name, fn in funcs.items():
        assert isinstance(name, str)
        assert callable(fn), f"{name} is not callable"


@pytest.mark.parametrize("vm, prim", _TARGETS, ids=_TARGET_IDS)
def test_single_input_nodes_returns_intrinsic_nodes(vm, prim):
    ref = InputRef("top")
    nodes = get_single_input_nodes(vm, prim, ref)
    assert len(nodes) > 0
    funcs = get_z3_functions(vm, prim)
    for node in nodes:
        assert isinstance(node, IntrinsicNode)
        assert (
            node.name in funcs
        ), f"single-input node {node.name!r} not in z3_functions for {vm}/{prim}"


@pytest.mark.parametrize("vm, prim", _TARGETS, ids=_TARGET_IDS)
def test_dual_input_nodes_returns_intrinsic_nodes(vm, prim):
    ref1 = InputRef("top")
    ref2 = InputRef("bottom")
    nodes = get_dual_input_nodes(vm, prim, ref1, ref2)
    assert len(nodes) > 0
    funcs = get_z3_functions(vm, prim)
    for node in nodes:
        assert isinstance(node, IntrinsicNode)
        assert (
            node.name in funcs
        ), f"dual-input node {node.name!r} not in z3_functions for {vm}/{prim}"


def test_symbolic_names_include_tag():
    tag = "mytag"
    ref = InputRef(tag)
    # Check single-input nodes across one target (the tag logic is the same everywhere)
    nodes = get_single_input_nodes(vector_machine.AVX2, primitive_type.i32, ref)
    for node in nodes:
        for operand in node.operands.values():
            if isinstance(operand, Symbolic):
                assert (
                    tag in operand.name
                ), f"Symbolic {operand.name!r} does not contain tag {tag!r}"


def test_unsupported_target_raises():
    with pytest.raises(NotImplementedError):
        get_z3_functions(vector_machine.AVX2, primitive_type.i16)


def test_intrinsic_node_isomorphic_order_default():
    """IntrinsicNode defaults isomorphic_order=True and excludes it from equality."""
    from bitonic_types import IntrinsicNode, InputRef

    a = IntrinsicNode("foo", {"x": InputRef("top")})
    b = IntrinsicNode("foo", {"x": InputRef("top")}, isomorphic_order=False)

    assert a.isomorphic_order is True
    assert b.isomorphic_order is False
    assert a == b  # compare=False: tag does not affect equality


# Expected node counts per target
_COUNT_CASES = [
    (vector_machine.AVX2, primitive_type.i32, 4, 6),
    (vector_machine.AVX2, primitive_type.i64, 4, 6),
    (vector_machine.AVX512, primitive_type.i32, 6, 12),
    (vector_machine.AVX512, primitive_type.i64, 6, 12),
]

_COUNT_IDS = ["AVX2_i32", "AVX2_i64", "AVX512_i32", "AVX512_i64"]


@pytest.mark.parametrize(
    "vm, prim, expected_single, expected_dual",
    _COUNT_CASES,
    ids=_COUNT_IDS,
)
def test_node_counts_match_expected(vm, prim, expected_single, expected_dual):
    ref = InputRef("top")
    ref2 = InputRef("bottom")
    single = get_single_input_nodes(vm, prim, ref)
    dual = get_dual_input_nodes(vm, prim, ref, ref2)
    assert (
        len(single) == expected_single
    ), f"Expected {expected_single} single-input nodes for {vm}/{prim}, got {len(single)}"
    assert (
        len(dual) == expected_dual
    ), f"Expected {expected_dual} dual-input nodes for {vm}/{prim}, got {len(dual)}"
