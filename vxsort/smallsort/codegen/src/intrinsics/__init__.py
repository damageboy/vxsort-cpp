"""Intrinsics registry: dispatches (vm, prim_type) to per-target modules."""

from __future__ import annotations

try:
    from ..utils import vector_machine, primitive_type
    from ..bitonic_types import InputRef, IntrinsicNode
    from . import avx2_i32, avx2_i64, avx512_i32, avx512_i64
except ImportError:
    from util.enums import vector_machine, primitive_type  # type: ignore
    from bitonic_types import InputRef, IntrinsicNode  # type: ignore
    import intrinsics.avx2_i32 as avx2_i32  # type: ignore
    import intrinsics.avx2_i64 as avx2_i64  # type: ignore
    import intrinsics.avx512_i32 as avx512_i32  # type: ignore
    import intrinsics.avx512_i64 as avx512_i64  # type: ignore

_DISPATCH = {
    (vector_machine.AVX2, primitive_type.i32): avx2_i32,
    (vector_machine.AVX2, primitive_type.i64): avx2_i64,
    (vector_machine.AVX512, primitive_type.i32): avx512_i32,
    (vector_machine.AVX512, primitive_type.i64): avx512_i64,
}


def _lookup(vm: vector_machine, prim_type: primitive_type):
    key = (vm, prim_type)
    mod = _DISPATCH.get(key)
    if mod is None:
        raise NotImplementedError(
            f"Intrinsics not implemented for {vm} and {prim_type}"
        )
    return mod


def get_z3_functions(
    vm: vector_machine, prim_type: primitive_type
) -> dict[str, object]:
    """Get available Z3 intrinsic functions for the given VM and primitive type."""
    return _lookup(vm, prim_type).z3_functions()


def get_single_input_nodes(
    vm: vector_machine, prim_type: primitive_type, input_ref: InputRef
) -> list[IntrinsicNode]:
    """Get single-input IntrinsicNode templates for the given VM and primitive type."""
    return _lookup(vm, prim_type).single_input_nodes(input_ref)


def get_dual_input_nodes(
    vm: vector_machine, prim_type: primitive_type, ref1: InputRef, ref2: InputRef
) -> list[IntrinsicNode]:
    """Get dual-input IntrinsicNode templates for the given VM and primitive type."""
    return _lookup(vm, prim_type).dual_input_nodes(ref1, ref2)
