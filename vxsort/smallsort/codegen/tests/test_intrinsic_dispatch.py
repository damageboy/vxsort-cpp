#!/usr/bin/env python3
"""Unit tests for intrinsic dispatch rule matching and call ordering."""

from bitonic_super_optimizer import (
    _dispatch_intrinsic_by_signature,
    _match_dispatch_rule,
)


def test_match_rule_prefers_specific_masked_immediate_pattern():
    keys = frozenset(("k", "src", "a", "b", "imm8"))
    rule = _match_dispatch_rule(keys)
    assert rule is not None
    assert rule.name == "masked_dual_with_immediate"


def test_match_rule_requires_exact_key_set():
    # This has a valid base shape plus one extra key.
    # With exact matching it must not match any rule.
    keys = frozenset(("k", "src", "a", "b", "imm8", "extra"))
    rule = _match_dispatch_rule(keys)
    assert rule is None


def test_dispatch_uses_rule_call_order_not_dict_insertion_order():
    captured = []

    def intrinsic(*values):
        captured.append(values)
        return values

    # Deliberately out-of-order insertion for rule:
    # masked_single_with_immediate -> call_order ("src", "k", "a", "imm8")
    args = {"a": 10, "imm8": 7, "k": 3, "src": 99}
    _dispatch_intrinsic_by_signature(intrinsic, args)
    assert captured[0] == (99, 3, 10, 7)


def test_dispatch_fallback_preserves_argument_insertion_order():
    captured = []

    def intrinsic(*values):
        captured.append(values)
        return values

    args = {"z": 1, "x": 2, "q": 3}
    _dispatch_intrinsic_by_signature(intrinsic, args)
    assert captured[0] == (1, 2, 3)
