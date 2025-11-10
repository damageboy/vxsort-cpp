#!/usr/bin/env python3
"""Demonstration of the BitonicSuperVectorizer system."""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from bitonic_compiler import BitonicSuperVectorizer, primitive_type, vector_machine, generate_bitonic_sorter


def demo_simple():
    """Demonstrate the super vectorizer on a simple 2-vector case."""
    print("=" * 70)
    print("BitonicSuperVectorizer Demo: 2 x AVX2 i32 vectors (16 elements)")
    print("=" * 70)
    print()

    # Create super vectorizer
    super_opt = BitonicSuperVectorizer(2, primitive_type.i32, vector_machine.AVX2)

    print(f"Total elements: {super_opt.total_elements}")
    print(f"Elements per vector: {super_opt.elements_per_vector}")
    print(f"Number of stages: {len(super_opt.bitonic_sorter.stages)}")
    print()

    # Show the stages
    print("Bitonic sorting stages:")
    for stage_id in sorted(super_opt.bitonic_sorter.stages.keys()):
        pairs = super_opt.bitonic_sorter.stages[stage_id]
        print(f"  Stage {stage_id}: {len(pairs)} pairs")
        print(f"    {pairs}")
    print()

    # Show initial state
    initial_state = super_opt._create_initial_state()
    print(f"Initial state:")
    print(f"  Top vector:    {initial_state.top}")
    print(f"  Bottom vector: {initial_state.bottom}")
    print()

    # Try to synthesize gadgets for the first stage only (demo)
    print("Attempting to synthesize gadgets for Stage 0...")
    stage_0_pairs = super_opt.bitonic_sorter.stages[0]
    print(f"  Target pairs: {stage_0_pairs}")

    # Check if input already matches target
    matches = super_opt.synthesizer._check_input_matches_target(initial_state, stage_0_pairs)
    if matches:
        print(f"  ✓ Input already matches target! No permutation needed.")
        print()
        print("This means the first stage requires ZERO instructions!")
        print("The elements are already aligned for the first min-max exchange.")
    else:
        print(f"  Input does not match target, gadget synthesis would be needed.")

    print()
    print("=" * 70)
    print("Note: Full synthesis with Z3 validation may take significant time")
    print("for all stages. This demo shows the structure is in place.")
    print("=" * 70)


def demo_instruction_catalogue():
    """Show the available instructions for synthesis."""
    print()
    print("=" * 70)
    print("Available AVX2 i32 Instructions for Synthesis")
    print("=" * 70)
    print()

    from bitonic_compiler import GadgetSynthesizer

    synthesizer = GadgetSynthesizer(vector_machine.AVX2, primitive_type.i32)

    print(f"Total intrinsics available: {len(synthesizer.available_intrinsics)}")
    print()

    for name in sorted(synthesizer.available_intrinsics.keys()):
        print(f"  • {name}")

    print()

    # Show some example instruction candidates
    print("Example single-input instruction candidates (first 5):")
    single_insts = synthesizer._enumerate_single_input_instructions("input")
    for inst in single_insts[:5]:
        print(f"  {inst}")

    print()
    print("Example dual-input instruction candidates (first 5):")
    dual_insts = synthesizer._enumerate_dual_input_instructions("top", "bottom")
    for inst in dual_insts[:5]:
        print(f"  {inst}")

    print()


if __name__ == "__main__":
    demo_simple()
    demo_instruction_catalogue()

    print()
    print("Demo complete!")
    print()
    print("To run full synthesis (may take time):")
    print("  python bitonic_compiler.py")
    print()

