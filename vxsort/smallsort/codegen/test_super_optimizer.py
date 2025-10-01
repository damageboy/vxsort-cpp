#!/usr/bin/env python3
"""
Tests for the super-optimizer.
"""

import pytest
# Import everything from super_optimizer to ensure we use the same instances
from super_optimizer import (
    BitonicSuperOptimizer,
    InstructionCatalog,
    PermutationSynthesizer,
    StageState,
    ElementPosition,
    BitonicSorter,
    vector_machine,
    primitive_type,
    width_dict,
)


class TestInstructionCatalog:
    """Test instruction catalog and cost model."""
    
    def test_avx2_32bit_instructions(self):
        """Test AVX2 32-bit instruction catalog."""
        instrs = InstructionCatalog.get_instructions(vector_machine.AVX2, 32)
        
        assert len(instrs) > 0
        assert any(i.name == '_mm256_permutexvar_epi32' for i in instrs)
        assert any(i.name == '_mm256_shuffle_ps' for i in instrs)
        
        # Check costs are reasonable
        for instr in instrs:
            assert instr.cost > 0
            assert instr.latency >= 0
            assert instr.throughput > 0
    
    def test_avx512_64bit_instructions(self):
        """Test AVX512 64-bit instruction catalog."""
        instrs = InstructionCatalog.get_instructions(vector_machine.AVX512, 64)
        
        assert len(instrs) > 0
        assert any(i.name == '_mm512_permutexvar_epi64' for i in instrs)
        assert any(i.name == '_mm512_permutex2var_epi64' for i in instrs)


class TestStageState:
    """Test state tracking."""
    
    def test_initial_state_avx2_i32(self):
        """Test creating initial state for AVX2/i32."""
        vm = vector_machine.AVX2
        prim_type = primitive_type.i32
        lanes_per_vector = (width_dict[vm] * 8) // (prim_type.value[0] * 8)
        
        state = StageState({}, num_vectors=2, lanes_per_vector=lanes_per_vector)
        
        # Populate with sequential elements
        for i in range(16):  # 2 vectors * 8 lanes
            vector = i // lanes_per_vector
            lane = i % lanes_per_vector
            state.positions[i] = ElementPosition(vector, lane)
        
        # Verify
        assert len(state.positions) == 16
        assert state.get_lane_contents(0, 0) == [0]
        assert state.get_lane_contents(1, 7) == [15]
    
    def test_state_copy(self):
        """Test deep copy of state."""
        state = StageState({0: ElementPosition(0, 0)}, num_vectors=2, lanes_per_vector=8)
        state2 = state.copy()
        
        state2.positions[0] = ElementPosition(1, 1)
        
        assert state.positions[0].vector == 0
        assert state2.positions[0].vector == 1


class TestPermutationSynthesizer:
    """Test permutation synthesis."""
    
    def test_synthesizer_creation(self):
        """Test creating a synthesizer."""
        synth = PermutationSynthesizer(vector_machine.AVX2, primitive_type.i32)
        
        assert synth.element_bits == 32
        assert synth.lanes_per_vector == 8
        assert len(synth.instructions) > 0
    
    def test_create_input_values(self):
        """Test creating input values for Z3."""
        synth = PermutationSynthesizer(vector_machine.AVX2, primitive_type.i32)
        
        # Create a simple state
        state = StageState({}, num_vectors=2, lanes_per_vector=8)
        for i in range(8):
            state.positions[i] = ElementPosition(0, i)  # All in vector 0
        
        # Target pairs
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        
        values = synth._create_input_values(state, pairs, vector_idx=0)
        
        assert len(values) == 8
        # Pairs should have matching values
        assert values[0] == values[1]  # Pair (0,1)
        assert values[2] == values[3]  # Pair (2,3)
        assert values[0] != values[2]  # Different pairs have different values


class TestBitonicSuperOptimizer:
    """Test the main super-optimizer."""
    
    def test_optimizer_creation(self):
        """Test creating optimizer."""
        # Generate a simple bitonic network
        total_elements = 16  # 2 AVX2 vectors of i32
        sorter = BitonicSorter(total_elements)
        
        optimizer = BitonicSuperOptimizer(
            sorter.stages,
            primitive_type.i32,
            vector_machine.AVX2,
            num_vectors=2
        )
        
        assert optimizer.total_elements == 16
        assert optimizer.lanes_per_vector == 8
    
    def test_initial_state_creation(self):
        """Test initial state is correctly created."""
        sorter = BitonicSorter(16)
        
        optimizer = BitonicSuperOptimizer(
            sorter.stages,
            primitive_type.i32,
            vector_machine.AVX2,
            num_vectors=2
        )
        
        initial = optimizer._create_initial_state()
        
        assert len(initial.positions) == 16
        # Elements should be in sequential positions
        assert initial.positions[0].vector == 0
        assert initial.positions[0].lane == 0
        assert initial.positions[8].vector == 1
        assert initial.positions[8].lane == 0
    
    @pytest.mark.skip(reason="Full optimization takes time, enable for integration testing")
    def test_optimize_small_network(self):
        """Test optimizing a small network."""
        # 8 elements = 1 AVX2 vector, but we'll use 2 for testing
        sorter = BitonicSorter(8)
        
        optimizer = BitonicSuperOptimizer(
            sorter.stages,
            primitive_type.i32,
            vector_machine.AVX2,
            num_vectors=2  # Artificially use 2 vectors
        )
        
        path = optimizer.optimize()
        
        assert path is not None
        assert len(path.stages) > 0
        print(f"Optimized path: {path}")


def test_instruction_costs_reasonable():
    """Test that all instruction costs are reasonable."""
    for vm in [vector_machine.AVX2, vector_machine.AVX512]:
        for bits in [32, 64]:
            instrs = InstructionCatalog.get_instructions(vm, bits)
            for instr in instrs:
                assert 0 < instr.cost < 100, f"{instr.name} has unreasonable cost {instr.cost}"
                assert instr.latency >= 1, f"{instr.name} latency too low"
                assert instr.throughput > 0, f"{instr.name} throughput invalid"


if __name__ == "__main__":
    # Run basic tests
    print("Testing Instruction Catalog...")
    test = TestInstructionCatalog()
    test.test_avx2_32bit_instructions()
    test.test_avx512_64bit_instructions()
    print("✓ Instruction Catalog tests passed")
    
    print("\nTesting Stage State...")
    test_state = TestStageState()
    test_state.test_initial_state_avx2_i32()
    test_state.test_state_copy()
    print("✓ Stage State tests passed")
    
    print("\nTesting Permutation Synthesizer...")
    test_synth = TestPermutationSynthesizer()
    test_synth.test_synthesizer_creation()
    test_synth.test_create_input_values()
    print("✓ Permutation Synthesizer tests passed")
    
    print("\nTesting Super Optimizer...")
    test_opt = TestBitonicSuperOptimizer()
    test_opt.test_optimizer_creation()
    test_opt.test_initial_state_creation()
    print("✓ Super Optimizer tests passed")
    
    print("\nTesting instruction costs...")
    test_instruction_costs_reasonable()
    print("✓ Instruction cost tests passed")
    
    print("\n" + "="*50)
    print("All basic tests passed!")
    print("="*50)

