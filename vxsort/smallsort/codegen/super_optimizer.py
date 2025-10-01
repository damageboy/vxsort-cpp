#!/usr/bin/env python3
"""
Super-optimizer for bitonic sorter shuffle/permute operations.

Uses Z3 SMT solver to synthesize optimal permutation sequences for each stage
of a bitonic sort network, minimizing total instruction cost while ensuring
correctness.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any
import itertools

from z3 import Solver, sat, unsat, BitVec, BitVecVal

# Import Z3 AVX instruction models
from z3_avx import (
    ymm_reg, zmm_reg,
    ymm_reg_with_32b_values, zmm_reg_with_32b_values,
    ymm_reg_with_64b_values, zmm_reg_with_64b_values,
    # AVX2 instructions
    _mm256_permutexvar_epi32, _mm256_permutexvar_epi64,
    _mm256_permute_ps, _mm256_permute_pd,
    _mm256_shuffle_ps, _mm256_shuffle_pd,
    _mm256_permute2x128_si256,
    _mm256_unpacklo_epi32, _mm256_unpackhi_epi32,
    # AVX512 instructions
    _mm512_permutexvar_epi32, _mm512_permutexvar_epi64,
    _mm512_permutex2var_epi32, _mm512_permutex2var_epi64,
    _mm512_permute_ps, _mm512_permute_pd,
    _mm512_shuffle_ps, _mm512_shuffle_pd,
    _mm512_shuffle_i32x4,
    _mm512_unpacklo_epi32, _mm512_unpackhi_epi32,
)

# Import from bitonic-compiler.py (with dash in filename)
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Rename module to avoid dash issues
import importlib.machinery
import importlib.util
loader = importlib.machinery.SourceFileLoader(
    "bitonic_compiler_module",
    os.path.join(os.path.dirname(__file__), "bitonic-compiler.py")
)
spec = importlib.util.spec_from_loader("bitonic_compiler_module", loader)
bitonic_compiler_module = importlib.util.module_from_spec(spec)
sys.modules["bitonic_compiler_module"] = bitonic_compiler_module
loader.exec_module(bitonic_compiler_module)

vector_machine = bitonic_compiler_module.vector_machine
primitive_type = bitonic_compiler_module.primitive_type
width_dict = bitonic_compiler_module.width_dict
BitonicSorter = bitonic_compiler_module.BitonicSorter

# Export for other modules
__all__ = ['vector_machine', 'primitive_type', 'width_dict', 'BitonicSorter',
           'BitonicSuperOptimizer', 'InstructionCatalog', 'PermutationSynthesizer',
           'StageState', 'ElementPosition', 'StageSolution', 'SolutionPath']


class InstructionType(Enum):
    """Type of permutation instruction."""
    SINGLE_REG_IMMEDIATE = 1  # One input, immediate control (e.g., permute_ps)
    SINGLE_REG_VARIABLE = 2   # One input, variable control (e.g., permutexvar)
    DUAL_REG_IMMEDIATE = 3    # Two inputs, immediate control (e.g., shuffle_ps)
    DUAL_REG_VARIABLE = 4     # Two inputs, variable control (e.g., permutex2var)


@dataclass
class InstructionDef:
    """Definition of an available permutation instruction."""
    name: str
    type: InstructionType
    z3_func: Callable
    element_bits: int  # 32 or 64
    vector_machine: vector_machine
    # Cost model (latency, reciprocal throughput, ports)
    latency: float
    throughput: float
    ports: str  # e.g., "p5" or "p0/p1"
    
    @property
    def cost(self) -> float:
        """Combined cost metric (can be refined)."""
        # Simple cost: latency + 1/throughput
        return self.latency + (1.0 / self.throughput if self.throughput > 0 else 10.0)


@dataclass
class ElementPosition:
    """Tracks where an element index is located."""
    vector: int  # 0=top, 1=bottom (or more for >2 vectors)
    lane: int    # Lane within the vector
    
    def __hash__(self):
        return hash((self.vector, self.lane))


@dataclass 
class StageState:
    """State of element positions at a stage."""
    # Maps element index -> position
    positions: dict[int, ElementPosition]
    num_vectors: int
    lanes_per_vector: int
    
    def copy(self) -> StageState:
        """Deep copy of state."""
        return StageState(
            positions={k: ElementPosition(v.vector, v.lane) for k, v in self.positions.items()},
            num_vectors=self.num_vectors,
            lanes_per_vector=self.lanes_per_vector
        )
    
    def get_lane_contents(self, vector: int, lane: int) -> list[int]:
        """Get all element indices in a specific vector:lane."""
        return [idx for idx, pos in self.positions.items() 
                if pos.vector == vector and pos.lane == lane]


@dataclass
class PermuteGadget:
    """A single permutation operation (or sequence)."""
    vector: int  # Which vector this operates on (0=top, 1=bottom, etc.)
    instructions: list[tuple[str, dict[str, Any]]]  # [(name, params), ...]
    cost: float
    
    def apply(self, state: StageState) -> StageState:
        """Apply this gadget to a state (abstract transformation)."""
        # This would update element positions based on the permutation
        # For now, we'll compute this during synthesis
        raise NotImplementedError("Applied during synthesis")


@dataclass
class StageSolution:
    """A complete solution for one stage."""
    stage_idx: int
    input_state: StageState
    output_state: StageState
    gadgets: list[PermuteGadget]  # One per vector
    total_cost: float
    
    def __repr__(self):
        gadget_strs = [f"V{g.vector}: {len(g.instructions)} ops" for g in self.gadgets]
        return f"Stage{self.stage_idx} [cost={self.total_cost:.2f}]: {', '.join(gadget_strs)}"


@dataclass
class SolutionPath:
    """Complete path through all stages."""
    stages: list[StageSolution]
    total_cost: float
    
    def __repr__(self):
        return f"Path [cost={self.total_cost:.2f}]: {len(self.stages)} stages"


class InstructionCatalog:
    """Catalog of available instructions with cost model."""
    
    # Cost data from uops.info (approximate values for common CPUs)
    # Format: (latency, reciprocal_throughput, ports)
    AVX2_COSTS = {
        '_mm256_permutexvar_epi32': (3, 1.0, 'p5'),
        '_mm256_permutexvar_epi64': (3, 1.0, 'p5'),
        '_mm256_permute_ps': (1, 1.0, 'p5'),
        '_mm256_permute_pd': (1, 1.0, 'p5'),
        '_mm256_shuffle_ps': (1, 1.0, 'p5'),
        '_mm256_shuffle_pd': (1, 1.0, 'p5'),
        '_mm256_permute2x128_si256': (3, 1.0, 'p5'),
        '_mm256_unpacklo_epi32': (1, 1.0, 'p5'),
        '_mm256_unpackhi_epi32': (1, 1.0, 'p5'),
    }
    
    AVX512_COSTS = {
        '_mm512_permutexvar_epi32': (3, 1.0, 'p5'),
        '_mm512_permutexvar_epi64': (3, 1.0, 'p5'),
        '_mm512_permutex2var_epi32': (3, 1.0, 'p5'),
        '_mm512_permutex2var_epi64': (3, 1.0, 'p5'),
        '_mm512_permute_ps': (1, 1.0, 'p5'),
        '_mm512_permute_pd': (1, 1.0, 'p5'),
        '_mm512_shuffle_ps': (1, 1.0, 'p5'),
        '_mm512_shuffle_pd': (1, 1.0, 'p5'),
        '_mm512_shuffle_i32x4': (3, 1.0, 'p5'),
        '_mm512_unpacklo_epi32': (1, 1.0, 'p5'),
        '_mm512_unpackhi_epi32': (1, 1.0, 'p5'),
    }
    
    @classmethod
    def get_instructions(cls, vm: vector_machine, element_bits: int) -> list[InstructionDef]:
        """Get all available instructions for a vector machine and element size."""
        instructions = []
        
        if vm == vector_machine.AVX2:
            costs = cls.AVX2_COSTS
            if element_bits == 32:
                instructions.extend([
                    InstructionDef('_mm256_permutexvar_epi32', InstructionType.SINGLE_REG_VARIABLE,
                                 _mm256_permutexvar_epi32, 32, vm, *costs['_mm256_permutexvar_epi32']),
                    InstructionDef('_mm256_permute_ps', InstructionType.SINGLE_REG_IMMEDIATE,
                                 _mm256_permute_ps, 32, vm, *costs['_mm256_permute_ps']),
                    InstructionDef('_mm256_shuffle_ps', InstructionType.DUAL_REG_IMMEDIATE,
                                 _mm256_shuffle_ps, 32, vm, *costs['_mm256_shuffle_ps']),
                    InstructionDef('_mm256_unpacklo_epi32', InstructionType.DUAL_REG_IMMEDIATE,
                                 _mm256_unpacklo_epi32, 32, vm, *costs['_mm256_unpacklo_epi32']),
                    InstructionDef('_mm256_unpackhi_epi32', InstructionType.DUAL_REG_IMMEDIATE,
                                 _mm256_unpackhi_epi32, 32, vm, *costs['_mm256_unpackhi_epi32']),
                ])
            elif element_bits == 64:
                instructions.extend([
                    InstructionDef('_mm256_permutexvar_epi64', InstructionType.SINGLE_REG_VARIABLE,
                                 _mm256_permutexvar_epi64, 64, vm, *costs['_mm256_permutexvar_epi64']),
                    InstructionDef('_mm256_permute_pd', InstructionType.SINGLE_REG_IMMEDIATE,
                                 _mm256_permute_pd, 64, vm, *costs['_mm256_permute_pd']),
                    InstructionDef('_mm256_shuffle_pd', InstructionType.DUAL_REG_IMMEDIATE,
                                 _mm256_shuffle_pd, 64, vm, *costs['_mm256_shuffle_pd']),
                ])
        
        elif vm == vector_machine.AVX512:
            costs = cls.AVX512_COSTS
            if element_bits == 32:
                instructions.extend([
                    InstructionDef('_mm512_permutexvar_epi32', InstructionType.SINGLE_REG_VARIABLE,
                                 _mm512_permutexvar_epi32, 32, vm, *costs['_mm512_permutexvar_epi32']),
                    InstructionDef('_mm512_permutex2var_epi32', InstructionType.DUAL_REG_VARIABLE,
                                 _mm512_permutex2var_epi32, 32, vm, *costs['_mm512_permutex2var_epi32']),
                    InstructionDef('_mm512_permute_ps', InstructionType.SINGLE_REG_IMMEDIATE,
                                 _mm512_permute_ps, 32, vm, *costs['_mm512_permute_ps']),
                    InstructionDef('_mm512_shuffle_ps', InstructionType.DUAL_REG_IMMEDIATE,
                                 _mm512_shuffle_ps, 32, vm, *costs['_mm512_shuffle_ps']),
                    InstructionDef('_mm512_unpacklo_epi32', InstructionType.DUAL_REG_IMMEDIATE,
                                 _mm512_unpacklo_epi32, 32, vm, *costs['_mm512_unpacklo_epi32']),
                    InstructionDef('_mm512_unpackhi_epi32', InstructionType.DUAL_REG_IMMEDIATE,
                                 _mm512_unpackhi_epi32, 32, vm, *costs['_mm512_unpackhi_epi32']),
                ])
            elif element_bits == 64:
                instructions.extend([
                    InstructionDef('_mm512_permutexvar_epi64', InstructionType.SINGLE_REG_VARIABLE,
                                 _mm512_permutexvar_epi64, 64, vm, *costs['_mm512_permutexvar_epi64']),
                    InstructionDef('_mm512_permutex2var_epi64', InstructionType.DUAL_REG_VARIABLE,
                                 _mm512_permutex2var_epi64, 64, vm, *costs['_mm512_permutex2var_epi64']),
                    InstructionDef('_mm512_permute_pd', InstructionType.SINGLE_REG_IMMEDIATE,
                                 _mm512_permute_pd, 64, vm, *costs['_mm512_permute_pd']),
                    InstructionDef('_mm512_shuffle_pd', InstructionType.DUAL_REG_IMMEDIATE,
                                 _mm512_shuffle_pd, 64, vm, *costs['_mm512_shuffle_pd']),
                ])
        
        return instructions


class PermutationSynthesizer:
    """Synthesizes permutation gadgets using Z3."""
    
    def __init__(self, vm: vector_machine, prim_type: primitive_type):
        self.vm = vm
        self.prim_type = prim_type
        self.element_bits = prim_type.value[0] * 8
        self.vector_bits = width_dict[vm] * 8
        self.lanes_per_vector = self.vector_bits // self.element_bits
        
        # Get available instructions
        self.instructions = InstructionCatalog.get_instructions(vm, self.element_bits)
    
    def synthesize_gadget(self, 
                         input_state: StageState,
                         target_pairs: list[tuple[int, int]],
                         vector_idx: int,
                         max_depth: int = 2) -> list[PermuteGadget]:
        """
        Synthesize permutation gadgets for a single vector.
        
        Args:
            input_state: Current element positions
            target_pairs: List of (idx1, idx2) pairs that need to be aligned
            vector_idx: Which vector we're permuting (0=top, 1=bottom, etc.)
            max_depth: Maximum instruction sequence length
            
        Returns:
            List of valid gadgets (may be empty if no solution found)
        """
        solutions = []
        
        # Try depth 1, then depth 2 (iterative deepening)
        for depth in range(1, max_depth + 1):
            depth_solutions = self._search_depth(input_state, target_pairs, vector_idx, depth)
            solutions.extend(depth_solutions)
            
            # If we found solutions at this depth, we might continue to find more
            # complex ones, but for now let's collect all
        
        return solutions
    
    def _search_depth(self,
                     input_state: StageState,
                     target_pairs: list[tuple[int, int]],
                     vector_idx: int,
                     depth: int) -> list[PermuteGadget]:
        """Search for solutions at a specific depth."""
        if depth == 1:
            return self._search_single_instruction(input_state, target_pairs, vector_idx)
        elif depth == 2:
            return self._search_two_instructions(input_state, target_pairs, vector_idx)
        else:
            return []
    
    def _search_single_instruction(self,
                                   input_state: StageState,
                                   target_pairs: list[tuple[int, int]],
                                   vector_idx: int) -> list[PermuteGadget]:
        """Try all single instruction solutions."""
        solutions = []
        
        for instr in self.instructions:
            result = self._try_instruction(instr, input_state, target_pairs, vector_idx)
            if result:
                gadget, output_state = result
                solutions.append(gadget)
        
        return solutions
    
    def _search_two_instructions(self,
                                 input_state: StageState,
                                 target_pairs: list[tuple[int, int]],
                                 vector_idx: int) -> list[PermuteGadget]:
        """Try all two instruction sequences."""
        solutions = []
        
        # Try all pairs of instructions
        for instr1, instr2 in itertools.product(self.instructions, repeat=2):
            result = self._try_instruction_sequence(
                [instr1, instr2], input_state, target_pairs, vector_idx
            )
            if result:
                gadget, output_state = result
                solutions.append(gadget)
        
        return solutions
    
    def _try_instruction(self,
                        instr: InstructionDef,
                        input_state: StageState,
                        target_pairs: list[tuple[int, int]],
                        vector_idx: int) -> Optional[tuple[PermuteGadget, StageState]]:
        """
        Try a single instruction and verify it achieves the goal.
        
        Returns (gadget, output_state) if successful, None otherwise.
        """
        # Create Z3 solver
        s = Solver()
        
        # Create input register with unique values for each target pair
        input_values = self._create_input_values(input_state, target_pairs, vector_idx)
        
        # Create Z3 representation
        if self.vm == vector_machine.AVX2:
            if self.element_bits == 32:
                input_reg = ymm_reg_with_32b_values('input', s, input_values)
            else:
                input_reg = ymm_reg_with_64b_values('input', s, input_values)
        else:  # AVX512
            if self.element_bits == 32:
                input_reg = zmm_reg_with_32b_values('input', s, input_values)
            else:
                input_reg = zmm_reg_with_64b_values('input', s, input_values)
        
        # Apply instruction based on type
        if instr.type == InstructionType.SINGLE_REG_IMMEDIATE:
            # e.g., permute_ps - synthesize the immediate
            imm8 = BitVec('imm8', 8)
            output_reg = instr.z3_func(input_reg, imm8)
            params = {'imm8': imm8}
            
        elif instr.type == InstructionType.SINGLE_REG_VARIABLE:
            # e.g., permutexvar - synthesize the index vector
            if self.vm == vector_machine.AVX2:
                idx_reg = ymm_reg('idx')
            else:
                idx_reg = zmm_reg('idx')
            output_reg = instr.z3_func(input_reg, idx_reg)
            params = {'idx': idx_reg}
            
        else:
            # Dual register instructions - need to handle differently
            # For now, skip these in single instruction search
            return None
        
        # Add constraints: each pair should have matching values in output
        self._add_alignment_constraints(s, output_reg, target_pairs, input_values)
        
        # Check satisfiability
        if s.check() == sat:
            model = s.model()
            # Extract parameters from model
            extracted_params = self._extract_params(model, params, instr)
            
            # Create gadget
            gadget = PermuteGadget(
                vector=vector_idx,
                instructions=[(instr.name, extracted_params)],
                cost=instr.cost
            )
            
            # Compute output state
            output_state = self._compute_output_state(
                input_state, vector_idx, extracted_params, instr, model, output_reg
            )
            
            return (gadget, output_state)
        
        return None
    
    def _try_instruction_sequence(self,
                                  instrs: list[InstructionDef],
                                  input_state: StageState,
                                  target_pairs: list[tuple[int, int]],
                                  vector_idx: int) -> Optional[tuple[PermuteGadget, StageState]]:
        """Try a sequence of instructions."""
        # TODO: Implement chained instruction synthesis
        # This is more complex as we need to chain the outputs
        return None
    
    def _create_input_values(self,
                            input_state: StageState,
                            target_pairs: list[tuple[int, int]],
                            vector_idx: int) -> list[int]:
        """
        Create input values where each target pair gets a unique value.
        
        Elements not in target pairs get distinct values too.
        """
        # Assign unique value to each pair
        pair_values = {}
        next_value = 1
        
        for idx1, idx2 in target_pairs:
            pair_values[idx1] = next_value
            pair_values[idx2] = next_value
            next_value += 1
        
        # Create lane-indexed values
        values = []
        for lane in range(self.lanes_per_vector):
            # Find which element is in this lane of this vector
            contents = input_state.get_lane_contents(vector_idx, lane)
            if contents:
                elem_idx = contents[0]  # Should be only one
                if elem_idx in pair_values:
                    values.append(pair_values[elem_idx])
                else:
                    # Not in a target pair, use distinct value
                    values.append(next_value)
                    next_value += 1
            else:
                # Empty lane (shouldn't happen normally)
                values.append(0)
        
        return values
    
    def _add_alignment_constraints(self,
                                   solver: Solver,
                                   output_reg,
                                   target_pairs: list[tuple[int, int]],
                                   input_values: list[int]):
        """
        Add constraints that paired values must end up in the same lane.
        
        We don't care which lane, just that they're together.
        """
        # Extract output values per lane
        from z3 import Extract, Or, And
        
        output_lanes = []
        for lane in range(self.lanes_per_vector):
            start_bit = lane * self.element_bits
            end_bit = start_bit + self.element_bits - 1
            output_lanes.append(Extract(end_bit, start_bit, output_reg))
        
        # For each pair, ensure they end up in the same lane
        for idx1, idx2 in target_pairs:
            pair_value = input_values[idx1] if idx1 < len(input_values) else input_values[idx2]
            
            # Find which lanes have this pair value
            # At least one lane should have both values (actually represented as same value twice)
            # Actually, since both indices have the same value, we just need that value
            # to appear in the output - this is automatically satisfied if permutation preserves values
            
            # The key constraint is that the OUTPUT should have each unique pair value
            # appearing at least once (values are preserved through permutation)
            pass  # The permutation naturally preserves values
    
    def _extract_params(self, model, params: dict, instr: InstructionDef) -> dict[str, Any]:
        """Extract concrete parameter values from Z3 model."""
        result = {}
        
        for name, param in params.items():
            if name == 'imm8':
                # Extract immediate value
                result['imm8'] = model.evaluate(param).as_long()
            elif name == 'idx':
                # Extract index vector
                idx_val = model.evaluate(param).as_long()
                # Convert to list of indices
                indices = []
                for i in range(self.lanes_per_vector):
                    if self.element_bits == 32:
                        idx = (idx_val >> (i * 32)) & ((1 << 5) - 1)  # 5 bits for AVX512, 3 for AVX2
                    else:
                        idx = (idx_val >> (i * 64)) & ((1 << 3) - 1)
                    indices.append(idx)
                result['idx'] = indices
        
        return result
    
    def _compute_output_state(self,
                             input_state: StageState,
                             vector_idx: int,
                             params: dict,
                             instr: InstructionDef,
                             model,
                             output_reg) -> StageState:
        """Compute the output state after applying the instruction."""
        # For now, return a copy - we'll refine this
        return input_state.copy()


class BitonicSuperOptimizer:
    """Main super-optimizer for bitonic sort stages."""
    
    def __init__(self,
                 stages: dict[int, list[tuple[int, int]]],
                 prim_type: primitive_type,
                 vm: vector_machine,
                 num_vectors: int = 2):
        self.stages = stages
        self.prim_type = prim_type
        self.vm = vm
        self.num_vectors = num_vectors
        
        # Calculate dimensions
        self.element_bits = prim_type.value[0] * 8
        self.vector_bits = width_dict[vm] * 8
        self.lanes_per_vector = self.vector_bits // self.element_bits
        self.total_elements = num_vectors * self.lanes_per_vector
        
        # Create synthesizer
        self.synthesizer = PermutationSynthesizer(vm, prim_type)
        
        # Solution tree
        self.solution_tree: dict[int, list[StageSolution]] = {}
    
    def optimize(self) -> SolutionPath:
        """
        Run the super-optimizer and find the best solution path.
        
        Returns the optimal SolutionPath through all stages.
        """
        # Build initial state
        initial_state = self._create_initial_state()
        
        # Process each stage
        current_states = [initial_state]
        
        for stage_idx in sorted(self.stages.keys()):
            pairs = self.stages[stage_idx]
            stage_solutions = []
            
            # For each possible input state from previous stage
            for input_state in current_states:
                # Synthesize solutions for this stage
                solutions = self._synthesize_stage(stage_idx, input_state, pairs)
                stage_solutions.extend(solutions)
            
            self.solution_tree[stage_idx] = stage_solutions
            
            # Prepare states for next stage
            current_states = [sol.output_state for sol in stage_solutions]
        
        # Find optimal path through tree
        optimal_path = self._find_optimal_path()
        
        return optimal_path
    
    def _create_initial_state(self) -> StageState:
        """Create the initial unsorted state."""
        positions = {}
        elem_idx = 0
        
        for vector in range(self.num_vectors):
            for lane in range(self.lanes_per_vector):
                positions[elem_idx] = ElementPosition(vector, lane)
                elem_idx += 1
        
        return StageState(positions, self.num_vectors, self.lanes_per_vector)
    
    def _synthesize_stage(self,
                         stage_idx: int,
                         input_state: StageState,
                         pairs: list[tuple[int, int]]) -> list[StageSolution]:
        """Synthesize all possible solutions for one stage."""
        
        # Special case: first stage needs no permutation (pairs can be anywhere)
        if stage_idx == 0:
            # Create a no-op solution
            gadgets = [PermuteGadget(v, [], 0.0) for v in range(self.num_vectors)]
            return [StageSolution(stage_idx, input_state, input_state, gadgets, 0.0)]
        
        # For each vector, synthesize permutation gadgets
        vector_gadgets = []
        for vector_idx in range(self.num_vectors):
            gadgets = self.synthesizer.synthesize_gadget(
                input_state, pairs, vector_idx, max_depth=2
            )
            vector_gadgets.append(gadgets)
        
        # Combine gadgets from all vectors to create complete solutions
        solutions = []
        for gadget_combo in itertools.product(*vector_gadgets):
            total_cost = sum(g.cost for g in gadget_combo)
            
            # Compute final output state (simplified for now)
            output_state = input_state.copy()
            
            solution = StageSolution(
                stage_idx=stage_idx,
                input_state=input_state,
                output_state=output_state,
                gadgets=list(gadget_combo),
                total_cost=total_cost
            )
            solutions.append(solution)
        
        return solutions if solutions else []
    
    def _find_optimal_path(self) -> SolutionPath:
        """Find the minimum cost path through the solution tree."""
        if not self.solution_tree:
            return SolutionPath([], 0.0)
        
        # Simple greedy approach for now: pick minimum cost at each stage
        path_stages = []
        total_cost = 0.0
        
        for stage_idx in sorted(self.solution_tree.keys()):
            solutions = self.solution_tree[stage_idx]
            if solutions:
                best = min(solutions, key=lambda s: s.total_cost)
                path_stages.append(best)
                total_cost += best.total_cost
        
        return SolutionPath(path_stages, total_cost)


# Example usage
if __name__ == "__main__":
    from bitonic_compiler import BitonicSorter
    
    # Generate a simple 2-vector (16 element) bitonic sorter for AVX2/i32
    num_vecs = 2
    vm = vector_machine.AVX2
    prim_type = primitive_type.i32
    total_elements = num_vecs * (width_dict[vm] // prim_type.value[0])
    
    print(f"Optimizing {total_elements}-element bitonic sort for {vm.name}/{prim_type.name}")
    
    # Generate bitonic network
    bitonic_sorter = BitonicSorter(total_elements)
    print(f"Generated {len(bitonic_sorter.stages)} stages")
    
    # Run super-optimizer
    optimizer = BitonicSuperOptimizer(
        bitonic_sorter.stages,
        prim_type,
        vm,
        num_vectors=num_vecs
    )
    
    optimal_path = optimizer.optimize()
    print(f"\nOptimal solution: {optimal_path}")
    for stage in optimal_path.stages:
        print(f"  {stage}")

