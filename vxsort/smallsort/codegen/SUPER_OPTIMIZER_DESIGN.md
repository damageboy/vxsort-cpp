# Bitonic Sort Super-Optimizer Design

## Overview

The super-optimizer synthesizes optimal permutation sequences for bitonic sort networks using Z3 SMT solving. It finds the most efficient combination of SIMD shuffle/permute instructions to align comparison pairs at each stage, minimizing total instruction cost.

## Architecture

### Key Components

```
BitonicSuperOptimizer
  ├── PermutationSynthesizer (Z3-based gadget synthesis)
  │     ├── InstructionCatalog (available instructions + costs)
  │     └── Z3 constraint generation
  ├── StageState (element position tracking)
  └── Solution tree (multiple paths through stages)
```

### Data Flow

1. **Input**: Bitonic network stages from `BitonicSorter`
   - Each stage: list of `(idx1, idx2)` comparison pairs

2. **Initial State**: Sequential element placement
   - Elements 0-7 in top vector (lanes 0-7)
   - Elements 8-15 in bottom vector (lanes 0-7)
   - For AVX2/i32: 8 lanes per vector

3. **Per-Stage Processing**:
   ```
   For each stage:
     For each vector (top/bottom):
       Try instruction sequences (depth 1-2):
         - Create Z3 input with unique values per pair
         - Apply instruction(s) symbolically
         - Add constraints: pairs must align (same lane)
         - If SAT: extract parameters from model
         - Record gadget + cost
   ```

4. **Path Selection**: Find minimum-cost path through solution tree

## Key Classes

### `StageState`
Tracks where each element index is located:
```python
positions: dict[int, ElementPosition]  # element_idx -> (vector, lane)
```

**Example** (AVX2/i32, 2 vectors):
```
Initial state:
  positions = {
    0: (vector=0, lane=0),
    1: (vector=0, lane=1),
    ...
    8: (vector=1, lane=0),
    ...
  }
```

### `PermuteGadget`
A permutation solution for one vector:
```python
vector: int  # Which vector (0=top, 1=bottom)
instructions: list[tuple[str, dict]]  # [(name, params), ...]
cost: float
```

**Example**:
```python
PermuteGadget(
    vector=0,
    instructions=[
        ('_mm256_permutexvar_epi32', {'idx': [7, 6, 5, 4, 3, 2, 1, 0]})
    ],
    cost=3.0
)
```

### `InstructionCatalog`
Maps instructions to cost model (from uops.info):
- **Latency**: Cycles from input ready to output ready
- **Throughput**: 1/Reciprocal throughput (ops/cycle)
- **Cost**: `latency + 1/throughput` (simple model for now)

## Z3 Synthesis Process

### 1. Create Input Values
For pairs `[(0,1), (2,3), (4,5), (6,7)]`:
```python
# Assign unique value to each pair
pair_values = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4}

# Map to lanes based on current state
input_values = [1, 1, 2, 2, 3, 3, 4, 4]  # If elements are in order
```

### 2. Symbolic Execution
```python
s = Solver()
input_reg = ymm_reg_with_32b_values('input', s, input_values)

# For variable permute: synthesize index vector
idx_reg = ymm_reg('idx')
output_reg = _mm256_permutexvar_epi32(input_reg, idx_reg)

# For immediate permute: synthesize immediate
imm8 = BitVec('imm8', 8)
output_reg = _mm256_permute_ps(input_reg, imm8)
```

### 3. Add Constraints
```python
# Pairs must align: same unique value must stay together
# (This is implicitly satisfied if permutation preserves values)
# Additional constraints can verify output structure
```

### 4. Extract Solution
If `s.check() == sat`:
```python
model = s.model()
# For variable permute:
idx_val = model.evaluate(idx_reg).as_long()
indices = extract_indices(idx_val)  # Convert bitvec to list

# For immediate permute:
imm_val = model.evaluate(imm8).as_long()
```

## Current Implementation Status

### ✅ Completed
- [x] Basic architecture and data structures
- [x] Instruction catalog with cost model
- [x] State tracking (`StageState`, `ElementPosition`)
- [x] Single instruction synthesis framework
- [x] Import/module structure
- [x] Basic tests

### 🚧 In Progress / TODO

#### High Priority

1. **Z3 Constraint Generation** (TODO #2)
   - Current: Placeholder in `_add_alignment_constraints`
   - Needed: Proper constraints ensuring pairs align
   - Challenge: Constraint must allow any lane, just enforce same-lane

2. **State Computation** (TODO #3)
   - Current: Returns copy of input
   - Needed: Compute actual output positions after permutation
   - Method: Simulate permutation on position map

3. **Dual-Register Instructions** (TODO #6, #7)
   - Current: Only single-register instructions supported
   - Needed: Handle `shuffle_ps`, `unpacklo`, `permutex2var`
   - Challenge: Two input vectors, need to coordinate

#### Medium Priority

4. **Two-Instruction Chaining** (TODO #4)
   - Current: Stub returns None
   - Needed: Chain two instructions, passing output->input
   - Example: `permute_ps` followed by `permute2x128`

5. **Better Path Finding** (TODO #5)
   - Current: Greedy (pick min cost per stage)
   - Needed: Dynamic programming for global optimum
   - Algorithm: Dijkstra's or A* through solution graph

#### Low Priority

6. **Model Validation** (TODO #8)
   - Verify synthesized gadgets are correct
   - Run test inputs through Z3 model

7. **Code Generation** (TODO #9)
   - Output C++ intrinsics
   - Output assembly
   - Generate test harness

8. **Comprehensive Tests** (TODO #10)
   - Full optimization runs
   - Correctness verification
   - Performance benchmarks

## Example Usage

```python
from super_optimizer import BitonicSuperOptimizer, BitonicSorter
from super_optimizer import vector_machine, primitive_type

# Create bitonic network (16 elements = 2 AVX2 vectors)
sorter = BitonicSorter(16)

# Run super-optimizer
optimizer = BitonicSuperOptimizer(
    stages=sorter.stages,
    prim_type=primitive_type.i32,
    vm=vector_machine.AVX2,
    num_vectors=2
)

optimal_path = optimizer.optimize()

# Examine solution
for stage in optimal_path.stages:
    print(f"Stage {stage.stage_idx}:")
    for gadget in stage.gadgets:
        print(f"  Vector {gadget.vector}:")
        for instr_name, params in gadget.instructions:
            print(f"    {instr_name}({params})")
```

## Design Decisions

### Why Unique Values Per Pair?
- Allows Z3 to track which elements belong together
- Doesn't constrain which lane they end up in
- Simplifies constraint generation

### Why Iterative Deepening?
- Most stages solvable with 1 instruction
- Trying depth 1 first is fast
- Only pay for depth 2 when needed

### Why Separate Gadgets Per Vector?
- Top and bottom vectors permute independently
- Later: min/max operation between vectors
- Allows parallel instruction selection

### Why Cost Model Instead of Just Instruction Count?
- Real performance depends on latency+throughput
- Some instructions slower than others
- Port pressure matters for scheduling

## Future Enhancements

1. **Register Pressure Tracking**
   - Account for number of temp registers needed
   - Prefer solutions using fewer registers

2. **Port-Aware Scheduling**
   - Model actual CPU port allocation
   - Avoid port conflicts

3. **Cross-Vector Optimizations**
   - Consider swapping elements between top/bottom
   - Joint optimization of vector pairs

4. **Machine Learning Cost Model**
   - Learn actual costs from benchmarks
   - CPU-specific optimization

5. **Blend/Mask Instructions**
   - Use masked operations where beneficial
   - AVX-512 mask registers

## Questions & Clarifications

### Q: What if no solution found for a stage?
**A**: Currently returns empty list. Should fall back to:
- Brute force permutation (multiple instructions)
- Cross-vector swaps
- Error/warning if truly impossible

### Q: How to handle first stage (no permutation needed)?
**A**: Special case - return no-op gadget with zero cost.
Pairs can land in any lane since input is unsorted.

### Q: What about element types (f32 vs i32)?
**A**: Currently handles via `element_bits` parameter.
Z3 models are bit-accurate, work for all types.

## Testing Strategy

1. **Unit Tests**: Individual components (✅ Done)
2. **Integration Tests**: Full optimization runs (TODO)
3. **Correctness Tests**: Verify synthesized code sorts correctly
4. **Performance Tests**: Compare against hand-written code
5. **Fuzzing**: Random bitonic networks, all parameters

## Performance Considerations

- Z3 solving can be slow for complex constraints
- Cache solutions per stage pattern
- Parallelize synthesis across vectors
- Early pruning of dominated solutions

## References

- [uops.info](https://uops.info) - Instruction latency/throughput data
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [Z3 Tutorial](https://ericpony.github.io/z3py-tutorial/guide-examples.htm)
- [Superoptimization Papers](https://en.wikipedia.org/wiki/Superoptimization)

