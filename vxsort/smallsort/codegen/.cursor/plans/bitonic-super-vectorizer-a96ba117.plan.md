<!-- a96ba117-11e9-4b44-843e-07becb0ae3b1 12ba0074-db73-4a6c-a595-f107afcb2aef -->
# Bitonic Super Vectorizer Implementation

## Overview

Create a Z3-based super-optimizer that synthesizes optimal permutation sequences for bitonic sorting networks on SIMD vectors.

## Core Architecture

### 1. Data Structures (`bitonic-compiler.py`)

Add classes to represent:

- **PermutationGadget**: Encapsulates instruction sequence for top/bottom vectors
  - `top_instructions: list[InstructionSpec]` (0-3 instructions)
  - `bottom_instructions: list[InstructionSpec]` (0-3 instructions)
  - `validated: bool`

- **InstructionSpec**: Represents a single AVX instruction
  - `intrinsic_name: str` (e.g., "_mm256_permute_ps")
  - `args: dict` (operands, immediates, masks)

- **SolutionNode**: Tree node for one stage's solutions
  - `stage: int`
  - `input_state: VectorState` (element positions in top/bottom)
  - `output_state: VectorState`
  - `gadget: PermutationGadget`
  - `children: list[SolutionNode]` (next stage solutions)
  - `cost: float`

- **VectorState**: Tracks which elements are in which lanes
  - `top: list[int]` (element indices)
  - `bottom: list[int]`

### 2. BitonicSuperVectorizer Class (`bitonic-compiler.py`)

Core methods:

- `__init__(num_vecs: int, prim_type: primitive_type, vm: vector_machine)`
  - Initialize BitonicSorter to get stage pairs
  - Set up initial VectorState from stage 0 pairs

- `synthesize_all_stages() -> list[SolutionNode]`
  - Entry point: builds solution tree for all stages
  - Returns root nodes (stage 0 solutions)

- `synthesize_stage(input_state: VectorState, target_pairs: list[tuple[int,int]]) -> list[PermutationGadget]`
  - For given input state, find all valid gadgets that align target pairs
  - Try combinations: (0,0), (0,1), (1,0), (1,1), (0,2), (2,0), ... up to (3,3)
  - Return validated gadgets

- `build_solution_tree() -> list[SolutionNode]`
  - Recursively explore all stage transitions
  - For each stage, try all valid gadgets and recurse to next stage

- `compute_costs(roots: list[SolutionNode], cost_model: CostModel)`
  - Traverse tree, compute cumulative costs for each path

- `export_solutions(roots: list[SolutionNode], output_path: str)`
  - Generate JSON with all solutions and costs

### 3. Gadget Synthesizer (`bitonic-compiler.py`)

- `GadgetSynthesizer` class:
  - `__init__(vm: vector_machine, prim_type: primitive_type)`
  - `available_intrinsics: dict[str, callable]` - filtered from z3_avx.py

  - `enumerate_gadgets(input_state: VectorState, target_pairs: list[tuple[int,int]], max_depth: int) -> list[PermutationGadget]`
    - Generate candidate gadgets up to max_depth instructions per vector
    - For depth 0-3 on top, depth 0-3 on bottom
    - Uses Z3 to validate each candidate

  - `try_single_instruction_gadgets()` - try each intrinsic alone
  - `try_two_instruction_gadgets()` - try compositions
  - `try_three_instruction_gadgets()` - including blends

### 4. Z3 Validation (`bitonic-compiler.py` + `z3_avx.py`)

- `validate_gadget(gadget: PermutationGadget, input_state: VectorState, target_pairs: list[tuple[int,int]]) -> bool`
  - Create Z3 Solver instance
  - Map each element to a unique pair_id based on target_pairs
  - Create symbolic input registers with pair_id values
  - Apply gadget instructions using z3_avx functions
  - Add constraints: `forall lane i: top_output[i] == bottom_output[i]` (same pair_id)
  - Return `solver.check() == sat`

Pair encoding example:

```
target_pairs = [(0,8), (1,9), (2,10), (3,11), (4,12), (5,13), (6,14), (7,15)]
input_state.top = [0,1,2,3,4,5,6,7]
input_state.bottom = [8,9,10,11,12,13,14,15]

Assign pair_ids: {0:1, 8:1, 1:2, 9:2, ...}
Input: top_reg has values [1,2,3,4,5,6,7,8], bottom has [1,2,3,4,5,6,7,8]
Constraint: top_output[i] == bottom_output[i] for all lanes i
```

### 5. Cost Model (`cost_model.py` - new file)

- `CostModel` class:
  - `__init__(target_cpu: str = "generic")`
  - `instruction_costs: dict[str, InstructionCost]`

- `InstructionCost` dataclass:
  - `latency: float`
  - `throughput: float` (reciprocal)
  - `ports: list[str]` (e.g., ["p0", "p1", "p5"])

- `load_costs_from_uops_info()` - scrape or use cached data
  - Parse uops.info for AVX2/AVX512 instruction characteristics
  - Map intrinsic names to instruction costs

- `calculate_gadget_cost(gadget: PermutationGadget) -> float`
  - Sum latencies, account for parallelism via port analysis

- Start with simple instruction count, then enhance with real data

### 6. Integration & Testing

**Update `generate_bitonic_sorter()` function**:

```python
def generate_bitonic_sorter(num_vecs: int, type: primitive_type, vm: vector_machine):
    super_opt = BitonicSuperVectorizer(num_vecs, type, vm)
    solutions = super_opt.synthesize_all_stages()
    super_opt.compute_costs(solutions, CostModel("zen5"))
    super_opt.export_solutions(solutions, "bitonic_solutions.json")
    return solutions
```

**Create test cases**:

- Test with AVX2 i32, 2 vectors (16 elements)
- Validate known solutions (manual gadgets)
- Verify solution tree structure

## Implementation Order

1. Add data structure classes (PermutationGadget, InstructionSpec, SolutionNode, VectorState)
2. Implement VectorState initialization from BitonicSorter stages
3. Create GadgetSynthesizer with intrinsic enumeration (AVX2 i32 subset from z3_avx.py)
4. Implement Z3 validation with pair_id encoding
5. Build BitonicSuperVectorizer.synthesize_stage() for single stage
6. Test single stage synthesis with concrete example
7. Implement solution tree building across all stages
8. Add simple cost model (instruction count)
9. Implement JSON export
10. Integrate with generate_bitonic_sorter()
11. Enhance cost model with uops.info data

## Files to Modify/Create

- `vxsort/smallsort/codegen/bitonic-compiler.py` - main implementation
- `vxsort/smallsort/codegen/cost_model.py` - new file for cost analysis
- `vxsort/smallsort/codegen/test_super_vectorizer.py` - new test file

## Notes

- Initial scope: AVX2 + i32 only (8 elements per vector, 16 total)
- Gadget search: Try (top_depth, bottom_depth) from (0,0) to (3,3)
- Min-max exchange is fixed after each permutation cloud
- Solution tree may be large; consider pruning strategies later
- JSON output enables downstream C++ code generation

### To-dos

- [ ] Implement core data structures: PermutationGadget, InstructionSpec, SolutionNode, VectorState
- [ ] Implement VectorState initialization from BitonicSorter stages
- [ ] Create GadgetSynthesizer class with AVX2 i32 intrinsic enumeration
- [ ] Implement Z3-based gadget validation with pair_id encoding
- [ ] Implement BitonicSuperVectorizer.synthesize_stage() for single stage
- [ ] Create tests for single stage synthesis with concrete examples
- [ ] Implement build_solution_tree() to explore all stages
- [ ] Add simple instruction count cost model
- [ ] Implement JSON export for solution trees
- [ ] Integrate BitonicSuperVectorizer with generate_bitonic_sorter()
- [ ] Enhance cost model with uops.info data for realistic instruction costs