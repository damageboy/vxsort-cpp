The code-generator attempts to generate bitonic sorters for all possible vector sizes and primitive types.

It does so by employing a mini super-optimizer, which tries to generate the most efficient
permutation/shuffle operations for each stage of the bitonic sort.

The super-optimizer makes use of Z3 to verify that the generated code is correct, in terms
of guaranteeing the correct ordering of the elements in the vector.
Each vector is then min/maxed to produce the final stage outcome.

## BitonicSuperVectorizer

The `BitonicSuperVectorizer` class is the main entry point for the super-optimizer. It:

1. **Generates bitonic comparison stages** using `BitonicSorter`
2. **Synthesizes permutation gadgets** for each stage using Z3-based validation
3. **Builds a solution tree** exploring all valid permutation sequences
4. **Computes costs** based on CPU-specific instruction latencies
5. **Exports solutions** to JSON for downstream code generation

### Architecture

- **VectorState**: Tracks which elements are in which lanes of top/bottom vectors
- **InstructionSpec**: Represents a single AVX instruction with arguments
- **PermutationGadget**: Sequence of 0-3 instructions for top and bottom vectors
- **SolutionNode**: Tree node containing a gadget and links to next stage solutions
- **GadgetSynthesizer**: Enumerates and validates instruction combinations using Z3
- **CostModel**: CPU-specific instruction cost database

### Usage

```python
from bitonic_compiler import generate_bitonic_sorter
from utils import vector_machine, primitive_type

# Generate optimized solutions for 2 AVX2 vectors of i32
solutions = generate_bitonic_sorter(2, primitive_type.i32, vector_machine.AVX2)
```

### Testing

```bash
# Run unit tests
uv run pytest

# Run demonstration
uv run python src/demo_super_vectorizer.py

# Run full synthesis (limited to depth 3 for speed)
uv run python src/bitonic_compiler.py --depth-limit=3
```

### Current Status

**Implemented:**
- ✅ Core data structures (VectorState, PermutationGadget, SolutionNode)
- ✅ BitonicSorter stage generation
- ✅ GadgetSynthesizer with AVX2 i32 instruction enumeration
- ✅ Z3-based validation using pair_id encoding
- ✅ Solution tree building infrastructure
- ✅ Basic cost model (instruction count)
- ✅ JSON export

**In Progress:**
- 🔄 Full gadget synthesis (depth 2-3 instructions)
- 🔄 Output state computation after gadget application
- 🔄 Enhanced cost model with uops.info data

**Planned:**
- ⏳ AVX2 f32, i64, f64 support
- ⏳ AVX512 support
- ⏳ Solution pruning strategies
- ⏳ C++ code generation from JSON solutions

## AVX2 Instructions

For AVX2, we support 32-bit and 64-bit elements, and the Z3-based super-optimizer
can search for the best permutation/shuffle operations for each stage from the following list of instructions.

### 32-bit Elements (AVX / AVX2)

| Done? | Mnemonic                        | Operates on | Rough Description                                                                                                     |
|-------|---------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------|
| ✅     | **VSHUFPS**                     | 128/256     | Arbitrary 2-input shuffle of 32-bit elements (control byte selects which of 4 elements from each input go to output). |
| ✅     | **VUNPCKLPS / VUNPCKHPS**       | 128/256     | Unpack low/high 32-bit elements from two vectors (interleave).                                                        |
| ✅     | **VPUNPCKLDQ / VPUNPCKHDQ**     | 128/256     | Integer variant: interleave low/high 32-bit ints.                                                                     |
| ✅     | **VPUNPCKLQDQ / VPUNPCKHQDQ**   | 128/256     | Interleave 64-bit chunks (affects grouping of 32-bit).                                                                |
| ✅     | **VPSHUFD**                     | 128/256     | Permute 32-bit elements within a 128- or 256-bit lane (4-element permute per 128-bit half).                           |
|       | **VPSHUFLW / VPSHUFHW**         | 128/256     | Permute 16-bit halves, but indirectly affects 32-bit grouping (rarely useful for pure 32-bit shuffle).                |
| ✅     | **VPERMILPS**                   | 128/256     | Permute 32-bit elements within 128-bit lane (control via immediate or vector).                                        |
| ✅     | **VPERM2F128 / VPERM2I128**     | 256         | Cross-lane permute: select which 128-bit half comes from which source, optional zeroing.                              |
| ✅     | **VPERMD** (AVX2)               | 256         | Full variable permute of 32-bit elements across 256-bit register.                                                     |
| ✅     | **VPERMPS** (AVX2)              | 256         | Same as above but for float32.                                                                                        |
|       | **VBLENDPS**                    | 128/256     | Blend 32-bit elements from two inputs under immediate mask.                                                           |
|       | **VPBLENDD** (AVX2)             | 128/256     | Blend 32-bit ints from two inputs under immediate mask.                                                               |
|       | **VBLENDVPS**                   | 128/256     | Variable blend (mask in XMM/YMM register).                                                                            |
|       | **VINSERTF128 / VINSERTI128**   | 256         | Insert 128-bit lane into a 256-bit vector.                                                                            |
|       | **VEXTRACTF128 / VEXTRACTI128** | 256         | Extract 128-bit lane from a 256-bit vector.                                                                           |


### 64-bit Elements (AVX / AVX2)

| Mnemonic                        | Operates on | Rough Description                                                               |
|---------------------------------|-------------|---------------------------------------------------------------------------------|
| **VUNPCKLPD / VUNPCKHPD**       | 128/256     | Interleave low/high 64-bit elements from two vectors.                           |
| **VPUNPCKLQDQ / VPUNPCKHQDQ**   | 128/256     | Interleave low/high 64-bit integers.                                            |
| **VSHUFPD**                     | 128/256     | Arbitrary 2-input shuffle of 64-bit elements.                                   |
| **VPERMILPD**                   | 128/256     | Permute 64-bit elements within 128-bit lane (immediate or vector).              |
| **VPERM2F128 / VPERM2I128**     | 256         | Cross-lane permute (128-bit granularity).                                       |
| **VPERMQ** (AVX2)               | 256         | Full permute of 64-bit elements across 256-bit register (immediate).            |
| **VPERMPD** (AVX2)              | 256         | Same as above but for float64.                                                  |
| **VPBLENDD** (AVX2)             | 128/256     | Blend 64-bit elements indirectly via 32-bit mask (two 32-bit parts per 64-bit). |
| **VBLENDPD**                    | 128/256     | Blend 64-bit FP elements from two inputs (immediate mask).                      |
| **VBLENDVPD**                   | 128/256     | Variable blend (mask in XMM/YMM register).                                      |
| **VINSERTF128 / VINSERTI128**   | 256         | Insert 128-bit lane into a 256-bit vector.                                      |
| **VEXTRACTF128 / VEXTRACTI128** | 256         | Extract 128-bit lane from a 256-bit vector.                                     |
