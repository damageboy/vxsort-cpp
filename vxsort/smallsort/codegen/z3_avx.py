import sys
from typing import Any
from z3.z3 import SeqRef, BitVecNumRef, BitVecRef, BitVec, BitVecVal, Solver, Extract, Concat, If, LShR, ZeroExt, simplify

zero = 0


def ymm_reg(name: str):
    return BitVec(name, 32 * 8)

def zmm_reg(name: str):
    return BitVec(name, 64 * 8)

def reg_with_values(name: str, s: Solver, raw_values, element_bits: int , total_bits: int):
    lanes = total_bits // element_bits
    assert len(raw_values) == lanes, f"Expected {lanes} values for {element_bits}-bit elements in {total_bits}-bit register, got {len(raw_values)}"
    
    # Create BitVec elements for each lane
    bv_elements = [BitVec(f"{name}_l_{i:02}", element_bits) for i in range(lanes)]
    
    # Add constraints for each element
    for i, raw_value in enumerate(raw_values):
        s.add(bv_elements[i] == BitVecVal(raw_value, element_bits))
    
    return simplify(Concat(bv_elements[::-1]))


def ymm_reg_with_32b_values(name: str, s: Solver, raw_values):
    return reg_with_values(name, s, raw_values, 32, 256)

def zmm_reg_with_32b_values(name: str, s: Solver, raw_values):
    return reg_with_values(name, s, raw_values, 32, 512)

def ymm_reg_with_64b_values(name: str, s: Solver, raw_values):
    return reg_with_values(name, s, raw_values, 64, 256)

def zmm_reg_with_64b_values(name: str, s: Solver, raw_values):
    return reg_with_values(name, s, raw_values, 64, 512)

def _reg_with_unique_values(name: str, s: Solver, lanes: int, bits: int):
    """
    Create a register with given number of lanes and element width, ensuring each lane is unique.
    """
    assert lanes * bits == 256 or lanes * bits == 512, "Total register size can only be 256 or 512 bits"

        # Create a new register
    if lanes * bits == 256:
        reg = ymm_reg(name)
    else:
        reg = zmm_reg(name)
    
    elems = [Extract(bits * (i + 1) - 1, bits * i, reg) for i in range(lanes)]
    for i in range(lanes):
        for j in range(i + 1, lanes):
            s.add(elems[i] != elems[j])
    return reg


def ymm_reg_with_unique_values(name: str, s: Solver, bits: int):
    lanes = 256 // bits
    return _reg_with_unique_values(name, s, lanes=lanes, bits=bits)


def zmm_reg_with_unique_values(name: str, s: Solver, bits: int):
    lanes = 512 // bits
    return _reg_with_unique_values(name, s, lanes=lanes, bits=bits)


def ymm_reg_pair_with_unique_values(name_prefix: str, s: Solver, bits: int):
    # Create two registers with internal uniqueness
    reg1 = ymm_reg_with_unique_values(f"{name_prefix}1", s, bits)
    reg2 = ymm_reg_with_unique_values(f"{name_prefix}2", s, bits)
    
    # Extract all elements from both registers
    lanes = 256 // bits
    reg1_elems = [Extract(bits * (i + 1) - 1, bits * i, reg1) for i in range(lanes)]
    reg2_elems = [Extract(bits * (i + 1) - 1, bits * i, reg2) for i in range(lanes)]
    
    # Add cross-register uniqueness constraints
    for reg1_elem in reg1_elems:
        for reg2_elem in reg2_elems:
            s.add(reg1_elem != reg2_elem)
    
    return reg1, reg2


def zmm_reg_pair_with_unique_values(name_prefix: str, s: Solver, bits: int):
    # Create two registers with internal uniqueness
    reg1 = zmm_reg_with_unique_values(f"{name_prefix}1", s, bits)
    reg2 = zmm_reg_with_unique_values(f"{name_prefix}2", s, bits)
    
    # Extract all elements from both registers
    lanes = 512 // bits
    reg1_elems = [Extract(bits * (i + 1) - 1, bits * i, reg1) for i in range(lanes)]
    reg2_elems = [Extract(bits * (i + 1) - 1, bits * i, reg2) for i in range(lanes)]
    
    # Add cross-register uniqueness constraints
    for reg1_elem in reg1_elems:
        for reg2_elem in reg2_elems:
            s.add(reg1_elem != reg2_elem)
    
    return reg1, reg2


# Type definition for element specifications
ElementSpecs = list[tuple[BitVecRef, int]]

def construct_reg_from_elements(bits: int, element_specs: ElementSpecs, total_bits: int):
    lanes = total_bits // bits
    assert len(element_specs) == lanes, f"Expected {lanes} element specs for {bits}-bit elements in {total_bits}-bit register, got {len(element_specs)}"
    
    # Extract each specified element
    elements: list[BitVecRef | SeqRef] = []
    for reg, elem_idx in element_specs:
        assert 0 <= elem_idx < lanes, f"Element index {elem_idx} out of range for {bits}-bit elements (0-{lanes-1})"
        start_bit = elem_idx * bits
        end_bit = start_bit + bits - 1
        elements.append(Extract(end_bit, start_bit, reg))
    
    # Concatenate in reverse order for Z3 (MSB first)
    return simplify(Concat(elements[::-1]))


def construct_ymm_reg_from_elements(bits: int, element_specs: ElementSpecs):
    return construct_reg_from_elements(bits, element_specs, 256)


def construct_zmm_reg_from_elements(bits: int, element_specs: ElementSpecs):
    return construct_reg_from_elements(bits, element_specs, 512)


def _reg_reversed(name: str, s: Solver, original_reg, lanes: int, bits: int):
    assert lanes * bits == 256 or lanes * bits == 512, "Total register size can only be 256 or 512 bits"
    
    # Create a new register
    if lanes * bits == 256:
        reversed_reg = ymm_reg(name)
    else:
        reversed_reg = zmm_reg(name)
    
    # Extract elements from both registers
    orig_elems = [Extract(bits * (i + 1) - 1, bits * i, original_reg) for i in range(lanes)]
    rev_elems = [Extract(bits * (i + 1) - 1, bits * i, reversed_reg) for i in range(lanes)]
    
    # Add constraints that reversed register elements equal original register elements in reverse order
    for i in range(lanes):
        s.add(rev_elems[i] == orig_elems[lanes - 1 - i])
    
    return reversed_reg


def ymm_reg_reversed(name, s, original_reg, bits):
    """Create a YMM register that is the reverse of the original register through constraints."""
    lanes = 256 // bits
    return _reg_reversed(name, s, original_reg, lanes, bits)


def zmm_reg_reversed(name, s, original_reg, bits):
    """Create a ZMM register that is the reverse of the original register through constraints."""  
    lanes = 512 // bits
    return _reg_reversed(name, s, original_reg, lanes, bits)

ymm_regs = [ymm_reg(f"ymm{i}") for i in range(16)]
zmm_regs = [zmm_reg(f"zmm{i}") for i in range(32)]


def to_num(v):
    d = v[0]
    for p in v[1:]:
        d = (d << 8) + p

    return d


def _MM_SHUFFLE2(x, y):
    """
    Mimics the standard _MM_SHUFFLE2 intrinsic macro.
    Returns (x << 1) | y
    """
    return (x << 1) | y


def _MM_SHUFFLE(z, y, x, w):
    """
    Mimics the standard _MM_SHUFFLE intrinsic macro.
    Returns (z<<6) | (y<<4) | (x<<2) | w
    """
    return (z << 6) | (y << 4) | (x << 2) | w


##
# Single vector variable permutes


# AVX2: vpermd/_mm256_permutevar_epi32
def _mm256_permutexvar_epi32(op1: BitVecRef, op_idx: BitVecRef):
    """
    Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
    Implements __m256i _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx)
    using ymm_regs and Z3 bitvector operations.

    Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
    Operation:
    ```
    FOR j := 0 to 7
        i := j*32
        id := idx[i+2:i]*32
        dst[i+31:i] := a[id+31:id]
    ENDFOR
    dst[MAX:256] := 0
    ```
    """
    elems = [None] * 8

    for j in range(8):
        i = j * 32

        # Extract 3 bits for index: idx[i+2:i] (need 3 bits to represent 0-7)
        idx_bits = Extract(i + 2, i, op_idx)

        # Use nested If statements to handle each possible index value (0-7)
        # Each index selects a different 32-bit chunk from the input
        elems[j] = simplify(
            If(
                idx_bits == 0,
                Extract(1 * 32 - 1, 0 * 32, op1),
                If(
                    idx_bits == 1,
                    Extract(2 * 32 - 1, 1 * 32, op1),
                    If(
                        idx_bits == 2,
                        Extract(3 * 32 - 1, 2 * 32, op1),
                        If(
                            idx_bits == 3,
                            Extract(4 * 32 - 1, 3 * 32, op1),
                            If(
                                idx_bits == 4,
                                Extract(5 * 32 - 1, 4 * 32, op1),
                                If(
                                    idx_bits == 5,
                                    Extract(6 * 32 - 1, 5 * 32, op1),
                                    If(
                                        idx_bits == 6,
                                        Extract(7 * 32 - 1, 6 * 32, op1),
                                        Extract(8 * 32 - 1, 7 * 32, op1),  # idx_bits == 7
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )

    return simplify(Concat(elems[::-1]))


# AVX512: vpermd/_mm512_permutexvar_epi32
def _mm512_permutexvar_epi32(op1: BitVecRef, op_idx: BitVecRef):
    """
    Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
    Implements __m512i _mm512_permutexvar_epi32 (__m512i idx, __m512i a)
    using zmm_regs and Z3 bitvector operations.


    Operation:
    ```
    FOR j := 0 to 15
        i := j*32
        id := idx[i+3:i]*32
        dst[i+31:i] := a[id+31:id]
    ENDFOR
    dst[MAX:512] := 0
    ```
    """

    chunks = [None] * 16  # Need 16 chunks for 512-bit register

    for j in range(16):
        i = j * 32

        # Extract 4 bits for index: idx[i+3:i] as per pseudocode
        idx_bits = Extract(i + 3, i, op_idx)

        # Use nested If statements to handle each possible index value (0-15)
        # Each index selects a different 32-bit chunk from the input
        chunks[j] = simplify(
            If(
                idx_bits == 0,
                Extract(1 * 32 - 1, 0 * 32, op1),
                If(
                    idx_bits == 1,
                    Extract(2 * 32 - 1, 1 * 32, op1),
                    If(
                        idx_bits == 2,
                        Extract(3 * 32 - 1, 2 * 32, op1),
                        If(
                            idx_bits == 3,
                            Extract(4 * 32 - 1, 3 * 32, op1),
                            If(
                                idx_bits == 4,
                                Extract(5 * 32 - 1, 4 * 32, op1),
                                If(
                                    idx_bits == 5,
                                    Extract(6 * 32 - 1, 5 * 32, op1),
                                    If(
                                        idx_bits == 6,
                                        Extract(7 * 32 - 1, 6 * 32, op1),
                                        If(
                                            idx_bits == 7,
                                            Extract(8 * 32 - 1, 7 * 32, op1),
                                            If(
                                                idx_bits == 8,
                                                Extract(9 * 32 - 1, 8 * 32, op1),
                                                If(
                                                    idx_bits == 9,
                                                    Extract(10 * 32 - 1, 9 * 32, op1),
                                                    If(
                                                        idx_bits == 10,
                                                        Extract(11 * 32 - 1, 10 * 32, op1),
                                                        If(
                                                            idx_bits == 11,
                                                            Extract(12 * 32 - 1, 11 * 32, op1),
                                                            If(
                                                                idx_bits == 12,
                                                                Extract(13 * 32 - 1, 12 * 32, op1),
                                                                If(
                                                                    idx_bits == 13,
                                                                    Extract(14 * 32 - 1, 13 * 32, op1),
                                                                    If(
                                                                        idx_bits == 14,
                                                                        Extract(15 * 32 - 1, 14 * 32, op1),
                                                                        Extract(16 * 32 - 1, 15 * 32, op1),  # idx_bits == 15
                                                                    ),
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )

    return simplify(Concat(chunks[::-1]))


# AVX512: vpermi2d/vpermt2d/_mm512_permutex2var_epi32
def _mm512_permutex2var_epi32(a: BitVecRef, idx: BitVecRef, b: BitVecRef):
    """
    Shuffle 32-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
    Implements __m512i _mm512_permutex2var_epi32 (__m512i a, __m512i idx, __m512i b)
    using zmm_regs and Z3 bitvector operations.

    Operation:
    ```
    FOR j := 0 to 15
        i := j*32
        off := idx[i+3:i]*32
        dst[i+31:i] := idx[i+4] ? b[off+31:off] : a[off+31:off]
    ENDFOR
    dst[MAX:512] := 0
    ```
    """
    elements = [None] * 16  # Need 16 elements for 512-bit register

    for j in range(16):
        i = j * 32

        # Extract offset: idx[i+3:i] (4 bits to represent indices 0-15)
        offset_bits = Extract(i + 3, i, idx)
        
        # Extract source selector: idx[i+4] (1 bit to choose between a and b)
        source_selector = Extract(i + 4, i + 4, idx)

        # First select the source vector based on source_selector
        # source_selector == 0 -> choose from a, source_selector == 1 -> choose from b
        selected_source = simplify(
            If(
                source_selector == 0,
                a,
                b
            )
        )

        # Then select element from the chosen source based on offset
        elements[j] = simplify(
            If(
                offset_bits == 0,
                Extract(1 * 32 - 1, 0 * 32, selected_source),
                If(
                    offset_bits == 1,
                    Extract(2 * 32 - 1, 1 * 32, selected_source),
                    If(
                        offset_bits == 2,
                        Extract(3 * 32 - 1, 2 * 32, selected_source),
                        If(
                            offset_bits == 3,
                            Extract(4 * 32 - 1, 3 * 32, selected_source),
                            If(
                                offset_bits == 4,
                                Extract(5 * 32 - 1, 4 * 32, selected_source),
                                If(
                                    offset_bits == 5,
                                    Extract(6 * 32 - 1, 5 * 32, selected_source),
                                    If(
                                        offset_bits == 6,
                                        Extract(7 * 32 - 1, 6 * 32, selected_source),
                                        If(
                                            offset_bits == 7,
                                            Extract(8 * 32 - 1, 7 * 32, selected_source),
                                            If(
                                                offset_bits == 8,
                                                Extract(9 * 32 - 1, 8 * 32, selected_source),
                                                If(
                                                    offset_bits == 9,
                                                    Extract(10 * 32 - 1, 9 * 32, selected_source),
                                                    If(
                                                        offset_bits == 10,
                                                        Extract(11 * 32 - 1, 10 * 32, selected_source),
                                                        If(
                                                            offset_bits == 11,
                                                            Extract(12 * 32 - 1, 11 * 32, selected_source),
                                                            If(
                                                                offset_bits == 12,
                                                                Extract(13 * 32 - 1, 12 * 32, selected_source),
                                                                If(
                                                                    offset_bits == 13,
                                                                    Extract(14 * 32 - 1, 13 * 32, selected_source),
                                                                    If(
                                                                        offset_bits == 14,
                                                                        Extract(15 * 32 - 1, 14 * 32, selected_source),
                                                                        Extract(16 * 32 - 1, 15 * 32, selected_source),  # offset_bits == 15
                                                                    ),
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )

    return simplify(Concat(elements[::-1]))


# AVX512: vpermi2q/vpermt2q/_mm512_permutex2var_epi64
def _mm512_permutex2var_epi64(a: BitVecRef, idx: BitVecRef, b: BitVecRef):
    """
    Shuffle 64-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
    Implements __m512i _mm512_permutex2var_epi64 (__m512i a, __m512i idx, __m512i b)
    using zmm_regs and Z3 bitvector operations.

    Operation:
    ```
    FOR j := 0 to 7
        i := j*64
        off := idx[i+2:i]*64
        dst[i+63:i] := idx[i+3] ? b[off+63:off] : a[off+63:off]
    ENDFOR
    dst[MAX:512] := 0
    ```
    """
    elements = [None] * 8  # Need 8 elements for 512-bit register with 64-bit elements

    for j in range(8):
        i = j * 64

        # Extract offset: idx[i+2:i] (3 bits to represent indices 0-7)
        offset_bits = Extract(i + 2, i, idx)
        
        # Extract source selector: idx[i+3] (1 bit to choose between a and b)
        source_selector = Extract(i + 3, i + 3, idx)

        # First select the source vector based on source_selector
        # source_selector == 0 -> choose from a, source_selector == 1 -> choose from b
        selected_source = simplify(
            If(
                source_selector == 0,
                a,
                b
            )
        )

        # Then select element from the chosen source based on offset
        elements[j] = simplify(
            If(
                offset_bits == 0,
                Extract(1 * 64 - 1, 0 * 64, selected_source),
                If(
                    offset_bits == 1,
                    Extract(2 * 64 - 1, 1 * 64, selected_source),
                    If(
                        offset_bits == 2,
                        Extract(3 * 64 - 1, 2 * 64, selected_source),
                        If(
                            offset_bits == 3,
                            Extract(4 * 64 - 1, 3 * 64, selected_source),
                            If(
                                offset_bits == 4,
                                Extract(5 * 64 - 1, 4 * 64, selected_source),
                                If(
                                    offset_bits == 5,
                                    Extract(6 * 64 - 1, 5 * 64, selected_source),
                                    If(
                                        offset_bits == 6,
                                        Extract(7 * 64 - 1, 6 * 64, selected_source),
                                        Extract(8 * 64 - 1, 7 * 64, selected_source),  # offset_bits == 7
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )

    return simplify(Concat(elements[::-1]))


# AVX512: vpermt2ps/_mm512_mask_permutex2var_ps (masked version)
def _mm512_mask_permutex2var_ps(a: BitVecRef, k: BitVecRef, idx: BitVecRef, b: BitVecRef):
    """
    Shuffle single-precision (32-bit) floating-point elements in a and b across lanes using the corresponding selector and index in idx, 
    and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
    Implements __m512 _mm512_mask_permutex2var_ps (__m512 a, __mmask16 k, __m512i idx, __m512 b)
    using zmm_regs and Z3 bitvector operations.

    Operation:
    ```
    FOR j := 0 to 15
        i := j*32
        off := idx[i+3:i]*32
        IF k[j]
            dst[i+31:i] := idx[i+4] ? b[off+31:off] : a[off+31:off]
        ELSE
            dst[i+31:i] := a[i+31:i]
        FI
    ENDFOR
    dst[MAX:512] := 0
    ```
    """
    elements = [None] * 16  # Need 16 elements for 512-bit register

    for j in range(16):
        i = j * 32

        # Extract the mask bit for this element position
        mask_bit = Extract(j, j, k)
        
        # Extract the corresponding element from a (fallback when mask bit is 0)
        fallback_element = Extract(i + 31, i, a)

        # Only compute permutation if mask bit is set
        # Extract offset: idx[i+3:i] (4 bits to represent indices 0-15)
        offset_bits = Extract(i + 3, i, idx)
        
        # Extract source selector: idx[i+4] (1 bit to choose between a and b)
        source_selector = Extract(i + 4, i + 4, idx)

        # First select the source vector based on source_selector
        # source_selector == 0 -> choose from a, source_selector == 1 -> choose from b
        selected_source = simplify(
            If(
                source_selector == 0,
                a,
                b
            )
        )

        # Then select element from the chosen source based on offset
        permuted_element = simplify(
            If(
                offset_bits == 0,
                Extract(1 * 32 - 1, 0 * 32, selected_source),
                If(
                    offset_bits == 1,
                    Extract(2 * 32 - 1, 1 * 32, selected_source),
                    If(
                        offset_bits == 2,
                        Extract(3 * 32 - 1, 2 * 32, selected_source),
                        If(
                            offset_bits == 3,
                            Extract(4 * 32 - 1, 3 * 32, selected_source),
                            If(
                                offset_bits == 4,
                                Extract(5 * 32 - 1, 4 * 32, selected_source),
                                If(
                                    offset_bits == 5,
                                    Extract(6 * 32 - 1, 5 * 32, selected_source),
                                    If(
                                        offset_bits == 6,
                                        Extract(7 * 32 - 1, 6 * 32, selected_source),
                                        If(
                                            offset_bits == 7,
                                            Extract(8 * 32 - 1, 7 * 32, selected_source),
                                            If(
                                                offset_bits == 8,
                                                Extract(9 * 32 - 1, 8 * 32, selected_source),
                                                If(
                                                    offset_bits == 9,
                                                    Extract(10 * 32 - 1, 9 * 32, selected_source),
                                                    If(
                                                        offset_bits == 10,
                                                        Extract(11 * 32 - 1, 10 * 32, selected_source),
                                                        If(
                                                            offset_bits == 11,
                                                            Extract(12 * 32 - 1, 11 * 32, selected_source),
                                                            If(
                                                                offset_bits == 12,
                                                                Extract(13 * 32 - 1, 12 * 32, selected_source),
                                                                If(
                                                                    offset_bits == 13,
                                                                    Extract(14 * 32 - 1, 13 * 32, selected_source),
                                                                    If(
                                                                        offset_bits == 14,
                                                                        Extract(15 * 32 - 1, 14 * 32, selected_source),
                                                                        Extract(16 * 32 - 1, 15 * 32, selected_source),  # offset_bits == 15
                                                                    ),
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )

        # Apply mask: if mask bit is set, use permuted element, otherwise use fallback from a
        elements[j] = simplify(
            If(
                mask_bit == 1,
                permuted_element,
                fallback_element
            )
        )

    return simplify(Concat(elements[::-1]))


# AVX2: vpermq/_mm256_permutexvar_epi64
def _mm256_permutexvar_epi64(op1: BitVecRef, op_idx: BitVecRef):
    chunks = [None] * 4  # 4 chunks for 64-bit elements in 256-bit register

    for j in range(4):
        i = j * 64

        # Extract 2 bits for index: idx[i+1:i] (need 2 bits to represent 0-3)
        idx_bits = Extract(i + 1, i, op_idx)

        # Use nested If statements to handle each possible index value (0-3)
        # Each index selects a different 64-bit chunk from the input
        chunks[j] = simplify(
            If(
                idx_bits == 0,
                Extract(1 * 64 - 1, 0 * 64, op1),
                If(
                    idx_bits == 1,
                    Extract(2 * 64 - 1, 1 * 64, op1),
                    If(
                        idx_bits == 2,
                        Extract(3 * 64 - 1, 2 * 64, op1),
                        Extract(4 * 64 - 1, 3 * 64, op1),  # idx_bits == 3
                    ),
                ),
            )
        )

    return simplify(Concat(chunks[::-1]))


# AVX512: vpermq/_mm512_permutexvar_epi64
def _mm512_permutexvar_epi64(op1: BitVecRef, op_idx: BitVecRef):
    chunks = [None] * 8  # 8 chunks for 64-bit elements in 512-bit register

    for j in range(8):
        i = j * 64

        # Extract 3 bits for index: idx[i+2:i] (need 3 bits to represent 0-7)
        idx_bits = Extract(i + 2, i, op_idx)

        # Use nested If statements to handle each possible index value (0-7)
        # Each index selects a different 64-bit chunk from the input
        chunks[j] = simplify(
            If(
                idx_bits == 0,
                Extract(1 * 64 - 1, 0 * 64, op1),
                If(
                    idx_bits == 1,
                    Extract(2 * 64 - 1, 1 * 64, op1),
                    If(
                        idx_bits == 2,
                        Extract(3 * 64 - 1, 2 * 64, op1),
                        If(
                            idx_bits == 3,
                            Extract(4 * 64 - 1, 3 * 64, op1),
                            If(
                                idx_bits == 4,
                                Extract(5 * 64 - 1, 4 * 64, op1),
                                If(
                                    idx_bits == 5,
                                    Extract(6 * 64 - 1, 5 * 64, op1),
                                    If(
                                        idx_bits == 6,
                                        Extract(7 * 64 - 1, 6 * 64, op1),
                                        Extract(8 * 64 - 1, 7 * 64, op1),  # idx_bits == 7
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )

    return simplify(Concat(chunks[::-1]))


##
# Single vector 128-bit static permutes


# Helper function for permutes/shuffles
def _select4_ps(src_128: BitVecRef, select: BitVecRef | BitVecNumRef) -> BitVecRef:
    """Selects a 32-bit element from a 128-bit vector based on a 2-bit control."""
    return simplify(
        If(
            select == 0,
            Extract(31, 0, src_128),
            If(
                select == 1,
                Extract(63, 32, src_128),
                If(
                    select == 2,
                    Extract(95, 64, src_128),
                    Extract(127, 96, src_128),  # select == 3
                ),
            ),
        )
    )


# Helper function for permutes/shuffles (64-bit elements)
def _select2_pd(src_128: BitVecRef, select: BitVecRef | BitVecNumRef) -> BitVecRef:
    """Selects a 64-bit element from a 128-bit vector based on a 1-bit control."""
    return simplify(
        If(
            select == 0,
            Extract(63, 0, src_128),
            Extract(127, 64, src_128),  # select == 1
        )
    )


# Helper function for permutes/shuffles
def _extract_ctl4(imm: BitVecRef | BitVecNumRef):
    ctrl01 = Extract(1, 0, imm)
    ctrl23 = Extract(3, 2, imm)
    ctrl45 = Extract(5, 4, imm)
    ctrl67 = Extract(7, 6, imm)
    return ctrl01, ctrl23, ctrl45, ctrl67


# Helper function for permutes/shuffles (2-bit controls for pd)
def _extract_ctl2(imm: BitVecRef | BitVecNumRef):
    ctrl0 = Extract(0, 0, imm)
    ctrl1 = Extract(1, 1, imm)
    return ctrl0, ctrl1

def extract_128b_lane(input: BitVecRef, lane_idx: int):
    lane_start_bit = lane_idx * 128
    lane_end_bit = lane_start_bit + 127
    return Extract(lane_end_bit, lane_start_bit, input)

def vpermilps_lane(lane_idx: int, a: BitVecRef, ctrl01: BitVecRef, ctrl23: BitVecRef, ctrl45: BitVecRef, ctrl67: BitVecRef):
    src_lane = extract_128b_lane(a, lane_idx)

    chunks: list[BitVecRef|None] = [None] * 4
    chunks[0] = _select4_ps(src_lane, ctrl01)
    chunks[1] = _select4_ps(src_lane, ctrl23)
    chunks[2] = _select4_ps(src_lane, ctrl45)
    chunks[3] = _select4_ps(src_lane, ctrl67)
    return chunks

def vpermilpd_lane(lane_idx: int, a: BitVecRef, ctrl0: BitVecRef, ctrl1: BitVecRef):
    src_lane = extract_128b_lane(a, lane_idx)

    chunks: list[BitVecRef|None] = [None] * 2
    chunks[0] = _select2_pd(src_lane, ctrl0)
    chunks[1] = _select2_pd(src_lane, ctrl1)
    return chunks

def vshufps_lane(lane_idx: int, a: BitVecRef, b: BitVecRef, ctrl01: BitVecRef, ctrl23: BitVecRef, ctrl45: BitVecRef, ctrl67: BitVecRef) -> None:
    a_lane = extract_128b_lane(a, lane_idx)
    b_lane = extract_128b_lane(b, lane_idx)

    chunks: list[BitVecRef] = [None] * 4
    chunks[0] = _select4_ps(a_lane, ctrl01)
    chunks[1] = _select4_ps(a_lane, ctrl23)
    chunks[2] = _select4_ps(b_lane, ctrl45)
    chunks[3] = _select4_ps(b_lane, ctrl67)
    return chunks

def vshufpd_lane(lane_idx: int, a: BitVecRef, b: BitVecRef, imm: BitVecRef):
    a_lane = extract_128b_lane(a, lane_idx)
    b_lane = extract_128b_lane(b, lane_idx)

    # Each lane uses 2 control bits: lane i uses imm[2*i] and imm[2*i+1]
    ctrl0 = Extract(2 * lane_idx, 2 * lane_idx, imm)      # Controls selection from a
    ctrl1 = Extract(2 * lane_idx + 1, 2 * lane_idx + 1, imm)  # Controls selection from b

    chunks: list[BitVecRef|None] = [None] * 2
    chunks[0] = _select2_pd(a_lane, ctrl0)
    chunks[1] = _select2_pd(b_lane, ctrl1)
    return chunks

# AVX2: vpermilps/vpshufd/AVX-512 (_mm512_permute_ps/_mm512_shuffle_epi32)
def _mm256_permute_ps(op1: BitVecRef, imm8: BitVecRef | int):
    """
    Permutes 32-bit elements within each 128-bit lane
    of the source vector 'a' using the control bits in 'imm8'.
    Operates on YMM registers.
    """
    a = op1
    # Support constants or BitVec
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    ctrl01, ctrl23, ctrl45, ctrl67 = _extract_ctl4(imm)

    # Process each 128-bit lane (AVX-2 has two lanes in a 256-bit register)
    chunks_128b = [vpermilps_lane(lane_idx, a, ctrl01, ctrl23, ctrl45, ctrl67) for lane_idx in range(2)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    return simplify(Concat(flat_chunks[::-1])) # MSBs go first (for Z3)

# AVX512:  vpermilps/vpshufd (_mm512_permute_ps/_mm512_shuffle_epi32)
def _mm512_permute_ps(op1: BitVecRef, imm8: BitVecRef | int):
    """
    Permutes 32-bit floating-point elements in each 128-bit lane
    of the source vector 'a' using the control bits in 'imm8'.
    """
    a = op1
 
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    ctrl01, ctrl23, ctrl45, ctrl67 = _extract_ctl4(imm)
    # Process each 128-bit lane (AVX-512 has four lanes in a 512-bit register)
    chunks_128b = [vpermilps_lane(lane_idx, a, ctrl01, ctrl23, ctrl45, ctrl67) for lane_idx in range(4)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    return simplify(Concat(flat_chunks[::-1]))  # Reverse for Z3

# AVX-2: vpermilpd (_mm256_permute_pd)
def _mm256_permute_pd(op1: BitVecRef, imm8: BitVecRef | int):
    """
    Permutes 64-bit double-precision floating-point elements within each 128-bit lane
    of the source vector 'a' using the control bits in 'imm8'.
    Operates on YMM registers.
    """
    a = op1
    # Support constants or BitVec
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    ctrl0, ctrl1 = _extract_ctl2(imm)

    # Process each 128-bit lane (AVX-2 has two lanes in a 256-bit register)
    chunks_128b = [vpermilpd_lane(lane_idx, a, ctrl0, ctrl1) for lane_idx in range(2)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    return simplify(Concat(flat_chunks[::-1])) # MSBs go first (for Z3)


# AVX512: vpermilpd (_mm512_permute_pd)
def _mm512_permute_pd(op1: BitVecRef, imm8: BitVecRef | int):
    """
    Permutes 64-bit double-precision floating-point elements in each 128-bit lane
    of the source vector 'a' using the control bits in 'imm8'.
    """
    a = op1
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    ctrl0, ctrl1 = _extract_ctl2(imm)
    # Process each 128-bit lane (AVX-512 has four lanes in a 512-bit register)
    chunks_128b = [vpermilpd_lane(lane_idx, a, ctrl0, ctrl1) for lane_idx in range(4)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    return simplify(Concat(flat_chunks[::-1])) # MSBs go first (for Z3)


##
# 2 vector 128-bit static permutes


# AVX2: vshufps (_mm256_shuffle_ps)
def _mm256_shuffle_ps(op1: BitVecRef, op2: BitVecRef, imm8: BitVecRef | int):
    """
    Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in imm8, and store the results in dst.
    Implements __m256 _mm256_shuffle_ps (__m256 a, __m256 b, const int imm8)
    according to the Intel spec.

    Operation
    ```
    DEFINE SELECT4(src, control) {
        CASE(control[1:0]) OF
        0:	tmp[31:0] := src[31:0]
        1:	tmp[31:0] := src[63:32]
        2:	tmp[31:0] := src[95:64]
        3:	tmp[31:0] := src[127:96]
        ESAC
        RETURN tmp[31:0]
    }
    dst[31:0] := SELECT4(a[127:0], imm8[1:0])
    dst[63:32] := SELECT4(a[127:0], imm8[3:2])
    dst[95:64] := SELECT4(b[127:0], imm8[5:4])
    dst[127:96] := SELECT4(b[127:0], imm8[7:6])
    dst[159:128] := SELECT4(a[255:128], imm8[1:0])
    dst[191:160] := SELECT4(a[255:128], imm8[3:2])
    dst[223:192] := SELECT4(b[255:128], imm8[5:4])
    dst[255:224] := SELECT4(b[255:128], imm8[7:6])
    dst[MAX:256] := 0
    ```
    """
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    ctrl01, ctrl23, ctrl45, ctrl67 = _extract_ctl4(imm)

    chunks_128b = [vshufps_lane(lane_idx, op1, op2, ctrl01, ctrl23, ctrl45, ctrl67) for lane_idx in range(2)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    return simplify(Concat(flat_chunks[::-1])) # MSBs go first (for Z3)


# AVX512: vshufps (_mm512_shuffle_ps)
def _mm512_shuffle_ps(op1: BitVecRef, op2: BitVecRef, imm8: BitVecRef | int):
    """
        Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in imm8, and store the results in dst.

    Implements __m512 _mm512_shuffle_ps (__m512 a, __m512 b, const int imm8)
    according to the Intel spec.

        Operation
    ```
    DEFINE SELECT4(src, control) {
        CASE(control[1:0]) OF
        0:	tmp[31:0] := src[31:0]
        1:	tmp[31:0] := src[63:32]
        2:	tmp[31:0] := src[95:64]
        3:	tmp[31:0] := src[127:96]
        ESAC
        RETURN tmp[31:0]
    }
    dst[31:0] := SELECT4(a[127:0], imm8[1:0])
    dst[63:32] := SELECT4(a[127:0], imm8[3:2])
    dst[95:64] := SELECT4(b[127:0], imm8[5:4])
    dst[127:96] := SELECT4(b[127:0], imm8[7:6])
    dst[159:128] := SELECT4(a[255:128], imm8[1:0])
    dst[191:160] := SELECT4(a[255:128], imm8[3:2])
    dst[223:192] := SELECT4(b[255:128], imm8[5:4])
    dst[255:224] := SELECT4(b[255:128], imm8[7:6])
    dst[287:256] := SELECT4(a[383:256], imm8[1:0])
    dst[319:288] := SELECT4(a[383:256], imm8[3:2])
    dst[351:320] := SELECT4(b[383:256], imm8[5:4])
    dst[383:352] := SELECT4(b[383:256], imm8[7:6])
    dst[415:384] := SELECT4(a[511:384], imm8[1:0])
    dst[447:416] := SELECT4(a[511:384], imm8[3:2])
    dst[479:448] := SELECT4(b[511:384], imm8[5:4])
    dst[511:480] := SELECT4(b[511:384], imm8[7:6])
    dst[MAX:512] := 0
    ```
    """
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    ctrl01, ctrl23, ctrl45, ctrl67 = _extract_ctl4(imm)

    chunks_128b = [vshufps_lane(lane_idx, op1, op2, ctrl01, ctrl23, ctrl45, ctrl67) for lane_idx in range(4)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    return simplify(Concat(flat_chunks[::-1])) # MSBs go first (for Z3)


# AVX2: vshufpd (_mm256_shuffle_pd)
def _mm256_shuffle_pd(op1: BitVecRef, op2: BitVecRef, imm8: BitVecRef | int):
    """
    Shuffle double-precision (64-bit) floating-point elements within 128-bit lanes using the control in imm8, and store the results in dst.
    Implements __m256d _mm256_shuffle_pd (__m256d a, __m256d b, const int imm8)
    according to the Intel spec.

    Operation:
    ```
    dst[63:0] := (imm8[0] == 0) ? a[63:0] : a[127:64]
    dst[127:64] := (imm8[1] == 0) ? b[63:0] : b[127:64]
    dst[191:128] := (imm8[2] == 0) ? a[191:128] : a[255:192]
    dst[255:192] := (imm8[3] == 0) ? b[191:128] : b[255:192]
    dst[MAX:256] := 0
    ```
    """
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    chunks_128b = [vshufpd_lane(lane_idx, op1, op2, imm) for lane_idx in range(2)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    return simplify(Concat(flat_chunks[::-1])) # MSBs go first (for Z3)


# AVX512: vshufpd (_mm512_shuffle_pd)
def _mm512_shuffle_pd(op1: BitVecRef, op2: BitVecRef, imm8: BitVecRef | int):
    """
    Shuffle double-precision (64-bit) floating-point elements within 128-bit lanes using the control in imm8, and store the results in dst.
    Implements __m512d _mm512_shuffle_pd (__m512d a, __m512d b, const int imm8)
    according to the Intel spec.

    Operation:
    ```
    dst[63:0] := (imm8[0] == 0) ? a[63:0] : a[127:64]
    dst[127:64] := (imm8[1] == 0) ? b[63:0] : b[127:64]
    dst[191:128] := (imm8[2] == 0) ? a[191:128] : a[255:192]
    dst[255:192] := (imm8[3] == 0) ? b[191:128] : b[255:192]
    dst[319:256] := (imm8[4] == 0) ? a[319:256] : a[383:320]
    dst[383:320] := (imm8[5] == 0) ? b[319:256] : b[383:320]
    dst[447:384] := (imm8[6] == 0) ? a[447:384] : a[511:448]
    dst[511:448] := (imm8[7] == 0) ? b[447:384] : b[511:448]
    dst[MAX:512] := 0
    ```
    """
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    chunks_128b = [vshufpd_lane(lane_idx, op1, op2, imm) for lane_idx in range(4)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    return simplify(Concat(flat_chunks[::-1])) # MSBs go first (for Z3)


# Helper function for permute2x128 intrinsics
def _select4_128b(src1: BitVecRef, src2: BitVecRef, control: BitVecRef | BitVecNumRef) -> BitVecRef:
    """
    Selects a 128-bit lane based on 4-bit control according to vperm2i128 semantics.
    
    DEFINE SELECT4(src1, src2, control) {
        CASE(control[1:0]) OF
        0:	tmp[127:0] := src1[127:0]
        1:	tmp[127:0] := src1[255:128]
        2:	tmp[127:0] := src2[127:0]
        3:	tmp[127:0] := src2[255:128]
        ESAC
        IF control[3]
            tmp[127:0] := 0
        FI
        RETURN tmp[127:0]
    }
    """
    # Extract the select bits [1:0] and zero flag [3]
    select_bits = Extract(1, 0, control)
    zero_flag = Extract(3, 3, control)
    
    # Select the appropriate 128-bit lane based on select_bits
    selected_lane = simplify(
        If(
            select_bits == 0,
            Extract(127, 0, src1),      # src1[127:0]
            If(
                select_bits == 1,
                Extract(255, 128, src1), # src1[255:128]
                If(
                    select_bits == 2,
                    Extract(127, 0, src2),   # src2[127:0]
                    Extract(255, 128, src2), # src2[255:128] - select_bits == 3
                ),
            ),
        )
    )
    
    # Apply zero flag if set
    return simplify(
        If(
            zero_flag == 1,
            BitVecVal(0, 128),
            selected_lane
        )
    )


# AVX2: vperm2i128/_mm256_permute2x128_si256
def _mm256_permute2x128_si256(a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int):
    """
    Shuffle 128-bits (composed of integer data) selected by imm8 from a and b, and store the results in dst.
    
    Implements __m256i _mm256_permute2x128_si256 (__m256i a, __m256i b, const int imm8)
    according to the Intel spec.
    
    Operation:
    ```
    DEFINE SELECT4(src1, src2, control) {
        CASE(control[1:0]) OF
        0:	tmp[127:0] := src1[127:0]
        1:	tmp[127:0] := src1[255:128]
        2:	tmp[127:0] := src2[127:0]
        3:	tmp[127:0] := src2[255:128]
        ESAC
        IF control[3]
            tmp[127:0] := 0
        FI
        RETURN tmp[127:0]
    }
    dst[127:0] := SELECT4(a[255:0], b[255:0], imm8[3:0])
    dst[255:128] := SELECT4(a[255:0], b[255:0], imm8[7:4])
    dst[MAX:256] := 0
    ```
    """
    # Support constants or BitVec
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)
    
    # Process each 128-bit lane
    lanes = [None] * 2
    for i in range(2):
        # Extract control bits for this lane: imm8[3+i*4:i*4]
        control_bits = Extract(3 + i * 4, i * 4, imm)
        lanes[i] = _select4_128b(a, b, control_bits)
    
    # Concatenate the lanes (reverse order since Concat puts first arg in MSB)
    return simplify(Concat(lanes[::-1]))


# Helper function for shuffle_i32x4 intrinsics (512-bit)
def _select4_4x32b(src: BitVecRef, control: BitVecRef | BitVecNumRef) -> BitVecRef:
    """
    Selects a 128-bit lane from a 512-bit source based on 2-bit control according to vshufi32x4 semantics.
    
    DEFINE SELECT4(src, control) {
        CASE(control[1:0]) OF
        0:	tmp[127:0] := src[127:0]
        1:	tmp[127:0] := src[255:128]
        2:	tmp[127:0] := src[383:256]
        3:	tmp[127:0] := src[511:384]
        ESAC
        RETURN tmp[127:0]
    }
    """
    # Extract the select bits [1:0]
    select_bits = Extract(1, 0, control)
    
    # Select the appropriate 128-bit lane based on select_bits
    return simplify(
        If(
            select_bits == 0,
            Extract(127, 0, src),       # src[127:0]
            If(
                select_bits == 1,
                Extract(255, 128, src),  # src[255:128]
                If(
                    select_bits == 2,
                    Extract(383, 256, src),  # src[383:256]
                    Extract(511, 384, src),  # src[511:384] - select_bits == 3
                ),
            ),
        )
    )


# AVX512: vshufi32x4/_mm512_shuffle_i32x4
def _mm512_shuffle_i32x4(a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int):
    """
    Shuffle 128-bits (composed of 4 32-bit integers) selected by imm8 from a and b, and store the results in dst.
    
    Implements __m512i _mm512_shuffle_i32x4 (__m512i a, __m512i b, const int imm8)
    according to the Intel spec.
    
    Operation:
    ```
    DEFINE SELECT4(src, control) {
        CASE(control[1:0]) OF
        0:	tmp[127:0] := src[127:0]
        1:	tmp[127:0] := src[255:128]
        2:	tmp[127:0] := src[383:256]
        3:	tmp[127:0] := src[511:384]
        ESAC
        RETURN tmp[127:0]
    }
    dst[127:0] := SELECT4(a[511:0], imm8[1:0])
    dst[255:128] := SELECT4(a[511:0], imm8[3:2])
    dst[383:256] := SELECT4(b[511:0], imm8[5:4])
    dst[511:384] := SELECT4(b[511:0], imm8[7:6])
    dst[MAX:512] := 0
    ```
    """
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)
    
    # Select 128-bit lanes
    lanes = [None] * 4
    
    for j in range(4):
        source = a if j < 2 else b
        ctrl = Extract(2*j + 1, 2*j, imm)
        lanes[j] = _select4_4x32b(source, ctrl)
    
    # Concatenate the lanes (highest lane goes to MSB)
    return simplify(Concat(lanes[::-1]))


# vpsrld
def shr(op1, const):
    src = ymm_regs[op1]
    chunks = [None] * 8

    for j in range(8):
        i = j * 32
        elem = simplify(
            If(
                const > 31,
                BitVecVal(0, 32),
                Extract(256 - j * 32 - 1, 256 - (j + 1) * 32, src),
            )
        )

        elem2 = simplify(
            Concat(
                Extract(7, 0, elem),
                Extract(15, 8, elem),
                Extract(23, 16, elem),
                Extract(31, 24, elem),
            )
        )
        # print (elem2)
        elem3 = simplify(LShR(elem2, const))
        # print (elem3)
        chunks[j] = Concat(
            Extract(7, 0, elem3),
            Extract(15, 8, elem3),
            Extract(23, 16, elem3),
            Extract(31, 24, elem3),
        )

    return simplify(Concat(chunks))


# vpslld
def shl(op1, const):
    src = ymm_regs[op1]
    chunks = [None] * 8

    for j in range(8):
        i = j * 32
        elem = simplify(
            If(
                const > 31,
                BitVecVal(0, 32),
                Extract(256 - j * 32 - 1, 256 - (j + 1) * 32, src),
            )
        )

        elem2 = simplify(
            Concat(
                Extract(7, 0, elem),
                Extract(15, 8, elem),
                Extract(23, 16, elem),
                Extract(31, 24, elem),
            )
        )
        elem3 = simplify(elem2 << const)
        chunks[j] = Concat(
            Extract(7, 0, elem3),
            Extract(15, 8, elem3),
            Extract(23, 16, elem3),
            Extract(31, 24, elem3),
        )

    return simplify(Concat(chunks))


# vpxor
def xor(op1, op2):
    return simplify(ymm_regs[op1] ^ ymm_regs[op2])


# vpand
def _and(op1, op2):
    return simplify(ymm_regs[op1] & ymm_regs[op2])


# vpor
def _or(op1, op2):
    return simplify(ymm_regs[op1] | ymm_regs[op2])


#  vpcmpeqb
def cmp(op1, op2):
    chunksA = [None] * 32
    chunksB = [None] * 32
    chunksC = [None] * 32

    a = ymm_regs[op1]
    b = ymm_regs[op2]

    for j in range(32):
        chunksA[j] = simplify(Extract((j + 1) * 8 - 1, j * 8, a))
        chunksB[j] = simplify(Extract((j + 1) * 8 - 1, j * 8, b))

    for j in range(32):
        chunksC[j] = If(simplify(chunksA[j] == chunksB[j]), BitVecVal(0xFF, 8), BitVecVal(0, 8))
    return simplify(Concat(chunksC))  # [::-1]


def to_dword(v):
    return simplify(Concat(Extract(7, 0, v), Extract(15, 8, v), Extract(23, 16, v), Extract(31, 24, v)))


def from_dword(v):
    return Concat(Extract(7, 0, v), Extract(15, 8, v), Extract(23, 16, v), Extract(31, 24, v))


# vpaddd
def add_dwords(op1, op2):
    src1 = ymm_regs[op1]
    chunksA = [None] * 8
    chunksB = [None] * 8
    chunksA[0] = to_dword(simplify(Extract(1 * 32 - 1, 0 * 32, src1)))
    chunksA[1] = to_dword(simplify(Extract(2 * 32 - 1, 1 * 32, src1)))
    chunksA[2] = to_dword(simplify(Extract(3 * 32 - 1, 2 * 32, src1)))
    chunksA[3] = to_dword(simplify(Extract(4 * 32 - 1, 3 * 32, src1)))
    chunksA[4] = to_dword(simplify(Extract(5 * 32 - 1, 4 * 32, src1)))
    chunksA[5] = to_dword(simplify(Extract(6 * 32 - 1, 5 * 32, src1)))
    chunksA[6] = to_dword(simplify(Extract(7 * 32 - 1, 6 * 32, src1)))
    chunksA[7] = to_dword(simplify(Extract(8 * 32 - 1, 7 * 32, src1)))

    src2 = ymm_regs[op2]
    chunksB[0] = to_dword(simplify(Extract(1 * 32 - 1, 0 * 32, src2)))
    chunksB[1] = to_dword(simplify(Extract(2 * 32 - 1, 1 * 32, src2)))
    chunksB[2] = to_dword(simplify(Extract(3 * 32 - 1, 2 * 32, src2)))
    chunksB[3] = to_dword(simplify(Extract(4 * 32 - 1, 3 * 32, src2)))
    chunksB[4] = to_dword(simplify(Extract(5 * 32 - 1, 4 * 32, src2)))
    chunksB[5] = to_dword(simplify(Extract(6 * 32 - 1, 5 * 32, src2)))
    chunksB[6] = to_dword(simplify(Extract(7 * 32 - 1, 6 * 32, src2)))
    chunksB[7] = to_dword(simplify(Extract(8 * 32 - 1, 7 * 32, src2)))

    result = []
    for i in range(len(chunksA)):
        result.append(simplify(from_dword(chunksA[i] + chunksB[i])))

    return simplify(Concat(result[::-1]))