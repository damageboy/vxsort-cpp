import sys
from typing import Any
from z3.z3 import SeqRef, BitVecNumRef, BitVecRef, BitVec, BitVecVal, Solver, Extract, Concat, If, LShR, ZeroExt, simplify

zero = 0


def ymm_reg(name: str):
    return BitVec(name, 32 * 8)


def zmm_reg(name: str):
    return BitVec(name, 64 * 8)


def reg_with_values(name: str, s: Solver, raw_values, element_bits: int, total_bits: int):
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
        assert 0 <= elem_idx < lanes, f"Element index {elem_idx} out of range for {bits}-bit elements (0-{lanes - 1})"
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


def _MM_SHUFFLE2(x: int, y: int) -> int:
    """
    Mimics the standard _MM_SHUFFLE2 intrinsic macro.
    Returns (x << 1) | y
    """
    return (x << 1) | y


def _MM_SHUFFLE(z: int, y: int, x: int, w: int) -> int:
    """
    Mimics the standard _MM_SHUFFLE intrinsic macro.
    Returns (z<<6) | (y<<4) | (x<<2) | w
    """
    return (z << 6) | (y << 4) | (x << 2) | w


##
# Single vector variable permutes


def _create_if_tree(idx_bits: BitVecRef, elements: list[BitVecRef | SeqRef]):
    """
    Create nested If statements for element selection.
    """

    assert len(elements) > 0, "Can't have 0 elements"
    end_idx = len(elements) - 1

    # Create nested If statements like the original code
    result = elements[end_idx]  # Default case
    for i in range(end_idx - 1, -1, -1):
        result = If(idx_bits == i, elements[i], result)

    return result


##
# 1xInput -> 1xOutput, fully variable index permutes:
# - vpermd:
#   -  _mm256_permutexvar_{epi32,ps}
#   -  _mm512_[mask]permute[x]var_{epi32,ps}
# - vpermq:
#   -  _mm256_permutevar_{epi64,pd}
#   -  _mm512_[mask]permutevar_{epi64,pd}
# NOTE: AVX2/AVX512 is *very weird* in that in the 512b version, the permute[x]var
#       are identical in implementation to each other
#       but in the 256b version, only the permutexvar does variable permutes
#       while the permutevar option exists, but does something else entirely
#       (see other groups in this file to find it)


def _create_element_selector(source_reg: BitVecRef, idx_bits: BitVecRef, num_elements: int, element_bits: int) -> BitVecRef:
    """
    Create a balanced tree of If statements for element selection.

    Args:
        source_reg: The source register to select elements from
        idx_bits: The index bits extracted from the index register
        num_elements: Number of elements to choose from (2, 4, 8, 16)
        element_bits: Number of bits per element (32 or 64)

    Returns:
        A Z3 expression that selects the appropriate element based on idx_bits
    """
    # Extract all elements
    elements: list[BitVecRef | SeqRef] = []
    for i in range(num_elements):
        start_bit = i * element_bits
        end_bit = start_bit + element_bits - 1
        elements.append(Extract(end_bit, start_bit, source_reg))

    # Create balanced tree of If statements
    return _create_if_tree(idx_bits, elements)


# Generic implementation for permutexvar instructions
def _generic_permutexvar(op1: BitVecRef, op_idx: BitVecRef, total_width: int, element_width: int, src: BitVecRef | None = None, mask: BitVecRef | None = None):
    """
    Generic implementation for permutexvar instructions that shuffle elements across lanes.

    These instructions use a variable index vector to permute elements from a single source vector.
    Each element in the output is selected from the source vector based on the corresponding
    index value in the index vector. Optional masking is supported for AVX512 variants.

    Args:
        op1: Source vector to permute
        op_idx: Index vector containing the indices for each destination element
        total_width: Total bit width of the vectors (256 or 512)
        element_width: Width of each element in bits (32 or 64)
        src: Optional source vector for masked operations (values used when mask bit is 0)
        mask: Optional predicate mask (if provided, src must also be provided)

    Returns:
        Permuted vector (optionally masked)

    Generic Operation (where N = total_width / element_width, IDX_BITS = log2(N)):
        Without mask:
        ```
        FOR j := 0 to N-1
            i := j * element_width
            index := op_idx[i + IDX_BITS - 1 : i]
            dst[i + element_width - 1 : i] := op1[index * element_width + element_width - 1 : index * element_width]
        ENDFOR
        dst[MAX:total_width] := 0
        ```

        With mask:
        ```
        FOR j := 0 to N-1
            i := j * element_width
            index := op_idx[i + IDX_BITS - 1 : i]
            IF mask[j]
                dst[i + element_width - 1 : i] := op1[index * element_width + element_width - 1 : index * element_width]
            ELSE
                dst[i + element_width - 1 : i] := src[i + element_width - 1 : i]
            FI
        ENDFOR
        dst[MAX:total_width] := 0
        ```

    Examples:
        - _mm256_permutexvar_epi32: total_width=256, element_width=32 → 8 elements, 3 index bits
        - _mm512_permutexvar_epi32: total_width=512, element_width=32 → 16 elements, 4 index bits
        - _mm256_permutexvar_epi64: total_width=256, element_width=64 → 4 elements, 2 index bits
        - _mm512_permutexvar_epi64: total_width=512, element_width=64 → 8 elements, 3 index bits
        - _mm512_mask_permutexvar_epi32: total_width=512, element_width=32, with src and mask
        - _mm512_mask_permutexvar_epi64: total_width=512, element_width=64, with src and mask
    """
    num_elements = total_width // element_width
    # Calculate number of bits needed to index all elements
    # For 4 elements: 2 bits, 8 elements: 3 bits, 16 elements: 4 bits
    idx_bits_needed = (num_elements - 1).bit_length()

    elems = [None] * num_elements

    for j in range(num_elements):
        i = j * element_width
        # Extract index bits: idx[i+idx_bits_needed-1:i]
        idx_bits = Extract(i + idx_bits_needed - 1, i, op_idx)
        # Use the generic element selector to get the permuted element
        permuted_elem = _create_element_selector(op1, idx_bits, num_elements, element_width)

        # Apply mask if provided
        if mask is not None and src is not None:
            # Extract mask bit for this element
            mask_bit = Extract(j, j, mask)
            # Extract source element for this position
            src_elem = Extract(i + element_width - 1, i, src)
            # If mask bit is set, use permuted element; otherwise use src element
            elems[j] = If(mask_bit == BitVecVal(1, 1), permuted_elem, src_elem)
        else:
            elems[j] = permuted_elem

    return simplify(Concat(elems[::-1]))


# AVX2: vpermd/_mm256_permutevar_epi32
def _mm256_permutexvar_epi32(op1: BitVecRef, op_idx: BitVecRef):
    """
    Shuffle 32-bit integers across lanes in a 256-bit vector.
    Implements __m256i _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx)
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(op1, op_idx, 256, 32)


# AVX512: vpermd/_mm512_permutexvar_epi32
def _mm512_permutexvar_epi32(op1: BitVecRef, op_idx: BitVecRef):
    """
    Shuffle 32-bit integers across lanes in a 512-bit vector.
    Implements __m512i _mm512_permutexvar_epi32 (__m512i idx, __m512i a)
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(op1, op_idx, 512, 32)


# AVX2: vpermq/_mm256_permutexvar_epi64
def _mm256_permutexvar_epi64(op1: BitVecRef, idx: BitVecRef):
    """
    Shuffle 64-bit integers across lanes in a 256-bit vector.
    Implements __m256i _mm256_permutexvar_epi64 (__m256i idx, __m256i a)
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(op1, idx, 256, 64)


# AVX512: vpermq/_mm512_permutexvar_epi64
def _mm512_permutexvar_epi64(op1: BitVecRef, idx: BitVecRef):
    """
    Shuffle 64-bit integers across lanes in a 512-bit vector.
    Implements __m512i _mm512_permutexvar_epi64 (__m512i idx, __m512i a)
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(op1, idx, 512, 64)


# AVX512: vpermd/_mm512_mask_permutexvar_epi32 (masked variant)
def _mm512_mask_permutexvar_epi32(src: BitVecRef, mask: BitVecRef, idx: BitVecRef, op1: BitVecRef):
    """
    Shuffle 32-bit integers across lanes in a 512-bit vector using writemask.
    Implements __m512i _mm512_mask_permutexvar_epi32 (__m512i src, __mmask16 k, __m512i idx, __m512i a)
    Elements are copied from src when the corresponding mask bit is not set.
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(op1, idx, 512, 32, src=src, mask=mask)


# AVX512: vpermq/_mm512_mask_permutexvar_epi64 (masked variant)
def _mm512_mask_permutexvar_epi64(src: BitVecRef, mask: BitVecRef, idx: BitVecRef, op1: BitVecRef):
    """
    Shuffle 64-bit integers across lanes in a 512-bit vector using writemask.
    Implements __m512i _mm512_mask_permutexvar_epi64 (__m512i src, __mmask8 k, __m512i idx, __m512i a)
    Elements are copied from src when the corresponding mask bit is not set.
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(op1, idx, 512, 64, src=src, mask=mask)


##
# 2xInput -> 1xOutput, fully variable index permutes:
# * vpermi2d,vpermt2d:
#   -  _mm512_permutex2var_{epi32,epi64}
#   -  _mm512_[mask]permutex2var_{epi32,epi64}


def _create_two_source_element_selector(a: BitVecRef, b: BitVecRef, offset_bits: BitVecRef, source_selector: BitVecRef, num_elements: int, element_bits: int) -> BitVecRef:
    """
    Create element selector for two-source permutation (permutex2var).

    Args:
        source_a: First source register
        source_b: Second source register
        offset_bits: Bits specifying which element to select from the chosen source
        source_selector: Bit specifying which source to choose from (0=a, 1=b)
        num_elements: Number of elements in each source register
        element_bits: Number of bits per element

    Returns:
        A Z3 expression that selects the appropriate element
    """
    # First select the source vector based on source_selector
    selected_source = If(source_selector == 0, a, b)

    # Then select element from the chosen source based on offset
    return _create_element_selector(selected_source, offset_bits, num_elements, element_bits)


# Generic implementation for permutex2var instructions
def _generic_permutex2var(a: BitVecRef, idx: BitVecRef, b: BitVecRef, element_width: int, src: BitVecRef | None = None, mask: BitVecRef | None = None):
    """
    Generic implementation for permutex2var instructions that shuffle elements from two source vectors.

    These instructions use an index vector where each element contains:
    - Offset bits: select which element from the chosen source
    - Source selector bit: choose between source a (0) or source b (1)
    - Optional: masking is supported for AVX512 variants.

    Args:
        a: First source vector
        idx: Index vector containing offsets and source selectors for each destination element
        b: Second source vector
        element_width: Width of each element in bits (32 or 64)
        src: Optional source vector for masked operations (when mask bit is 0, copy from this)
        mask: Optional predicate mask (if provided, src must also be provided)

    Returns:
        Permuted vector (optionally masked)

    Generic Operation (for 512-bit registers, N elements, OFFSET_BITS bits, SRC_BIT position):
        Without mask:
        ```
        FOR j := 0 to N-1
            i := j * element_width
            offset := idx[i + OFFSET_BITS - 1 : i]
            source_sel := idx[i + SRC_BIT]
            selected_vec := source_sel ? b : a
            dst[i + element_width - 1 : i] := selected_vec[offset * element_width + element_width - 1 : offset * element_width]
        ENDFOR
        dst[MAX:512] := 0
        ```

        With mask:
        ```
        FOR j := 0 to N-1
            i := j * element_width
            offset := idx[i + OFFSET_BITS - 1 : i]
            source_sel := idx[i + SRC_BIT]
            IF mask[j]
                selected_vec := source_sel ? b : a
                dst[i + element_width - 1 : i] := selected_vec[offset * element_width + element_width - 1 : offset * element_width]
            ELSE
                dst[i + element_width - 1 : i] := src[i + element_width - 1 : i]
            FI
        ENDFOR
        dst[MAX:512] := 0
        ```

    Examples:
        - _mm512_permutex2var_epi32: element_width=32 → 16 elements, 4 offset bits, bit 4 is source selector
        - _mm512_permutex2var_epi64: element_width=64 → 8 elements, 3 offset bits, bit 3 is source selector
        - _mm512_mask_permutex2var_ps: element_width=32 -> 16 elements, 4 offset bits, bit 4 is source selector, with src and mask
        - _mm512_mask_permutex2var_pd: element_width=64 -> 8 elements, 3 offset bits, bit 3 is source selector, with src and mask
    """
    # All permutex2var instructions are 512-bit
    total_width = 512
    num_elements = total_width // element_width

    # Calculate bit positions
    # For 32-bit elements: offset is bits [3:0], source selector is bit 4
    # For 64-bit elements: offset is bits [2:0], source selector is bit 3
    offset_bits_count = (num_elements - 1).bit_length()
    source_selector_bit = offset_bits_count

    elems = [None] * num_elements

    for j in range(num_elements):
        i = j * element_width

        # Extract offset bits: idx[i+offset_bits_count-1:i]
        offset_bits = Extract(i + offset_bits_count - 1, i, idx)

        # Extract source selector: idx[i+source_selector_bit]
        source_selector = Extract(i + source_selector_bit, i + source_selector_bit, idx)

        # Get the permuted element using the two-source selector
        permuted_elem = _create_two_source_element_selector(a, b, offset_bits, source_selector, num_elements, element_width)

        # Apply mask if provided
        if mask is not None and src is not None:
            # Extract mask bit for this element
            mask_bit = Extract(j, j, mask)
            # Extract source element for this position
            src_elem = Extract(i + element_width - 1, i, src)
            # If mask bit is set, use permuted element; otherwise use src element
            elems[j] = If(mask_bit == BitVecVal(1, 1), permuted_elem, src_elem)
        else:
            elems[j] = permuted_elem

    return simplify(Concat(elems[::-1]))


# AVX512: vpermi2d/vpermt2d/_mm512_permutex2var_epi32
def _mm512_permutex2var_epi32(a: BitVecRef, idx: BitVecRef, b: BitVecRef):
    """
    Shuffle 32-bit integers in a and b across lanes using two-source permutation.
    Implements __m512i _mm512_permutex2var_epi32 (__m512i a, __m512i idx, __m512i b)
    See _generic_permutex2var for operation details.
    """
    return _generic_permutex2var(a, idx, b, 32)


# AVX512: vpermi2q/vpermt2q/_mm512_permutex2var_epi64
def _mm512_permutex2var_epi64(a: BitVecRef, idx: BitVecRef, b: BitVecRef):
    """
    Shuffle 64-bit integers in a and b across lanes using two-source permutation.
    Implements __m512i _mm512_permutex2var_epi64 (__m512i a, __m512i idx, __m512i b)
    See _generic_permutex2var for operation details.
    """
    return _generic_permutex2var(a, idx, b, 64)


# AVX512: vpermi2d/vpermt2d/_mm512_mask_permutex2var_epi32 (masked version)
def _mm512_mask_permutex2var_epi32(a: BitVecRef, k: BitVecRef, idx: BitVecRef, b: BitVecRef):
    """
    Shuffle 32-bit integer elements in a and b across lanes using writemask.
    Implements __m512i _mm512_mask_permutex2var_epi32 (__m512i a, __mmask16 k, __m512i idx, __m512i b)
    Elements are copied from a when the corresponding mask bit is not set.
    See _generic_permutex2var for operation details.
    """
    return _generic_permutex2var(a, idx, b, 32, src=a, mask=k)


# AVX512: vpermi2q/vpermt2q/_mm512_mask_permutex2var_epi64 (masked version for 64-bit)
def _mm512_mask_permutex2var_epi64(a: BitVecRef, k: BitVecRef, idx: BitVecRef, b: BitVecRef):
    """
    Shuffle 64-bit integer elements in a and b across lanes using writemask.
    Implements __m512i _mm512_mask_permutex2var_epi64 (__m512i a, __mmask8 k, __m512i idx, __m512i b)
    Elements are copied from a when the corresponding mask bit is not set.
    See _generic_permutex2var for operation details.
    """
    return _generic_permutex2var(a, idx, b, 64, src=a, mask=k)


##
# Helpers function for permutes/shuffles
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


##
# 1xInput->1xOutput, within 128b lane static(imm) permutes
# - vpermilps,vpermilpd:
#   -  _mm256_permute_p{s,d}
#   -  _mm512_[mask_]permute_p{s,d}
def vpermilps_lane(lane_idx: int, a: BitVecRef, ctrl01: BitVecRef, ctrl23: BitVecRef, ctrl45: BitVecRef, ctrl67: BitVecRef):
    src_lane = extract_128b_lane(a, lane_idx)

    chunks: list[BitVecRef | None] = [None] * 4
    chunks[0] = _select4_ps(src_lane, ctrl01)
    chunks[1] = _select4_ps(src_lane, ctrl23)
    chunks[2] = _select4_ps(src_lane, ctrl45)
    chunks[3] = _select4_ps(src_lane, ctrl67)
    return chunks


def vpermilpd_lane(lane_idx: int, a: BitVecRef, ctrl0: BitVecRef, ctrl1: BitVecRef):
    src_lane = extract_128b_lane(a, lane_idx)

    chunks: list[BitVecRef | None] = [None] * 2
    chunks[0] = _select2_pd(src_lane, ctrl0)
    chunks[1] = _select2_pd(src_lane, ctrl1)
    return chunks


# Generic permute_ps function
def _permute_ps_generic(op1: BitVecRef, imm8: BitVecRef | int, num_lanes: int, k: BitVecRef | None = None, src: BitVecRef | None = None):
    """
    Generic permute_ps implementation for any number of 128-bit lanes.
    Permutes 32-bit elements within each 128-bit lane using control bits in imm8.

    If k (mask) and src are provided, applies masking: elements are copied from src when mask bit is not set.

    Operation:
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
    FOR lane := 0 to num_lanes-1
        dst[lane*128+31:lane*128] := SELECT4(a[lane*128+127:lane*128], imm8[1:0])
        dst[lane*128+63:lane*128+32] := SELECT4(a[lane*128+127:lane*128], imm8[3:2])
        dst[lane*128+95:lane*128+64] := SELECT4(a[lane*128+127:lane*128], imm8[5:4])
        dst[lane*128+127:lane*128+96] := SELECT4(a[lane*128+127:lane*128], imm8[7:6])
    ENDFOR
    ```
    """
    a = op1
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)
    ctrl01, ctrl23, ctrl45, ctrl67 = _extract_ctl4(imm)
    chunks_128b = [vpermilps_lane(lane_idx, a, ctrl01, ctrl23, ctrl45, ctrl67) for lane_idx in range(num_lanes)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    result = simplify(Concat(flat_chunks[::-1]))

    # Apply mask if provided
    if k is not None and src is not None:
        num_elements = num_lanes * 4  # 4 elements per 128-bit lane
        elements = [None] * num_elements
        for j in range(num_elements):
            i = j * 32
            mask_bit = Extract(j, j, k)
            tmp_elem = Extract(i + 31, i, result)
            src_elem = Extract(i + 31, i, src)
            elements[j] = simplify(If(mask_bit == 1, tmp_elem, src_elem))
        result = simplify(Concat(elements[::-1]))

    return result


# AVX2: vpermilps (_mm256_permute_ps)
def _mm256_permute_ps(op1: BitVecRef, imm8: BitVecRef | int):
    """Permutes 32-bit elements within 128-bit lanes. Operates on YMM registers (2 lanes)."""
    return _permute_ps_generic(op1, imm8, 2)


# AVX512: vpermilps (_mm512_permute_ps)
def _mm512_permute_ps(op1: BitVecRef, imm8: BitVecRef | int):
    """Permutes 32-bit elements within 128-bit lanes. Operates on ZMM registers (4 lanes)."""
    return _permute_ps_generic(op1, imm8, 4)


# AVX512: vpermilps (_mm512_mask_permute_ps)
def _mm512_mask_permute_ps(src: BitVecRef, k: BitVecRef, a: BitVecRef, imm8: BitVecRef | int):
    """
    Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in imm8,
    and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
    Implements __m512 _mm512_mask_permute_ps (__m512 src, __mmask16 k, __m512 a, const int imm8)
    """
    return _permute_ps_generic(a, imm8, 4, k=k, src=src)


# Generic permute_pd function
def _permute_pd_generic(op1: BitVecRef, imm8: BitVecRef | int, num_lanes: int, k: BitVecRef | None = None, src: BitVecRef | None = None):
    """
    Generic permute_pd implementation for any number of 128-bit lanes.
    Permutes 64-bit elements within each 128-bit lane using control bits in imm8.

    If k (mask) and src are provided, applies masking: elements are copied from src when mask bit is not set.

    Operation:
    ```
    DEFINE SELECT2(src, control) {
        CASE(control[0]) OF
        0:	tmp[63:0] := src[63:0]
        1:	tmp[63:0] := src[127:64]
        ESAC
        RETURN tmp[63:0]
    }
    FOR lane := 0 to num_lanes-1
        dst[lane*128+63:lane*128] := SELECT2(a[lane*128+127:lane*128], imm8[0])
        dst[lane*128+127:lane*128+64] := SELECT2(a[lane*128+127:lane*128], imm8[1])
    ENDFOR
    ```
    """
    a = op1
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)
    ctrl0, ctrl1 = _extract_ctl2(imm)
    chunks_128b = [vpermilpd_lane(lane_idx, a, ctrl0, ctrl1) for lane_idx in range(num_lanes)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    result = simplify(Concat(flat_chunks[::-1]))

    # Apply mask if provided
    if k is not None and src is not None:
        num_elements = num_lanes * 2  # 2 elements per 128-bit lane
        elements = [None] * num_elements
        for j in range(num_elements):
            i = j * 64
            mask_bit = Extract(j, j, k)
            tmp_elem = Extract(i + 63, i, result)
            src_elem = Extract(i + 63, i, src)
            elements[j] = simplify(If(mask_bit == 1, tmp_elem, src_elem))
        result = simplify(Concat(elements[::-1]))

    return result


# AVX2: vpermilpd (_mm256_permute_pd)
def _mm256_permute_pd(op1: BitVecRef, imm8: BitVecRef | int):
    """Permutes 64-bit elements within 128-bit lanes. Operates on YMM registers (2 lanes)."""
    return _permute_pd_generic(op1, imm8, 2)


# AVX512: vpermilpd (_mm512_permute_pd)
def _mm512_permute_pd(op1: BitVecRef, imm8: BitVecRef | int):
    """Permutes 64-bit elements within 128-bit lanes. Operates on ZMM registers (4 lanes)."""
    return _permute_pd_generic(op1, imm8, 4)


# AVX512: vpermilpd (_mm512_mask_permute_pd)
def _mm512_mask_permute_pd(src: BitVecRef, k: BitVecRef, a: BitVecRef, imm8: BitVecRef | int):
    """
    Shuffle double-precision (64-bit) floating-point elements in a within 128-bit lanes using the control in imm8,
    and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
    Implements __m512d _mm512_mask_permute_pd (__m512d src, __mmask8 k, __m512d a, const int imm8)
    """
    return _permute_pd_generic(a, imm8, 4, k=k, src=src)


##
# 2xInput->1xOutput, within 128b lane static(imm) permutes
# - vshufps,vshufpd:
#   -  _mm256_shuffle_p{s,d}
#   -  _mm512_[mask_]shuffle_p{s,d}


def vshufps_lane(lane_idx: int, a: BitVecRef, b: BitVecRef, ctrl01: BitVecRef, ctrl23: BitVecRef, ctrl45: BitVecRef, ctrl67: BitVecRef) -> None:
    a_lane = extract_128b_lane(a, lane_idx)
    b_lane = extract_128b_lane(b, lane_idx)

    chunks: list[BitVecRef] = [None] * 4
    chunks[0] = _select4_ps(a_lane, ctrl01)
    chunks[1] = _select4_ps(a_lane, ctrl23)
    chunks[2] = _select4_ps(b_lane, ctrl45)
    chunks[3] = _select4_ps(b_lane, ctrl67)
    return chunks


# Generic shuffle_ps function
def _shuffle_ps_generic(op1: BitVecRef, op2: BitVecRef, imm8: BitVecRef | int, num_lanes: int, k: BitVecRef | None = None, src: BitVecRef | None = None):
    """
    Generic shuffle_ps implementation for any number of 128-bit lanes.
    Shuffles 32-bit elements within 128-bit lanes using control in imm8.

    If k (mask) and src are provided, applies masking: elements are copied from src when mask bit is not set.

    Operation:
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
    FOR lane := 0 to num_lanes-1
        dst[lane*128+31:lane*128] := SELECT4(a[lane*128+127:lane*128], imm8[1:0])
        dst[lane*128+63:lane*128+32] := SELECT4(a[lane*128+127:lane*128], imm8[3:2])
        dst[lane*128+95:lane*128+64] := SELECT4(b[lane*128+127:lane*128], imm8[5:4])
        dst[lane*128+127:lane*128+96] := SELECT4(b[lane*128+127:lane*128], imm8[7:6])
    ENDFOR
    ```
    """
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)
    ctrl01, ctrl23, ctrl45, ctrl67 = _extract_ctl4(imm)
    chunks_128b = [vshufps_lane(lane_idx, op1, op2, ctrl01, ctrl23, ctrl45, ctrl67) for lane_idx in range(num_lanes)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    result = simplify(Concat(flat_chunks[::-1]))

    # Apply mask if provided
    if k is not None and src is not None:
        num_elements = num_lanes * 4  # 4 elements per 128-bit lane
        elements = [None] * num_elements
        for j in range(num_elements):
            i = j * 32
            mask_bit = Extract(j, j, k)
            tmp_elem = Extract(i + 31, i, result)
            src_elem = Extract(i + 31, i, src)
            elements[j] = simplify(If(mask_bit == 1, tmp_elem, src_elem))
        result = simplify(Concat(elements[::-1]))

    return result


# AVX2: vshufps (_mm256_shuffle_ps)
def _mm256_shuffle_ps(op1: BitVecRef, op2: BitVecRef, imm8: BitVecRef | int):
    """Shuffles 32-bit elements within 128-bit lanes. Operates on YMM registers (2 lanes)."""
    return _shuffle_ps_generic(op1, op2, imm8, 2)


# AVX512: vshufps (_mm512_shuffle_ps)
def _mm512_shuffle_ps(op1: BitVecRef, op2: BitVecRef, imm8: BitVecRef | int):
    """Shuffles 32-bit elements within 128-bit lanes. Operates on ZMM registers (4 lanes)."""
    return _shuffle_ps_generic(op1, op2, imm8, 4)


# AVX512: vshufps (_mm512_mask_shuffle_ps)
def _mm512_mask_shuffle_ps(src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int):
    """
    Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in imm8,
    and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
    Implements __m512 _mm512_mask_shuffle_ps (__m512 src, __mmask16 k, __m512 a, __m512 b, const int imm8)
    """
    return _shuffle_ps_generic(a, b, imm8, 4, k=k, src=src)


def vshufpd_lane(lane_idx: int, a: BitVecRef, b: BitVecRef, imm: BitVecRef):
    a_lane = extract_128b_lane(a, lane_idx)
    b_lane = extract_128b_lane(b, lane_idx)

    # Each lane uses 2 control bits: lane i uses imm[2*i] and imm[2*i+1]
    ctrl0 = Extract(2 * lane_idx, 2 * lane_idx, imm)  # Controls selection from a
    ctrl1 = Extract(2 * lane_idx + 1, 2 * lane_idx + 1, imm)  # Controls selection from b

    chunks: list[BitVecRef | None] = [None] * 2
    chunks[0] = _select2_pd(a_lane, ctrl0)
    chunks[1] = _select2_pd(b_lane, ctrl1)
    return chunks


# Generic shuffle_pd function
def _shuffle_pd_generic(op1: BitVecRef, op2: BitVecRef, imm8: BitVecRef | int, num_lanes: int, k: BitVecRef | None = None, src: BitVecRef | None = None):
    """
    Generic shuffle_pd implementation for any number of 128-bit lanes.
    Shuffles 64-bit elements within 128-bit lanes using control in imm8.

    If k (mask) and src are provided, applies masking: elements are copied from src when mask bit is not set.

    Operation:
    ```
    FOR lane := 0 to num_lanes-1
        dst[lane*128+63:lane*128] := (imm8[2*lane] == 0) ? a[lane*128+63:lane*128] : a[lane*128+127:lane*128+64]
        dst[lane*128+127:lane*128+64] := (imm8[2*lane+1] == 0) ? b[lane*128+63:lane*128] : b[lane*128+127:lane*128+64]
    ENDFOR
    ```
    """
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)
    chunks_128b = [vshufpd_lane(lane_idx, op1, op2, imm) for lane_idx in range(num_lanes)]
    flat_chunks = [e for sublist in chunks_128b for e in sublist]
    result = simplify(Concat(flat_chunks[::-1]))

    # Apply mask if provided
    if k is not None and src is not None:
        num_elements = num_lanes * 2  # 2 elements per 128-bit lane
        elements = [None] * num_elements
        for j in range(num_elements):
            i = j * 64
            mask_bit = Extract(j, j, k)
            tmp_elem = Extract(i + 63, i, result)
            src_elem = Extract(i + 63, i, src)
            elements[j] = simplify(If(mask_bit == 1, tmp_elem, src_elem))
        result = simplify(Concat(elements[::-1]))

    return result


# AVX2: vshufpd (_mm256_shuffle_pd)
def _mm256_shuffle_pd(op1: BitVecRef, op2: BitVecRef, imm8: BitVecRef | int):
    """Shuffles 64-bit elements within 128-bit lanes. Operates on YMM registers (2 lanes)."""
    return _shuffle_pd_generic(op1, op2, imm8, 2)


# AVX512: vshufpd (_mm512_shuffle_pd)
def _mm512_shuffle_pd(op1: BitVecRef, op2: BitVecRef, imm8: BitVecRef | int):
    """Shuffles 64-bit elements within 128-bit lanes. Operates on ZMM registers (4 lanes)."""
    return _shuffle_pd_generic(op1, op2, imm8, 4)


# AVX512: vshufpd (_mm512_mask_shuffle_pd)
def _mm512_mask_shuffle_pd(src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int):
    """
    Shuffle double-precision (64-bit) floating-point elements within 128-bit lanes using the control in imm8,
    and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
    Implements __m512d _mm512_mask_shuffle_pd (__m512d src, __mmask8 k, __m512d a, __m512d b, const int imm8)
    """
    return _shuffle_pd_generic(a, b, imm8, 4, k=k, src=src)


##
# 2xInput->1xOutput, within 128b lane variable index permutes
# - vpermilps/vpermilpd:
#   -  _mm256_permutevar_p{s,d}
#   -  _mm512_[mask_]permutevar_p{s,d}


# Generic implementation for permutevar instructions
def _generic_permutevar(a: BitVecRef, b: BitVecRef, total_width: int, element_width: int, k: BitVecRef | None = None, src: BitVecRef | None = None):
    """
    Generic implementation for permutevar instructions that shuffle elements within 128-bit lanes.

    These instructions use a variable index vector to permute elements within each 128-bit lane.
    Each element in the output is selected from the corresponding 128-bit lane based on control bits
    in the index vector. Optional masking is supported for AVX512 variants.

    Args:
        a: Source vector to permute
        b: Control/index vector containing the control bits for each destination element
        total_width: Total bit width of the vectors (256 or 512)
        element_width: Width of each element in bits (32 for ps, 64 for pd)
        k: Optional predicate mask (if provided, src must also be provided)
        src: Optional source vector for masked operations (values used when mask bit is 0)

    Returns:
        Permuted vector (optionally masked)

    Generic Operation (where N = total_width / element_width, LANE_ELEMENTS = 128 / element_width):

        For element_width=32 (ps - single precision):
            - 4 elements per 128-bit lane
            - Uses 2 control bits per element: b[i+1:i] where i = element_index * 32

        For element_width=64 (pd - double precision):
            - 2 elements per 128-bit lane
            - Uses 1 control bit per element at specific positions:
              b[1], b[65], b[129], b[193] for 256-bit (4 elements)
              b[1], b[65], b[129], b[193], b[257], b[321], b[385], b[449] for 512-bit (8 elements)

        Without mask:
        ```
        FOR j := 0 to N-1
            lane_idx := j / LANE_ELEMENTS
            lane := a[lane_idx*128+127 : lane_idx*128]
            control_bits := extract_control_bits(b, j, element_width)
            dst[j*element_width+element_width-1 : j*element_width] := SELECT(lane, control_bits)
        ENDFOR
        dst[MAX:total_width] := 0
        ```

        With mask:
        ```
        FOR j := 0 to N-1
            lane_idx := j / LANE_ELEMENTS
            lane := a[lane_idx*128+127 : lane_idx*128]
            control_bits := extract_control_bits(b, j, element_width)
            tmp_elem := SELECT(lane, control_bits)
            IF k[j]
                dst[j*element_width+element_width-1 : j*element_width] := tmp_elem
            ELSE
                dst[j*element_width+element_width-1 : j*element_width] := src[j*element_width+element_width-1 : j*element_width]
            FI
        ENDFOR
        dst[MAX:total_width] := 0
        ```

    Examples:
        - _mm256_permutevar_ps: total_width=256, element_width=32 → 8 elements, 2 lanes
        - _mm512_permutevar_ps: total_width=512, element_width=32 → 16 elements, 4 lanes
        - _mm256_permutevar_pd: total_width=256, element_width=64 → 4 elements, 2 lanes
        - _mm512_permutevar_pd: total_width=512, element_width=64 → 8 elements, 4 lanes
        - _mm512_mask_permutevar_ps: total_width=512, element_width=32, with src and mask
        - _mm512_mask_permutevar_pd: total_width=512, element_width=64, with src and mask
    """
    num_elements = total_width // element_width
    elements_per_lane = 128 // element_width

    elements = [None] * num_elements

    for j in range(num_elements):
        i = j * element_width
        lane_idx = j // elements_per_lane
        lane_start = lane_idx * 128

        # Extract the 128-bit lane from a
        lane = Extract(lane_start + 127, lane_start, a)

        # Extract control bits and select element based on element width
        if element_width == 32:  # ps (single-precision)
            # Extract 2 control bits at position [i+1:i]
            ctrl_bits = Extract(i + 1, i, b)
            selected = _select4_ps(lane, ctrl_bits)
        elif element_width == 64:  # pd (double-precision)
            # Control bit positions depend on element index
            # Pattern: bit 1, 65, 129, 193, 257, 321, 385, 449 for successive elements
            ctrl_bit_pos = i + 1
            ctrl_bit = Extract(ctrl_bit_pos, ctrl_bit_pos, b)
            selected = _select2_pd(lane, ctrl_bit)
        else:
            raise ValueError(f"Unsupported element_width: {element_width}")

        # Apply mask if provided
        if k is not None and src is not None:
            src_elem = Extract(i + element_width - 1, i, src)
            mask_bit = Extract(j, j, k)
            elements[j] = simplify(If(mask_bit == 1, selected, src_elem))
        else:
            elements[j] = selected

    return simplify(Concat(elements[::-1]))


# AVX2: vpermilps (_mm256_permutevar_ps)
def _mm256_permutevar_ps(a: BitVecRef, b: BitVecRef):
    """
    Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in b.
    Implements __m256 _mm256_permutevar_ps (__m256 a, __m256i b)
    """
    return _generic_permutevar(a, b, total_width=256, element_width=32)


# AVX512: vpermilps (_mm512_permutevar_ps)
def _mm512_permutevar_ps(a: BitVecRef, b: BitVecRef):
    """
    Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in b.
    Implements __m512 _mm512_permutevar_ps (__m512 a, __m512i b)
    """
    return _generic_permutevar(a, b, total_width=512, element_width=32)


# AVX512: vpermilps (_mm512_mask_permutevar_ps)
def _mm512_mask_permutevar_ps(src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef):
    """
    Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in b,
    and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
    Implements __m512 _mm512_mask_permutevar_ps (__m512 src, __mmask16 k, __m512 a, __m512i b)
    """
    return _generic_permutevar(a, b, total_width=512, element_width=32, k=k, src=src)


# AVX2: vpermilpd (_mm256_permutevar_pd)
def _mm256_permutevar_pd(a: BitVecRef, b: BitVecRef):
    """
    Shuffle double-precision (64-bit) floating-point elements in a within 128-bit lanes using the control in b.
    Implements __m256d _mm256_permutevar_pd (__m256d a, __m256i b)
    """
    return _generic_permutevar(a, b, total_width=256, element_width=64)


# AVX512: vpermilpd (_mm512_permutevar_pd)
def _mm512_permutevar_pd(a: BitVecRef, b: BitVecRef):
    """
    Shuffle double-precision (64-bit) floating-point elements in a within 128-bit lanes using the control in b.
    Implements __m512d _mm512_permutevar_pd (__m512d a, __m512i b)
    """
    return _generic_permutevar(a, b, total_width=512, element_width=64)


# AVX512: vpermilpd (_mm512_mask_permutevar_pd)
def _mm512_mask_permutevar_pd(src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef):
    """
    Shuffle double-precision (64-bit) floating-point elements in a within 128-bit lanes using the control in b,
    and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
    Implements __m512d _mm512_mask_permutevar_pd (__m512d src, __mmask8 k, __m512d a, __m512i b)
    """
    return _generic_permutevar(a, b, total_width=512, element_width=64, k=k, src=src)


##
# 2xInput -> 1xOutput, whole 128b lane static(imm) permutes
# - vperm2i128:
#   -  _mm256_permute2x128_si256
#   -  _mm512_[mask_]shuffle_i32x4
# Note that while both the AVX2 and AVX512 versions *generally* shuffle whole 128b lanes,
# The AVX2 version has a more complex semantics for the control bits.
# The same functionality also exists in the AVX512 version, but it is "split"
# into two separate functions: _mm512_shuffle_i32x4 and _mm512_mask_shuffle_i32x4


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
            Extract(127, 0, src1),  # src1[127:0]
            If(
                select_bits == 1,
                Extract(255, 128, src1),  # src1[255:128]
                If(
                    select_bits == 2,
                    Extract(127, 0, src2),  # src2[127:0]
                    Extract(255, 128, src2),  # src2[255:128] - select_bits == 3
                ),
            ),
        )
    )

    # Apply zero flag if set
    return simplify(If(zero_flag == 1, BitVecVal(0, 128), selected_lane))


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
            Extract(127, 0, src),  # src[127:0]
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
        ctrl = Extract(2 * j + 1, 2 * j, imm)
        lanes[j] = _select4_4x32b(source, ctrl)

    # Concatenate the lanes (highest lane goes to MSB)
    return simplify(Concat(lanes[::-1]))


##
# 2xInput -> 1xOutput, blend hi/lo half of each 128b lane
# - vpunpckldq:
#   -  _mm256_unpacklo_epi32
#   -  _mm512_[mask_]unpacklo_epi32
# - vpunpckhdq:
#   -  _mm256_unpackhi_epi32
#   -  _mm512_[mask_]unpackhi_epi32


def _unpack_epi32_generic(a: BitVecRef, b: BitVecRef, high: bool, total_bits: int, src: BitVecRef = None, k: BitVecRef = None):
    """
    Generic unpack implementation for 32-bit integers with optional masking.

    Args:
        a: First source register
        b: Second source register
        high: True for unpackhi (elements 2,3), False for unpacklo (elements 0,1)
        total_bits: Register size (256 or 512)
        src: Source register for masked operations (None for unmasked)
        k: Write mask (None for unmasked operations)

    Returns:
        BitVecRef representing the unpacked result
    """
    assert total_bits in [256, 512], "total_bits must be 256 or 512"

    num_lanes = total_bits // 128  # Number of 128-bit lanes
    num_elements = total_bits // 32  # Total number of 32-bit elements

    elements = [None] * num_elements

    # Process each 128-bit lane
    for lane in range(num_lanes):
        lane_start = lane * 128

        if high:
            # Extract high half elements (2 and 3) from each lane
            a_elem0 = Extract(lane_start + 95, lane_start + 64, a)  # a[lane][2]
            a_elem1 = Extract(lane_start + 127, lane_start + 96, a)  # a[lane][3]
            b_elem0 = Extract(lane_start + 95, lane_start + 64, b)  # b[lane][2]
            b_elem1 = Extract(lane_start + 127, lane_start + 96, b)  # b[lane][3]
        else:
            # Extract low half elements (0 and 1) from each lane
            a_elem0 = Extract(lane_start + 31, lane_start + 0, a)  # a[lane][0]
            a_elem1 = Extract(lane_start + 63, lane_start + 32, a)  # a[lane][1]
            b_elem0 = Extract(lane_start + 31, lane_start + 0, b)  # b[lane][0]
            b_elem1 = Extract(lane_start + 63, lane_start + 32, b)  # b[lane][1]

        # Interleave: a[elem0], b[elem0], a[elem1], b[elem1]
        base_idx = lane * 4
        elements[base_idx + 0] = a_elem0
        elements[base_idx + 1] = b_elem0
        elements[base_idx + 2] = a_elem1
        elements[base_idx + 3] = b_elem1

    # If masking is requested, apply the mask
    if src is not None and k is not None:
        masked_elements = [None] * num_elements
        for j in range(num_elements):
            i = j * 32

            # Extract mask bit for this element
            mask_bit = Extract(j, j, k)

            # Extract elements from both unpacked result and src
            unpack_elem = elements[j]
            src_elem = Extract(i + 31, i, src)

            # Apply mask: if mask bit is set, use unpacked result, otherwise use src
            masked_elements[j] = simplify(If(mask_bit == 1, unpack_elem, src_elem))
        elements = masked_elements

    return simplify(Concat(elements[::-1]))


def _mm256_unpacklo_epi32(a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 32-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst".
    Implements __m256i _mm256_unpacklo_epi32(__m256i a, __m256i b)

    Operation:
    ```
    DEFINE INTERLEAVE_DWORDS(src1[127:0], src2[127:0]) {
        dst[31:0] := src1[31:0]
        dst[63:32] := src2[31:0]
        dst[95:64] := src1[63:32]
        dst[127:96] := src2[63:32]
        RETURN dst[127:0]
    }
    dst[127:0] := INTERLEAVE_DWORDS(a[127:0], b[127:0])
    dst[255:128] := INTERLEAVE_DWORDS(a[255:128], b[255:128])
    dst[MAX:256] := 0
    ```
    """
    return _unpack_epi32_generic(a, b, high=False, total_bits=256)


def _mm256_unpackhi_epi32(a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 32-bit integers from the high half of each 128-bit lane in "a" and "b", and store the results in "dst".
    Implements __m256i _mm256_unpackhi_epi32(__m256i a, __m256i b)

    Operation:
    ```
    DEFINE INTERLEAVE_HIGH_DWORDS(src1[127:0], src2[127:0]) {
        dst[31:0] := src1[95:64]
        dst[63:32] := src2[95:64]
        dst[95:64] := src1[127:96]
        dst[127:96] := src2[127:96]
        RETURN dst[127:0]
    }
    dst[127:0] := INTERLEAVE_HIGH_DWORDS(a[127:0], b[127:0])
    dst[255:128] := INTERLEAVE_HIGH_DWORDS(a[255:128], b[255:128])
    dst[MAX:256] := 0
    ```
    """
    return _unpack_epi32_generic(a, b, high=True, total_bits=256)


def _mm512_unpacklo_epi32(a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 32-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst".
    Implements __m512i _mm512_unpacklo_epi32(__m512i a, __m512i b)

    Operation:
    ```
    DEFINE INTERLEAVE_DWORDS(src1[127:0], src2[127:0]) {
        dst[31:0] := src1[31:0]
        dst[63:32] := src2[31:0]
        dst[95:64] := src1[63:32]
        dst[127:96] := src2[63:32]
        RETURN dst[127:0]
    }
    dst[127:0] := INTERLEAVE_DWORDS(a[127:0], b[127:0])
    dst[255:128] := INTERLEAVE_DWORDS(a[255:128], b[255:128])
    dst[383:256] := INTERLEAVE_DWORDS(a[383:256], b[383:256])
    dst[511:384] := INTERLEAVE_DWORDS(a[511:384], b[511:384])
    dst[MAX:512] := 0
    ```
    """
    return _unpack_epi32_generic(a, b, high=False, total_bits=512)


def _mm512_unpackhi_epi32(a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 32-bit integers from the high half of each 128-bit lane in "a" and "b", and store the results in "dst".
    Implements __m512i _mm512_unpackhi_epi32(__m512i a, __m512i b)

    Operation:
    ```
    DEFINE INTERLEAVE_HIGH_DWORDS(src1[127:0], src2[127:0]) {
        dst[31:0] := src1[95:64]
        dst[63:32] := src2[95:64]
        dst[95:64] := src1[127:96]
        dst[127:96] := src2[127:96]
        RETURN dst[127:0]
    }
    dst[127:0] := INTERLEAVE_HIGH_DWORDS(a[127:0], b[127:0])
    dst[255:128] := INTERLEAVE_HIGH_DWORDS(a[255:128], b[255:128])
    dst[383:256] := INTERLEAVE_HIGH_DWORDS(a[383:256], b[383:256])
    dst[511:384] := INTERLEAVE_HIGH_DWORDS(a[511:384], b[511:384])
    dst[MAX:512] := 0
    ```
    """
    return _unpack_epi32_generic(a, b, high=True, total_bits=512)


def _mm512_mask_unpacklo_epi32(src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 32-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst"
    using writemask "k" (elements are copied from "src" when the corresponding mask bit is not set).
    Implements __m512i _mm512_mask_unpacklo_epi32(__m512i src, __mmask16 k, __m512i a, __m512i b)

    Operation:
    ```
    DEFINE INTERLEAVE_DWORDS(src1[127:0], src2[127:0]) {
        dst[31:0] := src1[31:0]
        dst[63:32] := src2[31:0]
        dst[95:64] := src1[63:32]
        dst[127:96] := src2[63:32]
        RETURN dst[127:0]
    }
    tmp_dst[127:0] := INTERLEAVE_DWORDS(a[127:0], b[127:0])
    tmp_dst[255:128] := INTERLEAVE_DWORDS(a[255:128], b[255:128])
    FOR j := 0 to 15
        i := j*32
        IF k[j]
            dst[i+31:i] := tmp_dst[i+31:i]
        ELSE
            dst[i+31:i] := src[i+31:i]
        FI
    ENDFOR
    dst[MAX:512] := 0
    ```
    """
    return _unpack_epi32_generic(a, b, high=False, total_bits=512, src=src, k=k)


def _mm512_mask_unpackhi_epi32(src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 32-bit integers from the high half of each 128-bit lane in "a" and "b", and store the results in "dst"
    using writemask "k" (elements are copied from "src" when the corresponding mask bit is not set).
    Implements __m512i _mm512_mask_unpackhi_epi32(__m512i src, __mmask16 k, __m512i a, __m512i b)

    Operation:
    ```
    DEFINE INTERLEAVE_HIGH_DWORDS(src1[127:0], src2[127:0]) {
        dst[31:0] := src1[95:64]
        dst[63:32] := src2[95:64]
        dst[95:64] := src1[127:96]
        dst[127:96] := src2[127:96]
        RETURN dst[127:0]
    }
    tmp_dst[127:0] := INTERLEAVE_HIGH_DWORDS(a[127:0], b[127:0])
    tmp_dst[255:128] := INTERLEAVE_HIGH_DWORDS(a[255:128], b[255:128])
    tmp_dst[383:256] := INTERLEAVE_HIGH_DWORDS(a[383:256], b[383:256])
    tmp_dst[511:384] := INTERLEAVE_HIGH_DWORDS(a[511:384], b[511:384])
    FOR j := 0 to 15
        i := j*32
        IF k[j]
            dst[i+31:i] := tmp_dst[i+31:i]
        ELSE
            dst[i+31:i] := src[i+31:i]
        FI
    ENDFOR
    dst[MAX:512] := 0
    ```
    """
    return _unpack_epi32_generic(a, b, high=True, total_bits=512, src=src, k=k)
