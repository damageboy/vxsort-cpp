from z3.z3 import (
    SeqRef,
    BitVecNumRef,
    BitVecRef,
    BitVec,
    BitVecVal,
    Solver,
    Context,
    Extract,
    Concat,
    If,
    LShR,
    ZeroExt,
    simplify,
)

zero = 0


def ymm_reg(name: str, ctx: Context):
    return BitVec(name, 32 * 8, ctx=ctx)


def zmm_reg(name: str, ctx: Context):
    return BitVec(name, 64 * 8, ctx=ctx)


def reg_with_values(
    name: str,
    s: Solver,
    raw_values,
    element_bits: int,
    total_bits: int,
    ctx: Context,
):
    lanes = total_bits // element_bits
    assert (
        len(raw_values) == lanes
    ), f"Expected {lanes} values for {element_bits}-bit elements in {total_bits}-bit register, got {len(raw_values)}"

    # Create BitVec elements for each lane
    bv_elements = [
        BitVec(f"{name}_l_{i:02}", element_bits, ctx=ctx) for i in range(lanes)
    ]

    # Add constraints for each element
    for i, raw_value in enumerate(raw_values):
        s.add(bv_elements[i] == BitVecVal(raw_value, element_bits, ctx=ctx))

    return simplify(Concat(bv_elements[::-1]))


def ymm_reg_with_32b_values(name: str, s: Solver, raw_values=None, *, ctx: Context):
    return reg_with_values(name, s, raw_values, 32, 256, ctx=ctx)


def zmm_reg_with_32b_values(name: str, s: Solver, raw_values=None, *, ctx: Context):
    return reg_with_values(name, s, raw_values, 32, 512, ctx=ctx)


def ymm_reg_with_64b_values(name: str, s: Solver, raw_values=None, *, ctx: Context):
    return reg_with_values(name, s, raw_values, 64, 256, ctx=ctx)


def zmm_reg_with_64b_values(name: str, s: Solver, raw_values=None, *, ctx: Context):
    return reg_with_values(name, s, raw_values, 64, 512, ctx=ctx)


def _reg_with_unique_values(name: str, s: Solver, lanes: int, bits: int, ctx: Context):
    """
    Create a register with given number of lanes and element width, ensuring each lane is unique.
    """
    assert (
        lanes * bits == 256 or lanes * bits == 512
    ), "Total register size can only be 256 or 512 bits"

    # Create a new register
    if lanes * bits == 256:
        reg = ymm_reg(name, ctx=ctx)
    else:
        reg = zmm_reg(name, ctx=ctx)

    elems = [Extract(bits * (i + 1) - 1, bits * i, reg) for i in range(lanes)]
    for i in range(lanes):
        for j in range(i + 1, lanes):
            s.add(elems[i] != elems[j])
    return reg


def ymm_reg_with_unique_values(name: str, s: Solver, bits: int = 32, *, ctx: Context):
    lanes = 256 // bits
    return _reg_with_unique_values(name, s, lanes=lanes, bits=bits, ctx=ctx)


def zmm_reg_with_unique_values(name: str, s: Solver, bits: int = 32, *, ctx: Context):
    lanes = 512 // bits
    return _reg_with_unique_values(name, s, lanes=lanes, bits=bits, ctx=ctx)


def ymm_reg_pair_with_unique_values(
    name_prefix: str,
    s: Solver,
    bits: int = 32,
    *,
    ctx: Context,
):
    # Create two registers with internal uniqueness
    reg1 = ymm_reg_with_unique_values(f"{name_prefix}1", s, bits, ctx=ctx)
    reg2 = ymm_reg_with_unique_values(f"{name_prefix}2", s, bits, ctx=ctx)

    # Extract all elements from both registers
    lanes = 256 // bits
    reg1_elems = [Extract(bits * (i + 1) - 1, bits * i, reg1) for i in range(lanes)]
    reg2_elems = [Extract(bits * (i + 1) - 1, bits * i, reg2) for i in range(lanes)]

    # Add cross-register uniqueness constraints
    for reg1_elem in reg1_elems:
        for reg2_elem in reg2_elems:
            s.add(reg1_elem != reg2_elem)

    return reg1, reg2


def zmm_reg_pair_with_unique_values(
    name_prefix: str,
    s: Solver,
    bits: int = 32,
    *,
    ctx: Context,
):
    # Create two registers with internal uniqueness
    reg1 = zmm_reg_with_unique_values(f"{name_prefix}1", s, bits, ctx=ctx)
    reg2 = zmm_reg_with_unique_values(f"{name_prefix}2", s, bits, ctx=ctx)

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


def construct_reg_from_elements(
    bits: int, element_specs: ElementSpecs, total_bits: int
):
    lanes = total_bits // bits
    assert (
        len(element_specs) == lanes
    ), f"Expected {lanes} element specs for {bits}-bit elements in {total_bits}-bit register, got {len(element_specs)}"

    # Extract each specified element
    elements: list[BitVecRef | SeqRef] = []
    for reg, elem_idx in element_specs:
        assert (
            0 <= elem_idx < lanes
        ), f"Element index {elem_idx} out of range for {bits}-bit elements (0-{lanes - 1})"
        start_bit = elem_idx * bits
        end_bit = start_bit + bits - 1
        elements.append(Extract(end_bit, start_bit, reg))

    # Concatenate in reverse order for Z3 (MSB first)
    return simplify(Concat(elements[::-1]))


def construct_ymm_reg_from_elements(bits: int, element_specs: ElementSpecs):
    return construct_reg_from_elements(bits, element_specs, 256)


def construct_zmm_reg_from_elements(bits: int, element_specs: ElementSpecs):
    return construct_reg_from_elements(bits, element_specs, 512)


def _reg_reversed(
    name: str,
    s: Solver,
    original_reg,
    lanes: int,
    bits: int,
    ctx: Context,
):
    assert (
        lanes * bits == 256 or lanes * bits == 512
    ), "Total register size can only be 256 or 512 bits"

    # Create a new register
    if lanes * bits == 256:
        reversed_reg = ymm_reg(name, ctx=ctx)
    else:
        reversed_reg = zmm_reg(name, ctx=ctx)

    # Extract elements from both registers
    orig_elems = [
        Extract(bits * (i + 1) - 1, bits * i, original_reg) for i in range(lanes)
    ]
    rev_elems = [
        Extract(bits * (i + 1) - 1, bits * i, reversed_reg) for i in range(lanes)
    ]

    # Add constraints that reversed register elements equal original register elements in reverse order
    for i in range(lanes):
        s.add(rev_elems[i] == orig_elems[lanes - 1 - i])

    return reversed_reg


def ymm_reg_reversed(name, s, original_reg, bits, *, ctx):
    """Create a YMM register that is the reverse of the original register through constraints."""
    lanes = 256 // bits
    return _reg_reversed(name, s, original_reg, lanes, bits, ctx=ctx)


def zmm_reg_reversed(name, s, original_reg, bits, *, ctx):
    """Create a ZMM register that is the reverse of the original register through constraints."""
    lanes = 512 // bits
    return _reg_reversed(name, s, original_reg, lanes, bits, ctx=ctx)


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


def decode_shuffle_mask(imm8: int) -> tuple[int, int, int, int]:
    """
    Decodes an 8-bit shuffle mask into _MM_SHUFFLE parameters.

    Args:
        imm8: 8-bit shuffle mask immediate value

    Returns:
        Tuple (z, y, x, w) where:
        - w = bits [1:0]
        - x = bits [3:2]
        - y = bits [5:4]
        - z = bits [7:6]

    Example:
        >>> decode_shuffle_mask(0x88)
        (2, 0, 2, 0)
        >>> decode_shuffle_mask(0xdd)
        (3, 1, 3, 1)
    """
    w = imm8 & 0b11
    x = (imm8 >> 2) & 0b11
    y = (imm8 >> 4) & 0b11
    z = (imm8 >> 6) & 0b11
    return (z, y, x, w)


def mm_shuffle_str(imm8: int) -> str:
    """
    Returns a string representation of a shuffle mask in _MM_SHUFFLE format.

    Args:
        imm8: 8-bit shuffle mask immediate value

    Returns:
        String in format "_MM_SHUFFLE(z, y, x, w)"

    Example:
        >>> mm_shuffle_str(0x88)
        '_MM_SHUFFLE(2, 0, 2, 0)'
        >>> mm_shuffle_str(0xdd)
        '_MM_SHUFFLE(3, 1, 3, 1)'
    """
    z, y, x, w = decode_shuffle_mask(imm8)
    return f"_MM_SHUFFLE({z}, {y}, {x}, {w})"


def decode_shuffle2_mask(imm8: int) -> tuple[int, int]:
    """
    Decode an imm8 shuffle mask for 2-element operations (e.g., shuffle_pd).

    For shuffle_pd with 256-bit registers:
    - Bit 0: selects element from first 128-bit lane (0 or 1)
    - Bit 1: selects element from first 128-bit lane (0 or 1)
    - Bit 2: selects element from second 128-bit lane (0 or 1)
    - Bit 3: selects element from second 128-bit lane (0 or 1)

    Returns the high and low bits as a tuple (y, x) where:
    - x = bits [1:0] (first lane selection)
    - y = bits [3:2] (second lane selection)

    Args:
        imm8: 8-bit immediate value (only lowest 4 bits used)

    Returns:
        Tuple of (y, x) where each is a 2-bit value (0-3)
    """
    x = imm8 & 0b11  # Bits [1:0]
    y = (imm8 >> 2) & 0b11  # Bits [3:2]
    return (y, x)


def mm_shuffle2_str(imm8: int) -> str:
    """
    Returns a string representation of a shuffle mask in _MM_SHUFFLE2 format
    for 2-element operations like shuffle_pd.

    Args:
        imm8: 8-bit shuffle mask immediate value (only lowest 4 bits used)

    Returns:
        String in format "_MM_SHUFFLE2(y, x)"

    Example:
        >>> mm_shuffle2_str(0x0)
        '_MM_SHUFFLE2(0, 0)'
        >>> mm_shuffle2_str(0x5)
        '_MM_SHUFFLE2(1, 1)'
        >>> mm_shuffle2_str(0xa)
        '_MM_SHUFFLE2(2, 2)'
    """
    y, x = decode_shuffle2_mask(imm8)
    return f"_MM_SHUFFLE2({y}, {x})"


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


def _create_element_selector(
    source_reg: BitVecRef, idx_bits: BitVecRef, num_elements: int, element_bits: int
) -> BitVecRef:
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


def _generic_permutexvar(
    a: BitVecRef,
    op_idx: BitVecRef,
    total_width: int,
    element_width: int,
    solver: Solver,
    src: BitVecRef | None = None,
    mask: BitVecRef | None = None,
):
    """
    Generic implementation for permutexvar instructions that shuffle elements across lanes.

    These instructions use a variable index vector to permute elements from a single source vector.
    Each element in the output is selected from the source vector based on the corresponding
    index value in the index vector. Optional masking is supported for AVX512 variants.

    Args:
        a: Source vector to permute
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
            dst[i + element_width - 1 : i] := a[index * element_width + element_width - 1 : index * element_width]
        ENDFOR
        dst[MAX:total_width] := 0
        ```

        With mask:
        ```
        FOR j := 0 to N-1
            i := j * element_width
            index := op_idx[i + IDX_BITS - 1 : i]
            IF mask[j]
                dst[i + element_width - 1 : i] := a[index * element_width + element_width - 1 : index * element_width]
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
        permuted_elem = _create_element_selector(
            a, idx_bits, num_elements, element_width
        )

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

    # Pin don't-care bits to zero to speed up Z3 constraint solving
    # and avoid duplicate solutions.
    for j in range(num_elements):
        i = j * element_width
        high_bits = Extract(i + element_width - 1, i + idx_bits_needed, op_idx)
        solver.add(high_bits == 0)

    return simplify(Concat(elems[::-1]))


def _mm256_permutexvar_epi32(a: BitVecRef, op_idx: BitVecRef, solver: Solver):
    """
    Shuffle 32-bit integers across lanes in a 256-bit vector.
    Implements __m256i _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx)
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(a, op_idx, 256, 32, solver=solver)


def _mm512_permutexvar_epi32(a: BitVecRef, op_idx: BitVecRef, solver: Solver):
    """
    Shuffle 32-bit integers across lanes in a 512-bit vector.
    Implements __m512i _mm512_permutexvar_epi32 (__m512i idx, __m512i a)
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(a, op_idx, 512, 32, solver=solver)


def _mm256_permutexvar_epi64(a: BitVecRef, op_idx: BitVecRef, solver: Solver):
    """
    Shuffle 64-bit integers across lanes in a 256-bit vector.
    Implements __m256i _mm256_permutexvar_epi64 (__m256i idx, __m256i a)
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(a, op_idx, 256, 64, solver=solver)


def _mm512_permutexvar_epi64(a: BitVecRef, op_idx: BitVecRef, solver: Solver):
    """
    Shuffle 64-bit integers across lanes in a 512-bit vector.
    Implements __m512i _mm512_permutexvar_epi64 (__m512i idx, __m512i a)
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(a, op_idx, 512, 64, solver=solver)


def _mm512_mask_permutexvar_epi32(
    src: BitVecRef,
    mask: BitVecRef,
    op_idx: BitVecRef,
    a: BitVecRef,
    solver: Solver,
):
    """
    Shuffle 32-bit integers across lanes in a 512-bit vector using writemask.
    Implements __m512i _mm512_mask_permutexvar_epi32 (__m512i src, __mmask16 k, __m512i idx, __m512i a)
    Elements are copied from src when the corresponding mask bit is not set.
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(a, op_idx, 512, 32, src=src, mask=mask, solver=solver)


def _mm512_mask_permutexvar_epi64(
    src: BitVecRef,
    mask: BitVecRef,
    op_idx: BitVecRef,
    a: BitVecRef,
    solver: Solver,
):
    """
    Shuffle 64-bit integers across lanes in a 512-bit vector using writemask.
    Implements __m512i _mm512_mask_permutexvar_epi64 (__m512i src, __mmask8 k, __m512i idx, __m512i a)
    Elements are copied from src when the corresponding mask bit is not set.
    See _generic_permutexvar for operation details.
    """
    return _generic_permutexvar(a, op_idx, 512, 64, src=src, mask=mask, solver=solver)


##
# 2xInput -> 1xOutput, fully variable index permutes:
# * vpermi2d,vpermt2d:
#   -  _mm512_permutex2var_{epi32,epi64}
#   -  _mm512_[mask]permutex2var_{epi32,epi64}


def _create_two_source_element_selector(
    a: BitVecRef,
    b: BitVecRef,
    offset_bits: BitVecRef,
    source_selector: BitVecRef,
    num_elements: int,
    element_bits: int,
) -> BitVecRef:
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
    return _create_element_selector(
        selected_source, offset_bits, num_elements, element_bits
    )


def _generic_permutex2var(
    a: BitVecRef,
    op_idx: BitVecRef,
    b: BitVecRef,
    element_width: int,
    solver: Solver,
    src: BitVecRef | None = None,
    mask: BitVecRef | None = None,
):
    """
    Generic implementation for permutex2var instructions that shuffle elements from two source vectors.

    These instructions use an index vector where each element contains:
    - Offset bits: select which element from the chosen source
    - Source selector bit: choose between source a (0) or source b (1)
    - Optional: masking is supported for AVX512 variants.

    Args:
        a: First source vector
        op_idx: Index vector containing offsets and source selectors for each destination element
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
        offset_bits = Extract(i + offset_bits_count - 1, i, op_idx)

        # Extract source selector: idx[i+source_selector_bit]
        source_selector = Extract(
            i + source_selector_bit, i + source_selector_bit, op_idx
        )

        # Get the permuted element using the two-source selector
        permuted_elem = _create_two_source_element_selector(
            a, b, offset_bits, source_selector, num_elements, element_width
        )

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

    # Pin don't-care bits to zero to speed up Z3 constraint solving
    # and avoid duplicate solutions.
    for j in range(num_elements):
        i = j * element_width
        high_bits = Extract(i + element_width - 1, i + source_selector_bit + 1, op_idx)
        solver.add(high_bits == 0)

    return simplify(Concat(elems[::-1]))


def _mm512_permutex2var_epi32(
    a: BitVecRef, op_idx: BitVecRef, b: BitVecRef, solver: Solver
):
    """
    Shuffle 32-bit integers in a and b across lanes using two-source permutation.
    Implements __m512i _mm512_permutex2var_epi32 (__m512i a, __m512i idx, __m512i b)
    See _generic_permutex2var for operation details.
    """
    return _generic_permutex2var(a, op_idx, b, 32, solver=solver)


def _mm512_permutex2var_epi64(
    a: BitVecRef, op_idx: BitVecRef, b: BitVecRef, solver: Solver
):
    """
    Shuffle 64-bit integers in a and b across lanes using two-source permutation.
    Implements __m512i _mm512_permutex2var_epi64 (__m512i a, __m512i idx, __m512i b)
    See _generic_permutex2var for operation details.
    """
    return _generic_permutex2var(a, op_idx, b, 64, solver=solver)


def _mm512_mask_permutex2var_epi32(
    a: BitVecRef,
    k: BitVecRef,
    op_idx: BitVecRef,
    b: BitVecRef,
    solver: Solver,
):
    """
    Shuffle 32-bit integer elements in a and b across lanes using writemask.
    Implements __m512i _mm512_mask_permutex2var_epi32 (__m512i a, __mmask16 k, __m512i idx, __m512i b)
    Elements are copied from a when the corresponding mask bit is not set.
    See _generic_permutex2var for operation details.
    """
    return _generic_permutex2var(a, op_idx, b, 32, src=a, mask=k, solver=solver)


def _mm512_mask_permutex2var_epi64(
    a: BitVecRef,
    k: BitVecRef,
    op_idx: BitVecRef,
    b: BitVecRef,
    solver: Solver,
):
    """
    Shuffle 64-bit integer elements in a and b across lanes using writemask.
    Implements __m512i _mm512_mask_permutex2var_epi64 (__m512i a, __mmask8 k, __m512i idx, __m512i b)
    Elements are copied from a when the corresponding mask bit is not set.
    See _generic_permutex2var for operation details.
    """
    return _generic_permutex2var(a, op_idx, b, 64, src=a, mask=k, solver=solver)


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


def _select2_pd(src_128: BitVecRef, select: BitVecRef | BitVecNumRef) -> BitVecRef:
    """Selects a 64-bit element from a 128-bit vector based on a 1-bit control."""
    return simplify(
        If(
            select == 0,
            Extract(63, 0, src_128),
            Extract(127, 64, src_128),  # select == 1
        )
    )


def _select4_epi64(src_256: BitVecRef, select: BitVecRef | BitVecNumRef) -> BitVecRef:
    """Selects a 64-bit element from a 256-bit vector based on a 2-bit control."""
    return simplify(
        If(
            select == 0,
            Extract(63, 0, src_256),
            If(
                select == 1,
                Extract(127, 64, src_256),
                If(
                    select == 2,
                    Extract(191, 128, src_256),
                    Extract(255, 192, src_256),  # select == 3
                ),
            ),
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
def vpermilps_lane(
    lane_idx: int,
    a: BitVecRef,
    ctrl01: BitVecRef,
    ctrl23: BitVecRef,
    ctrl45: BitVecRef,
    ctrl67: BitVecRef,
):
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


def _permute_ps_generic(
    a: BitVecRef,
    imm8: BitVecRef | int,
    num_lanes: int,
    k: BitVecRef | None = None,
    src: BitVecRef | None = None,
):
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
    a = a
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)
    ctrl01, ctrl23, ctrl45, ctrl67 = _extract_ctl4(imm)
    chunks_128b = [
        vpermilps_lane(lane_idx, a, ctrl01, ctrl23, ctrl45, ctrl67)
        for lane_idx in range(num_lanes)
    ]
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


def _mm256_permute_ps(a: BitVecRef, imm8: BitVecRef | int):
    """Permutes 32-bit elements within 128-bit lanes. Operates on YMM registers (2 lanes)."""
    return _permute_ps_generic(a, imm8, 2)


def _mm512_permute_ps(a: BitVecRef, imm8: BitVecRef | int):
    """Permutes 32-bit elements within 128-bit lanes. Operates on ZMM registers (4 lanes)."""
    return _permute_ps_generic(a, imm8, 4)


def _mm512_mask_permute_ps(
    src: BitVecRef, k: BitVecRef, a: BitVecRef, imm8: BitVecRef | int
):
    """
    Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in imm8,
    and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
    Implements __m512 _mm512_mask_permute_ps (__m512 src, __mmask16 k, __m512 a, const int imm8)
    """
    return _permute_ps_generic(a, imm8, 4, k=k, src=src)


def _permute_pd_generic(
    a: BitVecRef,
    imm8: BitVecRef | int,
    num_lanes: int,
    solver: Solver,
    k: BitVecRef | None = None,
    src: BitVecRef | None = None,
):
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
    a = a
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)
    ctrl0, ctrl1 = _extract_ctl2(imm)
    chunks_128b = [
        vpermilpd_lane(lane_idx, a, ctrl0, ctrl1) for lane_idx in range(num_lanes)
    ]
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

    # Pin don't-care bits to zero to speed up Z3 constraint solving
    # and avoid duplicate solutions.
    if isinstance(imm, BitVecRef):
        solver.add(Extract(7, 2, imm) == 0)

    return result


def _mm256_permute_pd(a: BitVecRef, imm8: BitVecRef | int, solver: Solver):
    """Permutes 64-bit elements within 128-bit lanes. Operates on YMM registers (2 lanes)."""
    return _permute_pd_generic(a, imm8, 2, solver=solver)


def _mm512_permute_pd(a: BitVecRef, imm8: BitVecRef | int, solver: Solver):
    """Permutes 64-bit elements within 128-bit lanes. Operates on ZMM registers (4 lanes)."""
    return _permute_pd_generic(a, imm8, 4, solver=solver)


def _mm512_mask_permute_pd(
    src: BitVecRef,
    k: BitVecRef,
    a: BitVecRef,
    imm8: BitVecRef | int,
    solver: Solver,
):
    """
    Shuffle double-precision (64-bit) floating-point elements in a within 128-bit lanes using the control in imm8,
    and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
    Implements __m512d _mm512_mask_permute_pd (__m512d src, __mmask8 k, __m512d a, const int imm8)
    """
    return _permute_pd_generic(a, imm8, 4, k=k, src=src, solver=solver)


##
# 1xInput->1xOutput, cross-lane static(imm) permutes
# - vpermq:
#   -  _mm256_permute4x64_epi64


def _mm256_permute4x64_epi64(a: BitVecRef, imm8: BitVecRef | int):
    """
    Shuffle 64-bit integers in "a" across lanes using the control in "imm8", and store the results in "dst".

    Implements __m256i _mm256_permute4x64_epi64 (__m256i a, const int imm8)

    Operation:
    ```
    DEFINE SELECT4(src, control) {
        CASE(control[1:0]) OF
        0:	tmp[63:0] := src[63:0]
        1:	tmp[63:0] := src[127:64]
        2:	tmp[63:0] := src[191:128]
        3:	tmp[63:0] := src[255:192]
        ESAC
        RETURN tmp[63:0]
    }
    dst[63:0] := SELECT4(a[255:0], imm8[1:0])
    dst[127:64] := SELECT4(a[255:0], imm8[3:2])
    dst[191:128] := SELECT4(a[255:0], imm8[5:4])
    dst[255:192] := SELECT4(a[255:0], imm8[7:6])
    dst[MAX:256] := 0
    ```

    Args:
        a: Source vector (256-bit)
        imm8: Immediate 8-bit control mask (uses all 8 bits for 4 elements, 2 bits each)

    Returns:
        Permuted 256-bit vector
    """
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    # Extract 2-bit control for each element position and select from source
    elements = [_select4_epi64(a, Extract(j * 2 + 1, j * 2, imm)) for j in range(4)]

    return simplify(Concat(elements[::-1]))


##
# 2xInput->1xOutput, within 128b lane static(imm) permutes
# - vshufps,vshufpd:
#   -  _mm256_shuffle_p{s,d}
#   -  _mm512_[mask_]shuffle_p{s,d}


def vshufps_lane(
    lane_idx: int,
    a: BitVecRef,
    b: BitVecRef,
    ctrl01: BitVecRef,
    ctrl23: BitVecRef,
    ctrl45: BitVecRef,
    ctrl67: BitVecRef,
) -> list[BitVecRef]:
    a_lane = extract_128b_lane(a, lane_idx)
    b_lane = extract_128b_lane(b, lane_idx)

    chunks = [
        _select4_ps(a_lane, ctrl01),
        _select4_ps(a_lane, ctrl23),
        _select4_ps(b_lane, ctrl45),
        _select4_ps(b_lane, ctrl67),
    ]
    return chunks


def _shuffle_ps_generic(
    a: BitVecRef,
    b: BitVecRef,
    imm8: BitVecRef | int,
    num_lanes: int,
    k: BitVecRef | None = None,
    src: BitVecRef | None = None,
):
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
    chunks_128b = [
        vshufps_lane(lane_idx, a, b, ctrl01, ctrl23, ctrl45, ctrl67)
        for lane_idx in range(num_lanes)
    ]
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


def _mm256_shuffle_ps(a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int):
    """Shuffles 32-bit elements within 128-bit lanes. Operates on YMM registers (2 lanes)."""
    return _shuffle_ps_generic(a, b, imm8, 2)


def _mm512_shuffle_ps(a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int):
    """Shuffles 32-bit elements within 128-bit lanes. Operates on ZMM registers (4 lanes)."""
    return _shuffle_ps_generic(a, b, imm8, 4)


def _mm512_mask_shuffle_ps(
    src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int
):
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
    ctrl1 = Extract(
        2 * lane_idx + 1, 2 * lane_idx + 1, imm
    )  # Controls selection from b

    chunks: list[BitVecRef | None] = [None] * 2
    chunks[0] = _select2_pd(a_lane, ctrl0)
    chunks[1] = _select2_pd(b_lane, ctrl1)
    return chunks


def _shuffle_pd_generic(
    a: BitVecRef,
    b: BitVecRef,
    imm8: BitVecRef | int,
    num_lanes: int,
    solver: Solver,
    k: BitVecRef | None = None,
    src: BitVecRef | None = None,
):
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
    chunks_128b = [vshufpd_lane(lane_idx, a, b, imm) for lane_idx in range(num_lanes)]
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

    # Pin don't-care bits to zero to speed up Z3 constraint solving
    # and avoid duplicate solutions.
    if isinstance(imm, BitVecRef) and num_lanes * 2 < 8:
        solver.add(Extract(7, num_lanes * 2, imm) == 0)

    return result


def _mm256_shuffle_pd(
    a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int, solver: Solver
):
    """Shuffles 64-bit elements within 128-bit lanes. Operates on YMM registers (2 lanes)."""
    return _shuffle_pd_generic(a, b, imm8, 2, solver=solver)


def _mm512_shuffle_pd(
    a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int, solver: Solver
):
    """Shuffles 64-bit elements within 128-bit lanes. Operates on ZMM registers (4 lanes)."""
    return _shuffle_pd_generic(a, b, imm8, 4, solver=solver)


def _mm512_mask_shuffle_pd(
    src: BitVecRef,
    k: BitVecRef,
    a: BitVecRef,
    b: BitVecRef,
    imm8: BitVecRef | int,
    solver: Solver,
):
    """
    Shuffle double-precision (64-bit) floating-point elements within 128-bit lanes using the control in imm8,
    and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
    Implements __m512d _mm512_mask_shuffle_pd (__m512d src, __mmask8 k, __m512d a, __m512d b, const int imm8)
    """
    return _shuffle_pd_generic(a, b, imm8, 4, k=k, src=src, solver=solver)


##
# 2xInput->1xOutput, within 128b lane variable index permutes
# - vpermilps/vpermilpd:
#   -  _mm256_permutevar_p{s,d}
#   -  _mm512_[mask_]permutevar_p{s,d}


def _generic_permutevar(
    a: BitVecRef,
    b: BitVecRef,
    total_width: int,
    element_width: int,
    solver: Solver,
    k: BitVecRef | None = None,
    src: BitVecRef | None = None,
):
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

    # Pin don't-care bits to zero to speed up Z3 constraint solving
    # and avoid duplicate solutions.
    used_bits_end = 2
    for j in range(num_elements):
        i = j * element_width
        high_bits = Extract(i + element_width - 1, i + used_bits_end, b)
        solver.add(high_bits == 0)
    if element_width == 64:
        for j in range(num_elements):
            i = j * element_width
            solver.add(Extract(i, i, b) == 0)

    return simplify(Concat(elements[::-1]))


def _mm256_permutevar_ps(a: BitVecRef, b: BitVecRef, solver: Solver):
    """
    Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in b.
    Implements __m256 _mm256_permutevar_ps (__m256 a, __m256i b)
    """
    return _generic_permutevar(a, b, total_width=256, element_width=32, solver=solver)


def _mm512_permutevar_ps(a: BitVecRef, b: BitVecRef, solver: Solver):
    """
    Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in b.
    Implements __m512 _mm512_permutevar_ps (__m512 a, __m512i b)
    """
    return _generic_permutevar(a, b, total_width=512, element_width=32, solver=solver)


def _mm512_mask_permutevar_ps(
    src: BitVecRef,
    k: BitVecRef,
    a: BitVecRef,
    b: BitVecRef,
    solver: Solver,
):
    """
    Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in b,
    and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
    Implements __m512 _mm512_mask_permutevar_ps (__m512 src, __mmask16 k, __m512 a, __m512i b)
    """
    return _generic_permutevar(
        a, b, total_width=512, element_width=32, k=k, src=src, solver=solver
    )


def _mm256_permutevar_pd(a: BitVecRef, b: BitVecRef, solver: Solver):
    """
    Shuffle double-precision (64-bit) floating-point elements in a within 128-bit lanes using the control in b.
    Implements __m256d _mm256_permutevar_pd (__m256d a, __m256i b)
    """
    return _generic_permutevar(a, b, total_width=256, element_width=64, solver=solver)


def _mm512_permutevar_pd(a: BitVecRef, b: BitVecRef, solver: Solver):
    """
    Shuffle double-precision (64-bit) floating-point elements in a within 128-bit lanes using the control in b.
    Implements __m512d _mm512_permutevar_pd (__m512d a, __m512i b)
    """
    return _generic_permutevar(a, b, total_width=512, element_width=64, solver=solver)


def _mm512_mask_permutevar_pd(
    src: BitVecRef,
    k: BitVecRef,
    a: BitVecRef,
    b: BitVecRef,
    solver: Solver,
):
    """
    Shuffle double-precision (64-bit) floating-point elements in a within 128-bit lanes using the control in b,
    and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
    Implements __m512d _mm512_mask_permutevar_pd (__m512d src, __mmask8 k, __m512d a, __m512i b)
    """
    return _generic_permutevar(
        a, b, total_width=512, element_width=64, k=k, src=src, solver=solver
    )


##
# 2xInput -> 1xOutput, whole 128b lane static(imm) permutes
# - vperm2i128:
#   -  _mm256_permute2x128_si256
#   -  _mm512_[mask_]shuffle_i32x4
# Note that while both the AVX2 and AVX512 versions *generally* shuffle whole 128b lanes,
# The AVX2 version has a more complex semantics for the control bits.
# The same functionality also exists in the AVX512 version, but it is "split"
# into two separate functions: _mm512_shuffle_i32x4 and _mm512_mask_shuffle_i32x4


def _select4_128b(
    src1: BitVecRef,
    src2: BitVecRef,
    control: BitVecRef | BitVecNumRef,
    solver: Solver,
) -> BitVecRef:
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
    result = simplify(If(zero_flag == 1, BitVecVal(0, 128), selected_lane))

    # Pin don't-care bits to zero to speed up Z3 constraint solving
    # and avoid duplicate solutions.
    if isinstance(control, BitVecRef):
        solver.add(Extract(2, 2, control) == 0)

    return result


def _mm256_permute2x128_si256(
    a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int, solver: Solver
):
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
        lanes[i] = _select4_128b(a, b, control_bits, solver=solver)

    # Concatenate the lanes (reverse order since Concat puts first arg in MSB)
    return simplify(Concat(lanes[::-1]))


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


def _mm512_mask_shuffle_i32x4(
    src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int
):
    """
    Shuffle 128-bit lanes from a and b using imm8, with merge masking.

    Implements __m512i _mm512_mask_shuffle_i32x4(__m512i src, __mmask16 k,
    __m512i a, __m512i b, const int imm8)

    Elements are copied from src when the corresponding mask bit is not set.
    """
    # First do the unmasked shuffle
    unmasked = _mm512_shuffle_i32x4(a, b, imm8)

    # Apply merge mask element-wise (32-bit elements, 16 total)
    elements = [None] * 16
    for j in range(16):
        lo = j * 32
        hi = lo + 31
        mask_bit = Extract(j, j, k)
        elements[j] = If(mask_bit == 1, Extract(hi, lo, unmasked), Extract(hi, lo, src))

    return simplify(Concat(elements[::-1]))


##
# 2xInput -> 1xOutput, blend hi/lo half of each 128b lane
# - vpunpckldq:
#   -  _mm256_unpacklo_epi32
#   -  _mm512_[mask_]unpacklo_epi32
# - vpunpckhdq:
#   -  _mm256_unpackhi_epi32
#   -  _mm512_[mask_]unpackhi_epi32


def _unpack_epi32_generic(
    a: BitVecRef,
    b: BitVecRef,
    high: bool,
    total_bits: int,
    src: BitVecRef = None,
    k: BitVecRef = None,
):
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

    Pseudocode:
    For each 128-bit lane in the input:
      If high is False (unpacklo), interleave elements 0 and 1 of a and b within each lane:
        dst[0] = a[0], dst[1] = b[0], dst[2] = a[1], dst[3] = b[1]
      If high is True (unpackhi), interleave elements 2 and 3 of a and b within each lane:
        dst[0] = a[2], dst[1] = b[2], dst[2] = a[3], dst[3] = b[3]

    For total_bits=256, process 2 lanes; for 512, process 4 lanes.
    If masking is requested (src and k are not None), for each 32-bit element, choose the result from
    the unpacked value if the corresponding mask bit is set, otherwise use the value from src.
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
    """
    return _unpack_epi32_generic(a, b, high=False, total_bits=256)


def _mm256_unpackhi_epi32(a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 32-bit integers from the high half of each 128-bit lane in "a" and "b", and store the results in "dst".
    Implements __m256i _mm256_unpackhi_epi32(__m256i a, __m256i b)
    """
    return _unpack_epi32_generic(a, b, high=True, total_bits=256)


def _mm512_unpacklo_epi32(a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 32-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst".
    Implements __m512i _mm512_unpacklo_epi32(__m512i a, __m512i b)
    """
    return _unpack_epi32_generic(a, b, high=False, total_bits=512)


def _mm512_unpackhi_epi32(a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 32-bit integers from the high half of each 128-bit lane in "a" and "b", and store the results in "dst".
    Implements __m512i _mm512_unpackhi_epi32(__m512i a, __m512i b)
    """
    return _unpack_epi32_generic(a, b, high=True, total_bits=512)


def _mm512_mask_unpacklo_epi32(
    src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef
):
    """
    Unpack and interleave 32-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst"
    using writemask "k" (elements are copied from "src" when the corresponding mask bit is not set).
    Implements __m512i _mm512_mask_unpacklo_epi32(__m512i src, __mmask16 k, __m512i a, __m512i b)
    """
    return _unpack_epi32_generic(a, b, high=False, total_bits=512, src=src, k=k)


def _mm512_mask_unpackhi_epi32(
    src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef
):
    """
    Unpack and interleave 32-bit integers from the high half of each 128-bit lane in "a" and "b", and store the results in "dst"
    using writemask "k" (elements are copied from "src" when the corresponding mask bit is not set).
    Implements __m512i _mm512_mask_unpackhi_epi32(__m512i src, __mmask16 k, __m512i a, __m512i b)
    """
    return _unpack_epi32_generic(a, b, high=True, total_bits=512, src=src, k=k)


def _unpack_epi64_generic(
    a: BitVecRef,
    b: BitVecRef,
    high: bool,
    total_bits: int,
    src: BitVecRef = None,
    k: BitVecRef = None,
):
    """
    Generic unpack implementation for 64-bit integers with optional masking.

    Args:
        a: First source register
        b: Second source register
        high: True for unpackhi (element 1), False for unpacklo (element 0)
        total_bits: Register size (256 or 512)
        src: Source register for masked operations (None for unmasked)
        k: Write mask (None for unmasked operations)

    Returns:
        BitVecRef representing the unpacked result

    Pseudocode:
    For each 128-bit lane in the input:
      If high is False (unpacklo), interleave element 0 of a and b within each lane:
        dst[0] = a[0], dst[1] = b[0]
      If high is True (unpackhi), interleave element 1 of a and b within each lane:
        dst[0] = a[1], dst[1] = b[1]

    For total_bits=256, process 2 lanes (4 elements total); for 512, process 4 lanes (8 elements total).
    If masking is requested (src and k are not None), for each 64-bit element, choose the result from
    the unpacked value if the corresponding mask bit is set, otherwise use the value from src.
    """
    assert total_bits in [256, 512], "total_bits must be 256 or 512"

    num_lanes = total_bits // 128  # Number of 128-bit lanes
    num_elements = total_bits // 64  # Total number of 64-bit elements

    elements = [None] * num_elements

    # Process each 128-bit lane
    for lane in range(num_lanes):
        lane_start = lane * 128

        if high:
            # Extract high element (1) from each lane
            a_elem = Extract(lane_start + 127, lane_start + 64, a)  # a[lane][1]
            b_elem = Extract(lane_start + 127, lane_start + 64, b)  # b[lane][1]
        else:
            # Extract low element (0) from each lane
            a_elem = Extract(lane_start + 63, lane_start + 0, a)  # a[lane][0]
            b_elem = Extract(lane_start + 63, lane_start + 0, b)  # b[lane][0]

        # Interleave: a[elem], b[elem]
        base_idx = lane * 2
        elements[base_idx + 0] = a_elem
        elements[base_idx + 1] = b_elem

    # If masking is requested, apply the mask
    if src is not None and k is not None:
        masked_elements = [None] * num_elements
        for j in range(num_elements):
            i = j * 64

            # Extract mask bit for this element
            mask_bit = Extract(j, j, k)

            # Extract elements from both unpacked result and src
            unpack_elem = elements[j]
            src_elem = Extract(i + 63, i, src)

            # Apply mask: if mask bit is set, use unpacked result, otherwise use src
            masked_elements[j] = simplify(If(mask_bit == 1, unpack_elem, src_elem))
        elements = masked_elements

    return simplify(Concat(elements[::-1]))


def _mm256_unpacklo_epi64(a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 64-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst".
    Implements __m256i _mm256_unpacklo_epi64(__m256i a, __m256i b)
    """
    return _unpack_epi64_generic(a, b, high=False, total_bits=256)


def _mm256_unpackhi_epi64(a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 64-bit integers from the high half of each 128-bit lane in "a" and "b", and store the results in "dst".
    Implements __m256i _mm256_unpackhi_epi64(__m256i a, __m256i b)
    """
    return _unpack_epi64_generic(a, b, high=True, total_bits=256)


def _mm512_unpacklo_epi64(a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 64-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst".
    Implements __m512i _mm512_unpacklo_epi64(__m512i a, __m512i b)
    """
    return _unpack_epi64_generic(a, b, high=False, total_bits=512)


def _mm512_unpackhi_epi64(a: BitVecRef, b: BitVecRef):
    """
    Unpack and interleave 64-bit integers from the high half of each 128-bit lane in "a" and "b", and store the results in "dst".
    Implements __m512i _mm512_unpackhi_epi64(__m512i a, __m512i b)
    """
    return _unpack_epi64_generic(a, b, high=True, total_bits=512)


def _mm512_mask_unpacklo_epi64(
    src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef
):
    """
    Unpack and interleave 64-bit integers from the low half of each 128-bit lane in "a" and "b", and store the results in "dst"
    using writemask "k" (elements are copied from "src" when the corresponding mask bit is not set).
    Implements __m512i _mm512_mask_unpacklo_epi64(__m512i src, __mmask8 k, __m512i a, __m512i b)
    """
    return _unpack_epi64_generic(a, b, high=False, total_bits=512, src=src, k=k)


def _mm512_mask_unpackhi_epi64(
    src: BitVecRef, k: BitVecRef, a: BitVecRef, b: BitVecRef
):
    """
    Unpack and interleave 64-bit integers from the high half of each 128-bit lane in "a" and "b", and store the results in "dst"
    using writemask "k" (elements are copied from "src" when the corresponding mask bit is not set).
    Implements __m512i _mm512_mask_unpackhi_epi64(__m512i src, __mmask8 k, __m512i a, __m512i b)
    """
    return _unpack_epi64_generic(a, b, high=True, total_bits=512, src=src, k=k)


##
# 2xInput -> 1xOutput, blend operations
# - vblendpd:
#   -  _mm256_blend_pd
# - vblendps:
#   -  _mm256_blend_ps
# - vblendvpd:
#   -  _mm256_blendv_pd
# - vblendvps:
#   -  _mm256_blendv_ps


def _generic_blend(
    a: BitVecRef,
    b: BitVecRef,
    imm8: BitVecRef | int,
    total_width: int,
    element_width: int,
    solver: Solver,
):
    """
    Generic implementation for immediate blend instructions that select elements from two source vectors.

    These instructions use an immediate mask where each bit controls the selection for one element.
    If the mask bit is 1, the element is selected from b; otherwise from a.

    Args:
        a: First source vector
        b: Second source vector
        imm8: Immediate 8-bit control mask
        total_width: Total bit width of the vectors (256)
        element_width: Width of each element in bits (32 or 64)

    Returns:
        Blended vector

    Generic Operation (where N = total_width / element_width):
        ```
        FOR j := 0 to N-1
            i := j * element_width
            IF imm8[j]
                dst[i + element_width - 1 : i] := b[i + element_width - 1 : i]
            ELSE
                dst[i + element_width - 1 : i] := a[i + element_width - 1 : i]
            FI
        ENDFOR
        dst[MAX:total_width] := 0
        ```

    Examples:
        - _mm256_blend_pd: total_width=256, element_width=64 → 4 elements
        - _mm256_blend_ps: total_width=256, element_width=32 → 8 elements
    """
    num_elements = total_width // element_width
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    elements = [None] * num_elements

    for j in range(num_elements):
        i = j * element_width
        # Extract mask bit for this element
        mask_bit = Extract(j, j, imm)
        # Extract elements from both sources
        a_elem = Extract(i + element_width - 1, i, a)
        b_elem = Extract(i + element_width - 1, i, b)
        # Blend: if mask bit is 1, use b; otherwise use a
        elements[j] = simplify(If(mask_bit == 1, b_elem, a_elem))

    # Pin don't-care bits to zero to speed up Z3 constraint solving
    # and avoid duplicate solutions.
    if isinstance(imm8, BitVecRef) and num_elements < 8:
        solver.add(Extract(7, num_elements, imm) == 0)

    return simplify(Concat(elements[::-1]))


def _generic_blendv(
    a: BitVecRef,
    b: BitVecRef,
    mask: BitVecRef,
    total_width: int,
    element_width: int,
    solver: Solver,
):
    """
    Generic implementation for variable blend instructions that select elements from two source vectors.

    These instructions use a mask vector where the sign bit (MSB) of each element controls the selection.
    If the sign bit is 1, the element is selected from b; otherwise from a.

    Args:
        a: First source vector
        b: Second source vector
        mask: Variable mask vector (uses sign bit of each element)
        total_width: Total bit width of the vectors (256)
        element_width: Width of each element in bits (32 or 64)

    Returns:
        Blended vector

    Generic Operation (where N = total_width / element_width):
        ```
        FOR j := 0 to N-1
            i := j * element_width
            IF mask[i + element_width - 1]  // sign bit (MSB)
                dst[i + element_width - 1 : i] := b[i + element_width - 1 : i]
            ELSE
                dst[i + element_width - 1 : i] := a[i + element_width - 1 : i]
            FI
        ENDFOR
        dst[MAX:total_width] := 0
        ```

    Examples:
        - _mm256_blendv_pd: total_width=256, element_width=64 → 4 elements
        - _mm256_blendv_ps: total_width=256, element_width=32 → 8 elements
    """
    num_elements = total_width // element_width

    elements = [None] * num_elements

    for j in range(num_elements):
        i = j * element_width
        # Extract sign bit (MSB) for this element: mask[i + element_width - 1]
        sign_bit = Extract(i + element_width - 1, i + element_width - 1, mask)
        # Extract elements from both sources
        a_elem = Extract(i + element_width - 1, i, a)
        b_elem = Extract(i + element_width - 1, i, b)
        # Blend: if sign bit is 1, use b; otherwise use a
        elements[j] = simplify(If(sign_bit == 1, b_elem, a_elem))

    # Pin don't-care bits to zero to speed up Z3 constraint solving
    # and avoid duplicate solutions.
    for j in range(num_elements):
        i = j * element_width
        low_bits = Extract(i + element_width - 2, i, mask)
        solver.add(low_bits == 0)

    return simplify(Concat(elements[::-1]))


def _mm256_blend_pd(a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int, solver: Solver):
    """
    Blend packed double-precision (64-bit) floating-point elements from "a" and "b" using control mask "imm8",
    and store the results in "dst".
    Implements __m256d _mm256_blend_pd (__m256d a, __m256d b, const int imm8)
    """
    return _generic_blend(a, b, imm8, 256, 64, solver=solver)


def _mm256_blend_ps(a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int, solver: Solver):
    """
    Blend packed single-precision (32-bit) floating-point elements from "a" and "b" using control mask "imm8",
    and store the results in "dst".
    Implements __m256 _mm256_blend_ps (__m256 a, __m256 b, const int imm8)
    """
    return _generic_blend(a, b, imm8, 256, 32, solver=solver)


def _mm256_blendv_pd(a: BitVecRef, b: BitVecRef, mask: BitVecRef, solver: Solver):
    """
    Blend packed double-precision (64-bit) floating-point elements from "a" and "b" using "mask",
    and store the results in "dst".
    Implements __m256d _mm256_blendv_pd (__m256d a, __m256d b, __m256d mask)
    """
    return _generic_blendv(a, b, mask, 256, 64, solver=solver)


def _mm256_blendv_ps(a: BitVecRef, b: BitVecRef, mask: BitVecRef, solver: Solver):
    """
    Blend packed single-precision (32-bit) floating-point elements from "a" and "b" using "mask",
    and store the results in "dst".
    Implements __m256 _mm256_blendv_ps (__m256 a, __m256 b, __m256 mask)
    """
    return _generic_blendv(a, b, mask, 256, 32, solver=solver)


##
# 2xInput -> 1xOutput, alignr (concatenate and shift right)
# - vpalignr:
#   -  _mm256_alignr_epi8
# - valignd:
#   -  _mm256_alignr_epi32
#   -  _mm512_alignr_epi32
#   -  _mm512_mask_alignr_epi32
# - valignq:
#   -  _mm256_alignr_epi64
#   -  _mm512_alignr_epi64
#   -  _mm512_mask_alignr_epi64


def _generic_alignr(
    a: BitVecRef,
    b: BitVecRef,
    imm8: BitVecRef | int,
    total_width: int,
    element_width: int,
    solver: Solver,
    src: BitVecRef | None = None,
    k: BitVecRef | None = None,
):
    """
    Generic implementation for alignr instructions that concatenate two vectors and shift right.

    These instructions concatenate vector a (high part) and vector b (low part) into a
    double-width temporary, shift the result right by imm8 elements, and store the
    low half in the destination. Optional masking is supported for AVX512 variants.

    Args:
        a: First source vector (becomes high part of concatenation)
        b: Second source vector (becomes low part of concatenation)
        imm8: Immediate value specifying shift amount in elements
        total_width: Total bit width of each vector (256 or 512)
        element_width: Width of each element in bits (32 or 64)
        src: Optional source vector for masked operations (values used when mask bit is 0)
        k: Optional predicate mask (if provided, src must also be provided)

    Returns:
        Aligned/shifted vector (optionally masked)

    Generic Operation (where N = total_width / element_width):
        Without mask:
        ```
        temp[2*total_width-1:total_width] := a[total_width-1:0]
        temp[total_width-1:0] := b[total_width-1:0]
        temp[2*total_width-1:0] := temp[2*total_width-1:0] >> (element_width * imm8)
        dst[total_width-1:0] := temp[total_width-1:0]
        dst[MAX:total_width] := 0
        ```

        With mask:
        ```
        temp[2*total_width-1:total_width] := a[total_width-1:0]
        temp[total_width-1:0] := b[total_width-1:0]
        temp[2*total_width-1:0] := temp[2*total_width-1:0] >> (element_width * imm8)
        FOR j := 0 to N-1
            i := j * element_width
            IF k[j]
                dst[i + element_width - 1 : i] := temp[i + element_width - 1 : i]
            ELSE
                dst[i + element_width - 1 : i] := src[i + element_width - 1 : i]
            FI
        ENDFOR
        dst[MAX:total_width] := 0
        ```

    Examples:
        - _mm256_alignr_epi8: total_width=256, element_width=8 → 32 elements, shift by 0-31
        - _mm256_alignr_epi32: total_width=256, element_width=32 → 8 elements, shift by 0-7
        - _mm512_alignr_epi32: total_width=512, element_width=32 → 16 elements, shift by 0-15
        - _mm256_alignr_epi64: total_width=256, element_width=64 → 4 elements, shift by 0-3
        - _mm512_alignr_epi64: total_width=512, element_width=64 → 8 elements, shift by 0-7
        - _mm512_mask_alignr_epi32: total_width=512, element_width=32, with src and k
        - _mm512_mask_alignr_epi64: total_width=512, element_width=64, with src and k
    """
    num_elements = total_width // element_width
    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)

    # Extract the relevant bits from imm8 based on the number of elements
    # For 32-bit elements: 256-bit uses 3 bits, 512-bit uses 4 bits
    # For 64-bit elements: 256-bit uses 2 bits, 512-bit uses 3 bits
    shift_bits_needed = (num_elements - 1).bit_length()
    shift_amount = Extract(shift_bits_needed - 1, 0, imm)

    # Extract all elements from both vectors to form the concatenated temp
    # temp = [a_elements | b_elements] (a is high, b is low)
    a_elements = [
        Extract(element_width * (i + 1) - 1, element_width * i, a)
        for i in range(num_elements)
    ]
    b_elements = [
        Extract(element_width * (i + 1) - 1, element_width * i, b)
        for i in range(num_elements)
    ]

    # Concatenate: b elements first (indices 0..N-1), then a elements (indices N..2N-1)
    all_elements = b_elements + a_elements

    # Select elements after shifting by shift_amount
    # After shifting right by shift_amount, we take elements [shift_amount : shift_amount + num_elements)
    result_elements = [None] * num_elements

    for j in range(num_elements):
        # For each output position, we need to select from all_elements[shift_amount + j]
        # Use nested If statements to handle all possible shift amounts
        selected = all_elements[
            -1
        ]  # Default to last element (shouldn't happen if shift is in range)

        # Build the selection tree from the end
        for shift_val in range(2 * num_elements - 1, -1, -1):
            if shift_val + j < 2 * num_elements:
                selected = If(
                    shift_amount == shift_val, all_elements[shift_val + j], selected
                )

        result_elements[j] = selected

    # Apply mask if provided
    if k is not None and src is not None:
        masked_elements = [None] * num_elements
        for j in range(num_elements):
            i = j * element_width
            mask_bit = Extract(j, j, k)
            src_elem = Extract(i + element_width - 1, i, src)
            masked_elements[j] = simplify(
                If(mask_bit == 1, result_elements[j], src_elem)
            )
        result_elements = masked_elements

    # Pin don't-care bits to zero to speed up Z3 constraint solving
    # and avoid duplicate solutions.
    if isinstance(imm, BitVecRef) and shift_bits_needed < 8:
        solver.add(Extract(7, shift_bits_needed, imm) == 0)

    return simplify(Concat(result_elements[::-1]))


def _mm256_alignr_epi8(
    a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int, solver: Solver
):
    """
    Align packed bytes from b and a within each 128-bit lane.

    FOR j := 0 to 1
        i := j*128
        tmp[255:0] := ((a[i+127:i] << 128)[255:0] OR b[i+127:i]) >> (imm8*8)
        dst[i+127:i] := tmp[127:0]
    ENDFOR

    Implements __m256i _mm256_alignr_epi8(__m256i a, __m256i b, const int imm8)
    (VPALIGNR YMM form; lane-local semantics).
    """
    del solver

    imm = imm8 if isinstance(imm8, BitVecRef) else BitVecVal(imm8, 8)
    shift_bits = ZeroExt(248, imm) * BitVecVal(8, 256)

    lane_results = []
    for lane in range(2):
        lo = lane * 128
        hi = lo + 127
        a_lane = Extract(hi, lo, a)
        b_lane = Extract(hi, lo, b)
        lane_concat = Concat(a_lane, b_lane)
        lane_shifted = LShR(lane_concat, shift_bits)
        lane_results.append(Extract(127, 0, lane_shifted))

    # Z3 concat is MSB-first; lane 1 is high half, lane 0 is low half.
    return simplify(Concat(lane_results[1], lane_results[0]))


def _mm256_alignr_epi32(
    a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int, solver: Solver
):
    """
    Concatenate a and b into a 64-byte result, shift right by imm8 32-bit elements,
    and store the low 32 bytes (8 elements) in dst.
    Implements __m256i _mm256_alignr_epi32(__m256i a, __m256i b, const int imm8)
    See _generic_alignr for operation details.
    """
    return _generic_alignr(a, b, imm8, 256, 32, solver=solver)


def _mm512_alignr_epi32(
    a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int, solver: Solver
):
    """
    Concatenate a and b into a 128-byte result, shift right by imm8 32-bit elements,
    and store the low 64 bytes (16 elements) in dst.
    Implements __m512i _mm512_alignr_epi32(__m512i a, __m512i b, const int imm8)
    See _generic_alignr for operation details.
    """
    return _generic_alignr(a, b, imm8, 512, 32, solver=solver)


def _mm512_mask_alignr_epi32(
    src: BitVecRef,
    k: BitVecRef,
    a: BitVecRef,
    b: BitVecRef,
    imm8: BitVecRef | int,
    solver: Solver,
):
    """
    Concatenate a and b into a 128-byte result, shift right by imm8 32-bit elements,
    and store the low 64 bytes (16 elements) in dst using writemask k.
    Elements are copied from src when the corresponding mask bit is not set.
    Implements __m512i _mm512_mask_alignr_epi32(__m512i src, __mmask16 k, __m512i a, __m512i b, const int imm8)
    See _generic_alignr for operation details.
    """
    return _generic_alignr(a, b, imm8, 512, 32, src=src, k=k, solver=solver)


def _mm256_alignr_epi64(
    a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int, solver: Solver
):
    """
    Concatenate a and b into a 64-byte result, shift right by imm8 64-bit elements,
    and store the low 32 bytes (4 elements) in dst.
    Implements __m256i _mm256_alignr_epi64(__m256i a, __m256i b, const int imm8)
    See _generic_alignr for operation details.
    """
    return _generic_alignr(a, b, imm8, 256, 64, solver=solver)


def _mm512_alignr_epi64(
    a: BitVecRef, b: BitVecRef, imm8: BitVecRef | int, solver: Solver
):
    """
    Concatenate a and b into a 128-byte result, shift right by imm8 64-bit elements,
    and store the low 64 bytes (8 elements) in dst.
    Implements __m512i _mm512_alignr_epi64(__m512i a, __m512i b, const int imm8)
    See _generic_alignr for operation details.
    """
    return _generic_alignr(a, b, imm8, 512, 64, solver=solver)


def _mm512_mask_alignr_epi64(
    src: BitVecRef,
    k: BitVecRef,
    a: BitVecRef,
    b: BitVecRef,
    imm8: BitVecRef | int,
    solver: Solver,
):
    """
    Concatenate a and b into a 128-byte result, shift right by imm8 64-bit elements,
    and store the low 64 bytes (8 elements) in dst using writemask k.
    Elements are copied from src when the corresponding mask bit is not set.
    Implements __m512i _mm512_mask_alignr_epi64(__m512i src, __mmask8 k, __m512i a, __m512i b, const int imm8)
    See _generic_alignr for operation details.
    """
    return _generic_alignr(a, b, imm8, 512, 64, src=src, k=k, solver=solver)


# ── Min / Max ────────────────────────────────────────────────────────────────


def _generic_minmax(a, b, total_width, element_width, take_min):
    """Element-wise signed min or max.

    Z3 BitVec ``<`` is ``bvslt`` (signed), matching the hardware instructions
    (vpminsd/vpmaxsd for i32, vpminsq/vpmaxsq for i64).

    Args:
        a, b: Z3 BitVecRef operands of *total_width* bits.
        total_width: Register width in bits (256 or 512).
        element_width: Element width in bits (32 or 64).
        take_min: If True return element-wise min, else max.
    """
    num_elements = total_width // element_width
    elements = []
    for j in range(num_elements):
        lo = j * element_width
        hi = lo + element_width - 1
        a_elem = Extract(hi, lo, a)
        b_elem = Extract(hi, lo, b)
        if take_min:
            elements.append(If(a_elem < b_elem, a_elem, b_elem))
        else:
            elements.append(If(a_elem < b_elem, b_elem, a_elem))
    return simplify(Concat(elements[::-1]))


def generic_min(a, b, total_width, element_width):
    """Element-wise signed minimum."""
    return _generic_minmax(a, b, total_width, element_width, take_min=True)


def generic_max(a, b, total_width, element_width):
    """Element-wise signed maximum."""
    return _generic_minmax(a, b, total_width, element_width, take_min=False)


def _mm256_min_epi32(a: BitVecRef, b: BitVecRef):
    """Element-wise signed 32-bit minimum across a 256-bit register."""
    return _generic_minmax(a, b, 256, 32, take_min=True)


def _mm256_max_epi32(a: BitVecRef, b: BitVecRef):
    """Element-wise signed 32-bit maximum across a 256-bit register."""
    return _generic_minmax(a, b, 256, 32, take_min=False)


def _mm256_min_epi64(a: BitVecRef, b: BitVecRef):
    """Element-wise signed 64-bit minimum across a 256-bit register."""
    return _generic_minmax(a, b, 256, 64, take_min=True)


def _mm256_max_epi64(a: BitVecRef, b: BitVecRef):
    """Element-wise signed 64-bit maximum across a 256-bit register."""
    return _generic_minmax(a, b, 256, 64, take_min=False)


def _mm512_min_epi32(a: BitVecRef, b: BitVecRef):
    """Element-wise signed 32-bit minimum across a 512-bit register."""
    return _generic_minmax(a, b, 512, 32, take_min=True)


def _mm512_max_epi32(a: BitVecRef, b: BitVecRef):
    """Element-wise signed 32-bit maximum across a 512-bit register."""
    return _generic_minmax(a, b, 512, 32, take_min=False)


def _mm512_min_epi64(a: BitVecRef, b: BitVecRef):
    """Element-wise signed 64-bit minimum across a 512-bit register."""
    return _generic_minmax(a, b, 512, 64, take_min=True)


def _mm512_max_epi64(a: BitVecRef, b: BitVecRef):
    """Element-wise signed 64-bit maximum across a 512-bit register."""
    return _generic_minmax(a, b, 512, 64, take_min=False)
