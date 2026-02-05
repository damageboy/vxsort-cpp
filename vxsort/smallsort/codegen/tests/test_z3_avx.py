from z3 import Solver, unsat, sat, BitVec, BitVecVal, Concat, Extract

# Assuming your z3s functions and registers are importable, e.g.:
from z3_avx import _MM_SHUFFLE, _MM_SHUFFLE2, decode_shuffle_mask, mm_shuffle_str
from z3_avx import _mm256_permute_ps
from z3_avx import _mm512_permute_ps
from z3_avx import _mm256_permutexvar_epi32
from z3_avx import _mm512_permutexvar_epi32
from z3_avx import _mm512_mask_permutexvar_epi32
from z3_avx import _mm512_permutex2var_epi32
from z3_avx import _mm512_permutex2var_epi64
from z3_avx import _mm512_mask_permutex2var_epi32
from z3_avx import _mm512_mask_permutex2var_epi64
from z3_avx import _mm256_permutexvar_epi64
from z3_avx import _mm512_permutexvar_epi64
from z3_avx import _mm512_mask_permutexvar_epi64
from z3_avx import _mm256_shuffle_ps
from z3_avx import _mm512_shuffle_ps
from z3_avx import _mm256_shuffle_pd
from z3_avx import _mm512_shuffle_pd
from z3_avx import _mm256_permute_pd
from z3_avx import _mm512_permute_pd
from z3_avx import _mm256_permute2x128_si256
from z3_avx import _mm512_shuffle_i32x4
from z3_avx import _mm256_unpacklo_epi32, _mm256_unpackhi_epi32
from z3_avx import _mm512_unpacklo_epi32, _mm512_unpackhi_epi32
from z3_avx import _mm512_mask_unpacklo_epi32, _mm512_mask_unpackhi_epi32
from z3_avx import _mm512_mask_permute_ps, _mm512_mask_permute_pd
from z3_avx import _mm512_mask_shuffle_ps, _mm512_mask_shuffle_pd
from z3_avx import _mm256_permutevar_ps, _mm512_permutevar_ps, _mm512_mask_permutevar_ps
from z3_avx import _mm256_permutevar_pd, _mm512_permutevar_pd, _mm512_mask_permutevar_pd
from z3_avx import _mm256_blend_pd, _mm256_blend_ps, _mm256_blendv_pd, _mm256_blendv_ps
from z3_avx import _mm256_permute4x64_epi64
from z3_avx import _mm256_alignr_epi32, _mm512_alignr_epi32, _mm512_mask_alignr_epi32
from z3_avx import _mm256_alignr_epi64, _mm512_alignr_epi64, _mm512_mask_alignr_epi64
from z3_avx import (
    ymm_reg,
    ymm_reg_with_32b_values,
    ymm_reg_with_64b_values,
    ymm_reg_with_unique_values,
    ymm_reg_pair_with_unique_values,
    construct_ymm_reg_from_elements,
)
from z3_avx import (
    zmm_reg,
    zmm_reg_with_32b_values,
    zmm_reg_with_64b_values,
    zmm_reg_with_unique_values,
    zmm_reg_pair_with_unique_values,
    construct_zmm_reg_from_elements,
)
from z3_avx import ymm_reg_reversed, zmm_reg_reversed

#    imm8 = 0b11100100 means:
#    - Lane bits [1:0] = 00 (select element 0 for position 0)
#    - Lane bits [3:2] = 01 (select element 1 for position 1)
#    - Lane bits [5:4] = 10 (select element 2 for position 2)
#    - Lane bits [7:6] = 11 (select element 3 for position 3)
#    This should result in each 128-bit lane's elements staying in place.

null_permute_epi32_imm8 = _MM_SHUFFLE(3, 2, 1, 0)
null_permute_pd_imm8 = _MM_SHUFFLE2(
    1, 0
)  # bit 1 = 1 (select elem 1 for pos 1), bit 0 = 0 (select elem 0 for pos 0)

null_shuffle_ps_imm8 = _MM_SHUFFLE(
    3, 2, 1, 0
)  # pos0: op1[0], pos1: op1[1], pos2: op1[2], pos3: op1[3]
null_shuffle_ps_2vec_imm8 = _MM_SHUFFLE(
    1, 0, 1, 0
)  # pos0: op1[0], pos1: op1[1], pos2: op2[0], pos3: op2[1]
null_shuffle_pd_avx2_imm8 = (
    0x0A  # 0b1010: identity permutation for AVX2 (2 lanes, uses bits 0-3)
)
null_shuffle_pd_avx512_imm8 = (
    0xAA  # 0b10101010: identity permutation for AVX512 (4 lanes, uses bits 0-7)
)

# For _mm256_permute2x128_si256 null permute:
# Low lane: select a[127:0] (control=0), High lane: select a[255:128] (control=1)
null_permute2x128_imm8 = (
    1 << 4
) | 0  # 0x10: high_lane=1 (a[255:128]), low_lane=0 (a[127:0])

# For _mm512_shuffle_i32x4 null permute:
# dst[127:0] := a[127:0] (imm8[1:0] = 0), dst[255:128] := a[255:128] (imm8[3:2] = 1)
# dst[383:256] := b[383:256] (imm8[5:4] = 2), dst[511:384] := b[511:384] (imm8[7:6] = 3)
null_shuffle_i32x4_imm8 = _MM_SHUFFLE(3, 2, 1, 0)  # 0xE4

null_permute_vector_epi32_avx2 = [i for i in range(8)]
null_permute_vector_epi32_avx512 = [i for i in range(16)]
null_permute_vector_epi64_avx2 = [i for i in range(4)]
null_permute_vector_epi64_avx512 = [i for i in range(8)]
null_permutex2var_vector_epi32_avx512 = [
    i for i in range(16)
]  # source selector = 0, offset = i
null_permutex2var_vector_epi64_avx512 = [
    i for i in range(8)
]  # source selector = 0, offset = i
reverse_permute_vector_epi32_avx2 = null_permute_vector_epi32_avx2[::-1]
reverse_permute_vector_epi32_avx512 = null_permute_vector_epi32_avx512[::-1]
reverse_permute_vector_epi64_avx2 = null_permute_vector_epi64_avx2[::-1]
reverse_permute_vector_epi64_avx512 = null_permute_vector_epi64_avx512[::-1]


def array_to_long(values, bits):
    """
    Convert a Python array of integers to a single long integer with Z3 bit ordering.

    Args:
        values: List of integer values
        bits: Number of bits per element (32 for epi32, 64 for epi64)


    Returns:
        Single integer representing the packed values in Z3 bit ordering
    """
    result = 0
    for val in reversed(values):  # Reverse to match Z3 bit ordering
        result = (result << bits) | val
    return result


class TestPermutePs:
    """Tests for _mm256_permute_ps and _mm512_permute_ps (permute_epi32)"""

    def test_mm256_permute_epi32_null_permute_works(self):
        s = Solver()
        input = ymm_reg("ymm0")
        output_vector = _mm256_permute_ps(input, null_permute_epi32_imm8)

        s.add(input != output_vector)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute_epi32_null_permute_found(self):
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=32)
        imm8 = BitVec("imm8", 8)
        output = _mm256_permute_ps(input, imm8)

        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_permute_epi32_imm8, (
            "Z3 found unexpected null permute: got 0x{model_imm8:08x}, expected 0x{null_permute_epi32_imm8:08x}"
        )

    def test_mm512_permute_epi32_null_permute(self):
        s = Solver()

        input = zmm_reg("zmm0")
        output = _mm512_permute_ps(input, null_permute_epi32_imm8)

        s.add(input != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permute_epi32_null_permute_found(self):
        s = Solver()
        input = zmm_reg_with_unique_values("zmm0", s, bits=32)
        imm8 = BitVec("imm8", 8)
        output = _mm512_permute_ps(input, imm8)

        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute failed"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_permute_epi32_imm8, (
            "Z3 found unexpected null permute: got 0x{model_imm8:08x}, expected 0x{null_permute_epi32_imm8:08x}"
        )


class TestPermutePd:
    """Tests for _mm256_permute_pd and _mm512_permute_pd (permute_epi64)"""

    def test_mm256_permute_epi64_null_permute_works(self):
        s = Solver()
        input = ymm_reg("ymm0")
        output = _mm256_permute_pd(input, null_permute_pd_imm8)

        s.add(input != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute_epi64_null_permute_found(self):
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        imm8 = BitVec("imm8", 8)
        output = _mm256_permute_pd(input, imm8)

        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_permute_pd_imm8, (
            "Z3 found unexpected null permute: got 0x{model_imm8:08x}, expected 0x{null_permute_pd_imm8:08x}"
        )

    def test_mm512_permute_epi64_null_permute_works(self):
        s = Solver()

        input = zmm_reg("zmm0")
        output = _mm512_permute_pd(input, null_permute_pd_imm8)

        s.add(input != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permute_epi64_null_permute_found(self):
        s = Solver()
        input = zmm_reg_with_unique_values("zmm0", s, bits=64)
        imm8 = BitVec("imm8", 8)
        output = _mm512_permute_pd(input, imm8)

        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_permute_pd_imm8, (
            "Z3 found unexpected null permute: got 0x{model_imm8:08x}, expected 0x{null_permute_pd_imm8:08x}"
        )


class TestPermutexvarEpi32:
    """Tests for _mm256_permutexvar_epi32 and _mm512_permutexvar_epi32"""

    def test_mm256_permutexvar_epi32_null_permute_works(self):
        s = Solver()
        input = ymm_reg("ymm0")
        indices = ymm_reg_with_32b_values("indices", s, null_permute_vector_epi32_avx2)
        output = _mm256_permutexvar_epi32(input, indices)

        s.add(input != output)
        result = s.check()

        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permutexvar_epi32_null_permute_found(self):
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=32)
        indices = ymm_reg("indices")
        output = _mm256_permutexvar_epi32(input, indices)

        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_indices = s.model().evaluate(indices).as_long()
        expected_long = array_to_long(null_permute_vector_epi32_avx2, bits=32)
        assert model_indices == expected_long, (
            f"Z3 found unexpected null permute: got 0x{model_indices:064x}, expected 0x{expected_long:064x}"
        )

    def test_mm256_permutexvar_epi32_reverse_permute_found(self):
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=32)
        indices = ymm_reg("indices")
        output = _mm256_permutexvar_epi32(input, indices)

        reversed_input = ymm_reg_reversed("ymm_reversed", s, input, bits=32)

        s.add(output == reversed_input)
        result = s.check()

        assert result == sat, "Z3 failed to find reverse permute"
        model_indices = s.model().evaluate(indices).as_long()
        expected_long = array_to_long(reverse_permute_vector_epi32_avx2, bits=32)
        assert model_indices == expected_long, (
            f"Z3 found unexpected reverse permute: got 0x{model_indices:064x}, expected 0x{expected_long:064x}"
        )

    def test_mm512_permutexvar_epi32_null_permute_works(self):
        s = Solver()
        input = zmm_reg("zmm0")
        indicew = zmm_reg_with_32b_values(
            "indices", s, null_permute_vector_epi32_avx512
        )
        output = _mm512_permutexvar_epi32(input, indicew)

        # Assert that the output is NOT equal to the input
        # If this is unsatisfiable, it means the output MUST be equal to the input
        # and that the null permute vector can only lead to an identity permutation
        s.add(input != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutexvar_epi32_null_permute_found(self):
        s = Solver()
        input = zmm_reg_with_unique_values("zmm0", s, bits=32)
        indices = zmm_reg("indices")
        output = _mm512_permutexvar_epi32(input, indices)

        # Assert that the output equals the input (seeking identity permutation)
        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_indices = s.model().evaluate(indices).as_long()
        expected_long = array_to_long(null_permute_vector_epi32_avx512, bits=32)
        assert model_indices == expected_long, (
            "Z3 found unexpected null permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"
        )

    def test_mm512_permutexvar_epi32_reverse_permute_found(self):
        s = Solver()
        input = zmm_reg_with_unique_values("zmm0", s, bits=32)
        indices = zmm_reg("indices")
        output = _mm512_permutexvar_epi32(input, indices)

        # Create reversed input using constraints
        reversed_input = zmm_reg_reversed("zmm_reversed", s, input, bits=32)

        # Assert that the output equals the reversed input (seeking reverse permutation)
        s.add(output == reversed_input)
        result = s.check()

        assert result == sat, "Z3 failed to find reverse permute"
        model_indices = s.model().evaluate(indices).as_long()
        expected_long = array_to_long(reverse_permute_vector_epi32_avx512, bits=32)
        assert model_indices == expected_long, (
            "Z3 found unexpected reverse permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"
        )


class TestPermutexvarEpi64:
    """Tests for _mm256_permutexvar_epi64 and _mm512_permutexvar_epi64"""

    def test_mm256_permutexvar_epi64_null_permute_works(self):
        s = Solver()
        input = ymm_reg("ymm0")
        indices = ymm_reg_with_64b_values("indices", s, null_permute_vector_epi64_avx2)
        output = _mm256_permutexvar_epi64(input, indices)

        s.add(input != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permutexvar_epi64_null_permute_found(self):
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        indices = ymm_reg("indices")
        output = _mm256_permutexvar_epi64(input, indices)

        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_indices = s.model().evaluate(indices).as_long()
        expected_long = array_to_long(null_permute_vector_epi64_avx2, bits=64)
        assert model_indices == expected_long, (
            "Z3 found unexpected null permute: got 0x{model_indices:064x}, expected 0x{expected_long:064x}"
        )

    def test_mm256_permutexvar_epi64_reverse_permute_found(self):
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        indices = ymm_reg("indices")
        output = _mm256_permutexvar_epi64(input, indices)

        reversed_input = ymm_reg_reversed("ymm_reversed", s, input, bits=64)

        s.add(output == reversed_input)
        result = s.check()

        assert result == sat, "Z3 failed to find reverse permute"
        model_indices = s.model().evaluate(indices).as_long()
        expected_long = array_to_long(reverse_permute_vector_epi64_avx2, bits=64)
        assert model_indices == expected_long, (
            "Z3 found unexpected reverse permute: got 0x{model_indices:064x}, expected 0x{expected_long:064x}"
        )

    def test_mm512_permutexvar_epi64_null_permute_works(self):
        s = Solver()
        input = zmm_reg("zmm0")
        indices = zmm_reg_with_64b_values(
            "indices", s, null_permute_vector_epi64_avx512
        )
        output = _mm512_permutexvar_epi64(input, indices)

        s.add(input != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutexvar_epi64_null_permute_found(self):
        s = Solver()
        input = zmm_reg_with_64b_values("zmm0", s, [i + 1 for i in range(8)])
        indices = zmm_reg("indices")
        output = _mm512_permutexvar_epi64(input, indices)

        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_indices = s.model().evaluate(indices).as_long()
        expected_long = array_to_long(null_permute_vector_epi64_avx512, bits=64)
        assert model_indices == expected_long, (
            "Z3 found unexpected null permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"
        )

    def test_mm512_permutexvar_epi64_reverse_permute_found(self):
        s = Solver()
        input = zmm_reg_with_64b_values("zmm0", s, [i + 1 for i in range(8)])
        indices = zmm_reg("indices")
        output = _mm512_permutexvar_epi64(input, indices)

        reversed_input = zmm_reg_reversed("zmm_reversed", s, input, bits=64)

        s.add(output == reversed_input)
        result = s.check()

        assert result == sat, "Z3 failed to find reverse permute"
        model_indices = s.model().evaluate(indices).as_long()
        expected_long = array_to_long(reverse_permute_vector_epi64_avx512, bits=64)
        assert model_indices == expected_long, (
            "Z3 found unexpected reverse permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"
        )


class TestMaskPermutexvarEpi32:
    """Tests for _mm512_mask_permutexvar_epi32 (512-bit masked variant)"""

    def test_mm512_mask_permutexvar_epi32_mask_all_zeros(self):
        """Test with mask all zeros (should preserve src)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, null_permute_vector_epi32_avx512
        )
        mask = BitVecVal(0, 16)  # All mask bits are 0

        output = _mm512_mask_permutexvar_epi32(src, mask, indices, a)

        s.add(output != src)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all zeros: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutexvar_epi32_mask_all_ones(self):
        """Test with mask all ones (should equal unmasked operation)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, null_permute_vector_epi32_avx512
        )
        mask = BitVecVal(0xFFFF, 16)  # All mask bits are 1

        masked_output = _mm512_mask_permutexvar_epi32(src, mask, indices, a)
        unmasked_output = _mm512_permutexvar_epi32(a, indices)

        s.add(masked_output != unmasked_output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all ones: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutexvar_epi32_alternating_mask(self):
        """Test with alternating mask pattern"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, reverse_permute_vector_epi32_avx512
        )
        mask = BitVecVal(0x5555, 16)  # Alternating: 0101010101010101

        output = _mm512_mask_permutexvar_epi32(src, mask, indices, a)
        unmasked = _mm512_permutexvar_epi32(a, indices)

        # Expected: unmasked result in even positions (mask bit 1), src in odd positions (mask bit 0)
        expected_specs = []
        for i in range(16):
            if i % 2 == 0:
                expected_specs.append((unmasked, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for alternating mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutexvar_epi32_single_bit_mask(self):
        """Test with only one bit set in mask"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, reverse_permute_vector_epi32_avx512
        )
        mask = BitVecVal(1 << 7, 16)  # Only bit 7 is set

        output = _mm512_mask_permutexvar_epi32(src, mask, indices, a)
        unmasked = _mm512_permutexvar_epi32(a, indices)

        # Expected: unmasked result only at position 7, src everywhere else
        expected_specs = []
        for i in range(16):
            if i == 7:
                expected_specs.append((unmasked, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for single bit mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutexvar_epi32_partial_mask(self):
        """Test with lower half masked"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, reverse_permute_vector_epi32_avx512
        )
        mask = BitVecVal(0x00FF, 16)  # Lower 8 bits set

        output = _mm512_mask_permutexvar_epi32(src, mask, indices, a)

        reversed_a = zmm_reg_reversed("a_reversed", s, a, bits=32)

        # Expected: reversed a in positions 0-7, src in positions 8-15
        expected_specs = []
        for i in range(16):
            if i < 8:
                expected_specs.append((reversed_a, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for partial mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutexvar_epi32_find_mask_for_identity(self):
        """Test that Z3 can find mask to preserve src (mask all zeros)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, reverse_permute_vector_epi32_avx512
        )
        mask = BitVec("mask", 16)

        output = _mm512_mask_permutexvar_epi32(src, mask, indices, a)

        s.add(output == src)
        result = s.check()

        assert result == sat, "Z3 failed to find mask for identity"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0, (
            f"Z3 found unexpected mask for identity: got 0x{model_mask:04x}, expected 0x0000"
        )

    def test_mm512_mask_permutexvar_epi32_find_mask_for_full_permute(self):
        """Test that Z3 can find mask for full permutation (mask all ones)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, null_permute_vector_epi32_avx512
        )
        mask = BitVec("mask", 16)

        output = _mm512_mask_permutexvar_epi32(src, mask, indices, a)

        s.add(output == a)
        result = s.check()

        assert result == sat, "Z3 failed to find mask for full permutation"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0xFFFF, (
            f"Z3 found unexpected mask for full permutation: got 0x{model_mask:04x}, expected 0xFFFF"
        )


class TestMaskPermutexvarEpi64:
    """Tests for _mm512_mask_permutexvar_epi64 (512-bit masked variant)"""

    def test_mm512_mask_permutexvar_epi64_mask_all_zeros(self):
        """Test with mask all zeros (should preserve src)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, null_permute_vector_epi64_avx512
        )
        mask = BitVecVal(0, 8)  # All mask bits are 0

        output = _mm512_mask_permutexvar_epi64(src, mask, indices, a)

        s.add(output != src)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all zeros: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutexvar_epi64_mask_all_ones(self):
        """Test with mask all ones (should equal unmasked operation)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, null_permute_vector_epi64_avx512
        )
        mask = BitVecVal(0xFF, 8)  # All mask bits are 1

        masked_output = _mm512_mask_permutexvar_epi64(src, mask, indices, a)
        unmasked_output = _mm512_permutexvar_epi64(a, indices)

        s.add(masked_output != unmasked_output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all ones: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutexvar_epi64_alternating_mask(self):
        """Test with alternating mask pattern"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, reverse_permute_vector_epi64_avx512
        )
        mask = BitVecVal(0x55, 8)  # Alternating: 01010101

        output = _mm512_mask_permutexvar_epi64(src, mask, indices, a)
        unmasked = _mm512_permutexvar_epi64(a, indices)

        # Expected: unmasked result in even positions (mask bit 1), src in odd positions (mask bit 0)
        expected_specs = []
        for i in range(8):
            if i % 2 == 0:
                expected_specs.append((unmasked, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for alternating mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutexvar_epi64_single_bit_mask(self):
        """Test with only one bit set in mask"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, reverse_permute_vector_epi64_avx512
        )
        mask = BitVecVal(1 << 3, 8)  # Only bit 3 is set

        output = _mm512_mask_permutexvar_epi64(src, mask, indices, a)
        unmasked = _mm512_permutexvar_epi64(a, indices)

        # Expected: unmasked result only at position 3, src everywhere else
        expected_specs = []
        for i in range(8):
            if i == 3:
                expected_specs.append((unmasked, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for single bit mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutexvar_epi64_partial_mask(self):
        """Test with lower half masked"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, reverse_permute_vector_epi64_avx512
        )
        mask = BitVecVal(0x0F, 8)  # Lower 4 bits set

        output = _mm512_mask_permutexvar_epi64(src, mask, indices, a)

        reversed_a = zmm_reg_reversed("a_reversed", s, a, bits=64)

        # Expected: reversed a in positions 0-3, src in positions 4-7
        expected_specs = []
        for i in range(8):
            if i < 4:
                expected_specs.append((reversed_a, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for partial mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutexvar_epi64_find_mask_for_identity(self):
        """Test that Z3 can find mask to preserve src (mask all zeros)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, reverse_permute_vector_epi64_avx512
        )
        mask = BitVec("mask", 8)

        output = _mm512_mask_permutexvar_epi64(src, mask, indices, a)

        s.add(output == src)
        result = s.check()

        assert result == sat, "Z3 failed to find mask for identity"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0, (
            f"Z3 found unexpected mask for identity: got 0x{model_mask:02x}, expected 0x00"
        )

    def test_mm512_mask_permutexvar_epi64_find_mask_for_full_permute(self):
        """Test that Z3 can find mask for full permutation (mask all ones)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, null_permute_vector_epi64_avx512
        )
        mask = BitVec("mask", 8)

        output = _mm512_mask_permutexvar_epi64(src, mask, indices, a)

        s.add(output == a)
        result = s.check()

        assert result == sat, "Z3 failed to find mask for full permutation"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0xFF, (
            f"Z3 found unexpected mask for full permutation: got 0x{model_mask:02x}, expected 0xFF"
        )

    def test_mm512_mask_permutexvar_epi64_find_indices_and_mask(self):
        """Test that Z3 can find both indices and mask to achieve a specific pattern"""
        s = Solver()

        src = zmm_reg_with_64b_values(
            "src", s, [0x100, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107]
        )
        a = zmm_reg_with_64b_values(
            "a", s, [0x200, 0x201, 0x202, 0x203, 0x204, 0x205, 0x206, 0x207]
        )
        indices = zmm_reg("indices")
        mask = BitVec("mask", 8)

        output = _mm512_mask_permutexvar_epi64(src, mask, indices, a)

        # We want: first 4 elements reversed from a, last 4 from src unchanged
        # Expected: [a[3], a[2], a[1], a[0], src[4], src[5], src[6], src[7]]
        #         = [0x203, 0x202, 0x201, 0x200, 0x104, 0x105, 0x106, 0x107]
        expected = construct_zmm_reg_from_elements(
            64, [(a, 3), (a, 2), (a, 1), (a, 0), (src, 4), (src, 5), (src, 6), (src, 7)]
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find indices and mask for pattern"
        model_mask = s.model().evaluate(mask).as_long()
        # Lower 4 bits should be set (positions 0-3 use permuted values)
        assert model_mask == 0x0F, (
            f"Z3 found unexpected mask: got 0x{model_mask:02x}, expected 0x0F"
        )


class TestPermutex2varEpi32:
    """Tests for _mm512_permutex2var_epi32 (512-bit only)"""

    def test_mm512_permutex2var_epi32_null_permute_works(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, null_permutex2var_vector_epi32_avx512
        )
        output = _mm512_permutex2var_epi32(a, indices, b)

        # If this is unsatisfiable, it means the output MUST be equal to source a
        s.add(a != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutex2var_epi32_null_permute_found(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg("indices")
        output = _mm512_permutex2var_epi32(a, indices, b)
        s.add(a == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_indices = s.model().evaluate(indices).as_long()
        expected_long = array_to_long(null_permutex2var_vector_epi32_avx512, bits=32)
        assert model_indices == expected_long, (
            f"Z3 found unexpected null permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"
        )

    def test_mm512_permutex2var_epi32_select_from_b(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)

        select_b_indices = [(1 << 4) | i for i in range(16)]
        indices = zmm_reg_with_32b_values("indices", s, select_b_indices)
        output = _mm512_permutex2var_epi32(a, indices, b)

        s.add(b != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where select from b failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutex2var_epi32_reverse_permute_from_a(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        reverse_a_indices = [(0 << 4) | (15 - i) for i in range(16)]
        indices = zmm_reg_with_32b_values("indices", s, reverse_a_indices)

        output = _mm512_permutex2var_epi32(a, indices, b)

        # Create reversed input using constraints
        reversed_a = zmm_reg_reversed("a_reversed", s, a, bits=32)

        # Assert that the output is NOT equal to the reversed source a
        # If this is unsatisfiable, it means the output MUST equal the reversed source a
        s.add(reversed_a != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where reverse permute from a failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutex2var_epi32_mixed_sources(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        mixed_indices = []
        for i in range(16):
            if i % 2 == 0:
                # Even position: select from source a
                mixed_indices.append((0 << 4) | i)
            else:
                # Odd position: select from source b
                mixed_indices.append((1 << 4) | i)

        indices = zmm_reg_with_32b_values("indices", s, mixed_indices)
        output = _mm512_permutex2var_epi32(a, indices, b)

        expected_specs = []
        for i in range(16):
            if i % 2 == 0:
                # Even position: element i from source a
                expected_specs.append((a, i))
            else:
                # Odd position: element i from source b
                expected_specs.append((b, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        # Assert that the output is NOT equal to the expected result
        # If this is unsatisfiable, it means the output MUST equal the expected result
        s.add(expected != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where mixed sources failed: {s.model() if result == sat else 'No model'}"
        )


class TestPermutex2varEpi64:
    """Tests for _mm512_permutex2var_epi64 (512-bit only)"""

    def test_mm512_permutex2var_epi64_null_permute_works(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, null_permutex2var_vector_epi64_avx512
        )
        output = _mm512_permutex2var_epi64(a, indices, b)
        s.add(a != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutex2var_epi64_null_permute_found(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        indices = zmm_reg("indices")
        output = _mm512_permutex2var_epi64(a, indices, b)
        s.add(a == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_indices = s.model().evaluate(indices).as_long()
        expected_long = array_to_long(null_permutex2var_vector_epi64_avx512, bits=64)
        assert model_indices == expected_long, (
            f"Z3 found unexpected null permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"
        )

    def test_mm512_permutex2var_epi64_select_from_b(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)

        select_b_indices = [(1 << 3) | i for i in range(8)]
        indices = zmm_reg_with_64b_values("indices", s, select_b_indices)
        output = _mm512_permutex2var_epi64(a, indices, b)
        s.add(b != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where select from b failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutex2var_epi64_reverse_permute_from_a(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)

        reverse_a_indices = [(0 << 3) | (7 - i) for i in range(8)]
        indices = zmm_reg_with_64b_values("indices", s, reverse_a_indices)

        output = _mm512_permutex2var_epi64(a, indices, b)

        reversed_a = zmm_reg_reversed("a_reversed", s, a, bits=64)

        s.add(reversed_a != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where reverse permute from a failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutex2var_epi64_mixed_sources(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)

        mixed_indices = []
        for i in range(8):
            if i % 2 == 0:
                mixed_indices.append((0 << 3) | i)
            else:
                mixed_indices.append((1 << 3) | i)

        indices = zmm_reg_with_64b_values("indices", s, mixed_indices)
        output = _mm512_permutex2var_epi64(a, indices, b)

        expected_specs = []
        for i in range(8):
            if i % 2 == 0:
                expected_specs.append((a, i))
            else:
                expected_specs.append((b, i))

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(expected != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where mixed sources failed: {s.model() if result == sat else 'No model'}"
        )


class TestShufflePs:
    """Tests for _mm256_shuffle_ps and _mm512_shuffle_ps"""

    def test_mm256_shuffle_ps_null_permute_works(self):
        s = Solver()

        input = ymm_reg("ymm0")
        output = _mm256_shuffle_ps(input, input, null_shuffle_ps_imm8)

        s.add(output != input)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_shuffle_ps_null_permute_found(self):
        s = Solver()

        input = ymm_reg_with_unique_values("ymm0", s, bits=32)
        imm8 = BitVec("imm8", 8)
        output = _mm256_shuffle_ps(input, input, imm8)

        s.add(output == input)
        result = s.check()

        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_ps_imm8, (
            f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_ps_2vec_imm8:02x}"
        )

    def test_mm256_shuffle_ps_null_permute_2vec_works(self):
        s = Solver()

        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=32)

        output = _mm256_shuffle_ps(op1, op2, null_shuffle_ps_2vec_imm8)

        expected = construct_ymm_reg_from_elements(
            32,
            [
                (op1, 0),
                (op1, 1),
                (op2, 0),
                (op2, 1),
                (op1, 4),
                (op1, 5),
                (op2, 4),
                (op2, 5),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_shuffle_ps_null_permute_2vec_found(self):
        s = Solver()

        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=32)

        imm8 = BitVec("imm8", 8)
        output = _mm256_shuffle_ps(op1, op2, imm8)

        expected = construct_ymm_reg_from_elements(
            32,
            [
                (op1, 0),
                (op1, 1),
                (op2, 0),
                (op2, 1),
                (op1, 4),
                (op1, 5),
                (op2, 4),
                (op2, 5),
            ],
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_ps_2vec_imm8, (
            f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_ps_2vec_imm8:02x}"
        )

    def test_mm512_shuffle_ps_null_permute_works(self):
        s = Solver()

        input_vector = zmm_reg("zmm0")
        output_vector = _mm512_shuffle_ps(
            input_vector, input_vector, null_shuffle_ps_2vec_imm8
        )

        expected = construct_zmm_reg_from_elements(
            32,
            [
                (input_vector, 0),
                (input_vector, 1),
                (input_vector, 0),
                (input_vector, 1),
                (input_vector, 4),
                (input_vector, 5),
                (input_vector, 4),
                (input_vector, 5),
                (input_vector, 8),
                (input_vector, 9),
                (input_vector, 8),
                (input_vector, 9),
                (input_vector, 12),
                (input_vector, 13),
                (input_vector, 12),
                (input_vector, 13),
            ],
        )

        s.add(output_vector != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_shuffle_ps_null_permute_found(self):
        s = Solver()

        input = zmm_reg_with_unique_values("zmm0", s, bits=32)
        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_ps(input, input, imm8)

        expected = construct_zmm_reg_from_elements(
            32,
            [
                (input, 0),
                (input, 1),
                (input, 0),
                (input, 1),
                (input, 4),
                (input, 5),
                (input, 4),
                (input, 5),
                (input, 8),
                (input, 9),
                (input, 8),
                (input, 9),
                (input, 12),
                (input, 13),
                (input, 12),
                (input, 13),
            ],
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_ps_2vec_imm8, (
            f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_ps_2vec_imm8:02x}"
        )

    def test_mm512_shuffle_ps_null_permute_2vec_works(self):
        s = Solver()

        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=32)

        output = _mm512_shuffle_ps(op1, op2, null_shuffle_ps_2vec_imm8)

        expected = construct_zmm_reg_from_elements(
            32,
            [
                (op1, 0),
                (op1, 1),
                (op2, 0),
                (op2, 1),
                (op1, 4),
                (op1, 5),
                (op2, 4),
                (op2, 5),
                (op1, 8),
                (op1, 9),
                (op2, 8),
                (op2, 9),
                (op1, 12),
                (op1, 13),
                (op2, 12),
                (op2, 13),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_shuffle_ps_null_permute_2vec_found(self):
        s = Solver()

        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=32)

        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_ps(op1, op2, imm8)

        expected = construct_zmm_reg_from_elements(
            32,
            [
                (op1, 0),
                (op1, 1),
                (op2, 0),
                (op2, 1),
                (op1, 4),
                (op1, 5),
                (op2, 4),
                (op2, 5),
                (op1, 8),
                (op1, 9),
                (op2, 8),
                (op2, 9),
                (op1, 12),
                (op1, 13),
                (op2, 12),
                (op2, 13),
            ],
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_ps_2vec_imm8, (
            f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_ps_2vec_imm8:02x}"
        )

    def test_mm256_shuffle_ps_bitonic_stage_masks(self):
        """
        Test finding shuffle masks for specific bitonic sorter vector states.

        Input vector state:
        Top:    [1,  2,  5,  6,  9,  10, 13, 14]
        Bottom: [4,  3,  8,  7,  12, 11, 16, 15]

        Find two shuffle masks:
        1. Target output: [1, 5, 4, 8, 9, 13, 12, 16]
        2. Target output: [2, 6, 3, 7, 10, 14, 11, 15]
        """
        # First target: [1, 5, 4, 8, 9, 13, 12, 16]
        s1 = Solver()
        op1, op2 = ymm_reg_pair_with_unique_values("vec", s1, bits=32)

        imm8_1 = BitVec("imm8_1", 8)
        out1 = _mm256_shuffle_ps(op1, op2, imm8_1)

        exp1 = construct_ymm_reg_from_elements(
            32,
            [
                (op1, 0),  # elem 1
                (op1, 2),  # elem 5
                (op2, 0),  # elem 4
                (op2, 2),  # elem 8
                (op1, 4),  # elem 9
                (op1, 6),  # elem 13
                (op2, 4),  # elem 12
                (op2, 6),  # elem 16
            ],
        )

        s1.add(out1 == exp1)
        res1 = s1.check()

        assert res1 == sat, "Z3 failed to find shuffle mask for first target"
        model_imm8_1 = s1.model().evaluate(imm8_1).as_long()
        print(
            f"First shuffle mask found: 0x{model_imm8_1:02x} | 0b{model_imm8_1:08b} = {mm_shuffle_str(model_imm8_1)}"
        )

        # Second target: [2, 6, 3, 7, 10, 14, 11, 15]
        s2 = Solver()
        op1, op2 = ymm_reg_pair_with_unique_values("vec", s2, bits=32)

        imm8_2 = BitVec("imm8_2", 8)
        out2 = _mm256_shuffle_ps(op1, op2, imm8_2)

        exp2 = construct_ymm_reg_from_elements(
            32,
            [
                (op1, 1),  # elem 2
                (op1, 3),  # elem 6
                (op2, 1),  # elem 3
                (op2, 3),  # elem 7
                (op1, 5),  # elem 10
                (op1, 7),  # elem 14
                (op2, 5),  # elem 11
                (op2, 7),  # elem 15
            ],
        )

        s2.add(out2 == exp2)
        res2 = s2.check()

        assert res2 == sat, "Z3 failed to find shuffle mask for second target"
        model_imm8_2 = s2.model().evaluate(imm8_2).as_long()
        print(
            f"Second shuffle mask found: 0x{model_imm8_2:02x} | 0b{model_imm8_2:08b} = {mm_shuffle_str(model_imm8_2)}"
        )

    def test_mm256_shuffle_ps_bitonic_stage_masks_literal(self):
        """
        Test finding shuffle masks using literal values for bitonic sorter vector states.

        Input vector state:
        Top:    [1,  2,  5,  6,  9,  10, 13, 14]
        Bottom: [4,  3,  8,  7,  12, 11, 16, 15]

        Find two shuffle masks:
        1. Target output: [1, 5, 4, 8, 9, 13, 12, 16]
        2. Target output: [2, 6, 3, 7, 10, 14, 11, 15]
        """
        # First target: [1, 5, 4, 8, 9, 13, 12, 16]
        s1 = Solver()
        op1 = ymm_reg_with_32b_values("op1", s1, [1, 2, 5, 6, 9, 10, 13, 14])
        op2 = ymm_reg_with_32b_values("op2", s1, [4, 3, 8, 7, 12, 11, 16, 15])

        imm8_1 = BitVec("imm8_1", 8)
        out1 = _mm256_shuffle_ps(op1, op2, imm8_1)
        exp1 = ymm_reg_with_32b_values("exp1", s1, [1, 5, 4, 8, 9, 13, 12, 16])

        s1.add(out1 == exp1)
        res1 = s1.check()

        assert res1 == sat, "Z3 failed to find shuffle mask for first target"
        model_imm8_1 = s1.model().evaluate(imm8_1).as_long()
        print(
            f"First shuffle mask found: 0x{model_imm8_1:02x} = 0b{model_imm8_1:08b} = {mm_shuffle_str(model_imm8_1)}"
        )

        # Second target: [2, 6, 3, 7, 10, 14, 11, 15]
        s2 = Solver()
        op1 = ymm_reg_with_32b_values("op1", s2, [1, 2, 5, 6, 9, 10, 13, 14])
        op2 = ymm_reg_with_32b_values("op2", s2, [4, 3, 8, 7, 12, 11, 16, 15])

        imm8_2 = BitVec("imm8_2", 8)
        out2 = _mm256_shuffle_ps(op1, op2, imm8_2)
        exp2 = ymm_reg_with_32b_values("exp2", s2, [2, 6, 3, 7, 10, 14, 11, 15])

        s2.add(out2 == exp2)
        res2 = s2.check()

        assert res2 == sat, "Z3 failed to find shuffle mask for second target"
        model_imm8_2 = s2.model().evaluate(imm8_2).as_long()
        print(
            f"Second shuffle mask found: 0x{model_imm8_2:02x} = 0b{model_imm8_2:08b} = {mm_shuffle_str(model_imm8_2)}"
        )


class TestShufflePd:
    """Tests for _mm256_shuffle_pd and _mm512_shuffle_pd"""

    def test_mm256_shuffle_pd_null_permute_works(self):
        s = Solver()

        input = ymm_reg("ymm0")
        output_vector = _mm256_shuffle_pd(input, input, null_shuffle_pd_avx2_imm8)
        s.add(output_vector != input)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_shuffle_pd_null_permute_found(self):
        s = Solver()

        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        imm8 = BitVec("imm8", 8)
        output = _mm256_shuffle_pd(input, input, imm8)

        s.add(output == input)
        result = s.check()

        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_pd_avx2_imm8, (
            f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_pd_avx2_imm8:02x}"
        )

    def test_mm256_shuffle_pd_null_permute_2vec_works(self):
        s = Solver()

        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=64)
        output = _mm256_shuffle_pd(op1, op2, null_shuffle_pd_avx2_imm8)
        expected = construct_ymm_reg_from_elements(
            64, [(op1, 0), (op2, 1), (op1, 2), (op2, 3)]
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_shuffle_pd_null_permute_2vec_found(self):
        s = Solver()

        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=64)
        imm8 = BitVec("imm8", 8)
        output = _mm256_shuffle_pd(op1, op2, imm8)
        expected = construct_ymm_reg_from_elements(
            64, [(op1, 0), (op2, 1), (op1, 2), (op2, 3)]
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_pd_avx2_imm8, (
            f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_pd_avx2_imm8:02x}"
        )

    def test_mm512_shuffle_pd_null_permute_works(self):
        s = Solver()

        input = zmm_reg("zmm0")
        output_vector = _mm512_shuffle_pd(input, input, null_shuffle_pd_avx512_imm8)

        s.add(output_vector != input)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_shuffle_pd_null_permute_found(self):
        s = Solver()

        input = zmm_reg_with_unique_values("zmm0", s, bits=64)
        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_pd(input, input, imm8)

        s.add(output == input)
        result = s.check()

        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_pd_avx512_imm8, (
            f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_pd_avx512_imm8:02x}"
        )

    def test_mm512_shuffle_pd_null_permute_2vec_works(self):
        s = Solver()

        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=64)

        output = _mm512_shuffle_pd(op1, op2, null_shuffle_pd_avx512_imm8)

        expected = construct_zmm_reg_from_elements(
            64,
            [
                (op1, 0),
                (op2, 1),
                (op1, 2),
                (op2, 3),
                (op1, 4),
                (op2, 5),
                (op1, 6),
                (op2, 7),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_shuffle_pd_null_permute_2vec_found(self):
        s = Solver()

        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=64)

        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_pd(op1, op2, imm8)

        expected = construct_zmm_reg_from_elements(
            64,
            [
                (op1, 0),
                (op2, 1),
                (op1, 2),
                (op2, 3),
                (op1, 4),
                (op2, 5),
                (op1, 6),
                (op2, 7),
            ],
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_pd_avx512_imm8, (
            f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_pd_avx512_imm8:02x}"
        )


class TestPermute2x128Si256:
    """Tests for _mm256_permute2x128_si256 (256-bit only)"""

    def test_mm256_permute2x128_si256_null_permute_works(self):
        s = Solver()

        input_vector = ymm_reg("ymm0")
        output_vector = _mm256_permute2x128_si256(
            input_vector, input_vector, null_permute2x128_imm8
        )

        s.add(input_vector != output_vector)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute2x128_si256_null_permute_found(self):
        s = Solver()

        input_vector = ymm_reg_with_unique_values("ymm0", s, bits=128)
        imm8 = BitVec("imm8", 8)
        output = _mm256_permute2x128_si256(input_vector, input_vector, imm8)

        s.add((imm8 & 0x88) == 0)  # No zero flags set

        s.add(input_vector == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_imm8 = s.model().evaluate(imm8).as_long()

        # When a==b, multiple identity permutations are valid (without zero flags):
        # 0x10: low=a[127:0], high=a[255:128]
        # 0x12: low=b[127:0], high=a[255:128] (same as 0x10 when a==b)
        # 0x30: low=a[127:0], high=b[255:128] (same as 0x10 when a==b)
        # 0x32: low=b[127:0], high=b[255:128] (same as 0x10 when a==b)
        valid_identity_permutes = {0x10, 0x12, 0x30, 0x32}
        assert model_imm8 in valid_identity_permutes, (
            f"Z3 found invalid null permute: got 0x{model_imm8:02x}, expected one of {[hex(x) for x in valid_identity_permutes]}"
        )

    def test_mm256_permute2x128_si256_null_permute_2vec_works(self):
        s = Solver()

        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=128)

        output = _mm256_permute2x128_si256(op1, op2, null_permute2x128_imm8)

        expected = construct_ymm_reg_from_elements(
            128,
            [
                (op1, 0),  # op1[127:0] -> low lane
                (op1, 1),  # op1[255:128] -> high lane
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute2x128_si256_null_permute_2vec_found(self):
        s = Solver()

        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=128)

        imm8 = BitVec("imm8", 8)
        output = _mm256_permute2x128_si256(op1, op2, imm8)

        s.add((imm8 & 0x88) == 0)  # No zero flags set

        expected = construct_ymm_reg_from_elements(
            128,
            [
                (op1, 0),  # op1[127:0] -> low lane
                (op1, 1),  # op1[255:128] -> high lane
            ],
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_permute2x128_imm8, (
            f"Z3 found unexpected null permute: got 0x{model_imm8:02x}, expected 0x{null_permute2x128_imm8:02x}"
        )

    def test_mm256_permute2x128_si256_swap_lanes(self):
        s = Solver()

        input_vector = ymm_reg_with_unique_values("ymm0", s, bits=128)

        swap_imm8 = 0x01
        output = _mm256_permute2x128_si256(input_vector, input_vector, swap_imm8)

        expected = construct_ymm_reg_from_elements(
            128,
            [
                (input_vector, 1),  # Was high lane (a[255:128]), now low
                (input_vector, 0),  # Was low lane (a[127:0]), now high
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where lane swap failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute2x128_si256_cross_vector(self):
        s = Solver()

        a, b = ymm_reg_pair_with_unique_values("input", s, bits=128)

        cross_imm8 = 0x23
        output = _mm256_permute2x128_si256(a, b, cross_imm8)

        expected = construct_ymm_reg_from_elements(
            128,
            [
                (b, 1),  # b[255:128] -> low lane
                (b, 0),  # b[127:0] -> high lane
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where cross-vector permute failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute2x128_si256_zero_lanes(self):
        s = Solver()

        input_vector = ymm_reg_with_unique_values("ymm0", s, bits=128)

        zero_high_imm8 = 0x80
        output = _mm256_permute2x128_si256(input_vector, input_vector, zero_high_imm8)

        low_lane = Extract(127, 0, input_vector)
        high_lane = BitVecVal(0, 128)
        expected = Concat(high_lane, low_lane)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where zero lane failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute2x128_si256_zero_both_lanes(self):
        s = Solver()

        input_vector = ymm_reg("ymm0")

        zero_both_imm8 = 0x88
        output = _mm256_permute2x128_si256(input_vector, input_vector, zero_both_imm8)

        expected = BitVecVal(0, 256)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where zero both lanes failed: {s.model() if result == sat else 'No model'}"
        )


class TestShuffleI32x4:
    """Tests for _mm512_shuffle_i32x4 (512-bit only)"""

    def test_mm512_shuffle_i32x4_null_permute_works(self):
        s = Solver()

        input_vector = zmm_reg("zmm0")
        output_vector = _mm512_shuffle_i32x4(
            input_vector, input_vector, null_shuffle_i32x4_imm8
        )

        s.add(input_vector != output_vector)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_shuffle_i32x4_null_permute_found(self):
        s = Solver()

        input_vector = zmm_reg_with_unique_values("zmm0", s, bits=128)
        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_i32x4(input_vector, input_vector, imm8)

        s.add(input_vector == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_i32x4_imm8, (
            f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_i32x4_imm8:02x}"
        )

    def test_mm512_shuffle_i32x4_null_permute_2vec_works(self):
        s = Solver()

        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=128)

        output = _mm512_shuffle_i32x4(op1, op2, null_shuffle_i32x4_imm8)

        expected = construct_zmm_reg_from_elements(
            128,
            [
                (op1, 0),  # a[127:0] -> dst[127:0]
                (op1, 1),  # a[255:128] -> dst[255:128]
                (op2, 2),  # b[383:256] -> dst[383:256]
                (op2, 3),  # b[511:384] -> dst[511:384]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_shuffle_i32x4_null_permute_2vec_found(self):
        s = Solver()

        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=128)

        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_i32x4(op1, op2, imm8)

        expected = construct_zmm_reg_from_elements(
            128,
            [
                (op1, 0),  # a[127:0] -> dst[127:0]
                (op1, 1),  # a[255:128] -> dst[255:128]
                (op2, 2),  # b[383:256] -> dst[383:256]
                (op2, 3),  # b[511:384] -> dst[511:384]
            ],
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_i32x4_imm8, (
            f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_i32x4_imm8:02x}"
        )

    def test_mm512_shuffle_i32x4_cross_lanes(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=128)

        cross_imm8 = _MM_SHUFFLE(0, 1, 2, 3)
        output = _mm512_shuffle_i32x4(a, b, cross_imm8)

        expected = construct_zmm_reg_from_elements(
            128,
            [
                (a, 3),  # a[511:384] -> dst[127:0]
                (a, 2),  # a[383:256] -> dst[255:128]
                (b, 1),  # b[255:128] -> dst[383:256]
                (b, 0),  # b[127:0] -> dst[511:384]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where cross-lane shuffle failed: {s.model() if result == sat else 'No model'}"
        )


class TestMaskPermutex2varEpi32:
    """Tests for _mm512_mask_permutex2var_ps (512-bit only)"""

    def test_mm512_mask_permutex2var_epi32_mask_all_zeros(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, null_permutex2var_vector_epi32_avx512
        )
        mask = BitVecVal(0, 16)
        output = _mm512_mask_permutex2var_epi32(a, mask, indices, b)

        s.add(a != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where mask all zeros failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi32_mask_all_ones(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, null_permutex2var_vector_epi32_avx512
        )
        mask = BitVecVal(0xFFFF, 16)

        masked_output = _mm512_mask_permutex2var_epi32(a, mask, indices, b)
        unmasked_output = _mm512_permutex2var_epi32(a, indices, b)

        s.add(masked_output != unmasked_output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where mask all ones failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi32_alternating_mask(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        select_b_indices = [(1 << 4) | i for i in range(16)]
        indices = zmm_reg_with_32b_values("indices", s, select_b_indices)
        mask = BitVecVal(0x5555, 16)

        output = _mm512_mask_permutex2var_epi32(a, mask, indices, b)

        expected_specs = []
        expected_specs = [(b, i) if i % 2 == 0 else (a, i) for i in range(16)]

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where alternating mask failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi32_reverse_with_partial_mask(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        reverse_a_indices = [(0 << 4) | (15 - i) for i in range(16)]
        indices = zmm_reg_with_32b_values("indices", s, reverse_a_indices)
        mask = BitVecVal(0x00FF, 16)

        output = _mm512_mask_permutex2var_epi32(a, mask, indices, b)

        expected_specs = []
        for i in range(16):
            if i < 8:
                expected_specs.append((a, 15 - i))
            else:
                expected_specs.append((a, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where reverse with partial mask failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi32_mixed_sources_with_mask(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        mixed_indices = []
        for i in range(16):
            if i % 2 == 0:
                mixed_indices.append((0 << 4) | i)
            else:
                mixed_indices.append((1 << 4) | i)

        indices = zmm_reg_with_32b_values("indices", s, mixed_indices)
        mask = BitVecVal(0x5555, 16)
        output = _mm512_mask_permutex2var_epi32(a, mask, indices, b)

        expected_specs = [(a, i) for i in range(16)]
        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where mixed sources with mask failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi32_single_bit_mask(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values("indices", s, [(1 << 4) | 10] * 16)
        mask = BitVecVal(1 << 5, 16)
        output = _mm512_mask_permutex2var_epi32(a, mask, indices, b)

        expected_specs = []
        for i in range(16):
            if i == 5:
                expected_specs.append((b, 10))
            else:
                expected_specs.append((a, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample where single bit mask failed: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi32_find_identity_mask(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, [(1 << 4) | 7] * 16
        )  # All select b[7]
        mask = BitVec("mask", 16)
        output = _mm512_mask_permutex2var_epi32(a, mask, indices, b)

        s.add(output == a)
        result = s.check()

        assert result == sat, "Z3 failed to find a mask for identity"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0, (
            f"Z3 found unexpected mask for identity: got 0x{model_mask:04x}, expected 0x0000"
        )

    def test_mm512_mask_permutex2var_epi32_find_full_permute_mask(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, [(1 << 4) | i for i in range(16)]
        )
        mask = BitVec("mask", 16)
        output = _mm512_mask_permutex2var_epi32(a, mask, indices, b)

        s.add(output == b)
        result = s.check()

        assert result == sat, "Z3 failed to find a mask for full permutation"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0xFFFF, (
            f"Z3 found unexpected mask for full permutation: got 0x{model_mask:04x}, expected 0xFFFF"
        )

    def test_mm512_mask_permutex2var_epi32_find_partial_mask(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values(
            "indices", s, [(1 << 4) | i for i in range(16)]
        )
        mask = BitVec("mask", 16)
        output = _mm512_mask_permutex2var_epi32(a, mask, indices, b)

        expected_specs = []
        for i in range(16):
            if i < 4:
                expected_specs.append((b, i))
            else:
                expected_specs.append((a, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find a mask for partial permutation"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0x000F, (
            f"Z3 found unexpected mask for partial permutation: got 0x{model_mask:04x}, expected 0x000F"
        )

    def test_mm512_mask_permutex2var_epi32_find_indices_with_mask(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        mask = BitVecVal(0x5555, 16)
        indices = zmm_reg("indices")
        output = _mm512_mask_permutex2var_epi32(a, mask, indices, b)

        expected_specs = []
        for i in range(16):
            if i % 2 == 0:
                expected_specs.append((b, 0))  # Want b[0] in even positions
            else:
                expected_specs.append((a, i))  # Original a[i] in odd positions

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output == expected)
        result = s.check()
        assert result == sat, "Z3 failed to find indices for target pattern"
        model_indices = s.model().evaluate(indices).as_long()

        # Extract and check some index values
        # For even positions, should have: source_selector=1 (b), offset=0
        # We'll check position 0: should be (1 << 4) | 0 = 16
        pos0_index = (model_indices >> (0 * 32)) & 0x1F  # Extract 5 bits for position 0
        assert pos0_index == 16, (
            f"Position 0 index should be 16 (select b[0]), got {pos0_index}"
        )

    def test_mm512_mask_permutex2var_epi32_find_reverse_partial(self):
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        mask = BitVec("mask", 16)
        indices = zmm_reg("indices")
        output = _mm512_mask_permutex2var_epi32(a, mask, indices, b)

        expected_specs = []
        for i in range(16):
            if i < 8:
                expected_specs.append((a, 7 - i))  # Reverse: a[7], a[6], ..., a[0]
            else:
                expected_specs.append((a, i))  # Unchanged: a[8], a[9], ..., a[15]

        expected = construct_zmm_reg_from_elements(32, expected_specs)
        s.add(output == expected)
        result = s.check()
        assert result == sat, "Z3 failed to find mask+indices for partial reverse"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0x00FF, (
            f"Expected mask 0x00FF for first 8 elements, got 0x{model_mask:04x}"
        )


class TestMaskPermutex2varEpi64:
    """Tests for _mm512_mask_permutex2var_pd (512-bit masked variant for 64-bit)"""

    def test_mm512_mask_permutex2var_epi64_mask_all_zeros(self):
        """Test with mask all zeros (should preserve a)"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, null_permutex2var_vector_epi64_avx512
        )
        mask = BitVecVal(0, 8)
        output = _mm512_mask_permutex2var_epi64(a, mask, indices, b)

        s.add(a != output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all zeros: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi64_mask_all_ones(self):
        """Test with mask all ones (should equal unmasked)"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, null_permutex2var_vector_epi64_avx512
        )
        mask = BitVecVal(0xFF, 8)

        masked_output = _mm512_mask_permutex2var_epi64(a, mask, indices, b)
        unmasked_output = _mm512_permutex2var_epi64(a, indices, b)

        s.add(masked_output != unmasked_output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all ones: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi64_alternating_mask(self):
        """Test with alternating mask pattern"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        select_b_indices = [(1 << 3) | i for i in range(8)]
        indices = zmm_reg_with_64b_values("indices", s, select_b_indices)
        mask = BitVecVal(0x55, 8)  # 01010101

        output = _mm512_mask_permutex2var_epi64(a, mask, indices, b)
        unmasked = _mm512_permutex2var_epi64(a, indices, b)

        # Expected: unmasked result in even positions, a in odd positions
        expected_specs = []
        for i in range(8):
            if i % 2 == 0:
                expected_specs.append((unmasked, i))
            else:
                expected_specs.append((a, i))

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for alternating mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi64_single_bit_mask(self):
        """Test with only one bit set in mask"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        indices = zmm_reg_with_64b_values("indices", s, [(1 << 3) | 5] * 8)
        mask = BitVecVal(1 << 3, 8)  # Only bit 3
        output = _mm512_mask_permutex2var_epi64(a, mask, indices, b)

        expected_specs = []
        for i in range(8):
            if i == 3:
                expected_specs.append((b, 5))
            else:
                expected_specs.append((a, i))

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for single bit mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi64_partial_mask(self):
        """Test with lower half masked"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        reverse_a_indices = [(0 << 3) | (7 - i) for i in range(8)]
        indices = zmm_reg_with_64b_values("indices", s, reverse_a_indices)
        mask = BitVecVal(0x0F, 8)  # Lower 4 bits set

        output = _mm512_mask_permutex2var_epi64(a, mask, indices, b)
        reversed_a = zmm_reg_reversed("a_reversed", s, a, bits=64)

        # Expected: reversed a in positions 0-3, original a in positions 4-7
        expected_specs = []
        for i in range(8):
            if i < 4:
                expected_specs.append((reversed_a, i))
            else:
                expected_specs.append((a, i))

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for partial mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi64_mixed_sources_with_mask(self):
        """Test with mixed sources and selective masking"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        mixed_indices = []
        for i in range(8):
            if i % 2 == 0:
                mixed_indices.append((0 << 3) | i)
            else:
                mixed_indices.append((1 << 3) | i)

        indices = zmm_reg_with_64b_values("indices", s, mixed_indices)
        mask = BitVecVal(0x55, 8)  # 01010101
        output = _mm512_mask_permutex2var_epi64(a, mask, indices, b)

        expected_specs = [(a, i) for i in range(8)]
        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mixed sources with mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutex2var_epi64_find_identity_mask(self):
        """Test that Z3 can find mask to preserve a (mask all zeros)"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        indices = zmm_reg_with_64b_values("indices", s, [(1 << 3) | 7] * 8)
        mask = BitVec("mask", 8)
        output = _mm512_mask_permutex2var_epi64(a, mask, indices, b)

        s.add(output == a)
        result = s.check()

        assert result == sat, "Z3 failed to find mask for identity"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0, (
            f"Z3 found unexpected mask for identity: got 0x{model_mask:02x}, expected 0x00"
        )

    def test_mm512_mask_permutex2var_epi64_find_full_permute_mask(self):
        """Test that Z3 can find mask for full permutation (mask all ones)"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, [(1 << 3) | i for i in range(8)]
        )
        mask = BitVec("mask", 8)
        output = _mm512_mask_permutex2var_epi64(a, mask, indices, b)

        s.add(output == b)
        result = s.check()

        assert result == sat, "Z3 failed to find mask for full permutation"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0xFF, (
            f"Z3 found unexpected mask for full permutation: got 0x{model_mask:02x}, expected 0xFF"
        )

    def test_mm512_mask_permutex2var_epi64_find_partial_mask(self):
        """Test that Z3 can find mask for partial permutation"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        indices = zmm_reg_with_64b_values(
            "indices", s, [(1 << 3) | i for i in range(8)]
        )
        mask = BitVec("mask", 8)
        output = _mm512_mask_permutex2var_epi64(a, mask, indices, b)

        expected_specs = []
        for i in range(8):
            if i < 3:
                expected_specs.append((b, i))
            else:
                expected_specs.append((a, i))

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find mask for partial permutation"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0x07, (
            f"Z3 found unexpected mask for partial permutation: got 0x{model_mask:02x}, expected 0x07"
        )

    def test_mm512_mask_permutex2var_epi64_find_indices_with_mask(self):
        """Test that Z3 can find indices to achieve pattern with fixed mask"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        mask = BitVecVal(0x55, 8)  # 01010101
        indices = zmm_reg("indices")
        output = _mm512_mask_permutex2var_epi64(a, mask, indices, b)

        expected_specs = []
        for i in range(8):
            if i % 2 == 0:
                expected_specs.append((b, 0))  # Want b[0] in even positions
            else:
                expected_specs.append((a, i))  # Original a[i] in odd positions

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output == expected)
        result = s.check()
        assert result == sat, "Z3 failed to find indices for target pattern"
        model_indices = s.model().evaluate(indices).as_long()

        # For even positions, should have: source_selector=1 (b), offset=0
        # Check position 0: should be (1 << 3) | 0 = 8
        pos0_index = (model_indices >> (0 * 64)) & 0xF  # Extract 4 bits for position 0
        assert pos0_index == 8, (
            f"Position 0 index should be 8 (select b[0]), got {pos0_index}"
        )

    def test_mm512_mask_permutex2var_epi64_cross_source_reverse(self):
        """Test reversing elements with cross-source selection"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)

        # Create indices that reverse and alternate between sources
        # Position 0 gets b[7] (source=1, offset=7), position 1 gets a[6] (source=0, offset=6), etc.
        cross_reverse_indices = []
        for i in range(8):
            offset = 7 - i
            # When i is even, select from b (source=1); when odd, select from a (source=0)
            source = 1 if i % 2 == 0 else 0
            cross_reverse_indices.append((source << 3) | offset)

        indices = zmm_reg_with_64b_values("indices", s, cross_reverse_indices)
        mask = BitVecVal(0xFF, 8)  # All bits set
        output = _mm512_mask_permutex2var_epi64(a, mask, indices, b)

        expected_specs = []
        for i in range(8):
            offset = 7 - i
            if i % 2 == 0:
                expected_specs.append((b, offset))
            else:
                expected_specs.append((a, offset))

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for cross-source reverse: {s.model() if result == sat else 'No model'}"
        )


class TestUnpackEpi32:
    """Tests for unpack 32-bit integer instructions"""

    def test_mm256_unpacklo_epi32_basic(self):
        """Test _mm256_unpacklo_epi32 with known values"""
        s = Solver()

        # Create test inputs with unique values per lane
        # a = [a0, a1, a2, a3 | a4, a5, a6, a7]
        # b = [b0, b1, b2, b3 | b4, b5, b6, b7]
        a = ymm_reg_with_32b_values(
            "a", s, [0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7]
        )
        b = ymm_reg_with_32b_values(
            "b", s, [0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7]
        )

        output = _mm256_unpacklo_epi32(a, b)

        # Expected: [a0, b0, a1, b1 | a4, b4, a5, b5] (low elements from each lane)
        expected = construct_ymm_reg_from_elements(
            32,
            [
                (a, 0),
                (b, 0),
                (a, 1),
                (b, 1),  # Lane 0: interleave a[0,1] with b[0,1]
                (a, 4),
                (b, 4),
                (a, 5),
                (b, 5),  # Lane 1: interleave a[4,5] with b[4,5]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for unpacklo: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_unpackhi_epi32_basic(self):
        """Test _mm256_unpackhi_epi32 with known values"""
        s = Solver()

        # Create test inputs with unique values per lane
        a = ymm_reg_with_32b_values(
            "a", s, [0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7]
        )
        b = ymm_reg_with_32b_values(
            "b", s, [0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7]
        )

        output = _mm256_unpackhi_epi32(a, b)

        # Expected: [a2, b2, a3, b3 | a6, b6, a7, b7] (high elements from each lane)
        expected = construct_ymm_reg_from_elements(
            32,
            [
                (a, 2),
                (b, 2),
                (a, 3),
                (b, 3),  # Lane 0: interleave a[2,3] with b[2,3]
                (a, 6),
                (b, 6),
                (a, 7),
                (b, 7),  # Lane 1: interleave a[6,7] with b[6,7]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for unpackhi: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_unpacklo_epi32_basic(self):
        """Test _mm512_unpacklo_epi32 with known values"""
        s = Solver()

        # Create test inputs with unique values
        a_vals = [
            0xA0,
            0xA1,
            0xA2,
            0xA3,
            0xA4,
            0xA5,
            0xA6,
            0xA7,
            0xA8,
            0xA9,
            0xAA,
            0xAB,
            0xAC,
            0xAD,
            0xAE,
            0xAF,
        ]
        b_vals = [
            0xB0,
            0xB1,
            0xB2,
            0xB3,
            0xB4,
            0xB5,
            0xB6,
            0xB7,
            0xB8,
            0xB9,
            0xBA,
            0xBB,
            0xBC,
            0xBD,
            0xBE,
            0xBF,
        ]

        a = zmm_reg_with_32b_values("a", s, a_vals)
        b = zmm_reg_with_32b_values("b", s, b_vals)

        output = _mm512_unpacklo_epi32(a, b)

        # Expected: interleave low elements from each 128-bit lane
        expected = construct_zmm_reg_from_elements(
            32,
            [
                (a, 0),
                (b, 0),
                (a, 1),
                (b, 1),  # Lane 0
                (a, 4),
                (b, 4),
                (a, 5),
                (b, 5),  # Lane 1
                (a, 8),
                (b, 8),
                (a, 9),
                (b, 9),  # Lane 2
                (a, 12),
                (b, 12),
                (a, 13),
                (b, 13),  # Lane 3
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 512-bit unpacklo: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_unpackhi_epi32_basic(self):
        """Test _mm512_unpackhi_epi32 with known values"""
        s = Solver()

        # Create test inputs with unique values
        a_vals = [
            0xA0,
            0xA1,
            0xA2,
            0xA3,
            0xA4,
            0xA5,
            0xA6,
            0xA7,
            0xA8,
            0xA9,
            0xAA,
            0xAB,
            0xAC,
            0xAD,
            0xAE,
            0xAF,
        ]
        b_vals = [
            0xB0,
            0xB1,
            0xB2,
            0xB3,
            0xB4,
            0xB5,
            0xB6,
            0xB7,
            0xB8,
            0xB9,
            0xBA,
            0xBB,
            0xBC,
            0xBD,
            0xBE,
            0xBF,
        ]

        a = zmm_reg_with_32b_values("a", s, a_vals)
        b = zmm_reg_with_32b_values("b", s, b_vals)

        output = _mm512_unpackhi_epi32(a, b)

        # Expected: interleave high elements from each 128-bit lane
        expected = construct_zmm_reg_from_elements(
            32,
            [
                (a, 2),
                (b, 2),
                (a, 3),
                (b, 3),  # Lane 0
                (a, 6),
                (b, 6),
                (a, 7),
                (b, 7),  # Lane 1
                (a, 10),
                (b, 10),
                (a, 11),
                (b, 11),  # Lane 2
                (a, 14),
                (b, 14),
                (a, 15),
                (b, 15),  # Lane 3
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 512-bit unpackhi: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_unpacklo_epi32_identity_check(self):
        """Test that _mm256_unpacklo_epi32 with identical inputs gives expected pattern"""
        s = Solver()

        input_reg = ymm_reg_with_unique_values("input", s, bits=32)
        output = _mm256_unpacklo_epi32(input_reg, input_reg)

        # When a == b, unpacklo should give [a0, a0, a1, a1 | a4, a4, a5, a5]
        expected = construct_ymm_reg_from_elements(
            32,
            [
                (input_reg, 0),
                (input_reg, 0),
                (input_reg, 1),
                (input_reg, 1),
                (input_reg, 4),
                (input_reg, 4),
                (input_reg, 5),
                (input_reg, 5),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for identity unpacklo: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_unpackhi_epi32_identity_check(self):
        """Test that _mm256_unpackhi_epi32 with identical inputs gives expected pattern"""
        s = Solver()

        input_reg = ymm_reg_with_unique_values("input", s, bits=32)
        output = _mm256_unpackhi_epi32(input_reg, input_reg)

        # When a == b, unpackhi should give [a2, a2, a3, a3 | a6, a6, a7, a7]
        expected = construct_ymm_reg_from_elements(
            32,
            [
                (input_reg, 2),
                (input_reg, 2),
                (input_reg, 3),
                (input_reg, 3),
                (input_reg, 6),
                (input_reg, 6),
                (input_reg, 7),
                (input_reg, 7),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for identity unpackhi: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_unpacklo_epi32_mask_all_zeros(self):
        """Test _mm512_mask_unpacklo_epi32 with mask all zeros (should preserve src)"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        src = zmm_reg_with_unique_values("src", s, bits=32)
        mask = BitVecVal(0, 16)  # All mask bits are 0

        output = _mm512_mask_unpacklo_epi32(src, mask, a, b)

        s.add(output != src)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all zeros: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_unpacklo_epi32_mask_all_ones(self):
        """Test _mm512_mask_unpacklo_epi32 with mask all ones (should equal unmasked)"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        src = zmm_reg_with_unique_values("src", s, bits=32)
        mask = BitVecVal(0xFFFF, 16)  # All mask bits are 1

        masked_output = _mm512_mask_unpacklo_epi32(src, mask, a, b)
        unmasked_output = _mm512_unpacklo_epi32(a, b)

        s.add(masked_output != unmasked_output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all ones: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_unpackhi_epi32_alternating_mask(self):
        """Test _mm512_mask_unpackhi_epi32 with alternating mask pattern"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        src = zmm_reg_with_unique_values("src", s, bits=32)
        mask = BitVecVal(0x5555, 16)  # 0101010101010101 in binary

        output = _mm512_mask_unpackhi_epi32(src, mask, a, b)

        # Expected: unpack result in even positions, src in odd positions
        unpack_result = _mm512_unpackhi_epi32(a, b)
        expected_specs = []
        for i in range(16):
            if i % 2 == 0:
                # Even position: use unpack result
                expected_specs.append((unpack_result, i))
            else:
                # Odd position: use src
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for alternating mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_unpacklo_epi32_single_bit_mask(self):
        """Test _mm512_mask_unpacklo_epi32 with only one bit set in mask"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        src = zmm_reg_with_unique_values("src", s, bits=32)
        mask = BitVecVal(1 << 3, 16)  # Only bit 3 is set

        output = _mm512_mask_unpacklo_epi32(src, mask, a, b)

        # Expected: unpack result only at position 3, src everywhere else
        unpack_result = _mm512_unpacklo_epi32(a, b)
        expected_specs = []
        for i in range(16):
            if i == 3:
                expected_specs.append((unpack_result, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for single bit mask: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_unpacklo_epi32_reconstruct_pattern(self):
        """Test that Z3 can find inputs that produce a specific output pattern"""
        s = Solver()

        a = ymm_reg("a")
        b = ymm_reg("b")
        output = _mm256_unpacklo_epi32(a, b)

        # Specify a target pattern: all elements should be the same value
        target_value = BitVecVal(0x12345678, 32)
        for i in range(8):
            element = Extract(i * 32 + 31, i * 32, output)
            s.add(element == target_value)

        result = s.check()
        assert result == sat, "Z3 should be able to find inputs for constant output"

        # Verify that the inputs produce the expected pattern
        model = s.model()
        model_a = model.evaluate(a).as_long()
        model_b = model.evaluate(b).as_long()

        # Extract some elements from the inputs
        a_elem0 = (model_a >> (0 * 32)) & 0xFFFFFFFF
        a_elem1 = (model_a >> (1 * 32)) & 0xFFFFFFFF
        b_elem0 = (model_b >> (0 * 32)) & 0xFFFFFFFF
        b_elem1 = (model_b >> (1 * 32)) & 0xFFFFFFFF

        # For constant output, we expect the input elements to all equal the target
        assert a_elem0 == 0x12345678, f"Expected a[0] = 0x12345678, got 0x{a_elem0:08x}"
        assert a_elem1 == 0x12345678, f"Expected a[1] = 0x12345678, got 0x{a_elem1:08x}"
        assert b_elem0 == 0x12345678, f"Expected b[0] = 0x12345678, got 0x{b_elem0:08x}"
        assert b_elem1 == 0x12345678, f"Expected b[1] = 0x12345678, got 0x{b_elem1:08x}"

    def test_mm512_mask_unpackhi_epi32_find_mask(self):
        """Test that Z3 can find the correct mask to achieve a specific pattern"""
        s = Solver()

        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        src = zmm_reg_with_unique_values("src", s, bits=32)
        mask = BitVec("mask", 16)

        output = _mm512_mask_unpackhi_epi32(src, mask, a, b)

        # We want: first 4 elements from unpack result, rest from src
        unpack_result = _mm512_unpackhi_epi32(a, b)
        expected_specs = []
        for i in range(16):
            if i < 4:
                expected_specs.append((unpack_result, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 should find a mask for the target pattern"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0x000F, (
            f"Expected mask 0x000F (first 4 bits), got 0x{model_mask:04x}"
        )

    def test_mm256_unpack_combo_lo_hi(self):
        """Test combining unpacklo and unpackhi operations"""
        s = Solver()

        a, b = ymm_reg_pair_with_unique_values("input", s, bits=32)

        lo_result = _mm256_unpacklo_epi32(a, b)
        hi_result = _mm256_unpackhi_epi32(a, b)

        # The lo and hi results should be different (unless inputs have a very specific pattern)
        s.add(lo_result == hi_result)
        result = s.check()

        # This should be satisfiable only in special cases (when certain elements are equal)
        if result == sat:
            # If it's satisfiable, verify that the pattern makes sense
            model = s.model()
            model_a = model.evaluate(a).as_long()
            model_b = model.evaluate(b).as_long()

            # Extract elements to understand the pattern
            a_elems = [(model_a >> (i * 32)) & 0xFFFFFFFF for i in range(8)]
            b_elems = [(model_b >> (i * 32)) & 0xFFFFFFFF for i in range(8)]

            # For lo == hi, we need specific relationships between elements
            # This is a complex condition, so we just verify that Z3 found a valid solution
            print(f"Found pattern where lo == hi: a={a_elems}, b={b_elems}")

    def test_mm512_unpack_lane_independence(self):
        """Test that unpack operations work independently on each 128-bit lane"""
        s = Solver()

        # Create inputs where each 128-bit lane has distinct patterns
        a_vals = [
            0x10,
            0x11,
            0x12,
            0x13,  # Lane 0
            0x20,
            0x21,
            0x22,
            0x23,  # Lane 1
            0x30,
            0x31,
            0x32,
            0x33,  # Lane 2
            0x40,
            0x41,
            0x42,
            0x43,
        ]  # Lane 3
        b_vals = [
            0x50,
            0x51,
            0x52,
            0x53,  # Lane 0
            0x60,
            0x61,
            0x62,
            0x63,  # Lane 1
            0x70,
            0x71,
            0x72,
            0x73,  # Lane 2
            0x80,
            0x81,
            0x82,
            0x83,
        ]  # Lane 3

        a = zmm_reg_with_32b_values("a", s, a_vals)
        b = zmm_reg_with_32b_values("b", s, b_vals)

        lo_result = _mm512_unpacklo_epi32(a, b)

        # Verify each lane is processed independently
        # Lane 0 should produce: [0x10, 0x50, 0x11, 0x51]
        # Lane 1 should produce: [0x20, 0x60, 0x21, 0x61]
        # etc.
        expected = construct_zmm_reg_from_elements(
            32,
            [
                (a, 0),
                (b, 0),
                (a, 1),
                (b, 1),  # Lane 0: 0x10, 0x50, 0x11, 0x51
                (a, 4),
                (b, 4),
                (a, 5),
                (b, 5),  # Lane 1: 0x20, 0x60, 0x21, 0x61
                (a, 8),
                (b, 8),
                (a, 9),
                (b, 9),  # Lane 2: 0x30, 0x70, 0x31, 0x71
                (a, 12),
                (b, 12),
                (a, 13),
                (b, 13),  # Lane 3: 0x40, 0x80, 0x41, 0x81
            ],
        )

        s.add(lo_result != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for lane independence: {s.model() if result == sat else 'No model'}"
        )


class TestMaskPermutePs:
    """Tests for _mm512_mask_permute_ps"""

    def test_mm512_mask_permute_ps_mask_all_zeros(self):
        """Test with mask all zeros (should preserve src)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        mask = BitVecVal(0, 16)

        output = _mm512_mask_permute_ps(src, mask, a, null_permute_epi32_imm8)

        s.add(output != src)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all zeros: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permute_ps_mask_all_ones(self):
        """Test with mask all ones (should equal unmasked)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        mask = BitVecVal(0xFFFF, 16)

        masked_output = _mm512_mask_permute_ps(src, mask, a, null_permute_epi32_imm8)
        unmasked_output = _mm512_permute_ps(a, null_permute_epi32_imm8)

        s.add(masked_output != unmasked_output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all ones: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permute_ps_alternating_mask(self):
        """Test with alternating mask pattern"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        mask = BitVecVal(0x5555, 16)  # Alternating: 0101010101010101
        imm8 = _MM_SHUFFLE(0, 1, 2, 3)  # Reverse within lanes

        output = _mm512_mask_permute_ps(src, mask, a, imm8)
        unmasked = _mm512_permute_ps(a, imm8)

        # Expected: unmasked result in even positions, src in odd positions
        expected_specs = []
        for i in range(16):
            if i % 2 == 0:
                expected_specs.append((unmasked, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for alternating mask: {s.model() if result == sat else 'No model'}"
        )


class TestMaskPermutePd:
    """Tests for _mm512_mask_permute_pd"""

    def test_mm512_mask_permute_pd_mask_all_zeros(self):
        """Test with mask all zeros (should preserve src)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        mask = BitVecVal(0, 8)

        output = _mm512_mask_permute_pd(src, mask, a, null_permute_pd_imm8)

        s.add(output != src)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all zeros: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permute_pd_mask_all_ones(self):
        """Test with mask all ones (should equal unmasked)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        mask = BitVecVal(0xFF, 8)

        masked_output = _mm512_mask_permute_pd(src, mask, a, null_permute_pd_imm8)
        unmasked_output = _mm512_permute_pd(a, null_permute_pd_imm8)

        s.add(masked_output != unmasked_output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all ones: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permute_pd_single_bit_mask(self):
        """Test with only one bit set in mask"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        mask = BitVecVal(1 << 3, 8)  # Only bit 3
        imm8 = _MM_SHUFFLE2(0, 1)  # Swap within lanes

        output = _mm512_mask_permute_pd(src, mask, a, imm8)
        unmasked = _mm512_permute_pd(a, imm8)

        # Expected: unmasked result only at position 3, src everywhere else
        expected_specs = []
        for i in range(8):
            if i == 3:
                expected_specs.append((unmasked, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for single bit mask: {s.model() if result == sat else 'No model'}"
        )


class TestMaskShufflePs:
    """Tests for _mm512_mask_shuffle_ps"""

    def test_mm512_mask_shuffle_ps_mask_all_zeros(self):
        """Test with mask all zeros (should preserve src)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        mask = BitVecVal(0, 16)

        output = _mm512_mask_shuffle_ps(src, mask, a, b, null_shuffle_ps_2vec_imm8)

        s.add(output != src)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all zeros: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_shuffle_ps_mask_all_ones(self):
        """Test with mask all ones (should equal unmasked)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        mask = BitVecVal(0xFFFF, 16)

        masked_output = _mm512_mask_shuffle_ps(
            src, mask, a, b, null_shuffle_ps_2vec_imm8
        )
        unmasked_output = _mm512_shuffle_ps(a, b, null_shuffle_ps_2vec_imm8)

        s.add(masked_output != unmasked_output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all ones: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_shuffle_ps_partial_mask(self):
        """Test with partial mask (lower half only)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        mask = BitVecVal(0x00FF, 16)  # Lower 8 bits set

        output = _mm512_mask_shuffle_ps(src, mask, a, b, null_shuffle_ps_2vec_imm8)
        unmasked = _mm512_shuffle_ps(a, b, null_shuffle_ps_2vec_imm8)

        # Expected: unmasked result in positions 0-7, src in positions 8-15
        expected_specs = []
        for i in range(16):
            if i < 8:
                expected_specs.append((unmasked, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(32, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for partial mask: {s.model() if result == sat else 'No model'}"
        )


class TestMaskShufflePd:
    """Tests for _mm512_mask_shuffle_pd"""

    def test_mm512_mask_shuffle_pd_mask_all_zeros(self):
        """Test with mask all zeros (should preserve src)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        mask = BitVecVal(0, 8)

        output = _mm512_mask_shuffle_pd(src, mask, a, b, null_shuffle_pd_avx512_imm8)

        s.add(output != src)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all zeros: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_shuffle_pd_mask_all_ones(self):
        """Test with mask all ones (should equal unmasked)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        mask = BitVecVal(0xFF, 8)

        masked_output = _mm512_mask_shuffle_pd(
            src, mask, a, b, null_shuffle_pd_avx512_imm8
        )
        unmasked_output = _mm512_shuffle_pd(a, b, null_shuffle_pd_avx512_imm8)

        s.add(masked_output != unmasked_output)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all ones: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_shuffle_pd_alternating_mask(self):
        """Test with alternating mask pattern"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        mask = BitVecVal(0x55, 8)  # 01010101

        output = _mm512_mask_shuffle_pd(src, mask, a, b, null_shuffle_pd_avx512_imm8)
        unmasked = _mm512_shuffle_pd(a, b, null_shuffle_pd_avx512_imm8)

        # Expected: unmasked result in even positions, src in odd positions
        expected_specs = []
        for i in range(8):
            if i % 2 == 0:
                expected_specs.append((unmasked, i))
            else:
                expected_specs.append((src, i))

        expected = construct_zmm_reg_from_elements(64, expected_specs)

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for alternating mask: {s.model() if result == sat else 'No model'}"
        )


class TestMaskPermutevarPs:
    """Tests for _mm512_mask_permutevar_ps"""

    def test_mm512_mask_permutevar_ps_mask_all_zeros(self):
        """Test with mask all zeros (should preserve src)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        # Create control vector for identity permute within lanes
        ctrl = zmm_reg_with_32b_values("ctrl", s, [i % 4 for i in range(16)])
        mask = BitVecVal(0, 16)

        output = _mm512_mask_permutevar_ps(src, mask, a, ctrl)

        s.add(output != src)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all zeros: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutevar_ps_identity_permute(self):
        """Test identity permutation within lanes"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        # Create control vector: each element selects itself within its lane
        # Lane 0: [0, 1, 2, 3], Lane 1: [0, 1, 2, 3], etc.
        ctrl = zmm_reg_with_32b_values("ctrl", s, [i % 4 for i in range(16)])
        mask = BitVecVal(0xFFFF, 16)

        output = _mm512_mask_permutevar_ps(src, mask, a, ctrl)

        s.add(output != a)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for identity permute: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutevar_ps_reverse_within_lanes(self):
        """Test reversing elements within each 128-bit lane"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        # Create control vector: reverse within each lane [3, 2, 1, 0, 3, 2, 1, 0, ...]
        ctrl = zmm_reg_with_32b_values("ctrl", s, [3 - (i % 4) for i in range(16)])
        mask = BitVecVal(0xFFFF, 16)

        output = _mm512_mask_permutevar_ps(src, mask, a, ctrl)

        # Expected: each 128-bit lane is reversed
        expected = construct_zmm_reg_from_elements(
            32,
            [
                (a, 3),
                (a, 2),
                (a, 1),
                (a, 0),  # Lane 0 reversed
                (a, 7),
                (a, 6),
                (a, 5),
                (a, 4),  # Lane 1 reversed
                (a, 11),
                (a, 10),
                (a, 9),
                (a, 8),  # Lane 2 reversed
                (a, 15),
                (a, 14),
                (a, 13),
                (a, 12),  # Lane 3 reversed
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for reverse within lanes: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutevar_ps_broadcast_within_lanes(self):
        """Test broadcasting first element within each lane"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=32)
        a = zmm_reg_with_unique_values("a", s, bits=32)
        # Create control vector: all zeros (broadcast element 0 of each lane)
        ctrl = zmm_reg_with_32b_values("ctrl", s, [0] * 16)
        mask = BitVecVal(0xFFFF, 16)

        output = _mm512_mask_permutevar_ps(src, mask, a, ctrl)

        # Expected: first element of each lane broadcast to all positions in that lane
        expected = construct_zmm_reg_from_elements(
            32,
            [
                (a, 0),
                (a, 0),
                (a, 0),
                (a, 0),  # Lane 0: all a[0]
                (a, 4),
                (a, 4),
                (a, 4),
                (a, 4),  # Lane 1: all a[4]
                (a, 8),
                (a, 8),
                (a, 8),
                (a, 8),  # Lane 2: all a[8]
                (a, 12),
                (a, 12),
                (a, 12),
                (a, 12),  # Lane 3: all a[12]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for broadcast within lanes: {s.model() if result == sat else 'No model'}"
        )


class TestMaskPermutevarPd:
    """Tests for _mm512_mask_permutevar_pd"""

    def test_mm512_mask_permutevar_pd_mask_all_zeros(self):
        """Test with mask all zeros (should preserve src)"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        # Create control vector for identity permute (bits at positions 1, 65, 129, 193, 257, 321, 385, 449 = 0)
        ctrl = zmm_reg("ctrl")
        mask = BitVecVal(0, 8)

        output = _mm512_mask_permutevar_pd(src, mask, a, ctrl)

        s.add(output != src)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for mask all zeros: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutevar_pd_identity_permute(self):
        """Test identity permutation within lanes"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        # Create control vector with bits at correct positions set to 0 for identity
        # Positions: [1, 65, 129, 193, 257, 321, 385, 449] should be [0, 1, 0, 1, 0, 1, 0, 1]
        ctrl = zmm_reg("ctrl")
        # Set control bits: element j%2 of each lane
        s.add(Extract(1, 1, ctrl) == 0)  # Element 0 selects from position 0
        s.add(Extract(65, 65, ctrl) == 1)  # Element 1 selects from position 1
        s.add(Extract(129, 129, ctrl) == 0)  # Element 2 selects from position 0
        s.add(Extract(193, 193, ctrl) == 1)  # Element 3 selects from position 1
        s.add(Extract(257, 257, ctrl) == 0)  # Element 4 selects from position 0
        s.add(Extract(321, 321, ctrl) == 1)  # Element 5 selects from position 1
        s.add(Extract(385, 385, ctrl) == 0)  # Element 6 selects from position 0
        s.add(Extract(449, 449, ctrl) == 1)  # Element 7 selects from position 1
        mask = BitVecVal(0xFF, 8)

        output = _mm512_mask_permutevar_pd(src, mask, a, ctrl)

        s.add(output != a)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for identity permute: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutevar_pd_swap_within_lanes(self):
        """Test swapping elements within each 128-bit lane"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        # Create control vector: swap within each lane
        ctrl = zmm_reg("ctrl")
        # Set control bits to swap: [1, 0, 1, 0, 1, 0, 1, 0]
        s.add(Extract(1, 1, ctrl) == 1)  # Element 0 selects from position 1
        s.add(Extract(65, 65, ctrl) == 0)  # Element 1 selects from position 0
        s.add(Extract(129, 129, ctrl) == 1)  # Element 2 selects from position 1
        s.add(Extract(193, 193, ctrl) == 0)  # Element 3 selects from position 0
        s.add(Extract(257, 257, ctrl) == 1)  # Element 4 selects from position 1
        s.add(Extract(321, 321, ctrl) == 0)  # Element 5 selects from position 0
        s.add(Extract(385, 385, ctrl) == 1)  # Element 6 selects from position 1
        s.add(Extract(449, 449, ctrl) == 0)  # Element 7 selects from position 0
        mask = BitVecVal(0xFF, 8)

        output = _mm512_mask_permutevar_pd(src, mask, a, ctrl)

        # Expected: each pair within 128-bit lanes is swapped
        expected = construct_zmm_reg_from_elements(
            64,
            [
                (a, 1),
                (a, 0),  # Lane 0 swapped
                (a, 3),
                (a, 2),  # Lane 1 swapped
                (a, 5),
                (a, 4),  # Lane 2 swapped
                (a, 7),
                (a, 6),  # Lane 3 swapped
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for swap within lanes: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_mask_permutevar_pd_broadcast_within_lanes(self):
        """Test broadcasting first element within each lane"""
        s = Solver()

        src = zmm_reg_with_unique_values("src", s, bits=64)
        a = zmm_reg_with_unique_values("a", s, bits=64)
        # Create control vector: all control bits = 0 (broadcast element 0 of each lane)
        ctrl = zmm_reg("ctrl")
        s.add(Extract(1, 1, ctrl) == 0)
        s.add(Extract(65, 65, ctrl) == 0)
        s.add(Extract(129, 129, ctrl) == 0)
        s.add(Extract(193, 193, ctrl) == 0)
        s.add(Extract(257, 257, ctrl) == 0)
        s.add(Extract(321, 321, ctrl) == 0)
        s.add(Extract(385, 385, ctrl) == 0)
        s.add(Extract(449, 449, ctrl) == 0)
        mask = BitVecVal(0xFF, 8)

        output = _mm512_mask_permutevar_pd(src, mask, a, ctrl)

        # Expected: first element of each lane broadcast
        expected = construct_zmm_reg_from_elements(
            64,
            [
                (a, 0),
                (a, 0),  # Lane 0: both a[0]
                (a, 2),
                (a, 2),  # Lane 1: both a[2]
                (a, 4),
                (a, 4),  # Lane 2: both a[4]
                (a, 6),
                (a, 6),  # Lane 3: both a[6]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for broadcast within lanes: {s.model() if result == sat else 'No model'}"
        )


class TestPermutevarPs:
    """Tests for _mm256_permutevar_ps and _mm512_permutevar_ps (non-masked variants)"""

    def test_mm256_permutevar_ps_identity_permute(self):
        """Test identity permutation within lanes for 256-bit"""
        s = Solver()

        a = ymm_reg_with_unique_values("a", s, bits=32)
        # Create control vector: each element selects itself within its lane
        # Lane 0: [0, 1, 2, 3], Lane 1: [0, 1, 2, 3]
        ctrl = ymm_reg_with_32b_values("ctrl", s, [i % 4 for i in range(8)])

        output = _mm256_permutevar_ps(a, ctrl)

        s.add(output != a)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 256-bit identity permute: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permutevar_ps_reverse_within_lanes(self):
        """Test reversing elements within each 128-bit lane for 256-bit"""
        s = Solver()

        a = ymm_reg_with_unique_values("a", s, bits=32)
        # Create control vector: reverse within each lane [3, 2, 1, 0, 3, 2, 1, 0]
        ctrl = ymm_reg_with_32b_values("ctrl", s, [3 - (i % 4) for i in range(8)])

        output = _mm256_permutevar_ps(a, ctrl)

        # Expected: each 128-bit lane is reversed
        expected = construct_ymm_reg_from_elements(
            32,
            [
                (a, 3),
                (a, 2),
                (a, 1),
                (a, 0),  # Lane 0 reversed
                (a, 7),
                (a, 6),
                (a, 5),
                (a, 4),  # Lane 1 reversed
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 256-bit reverse within lanes: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permutevar_ps_broadcast_within_lanes(self):
        """Test broadcasting first element within each lane for 256-bit"""
        s = Solver()

        a = ymm_reg_with_unique_values("a", s, bits=32)
        # Create control vector: all zeros (broadcast element 0 of each lane)
        ctrl = ymm_reg_with_32b_values("ctrl", s, [0] * 8)

        output = _mm256_permutevar_ps(a, ctrl)

        # Expected: first element of each lane broadcast to all positions in that lane
        expected = construct_ymm_reg_from_elements(
            32,
            [
                (a, 0),
                (a, 0),
                (a, 0),
                (a, 0),  # Lane 0: all a[0]
                (a, 4),
                (a, 4),
                (a, 4),
                (a, 4),  # Lane 1: all a[4]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 256-bit broadcast within lanes: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permutevar_ps_mixed_permute(self):
        """Test mixed permutation pattern for 256-bit"""
        s = Solver()

        a = ymm_reg_with_unique_values("a", s, bits=32)
        # Create control vector: [1, 0, 3, 2, 2, 3, 0, 1]
        ctrl = ymm_reg_with_32b_values("ctrl", s, [1, 0, 3, 2, 2, 3, 0, 1])

        output = _mm256_permutevar_ps(a, ctrl)

        # Expected: permuted according to control vector
        expected = construct_ymm_reg_from_elements(
            32,
            [
                (a, 1),  # Lane 0[0] = a[1]
                (a, 0),  # Lane 0[1] = a[0]
                (a, 3),  # Lane 0[2] = a[3]
                (a, 2),  # Lane 0[3] = a[2]
                (a, 6),  # Lane 1[0] = a[6] (4+2)
                (a, 7),  # Lane 1[1] = a[7] (4+3)
                (a, 4),  # Lane 1[2] = a[4] (4+0)
                (a, 5),  # Lane 1[3] = a[5] (4+1)
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 256-bit mixed permute: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutevar_ps_identity_permute(self):
        """Test identity permutation within lanes for 512-bit"""
        s = Solver()

        a = zmm_reg_with_unique_values("a", s, bits=32)
        # Create control vector: each element selects itself within its lane
        ctrl = zmm_reg_with_32b_values("ctrl", s, [i % 4 for i in range(16)])

        output = _mm512_permutevar_ps(a, ctrl)

        s.add(output != a)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 512-bit identity permute: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutevar_ps_reverse_within_lanes(self):
        """Test reversing elements within each 128-bit lane for 512-bit"""
        s = Solver()

        a = zmm_reg_with_unique_values("a", s, bits=32)
        # Create control vector: reverse within each lane [3, 2, 1, 0, ...]
        ctrl = zmm_reg_with_32b_values("ctrl", s, [3 - (i % 4) for i in range(16)])

        output = _mm512_permutevar_ps(a, ctrl)

        # Expected: each 128-bit lane is reversed
        expected = construct_zmm_reg_from_elements(
            32,
            [
                (a, 3),
                (a, 2),
                (a, 1),
                (a, 0),  # Lane 0 reversed
                (a, 7),
                (a, 6),
                (a, 5),
                (a, 4),  # Lane 1 reversed
                (a, 11),
                (a, 10),
                (a, 9),
                (a, 8),  # Lane 2 reversed
                (a, 15),
                (a, 14),
                (a, 13),
                (a, 12),  # Lane 3 reversed
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 512-bit reverse within lanes: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutevar_ps_broadcast_within_lanes(self):
        """Test broadcasting last element within each lane for 512-bit"""
        s = Solver()

        a = zmm_reg_with_unique_values("a", s, bits=32)
        # Create control vector: all 3s (broadcast element 3 of each lane)
        ctrl = zmm_reg_with_32b_values("ctrl", s, [3] * 16)

        output = _mm512_permutevar_ps(a, ctrl)

        # Expected: last element of each lane broadcast to all positions in that lane
        expected = construct_zmm_reg_from_elements(
            32,
            [
                (a, 3),
                (a, 3),
                (a, 3),
                (a, 3),  # Lane 0: all a[3]
                (a, 7),
                (a, 7),
                (a, 7),
                (a, 7),  # Lane 1: all a[7]
                (a, 11),
                (a, 11),
                (a, 11),
                (a, 11),  # Lane 2: all a[11]
                (a, 15),
                (a, 15),
                (a, 15),
                (a, 15),  # Lane 3: all a[15]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 512-bit broadcast within lanes: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutevar_ps_alternating_pattern(self):
        """Test alternating permutation pattern for 512-bit"""
        s = Solver()

        a = zmm_reg_with_unique_values("a", s, bits=32)
        # Create control vector: alternating [0, 2, 0, 2, ...]
        ctrl = zmm_reg_with_32b_values(
            "ctrl", s, [0 if i % 2 == 0 else 2 for i in range(16)]
        )

        output = _mm512_permutevar_ps(a, ctrl)

        # Expected: alternating between element 0 and 2 of each lane
        expected = construct_zmm_reg_from_elements(
            32,
            [
                (a, 0),
                (a, 2),
                (a, 0),
                (a, 2),  # Lane 0
                (a, 4),
                (a, 6),
                (a, 4),
                (a, 6),  # Lane 1
                (a, 8),
                (a, 10),
                (a, 8),
                (a, 10),  # Lane 2
                (a, 12),
                (a, 14),
                (a, 12),
                (a, 14),  # Lane 3
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 512-bit alternating pattern: {s.model() if result == sat else 'No model'}"
        )


class TestPermutevarPd:
    """Tests for _mm256_permutevar_pd and _mm512_permutevar_pd (non-masked variants)"""

    def test_mm256_permutevar_pd_identity_permute(self):
        """Test identity permutation within lanes for 256-bit"""
        s = Solver()

        a = ymm_reg_with_unique_values("a", s, bits=64)
        # Create control vector with bits at correct positions set for identity [0, 1, 0, 1]
        ctrl = ymm_reg("ctrl")
        s.add(Extract(1, 1, ctrl) == 0)  # Element 0 selects from position 0
        s.add(Extract(65, 65, ctrl) == 1)  # Element 1 selects from position 1
        s.add(Extract(129, 129, ctrl) == 0)  # Element 2 selects from position 0
        s.add(Extract(193, 193, ctrl) == 1)  # Element 3 selects from position 1

        output = _mm256_permutevar_pd(a, ctrl)

        s.add(output != a)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 256-bit identity permute: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permutevar_pd_swap_within_lanes(self):
        """Test swapping elements within each 128-bit lane for 256-bit"""
        s = Solver()

        a = ymm_reg_with_unique_values("a", s, bits=64)
        # Create control vector: swap within each lane [1, 0, 1, 0]
        ctrl = ymm_reg("ctrl")
        s.add(Extract(1, 1, ctrl) == 1)  # Element 0 selects from position 1
        s.add(Extract(65, 65, ctrl) == 0)  # Element 1 selects from position 0
        s.add(Extract(129, 129, ctrl) == 1)  # Element 2 selects from position 1
        s.add(Extract(193, 193, ctrl) == 0)  # Element 3 selects from position 0

        output = _mm256_permutevar_pd(a, ctrl)

        # Expected: each pair within 128-bit lanes is swapped
        expected = construct_ymm_reg_from_elements(
            64,
            [
                (a, 1),
                (a, 0),  # Lane 0 swapped
                (a, 3),
                (a, 2),  # Lane 1 swapped
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 256-bit swap within lanes: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permutevar_pd_broadcast_first_within_lanes(self):
        """Test broadcasting first element within each lane for 256-bit"""
        s = Solver()

        a = ymm_reg_with_unique_values("a", s, bits=64)
        # Create control vector: all control bits = 0 (broadcast element 0 of each lane)
        ctrl = ymm_reg("ctrl")
        s.add(Extract(1, 1, ctrl) == 0)
        s.add(Extract(65, 65, ctrl) == 0)
        s.add(Extract(129, 129, ctrl) == 0)
        s.add(Extract(193, 193, ctrl) == 0)

        output = _mm256_permutevar_pd(a, ctrl)

        # Expected: first element of each lane broadcast
        expected = construct_ymm_reg_from_elements(
            64,
            [
                (a, 0),
                (a, 0),  # Lane 0: both a[0]
                (a, 2),
                (a, 2),  # Lane 1: both a[2]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 256-bit broadcast first within lanes: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permutevar_pd_broadcast_second_within_lanes(self):
        """Test broadcasting second element within each lane for 256-bit"""
        s = Solver()

        a = ymm_reg_with_unique_values("a", s, bits=64)
        # Create control vector: all control bits = 1 (broadcast element 1 of each lane)
        ctrl = ymm_reg("ctrl")
        s.add(Extract(1, 1, ctrl) == 1)
        s.add(Extract(65, 65, ctrl) == 1)
        s.add(Extract(129, 129, ctrl) == 1)
        s.add(Extract(193, 193, ctrl) == 1)

        output = _mm256_permutevar_pd(a, ctrl)

        # Expected: second element of each lane broadcast
        expected = construct_ymm_reg_from_elements(
            64,
            [
                (a, 1),
                (a, 1),  # Lane 0: both a[1]
                (a, 3),
                (a, 3),  # Lane 1: both a[3]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 256-bit broadcast second within lanes: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutevar_pd_identity_permute(self):
        """Test identity permutation within lanes for 512-bit"""
        s = Solver()

        a = zmm_reg_with_unique_values("a", s, bits=64)
        # Create control vector with bits at correct positions set for identity
        ctrl = zmm_reg("ctrl")
        s.add(Extract(1, 1, ctrl) == 0)  # Element 0 selects from position 0
        s.add(Extract(65, 65, ctrl) == 1)  # Element 1 selects from position 1
        s.add(Extract(129, 129, ctrl) == 0)  # Element 2 selects from position 0
        s.add(Extract(193, 193, ctrl) == 1)  # Element 3 selects from position 1
        s.add(Extract(257, 257, ctrl) == 0)  # Element 4 selects from position 0
        s.add(Extract(321, 321, ctrl) == 1)  # Element 5 selects from position 1
        s.add(Extract(385, 385, ctrl) == 0)  # Element 6 selects from position 0
        s.add(Extract(449, 449, ctrl) == 1)  # Element 7 selects from position 1

        output = _mm512_permutevar_pd(a, ctrl)

        s.add(output != a)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 512-bit identity permute: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutevar_pd_swap_within_lanes(self):
        """Test swapping elements within each 128-bit lane for 512-bit"""
        s = Solver()

        a = zmm_reg_with_unique_values("a", s, bits=64)
        # Create control vector: swap within each lane [1, 0, 1, 0, 1, 0, 1, 0]
        ctrl = zmm_reg("ctrl")
        s.add(Extract(1, 1, ctrl) == 1)  # Element 0 selects from position 1
        s.add(Extract(65, 65, ctrl) == 0)  # Element 1 selects from position 0
        s.add(Extract(129, 129, ctrl) == 1)  # Element 2 selects from position 1
        s.add(Extract(193, 193, ctrl) == 0)  # Element 3 selects from position 0
        s.add(Extract(257, 257, ctrl) == 1)  # Element 4 selects from position 1
        s.add(Extract(321, 321, ctrl) == 0)  # Element 5 selects from position 0
        s.add(Extract(385, 385, ctrl) == 1)  # Element 6 selects from position 1
        s.add(Extract(449, 449, ctrl) == 0)  # Element 7 selects from position 0

        output = _mm512_permutevar_pd(a, ctrl)

        # Expected: each pair within 128-bit lanes is swapped
        expected = construct_zmm_reg_from_elements(
            64,
            [
                (a, 1),
                (a, 0),  # Lane 0 swapped
                (a, 3),
                (a, 2),  # Lane 1 swapped
                (a, 5),
                (a, 4),  # Lane 2 swapped
                (a, 7),
                (a, 6),  # Lane 3 swapped
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 512-bit swap within lanes: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutevar_pd_broadcast_first_within_lanes(self):
        """Test broadcasting first element within each lane for 512-bit"""
        s = Solver()

        a = zmm_reg_with_unique_values("a", s, bits=64)
        # Create control vector: all control bits = 0 (broadcast element 0 of each lane)
        ctrl = zmm_reg("ctrl")
        s.add(Extract(1, 1, ctrl) == 0)
        s.add(Extract(65, 65, ctrl) == 0)
        s.add(Extract(129, 129, ctrl) == 0)
        s.add(Extract(193, 193, ctrl) == 0)
        s.add(Extract(257, 257, ctrl) == 0)
        s.add(Extract(321, 321, ctrl) == 0)
        s.add(Extract(385, 385, ctrl) == 0)
        s.add(Extract(449, 449, ctrl) == 0)

        output = _mm512_permutevar_pd(a, ctrl)

        # Expected: first element of each lane broadcast
        expected = construct_zmm_reg_from_elements(
            64,
            [
                (a, 0),
                (a, 0),  # Lane 0: both a[0]
                (a, 2),
                (a, 2),  # Lane 1: both a[2]
                (a, 4),
                (a, 4),  # Lane 2: both a[4]
                (a, 6),
                (a, 6),  # Lane 3: both a[6]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 512-bit broadcast first within lanes: {s.model() if result == sat else 'No model'}"
        )

    def test_mm512_permutevar_pd_broadcast_second_within_lanes(self):
        """Test broadcasting second element within each lane for 512-bit"""
        s = Solver()

        a = zmm_reg_with_unique_values("a", s, bits=64)
        # Create control vector: all control bits = 1 (broadcast element 1 of each lane)
        ctrl = zmm_reg("ctrl")
        s.add(Extract(1, 1, ctrl) == 1)
        s.add(Extract(65, 65, ctrl) == 1)
        s.add(Extract(129, 129, ctrl) == 1)
        s.add(Extract(193, 193, ctrl) == 1)
        s.add(Extract(257, 257, ctrl) == 1)
        s.add(Extract(321, 321, ctrl) == 1)
        s.add(Extract(385, 385, ctrl) == 1)
        s.add(Extract(449, 449, ctrl) == 1)

        output = _mm512_permutevar_pd(a, ctrl)

        # Expected: second element of each lane broadcast
        expected = construct_zmm_reg_from_elements(
            64,
            [
                (a, 1),
                (a, 1),  # Lane 0: both a[1]
                (a, 3),
                (a, 3),  # Lane 1: both a[3]
                (a, 5),
                (a, 5),  # Lane 2: both a[5]
                (a, 7),
                (a, 7),  # Lane 3: both a[7]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for 512-bit broadcast second within lanes: {s.model() if result == sat else 'No model'}"
        )


class TestBlendPd:
    """Tests for _mm256_blend_pd (immediate blend for double-precision)"""

    def test_mm256_blend_pd_all_from_a(self):
        """Test blend_pd with all elements from a (imm8 = 0b0000)"""
        s = Solver()
        a = ymm_reg("a")
        b = ymm_reg("b")
        imm8 = 0b0000  # All bits 0: select all from a

        output = _mm256_blend_pd(a, b, imm8)

        # Output should equal a
        s.add(output != a)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for all-from-a blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blend_pd_all_from_b(self):
        """Test blend_pd with all elements from b (imm8 = 0b1111)"""
        s = Solver()
        a = ymm_reg("a")
        b = ymm_reg("b")
        imm8 = 0b1111  # All bits 1: select all from b

        output = _mm256_blend_pd(a, b, imm8)

        # Output should equal b
        s.add(output != b)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for all-from-b blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blend_pd_alternating(self):
        """Test blend_pd with alternating pattern (imm8 = 0b1010)"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=64)
        imm8 = 0b1010  # Pattern: b, a, b, a (from element 0 to 3)

        output = _mm256_blend_pd(a, b, imm8)

        # Expected: elements 0,2 from a; elements 1,3 from b
        expected = construct_ymm_reg_from_elements(
            64,
            [
                (a, 0),  # bit 0 = 0
                (b, 1),  # bit 1 = 1
                (a, 2),  # bit 2 = 0
                (b, 3),  # bit 3 = 1
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for alternating blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blend_pd_first_two_from_b(self):
        """Test blend_pd with first two elements from b (imm8 = 0b0011)"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=64)
        imm8 = 0b0011  # First two from b, last two from a

        output = _mm256_blend_pd(a, b, imm8)

        expected = construct_ymm_reg_from_elements(
            64,
            [
                (b, 0),  # bit 0 = 1
                (b, 1),  # bit 1 = 1
                (a, 2),  # bit 2 = 0
                (a, 3),  # bit 3 = 0
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for first-two-from-b blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blend_pd_symbolic_mask(self):
        """Test that Z3 can find the correct mask to produce a specific blend"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=64)
        imm8 = BitVec("imm8", 8)

        output = _mm256_blend_pd(a, b, imm8)

        # Want: [a[0], b[1], a[2], b[3]]
        expected = construct_ymm_reg_from_elements(
            64,
            [
                (a, 0),
                (b, 1),
                (a, 2),
                (b, 3),
            ],
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find blend mask"
        model_imm8 = s.model().evaluate(imm8).as_long()
        expected_mask = 0b1010
        assert (model_imm8 & 0xF) == expected_mask, (
            f"Z3 found unexpected mask: got 0x{model_imm8:02x}, expected 0x{expected_mask:02x}"
        )


class TestBlendPs:
    """Tests for _mm256_blend_ps (immediate blend for single-precision)"""

    def test_mm256_blend_ps_all_from_a(self):
        """Test blend_ps with all elements from a (imm8 = 0b00000000)"""
        s = Solver()
        a = ymm_reg("a")
        b = ymm_reg("b")
        imm8 = 0b00000000  # All bits 0: select all from a

        output = _mm256_blend_ps(a, b, imm8)

        # Output should equal a
        s.add(output != a)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for all-from-a blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blend_ps_all_from_b(self):
        """Test blend_ps with all elements from b (imm8 = 0b11111111)"""
        s = Solver()
        a = ymm_reg("a")
        b = ymm_reg("b")
        imm8 = 0b11111111  # All bits 1: select all from b

        output = _mm256_blend_ps(a, b, imm8)

        # Output should equal b
        s.add(output != b)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for all-from-b blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blend_ps_alternating(self):
        """Test blend_ps with alternating pattern (imm8 = 0b10101010)"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=32)
        imm8 = 0b10101010  # Pattern: a, b, a, b, a, b, a, b

        output = _mm256_blend_ps(a, b, imm8)

        # Expected: even indices from a, odd indices from b
        expected = construct_ymm_reg_from_elements(
            32,
            [
                (a, 0),  # bit 0 = 0
                (b, 1),  # bit 1 = 1
                (a, 2),  # bit 2 = 0
                (b, 3),  # bit 3 = 1
                (a, 4),  # bit 4 = 0
                (b, 5),  # bit 5 = 1
                (a, 6),  # bit 6 = 0
                (b, 7),  # bit 7 = 1
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for alternating blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blend_ps_first_four_from_b(self):
        """Test blend_ps with first four elements from b (imm8 = 0b00001111)"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=32)
        imm8 = 0b00001111  # First four from b, last four from a

        output = _mm256_blend_ps(a, b, imm8)

        expected = construct_ymm_reg_from_elements(
            32,
            [
                (b, 0),  # bit 0 = 1
                (b, 1),  # bit 1 = 1
                (b, 2),  # bit 2 = 1
                (b, 3),  # bit 3 = 1
                (a, 4),  # bit 4 = 0
                (a, 5),  # bit 5 = 0
                (a, 6),  # bit 6 = 0
                (a, 7),  # bit 7 = 0
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for first-four-from-b blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blend_ps_symbolic_mask(self):
        """Test that Z3 can find the correct mask to produce a specific blend"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=32)
        imm8 = BitVec("imm8", 8)

        output = _mm256_blend_ps(a, b, imm8)

        # Want: [b[0], a[1], b[2], a[3], b[4], a[5], b[6], a[7]]
        expected = construct_ymm_reg_from_elements(
            32,
            [
                (b, 0),
                (a, 1),
                (b, 2),
                (a, 3),
                (b, 4),
                (a, 5),
                (b, 6),
                (a, 7),
            ],
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find blend mask"
        model_imm8 = s.model().evaluate(imm8).as_long()
        expected_mask = 0b01010101
        assert model_imm8 == expected_mask, (
            f"Z3 found unexpected mask: got 0x{model_imm8:02x}, expected 0x{expected_mask:02x}"
        )


class TestBlendvPd:
    """Tests for _mm256_blendv_pd (variable blend for double-precision)"""

    def test_mm256_blendv_pd_all_from_a(self):
        """Test blendv_pd with all sign bits 0 (select all from a)"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=64)

        # Create mask with all sign bits = 0 (all positive)
        mask = ymm_reg("mask")
        for j in range(4):
            i = j * 64
            s.add(Extract(i + 63, i + 63, mask) == 0)

        output = _mm256_blendv_pd(a, b, mask)

        # Output should equal a
        s.add(output != a)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for all-from-a blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blendv_pd_all_from_b(self):
        """Test blendv_pd with all sign bits 1 (select all from b)"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=64)

        # Create mask with all sign bits = 1 (all negative)
        mask = ymm_reg("mask")
        for j in range(4):
            i = j * 64
            s.add(Extract(i + 63, i + 63, mask) == 1)

        output = _mm256_blendv_pd(a, b, mask)

        # Output should equal b
        s.add(output != b)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for all-from-b blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blendv_pd_alternating(self):
        """Test blendv_pd with alternating sign bits"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=64)

        # Create mask with alternating sign bits: 0, 1, 0, 1
        mask = ymm_reg("mask")
        s.add(Extract(63, 63, mask) == 0)  # Element 0: from a
        s.add(Extract(127, 127, mask) == 1)  # Element 1: from b
        s.add(Extract(191, 191, mask) == 0)  # Element 2: from a
        s.add(Extract(255, 255, mask) == 1)  # Element 3: from b

        output = _mm256_blendv_pd(a, b, mask)

        expected = construct_ymm_reg_from_elements(
            64,
            [
                (a, 0),
                (b, 1),
                (a, 2),
                (b, 3),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for alternating blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blendv_pd_symbolic_mask(self):
        """Test that Z3 can find the correct mask to produce a specific blend"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=64)
        mask = ymm_reg("mask")

        output = _mm256_blendv_pd(a, b, mask)

        # Want: [b[0], b[1], a[2], a[3]]
        expected = construct_ymm_reg_from_elements(
            64,
            [
                (b, 0),
                (b, 1),
                (a, 2),
                (a, 3),
            ],
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find blend mask"
        # Verify sign bits match expected pattern
        model = s.model()
        mask_val = model.evaluate(mask)
        sign_bit_0 = model.evaluate(Extract(63, 63, mask_val)).as_long()
        sign_bit_1 = model.evaluate(Extract(127, 127, mask_val)).as_long()
        sign_bit_2 = model.evaluate(Extract(191, 191, mask_val)).as_long()
        sign_bit_3 = model.evaluate(Extract(255, 255, mask_val)).as_long()

        assert sign_bit_0 == 1, f"Expected sign bit 0 to be 1, got {sign_bit_0}"
        assert sign_bit_1 == 1, f"Expected sign bit 1 to be 1, got {sign_bit_1}"
        assert sign_bit_2 == 0, f"Expected sign bit 2 to be 0, got {sign_bit_2}"
        assert sign_bit_3 == 0, f"Expected sign bit 3 to be 0, got {sign_bit_3}"


class TestBlendvPs:
    """Tests for _mm256_blendv_ps (variable blend for single-precision)"""

    def test_mm256_blendv_ps_all_from_a(self):
        """Test blendv_ps with all sign bits 0 (select all from a)"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=32)

        # Create mask with all sign bits = 0 (all positive)
        mask = ymm_reg("mask")
        for j in range(8):
            i = j * 32
            s.add(Extract(i + 31, i + 31, mask) == 0)

        output = _mm256_blendv_ps(a, b, mask)

        # Output should equal a
        s.add(output != a)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for all-from-a blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blendv_ps_all_from_b(self):
        """Test blendv_ps with all sign bits 1 (select all from b)"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=32)

        # Create mask with all sign bits = 1 (all negative)
        mask = ymm_reg("mask")
        for j in range(8):
            i = j * 32
            s.add(Extract(i + 31, i + 31, mask) == 1)

        output = _mm256_blendv_ps(a, b, mask)

        # Output should equal b
        s.add(output != b)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for all-from-b blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blendv_ps_alternating(self):
        """Test blendv_ps with alternating sign bits"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=32)

        # Create mask with alternating sign bits: 0, 1, 0, 1, 0, 1, 0, 1
        mask = ymm_reg("mask")
        for j in range(8):
            i = j * 32
            expected_bit = j % 2
            s.add(Extract(i + 31, i + 31, mask) == expected_bit)

        output = _mm256_blendv_ps(a, b, mask)

        expected = construct_ymm_reg_from_elements(
            32,
            [
                (a, 0),
                (b, 1),
                (a, 2),
                (b, 3),
                (a, 4),
                (b, 5),
                (a, 6),
                (b, 7),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for alternating blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blendv_ps_first_four_from_b(self):
        """Test blendv_ps with first four elements from b"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=32)

        # Create mask: first four sign bits = 1, last four = 0
        mask = ymm_reg("mask")
        for j in range(8):
            i = j * 32
            expected_bit = 1 if j < 4 else 0
            s.add(Extract(i + 31, i + 31, mask) == expected_bit)

        output = _mm256_blendv_ps(a, b, mask)

        expected = construct_ymm_reg_from_elements(
            32,
            [
                (b, 0),
                (b, 1),
                (b, 2),
                (b, 3),
                (a, 4),
                (a, 5),
                (a, 6),
                (a, 7),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for first-four-from-b blend: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_blendv_ps_symbolic_mask(self):
        """Test that Z3 can find the correct mask to produce a specific blend"""
        s = Solver()
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=32)
        mask = ymm_reg("mask")

        output = _mm256_blendv_ps(a, b, mask)

        # Want: [b[0], a[1], b[2], a[3], b[4], a[5], b[6], a[7]]
        expected = construct_ymm_reg_from_elements(
            32,
            [
                (b, 0),
                (a, 1),
                (b, 2),
                (a, 3),
                (b, 4),
                (a, 5),
                (b, 6),
                (a, 7),
            ],
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find blend mask"
        # Verify sign bits match expected pattern (alternating starting with 1)
        model = s.model()
        mask_val = model.evaluate(mask)

        for j in range(8):
            i = j * 32
            sign_bit = model.evaluate(Extract(i + 31, i + 31, mask_val)).as_long()
            expected_bit = 1 if j % 2 == 0 else 0
            assert sign_bit == expected_bit, (
                f"Expected sign bit {j} to be {expected_bit}, got {sign_bit}"
            )


class TestPermute4x64Epi64:
    """Tests for _mm256_permute4x64_epi64 (cross-lane 64-bit permute)"""

    def test_mm256_permute4x64_epi64_identity(self):
        """Test identity permutation"""
        s = Solver()
        input = ymm_reg("ymm0")
        # Identity: [0, 1, 2, 3] - each 2-bit field selects its corresponding element
        imm8 = _MM_SHUFFLE(
            3, 2, 1, 0
        )  # dst[0]=src[0], dst[1]=src[1], dst[2]=src[2], dst[3]=src[3]

        output = _mm256_permute4x64_epi64(input, imm8)

        # Output should equal input
        s.add(output != input)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for identity permute: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute4x64_epi64_reverse(self):
        """Test reverse permutation"""
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        # Reverse: [3, 2, 1, 0]
        imm8 = _MM_SHUFFLE(
            0, 1, 2, 3
        )  # dst[0]=src[3], dst[1]=src[2], dst[2]=src[1], dst[3]=src[0]

        output = _mm256_permute4x64_epi64(input, imm8)

        # Create reversed input using constraints
        reversed_input = ymm_reg_reversed("ymm_reversed", s, input, bits=64)

        # Output should equal reversed input
        s.add(output != reversed_input)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for reverse permute: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute4x64_epi64_broadcast_first(self):
        """Test broadcasting first element"""
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        # Broadcast element 0: [0, 0, 0, 0]
        imm8 = _MM_SHUFFLE(0, 0, 0, 0)  # dst[0..3]=src[0]

        output = _mm256_permute4x64_epi64(input, imm8)

        # Expected: all elements should be input[0]
        expected = construct_ymm_reg_from_elements(
            64,
            [
                (input, 0),
                (input, 0),
                (input, 0),
                (input, 0),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for broadcast first: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute4x64_epi64_broadcast_last(self):
        """Test broadcasting last element"""
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        # Broadcast element 3: [3, 3, 3, 3]
        imm8 = _MM_SHUFFLE(3, 3, 3, 3)  # dst[0..3]=src[3]

        output = _mm256_permute4x64_epi64(input, imm8)

        # Expected: all elements should be input[3]
        expected = construct_ymm_reg_from_elements(
            64,
            [
                (input, 3),
                (input, 3),
                (input, 3),
                (input, 3),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for broadcast last: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute4x64_epi64_swap_pairs(self):
        """Test swapping adjacent pairs"""
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        # Swap pairs: [1, 0, 3, 2]
        imm8 = _MM_SHUFFLE(
            2, 3, 0, 1
        )  # dst[0]=src[1], dst[1]=src[0], dst[2]=src[3], dst[3]=src[2]

        output = _mm256_permute4x64_epi64(input, imm8)

        expected = construct_ymm_reg_from_elements(
            64,
            [
                (input, 1),
                (input, 0),
                (input, 3),
                (input, 2),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for swap pairs: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute4x64_epi64_swap_halves(self):
        """Test swapping halves"""
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        # Swap halves: [2, 3, 0, 1]
        imm8 = _MM_SHUFFLE(
            1, 0, 3, 2
        )  # dst[0]=src[2], dst[1]=src[3], dst[2]=src[0], dst[3]=src[1]

        output = _mm256_permute4x64_epi64(input, imm8)

        expected = construct_ymm_reg_from_elements(
            64,
            [
                (input, 2),
                (input, 3),
                (input, 0),
                (input, 1),
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for swap halves: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute4x64_epi64_custom_pattern(self):
        """Test custom pattern [1, 3, 2, 0]"""
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        # Custom pattern: [1, 3, 2, 0] (from low to high)
        imm8 = _MM_SHUFFLE(
            0, 2, 3, 1
        )  # dst[0]=src[1], dst[1]=src[3], dst[2]=src[2], dst[3]=src[0]

        output = _mm256_permute4x64_epi64(input, imm8)

        expected = construct_ymm_reg_from_elements(
            64,
            [
                (input, 1),  # dst[63:0]
                (input, 3),  # dst[127:64]
                (input, 2),  # dst[191:128]
                (input, 0),  # dst[255:192]
            ],
        )

        s.add(output != expected)
        result = s.check()
        assert result == unsat, (
            f"Z3 found a counterexample for custom pattern: {s.model() if result == sat else 'No model'}"
        )

    def test_mm256_permute4x64_epi64_symbolic_imm(self):
        """Test that Z3 can find the imm8 value to produce a specific permutation"""
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        imm8 = BitVec("imm8", 8)

        output = _mm256_permute4x64_epi64(input, imm8)

        # Want: [input[2], input[0], input[3], input[1]]
        expected = construct_ymm_reg_from_elements(
            64,
            [
                (input, 2),
                (input, 0),
                (input, 3),
                (input, 1),
            ],
        )

        s.add(output == expected)
        result = s.check()

        assert result == sat, "Z3 failed to find permute mask"
        model_imm8 = s.model().evaluate(imm8).as_long()
        # Expected imm8: [1, 3, 0, 2] = 0b01110010 = 0x72
        expected_mask = _MM_SHUFFLE(1, 3, 0, 2)
        assert model_imm8 == expected_mask, (
            f"Z3 found unexpected mask: got 0x{model_imm8:02x}, expected 0x{expected_mask:02x}"
        )


class TestAlignrEpi32:
    """Tests for _mm256_alignr_epi32 and _mm512_alignr_epi32"""

    def test_mm256_alignr_epi32_shift_zero(self):
        """Shift by 0 should return b unchanged"""
        s = Solver()
        a = ymm_reg_with_32b_values("a", s, list(range(10, 18)))
        b = ymm_reg_with_32b_values("b", s, list(range(8)))

        output = _mm256_alignr_epi32(a, b, 0)
        expected = ymm_reg_with_32b_values("expected", s, list(range(8)))

        s.add(output == expected)
        assert s.check() == sat

    def test_mm256_alignr_epi32_shift_one(self):
        """Shift by 1 should shift one element from b to a"""
        s = Solver()
        a = ymm_reg_with_32b_values("a", s, list(range(10, 18)))
        b = ymm_reg_with_32b_values("b", s, list(range(8)))

        # Concatenated: [10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7]
        # Shift right by 1: [1, 2, 3, 4, 5, 6, 7, 10, ...]
        # Take low 8: [1, 2, 3, 4, 5, 6, 7, 10]
        output = _mm256_alignr_epi32(a, b, 1)
        expected = ymm_reg_with_32b_values("expected", s, list(range(1, 8)) + [10])

        s.add(output == expected)
        assert s.check() == sat

    def test_mm256_alignr_epi32_shift_seven(self):
        """Shift by 7 (max for 3 bits) should get mostly from a"""
        s = Solver()
        a = ymm_reg_with_32b_values("a", s, list(range(10, 18)))
        b = ymm_reg_with_32b_values("b", s, list(range(8)))

        # Concatenated: [10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7]
        # Shift right by 7: [7, 10, 11, 12, 13, 14, 15, 16, ...]
        # Take low 8: [7, 10, 11, 12, 13, 14, 15, 16]
        output = _mm256_alignr_epi32(a, b, 7)
        expected = ymm_reg_with_32b_values("expected", s, [7] + list(range(10, 17)))

        s.add(output == expected)
        assert s.check() == sat

    def test_mm256_alignr_epi32_shift_four(self):
        """Shift by 4 should get half from each"""
        s = Solver()
        a = ymm_reg_with_32b_values("a", s, list(range(10, 18)))
        b = ymm_reg_with_32b_values("b", s, list(range(8)))

        # Concatenated: [10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7]
        # Shift right by 4: [4, 5, 6, 7, 10, 11, 12, 13, ...]
        # Take low 8: [4, 5, 6, 7, 10, 11, 12, 13]
        output = _mm256_alignr_epi32(a, b, 4)
        expected = ymm_reg_with_32b_values(
            "expected", s, list(range(4, 8)) + list(range(10, 14))
        )

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_alignr_epi32_shift_zero(self):
        """Shift by 0 should return b unchanged"""
        s = Solver()
        a = zmm_reg_with_32b_values("a", s, list(range(20, 36)))
        b = zmm_reg_with_32b_values("b", s, list(range(16)))

        output = _mm512_alignr_epi32(a, b, 0)
        expected = zmm_reg_with_32b_values("expected", s, list(range(16)))

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_alignr_epi32_shift_fifteen(self):
        """Shift by 15 (max for 4 bits) should get mostly from a"""
        s = Solver()
        a = zmm_reg_with_32b_values("a", s, list(range(20, 36)))
        b = zmm_reg_with_32b_values("b", s, list(range(16)))

        # Shift right by 15: [15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        output = _mm512_alignr_epi32(a, b, 15)
        expected = zmm_reg_with_32b_values("expected", s, [15] + list(range(20, 35)))

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_alignr_epi32_shift_four(self):
        """Shift by 3 elements"""
        s = Solver()
        a = zmm_reg_with_32b_values("a", s, [i for i in range(16, 32)])
        b = zmm_reg_with_32b_values("b", s, [i for i in range(16)])

        output = _mm512_alignr_epi32(a, b, 4)
        expected = zmm_reg_with_32b_values("expected", s, [i for i in range(4, 20)])

        s.add(output == expected)
        assert s.check() == sat

    def test_mm256_alignr_epi32_find_shift(self):
        """Use Z3 to find the shift amount"""
        s = Solver()
        a = ymm_reg_with_32b_values("a", s, list(range(10, 18)))
        b = ymm_reg_with_32b_values("b", s, list(range(8)))
        imm8 = BitVec("imm8", 8)

        output = _mm256_alignr_epi32(a, b, imm8)
        expected = ymm_reg_with_32b_values(
            "expected", s, list(range(2, 8)) + list(range(10, 12))
        )

        s.add(output == expected)
        assert s.check() == sat
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == 2


class TestAlignrEpi64:
    """Tests for _mm256_alignr_epi64 and _mm512_alignr_epi64"""

    def test_mm256_alignr_epi64_shift_zero(self):
        """Shift by 0 should return b unchanged"""
        s = Solver()
        a = ymm_reg_with_64b_values("a", s, list(range(100, 104)))
        b = ymm_reg_with_64b_values("b", s, list(range(4)))

        output = _mm256_alignr_epi64(a, b, 0)
        expected = ymm_reg_with_64b_values("expected", s, list(range(4)))

        s.add(output == expected)
        assert s.check() == sat

    def test_mm256_alignr_epi64_shift_one(self):
        """Shift by 1 element"""
        s = Solver()
        a = ymm_reg_with_64b_values("a", s, list(range(100, 104)))
        b = ymm_reg_with_64b_values("b", s, list(range(4)))

        # Concatenated: [100, 101, 102, 103, 0, 1, 2, 3]
        # Shift right by 1: [1, 2, 3, 100, ...]
        # Take low 4: [1, 2, 3, 100]
        output = _mm256_alignr_epi64(a, b, 1)
        expected = ymm_reg_with_64b_values("expected", s, list(range(1, 4)) + [100])

        s.add(output == expected)
        assert s.check() == sat

    def test_mm256_alignr_epi64_shift_two(self):
        """Shift by 2 should get half from each"""
        s = Solver()
        a = ymm_reg_with_64b_values("a", s, list(range(100, 104)))
        b = ymm_reg_with_64b_values("b", s, list(range(4)))

        output = _mm256_alignr_epi64(a, b, 2)
        expected = ymm_reg_with_64b_values(
            "expected", s, list(range(2, 4)) + list(range(100, 102))
        )

        s.add(output == expected)
        assert s.check() == sat

    def test_mm256_alignr_epi64_shift_three(self):
        """Shift by 3 (max for 2 bits) should get mostly from a"""
        s = Solver()
        a = ymm_reg_with_64b_values("a", s, list(range(100, 104)))
        b = ymm_reg_with_64b_values("b", s, list(range(4)))

        # Shift right by 3: [3, 100, 101, 102]
        output = _mm256_alignr_epi64(a, b, 3)
        expected = ymm_reg_with_64b_values("expected", s, [3] + list(range(100, 103)))

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_alignr_epi64_shift_zero(self):
        """Shift by 0 should return b unchanged"""
        s = Solver()
        a = zmm_reg_with_64b_values("a", s, list(range(200, 208)))
        b = zmm_reg_with_64b_values("b", s, list(range(8)))

        output = _mm512_alignr_epi64(a, b, 0)
        expected = zmm_reg_with_64b_values("expected", s, list(range(8)))

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_alignr_epi64_shift_seven(self):
        """Shift by 7 (max for 3 bits) should get mostly from a"""
        s = Solver()
        a = zmm_reg_with_64b_values("a", s, list(range(200, 208)))
        b = zmm_reg_with_64b_values("b", s, list(range(8)))

        # Shift right by 7: [7, 200, 201, 202, 203, 204, 205, 206]
        output = _mm512_alignr_epi64(a, b, 7)
        expected = zmm_reg_with_64b_values("expected", s, [7] + list(range(200, 207)))

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_alignr_epi64_shift_four(self):
        """Shift by 4 should get half from each"""
        s = Solver()
        a = zmm_reg_with_64b_values("a", s, list(range(200, 208)))
        b = zmm_reg_with_64b_values("b", s, list(range(8)))

        output = _mm512_alignr_epi64(a, b, 4)
        expected = zmm_reg_with_64b_values(
            "expected", s, list(range(4, 8)) + list(range(200, 204))
        )

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_alignr_epi64_shift_three(self):
        """Shift by 3 elements"""
        s = Solver()
        a = zmm_reg_with_64b_values("a", s, list(range(200, 208)))
        b = zmm_reg_with_64b_values("b", s, list(range(8)))

        output = _mm512_alignr_epi64(a, b, 3)
        expected = zmm_reg_with_64b_values(
            "expected", s, list(range(3, 8)) + list(range(200, 203))
        )

        s.add(output == expected)
        assert s.check() == sat

    def test_mm256_alignr_epi64_find_shift(self):
        """Use Z3 to find the shift amount"""
        s = Solver()
        a = ymm_reg_with_64b_values("a", s, list(range(100, 104)))
        b = ymm_reg_with_64b_values("b", s, list(range(4)))
        imm8 = BitVec("imm8", 8)

        output = _mm256_alignr_epi64(a, b, imm8)
        expected = ymm_reg_with_64b_values("expected", s, list(range(1, 4)) + [100])

        s.add(output == expected)
        assert s.check() == sat
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == 1


class TestMaskAlignrEpi32:
    """Tests for _mm512_mask_alignr_epi32"""

    def test_mm512_mask_alignr_epi32_mask_all_zeros(self):
        """All mask bits zero should return src unchanged"""
        s = Solver()
        src = zmm_reg_with_32b_values("src", s, list(range(100, 116)))
        a = zmm_reg_with_32b_values("a", s, list(range(20, 36)))
        b = zmm_reg_with_32b_values("b", s, list(range(16)))
        k = BitVecVal(0x0000, 16)

        output = _mm512_mask_alignr_epi32(src, k, a, b, 4)
        expected = src

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_mask_alignr_epi32_mask_all_ones(self):
        """All mask bits set should perform normal alignr"""
        s = Solver()
        src = zmm_reg_with_32b_values("src", s, list(range(100, 116)))
        a = zmm_reg_with_32b_values("a", s, list(range(20, 36)))
        b = zmm_reg_with_32b_values("b", s, list(range(16)))
        k = BitVecVal(0xFFFF, 16)

        output = _mm512_mask_alignr_epi32(src, k, a, b, 4)
        expected = zmm_reg_with_32b_values(
            "expected", s, list(range(4, 16)) + list(range(20, 24))
        )

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_mask_alignr_epi32_alternating_mask(self):
        """Alternating mask bits"""
        s = Solver()
        src = zmm_reg_with_32b_values("src", s, list(range(100, 116)))
        a = zmm_reg_with_32b_values("a", s, list(range(20, 36)))
        b = zmm_reg_with_32b_values("b", s, list(range(16)))
        k = BitVecVal(0xAAAA, 16)  # 0b1010101010101010

        output = _mm512_mask_alignr_epi32(src, k, a, b, 2)
        # Alignr by 2: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21]
        # With mask 0xAAAA: [100, 3, 102, 5, 104, 7, 106, 9, 108, 11, 110, 13, 112, 15, 114, 21]
        expected = zmm_reg_with_32b_values(
            "expected",
            s,
            [100, 3, 102, 5, 104, 7, 106, 9, 108, 11, 110, 13, 112, 15, 114, 21],
        )

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_mask_alignr_epi32_partial_mask(self):
        """Partial mask - lower half masked"""
        s = Solver()
        src = zmm_reg_with_32b_values("src", s, list(range(100, 116)))
        a = zmm_reg_with_32b_values("a", s, list(range(20, 36)))
        b = zmm_reg_with_32b_values("b", s, list(range(16)))
        k = BitVecVal(0x00FF, 16)  # Lower 8 elements enabled

        output = _mm512_mask_alignr_epi32(src, k, a, b, 1)
        # Alignr by 1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
        # With mask 0x00FF: [1, 2, 3, 4, 5, 6, 7, 8, 108, 109, 110, 111, 112, 113, 114, 115]
        expected = zmm_reg_with_32b_values(
            "expected", s, list(range(1, 9)) + list(range(108, 116))
        )

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_mask_alignr_epi32_find_mask(self):
        """Use Z3 to find the mask"""
        s = Solver()
        src = zmm_reg_with_32b_values("src", s, list(range(100, 116)))
        a = zmm_reg_with_32b_values("a", s, list(range(20, 36)))
        b = zmm_reg_with_32b_values("b", s, list(range(16)))
        k = BitVec("k", 16)

        output = _mm512_mask_alignr_epi32(src, k, a, b, 8)
        # Alignr by 8: [8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27]
        # Want: [8, 101, 10, 103, 12, 105, 14, 107, 20, 109, 22, 111, 24, 113, 26, 115]
        expected = zmm_reg_with_32b_values(
            "expected",
            s,
            [8, 101, 10, 103, 12, 105, 14, 107, 20, 109, 22, 111, 24, 113, 26, 115],
        )

        s.add(output == expected)
        assert s.check() == sat
        model_k = s.model().evaluate(k).as_long()
        assert model_k == 0x5555  # 0b0101010101010101


class TestMaskAlignrEpi64:
    """Tests for _mm512_mask_alignr_epi64"""

    def test_mm512_mask_alignr_epi64_mask_all_zeros(self):
        """All mask bits zero should return src unchanged"""
        s = Solver()
        src = zmm_reg_with_64b_values("src", s, list(range(100, 108)))
        a = zmm_reg_with_64b_values("a", s, list(range(200, 208)))
        b = zmm_reg_with_64b_values("b", s, list(range(8)))
        k = BitVecVal(0x00, 8)

        output = _mm512_mask_alignr_epi64(src, k, a, b, 2)
        expected = src

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_mask_alignr_epi64_mask_all_ones(self):
        """All mask bits set should perform normal alignr"""
        s = Solver()
        src = zmm_reg_with_64b_values("src", s, list(range(100, 108)))
        a = zmm_reg_with_64b_values("a", s, list(range(200, 208)))
        b = zmm_reg_with_64b_values("b", s, list(range(8)))
        k = BitVecVal(0xFF, 8)

        output = _mm512_mask_alignr_epi64(src, k, a, b, 2)
        expected = zmm_reg_with_64b_values(
            "expected", s, list(range(2, 8)) + list(range(200, 202))
        )

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_mask_alignr_epi64_alternating_mask(self):
        """Alternating mask bits"""
        s = Solver()
        src = zmm_reg_with_64b_values("src", s, list(range(100, 108)))
        a = zmm_reg_with_64b_values("a", s, list(range(200, 208)))
        b = zmm_reg_with_64b_values("b", s, list(range(8)))
        k = BitVecVal(0xAA, 8)  # 0b10101010

        output = _mm512_mask_alignr_epi64(src, k, a, b, 1)
        # Alignr by 1: [1, 2, 3, 4, 5, 6, 7, 200]
        # With mask 0xAA: [100, 2, 102, 4, 104, 6, 106, 200]
        expected = zmm_reg_with_64b_values(
            "expected", s, [100, 2, 102, 4, 104, 6, 106, 200]
        )

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_mask_alignr_epi64_partial_mask(self):
        """Partial mask - lower half enabled"""
        s = Solver()
        src = zmm_reg_with_64b_values("src", s, list(range(100, 108)))
        a = zmm_reg_with_64b_values("a", s, list(range(200, 208)))
        b = zmm_reg_with_64b_values("b", s, list(range(8)))
        k = BitVecVal(0x0F, 8)  # 0b00001111 - lower 4 elements enabled

        output = _mm512_mask_alignr_epi64(src, k, a, b, 3)
        # Alignr by 3: [3, 4, 5, 6, 7, 200, 201, 202]
        # With mask 0x0F: [3, 4, 5, 6, 104, 105, 106, 107]
        expected = zmm_reg_with_64b_values(
            "expected", s, list(range(3, 7)) + list(range(104, 108))
        )

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_mask_alignr_epi64_single_bit_mask(self):
        """Single bit mask"""
        s = Solver()
        src = zmm_reg_with_64b_values("src", s, list(range(100, 108)))
        a = zmm_reg_with_64b_values("a", s, list(range(200, 208)))
        b = zmm_reg_with_64b_values("b", s, list(range(8)))
        k = BitVecVal(0x10, 8)  # 0b00010000 - only element 4 enabled

        output = _mm512_mask_alignr_epi64(src, k, a, b, 2)
        # Alignr by 2: [2, 3, 4, 5, 6, 7, 200, 201]
        # With mask 0x10: [100, 101, 102, 103, 6, 105, 106, 107]
        expected = zmm_reg_with_64b_values(
            "expected", s, list(range(100, 104)) + [6] + list(range(105, 108))
        )

        s.add(output == expected)
        assert s.check() == sat

    def test_mm512_mask_alignr_epi64_find_shift_and_mask(self):
        """Use Z3 to find both shift and mask"""
        s = Solver()
        src = zmm_reg_with_64b_values("src", s, list(range(100, 108)))
        a = zmm_reg_with_64b_values("a", s, list(range(200, 208)))
        b = zmm_reg_with_64b_values("b", s, list(range(8)))
        k = BitVec("k", 8)
        imm8 = BitVec("imm8", 8)

        output = _mm512_mask_alignr_epi64(src, k, a, b, imm8)
        # Want: [5, 6, 7, 103, 104, 105, 106, 107] (shift by 5, mask = 0x07)
        expected = zmm_reg_with_64b_values(
            "expected", s, list(range(5, 8)) + list(range(103, 108))
        )

        s.add(output == expected)
        assert s.check() == sat
        model_k = s.model().evaluate(k).as_long()
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == 5
        assert model_k == 0x07  # 0b00000111


class TestShuffleMaskDecode:
    """Tests for decode_shuffle_mask function"""

    def test_decode_shuffle_mask_round_trip_all_values(self):
        for imm8 in range(256):
            z, y, x, w = decode_shuffle_mask(imm8)
            assert 0 <= w <= 3, f"w={w} out of range for imm8=0x{imm8:02x}"
            assert 0 <= x <= 3, f"x={x} out of range for imm8=0x{imm8:02x}"
            assert 0 <= y <= 3, f"y={y} out of range for imm8=0x{imm8:02x}"
            assert 0 <= z <= 3, f"z={z} out of range for imm8=0x{imm8:02x}"

            assert _MM_SHUFFLE(z, y, x, w) == imm8, (
                f"Round-trip failed for 0x{imm8:02x}: "
                f"decoded to ({z}, {y}, {x}, {w}), "
                f"re-encoded to 0x{_MM_SHUFFLE(z, y, x, w):02x}"
            )

    def test_mm_shuffle_str_identity(self):
        """Test string representation of identity shuffle"""
        result = mm_shuffle_str(0xE4)
        assert result == "_MM_SHUFFLE(3, 2, 1, 0)"

    def test_mm_shuffle_str_0x88(self):
        """Test string representation of 0x88 mask from bitonic sorter"""
        result = mm_shuffle_str(0x88)
        assert result == "_MM_SHUFFLE(2, 0, 2, 0)"

    def test_mm_shuffle_str_0xdd(self):
        """Test string representation of 0xdd mask from bitonic sorter"""
        result = mm_shuffle_str(0xDD)
        assert result == "_MM_SHUFFLE(3, 1, 3, 1)"

    def test_mm_shuffle_str_all_zeros(self):
        """Test string representation of all zeros"""
        result = mm_shuffle_str(0x00)
        assert result == "_MM_SHUFFLE(0, 0, 0, 0)"

    def test_mm_shuffle_str_all_ones(self):
        """Test string representation of all ones"""
        result = mm_shuffle_str(0xFF)
        assert result == "_MM_SHUFFLE(3, 3, 3, 3)"

    def test_mm_shuffle_str_reverse(self):
        """Test string representation of reverse shuffle"""
        result = mm_shuffle_str(0x1B)
        assert result == "_MM_SHUFFLE(0, 1, 2, 3)"

    def test_mm_shuffle_str_format(self):
        """Test that all strings have correct format"""
        for imm8 in [0x00, 0x88, 0xDD, 0xE4, 0xFF, 0x1B, 0x4E, 0xB1]:
            result = mm_shuffle_str(imm8)
            # Check it starts with _MM_SHUFFLE(
            assert result.startswith("_MM_SHUFFLE("), (
                f"Bad format for 0x{imm8:02x}: {result}"
            )
            # Check it ends with )
            assert result.endswith(")"), f"Bad format for 0x{imm8:02x}: {result}"
            # Check it contains the right number of commas
            assert result.count(",") == 3, f"Bad format for 0x{imm8:02x}: {result}"
