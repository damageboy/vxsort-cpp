from z3 import Solver, unsat, sat, BitVec, BitVecVal, Concat, Extract

# Assuming your z3s functions and registers are importable, e.g.:
from z3_avx import _MM_SHUFFLE, _MM_SHUFFLE2
from z3_avx import _mm256_permute_ps
from z3_avx import _mm512_permute_ps
from z3_avx import _mm256_permutexvar_epi32
from z3_avx import _mm512_permutexvar_epi32
from z3_avx import _mm512_permutex2var_epi32
from z3_avx import _mm512_permutex2var_epi64
from z3_avx import _mm512_mask_permutex2var_ps
from z3_avx import _mm256_permutexvar_epi64
from z3_avx import _mm512_permutexvar_epi64
from z3_avx import _mm256_shuffle_ps
from z3_avx import _mm512_shuffle_ps
from z3_avx import _mm256_shuffle_pd
from z3_avx import _mm512_shuffle_pd
from z3_avx import _mm256_permute_pd
from z3_avx import _mm512_permute_pd
from z3_avx import _mm256_permute2x128_si256
from z3_avx import _mm512_shuffle_i32x4
from z3_avx import ymm_reg, ymm_reg_with_32b_values, ymm_reg_with_64b_values, ymm_reg_with_unique_values, ymm_reg_pair_with_unique_values, construct_ymm_reg_from_elements
from z3_avx import zmm_reg, zmm_reg_with_32b_values, zmm_reg_with_64b_values, zmm_reg_with_unique_values, zmm_reg_pair_with_unique_values, construct_zmm_reg_from_elements
from z3_avx import ymm_reg_reversed, zmm_reg_reversed

#    imm8 = 0b11100100 means:
#    - Lane bits [1:0] = 00 (select element 0 for position 0)
#    - Lane bits [3:2] = 01 (select element 1 for position 1)
#    - Lane bits [5:4] = 10 (select element 2 for position 2)
#    - Lane bits [7:6] = 11 (select element 3 for position 3)
#    This should result in each 128-bit lane's elements staying in place.

null_permute_epi32_imm8 = _MM_SHUFFLE(3, 2, 1, 0)
null_permute_pd_imm8 = _MM_SHUFFLE2(1, 0)  # bit 1 = 1 (select elem 1 for pos 1), bit 0 = 0 (select elem 0 for pos 0)

null_shuffle_ps_imm8 = _MM_SHUFFLE(3, 2, 1, 0)  # pos0: op1[0], pos1: op1[1], pos2: op1[2], pos3: op1[3]
null_shuffle_ps_2vec_imm8 = _MM_SHUFFLE(1, 0, 1, 0)  # pos0: op1[0], pos1: op1[1], pos2: op2[0], pos3: op2[1]
null_shuffle_pd_avx2_imm8 = 0x0A  # 0b1010: identity permutation for AVX2 (2 lanes, uses bits 0-3)
null_shuffle_pd_avx512_imm8 = 0xAA  # 0b10101010: identity permutation for AVX512 (4 lanes, uses bits 0-7)

# For _mm256_permute2x128_si256 null permute: 
# Low lane: select a[127:0] (control=0), High lane: select a[255:128] (control=1)
null_permute2x128_imm8 = (1 << 4) | 0  # 0x10: high_lane=1 (a[255:128]), low_lane=0 (a[127:0])

# For _mm512_shuffle_i32x4 null permute:
# dst[127:0] := a[127:0] (imm8[1:0] = 0), dst[255:128] := a[255:128] (imm8[3:2] = 1)
# dst[383:256] := b[383:256] (imm8[5:4] = 2), dst[511:384] := b[511:384] (imm8[7:6] = 3)
null_shuffle_i32x4_imm8 = _MM_SHUFFLE(3, 2, 1, 0)  # 0xE4

null_permute_vector_epi32_avx2 = [i for i in range(8)]
null_permute_vector_epi32_avx512 = [i for i in range(16)]
null_permute_vector_epi64_avx2 = [i for i in range(4)]
null_permute_vector_epi64_avx512 = [i for i in range(8)]
null_permutex2var_vector_epi32_avx512 = [i for i in range(16)]  # source selector = 0, offset = i
null_permutex2var_vector_epi64_avx512 = [i for i in range(8)]  # source selector = 0, offset = i
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
        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

    def test_mm256_permute_epi32_null_permute_found(self):
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=32)
        imm8 = BitVec("imm8", 8)
        output = _mm256_permute_ps(input, imm8)

        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_permute_epi32_imm8, "Z3 found unexpected null permute: got 0x{model_imm8:08x}, expected 0x{null_permute_epi32_imm8:08x}"

    def test_mm512_permute_epi32_null_permute(self):
        s = Solver()

        input = zmm_reg("zmm0")
        output = _mm512_permute_ps(input, null_permute_epi32_imm8)

        s.add(input != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_permute_epi32_null_permute_found(self):
        s = Solver()
        input = zmm_reg_with_unique_values("zmm0", s, bits=32)
        imm8 = BitVec("imm8", 8)
        output = _mm512_permute_ps(input, imm8)

        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute failed"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_permute_epi32_imm8, "Z3 found unexpected null permute: got 0x{model_imm8:08x}, expected 0x{null_permute_epi32_imm8:08x}"


class TestPermutePd:
    """Tests for _mm256_permute_pd and _mm512_permute_pd (permute_epi64)"""
    
    def test_mm256_permute_epi64_null_permute_works(self):
        s = Solver()
        input = ymm_reg("ymm0")
        output = _mm256_permute_pd(input, null_permute_pd_imm8)

        s.add(input != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

    def test_mm256_permute_epi64_null_permute_found(self):
        s = Solver()
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        imm8 = BitVec("imm8", 8)
        output = _mm256_permute_pd(input, imm8)

        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_permute_pd_imm8, "Z3 found unexpected null permute: got 0x{model_imm8:08x}, expected 0x{null_permute_pd_imm8:08x}"

    def test_mm512_permute_epi64_null_permute_works(self):
        s = Solver()

        input = zmm_reg("zmm0")
        output = _mm512_permute_pd(input, null_permute_pd_imm8)

        s.add(input != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_permute_epi64_null_permute_found(self):
        s = Solver()
        input = zmm_reg_with_unique_values("zmm0", s, bits=64)
        imm8 = BitVec("imm8", 8)
        output = _mm512_permute_pd(input, imm8)

        s.add(input == output)
        result = s.check()

        assert result == sat, "Z3 failed to find null permute"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_permute_pd_imm8, "Z3 found unexpected null permute: got 0x{model_imm8:08x}, expected 0x{null_permute_pd_imm8:08x}"


class TestPermutexvarEpi32:
    """Tests for _mm256_permutexvar_epi32 and _mm512_permutexvar_epi32"""
    
    def test_mm256_permutexvar_epi32_null_permute_works(self):
        s = Solver()
        input = ymm_reg("ymm0")
        indices = ymm_reg_with_32b_values("indices", s, null_permute_vector_epi32_avx2)
        output = _mm256_permutexvar_epi32(input, indices)

        s.add(input != output)
        result = s.check()

        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

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
        assert model_indices == expected_long, f"Z3 found unexpected null permute: got 0x{model_indices:064x}, expected 0x{expected_long:064x}"

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
        assert model_indices == expected_long, f"Z3 found unexpected reverse permute: got 0x{model_indices:064x}, expected 0x{expected_long:064x}"

    def test_mm512_permutexvar_epi32_null_permute_works(self):
        s = Solver()
        input = zmm_reg("zmm0")
        indicew = zmm_reg_with_32b_values("indices", s, null_permute_vector_epi32_avx512)
        output = _mm512_permutexvar_epi32(input, indicew)

        # Assert that the output is NOT equal to the input
        # If this is unsatisfiable, it means the output MUST be equal to the input
        # and that the null permute vector can only lead to an identity permutation
        s.add(input != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

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
        assert model_indices == expected_long, "Z3 found unexpected null permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"

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
        assert model_indices == expected_long, "Z3 found unexpected reverse permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"


class TestPermutexvarEpi64:
    """Tests for _mm256_permutexvar_epi64 and _mm512_permutexvar_epi64"""
    
    def test_mm256_permutexvar_epi64_null_permute_works(self):
        s = Solver()
        input = ymm_reg("ymm0")
        indices = ymm_reg_with_64b_values("indices", s, null_permute_vector_epi64_avx2)
        output = _mm256_permutexvar_epi64(input, indices)

        s.add(input != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

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
        assert model_indices == expected_long, "Z3 found unexpected null permute: got 0x{model_indices:064x}, expected 0x{expected_long:064x}"

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
        assert model_indices == expected_long, "Z3 found unexpected reverse permute: got 0x{model_indices:064x}, expected 0x{expected_long:064x}"

    def test_mm512_permutexvar_epi64_null_permute_works(self):
        s = Solver()
        input = zmm_reg("zmm0")
        indices = zmm_reg_with_64b_values("indices", s, null_permute_vector_epi64_avx512)
        output = _mm512_permutexvar_epi64(input, indices)

        s.add(input != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

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
        assert model_indices == expected_long, "Z3 found unexpected null permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"

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
        assert model_indices == expected_long, "Z3 found unexpected reverse permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"


class TestPermutex2varEpi32:
    """Tests for _mm512_permutex2var_epi32 (512-bit only)"""
    
    def test_mm512_permutex2var_epi32_null_permute_works(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values("indices", s, null_permutex2var_vector_epi32_avx512)
        output = _mm512_permutex2var_epi32(a, indices, b)
        
        # If this is unsatisfiable, it means the output MUST be equal to source a
        s.add(a != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

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
        assert model_indices == expected_long, f"Z3 found unexpected null permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"

    def test_mm512_permutex2var_epi32_select_from_b(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        
        select_b_indices = [(1 << 4) | i for i in range(16)]
        indices = zmm_reg_with_32b_values("indices", s, select_b_indices)
        output = _mm512_permutex2var_epi32(a, indices, b)
        
        s.add(b != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where select from b failed: {s.model() if result == sat else 'No model'}"

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
        assert result == unsat, f"Z3 found a counterexample where reverse permute from a failed: {s.model() if result == sat else 'No model'}"

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
        assert result == unsat, f"Z3 found a counterexample where mixed sources failed: {s.model() if result == sat else 'No model'}"


class TestPermutex2varEpi64:
    """Tests for _mm512_permutex2var_epi64 (512-bit only)"""
    
    def test_mm512_permutex2var_epi64_null_permute_works(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        indices = zmm_reg_with_64b_values("indices", s, null_permutex2var_vector_epi64_avx512)
        output = _mm512_permutex2var_epi64(a, indices, b)
        s.add(a != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

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
        assert model_indices == expected_long, f"Z3 found unexpected null permute: got 0x{model_indices:0128x}, expected 0x{expected_long:0128x}"

    def test_mm512_permutex2var_epi64_select_from_b(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        
        select_b_indices = [(1 << 3) | i for i in range(8)]
        indices = zmm_reg_with_64b_values("indices", s, select_b_indices)
        output = _mm512_permutex2var_epi64(a, indices, b)
        s.add(b != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where select from b failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_permutex2var_epi64_reverse_permute_from_a(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=64)
        
        reverse_a_indices = [(0 << 3) | (7 - i) for i in range(8)]
        indices = zmm_reg_with_64b_values("indices", s, reverse_a_indices)
        
        output = _mm512_permutex2var_epi64(a, indices, b)
        
        reversed_a = zmm_reg_reversed("a_reversed", s, a, bits=64)
        
        s.add(reversed_a != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where reverse permute from a failed: {s.model() if result == sat else 'No model'}"

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
        assert result == unsat, f"Z3 found a counterexample where mixed sources failed: {s.model() if result == sat else 'No model'}"


class Test_shuffle_ps:
    """Tests for _mm256_shuffle_ps and _mm512_shuffle_ps"""
    
    def test_mm256_shuffle_ps_null_permute_works(self):
        s = Solver()
        
        input = ymm_reg("ymm0")
        output = _mm256_shuffle_ps(input, input, null_shuffle_ps_imm8)
        
        s.add(output != input)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"

    def test_mm256_shuffle_ps_null_permute_found(self):
        s = Solver()
        
        input = ymm_reg_with_unique_values("ymm0", s, bits=32)
        imm8 = BitVec("imm8", 8)
        output = _mm256_shuffle_ps(input, input, imm8)
        
        s.add(output == input)
        result = s.check()
        
        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_ps_imm8, f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_ps_2vec_imm8:02x}"

    def test_mm256_shuffle_ps_null_permute_2vec_works(self):
        s = Solver()
        
        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=32)
        
        output = _mm256_shuffle_ps(op1, op2, null_shuffle_ps_2vec_imm8)
        
        expected = construct_ymm_reg_from_elements(32, [
            (op1, 0), (op1, 1), (op2, 0), (op2, 1),
            (op1, 4), (op1, 5), (op2, 4), (op2, 5)
        ])
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"

    def test_mm256_shuffle_ps_null_permute_2vec_found(self):
        s = Solver()
        
        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=32)
        
        imm8 = BitVec("imm8", 8)
        output = _mm256_shuffle_ps(op1, op2, imm8)
        
        expected = construct_ymm_reg_from_elements(32, [
            (op1, 0), (op1, 1), (op2, 0), (op2, 1),
            (op1, 4), (op1, 5), (op2, 4), (op2, 5)
        ])
        
        s.add(output == expected)
        result = s.check()
        
        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_ps_2vec_imm8, f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_ps_2vec_imm8:02x}"

    def test_mm512_shuffle_ps_null_permute_works(self):
        s = Solver()
        
        input_vector = zmm_reg("zmm0")
        output_vector = _mm512_shuffle_ps(input_vector, input_vector, null_shuffle_ps_2vec_imm8)
        
        expected = construct_zmm_reg_from_elements(32, [
            (input_vector, 0), (input_vector, 1), (input_vector, 0), (input_vector, 1),
            (input_vector, 4), (input_vector, 5), (input_vector, 4), (input_vector, 5),
            (input_vector, 8), (input_vector, 9), (input_vector, 8), (input_vector, 9),
            (input_vector, 12), (input_vector, 13), (input_vector, 12), (input_vector, 13)
        ])
        
        s.add(output_vector != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_shuffle_ps_null_permute_found(self):
        s = Solver()
        
        input = zmm_reg_with_unique_values("zmm0", s, bits=32)
        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_ps(input, input, imm8)
        
        expected = construct_zmm_reg_from_elements(32, [
            (input, 0), (input, 1), (input, 0), (input, 1),
            (input, 4), (input, 5), (input, 4), (input, 5),
            (input, 8), (input, 9), (input, 8), (input, 9),
            (input, 12), (input, 13), (input, 12), (input, 13)
        ])
        
        s.add(output == expected)
        result = s.check()
        
        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_ps_2vec_imm8, f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_ps_2vec_imm8:02x}"

    def test_mm512_shuffle_ps_null_permute_2vec_works(self):
        s = Solver()
        
        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=32)
        
        output = _mm512_shuffle_ps(op1, op2, null_shuffle_ps_2vec_imm8)
        
        expected = construct_zmm_reg_from_elements(32, [
            (op1, 0), (op1, 1), (op2, 0), (op2, 1),
            (op1, 4), (op1, 5), (op2, 4), (op2, 5),
            (op1, 8), (op1, 9), (op2, 8), (op2, 9),
            (op1, 12), (op1, 13), (op2, 12), (op2, 13)
        ])
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_shuffle_ps_null_permute_2vec_found(self):
        s = Solver()

        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=32)
        
        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_ps(op1, op2, imm8)
        
        expected = construct_zmm_reg_from_elements(32, [
            (op1, 0), (op1, 1), (op2, 0), (op2, 1),
            (op1, 4), (op1, 5), (op2, 4), (op2, 5),
            (op1, 8), (op1, 9), (op2, 8), (op2, 9),
            (op1, 12), (op1, 13), (op2, 12), (op2, 13)
        ])
        
        s.add(output == expected)
        result = s.check()
        
        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_ps_2vec_imm8, f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_ps_2vec_imm8:02x}"


class Test_shuffle_pd:
    """Tests for _mm256_shuffle_pd and _mm512_shuffle_pd"""
    
    def test_mm256_shuffle_pd_null_permute_works(self):
        s = Solver()
        
        input = ymm_reg("ymm0")
        output_vector = _mm256_shuffle_pd(input, input, null_shuffle_pd_avx2_imm8)
        s.add(output_vector != input)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"

    def test_mm256_shuffle_pd_null_permute_found(self):
        s = Solver()
        
        input = ymm_reg_with_unique_values("ymm0", s, bits=64)
        imm8 = BitVec("imm8", 8)
        output = _mm256_shuffle_pd(input, input, imm8)
        
        s.add(output == input)
        result = s.check()
        
        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_pd_avx2_imm8, f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_pd_avx2_imm8:02x}"

    def test_mm256_shuffle_pd_null_permute_2vec_works(self):
        s = Solver()
        
        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=64)
        output = _mm256_shuffle_pd(op1, op2, null_shuffle_pd_avx2_imm8)
        expected = construct_ymm_reg_from_elements(64, [
            (op1, 0), (op2, 1), (op1, 2), (op2, 3)
        ])
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"

    def test_mm256_shuffle_pd_null_permute_2vec_found(self):
        s = Solver()
        
        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=64)
        imm8 = BitVec("imm8", 8)
        output = _mm256_shuffle_pd(op1, op2, imm8)
        expected = construct_ymm_reg_from_elements(64, [
            (op1, 0), (op2, 1), (op1, 2), (op2, 3)
        ])
        
        s.add(output == expected)
        result = s.check()
        
        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_pd_avx2_imm8, f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_pd_avx2_imm8:02x}"

    def test_mm512_shuffle_pd_null_permute_works(self):
        s = Solver()
        
        input = zmm_reg("zmm0")
        output_vector = _mm512_shuffle_pd(input, input, null_shuffle_pd_avx512_imm8)
        
        s.add(output_vector != input)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_shuffle_pd_null_permute_found(self):
        s = Solver()
        
        input = zmm_reg_with_unique_values("zmm0", s, bits=64)
        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_pd(input, input, imm8)
        
        s.add(output == input)
        result = s.check()
        
        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_pd_avx512_imm8, f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_pd_avx512_imm8:02x}"

    def test_mm512_shuffle_pd_null_permute_2vec_works(self):
        s = Solver()
        
        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=64)
        
        output = _mm512_shuffle_pd(op1, op2, null_shuffle_pd_avx512_imm8)
        
        expected = construct_zmm_reg_from_elements(64, [
            (op1, 0), (op2, 1), (op1, 2), (op2, 3),
            (op1, 4), (op2, 5), (op1, 6), (op2, 7)
        ])
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_shuffle_pd_null_permute_2vec_found(self):
        s = Solver()

        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=64)
        
        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_pd(op1, op2, imm8)
        
        expected = construct_zmm_reg_from_elements(64, [
            (op1, 0), (op2, 1), (op1, 2), (op2, 3),
            (op1, 4), (op2, 5), (op1, 6), (op2, 7)
        ])
        
        s.add(output == expected)
        result = s.check()
        
        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_pd_avx512_imm8, f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_pd_avx512_imm8:02x}"


class TestPermute2x128Si256:
    """Tests for _mm256_permute2x128_si256 (256-bit only)"""
    
    def test_mm256_permute2x128_si256_null_permute_works(self):
        s = Solver()
        
        input_vector = ymm_reg("ymm0")
        output_vector = _mm256_permute2x128_si256(input_vector, input_vector, null_permute2x128_imm8)
        
        s.add(input_vector != output_vector)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

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
        assert model_imm8 in valid_identity_permutes, f"Z3 found invalid null permute: got 0x{model_imm8:02x}, expected one of {[hex(x) for x in valid_identity_permutes]}"

    def test_mm256_permute2x128_si256_null_permute_2vec_works(self):
        s = Solver()
        
        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=128)
        
        output = _mm256_permute2x128_si256(op1, op2, null_permute2x128_imm8)
        
        expected = construct_ymm_reg_from_elements(128, [
            (op1, 0),  # op1[127:0] -> low lane
            (op1, 1)   # op1[255:128] -> high lane  
        ])
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null permute failed: {s.model() if result == sat else 'No model'}"

    def test_mm256_permute2x128_si256_null_permute_2vec_found(self):
        s = Solver()
        
        op1, op2 = ymm_reg_pair_with_unique_values("op", s, bits=128)
        
        imm8 = BitVec("imm8", 8)
        output = _mm256_permute2x128_si256(op1, op2, imm8)
        
        s.add((imm8 & 0x88) == 0)  # No zero flags set
        
        expected = construct_ymm_reg_from_elements(128, [
            (op1, 0),  # op1[127:0] -> low lane  
            (op1, 1)   # op1[255:128] -> high lane
        ])
        
        s.add(output == expected)
        result = s.check()
        
        assert result == sat, "Z3 failed to find null permute"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_permute2x128_imm8, f"Z3 found unexpected null permute: got 0x{model_imm8:02x}, expected 0x{null_permute2x128_imm8:02x}"

    def test_mm256_permute2x128_si256_swap_lanes(self): 
        s = Solver()
        
        input_vector = ymm_reg_with_unique_values("ymm0", s, bits=128)
        
        swap_imm8 = 0x01
        output = _mm256_permute2x128_si256(input_vector, input_vector, swap_imm8)
        
        expected = construct_ymm_reg_from_elements(128, [
            (input_vector, 1),  # Was high lane (a[255:128]), now low
            (input_vector, 0)   # Was low lane (a[127:0]), now high  
        ])

        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where lane swap failed: {s.model() if result == sat else 'No model'}"

    def test_mm256_permute2x128_si256_cross_vector(self):
        s = Solver()
        
        a, b = ymm_reg_pair_with_unique_values("input", s, bits=128)
        
        cross_imm8 = 0x23
        output = _mm256_permute2x128_si256(a, b, cross_imm8)
        
        expected = construct_ymm_reg_from_elements(128, [
            (b, 1),  # b[255:128] -> low lane
            (b, 0)   # b[127:0] -> high lane
        ])
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where cross-vector permute failed: {s.model() if result == sat else 'No model'}"

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
        assert result == unsat, f"Z3 found a counterexample where zero lane failed: {s.model() if result == sat else 'No model'}"

    def test_mm256_permute2x128_si256_zero_both_lanes(self):
        s = Solver()
        
        input_vector = ymm_reg("ymm0")
        
        zero_both_imm8 = 0x88
        output = _mm256_permute2x128_si256(input_vector, input_vector, zero_both_imm8)
        
        expected = BitVecVal(0, 256)
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where zero both lanes failed: {s.model() if result == sat else 'No model'}"


class TestShuffleI32x4:
    """Tests for _mm512_shuffle_i32x4 (512-bit only)"""
    
    def test_mm512_shuffle_i32x4_null_permute_works(self):
        s = Solver()
        
        input_vector = zmm_reg("zmm0")
        output_vector = _mm512_shuffle_i32x4(input_vector, input_vector, null_shuffle_i32x4_imm8)
        
        s.add(input_vector != output_vector)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_shuffle_i32x4_null_permute_found(self):
        s = Solver()
        
        input_vector = zmm_reg_with_unique_values("zmm0", s, bits=128)
        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_i32x4(input_vector, input_vector, imm8)
        
        s.add(input_vector == output)
        result = s.check()
        
        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_i32x4_imm8, f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_i32x4_imm8:02x}"

    def test_mm512_shuffle_i32x4_null_permute_2vec_works(self):
        s = Solver()
        
        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=128)
        
        output = _mm512_shuffle_i32x4(op1, op2, null_shuffle_i32x4_imm8)
        
        expected = construct_zmm_reg_from_elements(128, [
            (op1, 0),  # a[127:0] -> dst[127:0]
            (op1, 1),  # a[255:128] -> dst[255:128]
            (op2, 2),  # b[383:256] -> dst[383:256]
            (op2, 3)   # b[511:384] -> dst[511:384]
        ])
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where null shuffle failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_shuffle_i32x4_null_permute_2vec_found(self):
        s = Solver()
        
        op1, op2 = zmm_reg_pair_with_unique_values("op", s, bits=128)
        
        imm8 = BitVec("imm8", 8)
        output = _mm512_shuffle_i32x4(op1, op2, imm8)
        
        expected = construct_zmm_reg_from_elements(128, [
            (op1, 0),  # a[127:0] -> dst[127:0]
            (op1, 1),  # a[255:128] -> dst[255:128]
            (op2, 2),  # b[383:256] -> dst[383:256]
            (op2, 3)   # b[511:384] -> dst[511:384]
        ])
        
        s.add(output == expected)
        result = s.check()
        
        assert result == sat, "Z3 failed to find null shuffle"
        model_imm8 = s.model().evaluate(imm8).as_long()
        assert model_imm8 == null_shuffle_i32x4_imm8, f"Z3 found unexpected null shuffle: got 0x{model_imm8:02x}, expected 0x{null_shuffle_i32x4_imm8:02x}"

    def test_mm512_shuffle_i32x4_cross_lanes(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=128)
        
        cross_imm8 = _MM_SHUFFLE(0, 1, 2, 3)
        output = _mm512_shuffle_i32x4(a, b, cross_imm8)
        
        expected = construct_zmm_reg_from_elements(128, [
            (a, 3),  # a[511:384] -> dst[127:0]
            (a, 2),  # a[383:256] -> dst[255:128]
            (b, 1),  # b[255:128] -> dst[383:256]
            (b, 0)   # b[127:0] -> dst[511:384]
        ])
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where cross-lane shuffle failed: {s.model() if result == sat else 'No model'}"


class TestMaskPermutex2varPs:
    """Tests for _mm512_mask_permutex2var_ps (512-bit only)"""
    
    def test_mm512_mask_permutex2var_ps_mask_all_zeros(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values("indices", s, null_permutex2var_vector_epi32_avx512)
        mask = BitVecVal(0, 16)
        output = _mm512_mask_permutex2var_ps(a, mask, indices, b)
        
        s.add(a != output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where mask all zeros failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_mask_permutex2var_ps_mask_all_ones(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values("indices", s, null_permutex2var_vector_epi32_avx512)
        mask = BitVecVal(0xFFFF, 16)
        
        masked_output = _mm512_mask_permutex2var_ps(a, mask, indices, b)
        unmasked_output = _mm512_permutex2var_epi32(a, indices, b)
        
        s.add(masked_output != unmasked_output)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where mask all ones failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_mask_permutex2var_ps_alternating_mask(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        select_b_indices = [(1 << 4) | i for i in range(16)]
        indices = zmm_reg_with_32b_values("indices", s, select_b_indices)
        mask = BitVecVal(0x5555, 16)
        
        output = _mm512_mask_permutex2var_ps(a, mask, indices, b)
        
        expected_specs = []
        expected_specs = [(b, i) if i % 2 == 0 else (a, i) for i in range(16)]
        
        expected = construct_zmm_reg_from_elements(32, expected_specs)
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where alternating mask failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_mask_permutex2var_ps_reverse_with_partial_mask(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        reverse_a_indices = [(0 << 4) | (15 - i) for i in range(16)]
        indices = zmm_reg_with_32b_values("indices", s, reverse_a_indices)
        mask = BitVecVal(0x00FF, 16)
        
        output = _mm512_mask_permutex2var_ps(a, mask, indices, b)
        
        expected_specs = []
        for i in range(16):
            if i < 8:
                expected_specs.append((a, 15 - i))
            else:
                expected_specs.append((a, i))
        
        expected = construct_zmm_reg_from_elements(32, expected_specs)
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where reverse with partial mask failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_mask_permutex2var_ps_mixed_sources_with_mask(self):
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
        output = _mm512_mask_permutex2var_ps(a, mask, indices, b)
        
        expected_specs = [(a, i) for i in range(16)]
        expected = construct_zmm_reg_from_elements(32, expected_specs)
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where mixed sources with mask failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_mask_permutex2var_ps_single_bit_mask(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values("indices", s, [(1 << 4) | 10] * 16)
        mask = BitVecVal(1 << 5, 16)
        output = _mm512_mask_permutex2var_ps(a, mask, indices, b)
        
        expected_specs = []
        for i in range(16):
            if i == 5:
                expected_specs.append((b, 10))
            else:
                expected_specs.append((a, i))
        
        expected = construct_zmm_reg_from_elements(32, expected_specs)
        
        s.add(output != expected)
        result = s.check()
        assert result == unsat, f"Z3 found a counterexample where single bit mask failed: {s.model() if result == sat else 'No model'}"

    def test_mm512_mask_permutex2var_ps_find_identity_mask(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values("indices", s, [(1 << 4) | 7] * 16)  # All select b[7]
        mask = BitVec("mask", 16)
        output = _mm512_mask_permutex2var_ps(a, mask, indices, b)
        
        s.add(output == a)
        result = s.check()
        
        assert result == sat, "Z3 failed to find a mask for identity"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0, f"Z3 found unexpected mask for identity: got 0x{model_mask:04x}, expected 0x0000"

    def test_mm512_mask_permutex2var_ps_find_full_permute_mask(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values("indices", s, [(1 << 4) | i for i in range(16)])
        mask = BitVec("mask", 16)
        output = _mm512_mask_permutex2var_ps(a, mask, indices, b)
        
        s.add(output == b)
        result = s.check()
        
        assert result == sat, "Z3 failed to find a mask for full permutation"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0xFFFF, f"Z3 found unexpected mask for full permutation: got 0x{model_mask:04x}, expected 0xFFFF"

    def test_mm512_mask_permutex2var_ps_find_partial_mask(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        indices = zmm_reg_with_32b_values("indices", s, [(1 << 4) | i for i in range(16)])
        mask = BitVec("mask", 16)
        output = _mm512_mask_permutex2var_ps(a, mask, indices, b)

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
        assert model_mask == 0x000F, f"Z3 found unexpected mask for partial permutation: got 0x{model_mask:04x}, expected 0x000F"

    def test_mm512_mask_permutex2var_ps_find_indices_with_mask(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        mask = BitVecVal(0x5555, 16)
        indices = zmm_reg("indices")
        output = _mm512_mask_permutex2var_ps(a, mask, indices, b)
        
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
        assert pos0_index == 16, f"Position 0 index should be 16 (select b[0]), got {pos0_index}"

    def test_mm512_mask_permutex2var_ps_find_reverse_partial(self):
        s = Solver()
        
        a, b = zmm_reg_pair_with_unique_values("input", s, bits=32)
        mask = BitVec("mask", 16)
        indices = zmm_reg("indices")
        output = _mm512_mask_permutex2var_ps(a, mask, indices, b)

        expected_specs = []
        for i in range(16):
            if i < 8:
                expected_specs.append((a, 7 - i))  # Reverse: a[7], a[6], ..., a[0]
            else:
                expected_specs.append((a, i))      # Unchanged: a[8], a[9], ..., a[15]
        
        expected = construct_zmm_reg_from_elements(32, expected_specs)
        s.add(output == expected)
        result = s.check()
        assert result == sat, "Z3 failed to find mask+indices for partial reverse"
        model_mask = s.model().evaluate(mask).as_long()
        assert model_mask == 0x00FF, f"Expected mask 0x00FF for first 8 elements, got 0x{model_mask:04x}"