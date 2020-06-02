//
// Created by dans on 6/1/20.
//

#ifndef GCSORT_MACHINE_TRAITS_H
#define GCSORT_MACHINE_TRAITS_H

#include <immintrin.h>
#include <cstdint>

namespace gcsort {

enum vector_machine {
  NONE,
  AVX2,
  AVX512,
  SVE,
};

/// Pre-specialized trait class outlining the required/possible vector operations provided by a vector ISA for the purposes of partitioning \tparam T - The underlying primitive type \tparam M - The vectorized ISA, see @enum vector_machines
template <typename T, vector_machine M>
struct vxsort_machine_traits {
 public:
  typedef __m256 TV;
  typedef uint32_t TMASK;

  static constexpr bool supports_compress_writes();

  static TV load_vec(TV* ptr);
  static void store_vec(TV* ptr, TV v);
  static void store_compress_vec(TV*ptr, TV v, TMASK mask);
  static TV partition_vector(TV v, int mask);
  static TV get_vec_pivot(T pivot);
  static TMASK get_cmpgt_mask(TV a, TV b);
};
}

#endif  // GCSORT_MACHINE_TRAITS_H
