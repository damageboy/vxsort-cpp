#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#include <cstdint>
#include <cstddef>

#include "../defs.h"
#include "../machine_traits.h"

namespace vxsort {
namespace smallsort {

// * We might read the last 4 bytes into a 128-bit vector for 64-bit element masking
// * We might read the last 8 bytes into a 128-bit vector for 32-bit element masking
// Most compilers actually fuse the pair of instructions: _mm256_cvtepi8_epiNN + _mm_loadu_si128
// into a single instruction that will not read more that 4/8 bytes..
// vpmovsxb[dq] ymm0, dword [...], eliminating the 128-bit load completely and effectively
// reading exactly 4/8 (depending if the instruction is vpmovsxbd or vpmovsxbq)
// without generating an out of bounds read at all.
// But, life is harsh, and we can't trust the compiler to do the right thing if it is not
// contractual, hence this flustercuck
const int M4_SIZE = 16 + 12;
const int M8_SIZE = 64 + 8;

extern const uint8_t mask_table_4[M4_SIZE];
extern const uint8_t mask_table_8[M8_SIZE];

template <typename T, vector_machine M>
struct bitonic {
 public:
  static void sort(T* ptr, size_t length);
  static void sort_alt(T* ptr, size_t length);
};
}  // namespace smallsort
}  // namespace gcsort
#endif
