#ifndef BITONIC_MACHINE_H
#define BITONIC_MACHINE_H

#include <cstdint>
#include "../defs.h"
#include "vector_machine/machine_traits.h"

namespace vxsort {
namespace smallsort {
using namespace std;

template <typename T, vector_machine M>
struct bitonic_machine {
public:
    typedef T TV;
    typedef T TMASK;

    static INLINE void sort_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04);
    static INLINE void merge_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04);
    static INLINE void cross_min_max(TV& d01, TV& d02);
    static INLINE void strided_min_max(TV& d01, TV& d02);

    static NOINLINE void sort_01v_full_ascending(T *ptr);
    static NOINLINE void sort_02v_full_ascending(T *ptr);
    static NOINLINE void sort_03v_full_ascending(T *ptr);
    static NOINLINE void sort_04v_full_ascending(T *ptr);
    static void sort_full_vectors_ascending(T *ptr, usize length);
    static void sort_full_vectors_descending(T *ptr, usize length);
};
}  // namespace smallsort
}  // namespace vxsort
#endif
