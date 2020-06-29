#ifndef BITONIC_MACHINE_H
#define BITONIC_MACHINE_H

#include <cstdint>
#include "../defs.h"
#include "../machine_traits.h"

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
};
}  // namespace smallsort
}  // namespace vxsort
#endif
