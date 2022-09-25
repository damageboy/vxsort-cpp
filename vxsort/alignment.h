#ifndef VXSORT_ALIGNNMENT_H
#define VXSORT_ALIGNNMENT_H

#include <cstdint>
#include "defs.h"

namespace vxsort {
using namespace vxsort::types;

using namespace std;

template <int N>
struct alignment_hint {
public:
    static constexpr usize ALIGN = N;
    static const i8 REALIGN = 0x66;
    static_assert(REALIGN > ALIGN, "REALIGN must be larger than ALIGN");

    alignment_hint() : left_align(REALIGN), right_align(REALIGN) {}
    alignment_hint realign_left() {
        alignment_hint copy = *this;
        copy.left_align = REALIGN;
        return copy;
        }

    alignment_hint realign_right() {
        alignment_hint copy = *this;
        copy.right_align = REALIGN;
        return copy;
        }

    static bool is_aligned(void* p) { return (usize)p % ALIGN == 0; }

    i32 left_align : 8;
    i32 right_align : 8;
};

} // namespace vxsort
#endif  // VXSORT_ALIGNNMENT_H
