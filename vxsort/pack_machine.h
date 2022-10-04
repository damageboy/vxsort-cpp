#ifndef VXSORT_PACK_MACHINE_H
#define VXSORT_PACK_MACHINE_H

#include <cstdint>
#include <limits>
#include <type_traits>
#include <cassert>
#include "defs.h"
#include "alignment.h"
#include "vector_machine/machine_traits.h"

#include <immintrin.h>
#include <cstdio>

namespace vxsort {

template<typename T, vector_machine M, int Shift>
class pack_machine {
    static_assert(Shift <= 31, "Shift must be in the range 0..31");

    using VMT = vxsort_machine_traits<T, M>;
    typedef typename VMT::TV TV;
    static const int N = sizeof(TV) / sizeof(T);
    typedef alignment_hint<T, M> AH;

public:

    static INLINE TV pack_vectors(TV dl, TV dr, const TV offset_v) {
        // This is statically compiled in/out
        if (Shift > 0) {
            dl = VMT::shift_right(dl, Shift);
            dr = VMT::shift_right(dr, Shift);
        }
        dl = VMT::sub(dl, offset_v);
        dr = VMT::sub(dr, offset_v);

        return VMT::pack_unordered(dl, dr);
    }

    static INLINE void unpack_vectors(const TV baseVec, TV d01, TV& u01, TV& u02) {
        VMT::unpack_ordered(d01, u01, u02);

        u01 = VMT::add(u01, baseVec);
        u02 = VMT::add(u02, baseVec);

        if (Shift > 0) { // This is statically compiled in/out
            u01 = VMT::shift_left(u01, Shift);
            u02 = VMT::shift_left(u02, Shift);
        }
    }

};

} // namespace vxsort

#endif  // VXSORT_PACK_MACHINE_H
