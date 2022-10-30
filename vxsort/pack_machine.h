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

template<typename T, vector_machine M, i32 Shift>
class pack_machine {
    static_assert(Shift <= 31, "Shift must be in the range 0..31");

    using VMT = vxsort_machine_traits<T, M>;
    typedef typename VMT::TV TV;
    static const i32 N = sizeof(TV) / sizeof(T);
    typedef alignment_hint<T, M> AH;

public:

    /// pack the provided vectors into a lower bit-width type after offestting them by a known base value
    /// \param[in] u1 a vector containing the first half of the input
    /// \param[in] u2 a vector containing the second half of the input
    /// \param[in] offset_v a vector containing the base value to use for offsetting each element before packing
    /// \return a vector containing the packed values after readjusting them to the supplied base value
    static INLINE TV prepare_offset(T min_value)
    {
        // Create a vectorized version of the offset by which we need to
        // correct the data before packing it
        auto constexpr MIN = T(std::numeric_limits<typename VMT::TPACK>::min());
        auto offset = VMT::template shift_n_sub<Shift>(min_value, MIN);
        return VMT::broadcast(offset);
    }

    static INLINE TV pack_vectors(TV u1, TV u2, const TV offset_v) {
        // This is statically compiled in/out
        if (Shift > 0) {
            u1 = VMT::shift_right(u1, Shift);
            u2 = VMT::shift_right(u2, Shift);
        }
        u1 = VMT::sub(u1, offset_v);
        u2 = VMT::sub(u2, offset_v);

        return VMT::pack_unordered(u1, u2);
    }

    /// unpack the provided vector into two higher bit-width vectors, then offset them by a known base value
    /// \param[in] offset_v a vector containing the base value to use for offsetting each element after unpacking
    /// \param[in] p a vector containing the packed input values
    /// \param[out] u1 a vector containing the first half of the output
    /// \param[out] u2 a vector containing the second half of the output
    static INLINE void unpack_vectors(const TV offset_v, TV p, TV& u1, TV& u2) {
        VMT::unpack_ordered(p, u1, u2);

        u1 = VMT::add(u1, offset_v);
        u2 = VMT::add(u2, offset_v);

        if (Shift > 0) { // This is statically compiled in/out
            u1 = VMT::shift_left(u1, Shift);
            u2 = VMT::shift_left(u2, Shift);
        }
    }
};

} // namespace vxsort

#endif  // VXSORT_PACK_MACHINE_H
