#ifndef VXSORT_MACHINE_TRAITS_H
#define VXSORT_MACHINE_TRAITS_H

#include <cstdint>
#include "defs.h"

namespace vxsort {
using namespace vxsort::types;

enum vector_machine {
    NONE,
    AVX2,
    AVX512,
    NEON,
    SVE,
};

#ifdef __cpp_concepts
#endif
/**
 * represent a subset of vector operations, relevant for vxsort (partitioning and bitonic sorting)
 * in a vector-isa (AVX2/AXX512/
 * @tparam T The primitive type being sorted
 * @tparam M The vector machine model/ISA (e.g. AVX2, AVX512 etc.)
 * @tparam Unroll The unroll factor, controls to some extent, the code-bloat/speedup ration at the call site
 *                Defaults to 1
 * @tparam Shift Optional; specify how many LSB bits are known to be zero in the original input. Can be used
 *               to further speed up sorting.
 */

template <typename T, vector_machine M>
struct vxsort_machine_traits {
public:
    typedef T TV;
    typedef T TLOADSTOREMASK;
    typedef T TMASK;
    typedef T TPACK;

    static constexpr bool supports_compress_writes() {
        static_assert(always_false<TV>, "must be specialized!");
        return false;
    }

    static constexpr bool supports_packing() {
        static_assert(always_false<TV>, "must be specialized!");
        return false;
    }

    template <int Shift>
    static constexpr bool can_pack(T) {
        static_assert(always_false<TV>, "must be specialized!");
        return false;
    }

    static TV load_vec(TV*) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    static TLOADSTOREMASK generate_prefix_mask(i32) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    static TLOADSTOREMASK generate_suffix_mask(i32) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    static void store_vec(TV*, TV) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    static TV load_masked_vec(TV *, TV, TLOADSTOREMASK) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    static void store_masked_vec(TV *, TV, TLOADSTOREMASK) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    static void store_compress_vec(TV*, TV, TMASK) {
        static_assert(always_false<TV>, "must be specialized!");
    }
    static TV partition_vector(TV v, i32 mask);
    static TV broadcast(T pivot);
    static TMASK get_cmpgt_mask(TV a, TV b);

    static TV shift_right(TV v, int i);
    static TV shift_left(TV v, int i);

    static TV add(TV a, TV b);
    static TV sub(TV a, TV b);

    static TV pack_unordered(TV a, TV b);

    static void unpack_ordered(TV, TV&, TV&) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    template <int Shift>
    static T shift_n_sub(T v, T) {
        static_assert(always_false<TV>, "must be specialized!");
        return v;
    }

    template <int Shift>
    static T unshift_and_add(TPACK, T add) {
        static_assert(always_false<TV>, "must be specialized!");
        return add;
    }
};

}

#endif  // VXSORT_MACHINE_TRAITS_H
