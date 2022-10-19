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

    static constexpr i32 N = 1;

    /// Indicates if this type/vector-machine combination supports
    /// compressed writes (e.g. when vector elements are stored in a
    /// packed fashion according to a predicate mask)
    /// \return true when supported, false otherwise
    static constexpr bool supports_compress_writes() {
        static_assert(always_false<TV>, "must be specialized!");
        return false;
    }

    /// Indicates if this type/vector-machine combination supports
    /// packing to a lower bit-width type opportunistically during
    /// sort operations and unpacking back to the original type/width
    /// before returning to the caller
    /// \return true when supported, false otherwise
    static constexpr bool supports_packing() {
        static_assert(always_false<TV>, "must be specialized!");
        return false;
    }

    /// Check if the supplied range can be packed into a lower bit-width
    /// primitive type (e.g. 32-bit integers can be packed into 16-bit)
    /// while sorting
    /// \param[in] range the range of the values inside the partition
    /// \return true if the range can be packed, false otherwise
    template <i32 Shift>
    static bool can_pack(T range) {
        static_assert(always_false<TV>, "must be specialized!");
        return false;
    }

    /// Load a vector from memory
    /// \param[in] p the pointer to the memory location to load from
    /// \return the loaded vector
    static TV load_vec(TV* p) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    /// Generate a vector mask that can be used to loads N consecutive
    /// elements from memory into the first N elements of a vector type, making off non-loaded
    /// elements from being read/stored.
    /// \param[in] n the number of elements to load
    /// \return the generated mask
    static TLOADSTOREMASK generate_prefix_mask(i32 n) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    /// Generate a vector mask that can be used to loads N consecutive
    /// elements from memory into the last N elements of a vector type, making off non-loaded
    /// elements from being read/stored.
    static TLOADSTOREMASK generate_suffix_mask(i32) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    /// Store a vector to memory
    /// \param[in] p the pointer to the memory location to store to
    /// \param[in] v the vector to store
    static void store_vec(TV* p, TV v) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    /// Load a vector from memory, using a mask to control which elements
    /// are to be skipped/not-read from
    /// \param[in] p the pointer to the memory location to load from
    /// \param[in] base A vector with elements to be used in-place of the masked-off elements
    /// \param[in] mask the mask to control which elements are to be read
    /// \return the loaded vector with a mix of elements from memory and the
    ///         base vector according to the provided mask
    static TV load_partial_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    /// Store a vector to memory, using a mask to control which elements
    /// are to be skipped/not-written to
    /// \param[in] p the pointer to the memory location to store to
    /// \param[in] v A vector with elements to be written to memory, according to the mask
    /// \param[in] mask the mask to control which elements are to be read
    static void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    /// Store a vector to memory, using a mask to control which elements
    /// are to be skipped/not-written to.
    /// \note Compressed stores are used when the vector elements are stored in a packed fashion
    ///       according to the provided mask, e.g. the compression mask instructs
    ///       the vector machine to store the required elements in a packed fashion
    ///       into memory.
    /// \param[in] p the pointer to the memory location to store to
    /// \param[in] v A vector with elements to be written to memory, according to the mask
    /// \param[in] mask the mask to control which elements are to be read
    static void store_compress_vec(TV* p, TV v, TMASK mask) {
        static_assert(always_false<TV>, "must be specialized!");
    }
    static TV partition_vector(TV v, i32 mask);
    static TV broadcast(T pivot);
    static TMASK get_cmpgt_mask(TV a, TV b);

    static TV shift_right(TV v, i32 i);
    static TV shift_left(TV v, i32 i);

    static TV add(TV a, TV b);
    static TV sub(TV a, TV b);

    static TV pack_unordered(TV a, TV b);

    static void unpack_ordered(TV, TV&, TV&) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    template <i32 Shift>
    static T shift_n_sub(T v, T) {
        static_assert(always_false<TV>, "must be specialized!");
        return v;
    }

    template <i32 Shift>
    static T unshift_and_add(TPACK, T add) {
        static_assert(always_false<TV>, "must be specialized!");
        return add;
    }
};

}

#endif  // VXSORT_MACHINE_TRAITS_H
