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
    typedef T TCMPMASK;
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
    /// \note This mask is mostly useful for accessing the <b>end</b> of an array, where the
    ///       the masked-off elements are potentially outside the array bounds and may cause a
    ///       page-fault/segmentation-fault when accessed.
    /// \param[in] n the number of elements to load into the beginning of the vector, while masking-off
    ///              (therefore discarding) the remaining elements.
    ///              a zero (0) value is a special value that denotes that all values
    ///              are to be used (e.g. 0 discarded)
    /// \return the generated mask
    static TLOADSTOREMASK generate_prefix_mask(i32 n) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    /// Generate a vector mask that can be used to loads N-m consecutive
    /// elements from memory into the last N-m elements of a vector type, making off non-loaded
    /// elements from being read/stored.
    /// \note This mask is mostly useful for accessing the <b>beginning</b> of an array, where the
    ///       the masked-off elements are potentially outside the array bounds and may cause a
    ///       page-fault/segmentation-fault when accessed.
    /// \param[in] m the number of elements to mask-off (therefore discard) from the beginning of
    ///              the vector, while loading the other elements into the end of the vector.
    ///              a zero (0) value is a special value that denotes that all values
    ///              are to be used (e.g. 0 discarded)
    /// \return the generated mask
    static TLOADSTOREMASK generate_suffix_mask(i32 m) {
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
    static void store_compress_vec(TV* p, TV v, TCMPMASK mask) {
        static_assert(always_false<TV>, "must be specialized!");
    }


    static TV partition_vector(TV v, i32 mask);

    /// Broadcast a scalar value to all elements of a vector
    /// \param[in] v the scalar value to braodcast vector-wide
    /// \return the vector containing the broadcasted value
    static TV broadcast(T v);

    /// perform a vectorized comparison between two vectors returning
    /// a vector wide mask of the comparison results
    /// \param[in] a the left hand side vector
    /// \param[in] b the right hand side vector
    /// \return The comparison result as a mask
    static TCMPMASK get_cmpgt_mask(TV a, TV b);

    /// Perform arithmetic right-shift on a vector
    /// \param[in] v the inpyt vector
    /// \param[in] i the amout of bits to shift each element right by
    /// \return the shifted vector
    static TV shift_right(TV v, i32 i);

    /// Perform left-shift on a vector
    /// \param[in] v the inpyt vector
    /// \param[in] i the amout of bits to shift each element left by
    /// \return the shifted vector
    static TV shift_left(TV v, i32 i);

    /// Perform vector addition between two vectors
    /// \param[in] a the first vector
    /// \param[in] b the second vector
    /// \return a vector containing the sum of the two input vectors
    static TV add(TV a, TV b);

    /// Perform vector subtraction between two vectors
    /// \param[in] a the first vector
    /// \param[in] b the second vector
    /// \return a vector containing the subtraction between the two input vectors
    static TV sub(TV a, TV b);

    /// Perform unordered vector packing between two vectors, halving the bit-width
    /// of the input vectors
    /// \param[in] u1 the first vector
    /// \param[in] u2 the second vector
    /// \return a vector containing the packed elements of the two input vectors
    /// \note it is assumed that the order of packing between both vectors need not be preserved
    static TV pack_unordered(TV u1, TV u2);

    /// Perform unordered vector packing between two vectors, halving the bit-width
    /// of the input vectors
    /// \param[in] p the packed input vector
    /// \param[out] u1 a destination vector for the first half of the unpacked elements
    /// \param[out] u2 a destination vector for the second half of the unpacked elements
    static void unpack_ordered(TV p, TV& u1, TV& u2) {
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
