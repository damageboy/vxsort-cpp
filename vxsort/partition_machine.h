#ifndef VXSORT_PARTITION_MACHINE_H
#define VXSORT_PARTITION_MACHINE_H

#include <cstdint>
#include <limits>
#include <immintrin.h>

#include "defs.h"
#include "vector_machine/machine_traits.h"


#ifdef VXSORT_STATS
#include "stats/vxsort_stats.h"
#endif


namespace vxsort {
using namespace std;
using namespace vxsort::types;

template <typename T, vector_machine M>
struct partition_machine {
    using VMT = vxsort_machine_traits<T, M>;
    typedef typename VMT::TV TV;
public:

    static INLINE void partition_block(TV& data_vec, const TV P,
                                       T* RESTRICT &left, T* RESTRICT &right) {
        static_assert(always_false<TV>, "must be specialized!");
    }


    /// Prime the partition "pump" by reading and aligning up to the one vector worth
    /// of elements fro each side of the input partition. The actual amount of data
    /// to be partitioned depends on the next alignment point for future vector reads:
    /// By reading some exact amount of each side, by the end of this function, future
    /// reads can perform 100% aligned loads, thereby reducing the internal resources
    /// consumed by modern HW when dealing with un-aligned, or worse-yet cache-line
    /// striped loads
    /// @param[in] left_masked_amount the amount of elements, prior to
    ///                               @p read_left that are to be discarded;
    ///                               a zero (0) value is a special value that denotes that all values
    ///                               are to be used (e.g. 0 discarded)
    /// @param[in] right_unmasked_amount the amount of elements, prior to
    ///                                  @p read_right that are to be partitioned;
    ///                                  a zero (0) value is a special value that denotes that all values
    ///                                  are to be used (e.g. 0 discarded)
    /// @param[in] P The vector pivot value
    /// @param[inout] read_left A reference to the current left-side read-position,
    ///                         modified to the next read-position by the end of this function
    /// @param[inout] read_right A reference to the current right-side read-position,
    ///                          modified to the next read-position by the end of this function
    /// @param[inout] spill_read_left A reference to the spill-buffer's copy-from left-side
    ///                              read-position. This will reflect the discarded elements
    ///                              by the end of the this function
    /// @param[inout] spill_write_left A reference to the spill-buffer's left-side write-position
    ///                                This will reflect the next valid vector write position by
    ///                                the end of this function.
    /// @param[inout] spill_read_right A reference to the spill-buffer's copy-from right-side
    ///                               read-position. This will reflect the discarded elements
    ///                               by the end of this function
    /// @param[inout] spill_write_right A reference to the spill-buffer's right-side write-position
    ///                                 This will reflect eh next valid vector-write position by
    ///                                 the end of this function.
    static inline void align_vectorized(const i32 left_masked_amount, const i32 right_unmasked_amount,
                                        const TV P,
                                        T* RESTRICT &read_left, T* RESTRICT &read_right,
                                        T* RESTRICT &spill_read_left, T* RESTRICT &spill_write_left,
                                        T* RESTRICT &spill_read_right, T* RESTRICT &spill_write_right) {
        static_assert(always_false<TV>, "must be specialized!");
    }
};

}  // namespace vxsort

#endif //VXSORT_PARTITION_MACHINE_H
