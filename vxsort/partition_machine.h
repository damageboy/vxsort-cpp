#ifndef VXSORT_PARTITION_MACHINE_H
#define VXSORT_PARTITION_MACHINE_H

#include <cstdint>
#include <immintrin.h>

#include "defs.h"
#include "alignment.h"
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
    typedef alignment_hint<T, M> AH;
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

template <typename T>
struct partition_machine<T, vector_machine::AVX2> {
    using VMT = vxsort_machine_traits<T, vector_machine::AVX2>;
    typedef typename VMT::TV TV;
    static constexpr i32 N = sizeof(TV) / sizeof(T);
public:

    static INLINE void partition_block(TV& data_vec, const TV P,
                                       T* RESTRICT &left, T* RESTRICT &right)  {
        static_assert(!VMT::supports_compress_writes());

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads();
        vxsort_stats<T>::bump_vec_stores(2);
        vxsort_stats<T>::bump_perms();
#endif

        auto mask = VMT::get_cmpgt_mask(data_vec, P);
        data_vec = VMT::partition_vector(data_vec, mask);
        VMT::store_vec(reinterpret_cast<TV*>(left), data_vec);
        VMT::store_vec(reinterpret_cast<TV*>(right), data_vec);
        auto popcnt = -_mm_popcnt_u64(mask);
        right += popcnt;
        left += popcnt + N;
    }

    static inline void align_vectorized(const i32 left_masked_amount, const i32 right_unmasked_amount,
                                        const TV P,
                                        T* RESTRICT &read_left, T* RESTRICT &read_right,
                                        T* RESTRICT &spill_read_left, T* RESTRICT &spill_write_left,
                                        T* RESTRICT &spill_read_right, T* RESTRICT &spill_write_right) {
        static_assert(!VMT::supports_compress_writes());
#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads(2);
        vxsort_stats<T>::bump_vec_stores(4);
#endif
        //                                        ┌  next alignment ┐
        //                            left        :                 :    right   ┌─ elements to be masked-out
        // elements to be masked  ─┐  │           :                 :        │   │  (replaced with MAX_VALUE)
        // out (replace with       │  │           :                 :        │   │  and discarded later
        // MIN_VALUE) and          ▼  ▼           :                 :        ▼   ▼
        // discarded later        ┌───┬───────────────────────────────────────┬─────┐
        //                        │< <│< > > > > > x x x x x x x x x < < > > P│> > >│
        //                        └┬──┴──────────┬───────────────────┬────────┴────┬┘
        //                         └──────┬──────┘:                 :└──────┬──────┘
        //                                │       :                 :       │
        //                                ▼       :                 :       ▼
        //                           lt_vec                                 rt_vec
        //
        // Legend:
        // x - elements we aren't processing during alignment
        // P - the pivot element, placed as the last element in the partition,
        //     pointed to by `right`
        // > - elements that we are partitioning, greater than (>) the pivot
        // < - elements that we are partitioning, smaller-or-equal than (<=) the pivot

        // Alignment with vectorization is tricky, so read carefully before changing code:
        // 1. We load data, from both ends, while masking out-of-partition
        //    (and potentially out of bounds!) elements with masked vector load operations
        //    replacing masked out elements with:
        //    * for the right-boundary read with numeric_limits<T>::max()
        //    * for the left-boundary read with numeric_limits<T>::min()
        // 2. We partition and store in the following order:
        //    a) right-portion of right vector to the right-side
        //    b) left-portion of left vector to the left side
        //
        //    -- at this point one-half of each partitioned vector has been committed
        //       back to memory: we know that the out-of-bounds elements that we
        //       NEVER read, and instead REPLACED with min/max values are *on* the edges
        //       of the temp spill buffer, allowing us to mark how many such elements were
        //       extraneously written to the spill buffer on both ends, to skip copying them
        //       back at the end of the partition operation!
        //
        //    c) we advance the right write (spill_write_right) pointer by how many elements
        //       were actually needed to be written to the right hand side
        //    d) We write the right portion of the left vector to the right side
        //       now that its write position has been updated

        // adjust the read position for vector reads
        read_left -= left_masked_amount;
        read_right -= right_unmasked_amount;

        // adjust write position for vector writes
        spill_write_right -= N;

        auto max_base = VMT::broadcast(numeric_limits<T>::max());
        auto min_base = VMT::broadcast(numeric_limits<T>::min());
        auto lt_vec = VMT::load_partial_vec((TV *)read_left,  min_base, VMT::generate_suffix_mask(left_masked_amount));
        auto rt_vec = VMT::load_partial_vec((TV *)read_right, max_base, VMT::generate_prefix_mask(right_unmasked_amount));
        read_left += N;
        read_right -= N;
        const auto rt_mask = VMT::get_cmpgt_mask(rt_vec, P);
        const auto lt_mask = VMT::get_cmpgt_mask(lt_vec, P);
        const auto rt_popcount_right_part = _mm_popcnt_u64(rt_mask);
        const auto lt_popcount_right_part = _mm_popcnt_u64(lt_mask);
        const auto rt_popcount_left_part = N - rt_popcount_right_part;
        const auto lt_popcount_left_part = N - lt_popcount_right_part;



#ifdef VXSORT_STATS
            vxsort_stats<T>::bump_perms(2);
#endif
        rt_vec = VMT::partition_vector(rt_vec, rt_mask);
        lt_vec = VMT::partition_vector(lt_vec, lt_mask);
        VMT::store_vec((TV*)spill_write_right, rt_vec);
        VMT::store_vec((TV*)spill_write_left, lt_vec);

        spill_write_left += lt_popcount_left_part;
        spill_write_right -= rt_popcount_right_part;

        // At this after (a)+(b) (consult the comment above) were completed
        // We will adjust the spill_read_right+spill_read_left to skip over the
        // extra elements

        // left_masked_amount == 0 means all the elements
        // should be preserved so bumping spill_read_left by 0 is desired
        spill_read_left += left_masked_amount;

        // When right_unmasked_amount == 0 we should not discard any element
        spill_read_right -= N - right_unmasked_amount;

        VMT::store_vec((TV*)spill_write_right, lt_vec);
        spill_write_right -= lt_popcount_right_part;

        VMT::store_vec((TV*)spill_write_left, rt_vec);
        spill_write_left += rt_popcount_left_part;
    }
};

template <typename T>
struct partition_machine<T, vector_machine::AVX512> {
    using VMT = vxsort_machine_traits<T, vector_machine::AVX512>;
    typedef typename VMT::TV TV;
    static constexpr i32 N = sizeof(TV) / sizeof(T);
public:
    static INLINE void partition_block(TV& data_vec, const TV P,
                                       T* RESTRICT &left, T* RESTRICT &right)  {
        static_assert(VMT::supports_compress_writes());

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads();
        vxsort_stats<T>::bump_vec_stores(2);
#endif

        auto mask = VMT::get_cmpgt_mask(data_vec, P);
        auto popcnt = -_mm_popcnt_u64(mask);
        VMT::store_compress_vec(reinterpret_cast<TV*>(left), data_vec, ~mask);
        VMT::store_compress_vec(reinterpret_cast<TV*>(right + N + popcnt), data_vec, mask);
        right += popcnt;
        left += popcnt + N;
    }

    static inline void align_vectorized(const i32 left_masked_amount, const i32 right_unmasked_amount,
                                        const TV P,
                                        T* RESTRICT &read_left, T* RESTRICT &read_right,
                                        T* RESTRICT &spill_read_left, T* RESTRICT &spill_write_left,
                                        T* RESTRICT &spill_read_right, T* RESTRICT &spill_write_right) {
        static_assert(VMT::supports_compress_writes());
#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads(2);
        vxsort_stats<T>::bump_vec_stores(4);
#endif
        //                                        ┌  next alignment ┐
        //                            left        :                 :    right   ┌─ elements to be masked-out
        // elements to be masked  ─┐  │           :                 :        │   │  (replaced with MAX_VALUE)
        // out (replace with       │  │           :                 :        │   │  and discarded later
        // MIN_VALUE) and          ▼  ▼           :                 :        ▼   ▼
        // discarded later        ┌───┬───────────────────────────────────────┬─────┐
        //                        │< <│< > > > > > x x x x x x x x x < < > > P│> > >│
        //                        └┬──┴──────────┬───────────────────┬────────┴────┬┘
        //                         └──────┬──────┘:                 :└──────┬──────┘
        //                                │       :                 :       │
        //                                ▼       :                 :       ▼
        //                           lt_vec                                 rt_vec
        //
        // Legend:
        // x - elements we aren't processing during alignment
        // P - the pivot element, placed as the last element in the partition,
        //     pointed to by `right`
        // > - elements that we are partitioning, greater than (>) the pivot
        // < - elements that we are partitioning, smaller-or-equal than (<=) the pivot

        // Alignment with vectorization is tricky, so read carefully before changing code:
        // 1. We load data, from both ends, while masking out-of-partition
        //    (and potentially out of bounds!) elements with masked vector load operations
        //    replacing masked out elements with:
        //    * for the right-boundary read with numeric_limits<T>::max()
        //    * for the left-boundary read with numeric_limits<T>::min()
        // 2. We partition and store in the following order:
        //    a) right-portion of right vector to the right-side
        //    b) left-portion of left vector to the left side
        //
        //    -- at this point one-half of each partitioned vector has been committed
        //       back to memory: we know that the out-of-bounds elements that we
        //       NEVER read, and instead REPLACED with min/max values are *on* the edges
        //       of the temp spill buffer, allowing us to mark how many such elements were
        //       extraneously written to the spill buffer on both ends, to skip copying them
        //       back at the end of the partition operation!
        //
        //    c) we advance the right write (spill_write_right) pointer by how many elements
        //       were actually needed to be written to the right hand side
        //    d) We write the right portion of the left vector to the right side
        //       now that its write position has been updated

        // adjust the read position for vector reads
        read_left -= left_masked_amount;
        read_right -= right_unmasked_amount;

        // adjust write position for vector writes
        spill_write_right -= N;


        auto max_base = VMT::broadcast(numeric_limits<T>::max());
        auto min_base = VMT::broadcast(numeric_limits<T>::min());
        auto lt_vec = VMT::load_partial_vec((TV *)read_left,  min_base, VMT::generate_suffix_mask(left_masked_amount));
        auto rt_vec = VMT::load_partial_vec((TV *)read_right, max_base, VMT::generate_prefix_mask(right_unmasked_amount));
        read_left += N;
        read_right -= N;
        const auto rt_mask = VMT::get_cmpgt_mask(rt_vec, P);
        const auto lt_mask = VMT::get_cmpgt_mask(lt_vec, P);
        const auto rt_popcount_right_part = _mm_popcnt_u64(rt_mask);
        const auto lt_popcount_right_part = _mm_popcnt_u64(lt_mask);
        const auto rt_popcount_left_part = N - rt_popcount_right_part;
        const auto lt_popcount_left_part = N - lt_popcount_right_part;

        // spill_write_right points to the next valid write position of
        // a full N-element vector, so we have to adjust the pointer
        // UP by the number of elements NOT written
        VMT::store_compress_vec((TV *)spill_write_left, lt_vec, ~lt_mask);
        VMT::store_compress_vec((TV *)(spill_write_right + rt_popcount_left_part), rt_vec, rt_mask);

        spill_write_left += lt_popcount_left_part;
        spill_write_right -= rt_popcount_right_part;


        // At this after (a)+(b) (consult the comment above) were completed
        // We will adjust the spill_read_right+spill_read_left to skip over the
        // extra elements

        // left_masked_amount == 0 means all the elements
        // should be preserved so bumping spill_read_left by 0 is desired
        spill_read_left += left_masked_amount;

        // When right_unmasked_amount == 0 we should not discard any element
        spill_read_right -= N - right_unmasked_amount;

        VMT::store_compress_vec((TV *) (spill_write_right + lt_popcount_left_part), lt_vec, lt_mask);
        spill_write_right -= lt_popcount_right_part;

        VMT::store_compress_vec((TV*)spill_write_left, rt_vec, ~rt_mask);
        spill_write_left += rt_popcount_left_part;
    }

};

}  // namespace vxsort

#endif //VXSORT_PARTITION_MACHINE_H
