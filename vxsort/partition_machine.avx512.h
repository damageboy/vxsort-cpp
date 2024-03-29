#ifndef VXSORT_PARTITION_MACHINE_AVX512_H
#define VXSORT_PARTITION_MACHINE_AVX512_H

#include "vector_machine/machine_traits.avx512.h"

#include "partition_machine.h"


#ifdef VXSORT_STATS
#include "stats/vxsort_stats.h"
#endif

namespace vxsort {
using namespace std;
using namespace vxsort::types;

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
        auto lt_vec = VMT::load_partial_vec((TV *)read_left, min_base, VMT::generate_suffix_mask(left_masked_amount));
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

#endif //VXSORT_PARTITION_MACHINE_AVX512_H
