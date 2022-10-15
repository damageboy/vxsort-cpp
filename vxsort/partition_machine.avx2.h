#ifndef VXSORT_PARTITION_MACHINE_AVX2_H
#define VXSORT_PARTITION_MACHINE_AVX2_H

#include "vector_machine/machine_traits.avx2.h"

#include "partition_machine.h"


#ifdef VXSORT_STATS
#include "stats/vxsort_stats.h"
#endif


namespace vxsort {
using namespace std;
using namespace vxsort::types;

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

template <>
struct partition_machine<i16, vector_machine::AVX2> {
    using T = i16;
    using VMT16 = vxsort_machine_traits<i16, vector_machine::AVX2>;
    using VMT32 = vxsort_machine_traits<i32, vector_machine::AVX2>;
    typedef typename VMT16::TV TV;
    static constexpr i32 N = sizeof(TV) / sizeof(T);
public:

    static INLINE void partition_block(TV& data_vec, const TV P,
                                       T* RESTRICT &left, T* RESTRICT &right)  {
        static_assert(!VMT16::supports_compress_writes());

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads();
        vxsort_stats<T>::bump_vec_stores(2);
        vxsort_stats<T>::bump_perms();
#endif

        auto mask = VMT16::get_cmpgt_mask(data_vec, P);

        TV u1, u2;

        auto m1 = mask &0xFF;
        auto m2 = mask >> 8;

        VMT32::unpack_ordered(data_vec, u1, u2);

        VMT32::partition_vector(u1, m1);
        VMT32::partition_vector(u2, m2);

        auto popcnt1 = -_mm_popcnt_u64(m1);
        auto popcnt2 = -_mm_popcnt_u64(m2);

        // A0|A0|A1|A1|A2|A2|A3|A3|A4|A4|A5|A5|A6|A6|A7|A7|
        // B0|B0|B1|B1|B2|B2|B3|B3|B4|B4|B5|B5|B6|B6|B7|B7|
        // A0|A1|A2|A3|B0|B1|B2|B3|A4|A5|A6|A7|B4|B5|B6|B7|
        auto packed = _mm256_packus_epi32(u1, u2);
        packed = _mm256_permute4x64_epi64(packed, 0b11'01'10'00);

        auto p1 = _mm256_extracti128_si256(packed, 0);
        auto p2 = _mm256_extracti128_si256(packed, 1);

        right += VMT32::N;

        _mm_storeu_si128(reinterpret_cast<__m128i *>(left), p1);
        left += popcnt1 + VMT32::N;
        _mm_storeu_si128(reinterpret_cast<__m128i *>(left), p2);
        left += popcnt2 + VMT32::N;
        _mm_storeu_si128(reinterpret_cast<__m128i *>(right), p2);
        right += popcnt2;
        _mm_storeu_si128(reinterpret_cast<__m128i *>(right), p1);
        right += popcnt1;

        right -= VMT32::N;
    }

    static inline void align_vectorized(const i32 left_masked_amount, const i32 right_unmasked_amount,
                                        const TV P,
                                        T* RESTRICT &read_left, T* RESTRICT &read_right,
                                        T* RESTRICT &spill_read_left, T* RESTRICT &spill_write_left,
                                        T* RESTRICT &spill_read_right, T* RESTRICT &spill_write_right) {
        static_assert(!VMT16::supports_compress_writes());
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

        auto max_base = VMT16::broadcast(numeric_limits<T>::max());
        auto min_base = VMT16::broadcast(numeric_limits<T>::min());
        auto lt_vec = VMT16::load_partial_vec((TV *)read_left, min_base, VMT16::generate_suffix_mask(left_masked_amount));
        auto rt_vec = VMT16::load_partial_vec((TV *)read_right, max_base, VMT16::generate_prefix_mask(right_unmasked_amount));
        read_left += N;
        read_right -= N;
        const auto rt_mask = VMT16::get_cmpgt_mask(rt_vec, P);
        const auto lt_mask = VMT16::get_cmpgt_mask(lt_vec, P);
        const auto rt_popcount_right_part = _mm_popcnt_u64(rt_mask);
        const auto lt_popcount_right_part = _mm_popcnt_u64(lt_mask);
        const auto rt_popcount_left_part = N - rt_popcount_right_part;
        const auto lt_popcount_left_part = N - lt_popcount_right_part;

#ifdef VXSORT_STATS
            vxsort_stats<T>::bump_perms(2);
#endif
        rt_vec = VMT16::partition_vector(rt_vec, rt_mask);
        lt_vec = VMT16::partition_vector(lt_vec, lt_mask);
        VMT16::store_vec((TV*)spill_write_right, rt_vec);
        VMT16::store_vec((TV*)spill_write_left, lt_vec);

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

        VMT16::store_vec((TV*)spill_write_right, lt_vec);
        spill_write_right -= lt_popcount_right_part;

        VMT16::store_vec((TV*)spill_write_left, rt_vec);
        spill_write_left += rt_popcount_left_part;
    }
};

}  // namespace vxsort

#endif //VXSORT_PARTITION_MACHINE_AVX2_H
