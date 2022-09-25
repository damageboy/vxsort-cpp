#ifndef VXSORT_PARTITION_MACHINE_H
#define VXSORT_PARTITION_MACHINE_H

#include <cstdint>
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
    typedef alignment_hint<sizeof(TV)> AH;
public:

    static INLINE void partition_block(TV& data_vec, const TV P,
                                       T* RESTRICT &left, T* RESTRICT &right) {
        static_assert(always_false<TV>, "must be specialized!");
    }

    void align_vectorized(const i32 left_align, const i32 right_align,
                          const TV P,
                          T* RESTRICT &read_left, T* RESTRICT &read_right,
                          T* RESTRICT &tmp_start_left, T* RESTRICT &tmp_left,
                          T* RESTRICT &tmp_start_right, T* RESTRICT &tmp_right) {
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
};

}  // namespace vxsort

#endif //VXSORT_PARTITION_MACHINE_H
