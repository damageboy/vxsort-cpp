#include "vxsort_stats.h"
#ifdef VXSORT_STATS

#include <fmt/format.h>

namespace vxsort {
using namespace vxsort::types;

i32 vxsort_stats_base::last_type = 0;
std::array<vxsort_type, 6> vxsort_stats_base::registered_types;

template<typename T>
void vxsort_stats<T>::print_stats()
{
    fmt::print("{:9} | {:7} | {:8} | {:7} | {:11} | {:12} | {:10} | {:9.2}% | {:>9.2f}\n",
               vxsort_type_to_str(typeid_to_vxsort_type<T>()),
               _num_sorts,
               _total_sort_size,
               _num_partitions,
               _num_vec_loads,
               _num_vec_stores,
               _num_small_sorts,
               (f64) _small_sorts_size * 100 / (f64) _total_partitioned_size,
               (f64) _small_sorts_size / (f64)  _num_small_sorts);
}

extern void print_all_stats() {
    fmt::print("type      | # sorts | # sorted | # parts | # vec loads | # vec stores | # sm. sort | % sm. sort | avg. sm. sort\n");
    fmt::print("----------|---------|----------|---------|-------------|--------------|------------|------------|--------------\n");
    for (auto i = 0; i < vxsort_stats_base::last_type; ++i) {
        switch (vxsort_stats_base::registered_types[i]) {
            case vxsort_type::I16:  vxsort_stats<i16>::print_stats(); break;
            case vxsort_type::U16:  vxsort_stats<u16>::print_stats(); break;
            case vxsort_type::I32:  vxsort_stats<i32>::print_stats(); break;
            case vxsort_type::U32:  vxsort_stats<u32>::print_stats(); break;
            case vxsort_type::I64:  vxsort_stats<i64>::print_stats(); break;
            case vxsort_type::U64:  vxsort_stats<u64>::print_stats(); break;
            case vxsort_type::F32:  vxsort_stats<f32>::print_stats(); break;
            case vxsort_type::F64:  vxsort_stats<f64>::print_stats(); break;
            case vxsort_type::NONE: break;
        }
    }
}

extern void reset_all_stats() {
    for (auto i = 0; i < vxsort_stats_base::last_type; i++) {
        switch (vxsort_stats_base::registered_types[i]) {
            case vxsort_type::I16:  vxsort_stats<i16>::reset(); break;
            case vxsort_type::U16:  vxsort_stats<u16>::reset(); break;
            case vxsort_type::I32:  vxsort_stats<i32>::reset(); break;
            case vxsort_type::U32:  vxsort_stats<u32>::reset(); break;
            case vxsort_type::I64:  vxsort_stats<i64>::reset(); break;
            case vxsort_type::U64:  vxsort_stats<u64>::reset(); break;
            case vxsort_type::F32:  vxsort_stats<f32>::reset(); break;
            case vxsort_type::F64:  vxsort_stats<f64>::reset(); break;
            case vxsort_type::NONE: break;
        }
    }
}

} // namespace vxsort

#endif