#include "vxsort_stats.h"
#ifdef VXSORT_STATS

namespace vxsort {
int vxsort_stats_base::last_type = 0;
std::array<vxsort_type, 6> vxsort_stats_base::registered_types;

template<typename T>
void vxsort_stats<T>::print_stats()
{
    printf("%9s | ", vxsort_stats_base::vxsort_type_to_str(vxsort_stats_base::typeid_to_vxsort_type<T>()));
    printf("%7ld | ", _num_sorts);
    printf("%8ld | ", _total_sort_size);
    printf("%8ld | ", _num_partitions);
    printf("%11ld | ", _num_vec_loads);
    printf("%12ld | ", _num_vec_stores);
    printf("%10ld | ", _num_small_sorts);
    printf("%9.2f%% | ", (double) _small_sorts_size * 100 / (double) _total_partitioned_size);
    printf("%9.2f\n", (double) _small_sorts_size / (double)  _num_small_sorts);
}

extern void print_all_stats() {
    printf("type      | # sorts | # sorted | # part's | # vec loads | # vec stores | # sm. sort | %% sm. sort | avg. sm. sort\n");
    printf("----------|---------|----------|----------|-------------|--------------|------------|------------|--------------\n");
    for (auto i = 0; i < vxsort_stats_base::last_type; i++) {
        switch (vxsort_stats_base::registered_types[i]) {
            case INT16:
                vxsort_stats<int16_t>::print_stats();
                break;
            case UINT16:
                vxsort_stats<uint16_t>::print_stats();
                break;
            case INT32:
                vxsort_stats<int32_t>::print_stats();
                break;
            case UINT32:
                vxsort_stats<uint32_t>::print_stats();
                break;
            case INT64:
                vxsort_stats<int64_t>::print_stats();
                break;
            case UINT64:
                vxsort_stats<uint64_t>::print_stats();
                break;
            case FLOAT:
                vxsort_stats<float>::print_stats();
                break;
            case DOUBLE:
                vxsort_stats<double>::print_stats();
                break;
        }
    }
}

extern void reset_all_stats() {
    for (auto i = 0; i < vxsort_stats_base::last_type; i++) {
        switch (vxsort_stats_base::registered_types[i]) {
            case INT16:
                vxsort_stats<int16_t>::reset();
                break;
            case UINT16:
                vxsort_stats<uint16_t>::reset();
                break;
            case INT32:
                vxsort_stats<int32_t>::reset();
                break;
            case UINT32:
                vxsort_stats<uint32_t>::reset();
                break;
            case INT64:
                vxsort_stats<int64_t>::reset();
                break;
            case UINT64:
                vxsort_stats<uint64_t>::reset();
                break;
            case FLOAT:
                vxsort_stats<float>::reset();
                break;
            case DOUBLE:
                vxsort_stats<double>::reset();
                break;
        }
    }
}


}

#endif