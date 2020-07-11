#ifndef VXSORT_VXSORT_STATS_H
#define VXSORT_VXSORT_STATS_H

#ifdef VXSORT_STATS

#include <array>
#include <cstdint>
#include <cstdio>
#include <typeinfo>


using namespace std;

namespace vxsort {

enum vxsort_type {
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    FLOAT,
    DOUBLE,
};


class vxsort_stats_base
{
   public:
    static std::array<vxsort_type, 6> registered_types;
    static int last_type;
   protected:
    static void reset()
    {
        last_type = 0;
    }

    template<typename T>
    static vxsort_type typeid_to_vxsort_type() {
        if (typeid(T) == typeid(int16_t))
            return vxsort_type::INT16;
        else if (typeid(T) == typeid(int32_t))
            return vxsort_type::INT32;
        else if (typeid(T) == typeid(int64_t))
            return vxsort_type::INT64;

        if (typeid(T) == typeid(uint16_t))
            return vxsort_type::UINT16;
        else if (typeid(T) == typeid(uint32_t))
            return vxsort_type::UINT32;
        else if (typeid(T) == typeid(uint64_t))
            return vxsort_type::UINT64;

        if (typeid(T) == typeid(float))
            return vxsort_type::FLOAT;
        if (typeid(T) == typeid(double))
            return vxsort_type::DOUBLE;

        throw new std::logic_error("Unsupported type");
    }

    static const char *vxsort_type_to_str(const vxsort_type type) {
        switch (type) {
            case INT16:  return "int16_t";
            case UINT16: return "uint16_t";
            case INT32:  return "int32_t";
            case UINT32: return "uint32_t";
            case INT64:  return "int64_t";
            case UINT64: return "uint64_t";
            case FLOAT:  return "float";
            case DOUBLE: return "double";
        }
    }

    static void register_stat(const vxsort_type type)
    {
        for (auto i = 0; i < last_type; i++)
            if (registered_types[i] == type)
                return;

        registered_types[last_type++] = type;
    }
};

template<typename T>
class vxsort_stats : vxsort_stats_base
{
   private:
    static uint64_t _num_sorts;
    static uint64_t _total_sort_size;
    static uint64_t _num_partitions;
    static uint64_t _total_partitioned_size;
    static uint64_t _num_small_sorts;
    static uint64_t _small_sorts_size;
    static uint64_t _packed_elements;
    static uint64_t _unpacked_elements;
    static uint64_t _num_perms;
    static uint64_t _num_vec_loads;
    static uint64_t _num_vec_stores;

   public:
    static void reset()
    {
        _num_sorts = 0;
        _total_sort_size = 0;
        _num_partitions = 0;
        _total_partitioned_size = 0;
        _num_small_sorts = 0;
        _small_sorts_size = 0;
        _packed_elements = 0;
        _unpacked_elements = 0;
        _num_perms = 0;
        _num_vec_loads = 0;
        _num_vec_stores = 0;

    }
    static void bump_sorts(size_t n) {
        _num_sorts++;
        _total_sort_size += n;
        vxsort_stats_base::register_stat(vxsort_stats_base::typeid_to_vxsort_type<T>());
    }
    static void bump_partitions(size_t n) {
        _num_partitions++;
        _total_partitioned_size += n;
    }
    static void bump_small_sorts(int n = 1) { _num_small_sorts++; }
    static void bump_perms(size_t perms = 1) { _num_perms += perms; }
    static void bump_vec_loads(size_t loads = 1) { _num_vec_loads += loads; }
    static void bump_vec_stores(size_t stores = 1) { _num_vec_stores += stores; }
    static void record_small_sort_size(size_t sort_size) { _small_sorts_size += sort_size; }
    static void bump_packs(size_t size) { _packed_elements += size; }
    static void bump_unpacks(size_t size) { _unpacked_elements += size; }

    static void print_stats();
};

extern void reset_all_stats();
extern void print_all_stats();

template <typename T>
uint64_t vxsort_stats<T>::_num_sorts = 0;
template <typename T>
uint64_t vxsort_stats<T>::_total_sort_size = 0;
template <typename T>
uint64_t vxsort_stats<T>::_num_partitions = 0;
template <typename T>
uint64_t vxsort_stats<T>::_total_partitioned_size = 0;
template <typename T>
uint64_t vxsort_stats<T>::_num_small_sorts = 0;
template <typename T>
uint64_t vxsort_stats<T>::_small_sorts_size = 0;
template <typename T>
uint64_t vxsort_stats<T>::_packed_elements = 0;
template <typename T>
uint64_t vxsort_stats<T>::_unpacked_elements = 0;
template <typename T>
uint64_t vxsort_stats<T>::_num_perms = 0;
template <typename T>
uint64_t vxsort_stats<T>::_num_vec_loads = 0;
template <typename T>
uint64_t vxsort_stats<T>::_num_vec_stores = 0;

}

#endif
#endif  // VXSORT_VXSORT_STATS_H
