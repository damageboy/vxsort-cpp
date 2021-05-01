#ifndef VXSORT_VXSORT_STATS_H
#define VXSORT_VXSORT_STATS_H

#ifdef VXSORT_STATS

#include <array>
#include <cstdint>
#include <cstdio>
#include <typeinfo>
#include "defs.h"


using namespace std;
namespace vxsort {
using namespace vxsort::types;
enum vxsort_type {
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
};

class vxsort_stats_base
{
   public:
    static std::array<vxsort_type, 6> registered_types;
    static int last_type;
   protected:
    static void reset()
    {
#ifdef VXSORT_STATS
        last_type = 0;
#endif
    }

    template<typename T>
    static vxsort_type typeid_to_vxsort_type() {
        if (typeid(T) == typeid(i16))
            return vxsort_type::I16;
        else if (typeid(T) == typeid(i32))
            return vxsort_type::I32;
        else if (typeid(T) == typeid(i64))
            return vxsort_type::I64;

        if (typeid(T) == typeid(u16))
            return vxsort_type::U16;
        else if (typeid(T) == typeid(u32))
            return vxsort_type::U32;
        else if (typeid(T) == typeid(u64))
            return vxsort_type::U64;

        if (typeid(T) == typeid(f32))
            return vxsort_type::F32;
        if (typeid(T) == typeid(f64))
            return vxsort_type::F64;

        throw new std::logic_error("Unsupported type");
    }

    static const char *vxsort_type_to_str(const vxsort_type type) {
        switch (type) {
            case I16: return "i16";
            case U16: return "u16";
            case I32: return "i32";
            case U32: return "u32";
            case I64: return "i64";
            case U64: return "u64";
            case F32: return "f32";
            case F64: return "f64";
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
