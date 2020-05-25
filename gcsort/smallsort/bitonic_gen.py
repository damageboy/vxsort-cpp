#!/usr/bin/env python3

max_bitonic_sort_verctors = 16

def next_power_of_2(v):
    v = v - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v = v + 1
    return int(v)


largest_merge_variant_needed = next_power_of_2(max_bitonic_sort_verctors) / 2;

## types to function suffix
bitonic_type_map = {
    "int32_t": "__m256i",
    "uint32_t": "__m256i",
    "float": "__m256",
    "int64_t": "__m256i",
    "uint64_t": "__m256i",
    "double": "__m256d",
}

bitonic_size_map = {
    "int32_t": 4,
    "uint32_t": 4,
    "float": 4,
    "int64_t": 8,
    "uint64_t": 8,
    "double": 8,
}

bitonic_types = bitonic_size_map.keys()


def generate_param_list(start, numParams):
    return str.join(", ", list(map(lambda p: f"d{p:02d}", range(start, start + numParams))))


def generate_param_def_list(numParams, nativeType):
    return str.join(", ", list(map(lambda p: f"{bitonic_type_map[nativeType]}& d{p:02d}", range(1, numParams + 1))))


def generate_shuffle_X1(v, type):
    if bitonic_size_map[type] == 4:
        return f"_mm256_shuffle_epi32({v}, 0xB1)"
    elif bitonic_size_map[type] == 8:
        return f"_mm256_shuffle_pd((__m256d) {v}, (__m256d) {v}, 0x5)"


def generate_shuffle_X2(v, type):
    if bitonic_size_map[type] == 4:
        return f"_mm256_shuffle_epi32({v}, 0x4E)"
    elif bitonic_size_map[type] == 8:
        return f"_mm256_permute4x64_pd((__m256d) {v}, 0x4E)"


def generate_shuffle_XR(v, type):
    if bitonic_size_map[type] == 4:
        return f"_mm256_shuffle_epi32({v}, 0x1B)"
    elif bitonic_size_map[type] == 8:
        return f"_mm256_permute4x64_pd((__m256d) {v}, 0x1B)"


def generate_blend_B1(v1, v2, type, ascending):
    if bitonic_size_map[type] == 4:
        if ascending:
            return f"_mm256_blend_epi32({v1}, {v2}, 0xAA)"
        else:
            return f"_mm256_blend_epi32({v2}, {v1}, 0xAA)"
    elif bitonic_size_map[type] == 8:
        if ascending:
            return f"_mm256_blend_pd((__m256d) {v1}, (__m256d) {v2}, 0xA)"
        else:
            return f"_mm256_blend_pd((__m256d) {v2}, (__m256d) {v1}, 0xA)"


def generate_blend_B2(v1, v2, type, ascending):
    if bitonic_size_map[type] == 4:
        if ascending:
            return f"_mm256_blend_epi32({v1}, {v2}, 0xCC)"
        else:
            return f"_mm256_blend_epi32({v2}, {v1}, 0xCC)"
    elif bitonic_size_map[type] == 8:
        if ascending:
            return f"_mm256_blend_pd((__m256d) {v1}, (__m256d) {v2}, 0xC)"
        else:
            return f"_mm256_blend_pd((__m256d) {v2}, (__m256d) {v1}, 0xC)"


def generate_blend_B4(v1, v2, type, ascending):
    if bitonic_size_map[type] == 4:
        if ascending:
            return f"_mm256_blend_epi32({v1}, {v2}, 0xF0)"
        else:
            return f"_mm256_blend_epi32({v2}, {v1}, 0xF0)"
    elif bitonic_size_map[type] == 8:
        raise Exception("WTF")


def generate_cross(v, type):
    if bitonic_size_map[type] == 4:
        return f"_mm256_permute4x64_pd((__m256d) {v}, 0x4E)"
    elif bitonic_size_map[type] == 8:
        raise Exception("WTF")


def generate_reverse(v, type):
    if bitonic_size_map[type] == 4:
        return f"_mm256_permute4x64_pd((__m256d) _mm256_shuffle_epi32({v}, 0x1B), 0x4E)"
    elif bitonic_size_map[type] == 8:
        return f"_mm256_permute4x64_pd((__m256d) {v}, 0x1B)"


def crappity_crap_crap(v1, v2, type):
    if type == "int64_t":
        return f"cmp = _mm256_cmpgt_epi64({v1}, {v2});"
    elif type == "uint64_t":
        return f"cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, {v1}), _mm256_xor_si256(topBit, {v2}));"

    return ""


def generate_min(v1, v2, type):
    if type == "int32_t":
        return f"_mm256_min_epi32({v1}, {v2})"
    elif type == "uint32_t":
        return f"_mm256_min_epu32({v1}, {v2})"
    elif type == "float":
        return f"_mm256_min_ps({v1}, {v2})"
    elif type == "int64_t":
        return f"_mm256_blendv_pd({v1}, {v2}, cmp)"
    elif type == "uint64_t":
        return f"_mm256_blendv_pd({v1}, {v2}, cmp)"
    elif type == "double":
        return f"_mm256_min_pd({v1}, {v2})"


def generate_max(v1, v2, type):
    if type == "int32_t":
        return f"_mm256_max_epi32({v1}, {v2})"
    elif type == "uint32_t":
        return f"_mm256_max_epu32({v1}, {v2})"
    elif type == "float":
        return f"_mm256_max_ps({v1}, {v2})"
    elif type == "int64_t":
        return f"_mm256_blendv_pd({v2}, {v1}, cmp)"
    elif type == "uint64_t":
        return f"_mm256_blendv_pd({v2}, {v1}, cmp)"
    elif type == "double":
        return f"_mm256_max_pd({v1}, {v2})"


def generate_1v_basic_sorters(f, type, ascending):
    maybe_cmp = lambda: ", cmp" if (type == "int64_t" or type == "uint64_t") else ""
    maybe_topbit = lambda: f"\n        {bitonic_type_map[type]} topBit = _mm256_set1_epi64x(1LLU << 63);" if (type == "uint64_t") else ""

    suffix = "ascending" if ascending else "descending"

    s = f"""    static inline void sort_01v_{suffix}({generate_param_def_list(1, type)}) {{
        {bitonic_type_map[type]}  min, max, s{maybe_cmp()};{maybe_topbit()}

        s = {generate_shuffle_X1("d01", type)};
        {crappity_crap_crap("s", "d01", type)}
        min = {generate_min("s", "d01", type)};
        max = {generate_max("s", "d01", type)};
        d01 = {generate_blend_B1("min", "max", type, ascending)};
    
        s = {generate_shuffle_XR("d01", type)};
        {crappity_crap_crap("s", "d01", type)}
        min = {generate_min("s", "d01", type)};
        max = {generate_max("s", "d01", type)};
        d01 = {generate_blend_B2("min", "max", type, ascending)};
    
        s = {generate_shuffle_X1("d01", type)};
        {crappity_crap_crap("s", "d01", type)}
        min = {generate_min("s", "d01", type)};
        max = {generate_max("s", "d01", type)};
        d01 = {generate_blend_B1("min", "max", type, ascending)};"""

    print(s, file=f)

    if bitonic_size_map[type] == 4:
        s = f"""
        s = {generate_reverse("d01", type)};
        min = {generate_min("s", "d01", type)};
        max = {generate_max("s", "d01", type)};
        d01 = {generate_blend_B4("min", "max", type, ascending)};
    
        s = {generate_shuffle_X2("d01", type)};
        min = {generate_min("s", "d01", type)};
        max = {generate_max("s", "d01", type)};
        d01 = {generate_blend_B2("min", "max", type, ascending)};
    
        s = {generate_shuffle_X1("d01", type)};
        min = {generate_min("s", "d01", type)};
        max = {generate_max("s", "d01", type)};
        d01 = {generate_blend_B1("min", "max", type, ascending)};"""
        print(s, file=f)
    print("}", file=f)


def generate_1v_merge_sorters(f, type, ascending):
    maybe_cmp = lambda: ", cmp" if (type == "int64_t" or type == "uint64_t") else ""
    maybe_topbit = lambda: f"\n        {bitonic_type_map[type]} topBit = _mm256_set1_epi64x(1LLU << 63);" if (type == "uint64_t") else ""

    suffix = "ascending" if ascending else "descending"

    s = f"""    static inline void sort_01v_merge_{suffix}({generate_param_def_list(1, type)}) {{
        {bitonic_type_map[type]}  min, max, s{maybe_cmp()};{maybe_topbit()}"""
    print(s, file=f)

    if bitonic_size_map[type] == 4:
        s = f"""
        s = {generate_cross("d01", type)};
        min = {generate_min("s", "d01", type)};
        max = {generate_max("s", "d01", type)};
        d01 = {generate_blend_B4("min", "max", type, ascending)};"""
        print(s, file=f)

    s = f"""
        s = {generate_shuffle_X2("d01", type)};
        {crappity_crap_crap("s", "d01", type)}
        min = {generate_min("s", "d01", type)};
        max = {generate_max("s", "d01", type)};
        d01 = {generate_blend_B2("min", "max", type, ascending)};

        s = {generate_shuffle_X1("d01", type)};
        {crappity_crap_crap("s", "d01", type)}
        min = {generate_min("s", "d01", type)};
        max = {generate_max("s", "d01", type)};
        d01 = {generate_blend_B1("min", "max", type, ascending)};"""

    print(s, file=f)
    print("}", file=f)


def generate_1v_sorters(f, type, ascending):
    generate_1v_basic_sorters(f, type, ascending)
    generate_1v_merge_sorters(f, type, ascending)


def generate_compounded_sorters(f, width, type, ascending):
    maybe_cmp = lambda: ", cmp" if (type == "int64_t" or type == "uint64_t") else ""
    maybe_topbit = lambda: f"\n        {bitonic_type_map[type]} topBit = _mm256_set1_epi64x(1LLU << 63);" if (type == "uint64_t") else ""

    w1 = int(next_power_of_2(width) / 2)
    w2 = int(width - w1)

    suffix = "ascending" if ascending else "descending"
    rev_suffix = "descending" if ascending else "ascending"

    s = f"""    static inline void sort_{width:02d}v_{suffix}({generate_param_def_list(width, type)}) {{
    {bitonic_type_map[type]}  tmp{maybe_cmp()};{maybe_topbit()}

    sort_{w1:02d}v_{suffix}({generate_param_list(1, w1)});
    sort_{w2:02d}v_{rev_suffix}({generate_param_list(w1 + 1, w2)});"""

    print(s, file=f)

    for r in range(w1 + 1, width + 1):
        x = w1 + 1 - (r - w1)
        s = f"""
    tmp = d{r:02d};
    {crappity_crap_crap(f"d{x:02d}", f"d{r:02d}", type)}
    d{r:02d} = {generate_max(f"d{x:02d}", f"d{r:02d}", type)};
    d{x:02d} = {generate_min(f"d{x:02d}", "tmp", type)};"""

        print(s, file=f)

    s = f"""
    sort_{w1:02d}v_merge_{suffix}({generate_param_list(1, w1)});
    sort_{w2:02d}v_merge_{suffix}({generate_param_list(w1 + 1, w2)});"""
    print(s, file=f)
    print("}", file=f)


def generate_compounded_mergers(f, width, type, ascending):
    maybe_cmp = lambda: ", cmp" if (type == "int64_t" or type == "uint64_t") else ""
    maybe_topbit = lambda: f"\n        {bitonic_type_map[type]} topBit = _mm256_set1_epi64x(1LLU << 63);" if (type == "uint64_t") else ""

    w1 = int(next_power_of_2(width) / 2)
    w2 = int(width - w1)

    suffix = "ascending" if ascending else "descending"
    rev_suffix = "descending" if ascending else "ascending"

    s = f"""    static inline void sort_{width:02d}v_merge_{suffix}({generate_param_def_list(width, type)}) {{
    {bitonic_type_map[type]}  tmp{maybe_cmp()};{maybe_topbit()}"""
    print(s, file=f)

    for r in range(w1 + 1, width + 1):
        x = r - w1
        s = f"""
    tmp = d{x:02d};
    {crappity_crap_crap(f"d{r:02d}", f"d{x:02d}", type)}
    d{x:02d} = {generate_min(f"d{r:02d}", f"d{x:02d}", type)};
    {crappity_crap_crap(f"d{r:02d}", "tmp", type)}
    d{r:02d} = {generate_max(f"d{r:02d}", "tmp", type)};"""
        print(s, file=f)

    s = f"""
    sort_{w1:02d}v_merge_{suffix}({generate_param_list(1, w1)});
    sort_{w2:02d}v_merge_{suffix}({generate_param_list(w1 + 1, w2)});"""
    print(s, file=f)
    print("}", file=f)


def get_load_intrinsic(type):
    if type == "double":
        return f"_mm256_loadu_pd(({type} const *)"
    if type == "float":
        return f"_mm256_loadu_ps(({type} const *)"
    return "_mm256_lddqu_si256((__m256i const *)"


def get_store_intrinsic(type):
    if type == "double":
        return f"_mm256_storeu_pd(({type} *)"
    if type == "float":
        return f"_mm256_storeu_ps(({type} *)"
    return f"_mm256_storeu_si256((__m256i *)"


def generate_entry_points(f, type):
    for m in range(1, max_bitonic_sort_verctors + 1):
        s = f"""
static __attribute__((noinline)) void sort_{m:02d}v({type} *ptr) {{"""
        print(s, file=f)

        for l in range(0, m):
            s = f"    {bitonic_type_map[type]} d{l + 1:02d} = {get_load_intrinsic(type)} ptr + {l});"
            print(s, file=f)

        s = f"    sort_{m:02d}v_ascending({generate_param_list(1, m)});"
        print(s, file=f)

        for l in range(0, m):
            s = f"    {get_store_intrinsic(type)} ptr + {l}, d{l + 1:02d});"
            print(s, file=f)

        print("}", file=f)


def generate_master_entry_point(f, type):
    s = f"""    static void sort({type} *ptr, int length) {{
    const int N = {int(32 / bitonic_size_map[type])};

    switch(length / N) {{"""
    print(s, file=f)

    for m in range(1, max_bitonic_sort_verctors + 1):
        s = f"        case {m}: sort_{m:02d}v(ptr); break;"
        print(s, file=f)
    print("    }", file=f)
    print("}", file=f)
    pass


def generate_per_type(f, type):
    s = f"""
#ifndef BITONIC_SORT_{type.upper()}_H
#define BITONIC_SORT_{type.upper()}_H

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"
#pragma ide diagnostic ignored "portability-simd-intrinsics"

#include <immintrin.h>
#include "bitonic_sort.h"
namespace gcsort {{
namespace smallsort {{
template<> class bitonic<{type}> {{
public:
"""
    print(s, file=f)
    generate_1v_sorters(f, type, ascending=True)
    generate_1v_sorters(f, type, ascending=False)
    for width in range(2, max_bitonic_sort_verctors + 1):
        generate_compounded_sorters(f, width, type, ascending=True)
        generate_compounded_sorters(f, width, type, ascending=False)
        if width <= largest_merge_variant_needed:
            generate_compounded_mergers(f, width, type, ascending=True)
            generate_compounded_mergers(f, width, type, ascending=False)

    generate_entry_points(f, type)
    generate_master_entry_point(f, type)
    print("};\n}\n}\n#endif", file=f)

def generate_all_types():
    for type in bitonic_types:
        with open(f"bitonic_sort.{type}.generated.h", "w") as f:
            generate_per_type(f, type)


if __name__ == '__main__':
    generate_all_types()
